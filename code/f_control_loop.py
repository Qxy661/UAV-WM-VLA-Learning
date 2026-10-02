"""
Demo F — 外环的延迟，进到闭环里是多大一盘菜？

对应文档：docs/03-VLA专题/05-机载部署与优化.md
          docs/04-VLM专题/03-LLM驱动的无人机Agent.md

要验证的结论：两篇正文各给了一个延迟数字，都没说这个数字是怎么来的。

  03/05 的分频控制架构把外环（VLA 推理）的延迟要求写成 < 50 ms，
        内环（PID）< 5 ms，低频层（任务规划）< 500 ms；
  04/03 说 LLM 推理延迟 1-4 秒，"无法满足实时控制需求"。

这一节把同一条曲线量出来：内环固定 50 Hz 正常跑，只让外环下发的目标点变陈旧，
从 0 ms 一路加到 4000 ms，看跟踪误差怎么长。两个文档的数字落在曲线的哪一段，
一眼就能看出来。

机制：外环每 1/f 秒才更新一次目标点，内环在这期间一直执行旧目标。
      所以"外环频率低"和"外环延迟大"是同一件事——都是指令陈旧，
      本节统一用陈旧度 τ 表示，τ = 1/(2f) 是两个口径之间的换算。

一个容易踩的坑（本脚本已避开，改动时请留意）：
  common/quad_sim.py 里的 expert() 不能直接拿来跑这条实验。它是**定点悬停**控制器，
  D 项作用于绝对速度，参考点一动，速度反馈就产生一个反向倾角把无人机拽停，
  实测 τ=0 时跟踪误差就有 1.86 m（几乎停在圆心）。跟踪移动参考点必须把
  D 项改成速度误差并加参考速度前馈，也就是本脚本的 track()。

  另外，陈旧度是拿周期轨迹量的：τ 等于整圈时长时，旧目标会绕回原位，
  误差假性归零。本脚本一圈 8000 ms、τ 最大 4000 ms（半圈），正好在单调区间内。

运行：py -3.9 code/f_control_loop.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim
from common import plotting

SEED = 0
N = 64                   # 并行无人机数，均匀铺在圆周上
T = 800                  # 800 步 × 0.02 s = 16 秒 = 2 圈
DT = 0.02                # 内环 50 Hz
R_REF = 2.0              # 参考圆半径 (m)
LAP = 8.0                # 一圈 8 秒
OMEGA = 2 * math.pi / LAP
H_REF = 1.5              # 巡航高度 (m)
WARMUP = 200             # 前 4 秒不计入统计，放掉起飞瞬态
STICK = 0.5              # "贴线"判据：离参考点 0.5 m 以内 (m)

TAUS = [0, 20, 50, 100, 200, 500, 1000, 2000, 4000]   # 外环指令陈旧度 (ms)


def reference(t, phase):
    """参考轨迹上 t 时刻的点，(N,3)。t 可以是负数——陈旧目标会回看过去。"""
    ang = OMEGA * t + phase
    return torch.stack([R_REF * torch.cos(ang), R_REF * torch.sin(ang),
                        torch.full_like(ang, H_REF)], dim=-1)


def ref_vel(t, phase):
    """参考轨迹在 t 时刻的速度，(N,3)，对 reference 求导得到（高度恒定，z 分量为 0）。

    形状必须和状态向量一致，否则第一帧 step 里 p + v*dt 会把位置压成 (N,2)。
    """
    ang = OMEGA * t + phase
    return torch.stack([-R_REF * OMEGA * torch.sin(ang),
                        R_REF * OMEGA * torch.cos(ang),
                        torch.zeros_like(ang)], dim=-1)


def track(env, p_ref, v_ref, kp=0.9, kd=0.7, ka=3.0, tilt_max=0.6):
    """跟踪移动参考点的串级控制器：位置误差 + 速度前馈。

    与 quad_sim.expert() 的差别只有一处：D 项比较的是**速度误差** `v - v_ref`
    而不是绝对速度 v。定点悬停时 v_ref=0，两者等价；参考点一动，前馈项让
    无人机提前建立倾角，位置项只需修正残差。
    """
    ex, ey = p_ref[:, 0] - env.p[:, 0], p_ref[:, 1] - env.p[:, 1]
    vx, vy = env.v[:, 0] - v_ref[:, 0], env.v[:, 1] - v_ref[:, 1]
    tilt = torch.stack([torch.clamp(kp * ex - kd * vx, -tilt_max, tilt_max),
                        torch.clamp(kp * ey - kd * vy, -tilt_max, tilt_max)], dim=-1)
    a = torch.zeros(env.n, 4)
    a[:, 0] = 0.5 + 0.10 * (H_REF - env.p[:, 2]) - 0.30 * env.v[:, 2]
    a[:, 1:3] = ka * (tilt - env.rpy[:, :2])
    return a.clamp(-1, 1)


@torch.no_grad()
def run(tau_ms):
    """内环 50 Hz 正常跑，外环下发的目标点陈旧 tau_ms。"""
    k = int(round(tau_ms / 1000.0 / DT))          # 陈旧的步数
    env = QuadSim(N)
    env.reset()
    phase = 2 * math.pi * torch.arange(N) / N     # 均匀铺在圆周上，互不重叠
    env.p = reference(0.0, phase).clone()         # 起步就站在各自的参考点上
    env.v = ref_vel(0.0, phase).clone()           # 且速度已经是参考速度
    env.rpy.zero_()
    env.om.zero_()

    err2, stick, dev = 0.0, 0, 0
    for t in range(T):
        ts = DT * (t - k)                         # <- 外环：下发的永远是 k 步前的点
        env.step(track(env, reference(ts, phase), ref_vel(ts, phase)))
        if t >= WARMUP:
            d = (env.p[:, :2] - reference(DT * t, phase)[:, :2]).norm(dim=-1)
            err2 += float((d ** 2).sum())
            stick += int((d < STICK).sum())
            dev = max(dev, float(d.max()))
    n = (T - WARMUP) * N
    return math.sqrt(err2 / n), stick / n, dev, env.p[:, :2].clone()


def chord(tau_ms):
    """相位滞后 τ 在半径 R 的圆上对应的弦长：2R·sin(ωτ/2)。"""
    return 2 * R_REF * math.sin(OMEGA * tau_ms / 1000.0 / 2)


def hold(env, goal, kp=0.9, kd=0.7, ka=3.0, tilt_max=0.6):
    """定点悬停控制器：D 项作用于绝对速度，收到目标点就停在那儿。

    外环下发离散航点时用它，比 track() 更贴近 LLM agent 的实际形态。
    """
    e = goal[:, :2] - env.p[:, :2]
    tilt = torch.stack([
        torch.clamp(kp * e[:, 0] - kd * env.v[:, 0], -tilt_max, tilt_max),
        torch.clamp(kp * e[:, 1] - kd * env.v[:, 1], -tilt_max, tilt_max)], dim=-1)
    a = torch.zeros(env.n, 4)
    a[:, 0] = 0.5 + 0.10 * (H_REF - env.p[:, 2]) - 0.30 * env.v[:, 2]
    a[:, 1:3] = ka * (tilt - env.rpy[:, :2])
    return a.clamp(-1, 1)


@torch.no_grad()
def run_waypoint(tau_ms):
    """外环每 tau_ms 重新下发一个航点，内环只做定点悬停。

    这是 LLM agent 的典型形态：慢外环给的是"去哪儿"，不是持续的轨迹指令。
    tau_ms=0 表示外环每步都更新，此时仍有误差——定点控制器跟踪移动参考点
    本身就滞后，这部分与延迟无关。
    """
    k = max(int(round(tau_ms / 1000.0 / DT)), 1)
    env = QuadSim(N)
    env.reset()
    phase = 2 * math.pi * torch.arange(N) / N
    env.p = reference(0.0, phase).clone()
    env.v.zero_()
    env.rpy.zero_()
    env.om.zero_()

    goal = reference(0.0, phase)
    err2 = 0.0
    for t in range(T):
        if t % k == 0:
            goal = reference(DT * t, phase)       # <- 外环重新规划
        env.step(hold(env, goal))
        if t >= WARMUP:
            d = (env.p[:, :2] - reference(DT * t, phase)[:, :2]).norm(dim=-1)
            err2 += float((d ** 2).sum())
    return math.sqrt(err2 / ((T - WARMUP) * N))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    tangential = R_REF * OMEGA
    print(f"参考轨迹：半径 {R_REF} m 的圆，{LAP} 秒一圈，切线速度 {tangential:.2f} m/s")
    print(f"内环 {1 / DT:.0f} Hz，外环目标点陈旧度 τ 从 0 扫到 {TAUS[-1]} ms；"
          f"{N} 架均匀铺在圆周上，前 {WARMUP * DT:.0f} 秒不计\n")
    print(f"{'τ (ms)':>8} {'等效外环频率':>14} {'连续跟踪':>10} {'理论弦长':>10} "
          f"{'阶梯航点':>10} {'贴线率':>9} {'最大偏离':>10}")
    res, wp = {}, {}
    for tau in TAUS:
        rmse, sticky, dev, p_end = run(tau)
        res[tau] = (rmse, sticky, dev, p_end)
        wp[tau] = run_waypoint(tau)
        eq = "—" if tau == 0 else f"{1000 / (2 * tau):.1f} Hz"
        print(f"{tau:>8} {eq:>14} {rmse:>8.2f} m {chord(tau):>8.2f} m "
              f"{wp[tau]:>8.2f} m {sticky * 100:>8.1f}% {dev:>8.2f} m")

    print(f"\n基准（τ=0）残差 {res[0][0] * 100:.0f} cm，这是内环自己的跟踪残差。"
          f"50 ms 时 {res[50][0] * 100:.0f} cm，与基准同量级")
    print(f"1 s 时 {res[1000][0]:.2f} m，4 s 时 {res[4000][0]:.2f} m"
          f"（圆直径 {2 * R_REF:.0f} m，已到上限）")
    big = [t for t in TAUS if t >= 500]
    print("τ ≥ 500 ms 时 实测/理论弦长 = "
          + " / ".join(f"{res[t][0] / chord(t):.2f}" for t in big)
          + "，说明内环几乎完美执行了陈旧指令，误差是纯几何的")

    # ---------------- 图 ----------------
    fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(13, 5.0))

    xs = [t for t in TAUS if t > 0]
    ys = [res[t][0] for t in xs]
    ax1.plot([1] + xs, [res[0][0]] + ys, "o-", color="#4C72B0", linewidth=2,
             markersize=5, label="连续跟踪（内环带速度前馈）")
    ax1.plot([1] + xs, [wp[0]] + [wp[t] for t in xs], "s-", color="#DD8452",
             linewidth=2, markersize=5, label="阶梯航点（内环定点悬停）")
    ax1.plot([1] + xs, [chord(1)] + [chord(t) for t in xs],
             "--", color="#8C8C8C", linewidth=1.8, label="理论弦长 2R·sin(ωτ/2)")
    ax1.axvspan(1000, 4000, color="#C44E52", alpha=0.12)
    ax1.axvline(50, color="#55A868", linestyle="--", linewidth=2)
    ax1.text(52, 3.6, "03/05 外环要求 <50 ms\n→ 滞后 7.8 cm", color="#55A868",
             fontsize=9, va="top")
    ax1.text(1150, 1.55, "04/03 LLM 延迟\n1–4 s → 1.6–4 m", color="#C44E52", fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("外环指令陈旧度 τ (ms，对数轴)")
    ax1.set_ylabel("跟踪误差 (m)")
    ax1.set_title("外环延迟 vs 跟踪误差")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3, which="both")

    ph0 = torch.tensor(0.0)                       # 画图只看第 0 号架
    th = torch.linspace(0, 2 * math.pi, 400)
    ax2.plot(R_REF * torch.cos(th), R_REF * torch.sin(th), color="#BBBBBB",
             linewidth=2, zorder=1, label="参考圆")
    p_end = reference((T - 1) * DT, ph0)          # t=16 s 时第 0 号架该在的位置
    ax2.plot([float(p_end[0])], [float(p_end[1])], "*", color="#333333",
             markersize=16, zorder=4)
    ax2.annotate("t = 16 s 时\n参考点在这里", xy=(float(p_end[0]), float(p_end[1])),
                 xytext=(0.0, -3.3), fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#333333"))
    for tau, color in ((0, "#55A868"), (200, "#DD8452"), (1000, "#8172B3"),
                       (2000, "#C44E52")):
        q = res[tau][3][0]                        # 第 0 号架的**实测**终点位置
        ax2.plot([float(p_end[0]), float(q[0])], [float(p_end[1]), float(q[1])],
                 color=color, linewidth=2, zorder=3)
        gap = float((q - p_end[:2]).norm())       # 图上画多长就标多长
        ax2.plot([float(q[0])], [float(q[1])], "o", color=color, markersize=8, zorder=4,
                 label=f"τ={tau} ms，落后 {gap:.2f} m")
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-3.6, 3.2)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("同一时刻的位置：差的是相位，不是半径")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle("外环延迟：内环再准也补不回外环的陈旧指令", y=1.0)
    plotting.save(fig, "f_latency.png")


if __name__ == "__main__":
    main()
