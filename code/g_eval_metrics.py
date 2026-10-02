"""
Demo G — 复现指南里那张表，换一列指标排名会变吗？

对应文档：docs/07-实践指南/02-复现指南-UAV-Flow.md
          docs/07-实践指南/03-复现指南-CognitiveDrone.md

要验证的结论：两篇复现指南各给了一张基线对比表——CognitiveDrone 那张是
ADE / FDE / Yaw Error / Task Completion 四列，UAV-Flow 是 Task Success Rate /
Position Error / Path Efficiency 那几项，并且都强调"只有环境、场景、回合数
完全一致时，这几项指标之间才可比"。

本节把这句话做实：三个策略跑同一个任务，算同一组指标，然后**按每一列
单独排名**。结论比"要小心"更具体——**七列排出七套名次，没有哪一列能
同时说清"到没到"和"到得好不好"**：

  · 完成率把 PID 和高增益都判成 100%，这一列分不开这两个；
  · 动作 MSE 里随机（0.43）和高增益（0.41）只差 5%，这一列会告诉你
    两者差不多——可它们的完成率是 0% 和 100%；
  · 跑得快的策略在"平均位置误差 / 终端误差"上赢，在"路径长度比 / SPL"
    上输；偏航误差甚至把随机策略排在高增益前面。

指标覆盖：UAV-Flow 那张表的 Task Success Rate / Position Error /
Path Efficiency / Action MSE 四项，CognitiveDrone 那张表的
ADE / FDE / Yaw Error / Task Completion 四项，本脚本全部量到。

两个指标口径上的坑（本脚本已避开，改动时请留意）：

  1. ADE 不要绑在"随时间推进的参考轨迹"上。若参考是"匀速直线、第 T 步
     正好到终点"，那么**提前到达并悬停的策略反而被判大误差**（参考点还在
     半路上）。本脚本按 UAV-Flow 的口径，ADE = 每一步到目标的距离在时间上
     的平均值，不引入额外的时间参数，也就没有这个反向惩罚。

  2. 偏航误差不要拿"指向目标"当基准。无人机贴近目标时这个方向是无定义的，
     悬停抖动会让误差飙到几十度——那不是策略的问题，是基准的问题。
     本脚本用**起点到终点的固定方向**当基准，且只在离目标 1 m 以外统计。

另外，偏航这一列在本仿真里要打折看：quad_sim 的 rpy 是整体按 TILT_LIMIT
限幅的，**偏航也被一起限在 ±1.2 rad**；而且偏航不参与位置动力学。
所以"偏航误差"只反映第四维动作的跟随质量，不代表飞行品质。
高增益那一行的 25.6% 步数贴着这个限幅——它摆得太猛，机头转不过来。

运行：py -3.9 code/g_eval_metrics.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim, expert, TILT_LIMIT
from common import plotting

SEED = 0
N = 128                  # 并行无人机数，每架算一个 episode
T = 400                  # 400 步 × 0.02 s = 8 秒
GOAL = torch.tensor([8.0, 0.0, 1.5])     # 起点在原点附近，直线距离 8 m
START_SPREAD = 0.6       # 起点散布 (m)
REACH_TOL = 1.0          # 到达判定容差 (m)
YAW_MIN_D = 1.0          # 离目标多远之外才统计偏航误差 (m)
YAW_K = 2.0              # 偏航角速率增益 (rad/s per rad)

# 三个策略：标准增益、拍脑袋调大 6 倍的增益、纯随机
KP_PID, KD = 0.30, 0.70
KP_HOT = 1.80


def make_start(n):
    """起点铺在原点附近的一小片里，终点固定。"""
    return START_SPREAD * torch.randn(n, 2)


def yaw_rate(env, goal, k=YAW_K):
    """第四维动作：让机头指向目标。返回归一化到 [-1,1] 的角速率指令。

    偏航在本仿真里不参与位置动力学（加速度只由横滚/俯仰决定），
    所以 Yaw Error 是一路与位置指标独立的量——正是 4D 动作第 4 维的考核项。
    """
    head = torch.atan2(goal[:, 1] - env.p[:, 1], goal[:, 0] - env.p[:, 0])
    err = head - env.rpy[:, 2]
    err = torch.atan2(torch.sin(err), torch.cos(err))       # 折到 [-π, π]
    return (k * err / 5.0).clamp(-1, 1)                     # 5.0 = quad_sim.OMEGA_MAX


def pol_random(env, goal):
    """随机动作：四个通道都在满量程里乱走。"""
    return 2 * torch.rand(env.n, 4) - 1


def pol_pid(env, goal):
    """标准串级 PID：位置环 kp=0.30，第四维对目标定向。"""
    a = expert(env, goal, kp=KP_PID, kd=KD)
    a[:, 3] = yaw_rate(env, goal)
    return a


def pol_hot(env, goal):
    """同一个控制器，位置环增益调大 6 倍——飞得急，但刹不住。"""
    a = expert(env, goal, kp=KP_HOT, kd=KD)
    a[:, 3] = yaw_rate(env, goal)
    return a


@torch.no_grad()
def rollout(policy, start):
    """跑一个策略，返回轨迹 (T+1, N, 2)、偏航角 (T+1, N)、状态 (T+1, N, 12)。"""
    env = QuadSim(N)
    env.reset()
    env.p[:, :2] = start
    env.p[:, 2] = GOAL[2]
    env.v.zero_()
    env.rpy.zero_()
    env.om.zero_()
    goal = GOAL[None, :].repeat(N, 1)

    ps, yaws, obs = [env.p[:, :2].clone()], [env.rpy[:, 2].clone()], [env.obs()]
    for _ in range(T):
        env.step(policy(env, goal))
        ps.append(env.p[:, :2].clone())
        yaws.append(env.rpy[:, 2].clone())
        obs.append(env.obs())
    return torch.stack(ps), torch.stack(yaws), torch.stack(obs)


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class Snapshot:
    """只带状态的空壳，好让策略函数能在一批离线状态上求动作。

    离线指标（动作 MSE）不该跑闭环——它是拿数据集里的一批状态，
    直接比较"策略会给什么动作"和"专家会给出什么动作"。
    """

    def __init__(self, obs, goal):
        self.n = obs.shape[0]
        self.p, self.v, self.rpy, self.om = obs[:, :3], obs[:, 3:6], obs[:, 6:9], obs[:, 9:]
        self.goal = goal


def action_mse(policy, obs, goal):
    """策略动作与专家动作的均方误差，在一批共享状态上算。

    只比前三维（位置控制的横滚/俯仰/油门），因为专家的第四维恒为 0
    （它不控偏航），拿它当基准会把任何偏航控制都判成误差。
    """
    snap = Snapshot(obs, goal)
    return ((policy(snap, goal) - expert(snap, goal))[:, :3] ** 2).mean().item()


def metrics(pos, yaw, start):
    """按 UAV-Flow / CognitiveDrone 的口径算六个指标。

    pos: (T+1, N, 2) 轨迹; yaw: (T+1, N) 偏航角; start: (N,2) 起点。
    """
    goal = GOAL[:2]
    d_goal = (pos - goal).norm(dim=-1)                 # (T+1, N) 到目标的距离

    ade = d_goal.mean(dim=0)                           # 平均位置误差（UAV-Flow 口径）
    fde = d_goal[-1]                                   # 终端误差（CognitiveDrone 的 FDE）

    # 偏航误差：机头方向 vs 起点->终点的固定方向，只在离目标 1 m 以外统计
    head = torch.atan2(goal[1] - start[:, 1], goal[0] - start[:, 0])   # (N,)
    err = wrap(yaw - head[None, :]).abs() * 180 / math.pi
    moving = d_goal > YAW_MIN_D
    yaw_err = (err * moving).sum(dim=0) / moving.sum(dim=0).clamp(min=1)

    # 路径长度比：实际走过的路程 / 直线距离
    seg = (pos[1:] - pos[:-1]).norm(dim=-1).sum(dim=0)
    straight = (goal - start).norm(dim=-1)
    ratio = seg / straight

    reached = fde < REACH_TOL
    spl = reached.float() * (straight / torch.maximum(seg, straight))
    sat = (yaw.abs() > TILT_LIMIT - 1e-6).float().mean().item()   # 偏航贴限幅的步数占比
    return dict(done=reached.float().mean().item(), ade=ade.mean().item(),
                fde=fde.mean().item(), yaw=yaw_err.mean().item(),
                ratio=ratio.mean().item(), spl=spl.mean().item(), sat=sat)


# 指标清单：(键, 中文名, 单位, 越大越好?)
METRICS = [("done", "完成率", "%", True), ("ade", "平均位置误差", "m", False),
           ("fde", "终端误差", "m", False), ("mse", "动作MSE", "", False),
           ("yaw", "偏航误差", "°", False), ("ratio", "路径长度比", "×", False),
           ("spl", "SPL", "", True)]


def rank_table(res, names):
    """按每一列单独排名。并列取平均名次。返回 {指标: {策略: 名次}}。"""
    out = {}
    for key, _, _, bigger_better in METRICS:
        vals = sorted((res[nm][key], nm) for nm in names)     # 升序
        if bigger_better:
            vals = vals[::-1]        # 越大越好 -> 降序，头名在第 1 位
        rk, i = {}, 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and abs(vals[j + 1][0] - vals[i][0]) < 1e-9:
                j += 1
            avg = (i + j) / 2 + 1          # 并列名次的平均
            for k in range(i, j + 1):
                rk[vals[k][1]] = avg
            i = j + 1
        out[key] = rk
    return out


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    start = make_start(N)
    straight = (GOAL[:2] - start).norm(dim=-1)
    print(f"{N} 架无人机，{T} 步（{T * 0.02:.0f} 秒），"
          f"直线距离 {straight.mean():.2f} ± {straight.std():.2f} m，"
          f"到达容差 {REACH_TOL} m")
    print(f"位置环增益：标准 PID kp={KP_PID}，高增益 kp={KP_HOT}（6 倍），"
          f"两者 kd={KD}\n")

    names = ["随机", "PID", "高增益"]
    pols = [("随机", pol_random), ("PID", pol_pid), ("高增益", pol_hot)]
    res, obs_pid = {}, None
    for nm, pol in pols:
        pos, yaw, obs = rollout(pol, start)
        res[nm] = metrics(pos, yaw, start)
        res[nm]["_pos"] = pos
        if nm == "PID":
            obs_pid = obs[::8].reshape(-1, 12)        # 每 8 步取一帧当离线数据集

    # 动作 MSE 是离线指标：所有策略都在同一批状态上比，不跑闭环
    goal_batch = GOAL[None, :].repeat(obs_pid.shape[0], 1)
    for nm, pol in pols:
        res[nm]["mse"] = action_mse(pol, obs_pid, goal_batch)

    head = f"{'策略':<8}" + "".join(f"{lab:>13}" for _, lab, _, _ in METRICS)
    print(head)
    for nm in names:
        row = f"{nm:<8}"
        for key, _, unit, _ in METRICS:
            v = res[nm][key] * (100 if unit == "%" else 1)
            row += f"{v:>11.2f}{unit:<2}"
        print(row)

    rk = rank_table(res, names)
    print(f"\n{'单列排名':<8}" + "".join(f"{lab:>13}" for _, lab, _, _ in METRICS))
    for nm in names:
        print(f"{nm:<8}" + "".join(f"{rk[key][nm]:>13.1f}" for key, _, _, _ in METRICS))

    gap = abs(res['随机']['mse'] - res['高增益']['mse']) / max(res['随机']['mse'],
                                                             res['高增益']['mse'])
    print(f"\n动作 MSE 这一列最刺眼：随机 {res['随机']['mse']:.3f} 与高增益 "
          f"{res['高增益']['mse']:.3f} 只差 {gap * 100:.0f}%，这一列会告诉你两者差不多，"
          f"实际上一个 8 秒飘出画面、一个稳稳到达（完成率 "
          f"{res['随机']['done'] * 100:.0f}% vs {res['高增益']['done'] * 100:.0f}%）")
    print(f"完成率这一列同样不济：PID 和高增益都是 100%，分不开。"
          f"能分开的是路径长度比（{res['PID']['ratio']:.2f}× vs "
          f"{res['高增益']['ratio']:.2f}×）和 SPL（{res['PID']['spl']:.2f} vs "
          f"{res['高增益']['spl']:.2f}）")
    print(f"\n高增益用 {res['高增益']['ratio'] / res['PID']['ratio']:.1f} 倍的路程换来了早到："
          f"平均位置误差 {res['高增益']['ade']:.2f} m vs {res['PID']['ade']:.2f} m，"
          f"终端误差 {res['高增益']['fde']:.2f} m vs {res['PID']['fde']:.2f} m")
    print(f"PID 用更短的路和更稳的机头换来了慢：偏航误差 "
          f"{res['PID']['yaw']:.1f}° vs {res['高增益']['yaw']:.1f}°")
    print(f"偏航误差这一列对随机策略没有意义：它 {res['随机']['yaw']:.1f}° 反而优于"
          f"高增益 {res['高增益']['yaw']:.1f}°，因为它压根不朝目标飞")
    print("偏航贴限幅（±{:.1f} rad）的步数占比：".format(TILT_LIMIT)
          + "、".join(f"{nm} {res[nm]['sat'] * 100:.1f}%" for nm in names)
          + " —— 高增益摆得太猛，机头转不过来")
    print('→ 七列指标排出七套名次；没有哪一列能同时说清"到没到"和"到得好不好"')

    # ---------------- 图 ----------------
    fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(13.5, 5.4),
                                            gridspec_kw={"width_ratios": [1, 1.15]})

    colors = {"随机": "C3", "PID": "C0", "高增益": "C1"}
    for nm in names:
        pos = res[nm]["_pos"][:, :16, :]               # 每档抽 16 条，免得糊成一片
        for j in range(pos.shape[1]):
            ax1.plot(pos[:, j, 0], pos[:, j, 1], color=colors[nm],
                     alpha=0.30, linewidth=0.8)
        ax1.plot([], [], color=colors[nm], linewidth=2.4,
                 label=f"{nm}（路径 {res[nm]['ratio']:.2f}×）")
    ax1.plot([float(start[:, 0].mean()), float(GOAL[0])],
             [float(start[:, 1].mean()), float(GOAL[1])],
             "k--", linewidth=1.6, label="直线参考（1.00×）")
    ax1.plot([GOAL[0]], [GOAL[1]], "k*", markersize=17, zorder=5)
    ax1.annotate("目标", xy=(float(GOAL[0]), float(GOAL[1])),
                 xytext=(float(GOAL[0]) - 0.4, float(GOAL[1]) + 1.2),
                 fontsize=10, ha="right",
                 arrowprops=dict(arrowstyle="->", color="#333333"))
    # 随机策略 8 秒能飘出上百米，不裁视野的话另外两条会被压成一个点
    ax1.set_xlim(-3.0, 11.0)
    ax1.set_ylim(-4.2, 4.2)
    ax1.text(-2.7, -3.8, f"随机策略已飘出画面（路径 {res['随机']['ratio']:.1f}×）",
             fontsize=8.5, color="C3")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("同一个任务，三个策略的轨迹")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_aspect("equal")

    # 名次折线图：x 轴是各个指标，y 轴是名次（1 在上）
    # 并列的策略名次相同，点会完全重叠——横向错开一点，否则看起来像少画了一条
    xoff, yoff = {nm: [] for nm in names}, {nm: [] for nm in names}
    for key, _, _, _ in METRICS:
        groups = {}
        for nm in names:
            groups.setdefault(rk[key][nm], []).append(nm)
        for _, members in groups.items():
            span = 0.17 * (len(members) - 1)
            for i, nm in enumerate(members):
                xoff[nm].append(-span / 2 + 0.17 * i if span else 0.0)
                yoff[nm].append(12 if i == 0 else -22)      # 并列时数值上下错开

    xs = range(len(METRICS))
    for nm in names:
        ys = [rk[key][nm] for key, _, _, _ in METRICS]
        xj = [x + d for x, d in zip(xs, xoff[nm])]
        ax2.plot(list(xs), ys, "-", color=colors[nm], linewidth=2.2, zorder=1)
        ax2.plot(xj, ys, "o", color=colors[nm], markersize=11, zorder=2,
                 label=nm)                       # 线走真实名次，点错开显示并列
        for x, y, dy, (key, _, unit, _) in zip(xj, ys, yoff[nm], METRICS):
            v = res[nm][key] * (100 if unit == "%" else 1)
            ax2.annotate(f"{v:.1f}{unit}", (x, y), textcoords="offset points",
                         xytext=(0, dy), ha="center", fontsize=8.5,
                         color=colors[nm])
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([lab for _, lab, _, _ in METRICS], fontsize=9.5)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(["第 1", "第 2", "第 3"])
    ax2.set_ylim(3.6, 0.4)
    ax2.set_ylabel("单列名次（越靠上越好）")
    ax2.set_title("同一组策略，换一列指标就换一套排名")
    ax2.legend(fontsize=9, loc="upper center", ncol=3, framealpha=0.95)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("复现指南那张表的七个指标，排出七套名次", y=1.0)
    plotting.save(fig, "g_metrics.png")


if __name__ == "__main__":
    main()
