"""
Demo T — 世界模型诊断：单步指标相等，不等于可用性相同

对应文档：docs/02-世界模型专题/07-世界模型评测与诊断.md

要验证的结论：世界模型"好不好"没有单一答案。三种典型退化——**常值偏置**、
**零均值抖动**、**遮挡期遗忘**——可以让同一个单步指标取到完全一样的值，而
在开环 rollout、在目标重现那一刻、在误差分布的尾部，它们的表现相差数倍到
一个数量级。单条诊断看不见全部三类失效，诊断要成套。

装置（复用 code/common/quad_sim.py）：
  真值   —— 脚本专家追一个横向正弦运动的移动目标，250 步 = 5 秒
  世界模型 —— 与真值同一套标称动力学的一步预测器，**退化按族解析注入**
  遮挡   —— 第 100~160 步（1.2 秒）目标不可观测，模型只能自己外推

三个退化族，强度都用同一个 ε 和同一个尺度 D = 1 m 表示，且**每步的期望位移
偏差都等于 ε·D**：
  drift  每步给预测位置加常值偏置 ε·D·ẑ（推力标定偏低在一步尺度上的表现）
  jitter 每步给预测位置加零均值噪声，按 E|N(0,I₃)| = 1.5958 归一，
         使每步的期望模长同样等于 ε·D
  forget 遮挡期间目标速度按 (1−ε_f) 缩放，ε_f=1 即"目标停在最后见到的位置"

**注意两族的 ε 不是同一个量**：drift/jitter 的 ε 是每步位移偏差（米），
forget 的 ε_f 是速度丧失比例。两者分列两张表、x 轴也不同，不做逐行对照。

四条诊断量：
  D1a 单步无人机位置误差均值          —— 只看无人机
  D1b 单步联合状态误差（无人机+目标） —— 两条都看
  D2  从真值状态开环推 30 步后的无人机位置误差
  D3  目标重现瞬间（遮挡期最后一步）的位置误差 —— 只看目标
  D4  单步无人机位置误差的 95 分位

**drift 与 jitter 的 D1a 在构造上相等**（每步期望位移偏差同为 ε·D），所以它们的
D1a 曲线应当重合——这是脚本的自检点。在此前提下：
  D2 之比应当接近 sqrt(30) = 5.477。常值偏置在线性累加，零均值噪声在随机游走，
  同样的单步误差在开环 30 步后差一个 sqrt(步数)。
  D4/D1a 则应当是 drift ≈ 1、jitter ≈ 1.75：常值偏置的误差分布是尖的，
  零均值噪声的模长服从 3 自由度 chi 分布，尾部更厚。

**要验证的主结论是失明是对称的**：D1a 只看无人机，对 forget 恒为 0（不是"不敏感"，
是解析意义下的 0——这一族压根不动无人机动力学）；D3 只看目标，对 drift/jitter
给出与 nominal 完全相同的 0.4963 m。所以每一条诊断量都有自己瞎掉的那一族，
只看一条量排名必然被它瞎掉的那一族带偏。nominal 一档的 D1a/D2/D4 恒为 0
（模型与真值同源），是自检点，不是结论。

运行：py -3.9 code/t_wm_diagnostics.py
纯 CPU，约 20 秒；不训练任何模型。
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim, expert
from common import plotting

SEED = 7
N = 256                    # 并行轨迹数
T = 250                    # 250 步 = 5 秒
DT = 0.02
OC_START, OC_END = 100, 160     # 遮挡窗口（含端点）
A_TARGET = 3.0             # 目标横向振幅 (m)
OMEGA = 0.6                # 目标横向角频率 (rad/s)
# ω=0.6 使目标周期约 10.5 秒，1.2 秒的遮挡窗只占几分之一个周期——窗口内速度
# 近似恒定。这是"有速度记忆 vs 无速度记忆"的正确对照：第一版取 ω=2.0，加速度
# 达 6 m/s²，匀速外推自己就过冲，"把目标冻结"反而更准（2.06 m 对 2.33 m），
# 结论被装置带反了。
GOAL_X, GOAL_Z = 8.0, 1.5
D_SCALE = 1.0              # 三族共用的扰动尺度 (m)
EPS_LIST = [0.01, 0.02, 0.05, 0.10, 0.20]
EPS_FORGET = [0.0, 0.25, 0.50, 0.75, 1.00]     # 遗忘族单独的量程，见 ⑤ 节
FAMILIES = ["nominal", "drift", "jitter", "forget"]
OPEN_LOOP = 30
STARTS = list(range(20, 220, 20))

# E|N(0, I_3)| = sqrt(2) * Gamma(2) / Gamma(1.5) = 1.5958
# 用它把 jitter 的每步期望位移归一化到与 drift 相同的 ε·D。
# 不除这个数，两族的"每步偏差量级"其实差 1.6 倍，D1a 曲线也就重合不了。
CHI3_MEAN = 1.5957691


def target_pos(t, phi):
    """目标位置：沿 x 匀速前进、沿 y 做正弦机动。phi 是每条的相位。"""
    s = t * DT
    g = torch.zeros(phi.shape[0], 3)
    g[:, 0] = GOAL_X
    g[:, 1] = A_TARGET * torch.sin(OMEGA * s + phi)
    g[:, 2] = GOAL_Z
    return g


def rollout(seed=SEED):
    """真值轨迹：专家追机动目标。返回状态序列、动作序列、目标相位。"""
    torch.manual_seed(seed)
    env = QuadSim(N, dt=DT)
    env.reset(rand=True, pos_std=0.4, vel_std=0.3)
    phi = 2 * math.pi * torch.rand(N)
    states, actions = [], []
    for t in range(T):
        a = expert(env, target_pos(t, phi))
        actions.append(a)
        states.append(env.obs())
        env.step(a)
    return torch.stack(states), torch.stack(actions), phi


def set_state(env, s):
    env.p, env.v = s[:, 0:3].clone(), s[:, 3:6].clone()
    env.rpy, env.om = s[:, 6:9].clone(), s[:, 9:12].clone()


def model_step(env, a, fam, eps, gen):
    """世界模型推进一步，就地更新 env 状态，返回预测位置。"""
    env.step(a)
    if fam == "drift":
        env.p = env.p + torch.tensor([0.0, 0.0, eps * D_SCALE])
    elif fam == "jitter":
        env.p = env.p + (eps * D_SCALE / CHI3_MEAN) * torch.randn(
            env.p.shape, generator=gen)
    return env.p


def target_belief(phi, fam, eps):
    """目标信念在整条时间轴上的传播，返回（真值, 信念），形状 (T, n, 3)。"""
    g_seq = torch.stack([target_pos(t, phi) for t in range(T)])
    hats = torch.zeros_like(g_seq)
    hats[0] = g_seq[0]
    v_hat = torch.zeros(N, 3)
    for t in range(1, T):
        if OC_START <= t <= OC_END:          # 遮挡：只能外推
            k = (1.0 - eps) if fam == "forget" else 1.0
            hats[t] = hats[t - 1] + k * v_hat * DT
        else:                                # 可观测：直接校正，并更新速度估计
            hats[t] = g_seq[t]
            v_hat = (g_seq[t] - g_seq[t - 1]) / DT
    return g_seq, hats


def diagnose(states, actions, phi, fam, eps):
    """跑一族一档，返回各诊断量。"""
    # 种子必须由 (族, 档) 确定性推出：用 hash() 会吃到 PYTHONHASHSEED 随机化，
    # 同一条命令两次跑出不同的数。
    # 遗忘族的 ε 是「速度丧失比例」，量程与另两族不同（见 ⑤ 节），种子按各自的档位表取
    grid = EPS_FORGET if fam == "forget" else EPS_LIST
    gi = FAMILIES.index(fam) * 100 + grid.index(eps)
    gen = torch.Generator().manual_seed(SEED * 1000 + gi)
    g_seq, g_hat = target_belief(phi, fam, eps)

    # ---- 单步误差 ----
    env = QuadSim(N, dt=DT)
    dp_list, dg_list = [], []
    for t in range(T - 1):
        set_state(env, states[t])
        p_hat = model_step(env, actions[t], fam, eps, gen)
        dp_list.append(torch.norm(p_hat - states[t + 1][:, 0:3], dim=-1))
        dg_list.append(torch.norm(g_hat[t + 1] - g_seq[t + 1], dim=-1))
    dp = torch.stack(dp_list)                 # (T-1, n) 无人机单步误差
    dg = torch.stack(dg_list)                 # (T-1, n) 目标单步误差
    d1a = dp.mean().item()
    d1b = torch.sqrt(dp ** 2 + dg ** 2).mean().item()
    d4 = torch.quantile(dp.flatten(), 0.95).item()

    # ---- D2：开环 30 步 ----
    d2s = []
    for t0 in STARTS:
        set_state(env, states[t0])
        for j in range(OPEN_LOOP):
            p_hat = model_step(env, actions[t0 + j], fam, eps, gen)
        d2s.append(torch.norm(p_hat - states[t0 + OPEN_LOOP][:, 0:3], dim=-1))
    d2 = torch.cat(d2s).mean().item()

    # ---- D3：目标重现瞬间 ----
    d3 = torch.norm(g_hat[OC_END] - g_seq[OC_END], dim=-1).mean().item()

    return {"D1a": d1a, "D1b": d1b, "D2": d2, "D3": d3, "D4": d4}


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print("=" * 78)
    print("Demo T — 世界模型诊断：四条诊断量，三种退化")
    print("=" * 78)
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}")
    print(f"轨迹 {N} 条 x {T} 步（{T * DT:.0f} 秒）｜遮挡第 {OC_START}~{OC_END} 步"
          f"（{(OC_END - OC_START) * DT:.1f} 秒，占 "
          f"{(OC_END - OC_START) / (T - 1) * 100:.0f}% 的步数）｜扰动尺度 D = {D_SCALE} m")

    states, actions, phi = rollout()
    g_seq, _ = target_belief(phi, "nominal", 0.0)
    travel = torch.norm(g_seq[OC_END] - g_seq[OC_START], dim=-1)
    print(f"\n目标在遮挡期位移：{travel.mean():.3f} ± {travel.std():.3f} m"
          f"（最大 {travel.max():.2f} m）")

    print("\n" + "-" * 78)
    print("① 各诊断量在不同 ε 下的取值")
    print("-" * 78)
    print("   连续性退化（ε = 每步位置偏差，米）：")
    print(f"{'ε':>6}  {'族':<8}{'D1a 单步(机)':>14}{'D1b 单步(机+目标)':>19}"
          f"{'D2 开环30步':>14}{'D3 重现瞬间':>14}{'D4 单步P95':>13}")
    table = {}
    for eps in EPS_LIST:
        for fam in ("nominal", "drift", "jitter"):
            r = diagnose(states, actions, phi, fam, eps)
            table[(fam, eps)] = r
            print(f"{eps:>6.2f}  {fam:<8}{r['D1a']:>14.4f}{r['D1b']:>19.4f}"
                  f"{r['D2']:>14.4f}{r['D3']:>14.4f}{r['D4']:>13.4f}")
        print()
    print("   遗忘型退化（ε_f = 遮挡期间速度丧失的比例）：")
    table_f = {}
    for eps in EPS_FORGET:
        r = diagnose(states, actions, phi, "forget", eps)
        table_f[eps] = r
        print(f"{eps:>6.2f}  {'forget':<8}{r['D1a']:>14.4f}{r['D1b']:>19.4f}"
              f"{r['D2']:>14.4f}{r['D3']:>14.4f}{r['D4']:>13.4f}")
    print("\n   两族分列，不要逐行横向对照：ε 是每步挪多少米，ε_f 是丢掉几成速度，"
          "量纲不同。两族都关掉时（ε = 0 等价于 ε_f = 0）数才可比，"
          "那是 ⑤ 节的自检点。")

    print("-" * 78)
    print("② 自检：drift 与 jitter 的每步期望位移偏差同为 ε·D，D1a 之比应约为 1")
    print("-" * 78)
    for eps in EPS_LIST:
        a, b = table[("drift", eps)]["D1a"], table[("jitter", eps)]["D1a"]
        print(f"   ε={eps:<5} drift / jitter = {a / b:.4f}"
              f"   （绝对量 {a:.4f} vs {b:.4f} m，理论同为 {eps * D_SCALE:.4f}）")
    print("   -> 单步均值这一条诊断，在这两族之间没有任何分辨力。")

    print("\n" + "-" * 78)
    print("③ 单步相等的前提下，开环误差之比（理论值 sqrt(30) = 5.477）")
    print("-" * 78)
    print(f"   {'ε':>6}{'D2 drift / jitter':>22}{'理论':>10}")
    for eps in EPS_LIST:
        r = table[("drift", eps)]["D2"] / table[("jitter", eps)]["D2"]
        print(f"   {eps:>6.2f}{r:>22.3f}{math.sqrt(OPEN_LOOP):>10.3f}")
    print("   -> 常值偏置每步线性累加，零均值噪声每步随机游走，"
          "同样的单步误差在开环下差一个 sqrt(步数)。")

    print("\n" + "-" * 78)
    print("④ 尾部分辨力：D4 / D1a（理论 drift ≈ 1，jitter ≈ 2.795/1.596 = 1.751）")
    print("-" * 78)
    print(f"   {'ε':>6}{'drift':>12}{'jitter':>12}")
    for eps in EPS_LIST:
        row = [table[(f, eps)]["D4"] / table[(f, eps)]["D1a"]
               for f in ("drift", "jitter")]
        print(f"   {eps:>6.2f}" + "".join(f"{v:>12.2f}" for v in row))
    print("   -> 常值偏置的误差分布是尖的，零均值噪声的模长服从 chi 分布、尾部更厚。"
          "只看均值同样分不出这两族。")
    print("   （nominal 这一档不出数：模型与真值共用同一套标称动力学时，"
          "D1a 恒为 0，比值无定义。这是本实验的自检点，不是结论。）")

    print("\n" + "-" * 78)
    print("⑤ 遗忘族单独扫：ε_f 是速度保留比例的补（ε_f = 1 即完全冻结目标）")
    print("-" * 78)
    print("   这一族的 ε 与上面两族不同量纲，不能共用一档量程：drift/jitter 的 ε")
    print("   是「每步位移偏差」（米），forget 的 ε 是「速度丧失多少比例」。共用 0.2")
    print("   只会得到「只忘掉两成速度」，量不出冻结这个失效模式。")
    print(f"\n   {'ε_f':>6}{'D1a':>10}{'D1b':>10}{'D3 (m)':>12}"
          f"{'D1b 相对':>12}{'D3 相对':>12}")
    fz = [(e, table_f[e]) for e in EPS_FORGET]
    base1b, base3 = fz[0][1]["D1b"], fz[0][1]["D3"]
    for e, r in fz:
        print(f"   {e:>6.2f}{r['D1a']:>10.4f}{r['D1b']:>10.4f}{r['D3']:>12.4f}"
              f"{r['D1b'] / base1b:>12.3f}{r['D3'] / base3:>12.3f}")
    nom = table[("nominal", 0.01)]
    # 自检：ε_f = 0 就是「不遗忘」，应与 nominal 逐项同值（同一段代码路径）
    for k in ("D1a", "D1b", "D3", "D4"):
        assert abs(fz[0][1][k] - nom[k]) < 1e-9, (k, fz[0][1][k], nom[k])
    print(f"\n   参照 nominal（匀速外推，不遗忘）：D1a {nom['D1a']:.4f}、"
          f"D1b {nom['D1b']:.4f}、D3 {nom['D3']:.4f} m"
          f"（与 ε_f = 0 逐项同值，脚本自检已断言）")
    e_last, r_last = fz[-1]
    oc = (OC_END - OC_START) / (T - 1) * 100
    print(f"   -> (a) D1a 在整档 ε_f 上恒为 0.0000：这一族不动无人机动力学，"
          f"只看无人机的诊断量对它是**完全失明**，不是「不敏感」。")
    print(f"      (b) D1b 把目标项并了进来，看见了（{base1b:.4f} -> "
          f"{r_last['D1b']:.4f}，{r_last['D1b'] / base1b:.2f} 倍），但它对整条时间轴取均值，"
          f"而误差只存在于 {oc:.0f}% 的遮挡步上——涨幅被这一步数占比摊薄。")
    print(f"      (c) D3 只读重现那一瞬：{base3:.4f} -> {r_last['D3']:.4f} m。"
          f"它自己的基线 0.4963 m 不是遗忘造成的，而是匀速外推本身在加速目标上的"
          f"误差（nominal 也是这个数），所以判断遗忘要扣基线：净增量 "
          f"{base3 - base3:.4f} -> {r_last['D3'] - base3:.4f} m。")
    print(f"   -> 把两边的净增量放在一起：同一个失效，在全程平均的单步指标上只体现为 "
          f"{r_last['D1b'] - base1b:.4f} m，在重现瞬间的指标上是 "
          f"{r_last['D3'] - base3:.4f} m，差 "
          f"{(r_last['D3'] - base3) / (r_last['D1b'] - base1b):.1f} 倍。"
          f"遮挡占的步数比例越小，这个差越大。")

    print("\n" + "-" * 78)
    print("读法")
    print("-" * 78)
    print("   (1) 每一条诊断量都有自己瞎掉的那一族，而且失明是对称的：")
    print("       D1a 只看无人机 -> 对 forget 恒为 0（失明）；"
          "D3 只看目标 -> 对 drift/jitter 恒为 0.4963（失明）。")
    print("       分开 drift 与 jitter 要靠 D2（累加方式不同，差 sqrt(30) 倍）"
          "或 D4（分布形状不同，差 1.75 倍）。")
    print("       -> 只看一条量做排序，名次必然被它瞎掉的那一族带偏。")
    print("   (2) 换口径能改变可见性，但换不掉失明。D1b 把目标并进来，看见了 forget，"
          "代价是无人机那两族的差异被目标项摊平；D3 能把 forget 单独拎出来，"
          "代价是对另两族彻底无感（它们给的数与 nominal 一模一样）。")
    print("   (3) 本实验里的模型与真值共用同一套标称动力学，退化是**解析注入**的，"
          "所以量的是**诊断量本身的分辨力**，不是某个真实模型的好坏。"
          "把结论外推到训练出来的模型上，需要另做实验。")

    # ---- 出图 ----
    # 前三格画连续三族（ε 同量纲，可共用一个对数横轴）；第四格单独画遗忘族，
    # 它的 ε_f 是另一套量纲，混在一个横轴上会让人以为两族可以按同一个 ε 比大小。
    fig, axes = plotting.plt.subplots(2, 2, figsize=(12.4, 8.4))
    colors = {"nominal": "#888888", "drift": "#C44E52",
              "jitter": "#4C72B0", "forget": "#55A868"}
    styles = {"nominal": ":", "drift": "o-", "jitter": "s-", "forget": "^-"}
    for key, ax, title in [
            ("D1a", axes[0][0], "① 单步无人机误差（均值）——两族重合"),
            ("D2", axes[0][1], "② 开环 30 步位置误差——drift 与 jitter 差 5.5 倍"),
            ("D4", axes[1][0], "④ 单步误差 95 分位——同一族换一个统计量")]:
        for fam in ("drift", "jitter"):
            ys = [table[(fam, eps)][key] for eps in EPS_LIST]
            ax.plot(EPS_LIST, ys, styles[fam], color=colors[fam], label=fam)
        ax.set_xlabel("退化强度 ε（米/步）")
        ax.set_ylabel("误差 (m)")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.plot([e for e, _ in fz], [r["D3"] for _, r in fz], "^-",
            color=colors["forget"], label="forget")
    ax.axhline(nom["D3"], color=colors["nominal"], ls=":", label="nominal")
    ax.set_xlabel("遗忘强度 ε_f（速度丧失比例）")
    ax.set_ylabel("误差 (m)")
    ax.set_title("③ 目标重现瞬间误差——对另两族恒为 nominal 基线，只有 forget 会升")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("同一个世界模型，四条诊断量给出四种说法", y=1.0)
    fig.tight_layout()
    plotting.save(fig, "wm7_diagnostics.png")


if __name__ == "__main__":
    main()
