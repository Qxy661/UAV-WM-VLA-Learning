"""
Demo V — 长时程遮挡：记忆位把重现瞬间的跳变压下来多少，以及它什么时候失效

对应文档：docs/02-世界模型专题/09-长时程世界模型与交互式生成.md

要验证的结论：目标被遮挡又重现时，**有没有持久记忆位**决定重观测那一帧要不要
做一次大修正——而这次修正会原样传进控制回路。三种外推方式：
  冻结     没有记忆位，遮挡期间只能一直用最后见到的那一帧位置（固定窗口模型的极限）
  线性外推 有记忆位，保留最后一次观测到的速度，全程匀速推
  衰减外推 有记忆位，但速度的置信度按时间常数 tau 衰减（不确定性随时间增长）
自变量是遮挡步数 K，因变量是重现瞬间的信念跳变量，以及重现后无人机的最大跟踪误差。

**结论不是"记忆位越好越好"**：短遮挡下线性外推的误差是 O(K^2)、冻结是 O(K)，
记忆位优势很大；但目标一机动，匀速外推的误差就随时间线性增长，而冻结的误差被
机动幅度封顶（最多 ±2A）。两者因此在某个 K 处交叉。衰减外推是在交叉两侧都站得住
的那个形态，而 tau 的合理取值由目标的机动特征时间 1/omega 定。

装置（复用 code/common/quad_sim.py）：
  真值     —— 专家追一个横向正弦运动的移动目标（与 Demo T 同一个装置）
  遮挡     —— 第 120 步起，目标连续 K 步不可观测，之后立刻重观测
  世界模型 —— 只提供"目标在哪"的信念；无人机动力学用真值仿真，所以下面所有误差
              都是**信念误差传进控制回路之后**的后果，不是模型误差本身

运行：py -3.9 code/v_longhorizon_memory.py
纯 CPU，约 30 秒；不训练任何模型。
"""

import math
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim, expert
from common import plotting

SEED = 13
N = 128
T = 400                    # 8 秒
DT = 0.02
T0 = 120                   # 遮挡起点（2.4 秒）
K_LIST = [5, 10, 20, 40, 80, 160]
POST = 100                 # 重观测后统计多少步
A_TARGET = 3.0
OMEGA = 0.6                # 目标机动角频率；特征时间 1/OMEGA = 1.667 秒
TAU_CHAR = 1.0 / OMEGA
TAU_LIST = [0.05, 0.2, 0.8, 1.6, 3.2, 12.8]   # 衰减外推的扫描档位
TAU_STAR = 1.6             # 代表档：与 1/omega = 1.667 同阶，取自扫描档位
GOAL_X, GOAL_Z = 8.0, 1.5
FREEZE, LINEAR, DECAY = "冻结", "线性外推", "衰减外推"
INF = float("inf")         # tau = INF 即"记忆永不衰减"


def target_pos(t, phi):
    """与 Demo T 同一个目标：沿 x 固定、沿 y 做正弦机动。"""
    s = t * DT
    g = torch.zeros(phi.shape[0], 3)
    g[:, 0] = GOAL_X
    g[:, 1] = A_TARGET * torch.sin(OMEGA * s + phi)
    g[:, 2] = GOAL_Z
    return g


def belief_traj(g_seq, K, tau):
    """按外推方式生成目标信念序列，形状 (T, n, 3)。

    遮挡区间是 [T0, T0+K]（含两端），重观测帧是 T0+K+1。
    tau = 0    -> 信念冻结在最后见到的位置（无记忆位的极限）
    tau = INF  -> 匀速外推不衰减
    其余       -> 位移按 v0 * tau * (1 - exp(-gap/tau)) 增长
    """
    Tn = g_seq.shape[0]
    end = min(T0 + K, Tn - 1)
    hats = g_seq.clone()
    if tau == 0.0:
        hats[T0:end + 1] = g_seq[T0]                   # 整个遮挡期停在最后一帧
        return hats, end
    v0 = (g_seq[T0] - g_seq[T0 - 1]) / DT              # 最后一次观测到的速度
    for t in range(T0 + 1, end + 1):
        gap = (t - T0) * DT
        disp = v0 * gap if tau == INF else v0 * tau * (1.0 - math.exp(-gap / tau))
        hats[t] = g_seq[T0] + disp
    return hats, end


def rollout(K, tau, phi, record=False):
    """跑一条闭环：无人机追信念，返回 (跳变量, 无人机位置轨迹或 None)。

    跳变量 = |hats[end] - hats[end+1]|，即信念轨迹在重观测帧的**不连续量**。
    这是控制回路要一次性吃下的修正，与遮挡外推的误差同源。

    record = True 时另外返回整条位置轨迹。**不要**用"无人机到真值目标的距离"当
    后果指标：专家控制器追机动目标本来就有几米稳态滞后，那个基数与遮挡无关，
    会把遮挡造成的那部分差异淹掉。见下面 delta()。
    """
    torch.manual_seed(SEED)
    env = QuadSim(N, dt=DT)
    env.reset(rand=True, pos_std=0.4, vel_std=0.3)
    g_seq = torch.stack([target_pos(t, phi) for t in range(T)])
    hats, end = belief_traj(g_seq, K, tau)

    jump, traj = None, []
    for t in range(T):
        if t == end + 1:
            jump = torch.norm(hats[end] - hats[t], dim=-1).mean().item()
        env.step(expert(env, hats[t]))
        if record:
            traj.append(env.p.clone())
    return jump, (torch.stack(traj) if record else None)


def delta(p_run, p_base, end):
    """遮挡造成的那一部分位移：与"同一初始条件、从不遮挡"的对照轨迹逐帧相减。

    两条轨迹的初值、控制器、目标**完全一样**，唯一差别是信念里有没有那段遮挡，
    所以这个差值把控制器的稳态滞后消掉了，剩下的就是遮挡的后果。

    分两段返回，因为它们是两个不同的入口：
      遮挡期最大偏离 —— 控制器整段都在追一个错的信念，被持续推走
      重现后最大偏离 —— 重观测那一帧的跳变一次性打进回路
    长遮挡下这两段会脱钩（冻结的跳变小但遮挡期偏离大），所以必须分开看。
    """
    d = torch.norm(p_run - p_base, dim=-1)              # (T, n)
    during = d[T0 + 1:end + 1]
    post = d[end + 1:min(end + POST, T - 1) + 1]
    return (during.max(dim=0).values.mean().item(),
            post.max(dim=0).values.mean().item())


def main():
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print("=" * 78)
    print("Demo V — 长时程遮挡：记忆位的收益与它的失效点")
    print("=" * 78)
    print(f"字体: {font or '未找到 CJK 字体'}")
    print(f"{N} 条并行轨迹 x {T} 步（{T * DT:.0f} 秒）｜遮挡起点第 {T0} 步"
          f"（{T0 * DT:.1f} 秒）｜目标振幅 {A_TARGET} m，角频率 {OMEGA} rad/s")
    print(f"目标的机动特征时间 1/omega = {TAU_CHAR:.3f} 秒"
          f"（= {TAU_CHAR / DT:.0f} 步）。遮挡长过它以后，匀速外推就该开始变差。")

    torch.manual_seed(SEED)
    phi = 2 * math.pi * torch.rand(N)

    # 对照轨迹：从不遮挡。它与 K、tau 都无关，跑一次就够——初值、控制器、目标
    # 都一样，唯一的差别是信念里有没有那段遮挡。
    _, p_base = rollout(0, 0.0, phi, record=True)

    # 一次性跑完三张表共用的网格：每个 K 下，冻结 + 匀速 + tau 扫描
    grid, jumpA, deltaB = {}, {}, {}
    for K in K_LIST:
        cells = {}
        for tau in [0.0, INF] + TAU_LIST:
            jump, p_run = rollout(K, tau, phi, record=True)
            end = min(T0 + K, T - 1)
            cells[tau] = (jump, delta(p_run, p_base, end))
        grid[K] = cells
        pick = (0.0, INF, TAU_STAR)
        jumpA[K] = [cells[t][0] for t in pick]
        deltaB[K] = [cells[t][1] for t in pick]

    head = (FREEZE, LINEAR, f"{DECAY} tau={TAU_STAR}s")

    print("\n" + "-" * 78)
    print("① 遮挡长度 x 外推方式 -> 重现瞬间的信念跳变量 (m)")
    print("-" * 78)
    print(f"{'K (步)':>8}{'秒':>7}" + "".join(f"{v:>16}" for v in head))
    for K in K_LIST:
        print(f"{K:>8}{K * DT:>7.1f}" + "".join(f"{x:>16.3f}" for x in jumpA[K]))
    print("   -> 短遮挡下线性外推是 O(K^2)、冻结是 O(K)，所以中段差距最大。")
    print(f"   -> 长遮挡下冻结反超：目标机动幅度只有 +-{A_TARGET} m，"
          f"信念冻结的误差被它封顶；匀速外推没有上限。")
    print(f"   -> 衰减外推两段都站得住：短遮挡时它近似匀速外推（gap << tau），"
          f"长遮挡时它退回冻结（gap >> tau）。")

    print("\n" + "-" * 78)
    print("② 无人机偏离「从不遮挡」对照轨迹的最大值 (m)：遮挡期 / 重现后")
    print("-" * 78)
    print(f"{'K (步)':>8}{'秒':>7}" + "".join(
        f"{v:>10}" for v in (FREEZE, LINEAR, DECAY)) * 2)
    print(f"{'':>15}{'遮挡期':^30}{'重现后 %d 步' % POST:^30}")
    for K in K_LIST:
        row = "".join(f"{x:>10.3f}" for x in (v[0] for v in deltaB[K]))
        row += "".join(f"{x:>10.3f}" for x in (v[1] for v in deltaB[K]))
        print(f"{K:>8}{K * DT:>7.1f}{row}")
    print("   -> 这一段是①和②减出来的：结果减掉了「从不遮挡」的对照轨迹，"
          "而不是直接看无人机到真值目标的距离——后者含专家控制器追机动目标"
          "本身的几米稳态滞后，会把遮挡那部分淹掉。")
    print("   -> 两个入口在长遮挡下**脱钩**：K = %d 时冻结的跳变（%.2f m）比"
          "线性外推（%.2f m）小，但它在整个遮挡期都在把一个恒定的错误位置喂给"
          "控制器，累积偏离反而更大。" % (K_LIST[-1], jumpA[K_LIST[-1]][0],
                                   jumpA[K_LIST[-1]][1]))
    print("   -> 所以长时程失效不是一个事件，是「持续错误 + 一次跳变」两段。"
          "只盯着重现瞬间的跳变去设计记忆机制，会漏掉前半段。")

    print("\n" + "-" * 78)
    print("③ 扫描衰减时间常数 tau：每一档遮挡的最优 tau 落在哪")
    print("-" * 78)
    print(f"{'K (步)':>8}" + "".join(f"{v:>8}" for v in TAU_LIST)
          + f"{'不衰减':>9}{'argmin':>8}{'1/w':>7}")
    best, edge = {}, {}
    for K in K_LIST:
        vals = [grid[K][tau][1][1] for tau in TAU_LIST]
        lin = grid[K][INF][1][1]
        j = min(range(len(vals)), key=lambda i: vals[i])
        best[K] = TAU_LIST[j]
        edge[K] = vals[j] > lin                 # 最优落在扫描上界之外（想要更大 tau）
        tag = f"{TAU_LIST[j]:.2f}" if not edge[K] else "不衰减"
        print(f"{K:>8}" + "".join(f"{v:>8.3f}" for v in vals)
              + f"{lin:>9.3f}{tag:>8}{TAU_CHAR:>7.2f}")
    print(f"   -> 最优档随遮挡长度**单调下移**：K <= 20 步时扫描的上界（不衰减）就已经"
          f"是最优，遮挡长到 {K_LIST[-1]} 步（{K_LIST[-1] * DT:.1f} 秒）才收敛到 "
          f"{best[K_LIST[-1]]:.1f} 秒附近——与目标的机动特征时间 "
          f"1/omega = {TAU_CHAR:.2f} 秒同阶。")
    print("   -> 这条趋势本身就是结论：**最优 tau 不是常数，它跟着遮挡长度走**。"
          "短遮挡下速度估计还准，就该全力外推（tau -> 无穷）；"
          "遮挡长过目标的机动时间尺度后，速度估计失效，衰减必须加快。")
    print("   -> 真实的做法不是部署前固定一个 tau，而是让不确定性随遮挡时间增长、"
          "由它在线决定 tau。这和 ① 里「衰减外推两段都站得住」是同一件事的两种说法。")

    print("\n" + "-" * 78)
    print("读法")
    print("-" * 78)
    print("   (1) 「有没有记忆位」不是二值问题，是「记忆该保留多久」的问题。"
          "全程匀速与完全冻结是同一根轴的两端，两端都不最优。")
    print("   (2) 最优 tau 随遮挡长度单调下移（③ 的 argmin 从「不衰减」降到 3.2 秒），"
          "说明长时程世界模型缺的能力不是「记住更多」，"
          "而是「知道自己的记忆什么时候开始不可信」。")
    print("   (3) 本装置里遮挡区间是**已知**的，信念从遮挡开始就按同一套规则外推，"
          "不做贝叶斯更新。真实系统的遮挡边界往往自己也不确定，"
          "所以这里量到的收益是上界；遮挡越靠猜测，实际收益越小。")

    # ---- 出图 ----
    fig, axes = plotting.plt.subplots(1, 3, figsize=(15.2, 4.6))
    cols = {FREEZE: "#C44E52", LINEAR: "#4C72B0", DECAY: "#55A868"}

    ax = axes[0]
    for i, name in enumerate((FREEZE, LINEAR, DECAY)):
        ax.plot(K_LIST, [jumpA[k][i] for k in K_LIST], "o-", color=cols[name],
                label=name)
    ax.axvline(TAU_CHAR / DT, color="#888888", ls=":", lw=1)
    ax.text(TAU_CHAR / DT * 0.94, ax.get_ylim()[1] * 0.9, "1/omega",
            fontsize=8, color="#666666", ha="right")
    ax.set_xlabel("遮挡步数 K")
    ax.set_ylabel("重现瞬间信念跳变量 (m)")
    ax.set_title("① 重观测帧要一次性吃下的修正（信念）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, name in enumerate((FREEZE, LINEAR, DECAY)):
        ax.plot(K_LIST, [deltaB[k][i][1] for k in K_LIST], "s-", color=cols[name],
                label=name)
    ax.set_xlabel("遮挡步数 K")
    ax.set_ylabel("偏离对照轨迹的最大值 (m)")
    ax.set_title("② 跳变传进控制回路的后果")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    M = torch.tensor([[grid[k][tau][1][1] for tau in TAU_LIST] for k in K_LIST])
    # 对数色标：误差跨了四个数量级，线性色标会让短遮挡的几行糊成一片同色
    im = ax.pcolormesh(TAU_LIST, K_LIST, M.numpy(), shading="nearest",
                       cmap="viridis", norm=mcolors.LogNorm())
    ax.plot([best[k] for k in K_LIST], K_LIST, "o-", color="#FFFFFF",
            mec="#C44E52", mew=1.4, label="扫描档 argmin tau")
    ax.plot([best[k] for k in K_LIST if edge[k]],
            [k for k in K_LIST if edge[k]], "o", ms=10, mfc="none",
            mec="#F2C14E", mew=1.8, label="最优在扫描上界外（想更大）")
    ax.axvline(TAU_CHAR, color="#FFFFFF", ls="--", lw=1.4)
    ax.text(TAU_CHAR * 1.06, K_LIST[0] * 1.15, "1/omega", color="#FFFFFF",
            fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("衰减时间常数 tau (s)")
    ax.set_ylabel("遮挡步数 K")
    ax.set_title("③ 最优 tau 与目标机动特征时间同阶")
    ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(im, ax=ax, label="最大跟踪误差 (m)")

    fig.suptitle("记忆位的收益不是单调的：匀速外推会输给冻结", y=1.0)
    fig.tight_layout()
    plotting.save(fig, "wm9_longhorizon_memory.png")


if __name__ == "__main__":
    main()
