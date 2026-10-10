"""
Demo E — 几何安全校正（GSC）管用到什么程度？

对应文档：docs/03-VLA专题/02-无人机VLA模型.md

要验证的结论：VLA-AN 用一个几十行的后处理把碰撞率从 18.7% 压到 1.2%，
公式只有一句"把指向障碍物的动作分量减掉"。这一节把它放进闭环里量出来。

两个场景的任务结构完全相同——目标都在障碍物正后方，都得从障碍物旁边过去，
唯一变量是障碍物的形状：

  凸形：一个球。绕过去的路只有一条，但走斜线时切向分量还在，能顺着球面滑过去。
  凹形：两个球留出一道 0.8 m 的窄缝。缝够一架半径 0.15 m 的无人机穿过，
        但滤波器看到的只有"最近的那个球"，它给出的法向量带横向分量，
        减掉之后反而把无人机推向另一个球——校正量指向陷阱内部。
        这是正文思考题 1 点名的那个边界（原答案说的是 U 形槽法向量朝陷阱内部，
        同一个机制：只看最近障碍，用的是局部信息）。

同一个凸形场景，GSC 把碰撞率从 100% 压到 0；同一个凹形场景，它把碰撞率
从 53% 抬到 73%。差别不在代码，在几何。

两个容易被忽略的地方（本脚本都已避开，改动时请留意）：

  1) n_obs 的方向约定。公式 `a - max(0, a·n)·n` 只有在 n 从无人机指向障碍物
     时才成立——此时 a·n > 0 表示"正在朝障碍物飞"，减掉它才叫校正。
     若把 n 定义成从障碍物指向无人机，a·n 恒为负，clamp 到 0 之后
     整段校正静默失效（不报错，只是什么都没做）。本脚本第一版就写反了。

  2) 触发距离要盖过刹车距离。倾角限幅 0.6 rad 对应最大水平加速度 5.3 m/s²，
     3 m/s 巡航的刹车距离是 0.85 m。D_THRESH 小于它时，滤掉速度指令
     也来不及减速——凸形场景同样会撞（实测 D_THRESH=1.0 时凸形还有 83.6% 撞）。

运行：py -3.9 code/e_gsc_safety.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim
from common import plotting

SEED = 0
N = 256                  # 并行无人机数
T = 400                  # 每回合步数，400 * 0.02 = 8 秒
KV, V_MAX = 0.8, 3.0     # 位置误差 -> 期望速度的增益 / 速度上限 (m/s)
KT = 1.2                 # 速度误差 -> 期望倾角的增益
KA, TILT_MAX = 3.0, 0.6  # 姿态内环增益 / 倾角限幅
D_THRESH = 1.5           # GSC 触发距离（到障碍表面的距离），须盖过 0.85 m 的刹车距离
DRONE_R = 0.15           # 无人机半径，用于判定碰撞
REACH_TOL = 0.8          # 到达判定容差 (m)


def make_scene(kind):
    """返回 (球心 (M,2), 半径 (M,), 目标 (3,))。障碍物一律用球近似。

    两个场景的**任务结构完全相同**：目标都在障碍物正后方，都得从障碍物旁边过去。
    唯一的差别是障碍物的形状——这正是要对照的变量。
    """
    goal = torch.tensor([7.0, 0.0, 1.5])
    if kind == "convex":
        # 一个球正挡在起终点连线上：凸形，绕过去的路只有一条但也够宽
        cen = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
        rad = torch.tensor([1.2])
    else:
        # 两个球留出一道 0.8 m 的缝：缝本身够一架 0.15 m 半径的无人机穿过，
        # 但两个球的法向量都朝前，滤波器分不清"墙"和"门"。
        cen = torch.tensor([[3.0, 1.15], [3.0, -1.15]], dtype=torch.float32)
        rad = torch.full((2,), 0.75)
    return cen, rad, goal


def gsc_filter(a_raw, p_xy, obstacles, d_thresh):
    """VLA-AN 的几何安全校正。

        if d_obs < d_threshold: a_safe = a_raw - max(0, a_raw·n_obs) * n_obs

    障碍物是球，n_obs 取"无人机 -> 最近球心"的单位向量（见模块 docstring）。
    只对最近的一个障碍生效——这正是原公式的做法，也是它在凹形障碍上失效的原因：
    最近的那个球给出的法向量是局部信息，看不到"整体凹进去"这件事。
    """
    cen, rad = obstacles
    d = cen[None, :, :] - p_xy[:, None, :]          # 无人机 -> 球心 (N,M,2)
    d_obs = d.norm(dim=-1) - rad[None, :]           # 到球面的距离 (N,M)
    idx = d_obs.argmin(dim=1)                       # 最近障碍的下标 (N,)

    ar = torch.arange(p_xy.shape[0])
    n = d[ar, idx]
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    d_min = d_obs[ar, idx]

    proj = (a_raw * n).sum(dim=-1, keepdim=True)    # a_raw · n_obs
    active = (d_min < d_thresh).unsqueeze(-1)
    return a_raw - active * proj.clamp(min=0.0) * n


def policy(env, goal, obstacles, use_gsc, d_thresh=D_THRESH):
    """两层结构，GSC 插在中间。

        位置误差 -> 期望水平速度  ──[GSC 在这里]──>  速度误差 -> 倾角 -> 角速率

    GSC 必须作用在**速度指令**上。放在倾角（≈加速度）这一层是错的：
    无人机以 2 m/s 巡航时，消掉朝障碍的加速度分量并不会消掉已经攒下的速度，
    照样撞上去——本脚本第一版就栽在这里。
    """
    v_des = torch.clamp(KV * (goal[:, :2] - env.p[:, :2]), -V_MAX, V_MAX)
    if use_gsc:
        v_des = gsc_filter(v_des, env.p[:, :2], obstacles, d_thresh)
    tilt_des = torch.clamp(KT * (v_des - env.v[:, :2]), -TILT_MAX, TILT_MAX)

    a = torch.zeros(env.n, 4)
    a[:, 0] = 0.5 + 0.10 * (goal[:, 2] - env.p[:, 2]) - 0.30 * env.v[:, 2]
    a[:, 1:3] = KA * (tilt_des - env.rpy[:, :2])
    return a.clamp(-1, 1)


def collided(p_xy, obstacles):
    """与任一球面相交即算碰撞。"""
    cen, rad = obstacles
    d = (p_xy[:, None, :] - cen[None, :, :]).norm(dim=-1)
    return (d < rad[None, :] + DRONE_R).any(dim=1)


@torch.no_grad()
def run(kind, use_gsc, n=N, d_thresh=D_THRESH):
    """跑一个场景，返回 (碰撞率, 到达率, 卡住率, 轨迹记录)。"""
    cen, rad, goal3 = make_scene(kind)
    obstacles = (cen, rad)
    goal = goal3[None, :].repeat(n, 1)

    env = QuadSim(n)
    # 起点铺在障碍带之前的一条横线上，横向铺开±3 m：
    # 这样接近角分散，才看得出 GSC 在斜向接近和正前方接近上的差别。
    # 不能用大 pos_std 的随机出生点——那样会有无人机直接出生在球内部，
    # 把碰撞率污染成接近 100%。
    env.reset()
    env.p[:, 0] = -2.5 + 0.3 * torch.randn(n)
    env.p[:, 1] = 3.0 * (torch.rand(n) * 2 - 1)
    env.p[:, 2] = 1.5

    start_hit = collided(env.p[:, :2], obstacles)
    if start_hit.any():
        raise RuntimeError(f"{int(start_hit.sum())} 架无人机出生在障碍物内部，请调整出生区")

    ever_hit = torch.zeros(n, dtype=torch.bool)
    traj = []
    for t in range(T):
        env.step(policy(env, goal, obstacles, use_gsc, d_thresh))
        ever_hit |= collided(env.p[:, :2], obstacles)
        if t % 5 == 0:
            traj.append(env.p[:, :2].clone())

    reached = (env.p - goal).norm(dim=-1) < REACH_TOL
    success = reached & ~ever_hit
    # 既没撞也没到：被安全滤波器挡住了，或卡在障碍前
    stalled = ~reached & ~ever_hit
    return (ever_hit.float().mean().item(), success.float().mean().item(),
            stalled.float().mean().item(), torch.stack(traj))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    a_max = math.tanh(TILT_MAX) * 9.81
    print(f"{N} 架无人机，每回合 {T} 步（{T * 0.02:.0f} 秒），"
          f"GSC 触发距离 {D_THRESH} m")
    print(f"倾角限幅 {TILT_MAX} rad -> 最大水平加速度 {a_max:.2f} m/s^2，"
          f"{V_MAX} m/s 巡航的刹车距离 {V_MAX ** 2 / (2 * a_max):.2f} m\n")
    print(f"{'场景':<10} {'GSC':<5} {'碰撞率':>8} {'到达率':>8} {'卡住率':>8}")
    res = {}
    for kind, label in (("convex", "凸形球"), ("concave", "凹形缝")):
        for use_gsc in (False, True):
            hit, suc, stl, traj = run(kind, use_gsc)
            res[(kind, use_gsc)] = (hit, suc, stl, traj)
            print(f"{label:<12} {'开' if use_gsc else '关':<5} "
                  f"{hit * 100:>7.1f}% {suc * 100:>7.1f}% {stl * 100:>7.1f}%")
        h0, s0, l0, _ = res[(kind, False)]
        h1, s1, l1, _ = res[(kind, True)]
        print(f"{'':<10} 变化   {(h1 - h0) * 100:>+7.1f} 点 {(s1 - s0) * 100:>+7.1f} 点 "
              f"{(l1 - l0) * 100:>+7.1f} 点\n")

    # ---------------- 图 ----------------
    fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(13, 5.0))

    cen, rad, goal3 = make_scene("concave")
    ax1.add_patch(plotting.plt.Circle((goal3[0].item(), goal3[1].item()), REACH_TOL,
                                      color="#55A868", alpha=0.35, zorder=1))
    for c, r in zip(cen, rad):
        ax1.add_patch(plotting.plt.Circle((c[0].item(), c[1].item()), r.item(),
                                          color="#8C8C8C", alpha=0.65, zorder=2))
    for use_gsc, color, lab in ((False, "#C44E52", "GSC 关"), (True, "#4C72B0", "GSC 开")):
        tr = res[("concave", use_gsc)][3][:, :40, :]     # 抽 40 条画，免得糊成一片
        for i in range(tr.shape[1]):
            ax1.plot(tr[:, i, 0].numpy(), tr[:, i, 1].numpy(),
                     color=color, alpha=0.28, linewidth=0.9)
        ax1.plot([], [], color=color, linewidth=1.8, label=lab)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("凹形（窄缝）：目标在缝后，缝宽 0.8 m")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.3)

    labels = ["凸形\nGSC 关", "凸形\nGSC 开", "凹形\nGSC 关", "凹形\nGSC 开"]
    hits = [res[k][0] * 100 for k in
            (("convex", False), ("convex", True), ("concave", False), ("concave", True))]
    sucs = [res[k][1] * 100 for k in
            (("convex", False), ("convex", True), ("concave", False), ("concave", True))]
    xs = range(4)
    ax2.bar([x - 0.2 for x in xs], hits, width=0.4, color="#C44E52", label="碰撞率")
    ax2.bar([x + 0.2 for x in xs], sucs, width=0.4, color="#55A868", label="到达率")
    for x, (h, s) in enumerate(zip(hits, sucs)):
        ax2.text(x - 0.2, h + 1.5, f"{h:.0f}%", ha="center", fontsize=9)
        ax2.text(x + 0.2, s + 1.5, f"{s:.0f}%", ha="center", fontsize=9)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("百分比 (%)")
    ax2.set_ylim(0, 112)
    ax2.set_title("同一段 GSC 代码，两种地形")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("几何安全校正：凸形障碍上有效，凹形障碍上失效", y=1.0)
    plotting.save(fig, "e_gsc_scene.png")


if __name__ == "__main__":
    main()
