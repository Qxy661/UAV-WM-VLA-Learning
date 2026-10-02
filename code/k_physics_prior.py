"""
Demo K — 物理残差损失，能救回多少分布外精度？

对应文档：docs/02-世界模型专题/05-无人机世界模型综述.md

要验证的结论：5.3 节列了物理引导的四条优势，其中「数据效率」和「外推能力」
是两条可以量的。本节在 quad_sim 上把它们拆开量：

  1. 物理残差的定义只用到状态和动作，**不需要标签**。所以它除了能施加在
     训练状态上，还能凭空施加在任意随机采样的分布外状态上——这是它区别于
     数据损失的唯一机制。
  2. 因此设计三组对照：纯数据 / 残差只加在训练状态 / 残差再加在随机
     分布外状态。中间那组是为了排除「加了正则项所以更好」这种平凡解释。
  3. quad_sim 里严格成立、且不依赖未知受力的三条关系是
         p'   - p   = v·dt              （位置积分）
         rpy' - rpy = a[1:]·ωmax·dt     （姿态积分，倾角未触限时）
         ω'   = a[1:]·ωmax               （角速率环的一阶响应）
     残差只写这三条，速度通道没有先验——受力未知，硬写就是抄答案。
  4. 最后要看的不是「物理先验有没有用」，而是**它的作用范围到哪为止**。
     所以除了单步外推，还要把开环 rollout 摆出来：单步精度不等于长期精度。

运行：py -3.9 code/k_physics_prior.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting
from common.quad_sim import OMEGA_MAX, TILT_LIMIT, QuadSim, expert

SEED = 0
DT = 0.02                   # 仿真步长，50 Hz，与 quad_sim 的默认值一致
HIDDEN = 64
STEPS = 1200                # 每个模型的训练步数
BATCH = 256
LR = 3e-3
LAMBDA = 1.0                # 物理残差项的权重
T_TRAJ = 150                # 每条轨迹的长度（步），150 * 0.02 s = 3 s
N_TRAJ = 32                 # 主实验的训练轨迹数
N_OOD = 2048                # 随机分布外状态数（只喂残差，不需要标签）
V_REF = 4.0                 # 速度归一化尺度，同时用作残差的无量纲化尺度
S_SCALE = torch.tensor([3.0, 3.0, 3.0, V_REF, V_REF, V_REF,
                        0.7, 0.7, 0.7, 5.0, 5.0, 5.0])
CH = ["位置", "速度", "姿态", "角速率"]
CH_UNIT = ["m", "m/s", "°", "rad/s"]
CH_SLICE = [(0, 3), (3, 6), (6, 9), (9, 12)]
COVERED = [True, False, True, True]   # 残差写死了哪几路

# 训练用的目标点盒子（小）与分布外测试的目标点（远）
TRAIN_GOAL_XY, TRAIN_GOAL_Z = 0.6, (0.4, 0.9)
OOD_GOAL = (3.0, 3.0, 2.5)


def collect(n_traj, T, seed):
    """用专家控制器采一批轨迹。目标点在小盒子里随机取，状态因此都靠近悬停。"""
    g = torch.Generator().manual_seed(seed)
    env = QuadSim(n_traj, dt=DT)
    env.reset()
    goal = torch.stack([
        (torch.rand(n_traj, generator=g) * 2 - 1) * TRAIN_GOAL_XY,
        (torch.rand(n_traj, generator=g) * 2 - 1) * TRAIN_GOAL_XY,
        TRAIN_GOAL_Z[0] + torch.rand(n_traj, generator=g) * (TRAIN_GOAL_Z[1] - TRAIN_GOAL_Z[0]),
    ], dim=-1)
    obs, act, nxt = [], [], []
    for _ in range(T):
        s = env.obs()
        a = expert(env, goal)
        obs.append(s)
        act.append(a)
        nxt.append(env.step(a))
    return torch.stack(obs, 1), torch.stack(act, 1), torch.stack(nxt, 1)


def rand_ood(n, seed):
    """随机分布外状态：速度、倾角、角速率都远超训练分布。残差不需要标签。"""
    g = torch.Generator().manual_seed(seed)
    s = torch.cat([
        (torch.rand(n, 3, generator=g) * 2 - 1) * 3.0,
        (torch.rand(n, 3, generator=g) * 2 - 1) * 5.0,
        (torch.rand(n, 3, generator=g) * 2 - 1) * 0.6,
        (torch.rand(n, 3, generator=g) * 2 - 1) * 5.0], -1)
    a = torch.rand(n, 4, generator=g) * 2 - 1
    return s, a


def true_delta(s, a):
    """把 (s, a) 原样喂给真仿真，拿真实的一步状态增量。用于单步外推评测。"""
    env = QuadSim(s.shape[0], dt=DT)
    env.p, env.v, env.rpy, env.om = s[:, 0:3], s[:, 3:6], s[:, 6:9], s[:, 9:12]
    return env.step(a) - s


def truth_rollout(n, init, goal, T):
    """在真仿真里跑一条开环轨迹，顺便把专家动作序列记下来。

    动作序列只由真状态决定，之后原样喂给学习到的模型，
    这样两条轨迹的唯一差别就是动力学模型本身。
    """
    env = QuadSim(n, dt=DT)
    env.p, env.v, env.rpy, env.om = init
    traj, acts = [env.obs()], []
    for _ in range(T):
        a = expert(env, goal)
        acts.append(a)
        traj.append(env.step(a))
    return torch.stack(traj, 1), torch.stack(acts, 0)


class DeltaModel(nn.Module):
    """学一阶状态增量 Δs = s' - s。输入按 S_SCALE 归一化，输出保持物理量纲。

    输出层乘上训练集里每个通道的增量标准差：否则数据损失会被增量的量级差异
    吃掉——位置通道的增量只有毫米级，速度通道是 0.1 级，差两个数量级，均方
    误差基本只看得见后者。
    """

    def __init__(self, scale, hidden=HIDDEN):
        super().__init__()
        self.register_buffer("scale", scale)
        self.net = nn.Sequential(
            nn.Linear(16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 12))

    def forward(self, s, a):
        return self.net(torch.cat([s / S_SCALE, a], -1)) * self.scale


def residual(ds_pred, s, a):
    """物理残差：只写与受力无关、因而必然成立的三条关系。

        p'   - p   - v·dt            = 0
        rpy' - rpy - a[1:]·ωmax·dt   = 0
        ω'   - a[1:]·ωmax            = 0

    每项按自身的量纲尺度归一化，所以 LAMBDA = 1 是有意义的权重。
    速度通道不在其中：受力未知，这一路只能交给数据。
    """
    dp = ds_pred[:, 0:3] - s[:, 3:6] * DT
    dr = ds_pred[:, 6:9] - a[:, 1:] * OMEGA_MAX * DT
    do = ds_pred[:, 9:12] - (a[:, 1:] * OMEGA_MAX - s[:, 9:12])
    return ((dp / (V_REF * DT)).pow(2).mean() + (dr / (OMEGA_MAX * DT)).pow(2).mean()
            + (do / OMEGA_MAX).pow(2).mean())


def train_model(states, actions, next_states, lam=0.0, ood=None, seed=SEED):
    """训练一个动力学模型。lam=0 即纯数据损失；ood 非空时残差也施加在随机分布外状态上。"""
    torch.manual_seed(seed)
    target = next_states - states
    model = DeltaModel(target.std(0).clamp(min=1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1)
    n = states.shape[0]
    s_ood, a_ood = ood if ood is not None else (None, None)
    for _ in range(STEPS):
        i = torch.randint(0, n, (BATCH,), generator=g)
        pred = model(states[i], actions[i])
        loss = (pred - target[i]).pow(2).mean()
        if lam > 0:
            reg = residual(pred, states[i], actions[i])
            if s_ood is not None:
                j = torch.randint(0, s_ood.shape[0], (BATCH,), generator=g)
                reg = reg + residual(model(s_ood[j], a_ood[j]), s_ood[j], a_ood[j])
            loss = loss + lam * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def step_errors(model, s, a, true_ds):
    """单步预测误差，逐通道返回（各自物理单位）。"""
    d = (model(s, a) - true_ds).abs()
    return [d[:, i:j].mean().item() * (180.0 / math.pi if u == "°" else 1.0)
            for (i, j), u in zip(CH_SLICE, CH_UNIT)]


@torch.no_grad()
def rollout_curve(model, s0, actions, true_traj):
    """开环 rollout 每一步的位置误差（对所有环境取平均），返回长度 T+1 的列表。"""
    s, out = s0, [s0]
    for t in range(actions.shape[0]):
        s = s + model(s, actions[t])
        out.append(s)
    pred = torch.stack(out, 1)
    return (pred[:, :, 0:3] - true_traj[:, :, 0:3]).norm(dim=-1).mean(0)


def synth_traj_set(n, seed):
    """一组分布内的测试轨迹：起点与训练同分布，开环喂同一串专家动作。"""
    g = torch.Generator().manual_seed(seed)
    init = (torch.stack([(torch.rand(n, generator=g) * 2 - 1) * 0.2,
                         (torch.rand(n, generator=g) * 2 - 1) * 0.2,
                         torch.zeros(n)], -1),
            (torch.rand(n, 3, generator=g) * 2 - 1) * 0.3,
            (torch.rand(n, 3, generator=g) * 2 - 1) * 0.1,
            torch.zeros(n, 3))
    goal = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * TRAIN_GOAL_XY,
                        (torch.rand(n, generator=g) * 2 - 1) * TRAIN_GOAL_XY,
                        TRAIN_GOAL_Z[0] + torch.rand(n, generator=g)
                        * (TRAIN_GOAL_Z[1] - TRAIN_GOAL_Z[0])], -1)
    true_traj, actions = truth_rollout(n, init, goal, T_TRAJ)
    return torch.cat(init, -1), true_traj, actions


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    # ---- 数据 ----
    obs, act, nxt = collect(N_TRAJ, T_TRAJ, SEED)
    states, actions, next_states = obs.reshape(-1, 12), act.reshape(-1, 4), nxt.reshape(-1, 12)
    print(f"训练集 {N_TRAJ} 条轨迹 x {T_TRAJ} 步 = {states.shape[0]} 条转移")
    print(f"  训练状态：|v| 上界 {states[:, 3:6].norm(dim=-1).max():.2f} m/s"
          f"，|rpy| 上界 {states[:, 6:9].norm(dim=-1).max():.3f} rad"
          f"，|ω| 上界 {states[:, 9:12].norm(dim=-1).max():.2f} rad")

    s_ood, a_ood = rand_ood(N_OOD, SEED + 7)
    ds_ood = true_delta(s_ood, a_ood)
    print(f"  随机分布外状态 {N_OOD} 条：|v| 上界 {s_ood[:, 3:6].norm(dim=-1).max():.2f} m/s"
          f"，|rpy| 上界 {s_ood[:, 6:9].norm(dim=-1).max():.3f} rad"
          f"，标注成本为 0")

    n_test = 64
    t_state, t_true, t_act = synth_traj_set(n_test, SEED + 13)
    ood_goal = torch.tensor(OOD_GOAL).repeat(n_test, 1)
    g = torch.Generator().manual_seed(SEED + 29)
    ood_init = (torch.stack([(torch.rand(n_test, generator=g) * 2 - 1) * 0.5,
                             (torch.rand(n_test, generator=g) * 2 - 1) * 0.5,
                             torch.rand(n_test, generator=g) * 0.5 + 0.6], -1),
                torch.stack([(torch.rand(n_test, generator=g) * 2 - 1) * 3.0,
                             (torch.rand(n_test, generator=g) * 2 - 1) * 3.0,
                             (torch.rand(n_test, generator=g) * 2 - 1) * 1.0], -1),
                torch.stack([(torch.rand(n_test, generator=g) * 2 - 1) * 0.5,
                             (torch.rand(n_test, generator=g) * 2 - 1) * 0.5,
                             (torch.rand(n_test, generator=g) * 2 - 1) * 0.2], -1),
                torch.zeros(n_test, 3))
    o_state = torch.cat(ood_init, -1)
    o_true, o_act = truth_rollout(n_test, ood_init, ood_goal, T_TRAJ)
    rpy_max = o_true[:, :, 6:9].abs().max().item()

    # ---- 1. 三组对照模型 ----
    print("\n[1] 训练三组模型（同一份数据、同一个初始化种子、同样的步数）")
    cfg = [("A 纯数据", 0.0, False),
           ("B 残差只在训练状态", LAMBDA, False),
           ("C 残差 + 随机分布外状态", LAMBDA, True)]
    models = [(name, train_model(states, actions, next_states, lam=lam,
                                 ood=(s_ood, a_ood) if use_ood else None))
              for name, lam, use_ood in cfg]

    # ---- 2. 单步预测误差 ----
    ds_in = true_delta(states[:1024], actions[:1024])
    print("\n[2] 单步预测误差（分布内 1024 条转移 / 随机分布外状态 1024 条）")
    print(f"    {'':<26s}" + "".join(f"{c:>12s}" for c in CH))
    print(f"    {'单位':<24s}" + "".join(f"{u:>12s}" for u in CH_UNIT))
    err_in, err_ood = {}, {}
    for name, m in models:
        err_in[name] = step_errors(m, states[:1024], actions[:1024], ds_in)
        err_ood[name] = step_errors(m, s_ood[:1024], a_ood[:1024], ds_ood[:1024])
        print(f"    {name + ' 分布内':<22s}" + "".join(f"{v:12.2e}" for v in err_in[name]))
        print(f"    {name + ' 分布外':<22s}" + "".join(f"{v:12.2e}" for v in err_ood[name]))

    # ---- 3. 先验的覆盖范围 ----
    base = err_ood["A 纯数据"]
    print("\n[3] 分布外单步误差相对于纯数据模型的比值（< 1 表示这一路被先验救回来了）")
    print(f"    {'':<22s}" + "".join(f"{c:>12s}" for c in CH))
    ratio = {}
    for name in ("B 残差只在训练状态", "C 残差 + 随机分布外状态"):
        ratio[name] = [x / max(y, 1e-12) for x, y in zip(err_ood[name], base)]
        print(f"    {name:<20s}" + "".join(f"{v:12.2f}" for v in ratio[name]))
    print(f"    {'残差是否写死这一路':<18s}" + "".join(
        f"{('是' if c else '否'):>12s}" for c in COVERED))
    drop_b = 1 / max(ratio["B 残差只在训练状态"][0], 1e-12)
    drop_c = 1 / max(ratio["C 残差 + 随机分布外状态"][0], 1e-12)
    print(f"    B 与 C 的差别只在残差加在哪批状态上：B 的位置通道只降 {drop_b:.1f} 倍，"
          f"C 降 {drop_c:.1f} 倍。")

    # ---- 4. 数据效率：把数据砍到 1/8 再看一次 ----
    print("\n[4] 数据效率（单步误差，2 个种子平均）——把轨迹数从 32 砍到 4")
    small = (obs[:4].reshape(-1, 12), act[:4].reshape(-1, 4), nxt[:4].reshape(-1, 12))
    print(f"    {'':<26s}{'分布内位置':>14s}{'分布外位置':>14s}{'分布外速度':>14s}")
    for tag, src in (("32 条轨迹（4800 转移）", (states, actions, next_states)),
                     ("4 条轨迹（600 转移）", small)):
        for key, lam, use_ood in (("A 纯数据", 0.0, False), ("C 残差", LAMBDA, True)):
            acc = []
            for sd in (0, 1):
                m = train_model(*src, lam=lam, ood=(s_ood, a_ood) if use_ood else None, seed=sd)
                acc.append(step_errors(m, states[:1024], actions[:1024], ds_in)[0:1]
                           + step_errors(m, s_ood[:1024], a_ood[:1024], ds_ood[:1024])[0:2])
            avg = [sum(a[k] for a in acc) / len(acc) for k in range(3)]
            print(f"    {(tag + ' ' + key):<24s}" + "".join(f"{v:14.2e}" for v in avg))
    print("    数据砍到 1/8，两边的误差几乎没变——这个任务上量不出「物理先验省数据」："
          "\n    600 条转移就已经把模型喂饱了，瓶颈不在数据量。要量数据效率得换一个"
          "\n    更难拟合的任务，或者把网络容量放大到数据成为瓶颈为止。")

    # ---- 5. 开环 rollout：单步精度不等于长期精度 ----
    print(f"\n[5] 开环 rollout（同一串专家动作分别喂给真仿真和学到的模型，{T_TRAJ} 步 = 3 秒）")
    print(f"    分布外触限检查：真值轨迹 |rpy| 最大 {rpy_max:.3f} rad，限幅 {TILT_LIMIT} rad，"
          f"{'未触限，姿态残差成立' if rpy_max < TILT_LIMIT else '有触限，姿态残差失效'}")
    curve_in, curve_ood = {}, {}
    print(f"    {'':<24s}{'分布内 0.5/1/3 s':>26s}{'分布外 0.5/1/3 s':>26s}")
    for name, m in models:
        ci = rollout_curve(m, t_state.clone(), t_act, t_true)
        co = rollout_curve(m, o_state.clone(), o_act, o_true)
        curve_in[name], curve_ood[name] = ci, co
        print(f"    {name:<22s}   {ci[25]:.3f} / {ci[50]:.3f} / {ci[150]:.3f} m"
              f"      {co[25]:.3f} / {co[50]:.3f} / {co[150]:.3f} m")

    print("\n结论：残差项只用到状态和动作，不需要标签，所以能凭空施加在任何随机采样的"
          "\n      分布外状态上。第 [3] 组显示这个手法确实管用，但只对它写死的那几路管用："
          f"\n      位置通道的分布外单步误差降了 {drop_c:.0f} 倍，速度通道纹丝不动。"
          "\n      第 [5] 组是一条警告——单步精度不等于长期精度。分布外 rollout 的前 0.5 秒，"
          "\n      带先验的两组确实领先；但位置每一拍都由 v 积分而来，速度通道没有先验兜底，"
          "\n      它的微小偏差随时间累积，1 秒后被纯数据模型反超，3 秒后落后接近一倍。")

    # ---- 出图 ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = {"A 纯数据": "#c0392b", "B 残差只在训练状态": "#e67e22",
              "C 残差 + 随机分布外状态": "#2471a3"}

    ax = axes[0]
    x = torch.arange(4, dtype=torch.float32)
    for k, name in enumerate(("B 残差只在训练状态", "C 残差 + 随机分布外状态")):
        r = [a / max(b, 1e-12) for a, b in zip(err_ood[name], base)]
        ax.bar(x + (k - 0.5) * 0.38, r, 0.36, color=colors[name], label=name)
    ax.axhline(1.0, color=colors["A 纯数据"], ls="--", lw=1.6, label="A 纯数据（基准 1.0）")
    ax.set_yscale("log"); ax.set_ylim(0.01, 6.0); ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{u}" for c, u in zip(CH, CH_UNIT)], fontsize=8.5)
    # 红色刻度 = 残差没有写这一路，别的都不解释
    for lab, cov in zip(ax.get_xticklabels(), COVERED):
        lab.set_color("#2471a3" if cov else "#c0392b")
    ax.text(2.6, 3.6, "红色刻度：残差没写这一路", ha="center", fontsize=8.5, color="#c0392b")
    ax.set_ylabel("分布外单步误差 / 纯数据模型的误差（对数轴）")
    ax.set_title("(a) 先验只按住它写死的那三路，\n速度通道纹丝不动")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both", axis="y")

    ax = axes[1]
    t = [i * DT for i in range(T_TRAJ + 1)]
    for name, _ in models:
        ax.plot(t, curve_in[name], "-", color=colors[name], lw=1.7, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("开环 rollout 时间（秒）"); ax.set_ylabel("位置误差（m，对数轴）")
    ax.set_title("(b) 分布内 rollout：数据管得到的地方，\n先验帮不上忙")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for name, _ in models:
        ax.plot(t, curve_ood[name], "-", color=colors[name], lw=1.7, label=name)
    ax.axvline(1.0, color="#555", ls=":", lw=1.2)
    ax.text(1.03, 0.03, "1 秒", fontsize=8.5, color="#555")
    ax.set_yscale("log")
    ax.set_xlabel("开环 rollout 时间（秒）"); ax.set_ylabel("位置误差（m，对数轴）")
    ax.set_title("(c) 分布外 rollout：前 0.5 秒先验领先，\n1 秒后被纯数据模型反超")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")

    fig.suptitle("物理残差不需要标签，所以能施加在分布外状态上；代价是它只管自己写到的那几路",
                 fontsize=12)
    fig.tight_layout()
    plotting.save(fig, "k_physics_prior.png")


if __name__ == "__main__":
    main()
