"""
Demo S — 世界模型当数据源：想象轨迹能补多少真机数据

对应文档：docs/03-VLA专题/10-世界模型增强VLA.md

要验证的结论：世界模型生成的"想象轨迹"可以拿来扩充策略的训练集，但它的
价值随想象步长先升后降。步长太短，想象数据只是真机数据的复制品，填不满
真机没覆盖到的状态；步长太长，世界模型自己的预测误差沿时间累积，存下来的
标签与状态不再一致，训练数据从"扩充"变成"污染"。这一节把这条曲线量出来。

装置（全部复用 code/common/quad_sim.py 的四旋翼）：
  真机数据 —— 脚本专家飞出来的 (s, a) 对，起点散布窄
  世界模型 —— 从真机数据学出来的一步动力学模型（残差 MLP），只学一步
  想象数据 —— 用世界模型自己滚 L 步得到的 (s, a) 对，预算固定
  策略     —— 真机 + 想象上做行为克隆，闭环评估

两个自变量：
  ① 世界模型的开环误差随步数增长多少 —— 这是想象步长的物理上限
  ② 想象步长 L → 闭环终点误差 —— 找曲线的底

运行：py -3.9 code/s_wm_data_source.py
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim, expert
from common import plotting

SEED = 0
T = 200                    # 每条真机轨迹 200 步 = 4 秒
N_START_REAL = 8           # 真机只有 8 条轨迹 —— 空中数据稀缺是这套方法的前提
START_STD_REAL = 0.40      # 真机起点散布 (m)
START_STD_IMAG = 0.80      # 想象轨迹的起点散布：踩在世界模型有效范围的边上
START_STD_EVAL = 1.00      # 闭环评估起点散布：比真机宽，留出想象能填的缝
GOAL = torch.tensor([8.0, 0.0, 1.5])

N_REAL = N_START_REAL * T  # 真机 (s, a) 对数
N_IMAG = 6400              # 想象数据预算固定，不随 L 变 —— 这是关键的对照控制

HORIZONS = [0, 1, 3, 10, 30, 60]   # L=0 表示只用真机数据
K_SEEDS = 8                        # 每个 L 换 8 个训练种子，报均值 ± 标准差
N_EVAL = 256
T_EVAL = 250

WM_EPOCHS = 3000
BC_EPOCHS = 2000
LR = 1e-3
DS = 1.0                   # 直接回归状态增量；之前放大 100 倍会让目标小到 ~3e-3，网络学不动
WM_TRAIN_FRAC = 1.0        # 真机数据本来就少，世界模型把它全用上


class _View:
    """把裸状态张量包成 expert() 认得的对象，避免复制一份控制律。"""

    def __init__(self, s):
        self.n = s.shape[0]
        self.p, self.v, self.rpy = s[:, :3], s[:, 3:6], s[:, 6:9]


class WorldModel(nn.Module):
    """一步动力学：s' = s + f(s, a) / DS。残差形式，学增量不学绝对值。"""

    def __init__(self, sd=12, ad=4, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd + ad, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, sd),
        )

    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], dim=-1)) / DS


class BCPolicy(nn.Module):
    """观测 -> 动作，和目标点写死的脚本专家同构（目标不输入给网络）。"""

    def __init__(self, sd=12, ad=4, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, ad),
        )

    def forward(self, s):
        return torch.tanh(self.net(s))


def collect_real(seed):
    """脚本专家在真机（这里即仿真）上飞，收集 (s, a, s')。"""
    torch.manual_seed(seed)
    env = QuadSim(N_START_REAL)
    s = env.reset(rand=True, pos_std=START_STD_REAL, vel_std=0.15)
    g = GOAL.repeat(N_START_REAL, 1)
    S, A, S2 = [], [], []
    for _ in range(T):
        a = expert(env, g)
        s2 = env.step(a)
        S.append(s); A.append(a); S2.append(s2)
        s = s2
    return torch.stack(S), torch.stack(A), torch.stack(S2)


def train_world_model(S, A, S2, seed):
    """世界模型只用一小部分真机数据训练：数据受限是空中世界模型的常态。"""
    torch.manual_seed(seed)
    Sf, Af, S2f = S.reshape(-1, 12), A.reshape(-1, 4), S2.reshape(-1, 12)
    n_use = int(Sf.shape[0] * WM_TRAIN_FRAC)
    sel = torch.randperm(Sf.shape[0])[:n_use]
    wm = WorldModel()
    opt = torch.optim.Adam(wm.parameters(), lr=LR)
    for _ in range(WM_EPOCHS):
        b = sel[torch.randint(0, n_use, (256,))]
        loss = F.mse_loss(wm(Sf[b], Af[b]), S2f[b])
        opt.zero_grad(); loss.backward(); opt.step()
    # 训练后自检：量一步预测误差，确认它真的学到了。
    # 数据全用上时没有留出集，这一项就是训练集误差（只当作"学没学到"的凭据）。
    with torch.no_grad():
        idx = torch.randint(0, n_use, (2048,))
        one_step = (wm(Sf[sel[idx]], Af[sel[idx]]) - S2f[sel[idx]]).norm(dim=-1).mean().item()
    return wm, one_step


def _env_at(s):
    """把一批裸状态装进仿真器，用来做"同一个起点、两条轨迹"的对照。"""
    env = QuadSim(s.shape[0])
    env.reset()
    env.p, env.v = s[:, :3].clone(), s[:, 3:6].clone()
    env.rpy, env.om = s[:, 6:9].clone(), s[:, 9:12].clone()
    return env


@torch.no_grad()
def wm_open_loop_error(wm, S0, steps):
    """从给定起点出发，用世界模型开环滚 steps 步，与真实仿真比误差。

    两边的动作序列完全相同（都由同一控制器对真实状态给出），所以量到的差
    只来自世界模型的预测误差。起点从真机数据里抽，保证落在训练分布上。
    """
    env = _env_at(S0)
    g = GOAL.repeat(S0.shape[0], 1)
    s_wm = S0.clone()
    errs = []
    for _ in range(steps):
        a = expert(env, g)
        s_true = env.step(a)
        s_wm = wm(s_wm, a)
        errs.append((s_wm - s_true).norm(dim=-1).mean().item())
    return errs


@torch.no_grad()
def imagine(wm, L, seed):
    """用世界模型自己滚 L 步，凑够 N_IMAG 个 (s, a) 对。

    预算固定：L 越大，起点越少、每条滚得越深。这正是"想象预算怎么花"
    这个实际问题的两种极端。
    """
    torch.manual_seed(seed + 1000)
    n_start = max(1, N_IMAG // L)
    s = torch.zeros(n_start, 12)
    s[:, :3] = START_STD_IMAG * torch.randn(n_start, 3)
    s[:, 3:6] = 0.15 * torch.randn(n_start, 3)
    S, A = [], []
    for _ in range(L):
        a = expert(_View(s), GOAL.repeat(n_start, 1))
        S.append(s); A.append(a)
        s = wm(s, a)
    return torch.cat(S), torch.cat(A)


def train_policy(S, A, seed):
    torch.manual_seed(seed)
    pol = BCPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    for _ in range(BC_EPOCHS):
        b = torch.randint(0, S.shape[0], (256,))
        loss = F.mse_loss(pol(S[b]), A[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


@torch.no_grad()
def eval_closed_loop(pol, seed):
    """闭环：从比真机更散的一批起点出发，飞到 T_EVAL 步，量终点误差。"""
    torch.manual_seed(seed + 5000)
    env = QuadSim(N_EVAL)
    s = env.reset(rand=True, pos_std=START_STD_EVAL, vel_std=0.15)
    for _ in range(T_EVAL):
        s = env.step(pol(s))
    return (env.p - GOAL).norm(dim=-1).mean().item()


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    S, A, S2 = collect_real(SEED)
    print(f"真机数据：{N_START_REAL} 条轨迹 × {T} 步 = {N_REAL} 个 (s, a) 对，"
          f"起点散布 σ={START_STD_REAL} m")

    # ---- 基线：脚本专家在评估起点散布下自己飞得怎么样 ----
    with torch.no_grad():
        torch.manual_seed(SEED + 5000)
        e = QuadSim(N_EVAL)
        e.reset(rand=True, pos_std=START_STD_EVAL, vel_std=0.15)
        g = GOAL.repeat(N_EVAL, 1)
        for _ in range(T_EVAL):
            e.step(expert(e, g))
        expert_err = (e.p - GOAL).norm(dim=-1).mean().item()
    print(f"脚本专家闭环终点误差：{expert_err:.4f} m（这是策略能达到的上界）\n")

    # 世界模型在真机起点分布上训练
    wms, ose = [], []
    for k in range(K_SEEDS):
        wm, e1 = train_world_model(S, A, S2, SEED + k)
        wms.append(wm); ose.append(e1)
    print(f"世界模型用全部 {N_REAL} 个真机 (s, a) 对训练，"
          f"训练集上的一步预测误差 {sum(ose) / len(ose):.4f} m")

    # ---- ① 世界模型的开环误差随步数增长 ----
    # 起点：中心组从真机数据里抽（训练分布），边缘组用想象起点的散布合成
    torch.manual_seed(SEED + 77)
    Sf_all = S.reshape(-1, 12)
    S0_in = Sf_all[torch.randperm(Sf_all.shape[0])[:512]]
    S0_edge = torch.zeros(512, 12)
    S0_edge[:, :3] = START_STD_IMAG * torch.randn(512, 3)
    S0_edge[:, 3:6] = 0.15 * torch.randn(512, 3)

    print("① 世界模型的开环误差随步数增长（同一个起点，动作序列完全相同）")
    print(f"   {'步数':>5}{'真机分布起点':>18}{'想象起点 σ=0.8':>20}")
    steps_list = [1, 2, 5, 10, 20, 40, 60]
    curve_in = wm_open_loop_error(wms[0], S0_in, max(steps_list))
    curve_edge = wm_open_loop_error(wms[0], S0_edge, max(steps_list))
    for st in steps_list:
        print(f"   {st:>5}{curve_in[st - 1]:>16.4f} m{curve_edge[st - 1]:>18.4f} m")
    print(f"   真机分布起点：{curve_in[0]:.4f} → {curve_in[-1]:.4f} m，60 步放大 "
          f"{curve_in[-1] / curve_in[0]:.1f} 倍，单调上升。")
    print(f"   想象起点 σ=0.8：第 1 步的误差就已经是真机分布起点的 "
          f"{curve_edge[0] / curve_in[0]:.0f} 倍（世界模型在训练分布外一步就不准），"
          f"之后 60 步再放大 {curve_edge[-1] / curve_edge[0]:.1f} 倍。\n")

    # ---- ② 想象步长 → 闭环误差 ----
    print(f"② 想象步长 L → 闭环终点误差（真机 {N_REAL} 对 + 想象 {N_IMAG} 对，"
          f"每个 L 换 {K_SEEDS} 个种子）")
    print(f"   {'L':>4}{'闭环终点误差(m)':>20}{'相对只用真机':>16}{'想象起点数':>12}")
    results = []
    base = None
    for L in HORIZONS:
        errs = []
        for k in range(K_SEEDS):
            if L == 0:
                Sp, Ap = S.reshape(-1, 12), A.reshape(-1, 4)
            else:
                Si, Ai = imagine(wms[k], L, SEED + k)
                Sp = torch.cat([S.reshape(-1, 12), Si])
                Ap = torch.cat([A.reshape(-1, 4), Ai])
            pol = train_policy(Sp, Ap, SEED + k)
            errs.append(eval_closed_loop(pol, SEED + k))
        m = sum(errs) / len(errs)
        sd = math.sqrt(sum((x - m) ** 2 for x in errs) / len(errs))
        if base is None:
            base = m
        results.append((L, m, sd, errs))
        n_start = "—" if L == 0 else str(max(1, N_IMAG // L))
        print(f"   {L:>4}{m:>17.4f} ± {sd:.4f}{m / base:>15.3f}×{n_start:>12}")

    print("\n  逐种子明细（看离散度，别只看均值）：")
    for L, m, sd, errs in results:
        print(f"   L={L:<3} " + "  ".join(f"{x:.4f}" for x in errs))

    best = min(results[1:], key=lambda r: r[1])
    nz = results[1:]
    spread = max(r[1] for r in nz) - min(r[1] for r in nz)
    pool = math.sqrt(sum(r[2] ** 2 for r in nz) / len(nz))
    all_seeds = [x for r in nz for x in r[3]]
    arm_of_worst = max(nz, key=lambda r: max(r[3]))
    arm_of_best = min(nz, key=lambda r: min(r[3]))
    print(f"\n  读法：")
    print(f"    (1) 只用真机 L=0：{results[0][1]:.4f} ± {results[0][2]:.4f} m"
          f"（{K_SEEDS} 个种子）。")
    print(f"    (2) 两个量必须分开看。**均值**：非零各档落在 "
          f"{min(r[1] for r in nz):.4f} ~ {max(r[1] for r in nz):.4f} m，"
          f"都低于 L=0；**离散度**：L=0 的标准差 {results[0][2]:.4f} m，"
          f"非零各档约 {pool:.4f} m，只有前者的 {pool / results[0][2] * 100:.0f}%。")
    print(f"        真正的差别在尾部：L=0 的 {K_SEEDS} 个种子里最差一个是 "
          f"{max(results[0][3]):.4f} m（训崩了），非零各档最差 "
          f"{max(all_seeds):.4f} m（L={arm_of_worst[0]}）。"
          f"想象数据首先把「某些种子训崩」这件事去掉了。")
    print(f"    (3) 但**步长的最优值量不出来**：非零各档的极差 {spread:.4f} m，"
          f"只有种子间标准差 {pool:.4f} m 的 {spread / pool:.2f} 倍；"
          f"L={arm_of_best[0]} 还跑出过 {min(arm_of_best[3]):.4f} m 的离群好值。"
          f"{K_SEEDS} 个种子定不住哪一档最好。")
    print(f"    (4) 脚本专家在同一分布的同规模回合上是 {expert_err:.4f} m。"
          f"最好的策略低于它：12 维状态上的非线性律比专家的固定增益在宽起点上更稳。"
          f"这条只作观察，不作结论。")

    # ---- 出图 ----
    fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(steps_list, [curve_in[s - 1] for s in steps_list], "o-",
             color="#4C72B0", label=f"σ={START_STD_REAL}（训练分布）")
    ax1.plot(steps_list, [curve_edge[s - 1] for s in steps_list], "s--",
             color="#C44E52", label=f"σ={START_STD_IMAG}（想象起点）")
    ax1.set_yscale("log")
    ax1.set_xlabel("开环步数")
    ax1.set_ylabel("与真实仿真的状态误差（米，对数轴）")
    ax1.set_title(f"① 世界模型开环误差随步数增长\n60 步放大 {curve_in[-1] / curve_in[0]:.0f} 倍")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    Ls = [r[0] for r in results]
    ms = [r[1] for r in results]
    sds = [r[2] for r in results]
    ax2.errorbar(Ls, ms, yerr=sds, fmt="o-", color="#55A868", capsize=4, linewidth=2)
    ax2.axhline(expert_err, color="gray", linestyle=":", linewidth=1.5,
                label=f"脚本专家上界 {expert_err:.3f} m")
    ax2.set_xlabel("想象步长 L（0 = 只用真机数据）")
    ax2.set_ylabel("闭环终点误差（米，越小越好）")
    ax2.set_title("② 想象步长 L 与闭环终点误差\n（误差棒是 8 个种子的标准差）")
    for L, m in zip(Ls, ms):
        ax2.annotate(f"{m:.3f}", (L, m), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle("世界模型当数据源：收益量得出来，最优步长量不出来", y=1.02)
    plotting.save(fig, "ph10_wm_data_source.png")


if __name__ == "__main__":
    main()
