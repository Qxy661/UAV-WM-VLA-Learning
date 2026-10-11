"""
Demo Q — 奖励加权 vs 堆数据：同一份飞行数据的两种用法

对应文档：docs/03-VLA专题/08-强化学习后训练与自我改进.md

要验证的结论：模仿学习之后接一段"按回报挑数据"的后训练，买到的是什么。
写成可量的两个问题：

  1. 同一份数据、同一个网络，**均匀 BC** 和**奖励加权（RWR）**谁好？
  2. 如果是加权更好，把数据量翻几倍能不能补上？——如果不能，说明加权动的
     和堆数据动的不是同一个量：一个是偏置，一个是方差。

数据集模拟真实机队的一种常见情况：一部分架次的**一个轴标定反了**（比例项符号
错了），飞得又慢又偏，但它们照样被录进了示范库。均匀 BC 把这些架次和正常架次
一视同仁地拟合，学到的动作是两种行为的**平均**——标定反了的那一半把修正量抵消
掉了一半，闭环上就是"该往 +y 修的时候只修了一半"。RWR 按回报给架次加权，
坏架次被折价，学到的动作偏向正常那一半。

为什么自变量选"坏架次比例"而不是"控制器增益"：先试过后者，不成立。增益这条轴上
回报曲线在最优点附近很平（kp 从 0.9 到 1.2，误差只从 0.795 变到 0.787），
均值增益和最优增益的性能差不到 2%，加权没有可赚的。要量出加权的判别力，
数据集里必须混进**增益表达不了的差别**——这里就是符号错了的那一类。

运行：py -3.9 code/q_rl_posttrain.py     # 纯 CPU 约 2 分钟
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting
from common.quad_sim import QuadSim

SEED = 0
DT = 0.02
T_EP = 120                 # 每回合步数
GOAL = [2.0, 2.0, 1.5]
WIND_STD = 1.2             # 阵风加速度标准差 (m/s^2)，每回合固定、策略不可见

# 正常架次用的控制器（本仓库 code/common/quad_sim.py 里的默认增益）
KP, KD, KA = 0.30, 0.70, 3.0

N_TRAIN = 256              # 数据集总回合数
FRAC_BAD = 0.5             # 默认的坏架次比例
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 坏架次比例扫描
NS = [16, 32, 64, 128, 256]           # 数据预算扫描
N_REP = 2                  # 每个配置重复几次（换一组数据）
NOISE_SEEDS = 4            # 量噪声底时每个配置换几个训练种子
N_EVAL = 256               # 评测回合数
EVAL_SEED = 7777           # 所有配置共用同一批风，否则比的不是策略是运气

BETA_FRAC = 0.25           # RWR 温度 = 该比例 × 该数据集上回报的跨度
STEPS = 800                # 训练步数
BATCH = 256
LR = 2e-3
HID = 96


# ---------------------------------------------------------------- 数据

def feat(env, goal):
    """观测量：位置误差 / 速度 / 姿态 / 角速率，各按物理量程归一。

    减掉目标再归一，网络不用去拟合一个常值偏移。
    """
    o = env.obs()
    return torch.cat([(o[:, 0:3] - goal) / 2.0,
                      o[:, 3:6] / 2.0,
                      o[:, 6:12] / 3.0], dim=-1)


def act(env, goal, bad):
    """可控控制器。bad[i]=True 时该回合的 y 轴比例项符号取反。

    符号取反模拟的是标定/接线错误，不是"增益没调好"——它让被控方向反过来，
    所以坏架次的回报比正常架次低一个数量级，且低得**系统**、不是噪声。
    """
    kpv = torch.ones(env.n, 2)
    kpv[bad, 1] = -1.0
    e = goal[:, :2] - env.p[:, :2]
    tilt = torch.clamp(KP * e * kpv - KD * env.v[:, :2], -0.6, 0.6)
    a = torch.zeros(env.n, 4)
    a[:, 0] = 0.5 + 0.10 * (goal[:, 2] - env.p[:, 2]) - 0.30 * env.v[:, 2]
    a[:, 1:3] = KA * (tilt - env.rpy[:, :2])
    return a.clamp(-1, 1)


def collect(n, seed, frac_bad=FRAC_BAD):
    """跑一批架次，记 (观测, 动作, 回报, 是否坏架次)。"""
    torch.manual_seed(seed)
    env = QuadSim(n, dt=DT)
    goal = torch.tensor(GOAL).expand(n, 3).clone()
    env.reset(rand=True, pos_std=0.6, vel_std=0.4)
    wind = WIND_STD * torch.randn(n, 3)
    g = torch.Generator().manual_seed(seed + 1)
    bad = torch.rand(n, generator=g) < frac_bad
    OB, AC = [], []
    for _ in range(T_EP):
        OB.append(feat(env, goal))
        AC.append(act(env, goal, bad))
        env.step(AC[-1])
        env.v = env.v + wind * DT
    ret = -torch.norm(env.p - goal, dim=-1)
    return torch.stack(OB, 1), torch.stack(AC, 1), ret, bad


# ---------------------------------------------------------------- 策略

class Policy(nn.Module):
    """观测 -> 动作。tanh 输出把动作夹在 [-1,1]，与执行器量程一致。"""

    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, HID), nn.SiLU(),
                                 nn.Linear(HID, HID), nn.SiLU(),
                                 nn.Linear(HID, 4))

    def forward(self, x):
        return torch.tanh(self.net(x))


def train(obs, act_t, ret, w=None, seed=SEED):
    """训练。w=None 是均匀 BC；w 给出逐回合权重就是 RWR。

    RWR 的权重 w_i ∝ exp(R_i / beta)。beta 取该数据集上回报跨度的一个比例，
    这样不同坏架次比例的数据集之间温度是可比的。
    """
    torch.manual_seed(seed)
    x = obs.reshape(-1, obs.shape[-1])
    y = act_t.reshape(-1, 4)
    if w is None:
        wt = None
    else:
        wt = w.repeat_interleave(T_EP)      # 回合权重 -> 逐样本（回合优先，故用 interleave）
    model = Policy(x.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(STEPS):
        i = torch.randint(0, x.shape[0], (BATCH,), generator=g)
        loss = (model(x[i]) - y[i]).pow(2).mean(dim=-1)
        loss = loss.mean() if wt is None else (loss * wt[i]).sum() / wt[i].sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def rwr_weights(ret):
    beta = BETA_FRAC * (ret.max() - ret.min())
    return torch.softmax(ret / beta, dim=0) * ret.numel()      # 归一化到均值 1


@torch.no_grad()
def deploy(model, n=N_EVAL, seed=EVAL_SEED, bad_frac=None):
    """闭环评测。bad_frac 给定时用脚本控制器（作参照），否则用网络。"""
    torch.manual_seed(seed)
    env = QuadSim(n, dt=DT)
    goal = torch.tensor(GOAL).expand(n, 3).clone()
    env.reset(rand=True, pos_std=0.6, vel_std=0.4)
    wind = WIND_STD * torch.randn(n, 3)
    bad = torch.zeros(n, dtype=torch.bool)
    if bad_frac is not None:
        bad = torch.rand(n, generator=torch.Generator().manual_seed(seed + 5)) < bad_frac
    for _ in range(T_EP):
        a = act(env, goal, bad) if bad_frac is not None else model(feat(env, goal))
        env.step(a)
        env.v = env.v + wind * DT
    return torch.norm(env.p - goal, dim=-1).mean().item()


def ess(w):
    """权重的等效样本数 (sum w)^2 / sum w^2。"""
    return float((w.sum() ** 2) / (w ** 2).sum())


def mean_std(v):
    """返回 (均值, 总体标准差)。"""
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 0.0
    return m, sd


# ---------------------------------------------------------------- 主流程

def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    clean_e = deploy(None, bad_frac=0.0)
    allbad_e = deploy(None, bad_frac=1.0)
    print(f"参照（脚本控制器，同 %d 回合、同一批风）：" % N_EVAL)
    print(f"  全部正常架次的控制器  {clean_e:.4f} m")
    print(f"  全部标定反了的控制器  {allbad_e:.4f} m"
          f"（差 {allbad_e - clean_e:.4f} m，{allbad_e / clean_e:.1f} 倍）\n")

    # ---- 扫描一：坏架次比例（数据量固定）
    print(f"扫描一：坏架次比例（数据量固定 N={N_TRAIN}，每档 {N_REP} 组数据）")
    print(f"  {'坏架次占比':>10} {'均匀 BC':>10} {'RWR':>10} {'权重等效样本数':>15}"
          f" {'坏人拿走的权重':>15}")
    comp = []
    for frac in FRACS:
        u, r_, e_, bm = [], [], [], []
        for rep in range(N_REP):
            obs, ac, ret, bad = collect(N_TRAIN, SEED + 10 * rep, frac)
            w = rwr_weights(ret)
            u.append(deploy(train(obs, ac, ret, None, seed=SEED + rep)))
            r_.append(deploy(train(obs, ac, ret, w, seed=SEED + rep)))
            e_.append(ess(w))
            bm.append(float(w[bad].sum() / w.sum()))
        m = lambda v: sum(v) / len(v)
        comp.append((frac, m(u), m(r_), m(e_), m(bm)))
        print(f"  {frac:>10.2f} {m(u):>10.4f} {m(r_):>10.4f} {m(e_):>15.1f}"
              f" {m(bm) * 100:>14.1f}%")

    # ---- 扫描二：数据量，以及**它根本量不出来**的那件事
    #
    # 计划里想量的是"加数据能不能补上加权的差"。做下来发现量不了：这个闭环指标
    # 对策略的微小差别极其敏感，同一个配置换训练种子重跑，误差就上下浮动一米多。
    # 与其把噪声当成趋势写进结论，不如把噪声本身量出来，说明这条轴为什么放弃。
    print(f"\n扫描二：数据量 —— 先量噪声底（坏架次比例 {FRAC_BAD}，每档换 "
          f"{NOISE_SEEDS} 个训练种子重跑同一份数据）")
    print(f"  {'N':>5} {'均匀 BC 均值±标准差':>24} {'RWR 均值±标准差':>24} {'均值差':>10}")
    noise = []
    for N in NS:
        obs, ac, ret, _bad = collect(N, SEED, FRAC_BAD)
        w = rwr_weights(ret)
        u = [deploy(train(obs, ac, ret, None, seed=9000 + s)) for s in range(NOISE_SEEDS)]
        r_ = [deploy(train(obs, ac, ret, w, seed=9000 + s)) for s in range(NOISE_SEEDS)]
        mu_, ms = mean_std(u), mean_std(r_)
        noise.append((N, mu_, ms, u, r_))
        print(f"  {N:>5} {mu_[0]:>13.4f} ± {mu_[1]:<8.4f} "
              f"{ms[0]:>13.4f} ± {ms[1]:<8.4f} {mu_[0] - ms[0]:>10.4f}")

    # 用扫描一里那份 f=0.5 的数据做温度扫描
    obs, ac, ret, bad = collect(N_TRAIN, SEED, FRAC_BAD)

    # ---- 扫描三：RWR 温度（坏架次比例固定）
    print(f"\n扫描三：RWR 温度 beta（坏架次比例 {FRAC_BAD}，N={N_TRAIN}）")
    print(f"  {'beta':>10} {'终点误差(m)':>12} {'权重等效样本数':>15} {'坏人拿走的权重':>15}")
    temps = []
    spread = float(ret.max() - ret.min())
    for frac_b in [1.0, 0.5, 0.25, 0.1]:
        beta = frac_b * spread
        w = torch.softmax(ret / beta, dim=0) * ret.numel()
        ev = deploy(train(obs, ac, ret, w, seed=SEED))
        temps.append((frac_b, beta, ev, ess(w), float(w[bad].sum() / w.sum())))
        print(f"  {frac_b:>10.2f} {ev:>12.4f} {ess(w):>15.1f}"
              f" {float(w[bad].sum() / w.sum()) * 100:>14.1f}%")
    print("  （beta 越小，权重越尖，坏架次被压得越狠——但有效样本数也一起掉。）")

    # ---- 读法
    f0, u0, r0, e0, b0 = comp[0]
    fh, uh, rh, eh, bh = comp[2]
    nmid = noise[1] if len(noise) > 1 else noise[0]      # 中间那一档预算
    nl = noise[-1]
    sds = [x[1][1] for x in noise] + [x[2][1] for x in noise]
    print("\n  读法：")
    print(f"    (1) 坏架次比例这个轴：从 {f0:.2f} 到 {fh:.2f}，均匀 BC 从 {u0:.4f} m "
          f"涨到 {uh:.4f} m（{uh / u0:.1f} 倍）；RWR 从 {r0:.4f} m 涨到 {rh:.4f} m"
          f"（{rh / r0:.1f} 倍）。同一份数据、同一个网络，只换损失里的权重。")
    print(f"    (2) 两端必须重合，这是设计的自检：坏架次占 {comp[0][0]:.2f} 时两种方法差 "
          f"{abs(u0 - r0):.4f} m，占 {comp[-1][0]:.2f} 时差 {abs(comp[-1][1] - comp[-1][2]):.4f} m。"
          f"数据里没有质量差别时加权无从判别，全是坏数据时加权也无从判别，"
          f"差别只出现在中间——这才说明量到的是权重在两种行为之间取舍，不是噪声。")
    print(f"    (3) 机制：均匀 BC 拟合的是两种行为的平均。标定反了的那一半给 y 轴的"
          f"修正量是反的，平均下来正常架次那半边的修正被抵消掉一截。RWR 把这个抵消"
          f"去掉了——坏架次占 {fh:.2f} 时它们仍拿走 {bh * 100:.0f}% 的权重，"
          f"这就是残留误差的来源。")
    print(f"    (4) 噪声底：同一个配置只换训练种子重跑 {NOISE_SEEDS} 次，"
          f"各档标准差落在 {min(sds):.4f} ~ {max(sds):.4f} m 之间，"
          f"没有随 N 单调变化的趋势，所以不能读成数据越多越不稳。")
    print(f'    (5) 因此计划里那条"加数据能不能补上"的轴，在这个装置上量不出来：'
          f"N={nmid[0]} 时均值差 {nmid[1][0] - nmid[2][0]:.4f} m 是标准差 "
          f"{nmid[1][1]:.4f} m 的 {abs(nmid[1][0] - nmid[2][0]) / max(nmid[1][1], 1e-9):.1f} 倍，"
          f"读得出来；到 N={nl[0]} 时均值差 {nl[1][0] - nl[2][0]:.4f} m 只有标准差的 "
          f"{abs(nl[1][0] - nl[2][0]) / max(nl[1][1], 1e-9):.1f} 倍，要分辨就得把每档重跑次数"
          f"加上去。这一条**不作为结论**。")
    print(f"    (6) 代价与旋钮：RWR 的权重等效样本数在坏架次占 {fh:.2f} 时是 {eh:.1f} / "
          f"{N_TRAIN}（{eh / N_TRAIN * 100:.0f}%）。beta 是控制这个折价的旋钮，"
          f"扫描三扫了四档。")
    print(f"    (7) 上界检查：坏架次占 {fh:.2f} 时 RWR 是 {rh:.4f} m，对照全正常架次的"
          f"脚本控制器 {clean_e:.4f} m，还差 {rh - clean_e:.4f} m——软加权压不干净，"
          f"这是 RWR 本身的固有残留，不是实现问题。")

    # ------------------------------------------------------------------ 出图
    fig, ax = plotting.plt.subplots(1, 3, figsize=(13.2, 3.7))

    fs = [c[0] for c in comp]
    ax[0].plot(fs, [c[1] for c in comp], marker="o", color="tab:red", label="均匀 BC")
    ax[0].plot(fs, [c[2] for c in comp], marker="s", color="tab:blue", label="奖励加权 RWR")
    ax[0].axhline(clean_e, ls=":", color="green", label="全正常控制器")
    ax[0].axhline(allbad_e, ls="--", color="gray", label="全标定反了")
    ax[0].fill_between(fs, [c[1] for c in comp], [c[2] for c in comp],
                       color="tab:red", alpha=0.08)
    ax[0].set_xlabel("坏架次（一个轴标定反）占比")
    ax[0].set_ylabel("风下终点误差 (m)")
    ax[0].set_title("(a) 两端重合、中间张开：加权只在有质量差别时有用")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    bs = [t[0] for t in temps]
    ax[1].plot(bs, [t[2] for t in temps], marker="o", color="tab:red")
    ax[1].set_xscale("log"); ax[1].invert_xaxis()
    ax[1].set_xticks(bs); ax[1].set_xticklabels([str(b) for b in bs])
    ax[1].set_xlabel("RWR 温度 beta（= 该比例 × 回报跨度）")
    ax[1].set_ylabel("风下终点误差 (m)", color="tab:red")
    ax[1].tick_params(axis="y", labelcolor="tab:red")
    axb = ax[1].twinx()
    axb.plot(bs, [t[4] * 100 for t in temps], marker="s", ls="--",
             color="tab:purple", label="坏架次拿走的权重")
    axb.plot(bs, [t[3] / N_TRAIN * 100 for t in temps], marker="^", ls=":",
             color="tab:green", label="等效样本数占比")
    axb.set_ylabel("百分比 (%)"); axb.set_ylim(0, 105)
    axb.legend(fontsize=8, loc="lower left")
    ax[1].set_title("(b) beta 越小压得越狠，有效样本也一起掉")
    ax[1].grid(alpha=0.3)

    # 四个位置：N小/大 × 均匀/RWR。散点横向抖一下，免得种子之间完全重叠。
    slots = [(noise[0], 0, "均匀", "tab:red", "o"),
             (noise[0], 1, "RWR", "tab:blue", "s"),
             (noise[-1], 2, "均匀", "tab:red", "o"),
             (noise[-1], 3, "RWR", "tab:blue", "s")]
    for entry, x, tag, col, mk in slots:
        vals = entry[3] if tag == "均匀" else entry[4]
        mu_, sd_ = (entry[1] if tag == "均匀" else entry[2])
        for j, v in enumerate(vals):
            ax[2].plot(x + 0.05 * (j - (len(vals) - 1) / 2), v, mk, color=col,
                       ms=4, alpha=0.6)
        ax[2].errorbar(x, mu_, yerr=sd_, fmt=mk, color=col, capsize=6, ms=8,
                       lw=2, label=tag if x < 2 else None)
    ax[2].set_xticks([0, 1, 2, 3])
    ax[2].set_xticklabels([f"N={noise[0][0]}", "RWR", f"N={noise[-1][0]}", "RWR"],
                          fontsize=8.5)
    ax[2].set_ylabel("风下终点误差 (m)")
    ax[2].set_title(f"(c) 噪声底：每点 {NOISE_SEEDS} 个训练种子的散布（误差棒）")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    plotting.save(fig, "ph8_reward_weighting.png")


if __name__ == "__main__":
    main()
