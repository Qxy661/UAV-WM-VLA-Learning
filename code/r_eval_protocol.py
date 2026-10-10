"""
Demo R — 同一批回合，换一种报法结论就变了

对应文档：docs/03-VLA专题/09-评测基准与报告口径.md

要验证的结论：评测的结论不只取决于"用哪个基准、跑多少回合"，还取决于
**这批回合被怎么汇总**。本脚本把三件汇总口径的坑各自量一遍：

  1. 用均值还是中位数。在同一批强风回合里，两种策略按均值排名与按中位数
     排名给出**相反**的结论——因为终点误差是重尾的，少数几个飞出去的回合
     把均值整个拖走。
  2. 报多少回合才够。用实测的每回合 0/1 结果，回看置信区间半宽随回合数的
     变化，再拿实测的两策略差距去比：常见的小回合数分辨不了常见的小差距。
  3. 独立单元是回合还是模型。同一个方法跑 4 次独立训练得到 4 个模型，
     把它们的回合全部合起来按二项分布算置信区间，会得到一个比"按模型算"
     窄得多的区间——因为回合之间共享同一个策略，不是独立的。

第 3 条和本仓库 [Demo G](../../code/g_eval_metrics.py) 不是一回事：G 换的是
**指标**（七列排出七套名次），本脚本换的是**汇总口径**（同一列指标、同一批
回合）。

本仓库 [demo Q](../../code/q_rl_posttrain.py) 也量到过第 3 条的近亲：同一个
配置只换训练种子重跑，闭环误差在 0.31 到 1.86 m 之间起伏。

运行：py -3.9 code/r_eval_protocol.py     # 纯 CPU 约 1 分钟
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting
from common.quad_sim import QuadSim, expert

SEED = 0
DT = 0.02
T = 300                    # 6 秒
GOAL = [8.0, 0.0, 1.5]     # 起点在原点附近，直线 8 m
START_STD = 0.6

KP_STD = 0.30              # 与本仓库其余 demo 一致的默认增益
KP_HOT = 1.00              # 高增益
KP_COLD = 0.50             # 第 2 格里那对"差得不小但也说不上大"的增益
KP_WARM = 0.60

WIND_CALM = 0.0            # 无风
WIND_STRONG = 3.0          # 强风加速度标准差 (m/s^2)

TOL = 0.25                 # 到达判定容差 (m)
N_MAIN = 512               # 第 1、2 格的回合数
NS = [16, 32, 64, 128, 256, 512]   # 置信区间回看的回合数
K_SEEDS = 4                # 独立训练次数（第 3 格）
M_PER_SEED = 256           # 每次训练的评测回合数
KP_PER_SEED = [0.30, 0.60, 1.00, 1.80]   # 4 次训练得到 4 个不同增益的模型
Z95 = 1.959964


def rollout(kp, wind_std, n, seed):
    """跑 n 个回合，返回终点误差 (n,)。风每回合固定、策略不可见。"""
    torch.manual_seed(seed)
    env = QuadSim(n, dt=DT)
    env.reset(rand=True, pos_std=START_STD, vel_std=0.3)
    goal = torch.tensor(GOAL).expand(n, 3).clone()
    wind = wind_std * torch.randn(n, 3)
    for _ in range(T):
        env.step(expert(env, goal, kp=kp))
        env.v = env.v + wind * DT
    return torch.norm(env.p - goal, dim=-1)


def mean_median(e):
    return float(e.mean()), float(e.median())


def wilson(k, n):
    """成功率的 Wilson 区间半宽。比正态近似在 p 靠近 0/1 时稳。"""
    if n == 0:
        return 0.0
    p = k / n
    z2 = Z95 ** 2
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return c, h


def halfwidth_at(x01, n, reps=400):
    """在 x01 里无放回抽 n 个回合，复现 k 次，报 Wilson 半宽的均值。"""
    g = torch.Generator().manual_seed(SEED + n)
    hs = []
    for _ in range(reps):
        idx = torch.randperm(x01.numel(), generator=g)[:n]
        hs.append(wilson(int(x01[idx].sum()), n)[1])
    return sum(hs) / len(hs)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")
    print(f"任务：{T * DT:.0f} 秒内飞到 ({GOAL[0]}, {GOAL[1]}, {GOAL[2]})，"
          f"起点散布 σ={START_STD} m｜风：无风 / σ={WIND_STRONG} m/s^2\n")

    # ---- 第 1 格：均值还是中位数
    print("第 1 格：两个策略 × 两种风况，终点误差的两个摘要")
    print(f"  {'策略':<12}{'风况':<8}{'均值(m)':>10}{'中位数(m)':>12}{'均值排名':>10}{'中位数排名':>12}")
    cells = {}
    for tag, kp in (("标准 kp=0.30", KP_STD), ("高增益 kp=1.00", KP_HOT)):
        for wtag, w in (("无风", WIND_CALM), ("强风", WIND_STRONG)):
            e = rollout(kp, w, N_MAIN, seed=SEED + 11)
            cells[(tag, wtag)] = (e, mean_median(e))
    for wtag in ("无风", "强风"):
        a, b = cells[("标准 kp=0.30", wtag)], cells[("高增益 kp=1.00", wtag)]
        rank_m = sorted([("标准 kp=0.30", a[1][0]), ("高增益 kp=1.00", b[1][0])], key=lambda x: x[1])
        rank_d = sorted([("标准 kp=0.30", a[1][1]), ("高增益 kp=1.00", b[1][1])], key=lambda x: x[1])
        rm = {nm: i + 1 for i, (nm, _) in enumerate(rank_m)}
        rd = {nm: i + 1 for i, (nm, _) in enumerate(rank_d)}
        for tag in ("标准 kp=0.30", "高增益 kp=1.00"):
            m, d = cells[(tag, wtag)][1]
            print(f"  {tag:<12}{wtag:<8}{m:>10.3f}{d:>12.3f}{rm[tag]:>10}{rd[tag]:>12}")
    # 合并两种风况
    both = {tag: torch.cat([cells[(tag, "无风")][0], cells[(tag, "强风")][0]])
            for tag in ("标准 kp=0.30", "高增益 kp=1.00")}
    print("  （把两种风况的回合合并后）")
    for tag, e in both.items():
        m, d = mean_median(e)
        print(f"  {tag:<12}{'合并':<8}{m:>10.3f}{d:>12.3f}")

    # ---- 第 2 格：多少回合才够
    e_cold = rollout(KP_COLD, WIND_CALM, N_MAIN, seed=SEED + 22)
    e_warm = rollout(KP_WARM, WIND_CALM, N_MAIN, seed=SEED + 22)
    x_cold = (e_cold < TOL).float()
    x_warm = (e_warm < TOL).float()
    p_cold, p_warm = float(x_cold.mean()), float(x_warm.mean())
    gap = abs(p_warm - p_cold)
    print(f"\n第 2 格：两个增益接近的策略（kp={KP_COLD} vs {KP_WARM}），容差 {TOL} m，无风")
    print(f"  实测成功率 {p_cold * 100:.1f}% vs {p_warm * 100:.1f}%，差 {gap * 100:.1f} 个百分点"
          f"（每档 {N_MAIN} 回合）")
    print(f"  {'回合数':>7}{'95% 置信区间半宽':>20}{'实测差距 / 半宽':>18}")
    print(f"  {'':>7}{'':>20}{'(>1 才分辨得出)':>18}")
    hw = {}
    for n in NS:
        h = halfwidth_at(x_cold, n)
        hw[n] = h
        print(f"  {n:>7}{h * 100:>18.1f}pp{gap / h:>17.2f}")
    need = None
    for n in range(8, 4097, 8):
        if halfwidth_at(x_cold, n, reps=60) <= gap:
            need = n
            break
    print(f"  半宽降到实测差距（{gap * 100:.1f}pp）以下，每臂约需 {need} 回合")

    # ---- 第 3 格：独立单元是回合还是模型
    print(f"\n第 3 格：同一个方法跑 {K_SEEDS} 次独立训练，评测回合数相同")
    print(f"  {'第几次训练':>10}{'增益':>8}{'成功率':>10}")
    seed_rates, seed_x = [], []
    for i, kp in enumerate(KP_PER_SEED):
        x = (rollout(kp, WIND_STRONG, M_PER_SEED, seed=SEED + 100 + i) < TOL * 4).float()
        seed_x.append(x)
        seed_rates.append(float(x.mean()))
        print(f"  {i + 1:>10}{kp:>8.2f}{seed_rates[-1] * 100:>9.1f}%")
    k_all = torch.cat(seed_x)
    p_pool = float(k_all.mean())
    sd_seed = (sum((r - sum(seed_rates) / K_SEEDS) ** 2 for r in seed_rates) / K_SEEDS) ** 0.5
    se_model = sd_seed / math.sqrt(K_SEEDS)
    se_episode = math.sqrt(p_pool * (1 - p_pool) / k_all.numel())
    print(f"  合并成功率 {p_pool * 100:.1f}%（{k_all.numel()} 个回合）")
    print(f"  按回合算（二项）：± {se_episode * 100:.1f}pp")
    print(f"  按模型算（4 次训练的标准误）：± {se_model * 100:.1f}pp"
          f"，是前者的 {se_model / max(se_episode, 1e-9):.2f} 倍")

    # ---- 读法
    m_c_std, d_c_std = cells[("标准 kp=0.30", "无风")][1]
    m_c_hot, d_c_hot = cells[("高增益 kp=1.00", "无风")][1]
    m_s_std, d_s_std = cells[("标准 kp=0.30", "强风")][1]
    m_s_hot, d_s_hot = cells[("高增益 kp=1.00", "强风")][1]
    mb_std, db_std = mean_median(both["标准 kp=0.30"])
    mb_hot, db_hot = mean_median(both["高增益 kp=1.00"])
    print("\n  读法：")
    print(f"    (1) 无风时两种策略差得远：均值 {m_c_std:.3f} vs {m_c_hot:.3f} m，"
          f"高增益好 {m_c_std / m_c_hot:.1f} 倍。这一步两个摘要说的是同一件事。")
    print(f"    (2) 强风时两个摘要打架：均值说标准增益好"
          f"（{m_s_std:.3f} vs {m_s_hot:.3f} m），中位数说高增益好"
          f"（{d_s_std:.3f} vs {d_s_hot:.3f} m）。方向相反。")
    print(f"    (3) 原因是重尾：强风下终点误差的均值被少数飞出去的回合拖走。"
          f"标准增益那组的均值 {m_s_std:.3f} 是它中位数 {d_s_std:.3f} 的 "
          f"{m_s_std / d_s_std:.1f} 倍；高增益是 {m_s_hot / d_s_hot:.1f} 倍。"
          f"谁更重尾，取决于谁的失败更彻底。")
    print(f"    (4) 合并两种风况同样掩盖方向：合并后均值 {mb_std:.3f} vs {mb_hot:.3f} m，"
          f"读起来是'两者接近'，而实际是无风差 {m_c_std / m_c_hot:.1f} 倍、"
          f"强风反过来。只看总分看不出这件事。")
    print(f"    (5) 回合数这条轴：实测差距 {gap * 100:.1f}pp 时，"
          f"{NS[1]} 回合的置信区间半宽是 {hw[NS[1]] * 100:.1f}pp，"
          f"是差距的 {hw[NS[1]] / gap:.1f} 倍——分辨不了；"
          f"{NS[-1]} 回合降到 {hw[NS[-1]] * 100:.1f}pp。")
    print(f"    (6) 独立单元这条轴：把 {K_SEEDS} 次训练的回合合起来按二项算，得到 "
          f"± {se_episode * 100:.1f}pp；按训练次数算得到 ± {se_model * 100:.1f}pp，"
          f"是前者的 {se_model / max(se_episode, 1e-9):.2f} 倍。"
          f"同一批数据，换个单位区间宽一倍。")
    print("    (7) 三条合起来的意思：论文里那个 ± 至少要说清"
          "（用哪个摘要、多少回合、独立单元是什么），否则数字不可比。")

    # ---------------------------------------------------------------- 出图
    fig, ax = plotting.plt.subplots(1, 3, figsize=(13.2, 3.9))

    labels = ["标准\nkp=0.30", "高增益\nkp=1.00"]
    xs = [0, 1]
    for j, (wtag, mk) in enumerate((("无风", "o"), ("强风", "s"))):
        ms = [cells[("标准 kp=0.30", wtag)][1][0], cells[("高增益 kp=1.00", wtag)][1][0]]
        ds = [cells[("标准 kp=0.30", wtag)][1][1], cells[("高增益 kp=1.00", wtag)][1][1]]
        c = "tab:blue" if j == 0 else "tab:red"
        ax[0].plot(xs, ms, mk + "-", color=c, label=f"{wtag}·均值")
        ax[0].plot(xs, ds, mk + "--", color=c, alpha=0.65, label=f"{wtag}·中位数")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("终点误差 (m，对数轴)")
    ax[0].set_title("(a) 强风下均值与中位数排名相反")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ns = list(hw.keys())
    ax[1].plot(ns, [hw[n] * 100 for n in ns], "o-", color="tab:red")
    ax[1].axhline(gap * 100, ls=":", color="tab:blue", lw=2,
                  label=f"实测差距 {gap * 100:.1f}pp")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(ns)
    ax[1].set_xticklabels([str(n) for n in ns], fontsize=8.5)
    ax[1].set_xlabel("评测回合数（每臂）")
    ax[1].set_ylabel("95% 置信区间半宽 (pp)")
    ax[1].set_title("(b) 半宽降到差距以下才算分辨得出")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # 横轴是成功率，纵轴三行：4 次训练各自的结果，以及两种口径的区间
    for r in seed_rates:
        ax[2].plot([r * 100], [2.6], "o", color="tab:purple", ms=9)
    ax[2].text(0.02, 0.97,
               "成功率 %：" + " / ".join(f"{r * 100:.1f}" for r in seed_rates)
               + "\n（对应 kp " + " / ".join(f"{kp:.2f}" for kp in KP_PER_SEED) + "）",
               transform=ax[2].transAxes, fontsize=8, va="top", color="tab:purple")
    ax[2].axhline(1.9, color="k", lw=0.8, alpha=0.3)
    ax[2].errorbar(p_pool * 100, 1.0, xerr=se_episode * 100, fmt="s", color="tab:gray",
                   capsize=7, ms=9, lw=2.2)
    ax[2].errorbar(p_pool * 100, 0.2, xerr=se_model * 100, fmt="s", color="tab:green",
                   capsize=7, ms=9, lw=2.2)
    ax[2].axvline(p_pool * 100, ls=":", color="k", alpha=0.5)
    ax[2].set_yticks([2.6, 1.0, 0.2])
    ax[2].set_yticklabels(["4 次训练\n各自结果", f"按回合算\n±{se_episode * 100:.1f}pp",
                           f"按模型算\n±{se_model * 100:.1f}pp"], fontsize=8)
    ax[2].set_ylim(-0.7, 3.2)
    ax[2].set_xlabel("强风下成功率 (%)")
    ax[2].set_title(f"(c) 同一批数据，区间宽 {se_model / max(se_episode, 1e-9):.1f} 倍")
    ax[2].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    plotting.save(fig, "ph9_eval_protocol.png")


if __name__ == "__main__":
    main()
