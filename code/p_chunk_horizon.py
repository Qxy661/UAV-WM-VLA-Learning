"""
Demo P — 提交长度 → 闭环精度

对应文档：docs/03-VLA专题/06-动作头与动作分块.md

要验证的结论：动作分块（一次预测 H 步、只执行前 k 步就重规划）里，
k 不是免费的参数。k 越大，基于**旧状态**算出的动作被开环执行的时间越长，
"策略没料到的扰动"积累得越多，而重规划之前没有任何机制能修正它。

自变量是**提交长度 k**，不是动作头类型。后者是另一题，见 code/c_action_heads.py。

为什么不用学出来的策略：本仓库 code/b_lang_cond_policy.py（Demo B）已经把
"模仿损失低、闭环飞不好"（BC 的分布漂移）量过一次了。如果这里也用 BC 策略，学到的策略本身
会在 100 步里发散到 6 m 以外，把要量的 k 效应整个淹掉，而且结论和那篇重复。
所以这里把变量隔离出来：动作块由**标称模型**（无风）把专家律展开 H 步得到，
再在**有阵风**的真实环境里执行前 k 步。策略对风一无所知，风就是
"预测时看不到、执行时才出现"的那部分偏差 —— 这正是分块执行要付的代价。

运行：py -3.9 code/p_chunk_horizon.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting
from common.quad_sim import QuadSim, expert

SEED = 0
DT = 0.02
H = 16                 # 预测的动作块长度（步）
T_EP = 120             # 每回合步数
N_EP = 256             # 并行回合数
N_REP = 5              # 重复次数（换一组阵风）
GOAL = [2.0, 2.0, 1.5]
WIND_STD = 1.2         # 阵风加速度标准差 (m/s^2)，每回合固定、策略不可见
KS = [1, 2, 4, 8, 16]  # 提交长度扫描


def predict_chunk(obs, goal):
    """在当前状态上，用**标称无风**动力学把专家律展开成 H 步动作块。

    这就是"分块策略"：它输出一段动作序列，但这段序列是按它以为的世界算的。
    """
    env = QuadSim(obs.shape[0], dt=DT)
    env.p = obs[:, 0:3].clone()
    env.v = obs[:, 3:6].clone()
    env.rpy = obs[:, 6:9].clone()
    env.om = obs[:, 9:12].clone()
    acts = []
    for _ in range(H):
        a = expert(env, goal)
        acts.append(a)
        env.step(a)
    return torch.stack(acts, dim=1)      # (n, H, 4)


def rollout(k, rep):
    """闭环：预测 H 步 -> 开环执行前 k 步（期间有阵风）-> 重规划。"""
    torch.manual_seed(1000 + rep)
    env = QuadSim(N_EP, dt=DT)
    goal = torch.tensor(GOAL).expand(N_EP, 3).clone()
    env.reset(rand=True, pos_std=0.6, vel_std=0.4)
    # 每回合一个固定方向的阵风，策略全程不知道它的存在
    wind = WIND_STD * torch.randn(N_EP, 3)

    calls, t = 0, 0
    while t < T_EP:
        chunk = predict_chunk(env.obs(), goal)
        calls += 1
        for j in range(k):
            env.step(chunk[:, j])
            env.v = env.v + wind * DT      # 风只在这里起作用
            t += 1
            if t >= T_EP:
                break
    err = torch.norm(env.p - goal, dim=-1)
    return err.mean().item(), err.std().item(), calls, err


def main():
    print(f"窗口：{N_EP} 回合 × {T_EP} 步｜动作块 H={H}｜阵风 σ={WIND_STD} m/s^2（策略不可见）")
    print(f"\n闭环扫描（每档重复 {N_REP} 次，报均值 ± 标准差）")
    print(f"  {'k':>4} {'终点误差(m)':>18} {'策略调用':>10}")
    rows = []
    for k in KS:
        errs = []
        calls = None
        for r in range(N_REP):
            m, _s, c, _e = rollout(k, r)
            errs.append(m)
            calls = c
        mu = sum(errs) / len(errs)
        sd = (sum((e - mu) ** 2 for e in errs) / len(errs)) ** 0.5
        rows.append((k, mu, sd, calls))
        print(f"  {k:>4} {mu:>11.4f} ± {sd:<6.4f} {calls:>10}")

    font = plotting.use_chinese_font()
    if font is None:
        print("  [警告] 没找到中文字体，图内标注退回英文")
    import matplotlib.pyplot as plt
    ks = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].errorbar(ks, [r[1] for r in rows], yerr=[r[2] for r in rows],
                   marker="o", capsize=4, color="tab:red")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks)
    ax[0].set_xticklabels([str(k) for k in ks])
    ax[0].set_xlabel("commit length k (steps)")
    ax[0].set_ylabel("final error to goal (m)")
    ax[0].set_title("longer commit -> larger uncorrected drift")
    ax[0].grid(alpha=0.3)
    ax[1].plot(ks, [r[3] for r in rows], marker="s", color="tab:orange")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(ks)
    ax[1].set_xticklabels([str(k) for k in ks])
    ax[1].set_xlabel("commit length k (steps)")
    ax[1].set_ylabel("policy calls per episode")
    ax[1].set_title("shorter commit -> more policy calls")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    plotting.save(fig, "ph6_chunk_horizon.png")


if __name__ == "__main__":
    main()
