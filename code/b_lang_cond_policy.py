"""
Demo B — 语言条件飞行策略：为什么"模仿损失低"不等于"飞得好"

对应文档：docs/03-VLA专题/03-语言条件飞行控制.md

要验证的结论：纯行为克隆（BC）学出来的策略，模仿损失可以低到 1e-5，
但闭环飞行反而更差 —— 这就是分布漂移（distribution shift）。
给专家数据加动作噪声 / 随机化初始状态能缓解它（DAgger 的核心思想）。

四个数据配置做对照：
  ① 纯专家数据          ② +随机初始状态
  ③ +随机初始+动作噪声   ④ 噪声加大到 0.30

运行：py -3.9 code/b_lang_cond_policy.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim, expert
from common import plotting

SEED = 0
N, T = 512, 250        # 512 条并行轨迹，每条 250 步 = 5 秒
EPOCHS = 1500
LR = 1e-3

# 语言指令 -> 目标点 (x, y, z)。四条指令对应四种不同的飞行意图。
INSTR = {
    "飞到红色建筑上方并悬停": [6.0, 3.0, 5.0],
    "飞到蓝色建筑上方并悬停": [-5.0, 4.0, 5.0],
    "沿直线向前巡航": [9.0, 0.0, 3.0],
    "绕到目标后方": [-4.0, -7.0, 3.0],
}

# 字符袋（bag-of-characters）编码：把指令里出现过的汉字打成 one-hot。
# 这里是最粗糙的文本编码，只为演示"语言作为条件"这件事本身；
# 真实 VLA 用的是 CLIP / T5 之类的预训练文本编码器。
vocab = sorted({c for s in INSTR for c in s})


def encode_text(s):
    v = torch.zeros(len(vocab))
    for c in s:
        v[vocab.index(c)] = 1.0
    return v


class LangPolicy(nn.Module):
    """obs + 语言 -> 动作。两个编码器各自升维后拼接，再融合出 4 维动作。"""

    def __init__(self, obs_dim=12, txt_dim=len(vocab), h=128, act_dim=4):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU())
        self.txt_enc = nn.Sequential(nn.Linear(txt_dim, h), nn.SiLU(), nn.Linear(h, h))
        self.fuse = nn.Sequential(
            nn.Linear(2 * h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, act_dim),
        )

    def forward(self, obs, txt):
        return torch.tanh(self.fuse(torch.cat([self.obs_enc(obs), self.txt_enc(txt)], dim=-1)))


names = list(INSTR)
texts = torch.stack([encode_text(s) for s in INSTR])
idx = torch.arange(N) % len(names)
goals = torch.stack([torch.tensor(INSTR[names[i]]) for i in idx])


@torch.no_grad()
def eval_policy(pol):
    """闭环评估：从原点起飞，跑到 T 步，量最终离目标多远。"""
    dists, paths = [], []
    for nm in names:
        g = torch.tensor(INSTR[nm])
        e = QuadSim(32)
        o = e.reset()
        txt = encode_text(nm).repeat(32, 1)
        traj = []
        for _ in range(T):
            o = e.step(pol(o, txt))
            traj.append(e.p.mean(0).clone())
        dists.append((e.p - g).norm(dim=-1).mean().item())
        paths.append(torch.stack(traj).numpy())
    return dists, paths


@torch.no_grad()
def expert_path(nm):
    """脚本专家的参考轨迹（闭环，用于对比）。"""
    g = torch.tensor(INSTR[nm])
    e = QuadSim(32)
    e.reset()
    gg = g.repeat(32, 1)
    traj = []
    for _ in range(T):
        e.step(expert(e, gg))
        traj.append(e.p.mean(0).clone())
    return torch.stack(traj).numpy()


def run(noise, rand_init, tag):
    """采数据 -> 训练 -> 闭环评估。返回 (最后一个 minibatch 的 BC 损失, 四条指令的终点误差)。"""
    torch.manual_seed(1)                      # 固定种子，让四组配置的对比公平
    env = QuadSim(N)
    obs = env.reset(rand=rand_init)
    O, Tt, A = [], [], []
    for t in range(T):
        a = expert(env, goals)
        if noise > 0:
            # 动作噪声 = 廉价版 DAgger：让数据里出现"专家没走过但策略会走偏到"的状态
            a = (a + noise * torch.randn(N, 4)).clamp(-1, 1)
        O.append(obs)
        Tt.append(texts[idx])
        A.append(a)
        obs = env.step(a)
    # 三个张量都沿 dim=1 堆叠 -> (batch, time, feat)
    O = torch.stack(O, 1)
    Tt = torch.stack(Tt, 1)
    A = torch.stack(A, 1)

    pol = LangPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    for ep in range(EPOCHS):
        b = torch.randint(0, N, (128,))
        t = int(torch.randint(0, T, (1,)))
        loss = F.mse_loss(pol(O[b, t], Tt[b, t]), A[b, t])
        opt.zero_grad()
        loss.backward()
        opt.step()
    dists, paths = eval_policy(pol)
    return loss.item(), dists, paths


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    # 专家基线：专家自己都到不了位的话，BC 不可能到位
    print("【专家基线】脚本专家闭环表现：")
    with torch.no_grad():
        for nm in names:
            g = torch.tensor(INSTR[nm])
            e = QuadSim(32)
            e.reset()
            gg = g.repeat(32, 1)
            for _ in range(T):
                e.step(expert(e, gg))
            print(f"  {nm:<24} 距目标 {(e.p - g).norm(dim=-1).mean():5.2f} m")

    configs = [
        (0.00, False, "① 纯专家数据",              "①\n纯专家"),
        (0.00, True,  "② +随机初始状态",            "②\n随机初始"),
        (0.15, True,  "③ +随机初始+动作噪声0.15",   "③\n随机初始\n+噪声0.15"),
        (0.30, True,  "④ 噪声加大到0.30",           "④\n随机初始\n+噪声0.30"),
    ]
    print(f"\n四指令闭环最终距目标 (m)   红建筑  蓝建筑  向前巡航  目标后方")
    results = []
    for noise, rand_init, tag, short in configs:
        loss, dists, paths = run(noise, rand_init, tag)
        results.append((short, loss, dists, paths))
        print(f"  [{tag:<24}] BC损失 {loss:.5f} | "
              + "  ".join(f"{d:5.2f}" for d in dists)
              + f" | 平均 {sum(dists) / 4:5.2f} m")

    # ---- 图 1：模仿损失 vs 闭环误差 ----
    fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(11, 4.2))
    xs = range(len(results))
    labels = [r[0] for r in results]
    ratio = results[-1][1] / results[0][1]
    ax1.bar(xs, [r[1] for r in results], color="#4C72B0")
    ax1.set_yscale("log")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("行为克隆损失（对数轴）")
    ax1.set_title(f"模仿损失：④ 是 ① 的约 {ratio:.0f} 倍")
    for x, r in zip(xs, results):
        ax1.text(x, r[1], f"{r[1]:.5f}", ha="center", va="bottom", fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(xs, [sum(r[2]) / 4 for r in results], color="#C44E52")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("闭环终点误差（米，越小越好）")
    ax2.set_title("闭环误差：损失最高的 ④ 反而最小")
    for x, r in zip(xs, results):
        v = sum(r[2]) / 4
        ax2.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("分布漂移：模仿损失低 ≠ 闭环飞得好（②③ 的次序有单次实验噪声）", y=1.02)
    plotting.save(fig, "bc_distribution_shift.png")

    # ---- 图 2：俯视轨迹，专家 vs 训练后策略 ----
    best = results[-1]
    fig, axes = plotting.plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, nm, path in zip(axes, names, best[3]):
        ep = expert_path(nm)
        ax.plot(ep[:, 0], ep[:, 1], "--", color="#55A868", linewidth=2, label="脚本专家")
        ax.plot(path[:, 0], path[:, 1], "-", color="#C44E52", linewidth=2, label="BC 策略（④）")
        g = INSTR[nm]
        ax.plot(g[0], g[1], "*", color="gold", markersize=16,
                markeredgecolor="black", zorder=5, label="目标点")
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("四条语言指令的闭环轨迹（俯视图）", y=1.02)
    plotting.save(fig, "bc_trajectory.png")


if __name__ == "__main__":
    main()
