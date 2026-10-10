"""
Demo A — 用 RSSM 给四旋翼学一个世界模型

对应文档：docs/02-世界模型专题/03-模型强化学习世界模型.md

要验证的结论：世界模型的"想象"（只用先验推演未来）能多准地预测未来的回报。
这正是 Dreamer 系列能拿世界模型替代真实环境的依据。

关键设计
  - 编码器 -> GRU 递推 -> 先验/后验 两条潜变量路径（RSSM 的核心）
  - KL 平衡：alpha*kl.detach() + (1-alpha)*kl，既训练先验又不让它塌掉
  - 训练用后验（teacher forcing），评估用先验（想象），两者的落差就是"想象误差"

运行：py -3.9 code/a_worldmodel_rssm.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.quad_sim import QuadSim, reward
from common import plotting

SEED = 0
OBS_DIM, ACT_DIM = 12, 4
T = 40                 # 每条轨迹 40 步 = 0.8 秒
N_ENV = 256            # 并行环境数
SEQ, BATCH = 16, 32    # 训练时的子序列长度与批大小
TRAIN_STEPS = 1200


class RSSM(nn.Module):
    """循环状态空间模型 (Recurrent State Space Model)。

    h 是确定性隐状态（GRU 承载），z 是随机隐状态（对角高斯）。
    先验 p(z|h) 用于想象，后验 q(z|h,obs) 用于训练时吸收观测。
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, h=128, z=16):
        super().__init__()
        self.h, self.z = h, z
        self.enc = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU())
        self.gru = nn.GRUCell(h + act_dim, h)
        self.prior = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2 * z))
        self.post = nn.Sequential(nn.Linear(h + obs_dim, h), nn.SiLU(), nn.Linear(h, 2 * z))
        self.dec = nn.Sequential(nn.Linear(h + z, h), nn.SiLU(), nn.Linear(h, obs_dim))
        self.rew = nn.Sequential(nn.Linear(h + z, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, obs, act, hx, z):
        hx = self.gru(torch.cat([self.enc(obs), act], dim=-1), hx)
        mu_p, ls_p = self.prior(hx).chunk(2, dim=-1)
        mu_q, ls_q = self.post(torch.cat([hx, obs], dim=-1)).chunk(2, dim=-1)
        # 训练时用后验采样（重参数化）
        z = mu_q + ls_q.clamp(-4, 4).exp() * torch.randn_like(mu_q)
        return hx, z, torch.cat([hx, z], dim=-1), (mu_p, ls_p), (mu_q, ls_q)

    def decode(self, feats):
        return self.dec(feats), self.rew(feats)


def kl_balanced(mu_p, ls_p, mu_q, ls_q, alpha=0.8):
    """KL(q||p) 的平衡版本。

    alpha*kl.detach() 这一项只推动先验去追后验（不回流到编码器），
    (1-alpha)*kl 则同时训练两边。DreamerV3 用这个技巧避免后验塌缩。
    """
    ls_p = ls_p.clamp(-4, 4)
    ls_q = ls_q.clamp(-4, 4)
    kl = 0.5 * ((mu_q - mu_p).pow(2) / ls_p.exp().pow(2)
                + ls_p.exp().pow(2) / ls_q.exp().pow(2)
                - 1 + 2 * (ls_q - ls_p))
    return (alpha * kl.detach() + (1 - alpha) * kl).sum(-1).mean()


def collect_data():
    """用"探索噪声 + 简易高度保持"采集一批轨迹（不做 RL，够训练世界模型即可）。"""
    env = QuadSim(N_ENV)
    goal = torch.zeros(3)
    goal[:2] = torch.tensor([6.0, 3.0])
    obs = env.reset()
    OBS, ACT, REW = [], [], []
    for t in range(T):
        if t % 3 == 0:
            a = torch.randn(N_ENV, ACT_DIM) * 0.25      # 周期性注入探索噪声
        else:
            a = torch.zeros(N_ENV, ACT_DIM)
            a[:, 0] = 0.5 + 0.02 * (goal[2] - env.p[:, 2])
            a[:, 1] = 0.05 * (goal[1] - env.p[:, 1])
            a[:, 2] = -0.05 * (goal[0] - env.p[:, 0])
        REW.append(reward(obs, goal))
        OBS.append(obs)
        ACT.append(a)
        obs = env.step(a)
    # 统一沿 dim=1 堆叠 -> (batch, time, feat)
    # 注意：如果这里一个用 dim=0、一个用 dim=-1，批次下标会当成时间下标用，
    # 报 "index 61 is out of bounds for dimension 0 with size 40" —— 踩过这个坑
    return (torch.stack(OBS, 1), torch.stack(ACT, 1), torch.stack(REW, 1))


def train(wm, OBS, ACT, REW):
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    hist = {"step": [], "重建": [], "KL": [], "奖励": [], "总损失": []}
    print(f"{'step':>6} {'重建':>9} {'KL':>9} {'奖励':>9} {'总损失':>10}")
    for step in range(1, TRAIN_STEPS + 1):
        i = torch.randint(0, N_ENV, (BATCH,))
        t0 = int(torch.randint(0, T - SEQ, (1,)))
        o = OBS[i, t0:t0 + SEQ]
        a = ACT[i, t0:t0 + SEQ]
        r = REW[i, t0:t0 + SEQ]

        hx = torch.zeros(BATCH, wm.h)
        z = torch.zeros(BATCH, wm.z)
        Lr = Lk = Lw = 0.0
        for k in range(SEQ):
            hx, z, feats, pp, qp = wm(o[:, k], a[:, k], hx, z)
            o_hat, r_hat = wm.decode(feats)
            Lr = Lr + F.mse_loss(o_hat, o[:, k])
            Lw = Lw + F.mse_loss(r_hat.squeeze(-1), r[:, k])
            Lk = Lk + kl_balanced(*pp, *qp)
        Lr, Lk, Lw = Lr / SEQ, Lk / SEQ, Lw / SEQ
        loss = Lr + 0.5 * Lk + Lw

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()

        if step % 20 == 0:
            hist["step"].append(step)
            hist["重建"].append(Lr.item())
            hist["KL"].append(Lk.item())
            hist["奖励"].append(Lw.item())
            hist["总损失"].append(loss.item())
        if step % 200 == 0:
            print(f"{step:>6} {Lr.item():>9.4f} {Lk.item():>9.4f} {Lw.item():>9.4f} {loss.item():>10.4f}")
    return hist


@torch.no_grad()
def rollout(wm, OBS, ACT, REW, t0=5, n=64, open_loop=False):
    """从 t0 开始在潜空间里往前推演，累积预测奖励。

    open_loop=False: 先验潜变量 rollout —— z 由先验采样，但 GRU 的输入
                     仍喂真实观测（编码器没被绕过，误差不会累积）
    open_loop=True : 完全开环 —— 把解码出来的观测再喂回去，误差会滚雪球
    """
    i = torch.arange(n)
    hx = torch.zeros(n, wm.h)
    z = torch.zeros(n, wm.z)
    for k in range(t0):
        hx, z, _, _, _ = wm(OBS[i, k], ACT[i, k], hx, z)

    pred, true = [], []
    obs_recon = OBS[i, t0 - 1]
    for k in range(t0, T):
        _, r_hat = wm.decode(torch.cat([hx, z], dim=-1))
        pred.append(r_hat.squeeze(-1))
        true.append(REW[i, k])
        mu_p, ls_p = wm.prior(hx).chunk(2, dim=-1)                  # 只用先验，不看观测
        z = mu_p + ls_p.clamp(-4, 4).exp() * torch.randn_like(mu_p)
        obs_in = obs_recon if open_loop else OBS[i, k]
        hx = wm.gru(torch.cat([wm.enc(obs_in), ACT[i, k]], dim=-1), hx)
        if open_loop:
            obs_recon, _ = wm.decode(torch.cat([hx, z], dim=-1))
    pred = torch.stack(pred, 1)
    true = torch.stack(true, 1)
    return pred, true


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    OBS, ACT, REW = collect_data()
    print(f"数据集 obs {tuple(OBS.shape)}  act {tuple(ACT.shape)}  rew {tuple(REW.shape)}\n")

    wm = RSSM()
    hist = train(wm, OBS, ACT, REW)

    # ---- 想象 rollout ----
    pred, true = rollout(wm, OBS, ACT, REW, open_loop=False)
    p_sum, t_sum = pred.sum(1), true.sum(1)
    dev = abs(p_sum.mean().item() - t_sum.mean().item()) / abs(t_sum.mean().item()) * 100
    print(f"\n想象 rollout 回报  预测 {p_sum.mean():.1f} | 真实 {t_sum.mean():.1f} | 偏差 {dev:.1f}%")

    p2, t2 = rollout(wm, OBS, ACT, REW, open_loop=True)
    p2_sum = p2.sum(1)
    dev2 = abs(p2_sum.mean().item() - t2.sum(1).mean().item()) / abs(t2.sum(1).mean().item()) * 100
    print(f"完全开环 rollout   预测 {p2_sum.mean():.1f} | 真实 {t2.sum(1).mean():.1f} | 偏差 {dev2:.1f}%")

    # ---- 图 1：损失曲线 ----
    fig, ax = plotting.plt.subplots(figsize=(8, 4.5))
    for k in ["重建", "KL", "奖励", "总损失"]:
        ax.plot(hist["step"], hist[k], label=k, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("损失（对数轴）")
    ax.set_title("RSSM 世界模型训练曲线")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plotting.save(fig, "wm_loss_curve.png")

    # ---- 图 2：想象 vs 真实 ----
    fig, ax = plotting.plt.subplots(figsize=(8, 4.5))
    ax.plot(range(5, T), true.mean(0).tolist(), label="真实奖励", linewidth=2)
    ax.plot(range(5, T), pred.mean(0).tolist(), "--", label="想象（先验）", linewidth=2)
    ax.plot(range(5, T), p2.mean(0).tolist(), ":", label="想象（完全开环）", linewidth=2)
    ax.set_xlabel("时间步（0.02 秒/步）")
    ax.set_ylabel("单步奖励")
    ax.set_title(f"潜空间想象 vs 真实奖励（累积回报偏差 {dev:.1f}%）")
    ax.grid(alpha=0.3)
    ax.legend()
    plotting.save(fig, "wm_imagination.png")


if __name__ == "__main__":
    main()
