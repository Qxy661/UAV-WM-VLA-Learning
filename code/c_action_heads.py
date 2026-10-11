"""
Demo C — 动作头对决：回归头 vs 扩散头

对应文档：docs/03-VLA专题/01-VLA架构演进.md

要验证的结论：当同一个观测对应多种合理动作（多模态）时，
用 MSE 回归的动作头会塌到"模态平均"—— 平均出来的动作两条路都不像；
扩散头因为建模的是整个分布，能采样出不同模态、且几乎不落在中间。

数据：一个 8 步的动作块，侧向通道只有两种走法（左绕 / 右绕），
      这是"同一状态下多种合理行为"的最小构造。

两个容易踩的坑（本脚本都已避开，改动时请留意）：
  1) 噪声调度。100 步线性 beta 在 t=99 时 alpha_bar 仍有 0.36，
     即"前向过程根本没到纯噪声"，而采样却从 N(0,I) 起步，两者不一致，
     去噪器会输出弥散到 ±8 的垃圾。这里改用余弦调度 (alpha_bar_99 ≈ 2e-4)。
  2) 探针位置。侧向真值随时间是斜坡 linspace(0.2,0.8)，所以
     "全时段均值 ±0.5" 和 "终点真值 ±0.8" 是两个不同的数，
     要跟采样终点比就必须用 ±0.8。

运行：py -3.9 code/c_action_heads.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
H, D = 8, 4          # 动作块：8 步，每步 4 维
N_DATA = 6000
NS = 100             # 扩散步数
LATERAL = 1          # 侧向所在的通道下标
TOL = 0.3            # 判定"落在某个模态附近"的容差


def make_data(n):
    """构造双模态动作数据：同一状态下"左绕"或"右绕"都合理。"""
    side = torch.randint(0, 2, (n,))              # 0=左绕 1=右绕
    sgn = side.float() * 2 - 1                    # -1 / +1
    a = torch.zeros(n, H, D)
    a[:, :, 0] = 0.5                              # thrust 恒定
    a[:, :, LATERAL] = sgn[:, None] * torch.linspace(0.2, 0.8, H)   # 侧向：两个模态
    a[:, :, 2] = 0.1 * torch.sin(torch.linspace(0, 3.14, H))[None, :]
    a += 0.03 * torch.randn(n, H, D)
    return a, side


def cosine_schedule(ns, s=0.008):
    """余弦噪声调度 (Nichol & Dhariwal 2021)，为少步数采样设计。"""
    f = lambda u: math.cos(((u / ns) + s) / (1 + s) * math.pi / 2) ** 2
    ab = torch.tensor([f(t) / f(0) for t in range(ns)], dtype=torch.float32).clamp(1e-5, 0.9999)
    alphas = (ab / torch.cat([torch.ones(1), ab[:-1]])).clamp(1e-8, 0.9999)
    betas = (1 - alphas).clamp(1e-8, 0.999)
    return alphas, ab, betas


# ---------------------------------------------------------------- ① 回归头
class RegHead(nn.Module):
    """MSE 回归：直接从条件映射到一整块动作。"""

    def __init__(self, cond_dim=12, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, H * D),
        )

    def forward(self, cond):
        return self.net(cond).view(-1, H, D)


# ---------------------------------------------------------------- ② 扩散头
class SinusoidalEmb(nn.Module):
    """时间步的正弦位置编码。直接喂原始标量 t 也能跑，但收敛慢很多。"""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        a = t[:, None].float() * freqs[None, :]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class Denoiser(nn.Module):
    """预测噪声 eps_theta(x_t, t, cond)，标准 DDPM 目标。"""

    def __init__(self, cond_dim=12, h=256):
        super().__init__()
        self.emb = SinusoidalEmb(64)
        self.net = nn.Sequential(
            nn.Linear(H * D + cond_dim + 64, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, H * D),
        )

    def forward(self, x, t, cond):
        return self.net(torch.cat([x.flatten(1), cond, self.emb(t)], dim=-1)).view(-1, H, D)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    X, SIDE = make_data(N_DATA)
    true_mode = torch.linspace(0.2, 0.8, H)[-1].item()      # 终点处的模态真值
    print(f"动作块 {tuple(X.shape)}；侧向通道两模态：")
    print(f"  全时段均值 {X[SIDE == 0, :, LATERAL].mean():+.2f} / {X[SIDE == 1, :, LATERAL].mean():+.2f}"
          f"（斜坡 linspace(0.2,0.8)，故与终点值不同）")
    print(f"  终点真值   {-true_mode:+.2f} / {true_mode:+.2f}   <- 采样要与这个比")

    # 占位条件向量：本 demo 只关心"动作头"本身，不接真实观测+语言条件
    cond = torch.cat([torch.zeros(N_DATA, 3), torch.zeros(N_DATA, 1),
                      torch.zeros(N_DATA, 8)], dim=-1)

    # ---------------- ① 回归头 ----------------
    reg = RegHead()
    opt = torch.optim.Adam(reg.parameters(), lr=1e-3)
    for _ in range(1500):
        b = torch.randint(0, N_DATA, (128,))
        loss = F.mse_loss(reg(cond[b]), X[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rp = reg(cond[:1])[0]
    lat_reg = rp[:, LATERAL]
    print(f"\n① 回归头预测的侧向轨迹: {[round(v, 2) for v in lat_reg.tolist()]}")
    print(f"   -> 全部落在 {lat_reg.mean():+.2f}，两个模态都不像 —— 这就是模态平均")

    # ---------------- ② 扩散头 ----------------
    alphas, ab, betas = cosine_schedule(NS)
    den = Denoiser()
    opt = torch.optim.Adam(den.parameters(), lr=2e-3)
    print(f"\n{'step':>6} {'扩散损失':>12}")
    for step in range(1, 8001):
        b = torch.randint(0, N_DATA, (128,))
        t = torch.randint(0, NS, (128,))
        x0 = X[b]
        eps = torch.randn_like(x0)
        a = ab[t].view(-1, 1, 1)
        pred = den(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, cond[b])
        loss = F.mse_loss(pred, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            print(f"{step:>6} {loss.item():>12.5f}")

    @torch.no_grad()
    def sample(n=400):
        """DDPM 祖先采样：从纯噪声出发，逐步去噪。"""
        x = torch.randn(n, H, D)
        for t in reversed(range(NS)):
            eps = den(x, torch.full((n,), t), cond[:n])
            at, abt = alphas[t], ab[t]
            x = (x - (1 - at) / (1 - abt).sqrt() * eps) / at.sqrt()
            if t > 0:
                x = x + (betas[t] * (1 - ab[t - 1]) / (1 - abt)).sqrt() * torch.randn_like(x)
        return x

    s = sample()
    lat = s[:, -1, LATERAL]
    near = ((lat - true_mode).abs() < TOL) | ((lat + true_mode).abs() < TOL)
    n_left = (lat < -TOL).sum().item()
    n_mid = ((lat >= -TOL) & (lat <= TOL)).sum().item()
    n_right = (lat > TOL).sum().item()
    print(f"\n② 扩散头采样 400 条：")
    print(f"   均值 {lat.mean():+.2f}  标准差 {lat.std():.2f}  范围 [{lat.min():.1f}, {lat.max():.1f}]")
    print(f"   落在两个模态附近（±{true_mode:.2f}±{TOL}）的比例：{near.float().mean() * 100:.1f}%")
    print(f"   左绕 {n_left} 条 | 中间（两不像）{n_mid} 条 | 右绕 {n_right} 条")
    print(f"   -> 中间只有 {n_mid} 条，说明扩散头几乎不做模态平均")

    # ---------------- 图 ----------------
    fig, (ax1, ax2) = plotting.plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax1.hist(X[:, -1, LATERAL].numpy(), bins=60, color="#DD8452", alpha=0.7)
    ax1.set_xlabel("侧向动作（终点）")
    ax1.set_ylabel("训练样本数")
    ax1.set_title("训练数据：两个模态（终点真值 ±0.80）")
    ax1.set_xlim(-3, 3)
    ax1.grid(alpha=0.3, axis="y")

    ax2.hist(lat.numpy(), bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
    for v, lab in ((-true_mode, f"模态 {-true_mode:+.2f}"), (true_mode, f"模态 {true_mode:+.2f}")):
        ax2.axvline(v, color="#55A868", linestyle="--", linewidth=2, label=lab)
    ax2.axvline(lat_reg.mean().item(), color="#C44E52", linewidth=2.5,
                label=f"① 回归头 {lat_reg.mean():+.2f}")
    ax2.set_xlabel("侧向动作（终点）")
    ax2.set_ylabel("采样条数")
    ax2.set_title(f"② 扩散头：{n_left} 左 / {n_mid} 中间 / {n_right} 右")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("多模态动作：MSE 回归塌到中间，扩散采样分成两簇", y=1.02)
    plotting.save(fig, "c_multimodal.png")


if __name__ == "__main__":
    main()
