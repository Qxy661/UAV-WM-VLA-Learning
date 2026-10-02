"""
Demo M — 像素自回归：teacher forcing 的误差不涨，一自由 rollout 就爆

对应文档：docs/05-综述论文精读/04-视频生成世界模型.md

要验证的结论：「长视频一致性」一节说多步生成后视频质量下降、累积误差。本节
把这个「累积」量出来，并追问它到底从哪来：

  1. 自回归视频模型训练时喂的是**真实**的上一帧（teacher forcing），推断时
     喂的却是**自己生成**的上一帧。训练和推断的输入分布不一致，这就是
     exposure bias。
  2. 「累积」不是自回归的必然结果。第一版拿一条正弦轨迹试，teacher forcing
     和自由 rollout 的误差**都是平的**——因为那条轨迹的动力学是稳定的，模型
     每步重看完整画面就把误差纠回来了。得动力学本身会把邻近轨迹推开，
     累积才显形。所以这一版把画面换成 Lorenz 吸引子（混沌）。
  3. 混沌给了个可量化的参照：两条只差 1e-8 的真轨迹按实测速率 0.046 /帧 分离。
     拿它去比自由 rollout 的误差放大速率，就能判断崩溃到底是「混沌固有的
     可预报性极限」，还是「模型自己离开训练分布后的行为」。
  4. 对照组：训练时往输入帧里掺噪声（DART 的做法）。扫三档，量它在
     「单步精度」和「放大速率」之间换到了什么。

画面是 24x24 像素的「视频」：一个高斯亮斑沿 Lorenz 的 (x, y) 投影移动，帧间
dt = 0.05。模型输入**连续两帧**、输出下一帧——因为 (x, y) 单帧定不出隐藏的
z，两帧才够。

运行：py -3.9 code/m_exposure_bias.py     # 约 2 分钟，CPU 即可，只需 torch + numpy
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
DT = 0.05                   # 每帧之间的仿真时间（秒）
DISCARD = 4000              # 先跑掉瞬态，让轨道落到吸引子上
T = 64                      # 每条序列的帧数
N_SEQ = 64                  # 训练序列条数
N_EVAL = 32                 # 评测序列条数
GRID = 24                   # 画面边长（像素），每帧 GRID*GRID 维
BLOB = 1.3                  # 亮斑宽度（像素）
MARGIN = 3.0                # 亮斑不许贴边，留出的余量（像素）
# (x, y) -> 像素的画布映射范围。取固定值而不是从数据算，这样文档里那段精简
# 片段和本脚本用的是同一套映射，读出来的数才对得上。Lorenz 实际范围约
# x∈[-18, 18]、y∈[-24.3, 24.3]，这里留了余量，不会裁到亮斑。
LO = [-20.0, -28.0]
HI = [20.0, 28.0]
STEPS = 2000                # 训练步数
BATCH = 64
LR = 2e-3
NOISE = 0.0035              # DART 对照组的噪声（与模型自身单步误差同量级）
NOISE_SWEEP = [0.01, 0.03]  # 再扫两档更大的噪声，看看这个权衡是不是单调
HORIZON = 60                # 自由 rollout 生成多少帧
CKPT = [1, 6, 12, 24, 48, 60]
FIT = (2, 12)               # 拟合误差「上涨段」斜率的步数区间
N_PAIR = 256                # 量轨迹分离速率用的初始点对数


# ---------------------------------------------------------------- Lorenz 吸引子

def lorenz(s):
    x, y, z = s
    return np.array([10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z])


def jac(s):
    x, y, z = s
    return np.array([[-10.0, 10.0, 0.0],
                     [28.0 - z, -1.0, -x],
                     [y, x, -8.0 / 3.0]])


def rk4(y, f, dt):
    """经典四阶 Runge-Kutta，y 与 f(y) 都按一维数组处理。"""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def attractor_points(n, seed):
    """在吸引子上随机取 n 个起点（先跑掉瞬态）。"""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3))
    for i in range(n):
        s = rng.normal(0.0, 8.0, 3)
        for _ in range(DISCARD):
            s = rk4(s, lorenz, DT)
        pts[i] = s
    return pts


def make_sequences(n, seed):
    """每条序列从吸引子上一点出发，记录 T 帧的 (x, y)。"""
    s = attractor_points(n, seed)
    out = np.empty((n, T, 2))
    for i in range(n):
        for t in range(T):
            out[i, t] = s[i, :2]
            s[i] = rk4(s[i], lorenz, DT)
    return out


def separation_rate(horizon, n=N_PAIR, seed=SEED + 7):
    """两条真轨迹在 horizon 帧内的平均分离速率（每帧的 log 增长率）。

    渐近的 Lyapunov 指数要跑很久才收敛（Lorenz 上有限时间指数起伏很大），
    而 rollout 只有几十帧。所以这里量的是**同一段时长上**的分离速率，
    才好和模型的误差增长率直接比。扰动取 1e-8，几十帧后仍远小于吸引子
    尺度，全程都在线性区，不会饱和。
    """
    a = attractor_points(n, seed)
    rng = np.random.default_rng(seed + 1)
    b = a + rng.normal(0.0, 1.0, a.shape) * 1e-8
    d0 = np.linalg.norm(b - a, axis=1)
    for _ in range(horizon):
        for i in range(n):
            a[i] = rk4(a[i], lorenz, DT)
            b[i] = rk4(b[i], lorenz, DT)
    return float(np.mean(np.log(np.linalg.norm(b - a, axis=1) / d0)) / horizon)


def mse_to_pixels(mse):
    """把逐帧 MSE 折算成「亮斑位置偏了几个像素」。

    模型崩掉时不只是位移，还会糊；所以这是个等效读数，不是恒等式。
    """
    c = (GRID - 1) / 2.0
    g = np.arange(GRID, dtype=float)
    dx0 = g[None, :, None] - c
    dy0 = g[None, None, :] - c
    base = np.exp(-(dx0 ** 2 + dy0 ** 2) / (2.0 * BLOB ** 2))
    d = np.arange(0.0, 8.001, 0.02)
    curve = np.array([((np.exp(-((g[None, :, None] - (c + dd)) ** 2 + dy0 ** 2)
                              / (2.0 * BLOB ** 2)) - base) ** 2).mean() for dd in d])
    return float(np.interp(mse, curve, d))


def render(track):
    """把 (x, y) 轨迹按固定画布映射画成高斯亮斑，返回 (n, T, GRID, GRID)。"""
    lo, hi = np.array(LO), np.array(HI)
    px = MARGIN + (track[..., 0] - lo[0]) / (hi[0] - lo[0]) * (GRID - 1 - 2 * MARGIN)
    py = MARGIN + (track[..., 1] - lo[1]) / (hi[1] - lo[1]) * (GRID - 1 - 2 * MARGIN)
    g = np.arange(GRID, dtype=float)
    dx = g[None, None, :, None] - px[..., None, None]      # (n, T, GRID, 1)
    dy = g[None, None, None, :] - py[..., None, None]      # (n, T, 1, GRID)
    return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * BLOB ** 2))


# ---------------------------------------------------------------- 世界模型

class FramePredictor(nn.Module):
    """看连续两帧，预测下一帧——最朴素的像素自回归视频模型。"""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, d))

    def forward(self, pair):
        return self.net(pair)


def train(frames, noise=0.0, steps=STEPS, seed=SEED):
    torch.manual_seed(seed)
    d = frames.shape[-1]
    model = FramePredictor(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1)
    cur = torch.cat([frames[:, :-2], frames[:, 1:-1]], dim=-1).reshape(-1, 2 * d)
    nxt = frames[:, 2:].reshape(-1, d)
    for _ in range(steps):
        i = torch.randint(0, cur.shape[0], (BATCH,), generator=g)
        inp = cur[i]
        if noise:
            # 训练时就把输入弄脏，模型会见过「不完美的上一帧」
            inp = (inp + torch.randn(inp.shape, generator=g) * noise).clamp(0.0, 1.0)
        loss = (model(inp) - nxt[i]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def rollout(model, frames, horizon=HORIZON):
    """同时跑 teacher forcing 与自由 rollout。

    返回 (逐步 MSE 两张, 自由 rollout 生成的全部帧)。生成的帧钳到 [0, 1]——
    真实像素生成器是有界的，不钳的话线性输出头会一路飘到 inf，
    量到的就不再是「画面崩了」而是「数值溢出了」。
    """
    tf_err, free_err = [], []
    free = torch.stack([frames[:, 0], frames[:, 1]], dim=1)      # 自己生成的最近两帧
    gen = [free[:, 0], free[:, 1]]
    for k in range(horizon):
        truth = frames[:, k + 2]
        tf_in = torch.cat([frames[:, k], frames[:, k + 1]], dim=-1)
        free_in = torch.cat([free[:, 0], free[:, 1]], dim=-1)
        pred = model(free_in).clamp(0.0, 1.0)
        tf_err.append((model(tf_in).clamp(0.0, 1.0) - truth).pow(2).mean(dim=-1))
        free_err.append((pred - truth).pow(2).mean(dim=-1))
        free = torch.stack([free[:, 1], pred], dim=1)
        gen.append(pred)
    return (torch.stack(tf_err).mean(1), torch.stack(free_err).mean(1),
            torch.stack(gen, dim=1))


def slope(y, lo, hi):
    """在 [lo, hi] 步上用最小二乘拟合 log(y) 对步数的斜率。"""
    k = np.arange(lo, hi + 1, dtype=float)
    v = np.log(np.asarray(y)[lo - 1:hi])
    return float(np.polyfit(k, v, 1)[0])


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    lam_f = separation_rate(HORIZON)
    print(f"Lorenz 吸引子（sigma=10, rho=28, beta=8/3），dt = {DT}")
    print(f"  真轨迹对的实测分离速率 = {lam_f:.4f} /帧"
          f"（{N_PAIR} 对初始点、{HORIZON} 帧上平均，全程在线性区）")
    print(f"  即两条只差 1e-8 的真实轨迹，每帧分离 e^{lam_f:.4f} = "
          f"{np.exp(lam_f):.4f} 倍——这是**动力学自己**的分离速率\n")

    tr_track = make_sequences(N_SEQ, SEED)
    te_track = make_sequences(N_EVAL, SEED + 99)
    tr4 = torch.tensor(render(tr_track), dtype=torch.float32)
    te4 = torch.tensor(render(te_track), dtype=torch.float32)
    tr = tr4.reshape(N_SEQ, T, -1)          # 展平成像素向量，模型只认一维
    te = te4.reshape(N_EVAL, T, -1)
    print(f"每条序列 {T} 帧，每帧 {GRID}x{GRID} = {GRID * GRID} 像素；"
          f"训练 {N_SEQ} 条，留出 {N_EVAL} 条\n")

    base = train(tr)
    noisy = train(tr, noise=NOISE)
    tf, free, gen = rollout(base, te)
    tf_n, free_n, _ = rollout(noisy, te)
    sweep = []
    for nz in [NOISE] + NOISE_SWEEP:
        m = noisy if nz == NOISE else train(tr, noise=nz)
        a, b, _ = rollout(m, te)
        sweep.append((nz, float(a[0]), float(b[CKPT[-1] - 1]), slope(b, *FIT) / 2.0))

    def row(name, y):
        return f"  {name:<15s}" + "".join(f"{float(y[k - 1]):>12.2e}" for k in CKPT)

    print(f"  {'rollout 步数':<15s}" + "".join(f"{k:>12d}" for k in CKPT))
    print(row("teacher forcing", tf))
    print(row("自由 rollout", free))
    print(row("自由+输入加噪", free_n))

    lo_k, hi_k = FIT
    sb, sn = slope(free, lo_k, hi_k), slope(free_n, lo_k, hi_k)
    f0, fn_ = float(free[CKPT[-1] - 1]), float(free_n[CKPT[-1] - 1])
    gb, gn = sb / 2.0, sn / 2.0

    print("\n  读法：")
    print(f"    teacher forcing 从第 1 步到第 {CKPT[-1]} 步，误差只在 "
          f"{float(tf.min()):.2e}–{float(tf.max()):.2e} 之间浮动，不随步数增长——"
          f"每步都从真值出发，误差没有机会留下来。")
    print(f"    同一套权重、同一批初始帧，只把「喂真值」换成「喂自己的输出」：第 "
          f"{CKPT[-1]} 步误差 {f0:.2e}，是第 1 步的 {f0 / float(free[0]):.0f} 倍，"
          f"比同时刻的 teacher forcing 高 {f0 / float(tf[CKPT[-1] - 1]):.0f} 倍。")
    print(f"    折算成画面：{f0:.2e} 的 MSE 相当于亮斑位置偏了 "
          f"{mse_to_pixels(f0):.1f} 个像素（画面宽 {GRID} 像素，亮斑 sigma={BLOB}）。"
          f"而 teacher forcing 的 {float(tf[CKPT[-1] - 1]):.2e} 只相当于 "
          f"{mse_to_pixels(float(tf[CKPT[-1] - 1])):.2f} 个像素。")
    print(f"    上涨集中在头十来步：第 {lo_k}→{hi_k} 步的等效放大速率是 {gb:.3f} /帧，"
          f"比真轨迹之间的 {lam_f:.3f} /帧 快 {gb / lam_f:.1f} 倍——所以这段崩溃"
          f"主要不是混沌的锅，是模型离开训练分布后外推不受约束。"
          f"第 {hi_k} 步之后曲线在 1e-2 附近趋平，画面崩到一个「糊掉」的稳定状态，"
          f"不是无限发散。")
    print(f"    加噪训练（DART 的做法：训练时往输入帧里掺噪声，让模型见过不完美的"
          f"上一帧）扫了三档：")
    print(f"      {'输入噪声 sigma':<16s}{'单步误差':>12s}{f'第{CKPT[-1]}步误差':>14s}"
          f"{'放大速率 /帧':>14s}")
    print(f"      {0.0:<16.4f}{float(tf[0]):>12.2e}{f0:>14.2e}{gb:>14.4f}")
    for nz, one, last, g in sweep:
        print(f"      {nz:<16.4f}{one:>12.2e}{last:>14.2e}{g:>14.4f}")
    print(f"    噪声的作用是**双向**的，而且两个方向都能看见：从 0 加到 "
          f"{NOISE_SWEEP[-1]}，单步误差涨了 {sweep[-1][1] / float(tf[0]):.1f} 倍，"
          f"放大速率从 {gb:.3f} 降到 {sweep[-1][3]:.3f} /帧（降 "
          f"{(1 - sweep[-1][3] / gb) * 100:.0f}%）——抗扰动这一面确实买到了。"
          f"但代价先到：第 {CKPT[-1]} 步的总误差反而从 {f0:.2e} 涨到 "
          f"{sweep[-1][2]:.2e}，比不加噪差 {sweep[-1][2] / f0:.1f} 倍。")
    print(f"    sigma={NOISE} 那一档是笔不赚不赔的买卖：单步几乎没动（差 "
          f"{sweep[0][1] / float(tf[0]):.2f} 倍），速率略升（{sweep[0][3]:.3f} 对 "
          f"{gb:.3f}），长程 {sweep[0][2]:.2e} 对 {f0:.2e}。这一档的差别不要再往下读，"
          f"理由见文档里的限制说明。")
    print(f"    合起来：想只靠「在输入上撒噪声」压住累积是划不来的——噪声有效的尺度"
          f"得跟最终偏差相当，可最终偏差 {mse_to_pixels(f0):.1f} 个像素远大于单步误差 "
          f"{mse_to_pixels(float(tf[0])):.2f} 个像素，等噪声大到那个份上，单步精度早就"
          f"赔光了。要真正压住，得直接对着 rollout 训练。")

    # ------------------------------------------------------------------ 出图
    fig = plotting.plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.16)

    ax = fig.add_subplot(gs[0, :])
    ks = np.arange(1, HORIZON + 1)
    ax.plot(ks, tf.tolist(), "-", color="#3b6ea5", lw=1.8, label="teacher forcing（喂真值）")
    ax.plot(ks, free.tolist(), "-", color="#c0504d", lw=1.8, label="自由 rollout（喂自己的输出）")
    ax.plot(ks, free_n.tolist(), "--", color="#4a9d5f", lw=1.8,
            label=f"自由 rollout（训练时输入加噪 {NOISE}）")
    seg = ks[lo_k - 1:hi_k]
    ax.plot(seg, np.exp(np.log(float(free[lo_k - 1])) + sb * (seg - lo_k)), ":",
            color="#7a5195", lw=2.2,
            label=f"上涨段拟合：{gb:.3f} /帧（真轨迹分离只有 {lam_f:.3f} /帧）")
    ax.set_yscale("log")
    ax.set_xlabel("rollout 步数"); ax.set_ylabel("逐帧 MSE（对数轴）")
    ax.set_title("(a) 同一套权重、同一批初始帧，只换推断时喂什么\n"
                 f"teacher forcing 被按住；自由 rollout 头十来步猛涨（比混沌本身快 "
                 f"{gb / lam_f:.1f} 倍），随后在 1e-2 附近趋平", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3, which="both")

    LAST = HORIZON + 1
    imgs = [("第 0 帧（真值，作为给定输入）", te4[0, 0]),
            (f"第 {LAST} 帧（真值）", te4[0, LAST]),
            (f"第 {LAST} 帧（自由 rollout）", gen[0, LAST].reshape(GRID, GRID))]
    for j, (title, img) in enumerate(imgs):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img.numpy(), cmap="magma", origin="lower", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Lorenz 吸引子上的像素自回归：teacher forcing 完全测不出累积误差，"
                 f"而崩溃比这团混沌自身的分离速率还快 {gb / lam_f:.1f} 倍", fontsize=11.5)
    plotting.save(fig, "m_exposure_bias.png")


if __name__ == "__main__":
    main()
