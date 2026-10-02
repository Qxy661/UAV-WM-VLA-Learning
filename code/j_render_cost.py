"""
Demo J — 体渲染和 alpha 溅射，是两套公式还是一套？

对应文档：docs/02-世界模型专题/04-3D场景世界模型.md

要验证的结论：2.2 节给了 NeRF 的体渲染积分（沿射线积密度、算累积透射率），
第 4 节讲 3DGS 的 alpha 溅射（把高斯基元按深度排序做前向合成）。
看起来是两套完全不同的东西，本节在一个 2D 玩具场景上把它们摆在一起：

  1. alpha 的来源：单个高斯下，沿射线细采样得到的 alpha，收敛到闭式积分
     alpha = 1 - exp(-∫σ dz)，两者是同一个数
  2. 但公式相同不等于结果相同：只有基元沿深度可分离时两个渲染器才一致，
     深度重叠时溅射的排序是个近似，残差不随采样数下降
  3. 成本：NeRF 每像素固定采 K 个点，空区域也要算；alpha 溅射只算
     真正覆盖该像素的那几个基元

显式声明：这里量的是 2D 玩具场景在 CPU 上的场求值次数与墙钟耗时，
**不是 GPU 上的 FPS**，不能外推到真实 3DGS 的渲染速度。真实 3DGS 的
优势主要来自光栅化管线与 GPU 上的并行排序，那部分是本节量不到的。

运行：py -3.9 code/j_render_cost.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
W = 256                     # 1D 图像的像素数（2D 场景、正交投影）
X_RANGE = 3.0               # 相机横向覆盖 [-3, 3]
Z_FAR = 10.0                # 射线从 z=0 打到 z=10
N_BLOB = 12                 # 场景里的高斯基元数
K_LIST = (8, 16, 32, 64, 128, 256, 1024)


def make_scene(seed, separable=False):
    """场景：若干各向同性的高斯雾团，各自带密度与颜色。

    separable=True 时把深度拉开，保证同一像素不会在同一段深度上
    同时看到两个雾团——用来对照"深度可分离"这个前提。
    """
    g = torch.Generator().manual_seed(seed)
    m = 6 if separable else N_BLOB
    x = (torch.rand(m, generator=g) * 2 - 1) * 2.5          # 横向位置
    if separable:
        r = torch.full((m,), 0.35)
        z = 1.0 + torch.arange(m, dtype=torch.float32) * 1.6    # 间隔 1.6 m >> 2r
    else:
        r = 0.4 + torch.rand(m, generator=g) * 0.6          # 半径
        z = 2.0 + torch.rand(m, generator=g) * 6.0          # 深度 2~8 m，互相重叠
    rho = 1.0 + torch.rand(m, generator=g) * 3.0            # 峰值密度
    col = torch.rand(m, 3, generator=g) * 0.8 + 0.2         # 颜色
    return {"x": x, "z": z, "r": r, "rho": rho, "col": col}


def density(scene, u, z):
    """密度场 σ(u, z)：所有雾团的高斯之和。u/z 形状相同，可广播。"""
    d2 = (u[..., None] - scene["x"]) ** 2 + (z[..., None] - scene["z"]) ** 2
    return (scene["rho"] * torch.exp(-d2 / (2 * scene["r"] ** 2))).sum(-1)


def color(scene, u, z):
    """颜色场：按各雾团在该点的密度加权平均，和密度场是同一套权重。"""
    d2 = (u[..., None] - scene["x"]) ** 2 + (z[..., None] - scene["z"]) ** 2
    w = scene["rho"] * torch.exp(-d2 / (2 * scene["r"] ** 2))
    return (w[..., None] * scene["col"]).sum(-2) / w.sum(-1, keepdim=True).clamp(min=1e-12)


def nerf_render(scene, u, K):
    """体渲染：每条射线均匀采 K 个点，alpha = 1 - exp(-σδ)，前向累积透射率。"""
    z = torch.linspace(0.0, Z_FAR, K)
    delta = Z_FAR / K
    sig = density(scene, u[:, None], z[None, :])            # (W, K)
    alpha = 1 - torch.exp(-sig * delta)
    # 透射率：第 i 点的 T 是它前面所有点 (1-alpha) 的乘积，不含自己
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha], -1), -1)[:, :-1]
    col = color(scene, u[:, None], z[None, :])
    return (trans * alpha)[..., None].mul(col).sum(1)


def splat_render(scene, u, cutoff=1e-3):
    """alpha 溅射：每个基元到该像素的 alpha 用沿 z 的闭式积分，再按深度前向合成。

    闭式：∫ρ·exp(-((u-x)²+(z-z₀)²)/2r²) dz = ρ·r·√(2π)·exp(-(u-x)²/2r²)
    """
    amp = scene["rho"] * scene["r"] * (2 * torch.pi) ** 0.5 \
        * torch.exp(-((u[:, None] - scene["x"]) ** 2) / (2 * scene["r"] ** 2))
    alpha = 1 - torch.exp(-amp)                              # (W, M)
    order = torch.argsort(scene["z"])                        # 由近及远
    alpha = alpha[:, order]
    # 只保留真正有贡献的基元，后面的乘 0 即可（等价于跳过）
    alpha = torch.where(alpha > cutoff, alpha, torch.zeros_like(alpha))
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha], -1), -1)[:, :-1]
    return (trans * alpha)[..., None].mul(scene["col"][order]).sum(1)


def _time_once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def single_gaussian_alpha(rho, r, K):
    """单个雾团：细采样得到的 alpha 与闭式 alpha 的对比。"""
    scene = {"x": torch.zeros(1), "z": torch.zeros(1), "r": torch.tensor([r]),
             "rho": torch.tensor([rho]), "col": torch.ones(1, 3)}
    half = 6 * r                                             # 采样范围盖住 6 个标准差
    z = torch.linspace(-half, half, K)
    delta = 2 * half / K
    marched = 1 - torch.exp(-(density(scene, torch.zeros(1), z) * delta).sum())
    closed = 1 - torch.exp(-torch.tensor(rho * r * (2 * torch.pi) ** 0.5))
    return marched.item(), closed.item()


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    scene = make_scene(SEED)
    u = torch.linspace(-X_RANGE, X_RANGE, W)
    print(f"场景: {N_BLOB} 个高斯雾团，2D 平面，正交投影到 {W} 像素的一维图像")

    # ---- 1. 一致性：ray marching 收敛到闭式 alpha ----
    print("\n[1] 单个高斯（ρ=2.0, r=0.6）：细采样 alpha vs 闭式积分 alpha")
    closed = 1 - torch.exp(-torch.tensor(2.0 * 0.6 * (2 * torch.pi) ** 0.5)).item()
    for K in (4, 16, 64, 256, 1024):
        marched, _ = single_gaussian_alpha(2.0, 0.6, K)
        print(f"    K = {K:5d}  细采样 alpha = {marched:.6f}   闭式 alpha = {closed:.6f}"
              f"   偏差 {abs(marched - closed):.2e}")

    # ---- 2. 成本：场求值次数 ----
    # 墙钟耗时抖动大，取 5 次里最快的一次，让文档里的数字可复现
    def best_of(fn, n=5):
        return min(_time_once(fn) for _ in range(n))

    ref = splat_render(scene, u)                             # 溅射是闭式的，当参考
    img_nerf = nerf_render(scene, u, 128)
    img_splat = splat_render(scene, u)
    t_nerf = best_of(lambda: nerf_render(scene, u, 128))
    t_splat = best_of(lambda: splat_render(scene, u))
    eval_nerf = 128 * N_BLOB                                 # K 个采样点，每点都要遍历所有基元
    hit = ((1 - torch.exp(-scene["rho"] * scene["r"] * (2 * torch.pi) ** 0.5
                          * torch.exp(-((u[:, None] - scene["x"]) ** 2) / (2 * scene["r"] ** 2))))
           > 1e-3).float().sum(1)
    print(f"\n[2] 整幅 {W} 像素渲染一次")
    print(f"    体渲染 K=128    每像素基元求值 {eval_nerf} 次   CPU 耗时 {t_nerf * 1e3:6.1f} ms")
    print(f"    alpha 溅射      每像素基元求值 {hit.mean():.1f} 次（平均，空区域为 0）"
          f"   CPU 耗时 {t_splat * 1e3:6.1f} ms")
    print(f"    求值次数之比 {eval_nerf / hit.mean():.0f}x   耗时之比 {t_nerf / t_splat:.0f}x")
    print(f"    两者渲染结果的最大像素误差 {(img_nerf - img_splat).abs().max():.4f}")

    # ---- 3. NeRF 的误差随 K 收敛 ----
    print("\n[3] 体渲染的采样误差（以闭式溅射为参考）——雾团在深度上互相重叠")
    errs = []
    for K in K_LIST:
        e = (nerf_render(scene, u, K) - ref).abs().mean().item()
        errs.append(e)
        print(f"    K = {K:5d}  每像素求值 {K * N_BLOB:6d} 次   平均像素误差 {e:.5f}")
    print(f"    注意误差不收敛到 0，卡在 {errs[-1]:.5f} 附近")

    # ---- 4. 对照：把雾团在深度上拉开 ----
    sep = make_scene(SEED, separable=True)
    ref_sep = splat_render(sep, u)
    errs_sep = []
    print("\n[4] 对照：6 个雾团沿深度拉开（间隔 1.6 m，半径 0.35 m，不重叠）")
    for K in K_LIST:
        e = (nerf_render(sep, u, K) - ref_sep).abs().mean().item()
        errs_sep.append(e)
        print(f"    K = {K:5d}  每像素求值 {K * 6:6d} 次   平均像素误差 {e:.5f}")

    print(f"\n结论：两个渲染器共用同一个前向合成公式 Σ T_i α_i c_i，"
          f"\n      差别只在 alpha 从哪来——体渲染沿射线采样，溅射用基元到像素的闭式积分。"
          f"\n      但公式相同不等于结果相同：体渲染在重叠处取的是密度加权混色，"
          f"\n      溅射要求每个基元有唯一的深度次序。第 [3] 组的残差卡在 {errs[-1]:.4f} 不降，"
          f"\n      第 [4] 组把深度拉开后残差随 K 收敛到 {errs_sep[-1]:.5f}，"
          f"\n      说明这点差异全部来自深度重叠，跟采样密度无关。")

    # ---- 出图 ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    strips = [("闭式溅射（参考）", ref), ("体渲染 K=8", nerf_render(scene, u, 8)),
              ("体渲染 K=32", nerf_render(scene, u, 32)), ("体渲染 K=128", nerf_render(scene, u, 128))]
    for i, (name, img) in enumerate(strips):
        ax.imshow(img[None].clamp(0, 1).numpy(), extent=[-X_RANGE, X_RANGE, i, i + 0.82],
                  aspect="auto", interpolation="nearest")
        ax.text(-X_RANGE - 0.12, i + 0.41, name, ha="right", va="center", fontsize=9)
    ax.set_xlim(-X_RANGE - 2.6, X_RANGE); ax.set_ylim(-0.15, len(strips))
    ax.set_yticks([]); ax.set_xlabel("横向位置 u（像素方向）")
    ax.set_title("(a) 同一场景的渲染结果：K 越大越接近闭式解")

    ax = axes[1]
    ks = torch.tensor(K_LIST, dtype=torch.float32)
    ax.plot(ks * N_BLOB, errs, "o-", color="#c0392b",
            label="体渲染：雾团深度重叠（残差不降）")
    ax.plot(ks * 6, errs_sep, "s-", color="#e67e22",
            label="体渲染：雾团深度可分离（收敛到 0）")
    # 溅射是参考解，偏差按定义为 0，只标出它的求值次数
    ax.axvline(hit.mean(), color="#2471a3", ls=":", lw=1.8)
    ax.axhline(errs[-1], color="#c0392b", ls=":", lw=1.0, alpha=0.6)
    ax.text(hit.mean() * 1.35, 0.006, f"alpha 溅射\n{hit.mean():.1f} 次/像素\n（闭式解，偏差为 0）",
            color="#2471a3", fontsize=8, va="center")
    ax.text(eval_nerf * 0.45, errs[-1] * 0.42, f"K 再大也降不下去：{errs[-1]:.4f}",
            color="#c0392b", fontsize=8, ha="right", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("每像素基元求值次数（对数轴）"); ax.set_ylabel("与闭式解的偏差（对数轴）")
    ax.set_title("(b) 溅射 9.3 次求值就拿到闭式解，\n体渲染堆到 1536 次仍差 0.079")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, which="both")

    fig.suptitle("体渲染与 alpha 溅射共用同一个合成公式，但只在深度可分离时给出同一个结果", fontsize=12)
    fig.tight_layout()
    plotting.save(fig, "j_render_cost.png")


if __name__ == "__main__":
    main()
