"""
Demo L — ATE 和 RPE 量的不是同一件事

对应文档：docs/02-世界模型专题/06-关键数据集与基准.md

要验证的结论：8.1 节把 ATE 写成「绝对轨迹误差」、RPE 写成「相对位姿误差」，
两个名字读起来都像「轨迹有多准」。本节把三种误差源**分开构造**，看两个
指标各自对什么敏感：

  1. 全局刚体偏移：整条估计轨迹平移 + 转一点。每一帧的位姿都错，但相邻帧
     之间的相对运动一点没变。
  2. 局部抖动：每帧独立加小噪声。相对运动被破坏，但误差不累积。
  3. 慢漂移：位置尺度偏 2%（标定没做准就是这个样子）。误差随距离累积，
     可每一帧的相对运动几乎没变。

预期结论是「RPE 小」远不等于「轨迹准」：全局错位与慢漂移这两类最能说明
轨迹质量的误差，RPE 都看不见。同时会看到刚体对齐（Umeyama）能消掉全局
偏移，却消不掉尺度漂移——所以 ATE 报出来之前必须说清有没有对齐。

顺带把 8.2 节的 SSIM / FID 换成不依赖 cv2 与 scipy.sqrtm 的实现：这两个库
都不在 code/requirements.txt 里，照抄现文是跑不起来的。

运行：py -3.9 code/l_seq_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
N = 200                     # 轨迹帧数，200 * 0.02 s = 4 s
DT = 0.02                   # 50 Hz
RADIUS = 2.0                # 圆周半径（米）
OMEGA = 0.8                 # 角速率（rad/s）
JITTER = 0.02               # 局部抖动标准差（米）
SCALE = 1.02                # 慢漂移：位置尺度偏 2%
SHIFT = np.array([0.5, 0.0, 0.0])   # 全局偏移（米）
YAW_SHIFT = np.radians(3.0)         # 全局偏航（弧度）
DELTA = 1                   # RPE 的相对间隔（帧）
IMG = 64                    # SSIM / FID 用的合成图边长
FLOOR = 1e-5                # 画对数轴时的下限，0 画不出来


def truth():
    """一条绕圆爬升的参考轨迹，机头朝切线方向。返回 (N, 4, 4)。"""
    t = np.arange(N) * DT
    yaw = OMEGA * t
    P = np.zeros((N, 4, 4))
    P[:, 3, 3] = 1.0
    P[:, 0, 0] = np.cos(yaw); P[:, 0, 1] = -np.sin(yaw)
    P[:, 1, 0] = np.sin(yaw); P[:, 1, 1] = np.cos(yaw)
    P[:, 2, 2] = 1.0
    P[:, :3, 3] = np.stack([RADIUS * np.cos(yaw),
                            RADIUS * np.sin(yaw),
                            0.4 * np.sin(1.7 * yaw)], axis=1)
    return P


def ate(est, gt):
    """绝对轨迹误差：位置误差的 RMSE。与 8.1 节的定义一致。"""
    e = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1)
    return float(np.sqrt(np.mean(e ** 2)))


def rpe(est, gt, delta=DELTA):
    """相对位姿误差：先各自取 delta 帧的相对运动，再比这两个相对运动。"""
    er = np.linalg.inv(est[:-delta]) @ est[delta:]
    gr = np.linalg.inv(gt[:-delta]) @ gt[delta:]
    d = np.linalg.inv(gr) @ er
    return float(np.sqrt(np.mean(np.linalg.norm(d[:, :3, 3], axis=1) ** 2)))


def align_rigid(est, gt):
    """Umeyama 刚体对齐：只允许旋转 + 平移，**不含尺度**。

    不含尺度是关键：估计轨迹整体放大 2% 这种误差，刚体对齐消不掉。
    """
    A, B = est[:, :3, 3].T, gt[:, :3, 3].T
    muA, muB = A.mean(1, keepdims=True), B.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd((B - muB) @ (A - muA).T)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))   # 防镜像
    R = U @ D @ Vt
    out = est.copy()
    out[:, :3, 3] = (R @ A + (muB - R @ muA)).T
    out[:, :3, :3] = R @ est[:, :3, :3]
    return out


def perturb(kind, rng):
    """按 kind 造一条「估计轨迹」。"""
    P = truth()
    if kind == "全局刚体偏移":
        c, s = np.cos(YAW_SHIFT), np.sin(YAW_SHIFT)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        P[:, :3, 3] = (Rz @ P[:, :3, 3].T).T + SHIFT
        P[:, :3, :3] = Rz @ P[:, :3, :3]
    elif kind == "局部抖动":
        P[:, :3, 3] += rng.normal(0.0, JITTER, size=(N, 3))
    elif kind == "2% 慢漂移":
        P[:, :3, 3] *= SCALE
    return P


def _blur(img, sigma=1.5, radius=5):
    """可分离高斯模糊，替代 cv2.GaussianBlur（不引 cv2）。"""
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)); k /= k.sum()
    out = np.apply_along_axis(np.convolve, 0,
                              np.pad(img, ((radius, radius), (0, 0)), mode="reflect"),
                              k, "valid")
    out = np.apply_along_axis(np.convolve, 1, np.pad(out, ((0, 0), (radius, radius)),
                                                     mode="reflect"), k, "valid")
    return out


def ssim(a, b, sigma=1.5):
    """整图平均 SSIM。替代 8.2 节那份用 cv2 的实现，公式逐项一致。"""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = _blur(a, sigma), _blur(b, sigma)
    s11 = _blur(a * a, sigma) - mu1 ** 2
    s22 = _blur(b * b, sigma) - mu2 ** 2
    s12 = _blur(a * b, sigma) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
        ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


def sqrtm_psd(A):
    """对称半正定阵的平方根，替代 scipy.linalg.sqrtm（不引 scipy）。

    eigh 出来的特征值可能带一点负的数值噪声，截断到 0 再开方。
    """
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def fid(real, gen):
    """FID 公式本体。喂进来的是自造特征，不是 Inception 特征。"""
    mu1, mu2 = real.mean(0), gen.mean(0)
    s1, s2 = np.cov(real, rowvar=False), np.cov(gen, rowvar=False)
    sh = sqrtm_psd(s1)
    M = sh @ s2 @ sh
    w = np.linalg.eigvalsh((M + M.T) / 2)          # 负特征值同样截断
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(s1) + np.trace(s2)
                 - 2.0 * np.sum(np.sqrt(np.clip(w, 0.0, None))))


def main():
    rng = np.random.default_rng(SEED)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    gt = truth()
    kinds = ["全局刚体偏移", "局部抖动", "2% 慢漂移"]
    print(f"参考轨迹 {N} 帧 @ {1 / DT:.0f} Hz，{N * DT:.1f} 秒"
          f"，半径 {RADIUS:.1f} m，RPE 间隔 delta = {DELTA} 帧\n")
    print(f"  {'扰动':<14s}{'ATE 未对齐':>12s}{'ATE 刚体对齐':>14s}{'RPE':>12s}")
    res, poses = {}, {}
    for kind in kinds:
        est = perturb(kind, rng)
        poses[kind] = est
        row = (ate(est, gt), ate(align_rigid(est, gt), gt), rpe(est, gt))
        res[kind] = row
        print(f"  {kind:<14s}{row[0]:>12.5f}{row[1]:>14.5f}{row[2]:>12.5f}")

    print("\n  读法：")
    d = res["全局刚体偏移"]
    print(f"    全局偏移：位姿整体错了 {d[0]:.3f} m，刚体对齐后剩 {d[1]:.2e} m，"
          f"RPE 只有 {d[2]:.2e} m —— 三者里 RPE 最看不见。")
    d = res["局部抖动"]
    print(f"    局部抖动：ATE 未对齐 {d[0]:.4f} m，RPE {d[2]:.4f} m，"
          f"同量级 —— 只有这一类误差 RPE 抓得住。")
    d = res["2% 慢漂移"]
    print(f"    2% 慢漂移：ATE 未对齐 {d[0]:.4f} m，刚体对齐后仍是 {d[1]:.4f} m"
          f"（刚体对齐不含尺度，消不掉），RPE 只有 {d[2]:.2e} m。")

    # ---- 8.2 节两段代码的数值自检 ----
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(float)
    # 加一层 3 像素周期的细纹理：图像太光滑的话 sigma=2 的模糊几乎抹不掉东西，
    # SSIM 反而显得比加噪还高，那样这个自检会给出误导性的读数。
    base = (128.0 + 60.0 * np.sin(2 * np.pi * xx / 24.0)
            + 30.0 * np.cos(2 * np.pi * yy / 16.0)
            + 25.0 * np.sin(2 * np.pi * (xx + yy) / 3.0))
    noisy = base + rng.normal(0.0, 10.0, size=base.shape)
    blur = _blur(base, sigma=2.0)
    print(f"\n  8.2 节自检（图 {IMG}x{IMG}，自造，不用 cv2 / scipy）：")
    print(f"    SSIM(原图, 原图) = {ssim(base, base):.6f}   ← 应为 1")
    print(f"    SSIM(原图, 加噪) = {ssim(base, noisy):.4f}")
    print(f"    SSIM(原图, 模糊) = {ssim(base, blur):.4f}")
    F = rng.normal(0.0, 1.0, size=(512, 16))
    G = F @ rng.normal(np.eye(16), 0.05, size=(16, 16)) + 0.1
    print(f"    FID(X, X)        = {fid(F, F):.6e}   ← 应为 0")
    print(f"    FID(X, X 微扰)   = {fid(F, G):.4f}")

    # ---- 出图 ----
    fig, axes = plotting.plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    labels = ["ATE 未对齐", "ATE 刚体对齐", "RPE"]
    cols = ["#3b6ea5", "#7ea8cc", "#c0504d"]
    w = 0.26
    xs = np.arange(len(kinds))
    for j, (lab, col) in enumerate(zip(labels, cols)):
        raw = [res[k][j] for k in kinds]
        v = np.maximum(raw, FLOOR)
        b = ax.bar(xs + (j - 1) * w, v, w, label=lab, color=col)
        # 触到下限的要写成「<1e-5」，写成 1e-05 会被读成一个真实读数
        ax.bar_label(b, labels=[f"<{FLOOR:.0e}" if r < FLOOR else
                                (f"{r:.0e}" if r < 1e-3 else f"{r:.3f}")
                                for r in raw], fontsize=7, padding=1)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 5.0)
    ax.set_xticks(xs); ax.set_xticklabels(kinds, fontsize=9)
    ax.set_ylabel("误差（米，对数轴）")
    ax.set_title("(a) 三类误差源下三个指标的读数\n不到 1e-5 的按 1e-5 画（对数轴画不出 0）",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1]
    ax.plot(gt[:, 0, 3], gt[:, 1, 3], "-", color="#222", lw=2.0, label="真值")
    for kind, style, col in (("全局刚体偏移", "--", cols[0]),
                             ("2% 慢漂移", "-.", cols[2])):
        P = poses[kind]
        ax.plot(P[:, 0, 3], P[:, 1, 3], style, color=col, lw=1.4, label=kind)
    ax.set_aspect("equal")
    ax.set_xlabel("x（米）"); ax.set_ylabel("y（米）")
    ax.set_title("(b) 俯视图：这两种误差 RPE 都报成「几乎没有」", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    fig.suptitle("ATE 抓得到全局偏移与慢漂移，RPE 只抓局部抖动；"
                 "刚体对齐能消掉前者，消不掉后者", fontsize=11.5)
    fig.tight_layout()
    plotting.save(fig, "l_seq_metrics.png")


if __name__ == "__main__":
    main()
