"""
Demo I — Plucker 坐标为什么能编码相机运动？

对应文档：docs/02-世界模型专题/02-生成式世界模型.md

要验证的结论：4.4 节说 Plucker 坐标"对相机运动有线性响应"，AirScape 把它
当像素级条件输入喂给扩散模型。这一节把这句话拆成三个可量的部分：

  1. 纯前向平移下，光轴中心那条射线的 Plucker 坐标完全不变
  2. 边缘射线变化量的模与平移量之比，恰好等于 sin(离轴角)
  3. Δm 对 Δp 严格线性，这就是"线性响应"的字面意思

三条合起来解释一件事：前进时画面从中心向外扩张，这个透视感全部来自
边缘射线自身的平移；中心射线只是沿着自己滑动，所在直线根本没动。

运行：py -3.9 code/i_plucker.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
W = H = 256                 # 像素网格边长
FOV_DEG = 60.0              # 水平视场角
DZ = 0.5                    # 前向平移量 (m)
Z_PLANE = 5.0               # 成像平面距离 (m)，用来算像素位移


def intrinsics(w, h, fov_deg):
    """针孔内参：主点在图像中心，焦距由视场角反推。"""
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0.0, w / 2 - 0.5],
                     [0.0, f, h / 2 - 0.5],
                     [0.0, 0.0, 1.0]])


def pixel_rays(K, w, h):
    """每个像素在相机坐标系下的单位方向，(w*h, 3)。相机朝 +z 看。"""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], -1)
    d = pix @ np.linalg.inv(K).T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def plucker(p, d):
    """Plucker 坐标 (d, m)，其中矩 m = p x d，p 取射线上任意一点。

    同一像素的 d 在纯平移下不变（方向不随平移改变），
    所以 Δm = Δp x d —— 对 Δp 是线性的，这正是它能当条件输入的原因。
    """
    return d, np.cross(p, d)


def project(points_cam, K):
    """相机坐标系下的三维点投影到像素坐标。先按 z 透视除法，再过内参。"""
    return (points_cam[:, :2] / points_cam[:, 2:3]) @ K[:2, :2].T + K[:2, 2]


def main():
    np.random.seed(SEED)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    K = intrinsics(W, H, FOV_DEG)
    d = pixel_rays(K, W, H)
    print(f"相机: {W}x{H}，水平视场角 {FOV_DEG:g}°，焦距 {K[0, 0]:.1f} px")

    # ---- 1. 纯前向平移：Δm = Δp x d ----
    p0, p1 = np.zeros(3), np.array([0.0, 0.0, DZ])
    _, m0 = plucker(p0, d)
    _, m1 = plucker(p1, d)
    dm = m1 - m0

    # 离轴角：射线与光轴 ẑ=(0,0,1) 的夹角
    theta = np.degrees(np.arccos(np.clip(d[:, 2], -1.0, 1.0)))
    ratio = np.linalg.norm(dm, axis=-1) / DZ            # |Δm| / |Δp|
    analytic = np.sin(np.radians(theta))                # 解析值 sin θ

    center = np.argmin(theta)                           # 最靠近光轴的那个像素
    # 光轴上那条射线单独算一遍：主点落在像素之间，网格里没有 θ 恰为 0 的像素
    dm_axis = np.cross(p1, np.array([0.0, 0.0, 1.0]))
    print(f"\n[1] 前向平移 {DZ} m 后，各像素射线的 |Δm|/|Δp|")
    print(f"    严格沿光轴的射线（d=(0,0,1)）    |Δm|/|Δp| = "
          f"{np.linalg.norm(dm_axis) / DZ:.3e}   （恒为 0）")
    print(f"    离它最近的像素  离轴角 {theta[center]:6.2f}°   |Δm|/|Δp| = "
          f"{ratio[center]:.4f}   sin θ = {analytic[center]:.4f}")
    for q in (10, 20, 30):
        i = np.argmin(np.abs(theta - q))
        print(f"    离轴角约 {theta[i]:5.1f}° 处   |Δm|/|Δp| = {ratio[i]:.4f}"
              f"   sin θ = {analytic[i]:.4f}")
    print(f"    实测与 sin θ 的最大偏差 {np.abs(ratio - analytic).max():.2e}"
          "   （浮点精度量级）")

    # ---- 2. Δm 对 Δp 是否严格线性 ----
    a, b = np.array([0.3, -0.4, 0.5]), np.array([-0.2, 0.1, 0.7])
    lhs = np.cross(a + b, d)
    rhs = np.cross(a, d) + np.cross(b, d)
    print(f"\n[2] 线性叠加：Δm(a+b) 与 Δm(a)+Δm(b) 的最大偏差 "
          f"{np.abs(lhs - rhs).max():.2e}")

    # ---- 3. 透视扩张：正对平面上的像素位移 ----
    # 每个像素方向对应平面 z=Z 上的一个点；相机前移后重新投影，量位移
    pts = d * (Z_PLANE / d[:, 2:3])
    pix0 = project(pts, K)
    pix1 = project(pts - np.array([0.0, 0.0, DZ]), K)
    move = np.linalg.norm(pix1 - pix0, axis=-1)
    radius = np.linalg.norm(pix0 - K[:2, 2], axis=-1)    # 到主点的像半径 (px)
    print(f"\n[3] 正对平面 z={Z_PLANE:g} m，相机前移 {DZ} m 造成的像素位移")
    print(f"    像半径 {radius[center]:4.1f} px 处，位移 {move[center]:.3f} px"
          f"   （位移/半径 = {move[center] / radius[center]:.4f}）")
    for r in (40, 80, 120, 160):
        i = np.argmin(np.abs(radius - r))
        print(f"    像半径 {radius[i]:5.0f} px 处，位移 {move[i]:6.2f} px"
              f"   （位移/半径 = {move[i] / radius[i]:.4f}）")
    # 位移应与像半径成正比：比值 = DZ / (Z - DZ)
    pred = DZ / (Z_PLANE - DZ)
    print(f"    位移/半径 的理论值 Δz/(Z-Δz) = {pred:.4f}，"
          f"实测均值 {np.mean(move / np.maximum(radius, 1e-9)):.4f}")

    # ---- 4. 对照：平移严格齐次，旋转不是 ----
    # 这正是 4.5 节要把 6-DoF 运动拆成平移与旋转两个分支的原因
    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    print("\n[4] 对照：把运动量翻倍，平移的响应严格翻倍，旋转的不是")
    r2 = np.linalg.norm(np.cross(2 * p1, d), axis=-1) / np.linalg.norm(np.cross(p1, d), axis=-1)
    d20 = d @ rot_y(np.radians(20.0)).T
    d40 = d @ rot_y(np.radians(40.0)).T
    r4 = (np.linalg.norm(d40 - d, axis=-1) / np.linalg.norm(d20 - d, axis=-1))
    print(f"    平移量 Δp 翻倍 -> |Δm| 比值 {r2.min():.6f} ~ {r2.max():.6f}"
          "   （严格 2，与离轴角无关）")
    print(f"    转角 φ 20°->40° -> |Δd| 比值 {r4.min():.6f} ~ {r4.max():.6f}"
          "   （明显小于 2）")
    # |Δd| = 2·sin(φ/2)·sin(与转轴的夹角)，所以比值是 sin20°/sin10°
    print(f"    解析对照：sin20°/sin10° = {np.sin(np.radians(20)) / np.sin(np.radians(10)):.6f}")

    print("\n结论：中心射线的 Plucker 矩在纯前向平移下严格不变（|Δm| = 0），"
          "\n      边缘射线的变化量按 sin(离轴角) 增长。同一件事在图像上就是："
          "\n      位移与像半径成正比，中心不动、边缘被推得最狠。"
          "\n      所以前进带来的透视扩张，全部由边缘射线自身的平移贡献。")

    # ---- 出图 ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    order = np.argsort(theta)
    sub = order[::211]                                    # 抽稀，否则 65536 个点糊成一片
    ax.plot(theta[order], analytic[order], "-", color="#c0392b", lw=1.6, label="解析值 sin θ")
    ax.scatter(theta[sub], ratio[sub], s=16, color="#2471a3", zorder=3, label="实测 |Δm|/|Δp|")
    ax.annotate("中心射线：恒为 0\n（沿自身滑动，所在直线没动）",
                xy=(0, 0), xytext=(9, 0.14), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlabel("离轴角 θ（度）"); ax.set_ylabel("|Δm| / |Δp|")
    ax.set_title("(a) 射线的 Plucker 响应只取决于离轴角")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    ax = axes[1]
    sub2 = order[::211]
    ax.scatter(radius[sub2], move[sub2], s=16, color="#2471a3", label="实测像素位移")
    rr = np.linspace(0, radius.max(), 50)
    ax.plot(rr, rr * pred, "--", color="#c0392b", lw=1.6, label=f"Δz/(Z-Δz) · r = {pred:.3f} r")
    ax.set_xlabel("像半径 r（像素，到主点的距离）"); ax.set_ylabel("位移（像素）")
    ax.set_title("(b) 前进时画面向外扩张：位移正比于像半径")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    fig.suptitle("Plucker 坐标把透视扩张编码成：离轴角越大，响应越强", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "i_plucker.png")


if __name__ == "__main__":
    main()
