"""感知：红色目标检测 + 反投影到三维。

做法刻意简单——阈值分割 + 几何反投影，不用神经网络。理由是这条线要回答的问题是
"给一个能用的检测，闭环误差还有多少被感知吃掉"，而不是"检测能做到多准"。用最简单的
检测器，量出来的误差就是**感知下限**：再准的检测器也只能改善这么多。

两个已知简化，写在代码里而不是脚注里：
- 相机俯仰当 0（本库视点都是水平拍货架面），所以只反投影到竖直平面 y = 货架面。
- 不建畸变模型；像素坐标直接用针孔模型。
"""

from __future__ import annotations

import numpy as np


def detect_red(img: np.ndarray, *, min_pixels: int = 8,
               r_floor: float = 0.55, ratio: float = 1.6) -> tuple[float, float, int] | None:
    """返回 (u, v, 像素数)，u/v 是归一化到 [-1,1] 的像素坐标；找不到返回 None。"""
    a = img.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > r_floor) & (r > ratio * g) & (r > ratio * b)
    n = int(mask.sum())
    if n < min_pixels:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    # 存储帧是**旋转 180°** 的（行 0 在下、列 0 在右），这是录制链路与 openpi 客户端
    # 共同约定；直接当普通像素坐标用会系统性地把目标算高、算偏，实测中位误差 66 cm。
    u = -(float(xs.mean()) / (w - 1) * 2.0 - 1.0)
    v = -(float(ys.mean()) / (h - 1) * 2.0 - 1.0)
    return u, v, n


def pixel_to_plane(u: float, v: float, *, pos, rot, plane_y: float,
                   hfov_deg: float = 70.0, aspect: float = 1.0) -> np.ndarray | None:
    """把像素方向打到 y = plane_y 平面上，返世界坐标；光线与该平面平行/反向时返回 None。

    `rot` 必须是**完整的机体旋转矩阵**，不是只取偏航：接近阶段机身有 10–20° 俯仰，
    只按偏航反投影会把目标算高半米（实测过，见 docs/13）。
    """
    half = np.tan(np.deg2rad(hfov_deg) / 2.0)
    # 机体系方向：x 前、y 左、z 上（与仿真里的机体约定一致）
    d = np.array([1.0, -u * half, -v * half * aspect], dtype=float)
    d /= np.linalg.norm(d)
    d_w = np.asarray(rot, dtype=float) @ d
    if abs(d_w[1]) < 1e-6:
        return None
    t = (plane_y - pos[1]) / d_w[1]
    if t <= 0:
        return None
    return np.asarray(pos, dtype=float) + t * d_w


def target_from_frame(img: np.ndarray, *, pos, rot, plane_y: float,
                      hfov_deg: float = 70.0) -> dict | None:
    hit = detect_red(img)
    if hit is None:
        return None
    u, v, n = hit
    p = pixel_to_plane(u, v, pos=pos, rot=rot, plane_y=plane_y, hfov_deg=hfov_deg)
    if p is None:
        return None
    return {"u": u, "v": v, "pixels": n, "xyz": p}
