"""解析式针孔相机：把场景渲染成小图，供世界模型与 VLA 用。

为什么不用 pybullet 的渲染：rigid 后端没有渲染器，而工程层的第一条纪律是
"关键路径必须真跑"。这里用光线投射（ray-AABB slab + 地面平面）给出与 rigid 后端
一致的几何视图——它不追求照片级，追求三件事：
  1. 遮挡是真的：墙和货架会挡住目标，覆盖率才不是自欺欺人；
  2. 分辨率可调且便宜（64×64 一帧约 1 ms），能撑起"每个 episode 上百帧"的数据量；
  3. 同一份几何既给图像又给深度，世界模型的预测误差可以落在像素与米两个单位上。

已知不真实：没有镜面反射/材质/运动模糊，光照是单一方向的兰伯特项，
所以在这上面训出来的模型不该指望迁移到真实相机。这条写进 ADR，不藏。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    t: np.ndarray        # (N,) 距离，inf 表示没打中
    normal: np.ndarray   # (N,3) 命中面法向
    box_id: np.ndarray   # (N,) int16，-1 表示地面，-2 表示天空


def _camera_basis(yaw: float, pitch: float) -> np.ndarray:
    """相机坐标系 → 世界系。列向量依次是 right / up / forward。"""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    forward = np.array([cy * cp, sy * cp, sp])
    right = np.array([-sy, cy, 0.0])
    up = np.cross(right, forward)      # 右手系：right × forward = up（指向图像上方）
    return np.column_stack([right, up, forward])


def rays(yaw: float, pitch: float, width: int, height: int, hfov_deg: float) -> np.ndarray:
    """返回 (H*W, 3) 的世界系单位方向。像素中心采样，行从上到下。"""
    R = _camera_basis(yaw, pitch)
    tan_h = np.tan(np.radians(hfov_deg) * 0.5)
    tan_v = tan_h * height / width
    xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (np.arange(height) + 0.5) / height * 2.0     # 图像第 0 行在上
    u, v = np.meshgrid(xs * tan_h, ys * tan_v)
    d_cam = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    d = d_cam @ R.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def raycast(origin: np.ndarray, dirs: np.ndarray, boxes, plane_z: float = 0.0) -> Hit:
    """一次投射全部光线。boxes 为带 .half / .position 属性的对象序列。"""
    origin = np.asarray(origin, dtype=float)
    n = len(dirs)
    best_t = np.full(n, np.inf)
    best_n = np.zeros((n, 3))
    best_id = np.full(n, -2, dtype=np.int16)

    # ① 地面：z = plane_z，只在下半空间有效
    dz = dirs[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = (plane_z - origin[2]) / dz
    ok = np.isfinite(t_plane) & (t_plane > 1e-6)
    best_t[ok] = t_plane[ok]
    best_n[ok] = np.array([0.0, 0.0, 1.0])
    best_id[ok] = -1

    # ② 轴对齐盒：slab 法
    for k, box in enumerate(boxes):
        half = np.asarray(box.half, dtype=float)
        c = np.asarray(box.center, dtype=float)
        tmin = np.full(n, -np.inf)
        tmax = np.full(n, np.inf)
        normal_axis = np.zeros((n, 3))
        for axis in range(3):
            d = dirs[:, axis]
            o = origin[axis] - c[axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-half[axis] - o) / d
                t2 = (half[axis] - o) / d
            near = np.minimum(t1, t2)
            far = np.maximum(t1, t2)
            upd = near > tmin
            normal_axis[upd] = 0.0
            normal_axis[upd, axis] = np.where(d[upd] > 0, -1.0, 1.0)
            tmin = np.where(upd, near, tmin)
            tmax = np.minimum(tmax, far)
        hit = (tmax >= tmin) & (tmax > 1e-6) & (tmin > 1e-6)
        closer = hit & (tmin < best_t)
        best_t[closer] = tmin[closer]
        best_n[closer] = normal_axis[closer]
        best_id[closer] = k
    return Hit(best_t, best_n, best_id)


def shade(hit: Hit, boxes, light_dir=(0.4, 0.3, 0.86), sky=(0.72, 0.80, 0.92),
          fill_dir=(0.0, -0.75, 0.35), ambient=0.22) -> np.ndarray:
    """把命中结果变成 RGB。地面灰、盒子用自带颜色、天空按方向渐变。

    两盏灯的近似：顶灯做明暗，-y 方向的补光专门照通道侧立面。
    只留顶灯的话，货架朝通道的那一面永远是暗的——而巡检目标就贴在那面上，
    渲染出来是一坨近黑的红，世界模型学不到任何东西。
    """
    n = len(hit.t)
    img = np.zeros((n, 3))
    L = np.asarray(light_dir, dtype=float)
    L = L / np.linalg.norm(L)
    F = np.asarray(fill_dir, dtype=float)
    F = F / np.linalg.norm(F)

    sky_mask = hit.box_id == -2
    if sky_mask.any():
        img[sky_mask] = np.asarray(sky)

    ground = hit.box_id == -1
    if ground.any():
        lam = np.abs(hit.normal[ground] @ L)
        img[ground] = np.array([0.42, 0.44, 0.40]) * (0.55 + 0.45 * lam)[:, None]

    for k, box in enumerate(boxes):
        m = hit.box_id == k
        if not m.any():
            continue
        rgba = getattr(box, "rgba", (0.8, 0.5, 0.2, 1.0))
        base = np.asarray(rgba[:3], dtype=float)
        nrm = hit.normal[m]
        lam = np.clip(0.72 * np.clip(nrm @ L, 0.0, 1.0) + 0.42 * np.clip(nrm @ F, 0.0, 1.0),
                      0.0, 1.0)
        em = float(getattr(box, "emissive", 0.0))
        lit = base * (ambient + (1.0 - ambient) * lam)[:, None]
        img[m] = lit * (1.0 - em) + base * em

    # 远处轻微雾化：给深度一点视觉线索，也避免世界模型把"远"当成"暗"
    fog = np.clip(hit.t / 40.0, 0.0, 0.35)
    fog = np.where(np.isfinite(fog), fog, 0.0)
    img = img * (1.0 - fog[:, None]) + np.asarray(sky)[None, :] * fog[:, None]
    return np.clip(img, 0.0, 1.0)


def render(
    origin,
    boxes,
    *,
    yaw: float,
    pitch: float,
    width: int = 64,
    height: int = 64,
    hfov_deg: float = 70.0,
    with_depth: bool = True,
):
    """返回 (image uint8 HxWx3, depth float32 HxW 或 None)。"""
    d = rays(yaw, pitch, width, height, hfov_deg)
    hit = raycast(origin, d, boxes)
    img = shade(hit, boxes).reshape(height, width, 3)
    depth = None
    if with_depth:
        depth = np.where(np.isfinite(hit.t), hit.t, 0.0).astype(np.float32)
        depth = depth.reshape(height, width)
    return (img * 255.0 + 0.5).astype(np.uint8), depth


def target_visible(origin, target_pos, **kw) -> bool:
    """目标是否落在视场内且未被遮挡（用单条射线判定）。"""
    d = np.asarray(target_pos, dtype=float) - np.asarray(origin, dtype=float)
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return True
    d = d / dist
    hit = raycast(np.asarray(origin, dtype=float), d[None, :], kw["boxes"])
    # 容差取"目标自身半径"量级：射线打中目标时 t 会比中心距离小半个尺寸，不是遮挡。
    return bool(not np.isfinite(hit.t[0]) or hit.t[0] >= dist - 0.15)


def in_view(origin, target_pos, *, yaw: float, hfov_deg: float,
            boxes, vfov_deg: float | None = None) -> bool:
    """目标是否在视锥内且未被遮挡：角度判定 + 单射线遮挡判定。

    只用 target_visible 会漏掉"目标在身后"的情况——射线照样能打中，
    但那不是"看见"。覆盖率指标必须两个条件都成立。
    """
    origin = np.asarray(origin, dtype=float)
    delta = np.asarray(target_pos, dtype=float) - origin
    horiz = float(np.linalg.norm(delta[:2]))
    if horiz < 1e-6:
        return False
    az = np.degrees(np.arctan2(delta[1], delta[0]) - yaw)
    az = (az + 180.0) % 360.0 - 180.0
    if abs(az) > hfov_deg * 0.5:
        return False
    el = np.degrees(np.arctan2(delta[2], horiz))
    vf = hfov_deg if vfov_deg is None else vfov_deg
    if abs(el) > vf * 0.5:
        return False
    return target_visible(origin, target_pos, boxes=boxes)
