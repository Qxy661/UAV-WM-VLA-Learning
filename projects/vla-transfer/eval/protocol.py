"""评测判据的唯一实现：M6 协议。

一条轨迹算不算"拍到了"某个视点，要同时满足三件事：
1. 机体距该视点 ≤ COVER_MAX_DIST（0.7 m）；
2. 目标落在视场角内；
3. 机体到目标的射线不被遮挡。

这跟数据生成器里的"到达判据"（离视点 ≤0.16 m 算到）不是一回事。两套口径混用会
给出两个完全不同的成功率——实测同一批 20 条轨迹，按到达判据是 0/20，按拍摄判据
是 4/20 完全覆盖、5/20 部分覆盖。所以判据只能有一份实现，这里就是它。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

COVER_MAX_DIST = 0.7
HFOV_DEG = 70.0


def marker_uv(pos: np.ndarray, yaw: float, target: np.ndarray, *,
              hfov_deg: float = HFOV_DEG) -> tuple[float, float]:
    """目标在画面里的归一化坐标 (u, v)：0 是正中心，±1 是视场边缘。

    相机是解析式针孔、方形画幅，所以水平/垂直视场相同。u 向右为正，v 向上为正。
    """
    d = np.asarray(target, float) - np.asarray(pos, float)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    x = float(d @ fwd)
    if x <= 1e-6:                     # 目标在身后，直接算作视场外
        return 9.0, 9.0
    half = np.tan(np.deg2rad(hfov_deg) / 2.0)
    return float((d @ right) / x / half), float((d @ up) / x / half)


def m6_coverage(pos: np.ndarray, yaw: np.ndarray, scene, *,
                max_dist: float = COVER_MAX_DIST, hfov_deg: float = HFOV_DEG,
                center_tol: float = 0.0,
                feasible: Sequence[bool] | None = None) -> tuple[int, int, list[str]]:
    """返回 (拍到数, 可行视点数, 拍到的视点标签)。yaw 是每帧的标量偏航角。

    center_tol > 0 时额外要求目标落在画面水平方向中心附近（|u| ≤ center_tol）。
    只看水平：视点的竖直位置锚在货架中心，目标的竖向抖动由视场角覆盖，不是
    需要"对准"的量。水平偏移才是位置与指令推不出来的那一个自由度。
    """
    from uavinspect.sim.camera import in_view

    if feasible is None:
        from uavinspect.sim.scene import visible_viewpoints
        feasible = visible_viewpoints(scene)

    covered: list[str] = []
    feasible_n = 0
    for vp, ok in zip(scene.viewpoints, feasible):
        if not ok:
            continue
        feasible_n += 1
        aim = np.asarray(vp[0], float)
        hit = False
        for k in range(len(pos)):
            if np.linalg.norm(pos[k] - aim) > max_dist:
                continue
            if not in_view(pos[k], scene.target_pos, yaw=float(yaw[k]),
                           hfov_deg=hfov_deg, boxes=scene.boxes):
                continue
            if center_tol > 0.0:
                u, _v = marker_uv(pos[k], float(yaw[k]), scene.target_pos,
                                  hfov_deg=hfov_deg)
                if abs(u) > center_tol:
                    continue
            hit = True
            break
        if hit:
            covered.append(vp[2])
    return len(covered), feasible_n, covered


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    """四元数序列 -> 偏航角序列（只取绕 z 的转角）。"""
    q = np.asarray(quat, float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
