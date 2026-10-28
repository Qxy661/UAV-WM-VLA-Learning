"""占据栅格建图：只用深度射线，不用真值地图。

存在的理由：评测里"覆盖率"一直是拿真值算的，而部署时只有带漂移的位姿。
这一层要回答的是——位姿漂移多少时，地图先坏在哪。

做法是标准的两件事：
1. 每条射线沿途打"空闲"，端点打"占据"（log-odds 累加）；
2. 与真值占据栅格比 IoU / 召回 / 误占。

不做的：不确定性建模、回环、射线投递的体素内插值。这些都是 SLAM 的活，这里只量"漂移敏感性"。
"""

from __future__ import annotations

import numpy as np

from ..sim import camera

FREE_DELTA = -0.4
OCC_DELTA = 0.85
L_HIT = 1.0
L_MISS = -0.2
L_MIN, L_MAX = -4.0, 4.0


class GridMap:
    """2D 占据栅格（z 方向先塌成 2D：巡检关心"这一层通不通"）。"""

    def __init__(self, bounds_xy, res: float = 0.10, z_slice=(0.2, 2.2)):
        (x0, y0), (x1, y1) = bounds_xy
        self.res = res
        self.x0, self.y0 = x0, y0
        self.nx = int(np.ceil((x1 - x0) / res))
        self.ny = int(np.ceil((y1 - y0) / res))
        self.z_lo, self.z_hi = z_slice
        self.logodds = np.zeros((self.nx, self.ny), dtype=np.float32)

    # --- 坐标换算 ---------------------------------------------------------
    def to_cell(self, xy):
        ix = np.floor((np.asarray(xy)[..., 0] - self.x0) / self.res).astype(int)
        iy = np.floor((np.asarray(xy)[..., 1] - self.y0) / self.res).astype(int)
        return ix, iy

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    # --- 射线积分 ---------------------------------------------------------
    def integrate(self, origin, yaw: float, boxes, *, n_rays: int = 24,
                  hfov_deg: float = 70.0, max_range: float = 4.0,
                  plane_z: float = 0.0, pitches=(0.0,)) -> int:
        """一帧：每个 pitch 打 n_rays 条射线，沿途打空闲、端点打占据。返回命中数。

        单层水平射线只能看见与它同高的面，货架与墙的上下沿都会漏；所以默认留了
        pitches 参数，调用方一般传三档俯仰（贴地/平视/俯视）。
        """
        hits = camera.raycast(np.asarray(origin, float),
                              np.concatenate([camera.rays(yaw, float(pi), n_rays, 1, hfov_deg)
                                              for pi in pitches], axis=0), boxes, plane_z=plane_z)
        dirs = np.concatenate([camera.rays(yaw, float(pi), n_rays, 1, hfov_deg)
                               for pi in pitches], axis=0)
        occ = 0
        for d, t, bid in zip(dirs, hits.t, hits.box_id):
            t_use = float(min(max(t, 0.0), max_range))
            # 沿途空闲
            n_steps = max(int(t_use / (self.res * 0.9)), 1)
            for k in range(n_steps):
                sk = t_use * k / n_steps
                # 端点附近不判空闲：同一条射线的端点会被自己打上空闲，把命中抹掉
                if bid >= 0 and (t_use - sk) < 1.5 * self.res:
                    continue
                p = np.asarray(origin, float)[:2] + d[:2] * sk
                ix, iy = self.to_cell(p)
                if self.inside(ix, iy):
                    self.logodds[ix, iy] = np.clip(self.logodds[ix, iy] + L_MISS, L_MIN, L_MAX)
            # 端点占据（打中实体且不是地面/天空）
            if bid >= 0 and t < max_range:
                p = np.asarray(origin, float) + d * t
                if self.z_lo <= p[2] <= self.z_hi:
                    ix, iy = self.to_cell(p[:2])
                    if self.inside(ix, iy):
                        self.logodds[ix, iy] = np.clip(self.logodds[ix, iy] + L_HIT, L_MIN, L_MAX)
                        occ += 1
        return occ

    def occupancy(self):
        return self.logodds > 0.0


def true_map(boxes, bounds_xy, res: float = 0.10, z_slice=(0.2, 2.2)):
    """真值占据：把每个盒子的 XY 足迹栅格化（只在 z 切片内算）。"""
    g = GridMap(bounds_xy, res=res, z_slice=z_slice)
    for b in boxes:
        c = np.asarray(b.center, float)
        h = np.asarray(b.half, float)
        if c[2] + h[2] < z_slice[0] or c[2] - h[2] > z_slice[1]:
            continue
        xs = np.arange(c[0] - h[0], c[0] + h[0] + 1e-9, res)
        ys = np.arange(c[1] - h[1], c[1] + h[1] + 1e-9, res)
        ix, iy = g.to_cell(np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1))
        ok = g.inside(ix, iy)
        g.logodds[ix[ok], iy[ok]] = 1.0
    return g.occupancy()


def iou(pred, gt) -> float:
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return inter / union if union > 0 else 0.0


def recall(pred, gt) -> float:
    """真值占据里有多少被建出来（漏墙率）。"""
    tot = float(gt.sum())
    return float((pred & gt).sum()) / tot if tot > 0 else 0.0


def false_occ(pred, gt) -> float:
    """建出来的占据里有多少是不存在的（幽灵墙率）。"""
    tot = float(pred.sum())
    return float((pred & ~gt).sum()) / tot if tot > 0 else 0.0


def mean_offset_cm(pred, gt, res: float = 0.10) -> float:
    """预测占据格到最近真值占据格的平均距离（cm）。漂移越大这个数越大，
    而且它对"只看见一面墙"这种系统性漏检不敏感——IoU 会因此被低估。"""
    import numpy as _np
    if not pred.any() or not gt.any():
        return float("nan")
    pi = _np.argwhere(pred).astype(float) * res
    gi = _np.argwhere(gt).astype(float) * res
    d = _np.sqrt(((pi[:, None, :] - gi[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(_np.mean(d) * 100.0)


def visible_mask(poses, gt, boxes, res: float = 0.10, origin=(0.0, 0.0),
                 plane_z: float = 0.0, tol: float = 0.12, max_range: float = 8.0):
    """真值占据格里，哪些格从这条路线几何上本来就看得见（用于归一化召回）。

    做法：从每个位姿向每个真值格中心打一条射线，若首次命中就落在该格附近，算可见。
    这一层不含传感器噪声与视角遮挡，只回答"换谁来飞也建不出来的那些格"。
    """
    import numpy as _np
    from ..sim import camera as _cam
    cells = _np.argwhere(gt)
    ox, oy = float(origin[0]), float(origin[1])
    centers = _np.column_stack([ox + cells[:, 0] * res + res / 2.0,
                                oy + cells[:, 1] * res + res / 2.0])
    # 世界坐标：GridMap 的原点偏移在调用方，这里用相对格心 + 由调用方补偏移
    vis = _np.zeros(len(cells), dtype=bool)
    for (origin, _yaw) in poses:
        o = _np.asarray(origin, float)
        d = _np.column_stack([centers[:, 0], centers[:, 1], _np.full(len(cells), o[2])]) - o
        n = _np.linalg.norm(d, axis=1)
        ok = (n > 1e-6) & (n < max_range)
        if not ok.any():
            continue
        dirs = d[ok] / n[ok, None]
        hits = _cam.raycast(o, dirs, boxes, plane_z=plane_z)
        near = _np.abs(hits.t - n[ok]) < tol
        idx = _np.flatnonzero(ok)
        vis[idx[near]] = True
    mask = _np.zeros_like(gt, dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = vis
    return mask


def dilate(mask, k: int = 1):
    """栅格膨胀 k 格（方形结构元）。用于容忍离散化误差：0.1 m 栅格遇上 0.12 m 厚的墙，
    命中格与真值格差一格是常态，不算幽灵墙。"""
    import numpy as _np
    out = mask.copy()
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            sh = _np.zeros_like(mask)
            xs = slice(max(dx, 0), mask.shape[0] + min(dx, 0))
            ys = slice(max(dy, 0), mask.shape[1] + min(dy, 0))
            src_xs = slice(max(-dx, 0), mask.shape[0] + min(-dx, 0))
            src_ys = slice(max(-dy, 0), mask.shape[1] + min(-dy, 0))
            sh[xs, ys] = mask[src_xs, src_ys]
            out |= sh
    return out
