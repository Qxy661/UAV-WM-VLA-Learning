"""E3 建图：位姿漂移多少时，占据栅格先坏在哪。

场景是布局 a 的仓储（四排货架 + 两排墙）。相机只给深度射线，位姿按"每米漂移 p%"注入
随机游走。每个漂移档飞同一条巡检路线（通道中线往返 + 两侧视点），比较建出来的 2D
占据栅格与真值栅格的 IoU / 漏墙率 / 幽灵墙率。

用法：
    py -3.9 projects/uav-inspect/scripts/e3_mapping.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from uavinspect.mapping import occupancy as om  # noqa: E402
from uavinspect.sim import camera, scene as sc  # noqa: E402

RES = 0.10
BOUNDS = ((-4.5, -2.8), (4.5, 2.8))
Z_SLICE = (0.2, 2.2)
N_RAYS = 48
PITCHES = (-0.22, 0.0, 0.22)
HFOV = 70.0
MAX_RANGE = 6.0


def route() -> list[tuple[tuple[float, float, float], float]]:
    """一条能看见四排货架的路线：沿通道中线走一趟，再斜看两个侧面。"""
    poses = []
    for x in np.linspace(-3.6, 3.6, 13):
        poses.append(((float(x), -1.9, 1.2), np.pi / 2))     # 面朝货架排
        poses.append(((float(x), 0.0, 1.2), np.pi / 2))
    for y in np.linspace(-1.6, 1.6, 7):
        poses.append(((-3.9, float(y), 1.2), 0.0))          # 侧墙
        poses.append(((3.9, float(y), 1.2), np.pi))
    # 两端墙：不专门回头看一眼就永远建不出来（这条本身就是探索策略问题）
    poses.append(((-4.2, 0.0, 1.2), np.pi))
    poses.append(((4.2, 0.0, 1.2), 0.0))
    poses.append(((0.0, -2.2, 1.2), -np.pi / 2))
    poses.append(((0.0, 2.2, 1.2), np.pi / 2))
    return poses


def drifted(poses, pct: float, rng) -> list:
    """按累计路程的比例注入漂移（随机游走，每步独立）。"""
    out = []
    err = np.zeros(3)
    prev = np.asarray(poses[0][0], float)
    for p, yaw in poses:
        p = np.asarray(p, float)
        step = float(np.linalg.norm(p - prev))
        if pct > 0:
            err = err + rng.normal(0.0, pct / 100.0 * step, 3)
        out.append((p + err, yaw))
        prev = p
    return out


def build(poses, boxes) -> np.ndarray:
    grid = om.GridMap(BOUNDS, res=RES, z_slice=Z_SLICE)
    for p, yaw in poses:
        grid.integrate(p, yaw, boxes, n_rays=N_RAYS, hfov_deg=HFOV, max_range=MAX_RANGE, pitches=PITCHES)
    return grid.occupancy()


def main() -> int:
    # 场景与物理引擎共用同一套盒子列表：建图只读 boxes，不推进物理
    from uavinspect import config as ucfg
    from uavinspect.sim.rigid import RigidEngine
    cfg = ucfg.load(REPO_ROOT / "configs" / "default.yaml")
    eng_obj = RigidEngine(cfg)
    sc.build_warehouse(eng_obj, seed=0)
    boxes = eng_obj.boxes

    gt = om.true_map(boxes, BOUNDS, res=RES, z_slice=Z_SLICE)
    poses = route()
    vis = om.visible_mask(poses, gt, boxes, res=RES, origin=(BOUNDS[0][0], BOUNDS[0][1]))
    rows = []
    for pct in (0.0, 0.5, 1.0, 2.0, 5.0):
        rng = np.random.default_rng(int(pct * 100))
        pred = build(drifted(poses, pct, rng), boxes)
        n_pred = int(pred.sum()); n_gtv = int((gt & vis).sum()); n_hit = int((pred & gt & vis).sum())
        rows.append({
            "drift_pct_per_m": pct,
            "iou": round(om.iou(pred, gt), 4),
            "recall": round(om.recall(pred, gt), 4),
            "false_occ": round(om.false_occ(pred, gt), 4),
            "mean_offset_cm": round(om.mean_offset_cm(pred, gt, RES), 2),
            "recall_visible": round(n_hit / n_gtv, 4) if n_gtv else 0.0,
            "false_occ_tol1": round(float((pred & ~om.dilate(gt, 1)).sum()) / max(n_pred, 1), 4),
            "pred_cells": int(pred.sum()),
            "gt_cells": int(gt.sum()),
        })
        print(rows[-1])
    res = {
        "scene": "warehouse_a",
        "res_m": RES,
        "poses": len(poses),
        "rays_per_pose": N_RAYS * len(PITCHES),
        "hfov_deg": HFOV,
        "rows": rows,
        "note": "2D 栅格；位姿漂移为按路程比例的随机游走；真值地图只由盒子足迹栅格化，不含传感器视角",
    }
    out = REPO_ROOT / "runs" / "e3_mapping"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
