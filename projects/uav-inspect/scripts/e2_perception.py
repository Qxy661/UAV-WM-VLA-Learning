"""E2 感知：检测率与反投影定位误差。

问的是两个工程问题，不是"检测器多强"：

1. 多远的距离还能检出红色目标？（决定 `d_safe` 与视点距离）
2. 检出来之后，反投影的三维位置差多少？（这部分误差会直接进子目标模型的标签上限）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.perception import detector  # noqa: E402
from uavinspect.util.mathx import quat_to_rot  # noqa: E402

HFOV = 70.0


def yaw_of(quat) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def control_index(frame_t: np.ndarray, t: np.ndarray) -> np.ndarray:
    """每帧对应哪个控制步：取 t 里最接近的那个（帧率低于控制率）。"""
    return np.abs(t[None, :] - frame_t[:, None]).argmin(axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2 感知误差")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "e2_perception"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, misses, total = [], 0, 0
    for ep in sorted(Path(a.data).glob("shard_*/episode_*/obs.npz")):
        meta = json.loads((ep.parent / "meta.json").read_text(encoding="utf-8"))
        tp = meta.get("target_pos")
        if not tp:
            continue
        tp = np.asarray(tp, dtype=float)
        with np.load(ep) as z:
            imgs, t, pos, quat = z["images"], z["t"], z["pos"], z["quat"]
            ft = z["frame_t"]
        idx = control_index(ft, t)
        hist: list[np.ndarray] = []
        for i, k in enumerate(idx):
            total += 1
            est = detector.target_from_frame(imgs[i], pos=pos[k], rot=quat_to_rot(quat[k]),
                                             plane_y=tp[1], hfov_deg=HFOV)
            if est is None:
                misses += 1
                continue
            raw = np.asarray(est["xyz"], dtype=float)
            hist.append(raw)
            hist = hist[-3:]
            sm = np.median(np.stack(hist), axis=0)
            jitter = float(np.linalg.norm(raw - hist[-2])) if len(hist) > 1 else 0.0
            rows.append({
                "dist_m": float(np.linalg.norm(pos[k] - tp)),
                "err_m": float(np.linalg.norm(raw - tp)),
                "err_sm_m": float(np.linalg.norm(sm - tp)),
                "jitter_m": jitter,
                "err_x_m": float(raw[0] - tp[0]),
                "err_z_m": float(raw[2] - tp[2]),
                "pixels": est["pixels"],
            })

    res = {"frames": total, "detected": len(rows), "detect_rate": round(len(rows) / max(total, 1), 4)}
    if rows:
        d = np.array([r["dist_m"] for r in rows])
        e = np.array([r["err_m"] for r in rows])
        ex = np.array([r["err_x_m"] for r in rows])
        ez = np.array([r["err_z_m"] for r in rows])
        res["localisation"] = {
            "err_median_cm": round(float(np.median(e)) * 100, 2),
            "err_p90_cm": round(float(np.percentile(e, 90)) * 100, 2),
            "err_max_cm": round(float(e.max()) * 100, 2),
            "bias_x_cm": round(float(ex.mean()) * 100, 2),
            "bias_z_cm": round(float(ez.mean()) * 100, 2),
            "dist_median_m": round(float(np.median(d)), 3),
        }
        esm = np.array([r["err_sm_m"] for r in rows])
        jit = np.array([r["jitter_m"] for r in rows])
        res["smoothing"] = {
            "window": 3,
            "raw_median_cm": round(float(np.median(e)) * 100, 2),
            "smoothed_median_cm": round(float(np.median(esm)) * 100, 2),
            "jitter_median_cm": round(float(np.median(jit)) * 100, 2),
            "jitter_p90_cm": round(float(np.percentile(jit, 90)) * 100, 2),
        }
        # 误差随距离是否增长（针孔模型下应该近似线性）
        bins = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 99.0)]
        res["by_distance"] = {}
        for lo, hi in bins:
            m = (d >= lo) & (d < hi)
            if m.any():
                res["by_distance"][f"{lo}-{hi}m"] = {
                    "n": int(m.sum()),
                    "err_median_cm": round(float(np.median(e[m])) * 100, 2),
                    "pixels_median": int(np.median([r["pixels"] for r, mm in zip(rows, m) if mm])),
                }
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
