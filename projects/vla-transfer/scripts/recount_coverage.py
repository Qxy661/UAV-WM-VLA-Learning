"""按 M6 协议（距视点 ≤0.7 m 且目标在视场内未被遮挡）重算覆盖率。

评测脚本沿用的是数据生成器的"到达判据"（离视点 ≤0.16 m 算到），
而 M6 基线表用的是"拍摄判据"（≤0.7 m 且真的看得见）。两套口径不能混着报，
所以这里只做一件事：拿已录制的轨迹，按 M6 口径重算每个可行视点是否被拍到。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "uav-inspect"))

from uavinspect.config import load as load_cfg  # noqa: E402
from uavinspect.sim.camera import in_view  # noqa: E402
from uavinspect.sim.scene import build_warehouse, visible_viewpoints  # noqa: E402
from uavinspect.sim.world import World  # noqa: E402
from uavinspect.util.mathx import quat_to_rot  # noqa: E402

COVER_MAX_DIST = 0.7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--config", default=str(ROOT / "projects" / "uav-inspect" / "configs" / "default.yaml"))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    rows = []
    for d in sorted(Path(args.runs).glob("ep_*"))[: args.limit]:
        seed = int(d.name.split("_")[1])
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obs = np.load(d / "obs.npz")
        pos, quat = np.asarray(obs["pos"]), np.asarray(obs["quat"])

        world = World(cfg, "rigid")
        scene = build_warehouse(world, seed)
        vis = visible_viewpoints(scene)

        covered = 0
        feasible = 0
        for vp, ok in zip(scene.viewpoints, vis):
            if not ok:
                continue
            feasible += 1
            aim = np.asarray(vp[0], float)
            hit = False
            for k in range(len(pos)):
                if np.linalg.norm(pos[k] - aim) > COVER_MAX_DIST:
                    continue
                R = quat_to_rot(quat[k])
                yaw = float(np.arctan2(R[1, 0], R[0, 0]))
                if in_view(pos[k], scene.target_pos, yaw=yaw, hfov_deg=70.0, boxes=scene.boxes):
                    hit = True
                    break
            covered += int(hit)

        rows.append(dict(seed=seed, covered=covered, feasible=feasible,
                         cov=round(covered / feasible, 3) if feasible else 0.0,
                         z_end=round(float(pos[-1, 2]), 2)))

    print(f"{'seed':>6} {'covered/feasible':>17} {'cov':>6} {'z_end':>6}")
    for r in rows:
        print(f"{r['seed']:>6} {str(r['covered']) + '/' + str(r['feasible']):>17} "
              f"{r['cov']:>6.3f} {r['z_end']:>6.2f}")
    if rows:
        n = len(rows)
        print(f"\nn={n}  平均覆盖率 {np.mean([r['cov'] for r in rows]):.3f}  "
              f"完全覆盖 {sum(1 for r in rows if r['cov'] >= 1.0)}/{n}  "
              f"部分覆盖 {sum(1 for r in rows if 0 < r['cov'] < 1)}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
