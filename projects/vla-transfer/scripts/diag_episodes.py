"""把评测回合拆成三件事：飞到了没有、看没看到、最后是不是沉了。

用法（开发机）：
    python diag_episodes.py --runs /root/work/repo/runs/bc_r5_eval --limit 20
读每个 ep_*/meta.json 与 obs.npz，重建场景（seed 即目录名），算：
- d_min：整条轨迹到"指令视点"的最小水平+垂直距离
- visible：轨迹里是否存在一帧目标在视场内且未被遮挡（用 m2 的判定）
- z_end / z_min：高度
- reach：终点与该视点的距离
输出一张表 + 汇总，不做任何训练或服务调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "uav-inspect"))

from uavinspect.sim.scene import build_warehouse, visible_viewpoints  # noqa: E402
from uavinspect.sim.world import World  # noqa: E402
from uavinspect.config import load as load_cfg  # noqa: E402
from uavinspect.sim.camera import in_view  # noqa: E402
from uavinspect.util.mathx import quat_to_rot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--data-cfg", default=str(ROOT / "projects" / "uav-inspect" / "configs" / "default.yaml"))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.data_cfg))
    runs = Path(args.runs)
    rows = []
    for d in sorted(runs.glob("ep_*"))[: args.limit]:
        seed = int(d.name.split("_")[1])
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obs = np.load(d / "obs.npz")
        pos, quat = np.asarray(obs["pos"]), np.asarray(obs["quat"])

        world = World(cfg, "rigid")
        scene = build_warehouse(world, seed)
        vis = visible_viewpoints(scene)
        # 指令里的 viewpoints 字段是这一集被要求飞的点；没写就退化成全部可行点
        want = meta.get("viewpoints") or meta.get("covered") or []
        aims = []
        for vp in scene.viewpoints:
            if vp[2] in want:
                aims.append(np.asarray(vp[0], float))
        if not aims:
            aims = [np.asarray(vp[0], float) for vp, ok in zip(scene.viewpoints, vis) if ok]
        d_min = float(min(np.linalg.norm(pos - a, axis=1).min() for a in aims)) if aims else float("nan")

        seen = False
        for k in range(0, len(pos), max(1, len(pos) // 200)):
            for vp, ok in zip(scene.viewpoints, vis):
                R = quat_to_rot(quat[k])
                yaw = float(np.arctan2(R[1, 0], R[0, 0]))
                if ok and in_view(pos[k], scene.target_pos, yaw=yaw, hfov_deg=70.0,
                                  boxes=scene.boxes):
                    seen = True
                    break
            if seen:
                break

        rows.append(dict(seed=seed, d_min=round(d_min, 2), visible=seen,
                         z_end=round(float(pos[-1, 2]), 2), z_min=round(float(pos[:, 2].min()), 2),
                         n=len(pos), want=str(want)))

    print(f"{'seed':>6} {'d_min':>6} {'visible':>8} {'z_end':>6} {'z_min':>6} {'n':>5}  want")
    for r in rows:
        print(f"{r['seed']:>6} {r['d_min']:>6.2f} {str(r['visible']):>8} {r['z_end']:>6.2f} "
              f"{r['z_min']:>6.2f} {r['n']:>5}  {r['want']}")
    if rows:
        print(f"\nn={len(rows)}  d_min 中位 {np.median([r['d_min'] for r in rows]):.2f} m  "
              f"visible {sum(r['visible'] for r in rows)}/{len(rows)}  "
              f"z_end 中位 {np.median([r['z_end'] for r in rows]):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
