"""M2：脚本专家飞巡检路线，录制可训练的数据集。

一条 episode 的流程：
    起飞 → 依次飞向三个巡检视点（每个视点悬停 1.5 s、以 10 Hz 拍图）
    → 回到通道中点 → 降落。全程 48 Hz 记状态与动作，10 Hz 记图像。

指标（写入 metrics.json 与 episode meta）：
    coverage          三个视点里，目标真正落进视场且未被遮挡的比例
    route_time_s      从起飞到最后一个视点悬停结束的耗时
    frames            本次录到的图像帧数
    reach_ratio       按时到达的视点数 / 总数（超时不算到达）

产物：
    runs/m2_dataset/episode_XXXX/{obs.npz, meta.json}
    runs/m2_dataset/{manifest.json, stats.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                                    # noqa: E402
from uavinspect.control.cascaded import CascadedController             # noqa: E402
from uavinspect.data.recorder import EpisodeRecorder, compute_stats, write_manifest  # noqa: E402
from uavinspect.sim.camera import in_view, render                      # noqa: E402
from uavinspect.sim.scene import build_warehouse, visible_viewpoints
from uavinspect.pipeline.inspect import (CAPTURE_EVERY, HFOV, IMG_SIZE, run_episode, wrap_pi)   # noqa: E402
from uavinspect.sim.world import World                                 # noqa: E402
from uavinspect.util.latency import StageTimer                          # noqa: E402

IMG_SIZE = 64
HFOV = 70.0
CAPTURE_EVERY = 5          # 48 Hz / 5 ≈ 9.6 Hz
DWELL_S = 2.5
WP_TIMEOUT_S = 14.0
WP_TOL = 0.16


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 巡检录制")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "m2_dataset"))
    ap.add_argument("--backend", default="auto", choices=["auto", "rigid", "pybullet"])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--capture-every", type=int, default=CAPTURE_EVERY)
    ap.add_argument("--only-label", default=None, choices=["front", "oblique", "behind-rack"])
    ap.add_argument("--held-out", action="store_true")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps: list[dict] = []
    wall0 = time.perf_counter()
    for i in range(args.episodes):
        seed = args.seed_base + i
        ep_id = f"episode_{seed:04d}"
        t0 = time.perf_counter()
        meta = run_episode(cfg, seed, out / ep_id, args.backend,
                           img_size=args.img_size, capture_every=args.capture_every,
                           only_label=args.only_label, held_out=args.held_out)
        meta["id"] = ep_id
        meta["dir"] = ep_id
        meta["wall_time_s"] = round(time.perf_counter() - t0, 2)
        eps.append(meta)
        print(f"{ep_id}  seed={seed}  覆盖 {meta['coverage'] * 100:5.1f}%  "
              f"{meta['frames']:3d} 帧  {meta['wall_time_s']:.1f} s  "
              f"已覆盖={meta['covered']}")

    manifest = write_manifest(out, eps)
    stats = compute_stats(out, manifest)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    train = [e for e in manifest["episodes"] if e["split"] == "train"]
    val = [e for e in manifest["episodes"] if e["split"] == "val"]
    cov = float(np.mean([e["coverage"] for e in eps]))
    metrics = {
        "task": "m2_dataset",
        "episodes": len(eps),
        "split": {"train": len(train), "val": len(val)},
        "coverage_mean": round(cov, 4),
        "coverage_all": [round(e["coverage"], 4) for e in eps],
        "frames_total": int(sum(e["frames"] for e in eps)),
        "wall_time_s": round(time.perf_counter() - wall0, 2),
        "seeds": [e["seed"] for e in eps],
        "backend": eps[0]["backend"] if eps else None,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print()
    print(f"数据集：{len(eps)} 集（train {len(train)} / val {len(val)}），"
          f"共 {metrics['frames_total']} 帧，平均覆盖率 {cov * 100:.1f}%")
    print(f"产物：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
