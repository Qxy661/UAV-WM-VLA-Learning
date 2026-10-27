"""T1 数据生成：按指令飞一趟，把 (图像, 位姿, 动作, 指令) 落到磁盘。

分片跑：`--shard 0 --shards 32`，32 个进程各领一段 seed，写各自的 manifest；
跑完用 `merge_dataset.py` 合成一份。分片是为了把 8000 集从 4 小时压到十几分钟。

划分按 seed 段，不按帧：train 0..6399 / val 6400..7199 / test 7200..7999。
test 段同时换掉指令表述（`held_out=True`），否则测的是记忆。
"""

from __future__ import annotations

import argparse
import os
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load as load_config          # noqa: E402
from uavinspect.data.instructions import make_instruction   # noqa: E402
from uavinspect.data.recorder import compute_stats, write_manifest   # noqa: E402
from uavinspect.pipeline.inspect import run_episode        # noqa: E402
from uavinspect.sim.scene import resolve_layout            # noqa: E402

SPLIT_SIZES = {"train": 6400, "val": 800, "test": 800}


def split_for(seed: int) -> str:
    if seed < SPLIT_SIZES["train"]:
        return "train"
    if seed < SPLIT_SIZES["train"] + SPLIT_SIZES["val"]:
        return "val"
    return "test"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T1 巡检数据集生成（可按 shard 并行）")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "t1_dataset"))
    ap.add_argument("--total", type=int, default=sum(SPLIT_SIZES.values()))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--capture-every", type=int, default=5)
    ap.add_argument("--backend", default="rigid", choices=["rigid", "pybullet", "auto"])
    ap.add_argument("--perturb-scale", type=float, default=1.0)
    ap.add_argument("--perturb", action="store_true", help="从扰动状态起步，让专家示范恢复（不建议用于 test）")
    ap.add_argument("--layout", default="a", choices=["a", "b"],
                    help="场景布局：a 为训练仓储，b 为换仓库/换间距的迁移测试")
    ap.add_argument("--action-space", default="torque", choices=["torque", "omega"],
                    help="torque: 记 [thrust, tau]；omega: 记 [thrust, omega_ref]，角速率环留给低层")
    args = ap.parse_args(argv)
    os.environ["UAV_LAYOUT"] = args.layout

    cfg = load_config(args.config)
    out = Path(args.out)
    shard_dir = out / f"shard_{args.shard:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_base, args.seed_base + args.total))
    mine = [s for i, s in enumerate(seeds) if i % args.shards == args.shard]

    eps: list[dict] = []
    t_all = time.perf_counter()
    skipped = 0
    for seed in mine:
        side = "front" if seed % 2 == 0 else "oblique"
        held = split_for(seed) == "test"
        ep_id = f"episode_{seed:04d}"
        try:
            meta = run_episode(cfg, seed, shard_dir / ep_id, args.backend, perturb=args.perturb, perturb_scale=args.perturb_scale,
                               img_size=args.img_size, capture_every=args.capture_every,
                               only_label=side, held_out=held, action_space=args.action_space)
        except ValueError:
            # 该 seed 的场景里这个侧向被挡住。跳过并在统计里记一笔，不静默改成别的视点。
            skipped += 1
            continue
        meta["id"] = ep_id
        meta["dir"] = ep_id
        meta["split"] = split_for(seed)
        meta["layout"] = args.layout
        meta["viewpoint"] = side
        # 货架号由场景决定：build_warehouse 用 seed % len(rack_x) 选槽位，这里同规则推回。
        rack = int(seed % len(resolve_layout(args.layout)[1]["rack_x"])) + 1
        meta["rack"] = rack
        meta["instruction"] = make_instruction(
            rack, side, random.Random(seed * 7919 + 13), held_out=held)
        # run_episode 已经写过一次 meta.json，这里补字段后重写，保证落盘的就是最终版
        (shard_dir / ep_id / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        eps.append(meta)

    manifest = write_manifest(shard_dir, eps, val_every=1 << 30)   # 划分已由 split 决定
    for e in manifest["episodes"]:
        e["split"] = split_for(e["seed"])
    stats = compute_stats(shard_dir, manifest)
    payload = {
        "task": "t1_dataset_shard",
        "shard": args.shard,
        "shards": args.shards,
        "episodes": len(eps),
        "skipped": skipped,
        "frames": int(sum(e["frames"] for e in eps)),
        "img_size": args.img_size,
        "capture_every": args.capture_every,
        "wall_time_s": round(time.perf_counter() - t_all, 1),
        "splits": {k: sum(1 for e in eps if e["split"] == k) for k in SPLIT_SIZES},
        "stats_train": stats,
    }
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (shard_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (shard_dir / "shard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("shard", "episodes", "skipped", "frames",
                                              "wall_time_s", "splits")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
