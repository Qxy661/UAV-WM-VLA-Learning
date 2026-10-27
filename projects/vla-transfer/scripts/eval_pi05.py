"""用 π0.5 服务端跑闭环评测，指标与 M6 / BC 同一套。

用法（开发机，先起大模型服务）：
    python projects/vla-transfer/scripts/eval_pi05.py \
        --host 127.0.0.1 --port 8000 --episodes 40 --seeds 8000 8100 \
        --instruction-file /root/work/data/uav96/meta/instructions_test.json \
        --out /root/work/runs/pi05_eval

注意：服务端策略与评测端必须用同一套 state/action 语义（10 维状态、4 维动作），
`--state-dim` 只为兼容 dummy 服务。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "uav-inspect"))
sys.path.insert(0, str(ROOT / "projects" / "vla-transfer"))

from eval.rollout import run_vla_episode                        # noqa: E402
from policy.pi05_client import RemotePolicy, RemotePolicyConfig  # noqa: E402


def bucket(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(v), size=(2000, len(v)))
    stats = v[idx].mean(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"mean": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def rate(values):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(v), size=(2000, len(v)))
    stats = v[idx].mean(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"rate": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def instruction_for(seed: int, held_out: bool) -> str:
    """评测指令与训练集用同一套模板，但 test 段用 held-out 表述。"""
    from uavinspect.data.instructions import make_instruction
    rack = 1 + (seed % 4)
    side = "front" if (seed // 4) % 2 == 0 else "oblique"
    return make_instruction(rack, side, random.Random(seed * 7919 + 13), held_out=held_out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs=2, default=(8000, 8100))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--state-dim", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "runs" / "pi05_eval"))
    ap.add_argument("--tag", default="pi05")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy = RemotePolicy(RemotePolicyConfig(host=args.host, port=args.port,
                                             action_dim=4, horizon=10, include_wrist=False))
    rows = []
    t0 = time.time()
    for i in range(args.episodes):
        seed = args.seeds[0] + i
        held_out = seed >= 7200
        text = instruction_for(seed, held_out)
        res, info = run_vla_episode(policy, seed, text, duration=args.duration)
        rows.append({**res.__dict__, **info, "held_out": held_out})
        print(f"[{i + 1}/{args.episodes}] seed={seed} succ={int(res.success)} "
              f"cov={res.coverage:.2f} clear={res.min_clearance:.2f} chunks={info['chunks']}")

    summary = {
        "agent": args.tag,
        "protocol": {"name": "t1-inspect", "version": "t1-v1", "duration": args.duration,
                     "action_hz": 9.6, "control_hz": 48, "img": 96},
        "success": rate([r["success"] for r in rows]),
        "coverage": bucket([r["coverage"] for r in rows]),
        "min_clearance": bucket([r["min_clearance"] for r in rows]),
        "time_to_coverage": bucket([r["time_to_coverage"] for r in rows
                                    if np.isfinite(r["time_to_coverage"])]),
        "latency_ms": policy.latency_summary(),
        "episodes": rows,
        "wall_s": round(time.time() - t0, 1),
    }
    (out / f"{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n成功 {summary['success']['rate'] * 100:.1f}% "
          f"[{summary['success']['ci95'][0] * 100:.1f}, {summary['success']['ci95'][1] * 100:.1f}] "
          f"覆盖 {summary['coverage']['mean']:.3f} 用时 {summary['wall_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
