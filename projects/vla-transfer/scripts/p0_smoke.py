"""P0 冒烟：一条 dummy 观测打到策略服务器，验证「服务能起来 + 动作块形状对 + 延迟有数」。

用 pi05_libero 的配置跑时动作是 7 维（机械臂）；换成无人机微调后的配置会是 4 维。
脚本不假设维度，只检查：二维形状、有限值、延迟落在能写进报告的范围。

用法（开发机，pi0 环境里）：
    python projects/vla-transfer/scripts/p0_smoke.py --port 8000 --out runs/p0_smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from policy.pi05_client import RemotePolicy, RemotePolicyConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--state-dim", type=int, default=8, help="pi05_libero 是 8；无人机是 12")
    ap.add_argument("--libero", action="store_true", help="给 pi05_libero 补第二路腕部相机")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "runs" / "p0_smoke"))
    args = ap.parse_args()

    policy = RemotePolicy(RemotePolicyConfig(host=args.host, port=args.port,
                                                          include_wrist=args.libero)).connect()
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    state = np.zeros(args.state_dim, dtype=np.float32)
    prompt = "去 2 号货架前，拍下目标"

    t0 = time.perf_counter()
    first = policy.infer(image, state, prompt)
    first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(args.runs):
        policy.infer(image, state, prompt)

    checks = {
        "shape_is_2d": first.ndim == 2,
        "finite": bool(np.isfinite(first).all()),
        "nonempty": first.shape[0] > 0 and first.shape[1] > 0,
        "latency_is_number": first_ms > 0,
    }
    summary = {
        "checks": checks,
        "passed": all(checks.values()),
        "action_shape": list(first.shape),
        "action_first_row": [round(float(x), 4) for x in first[0]],
        "action_range": [round(float(first.min()), 3), round(float(first.max()), 3)],
        "first_call_ms": round(first_ms, 1),
        "latency": policy.latency_summary(),
        "endpoint": f"{args.host}:{args.port}",
        "note": "P0 只验证管道；7 维动作来自 pi05_libero，无人机微调后的配置是 4 维",
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
