"""E8 服务端压测：动态批处理 × 到达率。

    python projects/vla-transfer/scripts/bench_serving.py --ckpt runs/sg_rel2/ckpt.pt \
        --data /root/work/data/uav96c --seconds 6

为什么要扫到达率：4 架机器 × 10 Hz = 40 请求/秒，单次 CPU 前向约 10 ms，一个核就能扛；
批处理的收益要到「到达率接近单核吞吐」时才出现。所以压测必须把到达率推到 100、200，
否则只能测出「批处理没用」这个假结论。

指标口径：
- `lat_ms`：进队到拿到结果（含排队、攒批、前向）。
- `stale_rate`：延迟超过该请求 deadline 的比例。stale 的语义是「沿用上一块动作」，
  不是丢请求——丢请求会让控制桥没有输入，只能走安全层降级，那是更差的结果。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

PV = Path(__file__).resolve().parents[1]  # projects/vla-transfer
sys.path.insert(0, str(PV))

from bc.model import STATE_DIM, BCNet, text_features  # noqa: E402
from serving.server import Request, Server  # noqa: E402


def load_model(ckpt: Path) -> tuple[torch.nn.Module, float]:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = BCNet(horizon=1, action_dim=3, state_dim=STATE_DIM)
    m.load_state_dict(obj["state_dict"])
    return m, float(obj.get("args", {}).get("goal_scale", 2.0))


def gather_frames(data: Path, n: int = 64) -> list[tuple[np.ndarray, np.ndarray, str]]:
    out = []
    for m in sorted(data.glob("shard_*/episode_*/meta.json")):
        z = np.load(m.parent / "obs.npz")
        if len(z["images"]) == 0:
            continue
        inst = json.loads(m.read_text(encoding="utf-8")).get("instruction", "")
        k = len(z["images"]) // 2
        ft = float(z["frame_t"][k])
        idx = int(np.argmin(np.abs(z["t"] - ft)))
        stv = np.concatenate([z["pos"][idx], z["vel"][idx], z["quat"][idx]]).astype(np.float32)
        out.append((z["images"][k], stv, inst))
        if len(out) >= n:
            break
    return out


def run_case(model, scale, frames, *, arrival_hz: float, max_batch: int, wait_ms: float,
             seconds: float, deadline_ms: float) -> dict:
    srv = Server(model, scale=scale, max_batch=max_batch, max_wait_ms=wait_ms)
    rng = np.random.default_rng(0)
    lats: list[float] = []
    stale = 0
    stop = threading.Event()
    n_sent = [0]

    def producer() -> None:
        # 生产者独立线程：真实系统里请求来自多架无人机，服务循环不该等它
        interval = 1.0 / arrival_hz
        t_next = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            if now >= t_next:
                img, stv, inst = frames[int(rng.integers(len(frames)))]
                srv.submit(Request(
                    rid=n_sent[0] % 4,
                    image=torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1),
                    state=torch.from_numpy(stv),
                    text=torch.from_numpy(text_features(inst)),
                    deadline_ms=deadline_ms,
                    t_arrive=now,
                ))
                n_sent[0] += 1
                t_next += interval
            else:
                time.sleep(min(0.0005, t_next - now))

    th = threading.Thread(target=producer, daemon=True)
    t0 = time.perf_counter()
    th.start()
    while time.perf_counter() - t0 < seconds:
        for rep in srv.step():
            lats.append(rep.lat_ms)
            stale += int(rep.stale)
        time.sleep(0.0005)
    stop.set()
    th.join(timeout=1.0)
    for _ in range(400):
        reps = srv.step()
        if not reps:
            break
        for rep in reps:
            lats.append(rep.lat_ms)
            stale += int(rep.stale)
    if not lats:
        return {}
    return {
        "arrival_hz": arrival_hz,
        "max_batch": max_batch,
        "wait_ms": wait_ms,
        "requests": len(lats),
        "batches": srv.batches,
        "req_per_batch": round(len(lats) / max(srv.batches, 1), 2),
        "p50_ms": round(float(np.percentile(lats, 50)), 2),
        "p90_ms": round(float(np.percentile(lats, 90)), 2),
        "p99_ms": round(float(np.percentile(lats, 99)), 2),
        "stale_rate": round(stale / len(lats), 4),
        "throughput_hz": round(len(lats) / seconds, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arrival-hz", type=float, default=40.0, help="最低一档到达率")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--deadline-ms", type=float, default=40.0)
    ap.add_argument("--wait-ms", type=float, default=2.0)
    a = ap.parse_args()

    model, scale = load_model(Path(a.ckpt))
    frames = gather_frames(Path(a.data))
    if not frames:
        raise SystemExit("没有可用帧")

    rows = []
    for arrival in (a.arrival_hz, 200.0, 1000.0, 4000.0):
        for max_batch in (1, 8):
            r = run_case(model, scale, frames, arrival_hz=arrival, max_batch=max_batch,
                         wait_ms=a.wait_ms, seconds=a.seconds, deadline_ms=a.deadline_ms)
            if r:
                rows.append(r)

    print(json.dumps({"deadline_ms": a.deadline_ms, "seconds": a.seconds, "rows": rows},
                     ensure_ascii=False, indent=2))
    print()
    print("| 到达 Hz | 批上限 | 每批请求 | P50 | P90 | P99 | 沿用上一块 | 吞吐 Hz |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['arrival_hz']:.0f} | {r['max_batch']} | {r['req_per_batch']} "
              f"| {r['p50_ms']:.1f} | {r['p90_ms']:.1f} | {r['p99_ms']:.1f} "
              f"| {r['stale_rate']:.1%} | {r['throughput_hz']:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
