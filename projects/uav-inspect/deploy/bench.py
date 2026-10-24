"""M7：把世界模型的三个推理路径摊开测延迟，并给出 P99 与预算的对照。

部署里真正跑在关键路径上的只有两个算子：

    enc   每帧编码一次（相机 10 Hz）
    step  GRU 单步 + 可见性头（世界模型查询 10 Hz）
    dec   解码成图，只用于遥测/可视化，不在关键路径

本脚本分别测它们的单次延迟，报 P50/P90/P99/max，并对着 100 ms（10 Hz）预算判定。
ONNX 导出单独一条：`onnx` 装不上时给出明确错误而不是静默跳过。
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uavinspect.util.latency import StageTimer  # noqa: E402
from wm.model import WorldModel  # noqa: E402

BUDGET_MS_10HZ = 100.0
BUDGET_MS_5HZ = 200.0


def load_model(ckpt_path: Path, device: str = "cpu") -> WorldModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck["args"]
    wm = WorldModel(dim=int(args.get("dim", 128)), act_dim=4).to(device)
    wm.load_state_dict(ck["model"])
    wm.eval()
    return wm


class EncOnly(torch.nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm

    def forward(self, frame):
        return self.wm.enc(frame)


class StepOnly(torch.nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm

    def forward(self, h, a):
        h2 = self.wm.step(h, a)
        return h2, self.wm.vis(h2)


class DecOnly(torch.nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm

    def forward(self, h):
        return self.wm.dec(h)


def timeit(fn, args, *, warmup=10, runs=200) -> dict:
    """单线程、固定输入，反复调用。返回毫秒分位数。"""
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(*args)
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    n = len(samples)

    def q(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return round(samples[idx], 4)

    return {
        "runs": n,
        "mean": round(statistics.fmean(samples), 4),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": round(samples[-1], 4),
    }


def export_onnx(wm: WorldModel, out: Path) -> dict:
    """静态形状导出。失败时把原因写进返回值，不抛。"""
    import tempfile

    step = StepOnly(wm).eval()
    h = torch.zeros(1, wm.dim)
    a = torch.zeros(1, 4)
    try:
        with torch.no_grad():
            torch.onnx.export(
                step, (h, a), str(out),
                input_names=["h", "action"], output_names=["h_next", "vis_logit"],
                opset_version=17, dynamo=False,
            )
        return {"ok": True, "path": str(out), "bytes": out.stat().st_size}
    except Exception as e:                       # 缺 onnx 包时走这里
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                "hint": "需要 `pip install onnx`；onnxruntime 已装但读不了 torch 的导出器输出"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "runs/m3_wm/ckpt.pt"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "runs/m7_deploy"))
    ap.add_argument("--onnx", action="store_true", help="尝试导出 ONNX（需要 onnx 包）")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[m7] 找不到 {ckpt}，先跑 wm/train.py 或从开发机取回 ckpt.pt")
        return 2

    wm = load_model(ckpt)
    frame = torch.rand(1, 3, 64, 64)
    h = torch.zeros(1, wm.dim)
    a = torch.zeros(1, 4)

    enc = timeit(EncOnly(wm), (frame,), runs=args.runs)
    step = timeit(StepOnly(wm), (h, a), runs=args.runs)
    dec = timeit(DecOnly(wm), (h,), runs=args.runs)

    budget = {
        "enc_p99_ok": enc["p99"] <= BUDGET_MS_10HZ,
        "step_p99_ok": step["p99"] <= BUDGET_MS_10HZ,
        "enc_headroom_x": round(BUDGET_MS_10HZ / max(enc["p99"], 1e-6), 2),
        "step_headroom_x": round(BUDGET_MS_10HZ / max(step["p99"], 1e-6), 2),
    }
    passed = bool(budget["enc_p99_ok"] and budget["step_p99_ok"])

    params = sum(p.numel() for p in wm.parameters())
    out = {
        "m7_deploy": {
            "params": params,
            "params_m": round(params / 1e6, 2),
            "threads": args.threads,
            "budget_ms": {"10hz": BUDGET_MS_10HZ, "5hz": BUDGET_MS_5HZ},
            "paths": {"enc": enc, "step": step, "dec": dec},
            "checks": budget,
            "passed": passed,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
    }
    if args.onnx:
        out["m7_deploy"]["onnx"] = export_onnx(wm, Path(args.out) / "step.onnx")

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(f"{'路径':<6} {'P50':>8} {'P90':>8} {'P99':>8} {'max':>8}  单位 ms")
    for name, r in (("enc", enc), ("step", step), ("dec", dec)):
        print(f"{name:<8} {r['p50']:>8.3f} {r['p90']:>8.3f} {r['p99']:>8.3f} {r['max']:>8.3f}")
    print(f"10 Hz 预算 {BUDGET_MS_10HZ:.0f} ms：enc 余量 {budget['enc_headroom_x']}x，"
          f"step 余量 {budget['step_headroom_x']}x -> {'通过' if passed else '未通过'}")
    if args.onnx:
        on = out["m7_deploy"]["onnx"]
        print(f"ONNX 导出：{'成功，' + str(on['bytes']) + ' 字节' if on['ok'] else '失败 ' + on['error']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
