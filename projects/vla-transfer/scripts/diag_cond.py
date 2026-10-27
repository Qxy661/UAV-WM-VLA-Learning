"""三个问题：BC 到底用不用图像、用不用指令。

指标是 teacher-forced 的动作 MSE，**归一化口径**（和 `train_bc.py` 的
`val_mse` 同一个数）。四列：

  true      原始输入
  text-     batch 内文本打乱（图像与状态不变）
  img-      图像置零
  zero      输出全零 = 直接预测训练集平均动作（归一化目标均值恒为 0）

判读：若 `text-` 与 `true` 接近（差 <5%），说明模型没读指令；
若 `img-` 与 `true` 接近，说明模型没读图像。三项都接近，就是"背了一条平均轨迹"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bc.data import ChunkDataset, fit_stats  # noqa: E402
from bc.model import BCNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/root/work/data/uav96m")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = obj.get("args", {})
    horizon = int(args.get("horizon", 8))
    residual = bool(args.get("residual", True))

    root = Path(a.data)
    mean, std = fit_stats(root, limit=400, require_coverage=0.0)
    for splits in (("val",), ("train",)):
        try:
            ds = ChunkDataset(root, splits, mean, std, horizon=horizon, every=4,
                              residual=residual, require_coverage=0.0,
                              max_episodes=64 if splits == ("train",) else 0)
            if len(ds) > 0:
                break
        except SystemExit:
            continue

    idx = np.linspace(0, len(ds) - 1, min(a.samples, len(ds))).astype(int)
    model = BCNet(horizon=horizon).to(a.device)
    sd = obj.get("model") or obj.get("state_dict") or obj
    model.load_state_dict(sd)
    model.eval()

    batch = [ds[int(i)] for i in idx]
    img = torch.stack([b[0] for b in batch]).to(a.device)
    st = torch.stack([b[1] for b in batch]).to(a.device)
    tx = torch.stack([b[2] for b in batch]).to(a.device)
    tgt = torch.stack([b[3] for b in batch]).to(a.device)

    def mse(x):
        return float(((x - tgt) ** 2).mean().item())

    with torch.no_grad():
        perm = torch.randperm(tx.shape[0], device=a.device)
        out = {
            "n": int(img.shape[0]),
            "splits": list(splits),
            "horizon": horizon,
            "residual": residual,
            "mse_true": round(mse(model(img, st, tx)), 4),
            "mse_text_shuffled": round(mse(model(img, st, tx[perm])), 4),
            "mse_img_zero": round(mse(model(torch.zeros_like(img), st, tx)), 4),
            "mse_zero_action": round(mse(torch.zeros_like(tgt)), 4),
            "target_var": round(float((tgt ** 2).mean().item()), 4),
        }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
