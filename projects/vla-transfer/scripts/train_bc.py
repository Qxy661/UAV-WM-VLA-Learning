"""训练行为克隆基线。判定标准写在 PLAN.md：test 成功率 < 20% 就先修任务，不进 π0.5。

用法（开发机）：
    /opt/miniforge3/envs/pi0/bin/python projects/vla-transfer/scripts/train_bc.py \
        --data /root/work/data/uav96 --epochs 3 --batch 128 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bc.data import ChunkDataset, fit_stats                     # noqa: E402
from bc.model import BCNet, n_params                            # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--blind", action="store_true", help="训练时把图像置零，用来判断策略是否真的用到了视觉")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=int, default=2, help="帧采样步长，控制每个 epoch 的大小")
    ap.add_argument("--out", default=str(ROOT / "runs" / "bc"))
    ap.add_argument("--require-coverage", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=8, help="动作块长度，闭环时一次推理走这么多帧")
    ap.add_argument("--residual", action="store_true",
                    help="学习相对保高前馈的残差（动作 - base_action），缓解闭环沉底")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0, help="只用前 N 条 train 轨迹（数据规模曲线用）")
    a = ap.parse_args(argv)

    # 固定种子：同一条命令两次跑出来的曲线必须一致，否则"哪一档更好"分不清是方法还是随机数。
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator().manual_seed(a.seed)

    root = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")

    # 残差模式下归一化必须用物理尺度（推力 2 N、力矩 0.5 N·m），不能走数据方差：
    # 目标已经是"相对保高的增量"，若仍用绝对动作的 mean/std（推力均值 9.85），
    # 残差会被缩到 1e-3 量级——模型学到的近似常数，评测时 pred*std+mean 又还原成
    # 9.85 N 叠在基底上，推力翻倍、飞机先冲高再坠地。
    mean, std = fit_stats(root, require_coverage=a.require_coverage,
                          max_episodes=a.max_episodes, residual=a.residual)
    tr = ChunkDataset(root, ("train",), mean=mean, std=std, every=a.every, residual=a.residual,
                      require_coverage=a.require_coverage, max_episodes=a.max_episodes,
                      horizon=a.horizon)
    va = ChunkDataset(root, ("val",), mean=mean, std=std, every=max(a.every * 2, 2),
                      require_coverage=a.require_coverage, horizon=a.horizon,
                      residual=a.residual)
    print(json.dumps({"train_samples": len(tr), "val_samples": len(va), "device": str(dev)}))

    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, generator=gen,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = BCNet(horizon=a.horizon).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    lossf = nn.MSELoss()

    hist = []
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        run, n = 0.0, 0
        for img, st, tx, act in tl:
            img, st, tx, act = img.to(dev, non_blocking=True), st.to(dev), tx.to(dev), act.to(dev)
            if a.blind:
                img = torch.zeros_like(img)
            pred = model(img, st, tx)
            loss = lossf(pred, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * img.shape[0]
            n += img.shape[0]
        model.eval()
        vrun, vn = 0.0, 0
        with torch.no_grad():
            for img, st, tx, act in vl:
                vi = torch.zeros_like(img.to(dev)) if a.blind else img.to(dev)
                loss = lossf(model(vi, st.to(dev), tx.to(dev)), act.to(dev))
                vrun += loss.item() * img.shape[0]
                vn += img.shape[0]
        rec = {"epoch": ep + 1, "train_mse": round(run / max(n, 1), 6),
               "val_mse": round(vrun / max(vn, 1), 6),
               "elapsed_s": round(time.perf_counter() - t0, 1)}
        hist.append(rec)
        print(json.dumps(rec))
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "args": vars(a), "epoch": ep + 1}, out / "ckpt.pt")

    metrics = {
        "task": "t1_bc_baseline",
        "params": n_params(model),
        "epochs": a.epochs,
        "batch": a.batch,
        "blind": bool(a.blind),
        "every": a.every,
        "train_samples": len(tr),
        "val_samples": len(va),
        "history": hist,
        "train_mse": hist[-1]["train_mse"],
        "val_mse": hist[-1]["val_mse"],
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("params", "val_mse", "wall_time_s")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
