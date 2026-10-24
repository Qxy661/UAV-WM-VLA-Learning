"""M3：在 M2 录的数据上训练世界模型，并与无模型基线比。

三个指标，缺一个都说明不了问题：
    rec_mse      看到真帧时的重建误差（模型能不能表示这一帧）
    roll_mse     只给动作、开环递推 horizon 步的误差（能不能预测未来）
    vis_auc      目标可见性的排序质量（语义量，不受像素 MSE 尺度影响）

基线：
    persistence  下一帧 = 上一帧；可见性 = 沿用当前值。这是世界模型必须赢的那个"什么都不做"。
    mean_frame   下一帧 = 训练集平均帧（下界，证明指标不是恒 0 也恒对）

验收：val 上 roll_mse 低于 persistence，且 vis_auc 高于 persistence。两个都赢才算过。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG))

from wm.dataset import build_windows, load_episodes, normalise   # noqa: E402
from wm.model import WorldModel                                   # noqa: E402


def to_tensors(win, device):
    """把 numpy 窗口包成 tensor 命名空间，后面 evaluate/pairs 只认属性名。"""
    from types import SimpleNamespace
    return SimpleNamespace(frames=torch.as_tensor(win.frames, device=device),
                           actions=torch.as_tensor(win.actions, device=device),
                           visible=torch.as_tensor(win.visible, device=device),
                           episode=win.episode)


def pairs(win, ctx: int, horizon: int):
    """(上下文帧, 上下文+未来动作, 未来帧, 未来可见性)。"""
    fut = slice(ctx, ctx + horizon)
    return (win.frames[:, :ctx], win.actions[:, :ctx + horizon],
            win.frames[:, fut], win.visible[:, fut])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """不依赖 sklearn 的 AUC。全是同号时返回 0.5（说明这批样本分不出正负）。"""
    pos = labels > 0.5
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


@torch.no_grad()
def evaluate(model, win, ctx: int, horizon: int, mean_frame: torch.Tensor) -> dict:
    model.eval()
    ctx_f, acts, fut_f, fut_v = pairs(win, ctx, horizon)
    rec_h, roll_h, roll_frames, vis_logits = model(ctx_f, acts, ctx, base=ctx_f[:, -1])
    roll_mse = float(torch.mean((roll_frames - fut_f) ** 2))
    rec_mse = float(torch.mean((model.dec(rec_h.reshape(-1, model.dim))
                                .view(ctx_f.shape[0], ctx, *ctx_f.shape[2:]) - ctx_f) ** 2))

    # 基线①：冻结最后一帧
    last = win.frames[:, ctx - 1:ctx]
    persist_mse = float(torch.mean((last - fut_f) ** 2))
    # 基线②：训练集平均帧
    mean_mse = float(torch.mean((mean_frame[None, None] - fut_f) ** 2))

    vis_score = torch.sigmoid(vis_logits[:, ctx:]).detach().cpu().numpy().reshape(-1)
    vis_true = fut_v.detach().cpu().numpy().reshape(-1)
    persist_vis = np.repeat(win.visible[:, ctx - 1:ctx].detach().cpu().numpy(),
                            horizon, axis=1).reshape(-1)
    return {
        "rec_mse": rec_mse, "roll_mse": roll_mse,
        "persist_mse": persist_mse, "mean_mse": mean_mse,
        "vis_auc": auc(vis_score, vis_true),
        "vis_auc_persist": auc(persist_vis, vis_true),
        "vis_pos_rate": float(vis_true.mean()),
        "n": int(len(vis_true)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M3 世界模型训练与评测")
    ap.add_argument("--data", default=str(PKG / "runs" / "m2_dataset"))
    ap.add_argument("--out", default=str(PKG / "runs" / "m3_wm"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--horizon-steps", type=int, default=10,
                    help="评测时开环递推的步数（可以比训练时更长）")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    eps = load_episodes(Path(args.data))
    tr_win, stats = normalise(build_windows(eps["train"], ctx=args.ctx,
                                            horizon=args.horizon_steps))
    va_win, _ = normalise(build_windows(eps["val"], ctx=args.ctx,
                                        horizon=args.horizon_steps), stats)
    tr_win, va_win = to_tensors(tr_win, device), to_tensors(va_win, device)

    model = WorldModel(dim=args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_par = sum(p.numel() for p in model.parameters())

    # 归一化空间的"平均帧"就是 0（原空间均值帧减去均值再除以标准差）
    mean_frame = torch.zeros(3, 1, 1, device=device)
    ctx_f, acts, fut_f, fut_v = pairs(tr_win, args.ctx, args.horizon_steps)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ctx_f))
        tot = {"rec": 0.0, "roll": 0.0, "vis": 0.0}
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            rec_h, roll_h, roll_frames, vis_logits = model(ctx_f[idx], acts[idx], args.ctx,
                                                    base=ctx_f[idx, -1])
            rec_frames = model.dec(rec_h.reshape(-1, model.dim)).view(
                len(idx), args.ctx, *ctx_f.shape[2:])
            l_rec = torch.mean((rec_frames - ctx_f[idx]) ** 2)
            l_roll = torch.mean((roll_frames - fut_f[idx]) ** 2)
            l_vis = torch.nn.functional.binary_cross_entropy_with_logits(
                vis_logits[:, args.ctx:], fut_v[idx])
            loss = l_rec + l_roll + 0.2 * l_vis
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot["rec"] += float(l_rec) * len(idx)
            tot["roll"] += float(l_roll) * len(idx)
            tot["vis"] += float(l_vis) * len(idx)
        n = len(perm)
        if (ep + 1) % 3 == 0 or ep == 0:
            va = evaluate(model, va_win, args.ctx, args.horizon_steps, mean_frame)
            print(f"epoch {ep + 1:2d}  train rec {tot['rec'] / n:.5f} roll {tot['roll'] / n:.5f} "
                  f"| val rec {va['rec_mse']:.5f} roll {va['roll_mse']:.5f} "
                  f"(persist {va['persist_mse']:.5f}, mean {va['mean_mse']:.5f}) "
                  f"| vis AUC {va['vis_auc']:.3f} vs persist {va['vis_auc_persist']:.3f}")
    train_s = time.perf_counter() - t0

    tr = evaluate(model, tr_win, args.ctx, args.horizon_steps, mean_frame)
    va = evaluate(model, va_win, args.ctx, args.horizon_steps, mean_frame)
    checks = {
        "beats_persistence_mse": va["roll_mse"] < va["persist_mse"],
        "beats_mean_mse": va["roll_mse"] < va["mean_mse"],
        "beats_persistence_vis": va["vis_auc"] > va["vis_auc_persist"],
    }
    passed = all(checks.values())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "task": "m3_wm", "passed": passed, "checks": checks,
        "params": vars(args),
        "n_params": n_par,
        "train_seconds": round(train_s, 2),
        "windows": {"train": int(tr_win.frames.shape[0]), "val": int(va_win.frames.shape[0])},
        "metrics": {"train": tr, "val": va},
    }
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)},
               out / "ckpt.pt")

    print(f"\n参数 {n_par / 1e3:.1f}k，窗口 train {tr_win.frames.shape[0]} / val {va_win.frames.shape[0]}，"
          f"训练 {train_s:.1f} s")
    print(f"val  重建 {va['rec_mse']:.5f}  开环 {va['roll_mse']:.5f}  "
          f"persistence {va['persist_mse']:.5f}  平均帧 {va['mean_mse']:.5f}")
    print(f"val  可见性 AUC {va['vis_auc']:.3f}  persistence {va['vis_auc_persist']:.3f}  "
          f"（正例占比 {va['vis_pos_rate']:.2f}）")
    print(f"判定：{'通过' if passed else '未通过'}  {checks}")
    print(f"产物：{out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
