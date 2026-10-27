"""E7 训练基础设施：离线误差与闭环成功率必须放在一张表里看。

为什么要有这个脚本：BC 的验证 MSE 是 0.0157、对齐后的推力误差 0.027 N，看着像学会了；
同一条策略闭环飞 20 集是 0/20。这两个数字放在不同文件里时，人会挑对自己有利的那个看。

判定规则（写死，不允许事后改）：
- 离线 RMSE < 动作标准差的 10%，且闭环成功率 == 0  → 报 `offline_good_closed_loop_dead`，
  含义是「离线指标不能作为可用性证据」，下一步只能查分布偏移或接口。
- 离线差且闭环差 → `both_bad`，是模型/训练的问题。
- 离线好且闭环好 → `ok`。
- 离线差但闭环好 → `suspicious`，八成是离线评测的划分或对齐有问题。

用法（开发机）：
    python projects/vla-transfer/scripts/e7_consistency.py \
        --ckpt runs/bc_p/ckpt.pt --data /root/work/data/uav96p \
        --closed-loop runs/bc_p_eval/metrics.json --out runs/e7/consistency.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "uav-inspect"))
sys.path.insert(0, str(ROOT / "projects" / "vla-transfer"))

from bc.model import ACTION_DIM, HORIZON, STATE_DIM, BCNet, text_features  # noqa: E402

CHANNELS = ("thrust", "tau_x", "tau_y", "tau_z")


def load_ckpt(path: Path, device: str):
    obj = torch.load(path, map_location=device, weights_only=False)
    horizon = int(obj.get("args", {}).get("horizon", HORIZON))
    model = BCNet(horizon=horizon, action_dim=ACTION_DIM, state_dim=STATE_DIM).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, np.asarray(obj["mean"], np.float32), np.asarray(obj["std"], np.float32), horizon


def offline_rmse(ckpt: Path, data: Path, split: str, limit: int, device: str) -> dict:
    model, mean, std, horizon = load_ckpt(ckpt, device)
    eps = []
    for meta_path in sorted(data.glob("shard_*/episode_*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("split") == split:
            eps.append((meta_path.parent, meta))
        if len(eps) >= limit:
            break
    if not eps:
        raise SystemExit(f"{split} 划分里没有 episode")

    errs: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for ep_dir, meta in eps:
        with np.load(ep_dir / "obs.npz") as z:
            images, pos, vel, quat = z["images"], z["pos"], z["vel"], z["quat"]
            action, t, frame_t = z["action"], z["t"], z["frame_t"]
        text = torch.from_numpy(text_features(str(meta.get("instruction", ""))))[None].to(device)
        # 帧序号对齐到控制步：图像是每 5 个控制步一张，动作是逐控制步
        # 与 bc/data.py 的训练侧对齐方式一致：searchsorted 默认 side='left'
        idx = np.searchsorted(t, frame_t)
        idx = np.clip(idx, 0, len(action) - horizon - 1)
        for k in range(0, len(idx), 3):
            i = int(idx[k])
            img = torch.from_numpy(images[k].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            state = np.concatenate([pos[i], vel[i], quat[i]]).astype(np.float32)
            with torch.no_grad():
                pred = model(img, torch.from_numpy(state)[None].to(device), text)[0].cpu().numpy()
            pred = pred * std + mean
            true = action[i:i + horizon]
            m = min(len(pred), len(true))
            errs.append((pred[:m] - true[:m]).reshape(-1, ACTION_DIM))
            trues.append(true[:m].reshape(-1, ACTION_DIM))

    err = np.concatenate(errs)
    tru = np.concatenate(trues)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    # 分母用 ckpt 里的训练集 std，不用这批 episode 的 std：
    # 悬停段样本的动作 std 只有 0.04 N，拿它当分母会把 0.05 N 的误差放大成 1.2 倍。
    return {
        "split": split, "episodes": len(eps), "samples": int(len(err)),
        "rmse": {c: round(float(v), 6) for c, v in zip(CHANNELS, rmse)},
        "subset_std": {c: round(float(v), 6) for c, v in zip(CHANNELS, tru.std(axis=0))},
        "train_std": {c: round(float(v), 6) for c, v in zip(CHANNELS, std)},
        "normalised_rmse": {c: round(float(r / (s + 1e-9)), 4)
                            for c, r, s in zip(CHANNELS, rmse, std)},
    }


def verdict(off: dict, success: float) -> tuple[str, str]:
    worst = max(off["normalised_rmse"].values())
    if worst < 0.10 and success == 0.0:
        return "offline_good_closed_loop_dead", (
            f"离线最差通道 nRMSE={worst:.3f}（<0.10），闭环成功率 0 —— "
            "离线误差不能当可用性证据，先查观测/动作接口与分布偏移")
    if worst < 0.10 and success > 0.0:
        return "ok", f"离线 nRMSE={worst:.3f}，闭环成功率 {success:.3f}"
    if worst >= 0.10 and success == 0.0:
        return "both_bad", f"离线 nRMSE={worst:.3f} 且闭环 0，先修模型/训练"
    return "suspicious", f"离线 nRMSE={worst:.3f} 但闭环 {success:.3f}，先查离线划分与对齐"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="npz 数据集根（只读 val/test 划分）")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=12, help="最多取多少集算离线误差")
    ap.add_argument("--closed-loop", required=True, help="eval_uav.py 写出的 metrics.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs" / "e7" / "consistency.json"))
    a = ap.parse_args()

    off = offline_rmse(Path(a.ckpt), Path(a.data), a.split, a.limit, a.device)
    closed = json.loads(Path(a.closed_loop).read_text(encoding="utf-8"))
    success = float(closed.get("success_rate", 0.0))
    tag, why = verdict(off, success)

    body = {"ckpt": str(a.ckpt), "data": str(a.data), "offline": off,
            "closed_loop": {"episodes": closed.get("episodes"), "success_rate": success,
                            "wilson95": closed.get("wilson95")},
            "verdict": tag, "reason": why}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"离线（{a.split}, {off['episodes']} 集 / {off['samples']} 样本）")
    for c in CHANNELS:
        print(f"  {c:<7} RMSE {off['rmse'][c]:>8.4f}  本批 std {off['subset_std'][c]:>7.4f}  "
              f"训练 std {off['train_std'][c]:>7.4f}  nRMSE {off['normalised_rmse'][c]:.3f}")
    print(f"闭环 {closed.get('episodes')} 集：成功率 {success:.3f}  区间 {closed.get('wilson95')}")
    print(f"判定：{tag} —— {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
