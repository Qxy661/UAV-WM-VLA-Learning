"""子目标接口的行为克隆：96² 图 + 10 维状态 + 一句中文指令 → 目标标记的世界坐标。

为什么单开一个脚本而不改 `train_bc.py`：标签不再是动作，而是**空间位置**，量纲是米；
损失、归一化、评测口径全都不一样（这里报的是定位误差 cm，不是力矩 MSE）。

标签来自 `meta.json` 的 `target_pos`（生成数据时由 `pipeline/inspect.py` 写入）。
目标在货架面内抖动 dx∈[-0.40,0.40]、dz∈[-0.20,0.20]，所以位置不能由「几号货架 + 哪一侧」
推出来——必须看图。这是把视觉变成必要条件的那个改动。

用法（开发机）：
    python projects/vla-transfer/scripts/train_subgoal.py \
        --data /root/work/data/uav96c --epochs 8 --batch 256 --device cuda \
        --out runs/subgoal_c
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "vla-transfer"))

from bc.model import BCNet, n_params, text_features  # noqa: E402

STATE_DIM = 10
GOAL_DIM = 3


def state_of(z: dict, k: int) -> np.ndarray:
    return np.concatenate([z["pos"][k], z["vel"][k], z["quat"][k]]).astype(np.float32)


class SubgoalFrames(Dataset):
    """把每集按帧展开成 (图, 状态, 指令, 目标) 四元组。

    画面是 9.6 Hz、状态是 48 Hz，必须用 `frame_t` 去 `t` 里找对应的控制步，
    不能按长度硬取——两者的行数不一样。
    """

    def __init__(self, root: Path, split: str, *, stride: int = 4, max_episodes: int = 0,
                 require_coverage: float = 0.0) -> None:
        self.rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        metas = sorted(Path(root).glob("shard_*/episode_*/meta.json"))
        if max_episodes:
            metas = metas[:max_episodes]
        used = 0
        for mp in metas:
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if m.get("split") != split or "target_pos" not in m:
                continue
            if float(m.get("coverage", 0.0)) < require_coverage:
                continue
            npz = mp.parent / "obs.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            images, frame_t, t = z["images"], z["frame_t"], z["t"]
            goal = np.asarray(m["target_pos"], dtype=np.float32)
            text = text_features(m.get("instruction", ""))
            for i in range(0, len(frame_t), stride):
                k = int(np.searchsorted(t, frame_t[i]))
                k = min(k, len(t) - 1)
                # 标签用**相对位移**而不是世界坐标：世界坐标要先从画面里认出"这是几号货架"
                # 才能给出绝对位置，等于把指令已经给出的信息再猜一遍；相对位移只问
                # "目标在我前方多少米"，与站位无关，泛化到没见过的货架也成立。
                rel = goal - np.asarray(z["pos"][k], dtype=np.float32)
                self.rows.append((images[i], state_of(z, k), text, rel))
            used += 1
        self.episodes = used

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img, st, tx, goal = self.rows[i]
        img = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        return img, torch.from_numpy(st), torch.from_numpy(tx), torch.from_numpy(goal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--require-coverage", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "runs" / "subgoal"))
    a = ap.parse_args()

    t0 = time.time()
    tr = SubgoalFrames(Path(a.data), "train", stride=a.stride, max_episodes=a.max_episodes,
                       require_coverage=a.require_coverage)
    va = SubgoalFrames(Path(a.data), "val", stride=max(a.stride, 8))
    if not len(tr) or not len(va):
        raise SystemExit(f"数据为空：train {len(tr)} / val {len(va)}")
    goals = np.stack([r[3] for r in tr.rows])
    # 相对位移模式下不用数据方差：靠近时位移本来就接近 0，用方差归一化会让
    # 远处样本主导损失；固定 2 m 的物理尺度让 1 cm 的误差在任何站位都同样重要。
    mu = np.zeros(3, dtype=np.float32)
    sd = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    print(json.dumps({"train_frames": len(tr), "val_frames": len(va), "train_episodes": tr.episodes,
                      "val_episodes": va.episodes, "goal_mean": mu.tolist(), "goal_std": sd.tolist(),
                      "rel_abs_mean_m": float(np.abs(goals).mean())}))

    # 放到 device 上一次，别在每次前向里 from_numpy（那会留在 CPU 上，
    # 混着 cuda 张量报 "found at least two devices"）。
    mu_t = torch.from_numpy(mu).to(a.device)
    sd_t = torch.from_numpy(sd).to(a.device)

    def norm(g: torch.Tensor) -> torch.Tensor:
        return (g - mu_t) / sd_t

    model = BCNet(horizon=1, action_dim=GOAL_DIM, state_dim=STATE_DIM).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    def run(dl, train: bool) -> tuple[float, float, list[float]]:
        tot_n, tot_m = 0.0, 0.0
        tot_a = np.zeros(3)
        for img, st, tx, goal in dl:
            img, st = img.to(a.device), st.to(a.device)
            tn, g = tx.to(a.device), goal.to(a.device)
            with torch.set_grad_enabled(train):
                pred = model(img, st, tn)[:, 0, :]
                loss = lossf(pred, norm(g))
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            # 还原成米，报的是定位误差而不是归一化 MSE；再按三个轴拆开——
            # 判据只看横向（|u|≈dx/0.805），沿航向差几厘米不要紧，横向差几厘米就废。
            delta = (pred.detach() * sd_t + mu_t) - g
            err = delta.norm(dim=-1)
            tot_n += float(loss.detach()) * len(g)
            tot_m += float(err.mean()) * len(g)
            tot_a += delta.abs().sum(0).cpu().numpy()
        n = len(dl.dataset)
        return tot_n / n, tot_m / n, (tot_a / n * 100).tolist()

    best = 1e9
    for ep in range(1, a.epochs + 1):
        trl, trm, tra = run(dl_tr, True)
        val, vam, vaa = run(dl_va, False)
        print(json.dumps({"epoch": ep, "train_mse_norm": round(trl, 5),
                          "train_err_cm": round(trm * 100, 2),
                          "val_mse_norm": round(val, 5), "val_err_cm": round(vam * 100, 2),
                          "val_axis_cm": [round(x, 2) for x in vaa],
                          "elapsed_s": round(time.time() - t0, 1)}))
        if vam < best:
            best = vam
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "goal_mean": mu, "goal_std": sd,
                        "args": vars(a), "val_err_cm": vam}, out / "ckpt.pt")
    print(json.dumps({"params": n_params(model), "best_val_err_cm": round(best * 100, 2),
                      "wall_time_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
