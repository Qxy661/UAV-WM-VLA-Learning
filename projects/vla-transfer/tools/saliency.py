"""遮挡敏感度：模型到底在看画面的哪一块。

做法是单点遮挡（occlusion sensitivity）：把图像切成 N×N 块，逐块用灰度均值填掉，
看 3 维子目标预测位移多少米。位移越大说明这块越重要。同时统计每块的红色像素占比，
给出「敏感度与红色目标位置的相关性」——如果模型真的在读那个红点，两者应该正相关。

    python projects/vla-transfer/tools/saliency.py --ckpt runs/sg_rel2/ckpt.pt \
        --data /root/work/data/uav96c --episodes 8 --grid 6 --out runs/saliency_sg
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bc.model import STATE_DIM, BCNet, text_features  # noqa: E402


def load_model(ckpt: Path, device: str):
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    args = obj.get("args", {})
    model = BCNet(horizon=1, action_dim=3, state_dim=STATE_DIM).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, float(args.get("goal_scale", 2.0))


def red_mask(img: np.ndarray) -> np.ndarray:
    """img: (3,H,W) 0–1"""
    r, g, b = img[0], img[1], img[2]
    return (r > 0.6) & (r > 1.8 * g) & (r > 1.8 * b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, scale = load_model(Path(a.ckpt), a.device)

    metas = sorted(Path(a.data).glob("shard_*/episode_*/meta.json"))
    val = [m for m in metas
           if json.loads(m.read_text(encoding="utf-8")).get("split") == "val"][:a.episodes]
    if not val:
        raise SystemExit("没有 val episode")

    H = W = None
    heat = None
    red = None
    used = 0
    for m in val:
        z = np.load(m.parent / "obs.npz")
        imgs = z["images"]
        if len(imgs) == 0:
            continue
        inst = json.loads(m.read_text(encoding="utf-8"))["instruction"]
        tx = torch.from_numpy(text_features(inst))[None].to(a.device)
        # 取该集最后几帧（接近目标时画面里才有红点）
        pick = [int(len(imgs) * f) for f in (0.5, 0.75, 0.95)]
        for k in pick:
            frame = imgs[k].astype(np.float32) / 255.0            # (H,W,3)
            img = torch.from_numpy(frame).permute(2, 0, 1)[None].to(a.device)
            # 状态用该帧对应的 pos/vel/quat
            ft = float(z["frame_t"][k])
            idx = int(np.argmin(np.abs(z["t"] - ft)))
            st = np.concatenate([z["pos"][idx], z["vel"][idx], z["quat"][idx]]).astype(np.float32)
            st = torch.from_numpy(st)[None].to(a.device)
            with torch.no_grad():
                base = model(img, st, tx)[0, 0].cpu().numpy() * scale

            if H is None:
                H, W = frame.shape[0], frame.shape[1]
                heat = np.zeros((a.grid, a.grid), np.float64)
                red = np.zeros((a.grid, a.grid), np.float64)
            rm = red_mask(torch.from_numpy(frame).permute(2, 0, 1).numpy())
            bh, bw = H // a.grid, W // a.grid
            for gi in range(a.grid):
                for gj in range(a.grid):
                    occ = frame.copy()
                    occ[gi * bh:(gi + 1) * bh, gj * bw:(gj + 1) * bw] = frame.mean(axis=(0, 1))
                    oi = torch.from_numpy(occ).permute(2, 0, 1)[None].to(a.device)
                    with torch.no_grad():
                        pred = model(oi, st, tx)[0, 0].cpu().numpy() * scale
                    heat[gi, gj] += float(np.linalg.norm(pred - base))
                    red[gi, gj] += float(rm[gi * bh:(gi + 1) * bh, gj * bw:(gj + 1) * bw].mean())
            used += 1

    heat /= max(used, 1)
    red /= max(used, 1)

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        x, y = x.ravel(), y.ravel()
        if x.std() < 1e-9 or y.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    summary = {
        "frames_used": used,
        "grid": a.grid,
        "sensitivity_mean_cm": round(float(heat.mean()) * 100, 2),
        "sensitivity_max_cm": round(float(heat.max()) * 100, 2),
        "sensitivity_p90_cm": round(float(np.percentile(heat, 90)) * 100, 2),
        "red_mass_total": round(float(red.sum()), 4),
        "corr_heat_red": round(corr(heat, red), 3) if math.isfinite(corr(heat, red)) else None,
        "heat": [[round(float(v) * 100, 2) for v in row] for row in heat],
        "red": [[round(float(v), 3) for v in row] for row in red],
    }
    (out / "saliency.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    # 直接把热力图画成 PNG（不依赖 matplotlib，手写 PPM→PNG 太麻烦，用 PGM 转）
    try:
        from PIL import Image  # noqa: F401

        img_arr = (heat / max(heat.max(), 1e-9) * 255).astype(np.uint8)
        Image.fromarray(img_arr).resize((W, H), Image.NEAREST).save(out / "heat.png")
    except Exception:
        np.savetxt(out / "heat.csv", heat, delimiter=",")

    print(json.dumps({k: summary[k] for k in
                      ("frames_used", "sensitivity_mean_cm", "sensitivity_max_cm",
                       "sensitivity_p90_cm", "corr_heat_red")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
