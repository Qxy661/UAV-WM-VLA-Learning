"""输入分布漂移：PSI（population stability index）与阈值。

为什么用 PSI 而不是 KL：PSI 对「分箱后比例变化」线性敏感、有通用阈值（<0.1 稳定、
0.1–0.25 关注、>0.25 报警），而且不需要平滑——KL 在某个箱子里出现 0 就发散，工程上
得手动加 eps，反而不好解释。

用在哪：换仓库布局、换相机曝光、换目标颜色，都会先在**输入**上反映出来。等到闭环
指标掉了才反应就晚了——那时已经在用旧模型飞新场景了。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def psi(ref: np.ndarray, cur: np.ndarray, *, bins: int = 10, eps: float = 1e-4) -> float:
    """两个样本的一维 PSI：按 ref 的分位切箱，比较两侧落在各箱的比例。"""
    ref = np.asarray(ref, dtype=float).ravel()
    cur = np.asarray(cur, dtype=float).ravel()
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if qs.size < 2:
        return 0.0
    r_hist, _ = np.histogram(ref, bins=qs)
    c_hist, _ = np.histogram(cur, bins=qs)
    r = np.clip(r_hist / max(r_hist.sum(), 1), eps, None)
    c = np.clip(c_hist / max(c_hist.sum(), 1), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def verdict(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "watch"
    return "alert"


def image_brightness(images: np.ndarray) -> np.ndarray:
    """每帧的平均亮度（0–1）。用它当输入分布的代表量：曝光、场景亮度、雾都会动它。"""
    a = np.asarray(images, dtype=np.float32)
    if a.ndim == 4 and a.shape[-1] == 3:
        return a.mean(axis=(1, 2, 3)) / (255.0 if a.max() > 1.5 else 1.0)
    raise ValueError(f"期望 (N,H,W,3) 的图像数组，拿到 {a.shape}")


def reference_from_episodes(root: str | Path, *, limit: int = 40) -> np.ndarray:
    """从数据根目录里取一批帧亮度当参考分布。"""
    root = Path(root)
    out = []
    for ep in sorted(root.glob("shard_*/episode_*/obs.npz"))[:limit]:
        with np.load(ep) as z:
            out.append(image_brightness(z["images"]))
    if not out:
        raise FileNotFoundError(f"{root} 下没找到 obs.npz")
    return np.concatenate(out)


def check(ref: np.ndarray, cur: np.ndarray) -> dict:
    v = psi(ref, cur)
    return {"psi": round(v, 4), "verdict": verdict(v), "bins": 10,
            "ref_mean": round(float(np.mean(ref)), 4), "cur_mean": round(float(np.mean(cur)), 4),
            "thresholds": {"watch": 0.10, "alert": 0.25}}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="输入分布漂移（亮度 PSI）")
    ap.add_argument("--ref", required=True, help="参考数据根（含 shard_*/episode_*/obs.npz）")
    ap.add_argument("--cur", required=True, help="待检数据根")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = check(reference_from_episodes(a.ref), reference_from_episodes(a.cur))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
