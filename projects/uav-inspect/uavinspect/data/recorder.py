"""episode 录制与数据集装配。

格式选择照 LeRobot 的思路（视频/数组分离、meta 记统计量），但这里没有视频编码依赖，
所以图像直接存 npz 的 uint8 数组：64×64×3 一帧 12 KB，300 帧一个 episode 约 3.6 MB，
三个 episode 十来兆，够小也够用。真要上规模，把 images 换成 mp4 只是换一个 writer。

每个 episode 落盘：
    obs.npz     images (T,H,W,3) uint8 / pos (T,3) / vel (T,3) / quat (T,4) / action (T,4)
    meta.json   seed / 场景 / 视点 / 覆盖率 / 帧率 / 时长 / 代码指纹

数据集级：
    manifest.json   episode 列表 + 划分（train/val）+ 每集种子
    stats.json      train 划分上的逐通道均值/标准差（归一化用；不跨划分重算）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.images: list[np.ndarray] = []
        self.pos: list[np.ndarray] = []
        self.vel: list[np.ndarray] = []
        self.quat: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.t: list[float] = []
        self.frame_t: list[float] = []
        self.phases: list[int] = []

    def add(self, *, pos, vel, quat, action, t: float, phase: int,
            image: np.ndarray | None = None) -> None:
        self.pos.append(np.asarray(pos, dtype=np.float32))
        self.vel.append(np.asarray(vel, dtype=np.float32))
        self.quat.append(np.asarray(quat, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.t.append(float(t))
        self.phases.append(int(phase))
        if image is not None:
            self.images.append(image)
            self.frame_t.append(float(t))

    def close(self, meta: dict) -> dict:
        meta = dict(meta)
        meta["steps"] = len(self.t)
        meta["frames"] = len(self.images)
        if self.images:
            meta["frame_rate_hz"] = float(
                (len(self.images) - 1) / max(self.frame_t[-1] - self.frame_t[0], 1e-9)
            ) if len(self.frame_t) > 1 else None
        np.savez_compressed(
            self.out / "obs.npz",
            images=np.stack(self.images) if self.images else np.zeros((0, 1, 1, 3), np.uint8),
            pos=np.stack(self.pos), vel=np.stack(self.vel), quat=np.stack(self.quat),
            action=np.stack(self.action), t=np.asarray(self.t, np.float32),
            phase=np.asarray(self.phases, np.int16),
            frame_t=np.asarray(self.frame_t, np.float32),
        )
        (self.out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta


def write_manifest(out_dir: Path, episodes: list[dict], *, val_every: int = 3) -> dict:
    """按种子排序后每 val_every 个取一个进验证集（划分按 episode，不按帧）。"""
    episodes = sorted(episodes, key=lambda e: e["seed"])
    manifest = {"format": "uav-inspect/episodes/v1", "episodes": []}
    for i, ep in enumerate(episodes):
        split = "val" if i % val_every == val_every - 1 else "train"
        manifest["episodes"].append({
            "id": ep["id"], "dir": ep["dir"], "seed": ep["seed"],
            "split": split, "frames": ep.get("frames"), "steps": ep.get("steps"),
            "coverage": ep.get("coverage"),
        })
    (Path(out_dir) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def compute_stats(root: Path, manifest: dict) -> dict:
    """只在 train 划分上算统计量。用 val 算归一化参数是最常见的一种泄漏。"""
    keys = ("pos", "vel", "action")
    acc = {k: [] for k in keys}
    for ep in manifest["episodes"]:
        if ep["split"] != "train":
            continue
        z = np.load(Path(root) / ep["dir"] / "obs.npz")
        for k in keys:
            acc[k].append(np.asarray(z[k], dtype=np.float64))
    stats = {"source_split": "train", "episodes": len(acc["pos"])}
    for k in keys:
        if not acc[k]:
            continue
        x = np.concatenate(acc[k], axis=0)
        stats[k] = {"mean": x.mean(axis=0).round(6).tolist(),
                    "std": x.std(axis=0).round(6).tolist(),
                    "min": x.min(axis=0).round(6).tolist(),
                    "max": x.max(axis=0).round(6).tolist()}
    stats["image_mean"] = None
    return stats
