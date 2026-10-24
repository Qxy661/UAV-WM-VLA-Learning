"""把 M2 录的数据集整理成世界模型能用的窗口。

一条窗口 = 上下文 8 帧 + 动作，预测接下来 4 帧；同时给"目标是否可见"的标签。
标签不是人工标的，是从图像里数红色像素得到的（目标自发光，红色占比 > 0.3% 即算可见）。
这样标签和输入同源，不存在"用几何真值去监督视觉"的泄漏。

划分只用 manifest 里的 train/val，不做二次切分：按 episode 划分已经在 M2 定过了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Windows:
    frames: np.ndarray        # (N, ctx+horizon, 3, 64, 64) float32，未归一化
    actions: np.ndarray       # (N, ctx+horizon, 4) float32
    visible: np.ndarray       # (N, ctx+horizon) float32，0/1
    episode: np.ndarray       # (N,) 来自哪一集，便于按集统计

    def __len__(self) -> int:
        return len(self.frames)


def red_fraction(images: np.ndarray) -> np.ndarray:
    """每帧的红色像素占比。目标自发光，所以这个量与"目标在画面里"高度相关。

    输入是 [0,1] 的 float（load_episodes 里已经除过 255）；阈值也按这个尺度写。
    早先这里写成 0.6*255，标签恒为 0，AUC 于是恒 0.5 —— 这种量纲错不会报错，
    只会让整条指标悄悄失效。
    """
    x = images.astype(np.float32)
    red = (x[..., 0] > 0.6) & (x[..., 0] > 1.8 * x[..., 1]) & (x[..., 0] > 1.8 * x[..., 2])
    return red.reshape(len(x), -1).mean(axis=1)


def load_episodes(root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "val": []}
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    for ep in manifest["episodes"]:
        z = np.load(Path(root) / ep["dir"] / "obs.npz")
        images = z["images"].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        out[ep["split"]].append({
            "frames": images,
            "actions": z["action"].astype(np.float32),
            "visible": (red_fraction(images) > 0.003).astype(np.float32),
        })
    return out


def build_windows(episodes: list[dict], *, ctx: int = 8, horizon: int = 4,
                  stride: int = 2) -> Windows:
    """滑窗。帧与动作按索引对齐（M2 每 5 个控制步拍一帧，动作取当前步）。"""
    fs, ac, vi, ei = [], [], [], []
    for i, ep in enumerate(episodes):
        f, a, v = ep["frames"], ep["actions"], ep["visible"]
        n = min(len(f), len(a), len(v))
        span = ctx + horizon
        for s in range(0, max(0, n - span + 1), stride):
            fs.append(f[s:s + span])
            ac.append(a[s:s + span])
            vi.append(v[s:s + span])
            ei.append(i)
    if not fs:
        raise ValueError("窗口为空：数据太短或 ctx+horizon 太大")
    return Windows(np.stack(fs), np.stack(ac), np.stack(vi), np.asarray(ei))


def normalise(win: Windows, stats: dict | None = None) -> tuple[Windows, dict]:
    """按 train 划分的统计量做逐通道标准化；动作按 train 的均值/方差缩放。"""
    if stats is None:
        x = win.frames.transpose(0, 1, 3, 4, 2).reshape(-1, 3)
        stats = {
            "img_mean": x.mean(axis=0).tolist(),
            "img_std": (x.std(axis=0) + 1e-6).tolist(),
            "act_mean": win.actions.reshape(-1, 4).mean(axis=0).tolist(),
            "act_std": (win.actions.reshape(-1, 4).std(axis=0) + 1e-6).tolist(),
        }
    m = np.asarray(stats["img_mean"], np.float32)[None, None, :, None, None]
    s = np.asarray(stats["img_std"], np.float32)[None, None, :, None, None]
    frames = (win.frames - m) / s
    am = np.asarray(stats["act_mean"], np.float32)[None, None, :]
    asd = np.asarray(stats["act_std"], np.float32)[None, None, :]
    actions = (win.actions - am) / asd
    return Windows(frames.astype(np.float32), actions.astype(np.float32),
                   win.visible, win.episode), stats
