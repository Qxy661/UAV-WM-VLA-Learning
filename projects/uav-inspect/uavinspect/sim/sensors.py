"""传感器与画质效应：把「相机不完美」写成可复现的参数。

仿真里最容易被忽略的一层：图像不是干净的 RGB。曝光、噪声、运动模糊、掉帧都会改变
模型看到的东西。这里只做能一句话说清的四种，每一种都能在文档里对上一条物理：

- `gain`      曝光增益（整幅乘一个系数，裁剪到 [0,1]）
- `noise_sigma` 加性高斯噪声（传感器读出噪声）
- `blur_k`    平均池化核（运动模糊的粗近似，k×k 均值）
- `dropout_p` 掉帧（返回上一帧）

设计取舍：不做 ISP、不做镜头畸变、不做卷帘——它们对闭环的影响需要专门的相机模型，
而 M3/M4 的模型分辨率只有 96×96，先保证「噪声变大时该报警」这条链是通的。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorConfig:
    gain: float = 1.0
    noise_sigma: float = 0.0
    blur_k: int = 1
    dropout_p: float = 0.0

    def __post_init__(self) -> None:
        if self.gain <= 0:
            raise ValueError("gain 必须为正")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma 不能为负")
        if self.blur_k < 1 or self.blur_k % 2 == 0:
            raise ValueError("blur_k 必须是正奇数")


def _blur(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return img
    pad = k // 2
    h, w, c = img.shape
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy + h, dx:dx + w, :]
    return out / (k * k)


def apply(img: np.ndarray, cfg: SensorConfig, rng: np.random.Generator,
          prev: np.ndarray | None = None) -> np.ndarray:
    """对一帧 (H,W,3) 图像施加效应，返回同形状、同 dtype 的图像。

    `dropout_p > 0` 时需要传 `prev`：掉帧就复用上一帧，而不是返回黑图——真实相机的
    丢帧是"卡住"，不是"变黑"。
    """
    x = img.astype(np.float32) / 255.0
    if cfg.dropout_p > 0 and prev is not None and rng.random() < cfg.dropout_p:
        return prev.copy()
    if cfg.blur_k > 1:
        x = _blur(x, cfg.blur_k)
    x = x * cfg.gain
    if cfg.noise_sigma > 0:
        x = x + rng.normal(0.0, cfg.noise_sigma, size=x.shape).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def apply_sequence(images: np.ndarray, cfg: SensorConfig, seed: int = 0) -> np.ndarray:
    """对一整段 (T,H,W,3) 施加效应：同一段用同一条随机流，保证可复现。"""
    rng = np.random.default_rng(seed)
    out = np.empty_like(images)
    prev = None
    for i, img in enumerate(images):
        out[i] = apply(img, cfg, rng, prev)
        prev = out[i]
    return out
