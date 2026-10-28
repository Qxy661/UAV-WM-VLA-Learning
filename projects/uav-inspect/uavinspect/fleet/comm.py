"""通信链路抽象：丢包、延迟、固定种子。

只做一件事——把「这条消息到底到没到」变成一个可复现的布尔值。真机的
通信（DDS / WiFi / 4G）比这复杂得多，但分配算法对通信的依赖只有两点：
消息会丢、消息会晚到。把它显式建模，就能测出「丢多少包会让分配退化」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Link:
    loss: float = 0.0          # 单条消息丢失概率
    delay_s: float = 0.0       # 单向延迟（只用于时间估计）
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def send(self) -> bool:
        """发一条消息，返回是否送达。"""
        if self.loss <= 0.0:
            return True
        return bool(self._rng.random() >= self.loss)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
