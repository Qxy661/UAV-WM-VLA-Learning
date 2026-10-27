"""行为克隆基线：96² 图像 + 10 维状态 + 一句中文指令 → 8 步 × 4 维动作块。

这个基线存在的意义不是刷分，而是**分离两种能力**：π0.5 比它高出的部分才是"预训练带来的迁移"，
两者相等说明数据本身就够教，跟预训练无关。所以它必须同数据、同动作块长度、同预算（30k 步）。

三个刻意的简单决定：
- 语言用字符 bigram 哈希到 512 维，不引入 tokenizer 与词表：合成指令的词汇量不到 60 个，
  用 BERT 级编码器只会让"预训练"与"数据"两个变量缠在一起。
- 单帧输入，不做历史：动作块 8 步已经隐含短时记忆，再叠 GRU 会让 BC 与 π0.5 的差距混入结构差异。
- MSE 回归到专家动作，不做离散分箱：专家是确定性控制器，动作分布近似单峰，回归足够；
  多峰问题在 `docs/10-横切判据/04-动作表示选择.md` 里单独讨论。
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

HORIZON = 8
ACTION_DIM = 4
STATE_DIM = 10
TEXT_BUCKETS = 512


def text_features(text: str, buckets: int = TEXT_BUCKETS) -> np.ndarray:
    """字符 bigram 哈希 + L2 归一化。不用词表，测试集的表述也能落到同一个空间里。"""
    v = np.zeros(buckets, dtype=np.float32)
    t = text.strip()
    grams = [t[i:i + 2] for i in range(max(len(t) - 1, 1))]
    for g in grams:
        h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=4).hexdigest(), 16)
        v[h % buckets] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class BCNet(nn.Module):
    def __init__(self, horizon: int = HORIZON, action_dim: int = ACTION_DIM,
                 state_dim: int = STATE_DIM, text_dim: int = TEXT_BUCKETS, h: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Linear(96 * 9, h), nn.SiLU(),
        )
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.SiLU())
        self.text = nn.Sequential(nn.Linear(text_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(h + 64 + 64, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, horizon * action_dim),
        )
        self.horizon, self.action_dim = horizon, action_dim

    def forward(self, image, state, text):
        x = self.enc(image)
        s = self.state(state)
        t = self.text(text)
        return self.head(torch.cat([x, s, t], dim=-1)).view(-1, self.horizon, self.action_dim)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
