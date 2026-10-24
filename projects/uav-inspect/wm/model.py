"""一个能跑通的小世界模型：卷积编码 → GRU 潜动力学 → 两路解码。

刻意做小：64×64、潜维 128、GRU 单层。M3 要的不是 SOTA，是"同口径下比无模型基线好，
且失败模式可解释"。所以只保留三个必须的结构件：

1. 动作进动力学（不少"世界模型"其实没接动作，于是只能复述历史）
2. 潜状态跨步递推（rollout 时不喂真帧，否则测的是单步纠偏）
3. 目标可见性作为单独头（这是 M2 场景的语义量，比像素 MSE 更贴近"遮挡补全"）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(_conv(3, 32), _conv(32, 64), _conv(64, 96), nn.Flatten(),
                                 nn.Linear(96 * 8 * 8, dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """潜态 → 64×64 图像。三级上采样，每级 2 倍：8 → 16 → 32 → 64。"""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(dim, 96 * 8 * 8)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(96, 96, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(96, 64, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 96, 8, 8)
        return torch.tanh(self.net(h))


class WorldModel(nn.Module):
    def __init__(self, dim: int = 128, act_dim: int = 4):
        super().__init__()
        self.dim = dim
        self.enc = Encoder(dim)
        self.act_emb = nn.Linear(act_dim, dim)
        self.rnn = nn.GRUCell(dim, dim)
        self.dec = Decoder(dim)
        self.vis = nn.Linear(dim, 1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        n, t = frames.shape[:2]
        z = self.enc(frames.reshape(n * t, *frames.shape[2:]))
        return z.view(n, t, self.dim)

    def step(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """一步潜动力学：动作先嵌到潜空间再进 GRU。"""
        return self.rnn(self.act_emb(a), h)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor, ctx: int,
                base: torch.Tensor | None = None):
        """返回 (重建潜态, 递推潜态, 递推帧, 可见性 logits)。

        重建分支看到真帧（teacher forcing），递推分支只看动作——两者分开算损失，
        这样"开环会漂"这件事是量出来的，不是被教师强制掩盖掉的。
        base 给定时，解码器只预测相对最后一帧的残差：冻结上一帧本身是免费且很强的基线，
        让模型从零重建整帧，指标上几乎不可能赢过它。
        """
        z = self.encode(frames)
        h = z[:, 0]
        rec = [h]
        for t in range(1, ctx):
            h = self.step(h, actions[:, t - 1])
            rec.append(h)
        roll = []
        total = actions.shape[1]          # 递推步数由动作序列长度决定，不是帧长度
        for t in range(ctx, total):
            h = self.step(h, actions[:, t - 1])
            roll.append(h)
        rec_h = torch.stack(rec, 1)
        roll_h = torch.stack(roll, 1)
        dec = self.dec(roll_h.reshape(-1, self.dim)).view(
            frames.shape[0], -1, *frames.shape[2:])
        if base is not None:
            dec = dec + base[:, None]
        return (rec_h, roll_h, dec,
                self.vis(torch.cat([rec_h, roll_h], 1)).squeeze(-1))
