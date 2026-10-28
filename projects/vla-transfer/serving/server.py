"""推理服务的最小可用实现：请求队列 + 动态批处理 + 每请求 deadline。

为什么要有这一层：策略在 10 Hz 上跑，而单次前向在 CPU 上要几十毫秒，四架机器各发各的
会让 GPU/CPU 空转在 batch=1 上。动态批处理把「同一小段时间窗内的请求」合成一次前向，
代价是排队延迟；deadline 则规定「宁可沿用上一块动作，也不返回一个已经过期的结果」。

    python projects/vla-transfer/scripts/e8_bench.py --ckpt runs/sg_rel2/ckpt.pt \
        --data /root/work/data/uav96c --arrival-hz 30 --seconds 10
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Request:
    rid: int
    image: torch.Tensor      # (3,H,W) 0-1
    state: torch.Tensor      # (10,)
    text: torch.Tensor       # (TEXT_BUCKETS,)
    deadline_ms: float = 40.0
    t_arrive: float = field(default_factory=time.perf_counter)


@dataclass
class Reply:
    rid: int
    goal: np.ndarray | None      # 3 维相对位移（米）；None 表示沿用上一块
    stale: bool
    lat_ms: float


class Server:
    """单线程事件循环式的服务：主循环里攒批、跑模型、回填结果。"""

    def __init__(self, model: torch.nn.Module, *, scale: float = 2.0,
                 max_batch: int = 8, max_wait_ms: float = 2.0,
                 device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.scale = scale
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.device = device
        self.queue: deque[Request] = deque()
        self.last_goal: dict[int, np.ndarray] = {}
        self.n_stale = 0
        self.n_served = 0
        self.batches = 0

    def submit(self, req: Request) -> None:
        self.queue.append(req)

    def _run(self, batch: list[Request]) -> list[np.ndarray]:
        imgs = torch.stack([r.image for r in batch]).to(self.device)
        sts = torch.stack([r.state for r in batch]).to(self.device)
        txs = torch.stack([r.text for r in batch]).to(self.device)
        with torch.no_grad():
            out = self.model(imgs, sts, txs)[:, 0].cpu().numpy() * self.scale
        return [out[i] for i in range(len(batch))]

    def step(self) -> list[Reply]:
        """跑一个服务周期：攒够一批（或队首等到 max_wait）才发车。"""
        if not self.queue:
            return []
        now = time.perf_counter()
        head = self.queue[0]
        # 没攒满、队首也还没到等待上限，就先不发车——这是批处理收益的来源
        if len(self.queue) < self.max_batch and (now - head.t_arrive) < self.max_wait:
            return []
        n = min(self.max_batch, len(self.queue))
        batch = [self.queue.popleft() for _ in range(n)]
        self.batches += 1
        goals = self._run(batch)
        replies = []
        now = time.perf_counter()
        for req, g in zip(batch, goals):
            lat = (now - req.t_arrive) * 1000.0
            stale = lat > req.deadline_ms
            if stale:
                self.n_stale += 1
                g = self.last_goal.get(req.rid)
            if g is not None:
                self.last_goal[req.rid] = g
            self.n_served += 1
            replies.append(Reply(rid=req.rid, goal=g, stale=stale, lat_ms=lat))
        return replies
