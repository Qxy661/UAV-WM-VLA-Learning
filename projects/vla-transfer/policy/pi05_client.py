"""把 uav-inspect 的观测喂给 openpi 策略服务器，再把动作块接回来。

设计约束（写在 PLAN.md §4 的 Embodiment 接口里）：
- 观测只走一个相机键 `observation/image`，状态一个向量 `observation/state`，指令走 `prompt`
- 动作块形状 (horizon, action_dim)，由服务端的训练配置决定，不在客户端假设
- 图像缩放用 openpi_client 自带的 `resize_with_pad`，与官方 LIBERO 客户端同一份代码

P0 阶段先拿 pi05_libero 验证管道（7 维动作）；无人机微调后换成自定义配置（4 维）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:  # 只在开发机上装了 openpi-client
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _ws
except ImportError:  # pragma: no cover - 本机没有这个包
    image_tools = None
    _ws = None


@dataclass
class RemotePolicyConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    resize: int = 224
    action_dim: int = 4
    horizon: int = 10
    include_wrist: bool = False   # pi05_libero 需要第二路腕部相机；无人机只有一路，所以默认关


@dataclass
class RemotePolicy:
    """一个最小的远程策略客户端，接口与 `uavinspect.eval.policies` 里的策略一致。"""

    cfg: RemotePolicyConfig = field(default_factory=RemotePolicyConfig)
    _client: object | None = None
    latencies_ms: list[float] = field(default_factory=list)

    def connect(self) -> "RemotePolicy":
        if _ws is None:
            raise RuntimeError("没有 openpi_client，请在开发机的 pi0 环境里运行")
        self._client = _ws.WebsocketClientPolicy(self.cfg.host, self.cfg.port)
        return self

    # ------------------------------------------------------------------ 观测
    def build_observation(self, image: np.ndarray, state: np.ndarray, prompt: str) -> dict:
        """image: HxWx3 uint8；state: 一维浮点向量；prompt: 指令文本。"""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        # 与官方客户端一致：先转 uint8，再等比缩放并填充到正方形
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, self.cfg.resize, self.cfg.resize))
        obs = {
            "observation/image": img,
            "observation/state": np.asarray(state, dtype=np.float32).ravel(),
            "prompt": prompt,
        }
        if self.cfg.include_wrist:
            obs["observation/wrist_image"] = img   # 单相机载体用它占位，只为跑通官方配置
        return obs

    # ------------------------------------------------------------------ 推理
    def infer(self, image: np.ndarray, state: np.ndarray, prompt: str) -> np.ndarray:
        assert self._client is not None, "先 connect()"
        obs = self.build_observation(image, state, prompt)
        t0 = time.perf_counter()
        out = self._client.infer(obs)
        self.latencies_ms.append((time.perf_counter() - t0) * 1e3)
        actions = np.asarray(out["actions"], dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"动作块应当是二维 (horizon, action_dim)，拿到 {actions.shape}")
        return actions

    def latency_summary(self) -> dict:
        if not self.latencies_ms:
            return {}
        xs = sorted(self.latencies_ms)
        pick = lambda q: xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]  # noqa: E731
        return {"n": len(xs), "p50_ms": round(pick(0.5), 1), "p90_ms": round(pick(0.9), 1),
                "p99_ms": round(pick(0.99), 1), "max_ms": round(xs[-1], 1)}

    # ------------------------------------------------------------------ 闭环
    def act(self, state, t=None) -> np.ndarray:
        """给 `uavinspect.eval.harness` 用的单步接口：这里只返回动作块的第一行动作。

        真正的分块执行在 `ChunkedExecutor` 里（见下），因为"每步都重新推理"会把
        P99 延迟放大到控制周期里，M7 实测 75 ms 时 48 Hz 的控制环会被拖垮。
        """
        raise NotImplementedError("用 ChunkedExecutor 包一层再交给 harness")


class ChunkedExecutor:
    """把动作块按固定步数执行，块用完再推理。这就是 ADR-003 里"分块 + 插值跟踪"的实现。"""

    def __init__(self, policy: RemotePolicy, replan_steps: int = 5):
        self.policy = policy
        self.replan_steps = replan_steps
        self._plan: list[np.ndarray] = []
        self.plan_age_ms: list[float] = []

    def reset(self) -> None:
        self._plan = []

    def act(self, image, state, prompt, dt: float) -> np.ndarray:
        if not self._plan:
            chunk = self.policy.infer(image, state, prompt)
            self._plan = [row for row in chunk[: self.replan_steps]]
            self.plan_age_ms.append(len(chunk) * dt * 1e3)
        return self._plan.pop(0)
