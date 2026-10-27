"""闭环接 π0.5 策略服务。

三个必须对齐的点，错一个就静默退化：
1. 观测键与训练时 repack 完全一致（observation/image、observation/state、prompt）；
2. 图像先 resize_with_pad 到 224 再 uint8，训练侧 LeRobot 存的是 96²，pad 由这里补；
3. 拿回的是 10 步动作块，按控制周期逐条弹出，块空才重算——不是每个控制步都请求，
   服务端 p50 约 78 ms，48 Hz 控制环等不起。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class _Buf:
    def __init__(self) -> None:
        self.items: list[np.ndarray] = []

    def __bool__(self) -> bool:
        return bool(self.items)

    def push(self, arr: np.ndarray) -> None:
        self.items = [np.asarray(a, dtype=np.float64) for a in arr]

    def pop(self) -> np.ndarray:
        return self.items.pop(0)


class Pi05Policy:
    """把策略服务包装成 inspect.run_episode 认的 policy(ctx) -> (thrust, torque)。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, instruction: str = "",
                 *, resize: int = 224, state_dim: int = 10, action_dim: int = 4,
                 horizon: int = 10, hover_thrust: float = 9.81) -> None:
        from openpi_client import image_tools  # 延迟导入：不接服务时不依赖
        from openpi_client import websocket_client_policy

        self._image_tools = image_tools
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.instruction = instruction
        self.resize = resize
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.hover_thrust = hover_thrust
        self.buf = _Buf()
        self.replans = 0
        self.latencies_ms: list[float] = []

    def _observation(self, image: np.ndarray, state: np.ndarray) -> dict[str, Any]:
        img = self._image_tools.resize_with_pad(image, self.resize, self.resize)
        img = self._image_tools.convert_to_uint8(img)
        st = np.zeros(self.state_dim, dtype=np.float32)
        st[: min(self.state_dim, state.shape[0])] = state[: self.state_dim]
        return {"observation/image": img, "observation/state": st, "prompt": self.instruction}

    def infer_vector(self, ctx: dict) -> np.ndarray | None:
        """返回这一拍的动作向量（不含任何动作语义解释）。

        默认 4 维（推力+三轴力矩）；``action_dim=3`` 时是相对位移子目标（米）。
        ctx 里还没有图像时返回 None，由调用方决定怎么兜底。
        """
        if not self.buf:
            image = ctx.get("image")
            if image is None:                      # 还没到拍摄周期
                return None
            st = ctx["st"]
            state = np.concatenate([st.pos, st.vel, st.quat]).astype(np.float32)
            obs = self._observation(np.asarray(image), state)
            result = self.client.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] < self.action_dim:
                raise ValueError(f"策略返回形状 {actions.shape}，期望 (T,>={self.action_dim})")
            self.buf.push(actions[: self.horizon, : self.action_dim])
            self.replans += 1
            if "server_timing" in result:
                self.latencies_ms.append(float(result["server_timing"]["infer_ms"]))
        return self.buf.pop()

    def __call__(self, ctx: dict) -> tuple[float, np.ndarray]:
        a = self.infer_vector(ctx)
        if a is None:
            return self.hover_thrust, np.zeros(3)
        return float(a[0]), np.asarray(a[1:4], dtype=float)

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {}
        v = np.asarray(self.latencies_ms, dtype=np.float64)
        return {"n": int(v.size), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
                "max": float(v.max())}
