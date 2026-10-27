"""把「期望角速率」动作翻译成力矩，用于 omega 动作空间实验。

为什么需要这层：动作空间实验里策略输出 `[thrust, omega_ref]`，而机体接口吃
`[thrust, torque]`。这个翻译必须和专家数据里的低层速率环**完全一致**，否则比较的是
两套控制器，而不是两套动作表示。

    torque = I · (rate_kp · (omega_ref − omega)) + omega × I·omega

`I` 与 `rate_kp` 都与 `uavinspect.control.cascaded` 同源：惯量取 QuadModel 的解析值
（四个电机当质点，Ixx=Iyy=m·arm²/2、Izz=m·arm²），增益取同一份配置。
"""

from __future__ import annotations

import numpy as np

from uavinspect.sim.dynamics import QuadModel


class OmegaToTorque:
    def __init__(self, cfg):
        self.inertia = QuadModel(cfg.drone, cfg.sim).inertia_diag()
        self.rate_kp = float(cfg.control.rate_kp)

    def __call__(self, omega_ref, omega) -> np.ndarray:
        """omega_ref 是 3 维期望角速率，返回 3 维力矩。"""
        wr = np.asarray(omega_ref, dtype=float).reshape(3)
        w = np.asarray(omega, dtype=float).reshape(3)
        return self.inertia * (self.rate_kp * (wr - w)) + np.cross(w, self.inertia * w)
