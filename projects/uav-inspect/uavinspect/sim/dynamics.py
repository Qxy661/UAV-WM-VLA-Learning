"""机体中与积分器无关的那一部分：混控、电机滞后、悬停反解。

同一个 `QuadModel` 会被两种后端共用：
- `rigid.py`：自研 RK4 积分 + 冲量式接触（只依赖 numpy）
- `quad.py`：PyBullet 刚体（联网装得上 pybullet 时）

分成两层不是为了好看。M0 的验收数字要能回答"差多少是控制器的问题、差多少是
积分器的问题"，所以两种后端必须跑同一条控制回路、同一套混控、同一个电机滞后；
能不同的只有"力怎么变成加速度"这一段。

不真实的地方写在 README 的 ADR-001：没有桨叶挥摆、地效、涡旋、电池电压衰减。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DroneCfg, SimCfg


@dataclass
class State:
    pos: np.ndarray      # world，m
    vel: np.ndarray      # world，m/s
    quat: np.ndarray     # (w,x,y,z)，body→world
    R: np.ndarray        # body→world
    omega: np.ndarray    # body 角速度，rad/s
    rotor_speed: np.ndarray

    @property
    def tilt(self) -> float:
        """机体 z 轴与 world z 的夹角（rad）。悬停质量、倾角限制都看这个。"""
        return float(np.arccos(np.clip(self.R[2, 2], -1.0, 1.0)))


class QuadModel:
    """X 型四旋翼的混控与电机。0/2 逆时针、1/3 顺时针。"""

    def __init__(self, cfg: DroneCfg, sim: SimCfg) -> None:
        self.cfg = cfg
        self.g = abs(sim.gravity)

        # kf 由悬停转速反解：4·kf·w_hover² = m·g。悬停不是"令推力 = mg"，
        # 而是"把四个转速稳到这个值"。
        self.kf = cfg.mass * self.g / (4.0 * cfg.rotor.w_hover**2)
        self.max_thrust = 4.0 * self.kf * cfg.rotor.w_max**2

        s = cfg.arm / np.sqrt(2.0)
        self.rotor_xy = np.array([[s, -s], [-s, -s], [-s, s], [s, s]])
        self.spin = np.array([1.0, -1.0, 1.0, -1.0])   # +1 逆时针

        # (总推力, roll 力矩, pitch 力矩, yaw 力矩) 与四个电机推力之间的混控矩阵。
        self.A = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                self.rotor_xy[:, 1],
                -self.rotor_xy[:, 0],
                cfg.rotor.torque_ratio * self.spin,
            ]
        )
        self.A_inv = np.linalg.inv(self.A)

        self.w = np.full(4, cfg.rotor.w_hover)
        self.w_des = self.w.copy()

    # ---------- 指令 ----------
    def set_wrench(self, thrust: float, torque) -> None:
        """把期望总推力与机体力矩解混控成期望转速。越界时饱和，不静默缩放。"""
        wrench = np.array([float(thrust), *[float(t) for t in torque]])
        t_each = self.A_inv @ wrench
        t_max = self.kf * self.cfg.rotor.w_max**2
        t_each = np.clip(t_each, 0.0, t_max)
        self.w_des = np.sqrt(t_each / self.kf)

    def mixer_roundtrip(self, thrust: float, torque) -> tuple[float, np.ndarray]:
        """测试用：期望扳手 → 转速 → 扳手，验证混控矩阵与其逆一致。"""
        wrench = np.array([float(thrust), *[float(t) for t in torque]])
        t_each = self.A_inv @ wrench
        w = np.sqrt(t_each / self.kf)
        t_back = self.kf * w**2
        return float(np.sum(t_back)), self.A[1:] @ t_back

    # ---------- 电机 ----------
    def motor_step(self, dt: float) -> None:
        alpha = min(dt / self.cfg.rotor.time_constant, 1.0)
        self.w += (self.w_des - self.w) * alpha
        np.clip(self.w, 0.0, self.cfg.rotor.w_max, out=self.w)

    def rotor_thrust(self) -> np.ndarray:
        return self.kf * self.w**2

    def rotor_wrench(self) -> tuple[float, np.ndarray]:
        t_each = self.rotor_thrust()
        return float(np.sum(t_each)), self.A[1:] @ t_each

    def reset_rotors(self) -> None:
        self.w = np.full(4, self.cfg.rotor.w_hover)
        self.w_des = self.w.copy()

    def inertia_diag(self) -> np.ndarray:
        """把四个电机当质点放在机臂末端（距中心 arm）反解的主惯量。

        motor_x/motor_y 在 (±s, ±s)，s = arm/√2，所以
        Ixx = Iyy = m·s² = m·arm²/2，Izz = 2·m·s² = m·arm²。
        PyBullet 后端不读这个值，它自己的 `getDynamicsInfo` 给什么就用什么；
        两种后端的 Izz/Ixx 都等于 2，量级可比。
        """
        m, a = self.cfg.mass, self.cfg.arm
        return np.array([m * a * a / 2.0, m * a * a / 2.0, m * a * a])
