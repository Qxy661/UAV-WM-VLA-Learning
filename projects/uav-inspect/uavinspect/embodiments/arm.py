"""机械臂具身：只做运动学，不做动力学。

存在的理由只有一个：证明 `uavinspect` 的评测 harness 与 `Agent(obs)->action` 接口
不是为多旋翼量身定做的。所以这里刻意不引第二套仿真器、不引第二个物理引擎，
只用解析 FK + 阻尼最小二乘 IK，把「动作」定义成关节速度指令。

能证明：同一个 harness、同一套指标（成功率、到点误差、用时分位）能跑两种具身。
不能证明：操作能力。没有接触、没有抓取、没有关节力矩与重力补偿——那是另一条线。
"""

from __future__ import annotations

import numpy as np

# 三段连杆：基座在 (0,0,0)，绕 z 转的基础偏航 + 两个俯仰关节
LINK = np.array([0.30, 0.28, 0.12])   # m
Q_MIN = np.array([-2.8, -1.2, -2.4])
Q_MAX = np.array([2.8, 1.6, 2.4])


def fk(q: np.ndarray) -> np.ndarray:
    """关节角 → 末端位置（基座旋转 + 两段俯仰，平面内累加，坐标系同无人机）。"""
    q = np.asarray(q, float)
    yaw, a1, a2 = q[0], q[1], q[2]
    # 平面内两连杆（LINK[0], LINK[1]）加末端小臂 LINK[2] 同向延长
    r = LINK[0] * np.cos(a1) + (LINK[1] + LINK[2]) * np.cos(a1 + a2)
    z = LINK[0] * np.sin(a1) + (LINK[1] + LINK[2]) * np.sin(a1 + a2)
    return np.array([r * np.cos(yaw), r * np.sin(yaw), z])


def jacobian(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    q = np.asarray(q, float)
    j = np.zeros((3, 3))
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        j[:, i] = (fk(q + dq) - fk(q - dq)) / (2 * eps)
    return j


def ik(target: np.ndarray, q0: np.ndarray, *, iters: int = 60, damping: float = 0.05) -> np.ndarray:
    """阻尼最小二乘 IK。失败时就返回最后一步——调用方按误差判定，不猜。"""
    q = np.asarray(q0, float).copy()
    for _ in range(iters):
        e = np.asarray(target, float) - fk(q)
        if np.linalg.norm(e) < 1e-4:
            break
        j = jacobian(q)
        dq = j.T @ np.linalg.solve(j @ j.T + damping ** 2 * np.eye(3), e)
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    return q


def joint_vel_to_target(q: np.ndarray, target: np.ndarray, *, kp: float = 2.0,
                        vmax: float = 1.5) -> np.ndarray:
    """位置误差 → 关节速度指令。这就是机械臂这一侧的「动作」。"""
    e = np.asarray(target, float) - fk(q)
    j = jacobian(q)
    dq = j.T @ np.linalg.solve(j @ j.T + 0.05 ** 2 * np.eye(3), kp * e)
    n = float(np.linalg.norm(dq))
    if n > vmax:
        dq = dq * (vmax / n)
    return dq


class ArmState:
    """和无人机 `State` 对齐的最小字段：pos 是末端位置。"""

    __slots__ = ("q",)

    def __init__(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, float)

    @property
    def pos(self) -> np.ndarray:
        return fk(self.q)


class Arm:
    """一阶关节伺服：dq 指令直接积分，30 Hz 更新。

    没有惯量、没有力矩限幅、没有重力——所以这里得到的收敛时间不能和真机比，
    只能和同一套 harness 下的无人机比「能不能在预算步数内到点」。
    """

    def __init__(self, q0: np.ndarray | None = None, *, dt: float = 1 / 30.0) -> None:
        self.q = np.zeros(3) if q0 is None else np.asarray(q0, float).copy()
        self.dt = dt

    def state(self) -> ArmState:
        return ArmState(self.q)

    def set_joint_vel(self, dq: np.ndarray) -> None:
        self.q = np.clip(self.q + np.asarray(dq, float) * self.dt, Q_MIN, Q_MAX)

    def reset(self, q0: np.ndarray) -> None:
        self.q = np.asarray(q0, float).copy()


def reachable(target: np.ndarray) -> bool:
    """末端到基座的距离是否在连杆总长之内（球壳判据，忽略关节限位）。"""
    return float(np.linalg.norm(target)) <= float(LINK.sum()) + 1e-9
