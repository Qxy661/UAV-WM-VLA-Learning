"""只依赖 numpy 的刚体积分后端。

为什么要有它：M0 的验收要能在任何一台只装了 numpy 的机器上重跑。PyBullet 在
某些 Python 版本上没有 Windows wheel（本机 3.11 就是），把"M0 跑不通"归因到
装包问题上，会把仿真的可信度和环境的可用性混成一件事。

积分用定步长 RK4，状态是 13 维：位置(3) + 速度(3) + 四元数(4) + 机体系角速度(3)。
四元数每步归一化；接触用"位置投影 + 法向冲量"处理，不是弹簧阻尼——
刚度惩罚力会让 RK4 在 240 Hz 下发散，而冲量式接触至少不会自己产生能量。

已知不真实（与 PyBullet 后端共有的差异写在 README 的 ADR-001，这里只列本后端特有的）：
- 接触只作用在质心（球体近似），没有翻滚力矩，撞墙后姿态不会被打歪。
- 没有摩擦锥，切向只做线性阻尼，所以"贴墙滑行"会比真实情况顺滑。
- 没有关节/柔性，机体之间是刚性球。
这些不影响 M0/M1 的"到不到、撞不撞、延迟多少"，影响"撞了以后怎么翻"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import DroneCfg, SimCfg
from ..util.mathx import quat_to_rot
from .dynamics import QuadModel, State

RESTITUTION = 0.2
FRICTION = 0.6
GROUND_EPS = 1e-4
STOP_SPEED = 0.05          # 低于这个法向速度就按静止处理，避免接触抖动


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass
class Box:
    half: np.ndarray
    center: np.ndarray
    rgba: tuple = (0.8, 0.5, 0.2, 1.0)
    index: int = 0


class RigidQuad(QuadModel):
    """一架四旋翼 + 自己的 13 维状态。"""

    def __init__(
        self,
        cfg: DroneCfg,
        sim: SimCfg,
        position=(0.0, 0.0, 0.2),
        yaw: float = 0.0,
        name: str = "uav",
    ) -> None:
        super().__init__(cfg, sim)
        self.name = name
        self.g_vec = np.array([0.0, 0.0, -self.g])
        self.inertia = self.inertia_diag()
        self.inv_inertia = 1.0 / self.inertia
        self.pos = np.asarray(position, dtype=float).copy()
        self.vel = np.zeros(3)
        self.quat = quat_from_yaw(yaw)
        self.omega = np.zeros(3)
        self.reset_rotors()

    # ---------- 状态 ----------
    def state(self) -> State:
        R = quat_to_rot(self.quat)
        return State(
            pos=self.pos.copy(), vel=self.vel.copy(), quat=self.quat.copy(),
            R=R, omega=self.omega.copy(), rotor_speed=self.w.copy(),
        )

    def reset(self, position, yaw: float = 0.0) -> None:
        self.pos = np.asarray(position, dtype=float).copy()
        self.vel = np.zeros(3)
        self.quat = quat_from_yaw(yaw)
        self.omega = np.zeros(3)
        self.reset_rotors()

    # ---------- 积分 ----------
    def _deriv(self, y: np.ndarray) -> np.ndarray:
        pos, vel, quat, omega = y[:3], y[3:6], y[6:10], y[10:13]
        R = quat_to_rot(quat)
        fz, tau = self.rotor_wrench()          # 一个子步内转速不变
        acc = (R @ np.array([0.0, 0.0, fz]) + self.g_vec * self.cfg.mass) / self.cfg.mass
        speed = float(np.linalg.norm(vel))
        acc = acc - self.cfg.drag_quadratic * speed * vel / self.cfg.mass
        dquat = 0.5 * quat_mul(quat, np.array([0.0, *omega]))
        domega = self.inv_inertia * (tau - np.cross(omega, self.inertia * omega))
        return np.concatenate([vel, acc, dquat, domega])

    def integrate(self, dt: float) -> None:
        y = np.concatenate([self.pos, self.vel, self.quat, self.omega])
        k1 = self._deriv(y)
        k2 = self._deriv(y + 0.5 * dt * k1)
        k3 = self._deriv(y + 0.5 * dt * k2)
        k4 = self._deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.pos, self.vel = y[:3].copy(), y[3:6].copy()
        self.quat = quat_normalize(y[6:10])
        self.omega = y[10:13].copy()

    # ---------- 接触 ----------
    def _apply_contact(self, normal: np.ndarray, pen: float) -> None:
        """位置投影 + 法向冲量。pen 沿 normal 方向把球推出去。"""
        self.pos = self.pos + normal * pen
        vn = float(self.vel @ normal)
        if vn < 0.0:
            bounce = 0.0 if -vn < STOP_SPEED else (1.0 + RESTITUTION) * vn
            self.vel = self.vel - bounce * normal
            tangent = self.vel - float(self.vel @ normal) * normal
            self.vel = self.vel - FRICTION * tangent
            self.omega *= 1.0 - min(0.5, abs(vn))

    def resolve_contacts(self, boxes: list[Box]) -> None:
        r = self.cfg.collision_radius
        if self.pos[2] < r:
            self._apply_contact(np.array([0.0, 0.0, 1.0]), r - self.pos[2])
        for b in boxes:
            d = self.pos - b.center
            closest = np.clip(d, -b.half, b.half)
            delta = d - closest
            dist = float(np.linalg.norm(delta))
            if dist > r:
                continue
            if dist > 1e-9:
                self._apply_contact(delta / dist, r - dist)
            else:
                # 球心在盒内：沿穿透最浅的轴推出去，避免除零
                gaps = b.half - np.abs(d)
                axis = int(np.argmin(gaps))
                n = np.zeros(3)
                n[axis] = 1.0 if d[axis] >= 0 else -1.0
                self._apply_contact(n, gaps[axis] + r)


class RigidEngine:
    """与 `World` 相同的对外接口，只是不用物理引擎。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.sim = cfg.sim
        self.backend = "rigid"
        self.drone_cfg = cfg.drone
        self.gravity = cfg.sim.gravity
        self.drones: dict[str, RigidQuad] = {}
        self.boxes: list[Box] = []
        self.t = 0.0
        self.steps = 0
        self.real_time = cfg.sim.real_time
        self._wall_start = time.perf_counter()
        self._box_index = 0

    def add_drone(self, name: str, position, yaw: float = 0.0) -> RigidQuad:
        if name in self.drones:
            raise ValueError(f"机体名重复：{name}")
        q = RigidQuad(self.drone_cfg, self.sim, position, yaw, name)
        self.drones[name] = q
        return q

    def add_box(self, half_extents, position, rgba=(0.8, 0.5, 0.2, 1.0), mass: float = 0.0) -> int:
        """静态障碍。mass > 0 会被忽略并报错——本后端只支持静态墙体。"""
        if mass:
            raise ValueError("RigidEngine 只支持静态障碍（mass=0）")
        self._box_index += 1
        self.boxes.append(Box(np.asarray(half_extents, dtype=float),
                              np.asarray(position, dtype=float), rgba, self._box_index))
        return self._box_index

    def _resolve_drone_pairs(self) -> None:
        names = sorted(self.drones)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = self.drones[names[i]], self.drones[names[j]]
                r = a.cfg.collision_radius + b.cfg.collision_radius
                delta = b.pos - a.pos
                dist = float(np.linalg.norm(delta))
                if dist >= r or dist < 1e-9:
                    continue
                n = delta / dist
                push = 0.5 * (r - dist)
                a.pos -= n * push
                b.pos += n * push
                rel = float((b.vel - a.vel) @ n)
                if rel < 0.0:
                    imp = -(1.0 + RESTITUTION) * rel / 2.0
                    a.vel -= imp * n
                    b.vel += imp * n

    def advance(self, dt: float) -> None:
        phys_dt = self.sim.physics_dt
        n = int(round(dt / phys_dt))
        if n < 1 or abs(n * phys_dt - dt) > 1e-9:
            raise ValueError(f"dt={dt} 不是 physics_dt={phys_dt} 的整数倍")
        for _ in range(n):
            for q in self.drones.values():
                q.motor_step(phys_dt)
            for q in self.drones.values():
                q.integrate(phys_dt)
                q.resolve_contacts(self.boxes)
            self._resolve_drone_pairs()
            self.t += phys_dt
            self.steps += 1
        if self.real_time:
            delay = (self._wall_start + self.t) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def state_vector(self) -> np.ndarray:
        return np.stack([self.drones[n].state().pos for n in sorted(self.drones)])

    def close(self) -> None:
        for q in self.drones.values():
            q.w_des = q.w.copy()

    def __enter__(self) -> "RigidEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
