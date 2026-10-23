"""仿真世界：物理步进、多机、障碍、渲染。

关键设计：`advance(dt)` 只按物理步长推进，并且在**每个物理子步**里更新电机与外力。
把"控制一次、物理多步"做对，才谈得上频率分离（ADR-003）；否则所谓 240 Hz 物理
只是把同一个力重复施加 5 次，等于隐式改写了推力曲线。
"""

from __future__ import annotations

import time

import numpy as np
import pybullet as p

from ..config import Config
from .quad import Quadrotor


class World:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = p.connect(p.DIRECT)
        if self.client < 0:
            raise RuntimeError("PyBullet 连接失败（DIRECT 模式）")
        p.setGravity(0.0, 0.0, cfg.sim.gravity, physicsClientId=self.client)
        p.setTimeStep(cfg.sim.physics_dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=60, physicsClientId=self.client
        )
        self._add_ground()
        self.drones: dict[str, Quadrotor] = {}
        self.t = 0.0
        self.steps = 0
        self._wall_start = time.perf_counter()

    def _add_ground(self) -> None:
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_PLANE, planeNormal=[0, 0, 1], rgbaColor=[0.35, 0.35, 0.35, 1.0],
            physicsClientId=self.client,
        )
        self.ground = p.createMultiBody(0.0, col, vis, physicsClientId=self.client)

    def add_drone(self, name: str, position, yaw: float = 0.0) -> Quadrotor:
        if name in self.drones:
            raise ValueError(f"机体名重复：{name}")
        q = Quadrotor(self.client, self.cfg.drone, self.cfg.sim, position, yaw, name)
        self.drones[name] = q
        return q

    def add_box(self, half_extents, position, rgba=(0.8, 0.5, 0.2, 1.0), mass: float = 0.0) -> int:
        """静态障碍（墙/柱/楼）。M1 的遮挡就靠它。"""
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[float(v) for v in half_extents], physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[float(v) for v in half_extents], rgbaColor=list(rgba),
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            mass, col, vis, basePosition=[float(v) for v in position], physicsClientId=self.client
        )

    def advance(self, dt: float) -> None:
        """推进 dt 秒仿真时间；dt 必须是 physics_dt 的整数倍。"""
        phys_dt = self.cfg.sim.physics_dt
        n = int(round(dt / phys_dt))
        if n < 1 or abs(n * phys_dt - dt) > 1e-9:
            raise ValueError(f"dt={dt} 不是 physics_dt={phys_dt} 的整数倍")
        for _ in range(n):
            for q in self.drones.values():
                q.motor_step(phys_dt)
            for q in self.drones.values():
                q.apply()
            p.stepSimulation(physicsClientId=self.client)
            self.t += phys_dt
            self.steps += 1
        if self.cfg.sim.real_time:
            target = self._wall_start + self.t
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def state_vector(self) -> np.ndarray:
        """所有机体的位置，按名字排序；供最小间距等集合指标使用。"""
        return np.stack([self.drones[n].state().pos for n in sorted(self.drones)])

    def close(self) -> None:
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
