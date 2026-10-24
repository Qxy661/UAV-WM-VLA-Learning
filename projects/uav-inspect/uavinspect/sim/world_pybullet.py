"""PyBullet 后端的世界（只在装得上 pybullet 的环境里用）。

接口与 `rigid.RigidEngine` 完全一致，`world.World(cfg, backend="pybullet")` 返回它。
"""

from __future__ import annotations

import time

import numpy as np

from ..config import Config
from .quad import PyBulletQuad


class PyBulletWorld:
    def __init__(self, cfg: Config) -> None:
        import pybullet as p

        self.p = p
        self.cfg = cfg
        self.client = p.connect(p.DIRECT)
        if self.client < 0:
            raise RuntimeError("PyBullet 连接失败（DIRECT 模式）")
        self.backend = "pybullet"
        p.setGravity(0.0, 0.0, cfg.sim.gravity, physicsClientId=self.client)
        p.setTimeStep(cfg.sim.physics_dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=60, physicsClientId=self.client)
        self._add_ground()
        self.drones: dict[str, PyBulletQuad] = {}
        self.boxes: list[int] = []
        self.t = 0.0
        self.steps = 0
        self._wall_start = time.perf_counter()

    def _add_ground(self) -> None:
        p = self.p
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_PLANE, planeNormal=[0, 0, 1],
            rgbaColor=[0.35, 0.35, 0.35, 1.0], physicsClientId=self.client)
        self.ground = p.createMultiBody(0.0, col, vis, physicsClientId=self.client)

    def add_drone(self, name: str, position, yaw: float = 0.0) -> PyBulletQuad:
        if name in self.drones:
            raise ValueError(f"机体名重复：{name}")
        q = PyBulletQuad(self.client, self.cfg.drone, self.cfg.sim, position, yaw, name)
        self.drones[name] = q
        return q

    def add_box(self, half_extents, position, rgba=(0.8, 0.5, 0.2, 1.0), mass: float = 0.0) -> int:
        """静态障碍（墙/柱/楼）。M1 的遮挡就靠它。"""
        p = self.p
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[float(v) for v in half_extents],
            physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[float(v) for v in half_extents],
            rgbaColor=list(rgba), physicsClientId=self.client)
        body = p.createMultiBody(
            mass, col, vis, basePosition=[float(v) for v in position],
            physicsClientId=self.client)
        self.boxes.append(body)
        return body

    def advance(self, dt: float) -> None:
        """推进 dt 秒仿真时间；dt 必须是 physics_dt 的整数倍。"""
        p = self.p
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
            delay = (self._wall_start + self.t) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def state_vector(self) -> np.ndarray:
        return np.stack([self.drones[n].state().pos for n in sorted(self.drones)])

    def close(self) -> None:
        if self.client >= 0:
            self.p.disconnect(physicsClientId=self.client)
            self.client = -1

    def __enter__(self) -> "PyBulletWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
