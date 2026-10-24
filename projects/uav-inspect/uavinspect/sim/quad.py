"""四旋翼刚体模型（PyBullet 后端）。

混控、电机滞后、悬停反解都在 `dynamics.py` 里，两种后端共用；这里只负责
"把扳手交给引擎、再从引擎读回状态"。

仍不真实的地方与 rigid.py 一致（写进 README 的 ADR-001）：没有桨叶挥摆、地效、
涡旋、电池电压衰减；电机是一阶而不是电流环。
"""

from __future__ import annotations

import numpy as np

from ..util.mathx import quat_to_rot
from .dynamics import QuadModel, State


class PyBulletQuad(QuadModel):
    """一架四旋翼，挂在 PyBullet 的 client 上。X 型布局。"""

    def __init__(
        self,
        client: int,
        cfg,
        sim,
        position=(0.0, 0.0, 0.2),
        yaw: float = 0.0,
        name: str = "uav",
    ) -> None:
        super().__init__(cfg, sim)
        import pybullet as p            # 延迟导入：没有 pybullet 的机器也能 import 本模块

        self.p = p
        self.client = client
        self.name = name

        col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=cfg.collision_radius, physicsClientId=client)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cfg.arm * 0.8, cfg.arm * 0.8, 0.02],
            rgbaColor=[0.2, 0.6, 1.0, 1.0],
            physicsClientId=client,
        )
        self.body = p.createMultiBody(
            baseMass=cfg.mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(v) for v in position],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=client,
        )
        # 控制器必须用引擎里真实的惯量，否则"参数"和"实际被积分的对象"是两套。
        self.inertia = np.array(
            p.getDynamicsInfo(self.body, -1, physicsClientId=client)[2], dtype=float)

    # ---------- 状态 ----------
    def state(self) -> State:
        p = self.p
        pos, quat = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        vel, ang = p.getBaseVelocity(self.body, physicsClientId=self.client)
        pos = np.asarray(pos, dtype=float)
        quat = np.asarray(quat, dtype=float)
        R = quat_to_rot(quat)
        omega_world = np.asarray(ang, dtype=float)
        return State(
            pos=pos,
            vel=np.asarray(vel, dtype=float),
            quat=quat,
            R=R,
            omega=R.T @ omega_world,   # 旋到机体系，角速率环在机体系里定义
            rotor_speed=self.w.copy(),
        )

    def apply(self) -> None:
        """把当前转速对应的力/力矩加进引擎。必须在 stepSimulation 之前调用。"""
        p = self.p
        st = self.state()
        fz, tau_body = self.rotor_wrench()
        force = st.R @ np.array([0.0, 0.0, fz])
        speed = float(np.linalg.norm(st.vel))
        force = force - self.cfg.drag_quadratic * speed * st.vel
        torque_world = st.R @ tau_body
        p.applyExternalForce(
            self.body, -1, force.tolist(), st.pos.tolist(), p.WORLD_FRAME,
            physicsClientId=self.client)
        p.applyExternalTorque(
            self.body, -1, torque_world.tolist(), p.WORLD_FRAME,
            physicsClientId=self.client)

    def reset(self, position, yaw: float = 0.0) -> None:
        p = self.p
        p.resetBasePositionAndOrientation(
            self.body, [float(v) for v in position],
            p.getQuaternionFromEuler([0.0, 0.0, yaw]), physicsClientId=self.client)
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self.reset_rotors()
