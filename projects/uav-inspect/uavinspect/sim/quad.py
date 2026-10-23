"""四旋翼刚体模型（PyBullet）。

与主库 `code/common/quad_sim.py` 的区别（这也是 ADR-001/002 的正文）：
- 姿态不是"无惯量直通量"，而是真实刚体：力矩 → 角加速度 → 角速度 → 姿态。
- 推力来自 4 个电机的当前转速平方，电机有一阶滞后；悬停不是"令 thrust = mg 即可"，
  而是"把转速稳到 1500 rad/s"。
- 反扭矩（yaw）由转子阻力矩产生，所以偏航是靠两侧转速差，不是直接给角速度。
- 阻力按 |v|·v 二次，符合真实气动；主库是线性阻尼。
- 重力、接触、地面反弹由引擎积分，不写在自研积分器里。

仍不真实的地方，必须写清楚，不假装等价：
- 没有桨叶挥摆（flapping）、地效、涡旋、电池电压随电量下降。
- 电机模型是一阶，不是真实电调的电流环。
- 这些差异对"控制/避让/评测"类结论影响有限，对"能耗/续航"类结论影响很大，
  所以本项目不做能耗结论。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from ..config import DroneCfg, SimCfg
from ..util.mathx import quat_to_rot


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


class Quadrotor:
    """一架四旋翼。X 型布局，0/2 逆时针、1/3 顺时针。"""

    def __init__(
        self,
        client: int,
        cfg: DroneCfg,
        sim: SimCfg,
        position=(0.0, 0.0, 0.2),
        yaw: float = 0.0,
        name: str = "uav",
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.name = name
        self.g = abs(sim.gravity)

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=cfg.collision_radius, physicsClientId=client)
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
            p.getDynamicsInfo(self.body, -1, physicsClientId=client)[2], dtype=float
        )

        # kf 由悬停转速反解：4·kf·w_hover² = m·g
        self.kf = cfg.mass * self.g / (4.0 * cfg.rotor.w_hover**2)
        self.max_thrust = 4.0 * self.kf * cfg.rotor.w_max**2

        s = cfg.arm / np.sqrt(2.0)
        self.rotor_xy = np.array([[s, -s], [-s, -s], [-s, s], [s, s]])
        self.spin = np.array([1.0, -1.0, 1.0, -1.0])   # +1 逆时针

        # (推力, 力矩) 与四个电机推力之间的混控矩阵，可逆即"可解混控"。
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

    # ---------- 状态 ----------
    def state(self) -> State:
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

    def motor_step(self, dt: float) -> None:
        alpha = min(dt / self.cfg.rotor.time_constant, 1.0)
        self.w += (self.w_des - self.w) * alpha
        np.clip(self.w, 0.0, self.cfg.rotor.w_max, out=self.w)

    def _rotor_wrench(self) -> tuple[float, np.ndarray]:
        t_each = self.kf * self.w**2
        return float(np.sum(t_each)), self.A[1:] @ t_each

    def apply(self) -> None:
        """把当前转速对应的力/力矩加进引擎。必须在 stepSimulation 之前调用。"""
        st = self.state()
        fz, tau_body = self._rotor_wrench()
        force = st.R @ np.array([0.0, 0.0, fz])
        speed = float(np.linalg.norm(st.vel))
        force = force - self.cfg.drag_quadratic * speed * st.vel
        torque_world = st.R @ tau_body
        p.applyExternalForce(
            self.body, -1, force.tolist(), st.pos.tolist(), p.WORLD_FRAME, physicsClientId=self.client
        )
        p.applyExternalTorque(
            self.body, -1, torque_world.tolist(), p.WORLD_FRAME, physicsClientId=self.client
        )

    def reset(self, position, yaw: float = 0.0) -> None:
        p.resetBasePositionAndOrientation(
            self.body, [float(v) for v in position], p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=self.client,
        )
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        self.w = np.full(4, self.cfg.rotor.w_hover)
        self.w_des = self.w.copy()
