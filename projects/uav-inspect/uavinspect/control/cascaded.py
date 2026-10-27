"""串级控制器：位置 → 期望姿态 → 角速率 → 力矩。

为什么不用主库 demo 里的"串级 P-D + 开环角速率"：开环角速率意味着姿态不是被控量，
真实刚体下会把姿态推到极限自激（`code/common/quad_sim.py` 的注释自己也承认了）。
这里最后一个环闭合在角速率上，力矩由惯量反解，才是可上机的结构。

几何控制律（Lee 型）：
    a_des = kp·e_p + kd·e_v                     （外环，产生期望加速度）
    F_des = m·(a_des + g·e_z)                   （含重力前馈）
    R_des = align(b3 = F_des/|F_des|, yaw)      （期望姿态）
    e_R   = ½·vee(R_desᵀR − RᵀR_des)
    ω_ref = −k_att·e_R                          （姿态误差转期望角速率）
    τ     = I·(k_rate·(ω_ref − ω)) + ω×(I·ω)    （角速率环 + 陀螺补偿）

符号约定容易写错，实测用 `tests` 里的 tx 符号测试兜底（参考点 +y 时应产生负 roll 力矩）。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..sim.dynamics import State
from ..util.mathx import vee


class CascadedController:
    def __init__(self, cfg: Config, quad) -> None:
        self.cfg = cfg
        self.mass = quad.cfg.mass
        self.inertia = quad.inertia
        self.max_thrust = quad.max_thrust
        self.g = quad.g
        self.max_tilt = np.deg2rad(cfg.control.max_tilt_deg)

    def compute(
        self,
        st: State,
        ref_pos,
        ref_vel=(0.0, 0.0, 0.0),
        yaw_ref: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        c = self.cfg.control
        ref_pos = np.asarray(ref_pos, dtype=float)
        ref_vel = np.asarray(ref_vel, dtype=float)

        # 外环：位置/速度误差 → 期望加速度（含重力前馈，否则悬停需要稳态误差）
        a_des = c.pos_kp * (ref_pos - st.pos) + c.pos_kd * (ref_vel - st.vel)
        f_des = self.mass * (a_des + np.array([0.0, 0.0, self.g]))

        # 倾角约束：限制水平分量，保证 |tilt| ≤ max_tilt。写成投影而不是事后裁剪推力，
        # 因为事后裁剪会同时削掉高度通道，导致"限倾角就掉高"。
        fz = max(f_des[2], 0.05 * self.mass * self.g)
        fh = f_des[:2]
        lim = np.tan(self.max_tilt) * fz
        n_h = float(np.linalg.norm(fh))
        if n_h > lim and n_h > 1e-9:
            fh = fh * (lim / n_h)
        f_des = np.array([fh[0], fh[1], fz])

        b3 = f_des / np.linalg.norm(f_des)
        c1 = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        b2 = np.cross(b3, c1)
        n2 = float(np.linalg.norm(b2))
        if n2 < 1e-6:                      # 期望推力与偏航参考共线时的退化处理
            c1 = np.array([1.0, 0.0, 0.0])
            b2 = np.cross(b3, c1)
            n2 = float(np.linalg.norm(b2))
        b2 = b2 / n2
        b1 = np.cross(b2, b3)
        R_des = np.column_stack([b1, b2, b3])

        e_R = vee(R_des.T @ st.R - st.R.T @ R_des)
        omega_ref = -c.att_kp * e_R
        # 期望角速率留一个出口：动作空间实验把它当"动作"记录下来，
        # 再由低层速率环（rate_kp）转成力矩。控制器本身行为不变。
        self.omega_ref = omega_ref
        torque = self.inertia * (c.rate_kp * (omega_ref - st.omega)) + np.cross(
            st.omega, self.inertia * st.omega
        )

        thrust = float(f_des @ st.R[:, 2])
        t_max = c.thrust_margin * self.max_thrust
        thrust = float(np.clip(thrust, 0.0, t_max))
        return thrust, torque
