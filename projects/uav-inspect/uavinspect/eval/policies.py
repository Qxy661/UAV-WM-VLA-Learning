"""三种基线策略，都实现同一个接口，便于 harness 一视同仁地评。

    policy.reset(quad)               每集开始调一次（策略可以读机体参数）
    policy.act(state, t) -> (thrust, torque)

注意：策略只能看到 `state` 与 `t`，看不到场景真值、看不到碰撞盒位置——
这是"评测要公平"的最低要求。巡检任务的目标点写在策略自己的参数里，
和人类飞手拿到的任务书一样。
"""

from __future__ import annotations

import numpy as np

from ..control.cascaded import CascadedController


class HoverPolicy:
    """什么都不做，原地定高。它给出"任务有多难"的下界。"""

    name = "hover-only"

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, quad):
        self.quad = quad

    def act(self, st, t):
        h = self.cfg.task.hover_height
        thrust = self.cfg.drone.mass * 9.81 + 2.0 * (h - st.pos[2]) - 1.5 * st.vel[2]
        return float(np.clip(thrust, 0.0, 20.0)), np.zeros(3)


class RandomPolicy:
    """悬停推力的随机扰动。它给出"乱飞能不能蒙对"的参照。"""

    name = "random"

    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def reset(self, quad):
        self.quad = quad

    def act(self, st, t):
        hover = self.cfg.drone.mass * 9.81
        thrust = hover * (1.0 + 0.35 * self.rng.standard_normal())
        thrust += 0.8 * (self.cfg.task.hover_height - st.pos[2])
        torque = self.rng.uniform(-0.12, 0.12, size=3)
        return float(np.clip(thrust, 0.0, 20.0)), torque


class WaypointPolicy:
    """脚本专家：按视点顺序飞，到点后悬停固定时长。

    它对应"人工示教"那条路线，也是 M2 数据采集用的同一种策略。
    """

    name = "scripted-waypoints"

    def __init__(self, cfg, waypoints=None, dwell_s: float = 2.5, tol: float = 0.16):
        self.cfg = cfg
        self.waypoints = []
        self.dwell_s = dwell_s
        self.tol = tol
        self.ctrl = None
        self.idx = 0
        self.dwell_start = None
        self.set_task(waypoints or [])

    def set_task(self, waypoints):
        """任务书：飞手拿到的视点清单（坐标 + 朝向），不含遮挡真值。"""
        out = []
        for wp in waypoints:
            if isinstance(wp, (tuple, list)) and len(wp) == 2 and np.ndim(wp[0]) == 1:
                out.append((np.asarray(wp[0], float), float(wp[1])))
            else:
                out.append((np.asarray(wp, float), 0.0))
        self.waypoints = out

    def reset(self, quad):
        self.quad = quad
        self.ctrl = CascadedController(self.cfg, quad)
        self.idx = 0
        self.dwell_start = None

    def act(self, st, t):
        if self.idx >= len(self.waypoints):
            return HoverPolicy(self.cfg).act(st, t)
        target, yaw_wp = self.waypoints[self.idx]
        err = float(np.linalg.norm(st.pos - target))
        if err <= self.tol:
            if self.dwell_start is None:
                self.dwell_start = t
            elif t - self.dwell_start >= self.dwell_s:
                self.idx += 1
                self.dwell_start = None
        else:
            self.dwell_start = None
        # 偏航要限速：一步从 0 转到 1.57 rad 会让姿态环瞬间吃到 1.5 rad 的误差，
        # 推力方向跟着甩，机体被甩到货架上。真实飞手也不会瞬间转 90°。
        dt = float(self.cfg.sim.control_dt)
        step = 0.8 * dt
        delta = float(np.arctan2(np.sin(yaw_wp - st.yaw), np.cos(yaw_wp - st.yaw)))
        yaw_cmd = st.yaw + float(np.clip(delta, -step, step))
        return self.ctrl.compute(st, target, yaw_ref=yaw_cmd)
