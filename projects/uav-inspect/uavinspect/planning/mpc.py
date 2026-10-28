"""E4 第二条：局部层用采样式 MPC（MPPI）做前瞻，替掉"一路追终点"。

为什么必须有这一层：全局 A\\* 给出的折线在 0.1 m 栅格上过一遍视线剪枝就够了，但真飞时
外层 P-D 是"朝终点直着拉"，遇到缝口或墙边就会顶着飞。E4 第一条量到反应式斥力 0/20 全
超时、A\\* 20/20，说明差别在**全局**；这里再往下一层问一次：把"追终点"换成"追前瞻点"，
能不能在同样的 A\\* 路线上把跟踪误差压下去。

模型与代价：

- 预测模型是可积的三维质点 `p ← p + v·dt + ½a·dt²`、`v ← v + a·dt`，`|a| ≤ a_max`。
  用质点而不是六自由度，理由写在 RESULTS 里：姿态内环已经有 30 Hz 的级联 P-D，MPC 只负责
  "往哪走"，这两件事分开才好定位谁坏。
- 采样：K 条轨迹、每条 H 步，噪声只在控制序列上（σ 随迭代衰减）。权重 `w ∝ exp(-cost/λ)`。
- 代价：终点距离（权重最大）、沿途离障碍的间隙（小于 `clear_margin` 时平方罚）、高过低或
  过高、控制量大小。

对外只暴露 `plan(state, goal, boxes) -> waypoint`：把最优轨迹上第 `lookahead` 步的
位置交回给外层 P-D 当参考点。这样接进来的东西和 E4 第一条的 `astar_reactive` 完全同构，
可以直接对比。
"""

from __future__ import annotations

import numpy as np

from .global_astar import clearance_2d


class PointMassMPPI:
    def __init__(
        self,
        *,
        horizon: int = 12,
        samples: int = 256,
        dt: float = 0.05,
        a_max: float = 4.0,
        sigma: float = 1.6,
        lam: float = 0.6,
        clear_margin: float = 0.55,
        w_clear: float = 40.0,
        w_goal: float = 12.0,
        w_ctrl: float = 0.05,
        lookahead: int = 4,
        z_ref: float = 1.5,
        w_z: float = 6.0,
        seed: int = 0,
    ) -> None:
        self.H = int(horizon)
        self.K = int(samples)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.clear_margin = float(clear_margin)
        self.w_clear = float(w_clear)
        self.w_goal = float(w_goal)
        self.w_ctrl = float(w_ctrl)
        self.lookahead = int(lookahead)
        self.z_ref = float(z_ref)
        self.w_z = float(w_z)
        self.rng = np.random.default_rng(seed)
        self._u = np.zeros((self.H, 3))          # 上一拍的最优序列，热启动用

    # ------------------------------------------------------------------ 代价
    def _cost(self, traj: np.ndarray, goal: np.ndarray, boxes) -> np.ndarray:
        """traj: (K, H, 3) 位置。返回 (K,) 代价。"""
        K, H, _ = traj.shape
        cost = np.zeros(K)
        # 终点
        cost += self.w_goal * np.sum((traj[:, -1, :] - goal[None, :]) ** 2, axis=1)
        # 沿途：越靠近终点越不重要（否则会在起点就想着终点）
        for h in range(H):
            frac = (h + 1) / H
            cost += (0.35 + 0.65 * frac) * np.sum((traj[:, h, :] - goal[None, :]) ** 2, axis=1)
        # 高度
        cost += self.w_z * np.sum((traj[:, :, 2] - self.z_ref) ** 2, axis=1)
        # 障碍间隙（只看 xy）
        for h in range(H):
            xy = traj[:, h, :2]
            clear = np.array([clearance_2d(p, boxes) for p in xy])
            pen = np.clip(self.clear_margin - clear, 0.0, None)
            cost += self.w_clear * pen ** 2
        return cost

    def _rollout(self, p0: np.ndarray, v0: np.ndarray, U: np.ndarray):
        """U: (K, H, 3) 控制序列 → (K, H, 3) 位置轨迹。"""
        K, H, _ = U.shape
        p = np.repeat(p0[None, :], K, axis=0)
        v = np.repeat(v0[None, :], K, axis=0)
        traj = np.empty((K, H, 3))
        for h in range(H):
            a = U[:, h, :]
            p = p + v * self.dt + 0.5 * a * self.dt ** 2
            v = v + a * self.dt
            traj[:, h, :] = p
        return traj

    # ------------------------------------------------------------------ 规划
    def plan(self, pos, vel, goal, boxes) -> np.ndarray:
        """返回一个前瞻参考点（世界坐标）。"""
        p0 = np.asarray(pos, dtype=float)
        v0 = np.asarray(vel, dtype=float)
        goal = np.asarray(goal, dtype=float)

        base = np.repeat(self._u[None, :, :], self.K, axis=0)      # (K,H,3) 热启动
        noise = self.rng.normal(0.0, self.sigma, size=(self.K, self.H, 3))
        U = np.clip(base + noise, -self.a_max, self.a_max)

        traj = self._rollout(p0, v0, U)
        cost = self._cost(traj, goal, boxes)
        cost += self.w_ctrl * np.sum(U ** 2, axis=(1, 2))

        cmin = float(cost.min())
        w = np.exp(-(cost - cmin) / max(self.lam, 1e-6))
        w /= max(float(w.sum()), 1e-9)
        self._u = np.clip(np.tensordot(w, U, axes=(0, 0)), -self.a_max, self.a_max)

        # 用更新后的序列再滚一次，取前瞻点
        best = self._rollout(p0, v0, np.repeat(self._u[None, :, :], 1, axis=0))[0]
        idx = min(self.lookahead, self.H - 1)
        return best[idx].copy()

    def reset(self) -> None:
        self._u = np.zeros((self.H, 3))
