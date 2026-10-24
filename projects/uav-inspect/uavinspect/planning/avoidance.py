"""多机冲突避让：互惠斥力 + 切向破对称。

为什么不用势场碰撞锥（VO/ORCA）：ORCA 需要每个体的"允许速度半平面"求交，实现和调参
成本高，而 M1 要验的是一条更弱但更硬的命题——"四架机交叉穿越时任意两机不近于 0.6 m"。
在这个密度下，带切向分量的互惠斥力够用，且失败模式清楚（顶住不动或绕圈），
可以被代码测出来。

真实的坑（本文件要处理的）：
- 纯径向斥力在正面对穿时是对称的，两机会互相顶住、谁也不让。所以加切向分量，
  方向由 (i+j) 的奇偶决定，保证同一次仿真里是确定性的。
- 斥力只能改速度参考，不能改位置参考——否则会与到达判据打架（机体被"推离"目标，
  误差不收敛，看起来永远到不了）。
"""

from __future__ import annotations

import numpy as np


def min_separation(positions: np.ndarray) -> float:
    """任意两机的最小间距；不足两架时返回 inf。"""
    n = len(positions)
    if n < 2:
        return float("inf")
    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, float(np.linalg.norm(positions[i] - positions[j])))
    return best


def pairwise_separations(positions: np.ndarray) -> list[tuple[int, int, float]]:
    n = len(positions)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append((i, j, float(np.linalg.norm(positions[i] - positions[j]))))
    return out


def repulsive_velocity(
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    d_safe: float,
    d_influence: float,
    k_rep: float,
    v_max: float,
    tangential: float = 0.6,
) -> np.ndarray:
    """返回每架机应该叠加的避让速度（world 系，m/s）。

    d_safe      硬间距下限，低于它斥力迅速增大（1/d 型）
    d_influence 影响半径，超出不干预
    k_rep       增益，单位 m/s
    tangential  切向分量占比，打破正面对穿的对称僵持
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    n = len(pos)
    out = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9 or dist > d_influence:
                continue
            n_ij = delta / dist
            # 接近速度为正：越近、越对冲，让得越多
            closing = float((vel[i] - vel[j]) @ n_ij)
            w = k_rep * (1.0 / max(dist, d_safe * 0.5) - 1.0 / d_influence)
            w *= 1.0 + max(0.0, closing)
            tangent = np.array([-n_ij[1], n_ij[0], 0.0])
            if (i + j) % 2 == 1:
                tangent = -tangent
            push = w * (n_ij + tangential * tangent)
            out[i] -= push
            out[j] += push
    norms = np.linalg.norm(out, axis=1)
    scale = np.ones(n)
    mask = norms > v_max
    scale[mask] = v_max / norms[mask]
    return out * scale[:, None]


def velocity_command(
    pos: np.ndarray,
    vel: np.ndarray,
    v_ref: np.ndarray,
    *,
    d_safe: float,
    d_influence: float,
    k_rep: float,
    v_max: float,
    tangential: float = 0.6,
    yield_gain: float = 0.8,
) -> np.ndarray:
    """期望速度 = 名义速度 + 斥力，再对"正在闭合"的邻居让步减速。

    只加斥力不够：四机同时冲向中心时，斥力方向两两抵消，谁也不让，最近距离会掉到
    d_safe 以下。所以补一条"谁正在靠近谁减速"的规则——减速是单向的，不会和斥力抵消。
    让步量随距离线性衰减，在 d_influence 处归零，避免远处莫名减速。
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    n = len(pos)

    v = v_ref + repulsive_velocity(
        pos, vel, d_safe=d_safe, d_influence=d_influence,
        k_rep=k_rep, v_max=v_max, tangential=tangential)

    yield_scale = np.ones(n)
    for i in range(n):
        worst = 0.0
        for j in range(n):
            if i == j:
                continue
            delta = pos[j] - pos[i]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9 or dist > d_influence:
                continue
            closing = float((vel[i] - vel[j]) @ (delta / dist))
            if closing <= 0.0:
                continue
            # 距离越近、闭合越快，让得越多
            weight = (1.0 - dist / d_influence) * min(closing / max(v_max, 1e-6), 1.0)
            worst = max(worst, weight)
        yield_scale[i] = 1.0 - yield_gain * worst

    v = v * np.clip(yield_scale, 0.0, 1.0)[:, None]
    norms = np.linalg.norm(v, axis=1)
    mask = norms > v_max
    if mask.any():
        v[mask] *= (v_max / norms[mask])[:, None]
    return v

