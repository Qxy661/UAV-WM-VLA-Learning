"""E4 第二条：在 A\\* 路线之上加一层采样式 MPC（MPPI）做前瞻，和"追路点 + 斥力"对照。

同一批 20 个场景、同一台动力学、同一个级联控制器、同一条 A\\* 折线，只换"参考点从哪来"：

- `astar`：沿折线走，到 0.30 m 内切下一个路点，外层 P-D 直追。
- `astar_reactive`：同上再加一档小尺度斥力补余量（E4 第一条的最优解）。
- `astar_mpc`：用 MPPI 在当前路点方向滚 12 步、256 采样，把前瞻点交回外层 P-D。

指标与 E4 第一条完全相同（success / clean / length / clear / wall_t），否则不能并表。

为什么要这一层：反应式是"见墙才躲"，A\\* 是"事前绕开"，MPPI 是"事前三步内就知道要不要
提前减速转进去"。三者的差别只有一种量能分清——**同一路线上的最小间隙与到点时间**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                                  # noqa: E402
from uavinspect.control.cascaded import CascadedController          # noqa: E402
from uavinspect.planning.global_astar import (                      # noqa: E402
    Grid, astar, clearance_2d, racks_2d)
from uavinspect.planning.mpc import PointMassMPPI                   # noqa: E402
from uavinspect.sim.scene import RACK, RACK_X, ROOM, resolve_layout  # noqa: E402
from uavinspect.sim.world import World                              # noqa: E402

SCENARIOS = 20
VMAX = 1.6


def scenario(k: int) -> tuple[np.ndarray, np.ndarray]:
    """与 e4_compare.py 完全一致的 20 个起终点（起点在通道另一侧，终点在货架缝里）。"""
    rng = np.random.default_rng(1000 + k)
    i = k % (len(RACK_X) - 1)
    gap_x = 0.5 * (RACK_X[i] + RACK_X[i + 1])
    goal = np.array([gap_x + 0.10 * (rng.random() - 0.5), 0.0])
    start = np.array([gap_x + 0.60 * (rng.random() - 0.5), -1.8 - 0.4 * rng.random()])
    return start, goal


def follow_target(pos, path, idx: int, *, tol: float = 0.30) -> tuple[np.ndarray, int]:
    while idx < len(path) - 1 and np.linalg.norm(path[idx] - pos) < tol:
        idx += 1
    return path[idx], idx


def obstacle_push(pos: np.ndarray, boxes, *, d_infl: float = 0.65, gain: float = 1.0) -> np.ndarray:
    """与 e4_compare 里 astar_reactive 用的同一档斥力（小尺幅、只补余量）。"""
    out = np.zeros(2)
    for cx, cy, hx, hy in boxes:
        dx, dy = pos[0] - cx, pos[1] - cy
        qx = float(np.clip(dx, -hx, hx))
        qy = float(np.clip(dy, -hy, hy))
        vx, vy = dx - qx, dy - qy
        d = float(np.hypot(vx, vy))
        if d < 1e-6:
            push = np.array([0.0, np.sign(dy) or 1.0]) * (hy - abs(dy) + hy)
            out += push * gain * 2.0
            continue
        if d < d_infl:
            out += np.array([vx, vy]) / d * gain * (d_infl - d) / d_infl
    return out


def run_one(cfg, k: int, planner: str, *, backend: str, duration: float, arrive_tol: float):
    start, goal = scenario(k)
    z = cfg.task.hover_height
    lay = resolve_layout("a")
    boxes = racks_2d(lay[1]["rack_x"], lay[1]["rack_y"], RACK[0], RACK[1])
    grid = Grid(ROOM, boxes, res=0.10, inflate=0.30)
    path = astar(grid, start, goal)
    if not path:
        return dict(success=False, clean=False, length=0.0, clear=-9.0, wall_t=0.0,
                    note="astar 无解")

    mpc = PointMassMPPI(z_ref=z, seed=k) if planner == "astar_mpc" else None

    with World(cfg, backend=backend) as world:
        quad = world.add_drone("uav", (float(start[0]), float(start[1]), z))
        ctrl = CascadedController(cfg, quad)
        dt = cfg.sim.control_dt
        t, idx, length, clear_min = 0.0, 0, 0.0, float("inf")
        last = start.copy()
        reached_at = None
        while t < duration:
            st = quad.state()
            pos = np.array([st.pos[0], st.pos[1]])
            length += float(np.linalg.norm(pos - last))
            last = pos
            clear_min = min(clear_min, clearance_2d(pos, boxes))

            if np.linalg.norm(pos - goal) < arrive_tol:
                reached_at = reached_at or t
                if t - reached_at > 1.0:
                    break
            else:
                reached_at = None

            wp, idx = follow_target(pos, path, idx)
            if planner == "astar_mpc":
                st3 = np.array([pos[0], pos[1], float(st.pos[2])])
                goal3 = np.array([wp[0], wp[1], z])
                look = mpc.plan(st3, np.asarray(st.vel, float), goal3, boxes)
                d2 = look[:2] - pos
                if float(np.linalg.norm(d2)) < 1e-6:
                    d2 = wp - pos
                step = min(0.80, float(np.linalg.norm(wp - pos)))
                target = pos + d2 / max(float(np.linalg.norm(d2)), 1e-9) * step
            else:
                target = wp

            v = cfg.control.pos_kp * (target - pos)
            n = float(np.linalg.norm(v))
            if n > VMAX:
                v = v * (VMAX / n)
            if planner == "astar_reactive" and np.linalg.norm(goal - pos) > 0.6:
                v = v + obstacle_push(pos, boxes)
                n2 = float(np.linalg.norm(v))
                if n2 > VMAX:
                    v = v * (VMAX / n2)
            ref_pos = pos + v / cfg.control.pos_kp
            thrust, torque = ctrl.compute(
                st, np.array([ref_pos[0], ref_pos[1], z]),
                ref_vel=np.array([v[0], v[1], 0.0]))
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt

    ok = bool(np.linalg.norm(np.array([last[0], last[1]]) - goal) < arrive_tol)
    return dict(success=ok, clean=bool(ok and clear_min >= 0.0), length=length,
                clear=clear_min, wall_t=t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4 局部 MPC 对照")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "e4_mpc"))
    ap.add_argument("--backend", default="rigid", choices=["auto", "rigid", "pybullet"])
    ap.add_argument("--n", type=int, default=SCENARIOS)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--arrive-tol", type=float, default=0.30)
    a = ap.parse_args(argv)

    cfg = load(Path(a.config))
    planners = ("astar", "astar_reactive", "astar_mpc")
    rows = {p: [] for p in planners}
    for k in range(a.n):
        for p in planners:
            r = run_one(cfg, k, p, backend=a.backend, duration=a.duration, arrive_tol=a.arrive_tol)
            r["k"] = k
            rows[p].append(r)
        if (k + 1) % 5 == 0:
            print(f"[{k+1}/{a.n}]", {p: round(np.mean([x["success"] for x in rows[p]]), 3) for p in planners})

    summary = {}
    for p in planners:
        rs = rows[p]
        succ = [x for x in rs if x["success"]]
        summary[p] = {
            "success": float(np.mean([x["success"] for x in rs])),
            "clean": float(np.mean([x["clean"] for x in rs])),
            "length_mean": float(np.mean([x["length"] for x in succ])) if succ else 0.0,
            "clear_min": float(np.min([x["clear"] for x in rs])),
            "wall_mean": float(np.mean([x["wall_t"] for x in rs])),
        }
        print(p, summary[p])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"n": a.n, "duration": a.duration, "summary": summary, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({p: round(summary[p]["success"], 3) for p in planners}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
