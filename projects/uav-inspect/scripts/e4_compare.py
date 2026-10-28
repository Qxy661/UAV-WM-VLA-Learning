"""E4 规划栈对照：反应式斥力 / 全局 A* / A*+斥力，同一批 20 个场景。

为什么要做这个对照：本库里所有「能飞到」的说法都靠反应式斥力，而反应式在
「货架正好挡在中间」时只能把飞机顶在货架前。这里用同一套动力学、同一个控制器、
同一批起终点，只换规划器，看差多少。

指标（三项都要报，不许只报成功率）：
- success：末态到目标 < --arrive-tol 并保持 1 s
- length：轨迹积分长度（成功回合才统计，否则短＝提前放弃）
- clear：全程到货架表面的最小距离（负值即撞上）

输出 runs/e4_compare/{metrics.json, trace_<planner>_<k>.csv}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                                      # noqa: E402
from uavinspect.control.cascaded import CascadedController              # noqa: E402
from uavinspect.planning.global_astar import (                          # noqa: E402
    Grid, astar, clearance_2d, path_length, racks_2d)
from uavinspect.sim.scene import RACK, RACK_X, RACK_Y, ROOM, resolve_layout  # noqa: E402
from uavinspect.sim.world import World                                  # noqa: E402

PLANNERS = ("reactive", "astar", "astar_reactive")


def scenario(k: int) -> tuple[np.ndarray, np.ndarray]:
    """20 个起终点：起点在通道另一侧，终点落在两排货架之间的缝里。

    这样直连航线必然要穿过货架线，而唯一通路是 0.9 m 宽的货架缝——
    反应式斥力在缝口会被两侧同时推开（这正是它该暴露的失效模式）。
    """
    rng = np.random.default_rng(1000 + k)
    i = k % (len(RACK_X) - 1)
    gap_x = 0.5 * (RACK_X[i] + RACK_X[i + 1])
    goal = np.array([gap_x + 0.10 * (rng.random() - 0.5), 0.0])
    start = np.array([gap_x + 0.60 * (rng.random() - 0.5), -1.8 - 0.4 * rng.random()])
    return start, goal


def obstacle_push(pos: np.ndarray, boxes, *, d_infl: float = 1.2, gain: float = 2.6) -> np.ndarray:
    """货架斥力：只改速度参考，不改位置参考（否则与到达判据打架）。"""
    out = np.zeros(2)
    for cx, cy, hx, hy in boxes:
        dx = pos[0] - cx
        dy = pos[1] - cy
        # 到矩形的最短向量（矩形外用投影点，矩形内用最浅轴）
        qx = float(np.clip(dx, -hx, hx))
        qy = float(np.clip(dy, -hy, hy))
        vx, vy = dx - qx, dy - qy
        d = float(np.hypot(vx, vy))
        if d < 1e-6:
            # 在矩形内部：沿最浅轴推出去
            push = np.array([0.0, np.sign(dy) or 1.0]) * (hy - abs(dy) + hy)
            out += push * gain * 2.0
            continue
        if d < d_infl:
            out += np.array([vx, vy]) / d * gain * (d_infl - d) / d_infl
    return out


def follow_target(pos, path, idx: int, *, tol: float = 0.30) -> tuple[np.ndarray, int]:
    while idx < len(path) - 1 and np.linalg.norm(path[idx] - pos) < tol:
        idx += 1
    return path[idx], idx


def run_one(cfg, k: int, planner: str, *, backend: str, duration: float, arrive_tol: float):
    start, goal = scenario(k)
    z = cfg.task.hover_height
    lay = resolve_layout("a")
    boxes = racks_2d(lay[1]["rack_x"], lay[1]["rack_y"], RACK[0], RACK[1])
    grid = Grid(ROOM, boxes, res=0.10, inflate=0.30)
    path = astar(grid, start, goal) if planner != "reactive" else []

    with World(cfg, backend=backend) as world:
        quad = world.add_drone("uav", (float(start[0]), float(start[1]), z))
        ctrl = CascadedController(cfg, quad)
        dt = cfg.sim.control_dt
        t, idx, length, clear_min = 0.0, 0, 0.0, float("inf")
        last = start.copy()
        reached_at = None
        traj = []
        while t < duration:
            st = quad.state()
            pos = np.array([st.pos[0], st.pos[1]])
            length += float(np.linalg.norm(pos - last))
            last = pos
            clear_min = min(clear_min, clearance_2d(pos, boxes))
            traj.append((t, pos[0], pos[1], st.pos[2]))

            if np.linalg.norm(pos - goal) < arrive_tol:
                reached_at = reached_at or t
                if t - reached_at > 1.0:
                    break
            else:
                reached_at = None

            if planner == "reactive":
                target = goal
            else:
                target, idx = follow_target(pos, path, idx)

            v = cfg.control.pos_kp * (target - pos)
            n = float(np.linalg.norm(v))
            vmax = 1.6
            if n > vmax:
                v = v * (vmax / n)
            if planner in ("reactive", "astar_reactive") and np.linalg.norm(goal - pos) > 0.6:
                # 两种规划器的斥力尺度不同：纯反应式没有全局信息，只能调大感应半径硬顶；
                # A* 已经规划出穿缝的路线，斥力只需补一点余量，调大反而把飞机顶在缝口。
                if planner == "reactive":
                    v = v + obstacle_push(pos, boxes, d_infl=1.2, gain=2.6)
                else:
                    v = v + obstacle_push(pos, boxes, d_infl=0.65, gain=1.0)
                n2 = float(np.linalg.norm(v))
                if n2 > vmax:
                    v = v * (vmax / n2)
            ref_pos = pos + v / cfg.control.pos_kp
            thrust, torque = ctrl.compute(st, np.array([ref_pos[0], ref_pos[1], z]), ref_vel=np.array([v[0], v[1], 0.0]))
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt

    ok = bool(np.linalg.norm(np.array([last[0], last[1]]) - goal) < arrive_tol)
    clean = ok and clear_min >= 0.0
    return dict(success=ok, clean=clean, length=length, clear=clear_min, wall_t=t,
                plan_pts=len(path), plan_len=path_length(path) if path else 0.0, traj=traj)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4 规划器对照")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "e4_compare"))
    ap.add_argument("--backend", default="rigid", choices=["auto", "rigid", "pybullet"])
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--arrive-tol", type=float, default=0.35)
    args = ap.parse_args(argv)

    cfg = load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {p: [] for p in PLANNERS}
    t0 = time.perf_counter()

    for k in range(args.scenarios):
        for p in PLANNERS:
            r = run_one(cfg, k, p, backend=args.backend, duration=args.duration, arrive_tol=args.arrive_tol)
            traj = r.pop("traj")
            results[p].append(r)
            with (out / f"trace_{p}_{k:02d}.csv").open("w", encoding="utf-8") as f:
                f.write("t,x,y,z\n")
                for row in traj:
                    f.write(",".join(f"{v:.4f}" for v in row) + "\n")

    summary = {}
    for p in PLANNERS:
        rows = results[p]
        succ = [r for r in rows if r["success"]]
        clean = [r for r in rows if r.get("clean")]
        summary[p] = dict(
            success_rate=round(len(succ) / len(rows), 4),
            success_clean_rate=round(len(clean) / len(rows), 4),
            n=len(rows),
            mean_length_m=round(float(np.mean([r["length"] for r in succ])), 3) if succ else None,
            median_clear_m=round(float(np.median([r["clear"] for r in rows])), 3),
            min_clear_m=round(float(np.min([r["clear"] for r in rows])), 3),
            collisions=int(sum(1 for r in rows if r["clear"] < 0.0)),
            mean_wall_s=round(float(np.mean([r["wall_t"] for r in rows])), 3),
        )
    payload = dict(backend=args.backend, scenarios=args.scenarios, duration_s=args.duration,
                   arrive_tol=args.arrive_tol, summary=summary,
                   wall_time_s=round(time.perf_counter() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
