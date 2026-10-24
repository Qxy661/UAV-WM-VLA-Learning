"""M1：四机同场交叉穿越 + 互惠斥力避让。

验收（代码判定，README §5）：
    1. 全程任意两机间距 ≥ d_safe（默认 0.6 m）
    2. 四架都在时限内到达各自目标并停住（末 2 s 误差 ≤ arrive_tol）
    3. 全程不触地（z ≥ 0.05）

产物写入 runs/m1_multi/：metrics.json / env.json / trace.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                                   # noqa: E402
from uavinspect.control.cascaded import CascadedController           # noqa: E402
from uavinspect.planning.avoidance import min_separation, pairwise_separations, velocity_command  # noqa: E402
from uavinspect.sim.world import World                               # noqa: E402
from uavinspect.util.latency import StageTimer                        # noqa: E402

NAMES = ["uav_a", "uav_b", "uav_c", "uav_d"]
START = np.array([
    [-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0],
])
# 目标取对角线：四条航迹在中心交叉，冲突是设计出来的，不是撞运气撞上的
GOAL = np.array([
    [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0],
])
PILLARS = [((0.12, 0.12, 1.2), (0.0, 1.4, 1.2)), ((0.12, 0.12, 1.2), (0.0, -1.4, 1.2))]
HEADER = ["t", "min_sep", "vmax", "zmin"] + [f"{n}_{a}" for n in NAMES for a in ("x", "y", "z")]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 四机交叉穿越 + 避让")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "m1_multi"))
    ap.add_argument("--backend", default="auto", choices=["auto", "rigid", "pybullet"])
    ap.add_argument("--d-safe", type=float, default=0.6)
    ap.add_argument("--d-influence", type=float, default=2.5)
    ap.add_argument("--k-rep", type=float, default=1.5)
    ap.add_argument("--v-avoid", type=float, default=1.2)
    ap.add_argument("--yield-gain", type=float, default=0.8)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--arrive-tol", type=float, default=0.25)
    args = ap.parse_args(argv)

    cfg = load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dt = cfg.sim.control_dt
    z = cfg.task.hover_height
    goals = np.column_stack([GOAL, np.full(len(GOAL), z)])
    timer = StageTimer()

    rows: list[list[float]] = []
    sep_series: list[float] = []
    tol_series: list[float] = []
    worst = (float('inf'), None, 0.0)
    events: list[tuple[float, str, float]] = []

    wall0 = time.perf_counter()
    with World(cfg, backend=args.backend) as world:
        backend = getattr(world, "backend", args.backend)
        quads = [world.add_drone(n, (START[i, 0], START[i, 1], 0.25)) for i, n in enumerate(NAMES)]
        for half, pos in PILLARS:
            world.add_box(half, pos)
        ctrls = [CascadedController(cfg, q) for q in quads]

        t = 0.0
        while t < args.duration:
            pos = np.array([q.state().pos for q in quads])
            vel = np.array([q.state().vel for q in quads])

            # 名义速度由位置误差给出，再叠加斥力与让步；位置参考按同一速度反推，
            # 这样避让改的是"往哪飞、飞多快"，而不是把目标点搬来搬去（后者会与到达判据打架）。
            v_raw = cfg.control.pos_kp * (goals - pos)
            n_raw = np.linalg.norm(v_raw, axis=1)
            if np.any(n_raw > args.v_avoid):
                v_raw[n_raw > args.v_avoid] *= (args.v_avoid / n_raw[n_raw > args.v_avoid])[:, None]
            v_ref = velocity_command(
                pos, vel, v_raw, d_safe=args.d_safe, d_influence=args.d_influence,
                k_rep=args.k_rep, v_max=args.v_avoid, yield_gain=args.yield_gain)
            ref_pos = pos + v_ref / cfg.control.pos_kp

            for i, q in enumerate(quads):
                st = q.state()
                t0 = time.perf_counter()
                thrust, torque = ctrls[i].compute(st, ref_pos[i], ref_vel=v_ref[i])
                timer.record("control", time.perf_counter() - t0)
                t0 = time.perf_counter()
                q.set_wrench(thrust, torque)
                timer.record("command", time.perf_counter() - t0)

            t0 = time.perf_counter()
            world.advance(dt)
            timer.record("physics", time.perf_counter() - t0)
            t += dt

            pos = np.array([q.state().pos for q in quads])
            vel = np.array([q.state().vel for q in quads])
            sep = min_separation(pos)
            sep_series.append(sep)
            if sep < worst[0]:
                pr = min(pairwise_separations(pos), key=lambda x: x[2])
                worst = (sep, (NAMES[pr[0]], NAMES[pr[1]]), t)
            if sep < args.d_safe + 0.1:
                pr = min(pairwise_separations(pos), key=lambda x: x[2])
                events.append((round(t, 3), f"{NAMES[pr[0]]}-{NAMES[pr[1]]}", round(sep, 3)))
            tol = float(np.max(np.linalg.norm(pos - goals, axis=1)))
            tol_series.append(tol)
            rows.append([t, sep, float(np.max(np.linalg.norm(vel, axis=1))),
                         float(pos[:, 2].min()), *pos.reshape(-1).tolist()])

        sim_time = world.t

    wall = time.perf_counter() - wall0
    arr = np.asarray(rows, dtype=float)
    pos_series = arr[:, 4:].reshape(len(arr), len(NAMES), 3)
    final_err = np.linalg.norm(pos_series[-1] - goals, axis=1)
    # 末 2 s 内每架机的最大误差：证明"到了并停住"，不是"路过"
    tail = pos_series[arr[:, 0] >= sim_time - 2.0]
    tail_err = np.linalg.norm(tail - goals[None, :, :], axis=2).max(axis=0)

    min_sep = float(np.min(sep_series))
    conflicts = int(np.sum(np.asarray(sep_series) < args.d_influence))
    zmin = float(arr[:, 3].min())
    vmax = float(arr[:, 2].max())

    checks = {
        "separation_ok": min_sep >= args.d_safe - 1e-6,
        "all_arrived": bool(np.all(tail_err <= args.arrive_tol)),
        "no_ground_hit": zmin >= 0.05,
    }
    passed = all(checks.values())

    metrics = {
        "task": "m1_multi",
        "backend": backend,
        "passed": passed,
        "checks": checks,
        "metrics": {
            "min_separation_m": min_sep,
            "worst_pair": list(worst[1]) if worst[1] else None,
            "worst_time_s": worst[2],
            "close_steps": len(events),
            "d_safe_m": args.d_safe,
            "conflict_steps": conflicts,
            "conflict_step_ratio": conflicts / len(sep_series),
            "max_speed_mps": vmax,
            "min_z_m": zmin,
            "final_err_m": final_err.tolist(),
            "tail_err_m": tail_err.tolist(),
        },
        "params": {"d_influence": args.d_influence, "k_rep": args.k_rep,
                   "v_avoid": args.v_avoid, "yield_gain": args.yield_gain, "duration": args.duration,
                   "arrive_tol": args.arrive_tol},
        "latency": timer.summary(),
        "sim": {"sim_time_s": sim_time, "wall_time_s": wall,
                "real_time_factor": sim_time / wall if wall > 0 else None,
                "physics_steps": world.steps if hasattr(world, "steps") else None},
        "seed": cfg.seed,
    }
    env = {
        "argv": sys.argv, "cwd": str(Path.cwd()), "git_commit": _git_commit(),
        "platform": platform.platform(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    with (out / "trace.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(HEADER) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")

    print(f"M1 四机交叉穿越  backend={backend}")
    print(f"  最小机间距    {min_sep:.3f} m  （下限 {args.d_safe:.2f}）")
    print(f"  进入影响区    {conflicts} 步 / {len(sep_series)} 步 "
          f"({conflicts / len(sep_series) * 100:.1f}%)")
    print(f"  最大速度      {vmax:.2f} m/s    最低高度 {zmin:.3f} m")
    print(f"  最近一次      {worst[0]:.3f} m  {worst[1]}  t={worst[2]:.2f}s")
    print(f"  贴脸事件      {len(events)} 步（< {args.d_safe + 0.1:.2f} m）前 5: {events[:5]}")
    print(f"  末 2 s 误差   " + "  ".join(f"{n}={e:.3f}" for n, e in zip(NAMES, tail_err)))
    print(f"  实时因子      {sim_time / wall:.2f}x")
    print()
    print(timer.table())
    print()
    print(f"判定：{'通过' if passed else '未通过'}  {checks}")
    print(f"产物：{out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
