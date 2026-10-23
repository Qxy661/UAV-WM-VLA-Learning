"""M0：单机定点悬停 + 四层频率 + 分阶段延迟。

验收由代码判定（README §5），不看"图好看"：
    1. 稳定后 RMSE ≤ task.accept_rmse
    2. 全程不触地（z ≥ 0.05）
    3. 末态高度偏差 ≤ task.accept_final_z

产物写入 runs/m0_hover/：metrics.json / env.json / trace.csv
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
PKG_ROOT = HERE.parents[1]          # projects/uav-inspect
REPO_ROOT = HERE.parents[2]         # 仓库根
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                       # noqa: E402
from uavinspect.control.cascaded import CascadedController  # noqa: E402
from uavinspect.sim.world import World                   # noqa: E402
from uavinspect.util.latency import StageTimer           # noqa: E402

HEADER = ["t", "x", "y", "z", "vx", "vy", "vz", "tilt_deg", "err", "thrust"]


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except Exception:
        return None


def _dependency_versions() -> dict[str, str | None]:
    import pybullet

    vers = {"python": platform.python_version(), "numpy": np.__version__,
            "pybullet": pybullet.getAPIVersion()}
    try:
        import yaml

        vers["pyyaml"] = yaml.__version__
    except Exception:
        vers["pyyaml"] = None
    return vers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M0 单机定点悬停")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "m0_hover"))
    ap.add_argument("--seed", type=int, default=None, help="覆盖配置里的 seed")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    seed = cfg.seed if args.seed is None else int(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = np.array([0.0, 0.0, cfg.task.hover_height])
    dt = cfg.sim.control_dt
    n_steps = int(round(cfg.task.duration / dt))
    timer = StageTimer()
    rows: list[list[float]] = []

    wall0 = time.perf_counter()
    with World(cfg) as world:
        quad = world.add_drone("uav0", (0.0, 0.0, 0.2))
        ctrl = CascadedController(cfg, quad)

        for _ in range(n_steps):
            t = world.t

            t0 = time.perf_counter()
            st = quad.state()
            timer.record("sense", time.perf_counter() - t0)

            t0 = time.perf_counter()
            thrust, torque = ctrl.compute(st, ref)
            timer.record("control", time.perf_counter() - t0)

            t0 = time.perf_counter()
            quad.set_wrench(thrust, torque)
            timer.record("command", time.perf_counter() - t0)

            t0 = time.perf_counter()
            world.advance(dt)
            timer.record("physics", time.perf_counter() - t0)

            st2 = quad.state()
            rows.append([
                t, *st2.pos.tolist(), *st2.vel.tolist(), float(np.rad2deg(st2.tilt)),
                float(np.linalg.norm(st2.pos - ref)), float(thrust),
            ])
        sim_time = world.t
        steps = world.steps
        thrust_ceiling = quad.max_thrust

    wall = time.perf_counter() - wall0
    arr = np.asarray(rows, dtype=float)
    t = arr[:, 0]
    win = t >= cfg.task.settle_time
    err = arr[win, 8]
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(err.max())
    p95_err = float(np.percentile(err, 95))
    final_z = float(arr[-1, 3])
    min_z = float(arr[:, 3].min())
    max_tilt_deg = float(arr[:, 7].max())
    max_thrust = float(arr[:, 9].max())

    checks = {
        "rmse_ok": rmse <= cfg.task.accept_rmse,
        "no_crash": min_z >= 0.05,
        "final_z_ok": abs(final_z - cfg.task.hover_height) <= cfg.task.accept_final_z,
    }
    passed = all(checks.values())

    metrics = {
        "task": "m0_hover",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "rmse_m": rmse, "max_err_m": max_err, "p95_err_m": p95_err,
            "final_z_m": final_z, "min_z_m": min_z,
            "max_tilt_deg": max_tilt_deg, "max_thrust_n": max_thrust,
            "max_thrust_ratio": max_thrust / thrust_ceiling,
        },
        "latency": timer.summary(),
        "sim": {
            "sim_time_s": sim_time, "wall_time_s": wall,
            "real_time_factor": sim_time / wall if wall > 0 else None,
            "physics_steps": steps, "control_steps": n_steps,
        },
        "task_cfg": {
            "hover_height": cfg.task.hover_height,
            "accept_rmse": cfg.task.accept_rmse,
            "settle_time": cfg.task.settle_time,
        },
        "seed": seed,
    }
    env = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "deps": _dependency_versions(),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "config": cfg.to_dict(),
    }

    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "trace.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(HEADER) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")

    print(f"M0 单机定点悬停  seed={seed}")
    print(f"  目标高度      {cfg.task.hover_height:.2f} m，时长 {sim_time:.1f} s，"
          f"评测窗口 t ≥ {cfg.task.settle_time:.1f} s")
    print(f"  RMSE          {rmse:.4f} m   （阈值 {cfg.task.accept_rmse:.2f}）")
    print(f"  P95 位置误差  {p95_err:.4f} m   最大 {max_err:.4f} m")
    print(f"  末态高度      {final_z:.4f} m   （|偏差| ≤ {cfg.task.accept_final_z:.2f}）")
    print(f"  最低高度      {min_z:.4f} m   最大倾角 {max_tilt_deg:.2f}°")
    print(f"  实时因子      {sim_time / wall:.2f}x（{steps} 物理步 / {n_steps} 控制步）")
    print()
    print("阶段耗时（毫秒）")
    print(timer.table())
    print()
    verdict = "通过" if passed else "未通过"
    print(f"判定：{verdict}  {checks}")
    print(f"产物：{out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
