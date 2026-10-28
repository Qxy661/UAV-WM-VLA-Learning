"""E5 控制进阶：指令延迟下的闭环裕度。

仿真里最容易骗自己的一条：控制周期 20 ms，于是默认"指令立刻生效"。真实链路上，
视觉推理、通信、调度加起来是 100–300 ms。这一节把延迟显式插进回路，量三件事：

1. 悬停还能不能稳住（RMSE、最大倾角）
2. 阶跃响应的超调与稳定时间怎么变
3. 到多少毫秒开始发散

延迟按控制周期整数倍注入（只延迟指令，不延迟状态），这是最有利于控制器的假设——
真实系统里状态也是旧的。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load  # noqa: E402
from uavinspect.control.cascaded import CascadedController  # noqa: E402
from uavinspect.sim.world import World  # noqa: E402

HOVER = 1.5
STEP_TO = 2.2
STEP_X = 1.2


def run(cfg, delay_steps: int, *, mode: str = "hover", duration: float = 20.0) -> dict:
    dt = cfg.sim.control_dt
    n = int(round(duration / dt))
    buf: collections.deque = collections.deque()
    idle = (float(cfg.drone.mass) * 9.81, np.zeros(3))
    errs, tilts, zs, xs_all = [], [], [], []
    with World(cfg, backend="rigid") as world:
        quad = world.add_drone("uav0", (0.0, 0.0, 0.2))
        ctrl = CascadedController(cfg, quad)
        for k in range(n):
            t = k * dt
            if mode == "hover":
                ref = np.array([0.0, 0.0, HOVER])
            elif mode == "step":
                ref = np.array([0.0, 0.0, HOVER if t < 5.0 else STEP_TO])
            else:
                ref = np.array([0.0 if t < 5.0 else STEP_X, 0.0, HOVER])
            st = quad.state()
            thrust, torque = ctrl.compute(st, ref)
            if delay_steps <= 0:
                quad.set_wrench(thrust, torque)
            else:
                # 队列里最多压 delay_steps 条；超过才把最老的一条真正下发
                buf.append((thrust, torque))
                if len(buf) > delay_steps:
                    quad.set_wrench(*buf.popleft())
                else:
                    quad.set_wrench(*idle)
            world.advance(dt)
            st = quad.state()
            errs.append(float(np.linalg.norm(st.pos - ref)))
            tilts.append(float(np.rad2deg(st.tilt)))
            xs_all.append(float(st.pos[0]))
            zs.append(float(st.pos[2]))
    e = np.array(errs)
    z = np.array(zs)
    out = {
        "delay_ms": int(round(delay_steps * cfg.sim.control_dt * 1000)),
        "rmse_m": round(float(np.sqrt(np.mean(e ** 2))), 4),
        "final_err_m": round(float(e[-1]), 4),
        "max_tilt_deg": round(float(np.max(tilts)), 2),
        "min_z_m": round(float(z.min()), 3),
        "max_z_m": round(float(z.max()), 3),
        "diverged": bool(np.max(np.abs(z)) > 20.0 or not np.isfinite(z).all()),
        "z_end_m": round(float(np.mean(z[-10:])), 3),
        "unstable": bool(np.max(tilts) > 80.0 or float(np.mean(z[-10:])) < 1.0),
    }
    if mode in ("step", "lateral"):
        # 需要横向位移才能看倾角，所以要记 x
        pass
    if mode == "step":
        settle = np.nonzero(np.abs(z - STEP_TO) > 0.05 * (STEP_TO - HOVER))[0]
        out["overshoot_pct"] = round(100.0 * (z.max() - STEP_TO) / (STEP_TO - HOVER), 1)
        out["settle_s"] = round(float((settle[-1] + 1) * cfg.sim.control_dt) if settle.size else 0.0, 2)
    if mode == "lateral":
        xs = np.array(xs_all)
        settle = np.nonzero(np.abs(xs - STEP_X) > 0.05 * STEP_X)[0]
        out["overshoot_pct"] = round(100.0 * (xs.max() - STEP_X) / STEP_X, 1)
        out["settle_s"] = round(float((settle[-1] + 1) * cfg.sim.control_dt) if settle.size else 0.0, 2)
        out["max_tilt_deg"] = round(float(np.max(tilts)), 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E5 延迟裕度")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "e5_control"))
    a = ap.parse_args(argv)
    cfg = load(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    delays = [0, 1, 2, 4, 6, 8, 10, 14]
    hover = [run(cfg, d, mode="hover") for d in delays]
    step = [run(cfg, d, mode="step") for d in delays]
    lat = [run(cfg, d, mode="lateral") for d in delays]
    res = {"control_dt_ms": round(cfg.sim.control_dt * 1000, 1), "hover": hover, "step": step, "lateral": lat}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'delay_ms':>8} {'hover_rmse':>10} {'hover_tilt':>10} {'lat_tilt':>8} {'lat_over':>9} {'lat_settle':>10} {'lat_unstable':>12}")
    for h, s in zip(hover, lat):
        print(f"{h['delay_ms']:>8} {h['rmse_m']:>10.4f} {h['max_tilt_deg']:>10.1f} {s['max_tilt_deg']:>8.1f} "
              f"{s.get('overshoot_pct', float('nan')):>9.1f} {s.get('settle_s', float('nan')):>10.2f} {s['unstable']!s:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
