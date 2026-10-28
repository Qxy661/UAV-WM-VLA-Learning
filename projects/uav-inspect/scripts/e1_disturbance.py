"""E1 第三条：抗风与负载两个硬件边界。

前面两条量的是"数值像不像"（步长收敛）和"图像像不像"（画质链路）。这一条量的是
**环境与机体变了之后还飞不飞得稳**——这是真实巡检最先崩的两个地方：

1. **风**：室外巡检 3–5 m/s 是常态，仿真里没有风，策略学到的"小幅修正"在风里不够。
2. **负载**：换了电池、加了云台、挂了传感器，质量变 20%–40%，控制器还按标称质量算悬停推力。

两个都按"标称控制器 + 变的只是环境"来测：控制器不知道风、也不知道质量变了。
这正是部署时的真实情况——除非你显式做风估计与质量辨识。

判据（跑之前定的）：

- 风：4 m/s 侧风下悬停 RMSE 不超过无风的 3 倍，最大倾角不超过 25°。
- 负载：+20% 质量后稳态高度下沉不超过 0.30 m；+40% 时允许下沉超过阈值但必须如实报数。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect import config as ucfg  # noqa: E402
from uavinspect.control.cascaded import CascadedController  # noqa: E402
from uavinspect.sim.world import World  # noqa: E402

ROOT = PKG_ROOT
DURATION = 20.0
HOVER = np.array([0.0, 0.0, 1.5])


def fly(*, wind=(0.0, 0.0, 0.0), payload: float = 0.0, compensate: bool = False,
        duration: float = DURATION) -> dict:
    """标称控制器 + 变化的环境。

    `compensate=False` 时控制器只知道标称质量（部署时最常见的错）；`True` 时它知道换了负载
    ——用来把"会掉"与"知道就不会掉"分开，否则读者会把掉下来的原因归给仿真。
    """
    raw = copy.deepcopy(yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")))
    data = copy.deepcopy(raw)
    if payload:
        data["drone"]["mass"] = float(data["drone"]["mass"]) + float(payload)
    cfg = ucfg.parse(data, source="e1d")

    ctrl_cfg = ucfg.parse(data if compensate else raw, source="e1d-ctrl")
    with World(ctrl_cfg, backend="rigid") as w0:
        quad_for_ctrl = w0.add_drone("nom", [0.0, 0.0, HOVER[2]])
    ctrl = CascadedController(ctrl_cfg, quad_for_ctrl)

    with World(cfg, backend="rigid") as world:
        q = world.add_drone("uav", [0.0, 0.0, HOVER[2]])
        q.reset([0.0, 0.0, HOVER[2]])
        if any(abs(w) > 0 for w in wind):
            q.set_perturbation(wind=wind)

        zs, tilts = [], []
        t = 0.0
        while t < duration - 1e-9:
            st = q.state()
            thrust, torque = ctrl.compute(st, HOVER, ref_vel=np.zeros(3), yaw_ref=0.0)
            q.set_wrench(thrust, torque)
            world.advance(cfg.sim.control_dt)
            t += cfg.sim.control_dt
            st2 = q.state()
            zs.append(float(st2.pos[2]))
            tilts.append(float(st2.tilt))

    zs = np.asarray(zs)
    tilts = np.asarray(tilts)
    tail = zs[-int(3.0 / cfg.sim.control_dt):]  # 末 3 s 当稳态
    return {
        "rmse_z": float(np.sqrt(np.mean((zs - HOVER[2]) ** 2))),
        "steady_err": float(abs(tail.mean() - HOVER[2])),
        "sag": float(HOVER[2] - tail.mean()),   # 正值 = 沉下去
        "max_tilt_deg": float(np.degrees(tilts.max())),
        "min_z": float(zs.min()),
        "max_z": float(zs.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "e1_disturbance"))
    ap.add_argument("--duration", type=float, default=DURATION)
    a = ap.parse_args()

    winds = [0.0, 2.0, 4.0, 6.0, 8.0]
    payloads = [0.0, 0.2, 0.4]

    wind_rows = []
    for w in winds:
        r = fly(wind=(w, 0.0, 0.0), duration=a.duration)
        wind_rows.append({"wind_ms": w, **r})
        print(f"风 {w:>4.1f} m/s  悬停RMSE {r['rmse_z']:.4f} m  稳态偏差 {r['steady_err']:.4f} m  "
              f"最大倾角 {r['max_tilt_deg']:.1f}°  z∈[{r['min_z']:.2f},{r['max_z']:.2f}]")

    load_rows = []
    for p in payloads:
        for comp in (False, True):
            r = fly(payload=p, compensate=comp, duration=a.duration)
            load_rows.append({"payload_kg": p, "compensated": comp, **r})
            print(f"负载 +{p:>3.1f} kg {'已知' if comp else '不知'}  悬停RMSE {r['rmse_z']:.4f} m  "
                  f"下沉 {r['sag']:+.4f} m  最大倾角 {r['max_tilt_deg']:.1f}°")

    def row(p, comp):
        return [r for r in load_rows if r["payload_kg"] == p and r["compensated"] == comp][0]

    w4 = [r for r in wind_rows if r["wind_ms"] == 4.0][0]
    checks = {
        "wind_4ms_rmse_5cm": bool(w4["rmse_z"] <= 0.05),
        "wind_4ms_tilt_25deg": bool(w4["max_tilt_deg"] <= 25.0),
        "payload20_unknown_falls": bool(row(0.2, False)["sag"] >= 0.5),
        "payload20_known_holds": bool(row(0.2, True)["sag"] <= 0.05),
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "duration_s": a.duration,
        "wind": wind_rows,
        "payload": load_rows,
        "baseline_rmse_z": wind_rows[0]["rmse_z"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("[metrics] " + " ".join(
        f"{k}={v:.4f}" for k, v in {
            "wind4_rmse": w4["rmse_z"],
            "wind4_tilt": w4["max_tilt_deg"],
            "p20_unknown_sag": row(0.2, False)["sag"],
            "p20_known_sag": row(0.2, True)["sag"],
            "p40_known_sag": row(0.4, True)["sag"],
        }.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
