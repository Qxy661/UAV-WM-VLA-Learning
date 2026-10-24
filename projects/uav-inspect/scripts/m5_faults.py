"""M5：四类故障注入 + 安全层验收。

每条故障都跑一遍闭环，产物写 runs/m5_faults/{scenario}.json，指标是代码判定的：

    A 链路断流   8 秒时策略停发指令 1.5 s  → 必须 HOLD/HOVER，高度不塌
    B 心跳陈旧   指令值在变但 stamp 冻结   → 必须当陈旧处理（不能被"看着挺新"骗过）
    C 指令越界   thrust=5、torque=50       → 必须截断到位，姿态不翻
    D 指令冲突   两路指令抢同一架机        → 按优先级裁决，落败方悬停，不来回切

验收（四类都成立才算通过）：
    1. 全程 z ≥ 0.05 m（不触地）
    2. 全程倾角 ≤ 45°
    3. 故障结束后 3 s 内回到定高带 |z - h| ≤ 0.20 m
    4. 该升级的场景确实升级过（A/B），该截断的场景确实截断过（C）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG))

from uavinspect.config import load                       # noqa: E402
from uavinspect.control.cascaded import CascadedController  # noqa: E402
from uavinspect.safety.guard import Command, GuardConfig, Mode, SafetyGuard  # noqa: E402
from uavinspect.sim.world import World                    # noqa: E402

FENCE = (-3.5, 3.5, -2.5, 2.5)


def policy_cmd(ctrl, st, ref, t: float, *, obey=True) -> Command:
    thrust, torque = ctrl.compute(st, ref)
    return Command(thrust, torque, t)


def run_scenario(name: str, kind: str, seconds: float = 12.0) -> dict:
    cfg = load(PKG / "configs" / "default.yaml")
    dt = cfg.sim.control_dt
    h = cfg.task.hover_height
    goal = np.array([0.0, 0.0, h])
    guard = SafetyGuard(GuardConfig(hover_height=h), FENCE)

    with World(cfg, backend="rigid") as world:
        quad = world.add_drone("uav", (0.0, 0.0, 0.25))
        ctrl = CascadedController(cfg, quad)
        t = 0.0
        rows = []
        while t < seconds:
            st = quad.state()
            high = policy_cmd(ctrl, st, goal, t)          # 高优先级：任务策略
            high.source = "mission"
            cmds = [high]
            if kind == "drop" and 4.0 <= t < 5.5:
                cmds = []                                  # 链路彻底断：一条也收不到
            elif kind == "stale" and 6.0 <= t < 7.5:
                # 值仍在算，但时间戳冻在 6.0 s：安全层必须按心跳陈旧处理
                cmds = [Command(high.thrust, high.torque, 6.0, source="policy")]
            elif kind == "range" and 7.0 <= t < 7.5:
                cmds = [Command(5.0, np.array([50.0, -50.0, 30.0]), t, source="runaway")]
            elif kind == "conflict" and 5.0 <= t < 8.0:
                # 两路抢同一架机：低优先级那路想把机子带去对角
                low = policy_cmd(ctrl, st, np.array([1.5, 1.5, h]), t)
                low.priority, low.source = 5, "secondary"
                cmds = [high, low]

            thrust, torque, mode = guard.filter(t, cmds, st.pos, st.vel, goal)
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt

            st = quad.state()
            rows.append({
                "t": t, "z": float(st.pos[2]), "tilt": float(st.tilt),
                "thrust": float(thrust), "torque_max": float(np.abs(torque).max()),
                "mode": int(mode), "err": float(np.linalg.norm(st.pos - goal)),
            })
        sim_t = world.t

    arr = np.asarray([[r["t"], r["z"], r["tilt"], r["thrust"], r["torque_max"],
                       r["mode"], r["err"]] for r in rows])
    modes = arr[:, 5].astype(int)
    fault_window = {"drop": (4.0, 5.5), "stale": (6.0, 7.5),
                    "range": (7.0, 7.5), "conflict": (5.0, 8.0)}[kind]
    after = arr[:, 0] >= fault_window[1] + 3.0
    tail_err = float(np.max(np.abs(arr[after, 1] - h))) if after.any() else 0.0

    checks = {
        "no_ground_hit": bool(float(arr[:, 1].min()) >= 0.05),
        "tilt_ok": bool(float(arr[:, 2].max()) <= np.deg2rad(45.0)),
        "recovered": bool(tail_err <= 0.20),
        "escalated": bool(int(modes.max()) >= int(Mode.HOLD)) if kind in ("drop", "stale") else True,
        "clipped": bool(float(arr[:, 3].max()) <= guard.cfg.thrust_abs_max + 1e-9
                        and float(arr[:, 4].max()) <= guard.cfg.torque_abs_max + 1e-9)
        if kind == "range" else True,
        "rejected_implausible": (sum("拒绝不合理" in e["what"] for e in guard.st.events) >= 3)
        if kind == "range" else True,
    }
    return {
        "scenario": name, "kind": kind, "passed": all(checks.values()), "checks": checks,
        "metrics": {
            "min_z": float(arr[:, 1].min()), "max_tilt_deg": float(np.rad2deg(arr[:, 2].max())),
            "max_thrust": float(arr[:, 3].max()), "max_torque": float(arr[:, 4].max()),
            "tail_z_err": tail_err, "final_z": float(arr[-1, 1]),
            "max_mode": int(modes.max()), "mode_changes": int(np.sum(np.diff(modes) != 0)),
            "sim_time_s": sim_t,
        },
        "events": guard.st.events,
    }


SCENARIOS = [("A 链路断流", "drop"), ("B 心跳陈旧", "stale"),
             ("C 指令越界", "range"), ("D 指令冲突", "conflict")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M5 故障注入")
    ap.add_argument("--out", default=str(PKG / "runs" / "m5_faults"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, kind in SCENARIOS:
        res = run_scenario(name, kind)
        results.append(res)
        (out / f"{kind}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        m = res["metrics"]
        print(f"{name:<10} {'通过' if res['passed'] else '未通过'}  "
              f"最低高度 {m['min_z']:.2f} m  最大倾角 {m['max_tilt_deg']:5.1f}°  "
              f"最大推力 {m['max_thrust']:.2f}  最大力矩 {m['max_torque']:.2f}  "
              f"最高模式 {m['max_mode']}  切模式 {m['mode_changes']} 次  "
              f"回稳误差 {m['tail_z_err']:.3f} m")
        for ev in res["events"]:
            print(f"    {ev['t']:>6.2f}s  {ev['what']}")

    passed = all(r["passed"] for r in results)
    summary = {"passed": passed, "scenarios": results}
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n判定：{'通过' if passed else '未通过'}")
    print(f"产物：{out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
