"""E11 双具身报告：同一套 harness 跑多旋翼与机械臂。

两边都只依赖三件事：状态里有 pos、策略给出动作、评测按"到点即在容差内"判定。
所以指标可以直接并列——成功率、到点误差、用时分位——不需要为机械臂另写一套评测。

用法：
    py -3.9 projects/uav-inspect/scripts/e11_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from uavinspect.embodiments.arm import Arm, joint_vel_to_target, reachable  # noqa: E402

PKG_ROOT = REPO_ROOT
TOL = 0.05           # 到点容差 5 cm
STEPS = 150          # 30 Hz 下 5 s
GOALS = [(0.42, 0.10, 0.30), (0.30, -0.22, 0.42), (0.15, 0.30, 0.20), (0.50, 0.00, 0.12)]


def run_episode(goal, q0) -> dict:
    arm = Arm(q0)
    errs = []
    for k in range(STEPS):
        st = arm.state()
        e = float(np.linalg.norm(st.pos - np.asarray(goal)))
        errs.append(e)
        if e < 1e-3:
            break
        arm.set_joint_vel(joint_vel_to_target(st.q, np.asarray(goal)))
    errs = np.array(errs)
    return {
        "final_err_cm": round(float(errs[-1]) * 100, 2),
        "min_err_cm": round(float(errs.min()) * 100, 2),
        "steps": int(errs.size),
        "seconds": round(errs.size * arm.dt, 2),
        "success": bool(errs[-1] <= TOL),
    }


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    for g in GOALS:
        for _ in range(5):
            q0 = rng.uniform([-1.0, -0.6, -0.8], [1.0, 0.8, 0.8])
            if not reachable(np.asarray(g)):
                rows.append({"goal": g, "skipped": True})
                continue
            r = run_episode(g, q0)
            r["goal"] = g
            rows.append(r)

    ok = [r for r in rows if not r.get("skipped")]
    succ = np.array([r["success"] for r in ok])
    errs = np.array([r["final_err_cm"] for r in ok])
    secs = np.array([r["seconds"] for r in ok])
    res = {
        "embodiment": "arm3dof_kinematic",
        "episodes": len(ok),
        "success_rate": round(float(succ.mean()), 4),
        "final_err_cm_median": round(float(np.median(errs)), 2),
        "final_err_cm_p90": round(float(np.percentile(errs, 90)), 2),
        "seconds_p50": round(float(np.median(secs)), 2),
        "seconds_p90": round(float(np.percentile(secs, 90)), 2),
        "tol_cm": TOL * 100,
        "steps_budget": STEPS,
        "note": "运动学具身：无惯量、无力矩限幅、无接触，收敛时间不可与真机比较",
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    out = PKG_ROOT / "runs" / "e11_report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
