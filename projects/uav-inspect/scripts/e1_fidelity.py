"""E1 仿真保真度：积分步长收敛 + 传感器效应对输入分布的影响。

两件事分开量：

1. **数值收敛**：物理步长 dt 减半时，同一控制律的 20 s 悬停轨迹差多少。差得多，说明
   报告里写的实时因子和 RMSE 都是"这一档步长特有的"，换个步长就变。
2. **画质链路**：曝光/噪声/模糊/掉帧加进去之后，E9 的亮度 PSI 会不会响。这条链不通，
   "输入漂移报警"在真实相机上永远不会触发。
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
from uavinspect.mlops import drift  # noqa: E402
from uavinspect.sim import sensors  # noqa: E402
from uavinspect.sim.world import World  # noqa: E402

ROOT = PKG_ROOT
DURATION = 20.0


def fly(dt: float, control_dt: float = 0.02) -> dict:
    """固定控制周期 0.02 s，只变物理步长——否则比的是"控制器跑多快"而不是积分精度。"""
    every = int(round(control_dt / dt))
    data = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    data["sim"]["physics_dt"] = float(dt)
    data["sim"]["control_dt"] = float(dt) * every
    cfg = ucfg.parse(data, source="e1")
    ref = np.array([0.0, 0.0, cfg.task.hover_height])
    n_steps = int(round(DURATION / cfg.sim.control_dt))
    errs, zs, omegas = [], [], []
    with World(cfg, backend="rigid") as world:
        quad = world.add_drone("uav0", (0.0, 0.0, 0.2))
        ctrl = CascadedController(cfg, quad)
        for _ in range(n_steps):
            st = quad.state()
            thrust, torque = ctrl.compute(st, ref)
            quad.set_wrench(thrust, torque)
            world.advance(cfg.sim.control_dt)
            st = quad.state()
            errs.append(float(np.linalg.norm(st.pos - ref)))
            zs.append(float(st.pos[2]))
            omegas.append(float(np.linalg.norm(st.omega)))
    return {"physics_dt": round(dt, 5), "control_dt": round(dt * every, 5),
            "control_every": every,
            "control_steps": n_steps,
            "rmse_m": round(float(np.sqrt(np.mean(np.square(errs)))), 5),
            "final_err_m": round(errs[-1], 5),
            "min_z_m": round(float(np.min(zs)), 4),
            "omega_p95": round(float(np.percentile(omegas, 95)), 4)}


def make_scene_images(n: int = 200, size: int = 96, seed: int = 3) -> np.ndarray:
    """造一批"仓库感"图像：渐变灰底 + 红色块 + 轻微噪声，用来算亮度分布。"""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.75, 0.35, size)[None, :] * np.linspace(0.9, 0.6, size)[:, None]
    out = np.zeros((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        img = np.repeat(base[:, :, None], 3, axis=2)
        x, y = int(rng.integers(0, size - 24)), int(rng.integers(0, size - 24))
        img[y:y + 16, x:x + 20] = [0.95, 0.16, 0.12]
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
        out[i] = (img * 255).round().astype(np.uint8)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E1 仿真保真度")
    ap.add_argument("--out", default=str(ROOT / "runs" / "e1_fidelity"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    conv = [fly(dt) for dt in (0.02, 0.01, 0.005, 0.0025)]
    ref_rmse = conv[-1]["rmse_m"]
    for row in conv:
        row["rmse_vs_finest_pct"] = round(100.0 * (row["rmse_m"] - ref_rmse) / ref_rmse, 2) if ref_rmse else None

    clean = make_scene_images()
    ref_b = drift.image_brightness(clean)
    cases = {
        "clean": sensors.SensorConfig(),
        "exposure+20%": sensors.SensorConfig(gain=1.2),
        "exposure-25%": sensors.SensorConfig(gain=0.75),
        "noise_0.03": sensors.SensorConfig(noise_sigma=0.03),
        "noise_0.10": sensors.SensorConfig(noise_sigma=0.10),
        "blur_5": sensors.SensorConfig(blur_k=5),
        "dropout_30%": sensors.SensorConfig(dropout_p=0.30),
    }
    sensor_rows = {}
    for k, cfg in cases.items():
        imgs = sensors.apply_sequence(clean, cfg, seed=7)
        sensor_rows[k] = drift.check(ref_b, drift.image_brightness(imgs))

    res = {"convergence": conv, "sensor_drift": sensor_rows,
           "units": {"rmse_m": "米（20 s 悬停，rigid 后端）", "psi": "无量纲"}}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
