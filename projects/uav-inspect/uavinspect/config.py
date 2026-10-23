"""配置加载与校验。

为什么单独成文件：主库 demo 把常数写在模块级（`OMEGA_MAX = 5.0`），
外部无法覆盖、也无法在加载时发现矛盾。这里把"坏配置"变成"加载即失败"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置不合法。取值范围问题统一用这个异常，便于测试断言。"""


@dataclass(frozen=True)
class RotorCfg:
    w_hover: float
    w_max: float
    time_constant: float
    torque_ratio: float


@dataclass(frozen=True)
class DroneCfg:
    mass: float
    arm: float
    collision_radius: float
    drag_quadratic: float
    rotor: RotorCfg


@dataclass(frozen=True)
class SimCfg:
    physics_dt: float
    control_dt: float
    policy_dt: float
    camera_dt: float
    gravity: float
    real_time: bool


@dataclass(frozen=True)
class ControlCfg:
    pos_kp: float
    pos_kd: float
    att_kp: float
    rate_kp: float
    max_tilt_deg: float
    thrust_margin: float


@dataclass(frozen=True)
class TaskCfg:
    hover_height: float
    duration: float
    settle_time: float
    accept_rmse: float
    accept_final_z: float


@dataclass(frozen=True)
class Config:
    seed: int
    sim: SimCfg
    drone: DroneCfg
    control: ControlCfg
    task: TaskCfg
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source", None)
        return d


def _num(section: str, key: str, raw: Any, lo: float, hi: float) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ConfigError(f"{section}.{key} 必须是数字，收到 {raw!r}")
    v = float(raw)
    if not math.isfinite(v):
        raise ConfigError(f"{section}.{key} 必须是有限数，收到 {v}")
    if not (lo <= v <= hi):
        raise ConfigError(f"{section}.{key} 应在 [{lo}, {hi}]，收到 {v}")
    return v


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    sec = data.get(name)
    if not isinstance(sec, dict):
        raise ConfigError(f"缺少配置段 {name!r}")
    return sec


def _need(sec: dict[str, Any], section: str, key: str) -> Any:
    if key not in sec:
        raise ConfigError(f"缺少配置项 {section}.{key}")
    return sec[key]


def parse(data: dict[str, Any], source: str = "") -> Config:
    sim = _section(data, "sim")
    physics_dt = _num("sim", "physics_dt", _need(sim, "sim", "physics_dt"), 1e-4, 0.02)
    control_dt = _num("sim", "control_dt", _need(sim, "sim", "control_dt"), 1e-3, 0.1)
    # 频率必须整倍嵌套，否则"四层解耦"只是口号，实际会漂相位。
    ratio = control_dt / physics_dt
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
        raise ConfigError(
            f"sim.control_dt 必须是 sim.physics_dt 的整数倍，当前比值为 {ratio:.6f}"
        )
    sim_cfg = SimCfg(
        physics_dt=physics_dt,
        control_dt=control_dt,
        policy_dt=_num("sim", "policy_dt", _need(sim, "sim", "policy_dt"), 1e-2, 2.0),
        camera_dt=_num("sim", "camera_dt", _need(sim, "sim", "camera_dt"), 1e-2, 5.0),
        gravity=_num("sim", "gravity", _need(sim, "sim", "gravity"), -30.0, -1.0),
        real_time=bool(sim.get("real_time", False)),
    )

    drone = _section(data, "drone")
    rotor = drone.get("rotor")
    if not isinstance(rotor, dict):
        raise ConfigError("缺少配置段 drone.rotor")
    drone_cfg = DroneCfg(
        mass=_num("drone", "mass", _need(drone, "drone", "mass"), 0.05, 20.0),
        arm=_num("drone", "arm", _need(drone, "drone", "arm"), 0.02, 1.0),
        collision_radius=_num(
            "drone", "collision_radius", drone.get("collision_radius", 0.08), 0.01, 0.5
        ),
        drag_quadratic=_num(
            "drone", "drag_quadratic", drone.get("drag_quadratic", 0.0), 0.0, 10.0
        ),
        rotor=RotorCfg(
            w_hover=_num("drone.rotor", "w_hover", _need(rotor, "drone.rotor", "w_hover"), 100.0, 5000.0),
            w_max=_num("drone.rotor", "w_max", _need(rotor, "drone.rotor", "w_max"), 200.0, 8000.0),
            time_constant=_num(
                "drone.rotor", "time_constant", _need(rotor, "drone.rotor", "time_constant"), 1e-3, 0.5
            ),
            torque_ratio=_num(
                "drone.rotor", "torque_ratio", _need(rotor, "drone.rotor", "torque_ratio"), 0.0, 0.2
            ),
        ),
    )
    if drone_cfg.rotor.w_max <= drone_cfg.rotor.w_hover:
        raise ConfigError("drone.rotor.w_max 必须大于 w_hover")
    # 悬停时推力必须占满整车重力，否则起飞即坠。
    kf = drone_cfg.mass * abs(sim_cfg.gravity) / (4.0 * drone_cfg.rotor.w_hover**2)
    if kf <= 0:
        raise ConfigError("kf 反解为非正数，检查 mass / w_hover")

    control = _section(data, "control")
    control_cfg = ControlCfg(
        pos_kp=_num("control", "pos_kp", _need(control, "control", "pos_kp"), 1e-3, 100.0),
        pos_kd=_num("control", "pos_kd", _need(control, "control", "pos_kd"), 1e-3, 200.0),
        att_kp=_num("control", "att_kp", _need(control, "control", "att_kp"), 1e-3, 100.0),
        rate_kp=_num("control", "rate_kp", _need(control, "control", "rate_kp"), 1e-3, 200.0),
        max_tilt_deg=_num("control", "max_tilt_deg", _need(control, "control", "max_tilt_deg"), 1.0, 80.0),
        thrust_margin=_num("control", "thrust_margin", _need(control, "control", "thrust_margin"), 0.1, 1.0),
    )

    task = _section(data, "task")
    task_cfg = TaskCfg(
        hover_height=_num("task", "hover_height", _need(task, "task", "hover_height"), 0.1, 50.0),
        duration=_num("task", "duration", _need(task, "task", "duration"), 0.1, 3600.0),
        settle_time=_num("task", "settle_time", _need(task, "task", "settle_time"), 0.0, 3600.0),
        accept_rmse=_num("task", "accept_rmse", _need(task, "task", "accept_rmse"), 0.0, 10.0),
        accept_final_z=_num("task", "accept_final_z", task.get("accept_final_z", 0.15), 0.0, 10.0),
    )
    if task_cfg.settle_time >= task_cfg.duration:
        raise ConfigError("task.settle_time 必须小于 task.duration，否则没有任何评测窗口")

    seed = data.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError(f"seed 必须是非负整数，收到 {seed!r}")
    return Config(seed=seed, sim=sim_cfg, drone=drone_cfg, control=control_cfg, task=task_cfg, source=source)


def load(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是映射：{path}")
    return parse(raw, source=str(path))
