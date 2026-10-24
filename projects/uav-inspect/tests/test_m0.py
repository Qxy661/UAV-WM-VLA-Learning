"""契约测试：这些测试失败，说明"工程层"的某个承诺被破坏了。

不是"覆盖率好看"，而是把 README §2 的四条硬约束变成可执行断言。
两个后端都跑：`rigid` 永远在，`pybullet` 只有在本机装得上时才加入参数化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from uavinspect.config import ConfigError, load, parse  # noqa: E402
from uavinspect.control.cascaded import CascadedController  # noqa: E402
from uavinspect.sim.world import World, pybullet_available  # noqa: E402

CFG_PATH = PKG / "configs" / "default.yaml"
BACKENDS = ["rigid"] + (["pybullet"] if pybullet_available() else [])


@pytest.fixture(scope="module")
def cfg():
    return load(CFG_PATH)


@pytest.fixture(params=BACKENDS)
def world(cfg, request):
    w = World(cfg, backend=request.param)
    yield w
    w.close()


# ---------------------------------------------------------------- 配置契约
def test_config_is_valid_and_frequencies_nest(cfg):
    assert cfg.sim.control_dt % cfg.sim.physics_dt < 1e-9
    assert cfg.sim.policy_dt >= cfg.sim.control_dt


def test_config_rejects_non_multiple_control_dt(cfg):
    data = cfg.to_dict()
    data["sim"]["control_dt"] = 0.017
    with pytest.raises(ConfigError, match="整数倍"):
        parse(data)


def test_config_rejects_negative_mass(cfg):
    data = cfg.to_dict()
    data["drone"]["mass"] = -1.0
    with pytest.raises(ConfigError, match="drone.mass"):
        parse(data)


def test_config_rejects_wmax_below_hover(cfg):
    data = cfg.to_dict()
    data["drone"]["rotor"]["w_max"] = 1000.0
    with pytest.raises(ConfigError, match="w_max"):
        parse(data)


# ---------------------------------------------------------------- 混控与电机
def test_mixer_roundtrip(world):
    """期望扳手 → 转速 → 扳手必须回到原值，否则混控矩阵与其逆不一致。"""
    q = world.add_drone("uav0", (0, 0, 1.0))
    thrust = 0.55 * q.max_thrust
    torque = np.array([0.01, -0.02, 0.005])
    t_back, tau_back = q.mixer_roundtrip(thrust, torque)
    assert t_back == pytest.approx(thrust, rel=1e-9)
    np.testing.assert_allclose(tau_back, torque, rtol=1e-9, atol=1e-12)


def test_motor_lag_converges(world, cfg):
    q = world.add_drone("uav0", (0, 0, 1.0))
    q.set_wrench(0.8 * q.max_thrust, np.zeros(3))
    for _ in range(int(1.0 / cfg.sim.physics_dt)):
        q.motor_step(cfg.sim.physics_dt)
    assert np.allclose(q.w, q.w_des, rtol=1e-2)


# ---------------------------------------------------------------- 控制符号
def test_control_sign_is_correct(world, cfg):
    """参考点在 +y：必须产生负 roll 力矩，否则正反馈直接翻机。"""
    q = world.add_drone("uav0", (0, 0, 1.5))
    ctrl = CascadedController(cfg, q)
    _, torque = ctrl.compute(q.state(), np.array([0.0, 0.5, 1.5]))
    assert torque[0] < 0.0


# ---------------------------------------------------------------- 世界契约
def test_world_rejects_bad_advance(world, cfg):
    world.add_drone("uav0", (0, 0, 1.0))
    with pytest.raises(ValueError, match="整数倍"):
        world.advance(cfg.sim.physics_dt * 0.5)


def test_hover_is_stable(world, cfg):
    """6 秒定点悬停，最后 1 秒 RMSE < 0.30 m。比 M0 验收阈值宽，用于快速反馈。"""
    q = world.add_drone("uav0", (0, 0, 0.2))
    ctrl = CascadedController(cfg, q)
    ref = np.array([0.0, 0.0, cfg.task.hover_height])
    errs = []
    n = int(round(6.0 / cfg.sim.control_dt))
    for _ in range(n):
        st = q.state()
        thrust, torque = ctrl.compute(st, ref)
        q.set_wrench(thrust, torque)
        world.advance(cfg.sim.control_dt)
        if world.t >= 5.0:
            errs.append(float(np.linalg.norm(q.state().pos - ref)))
    assert len(errs) > 10
    assert np.sqrt(np.mean(np.square(errs))) < 0.30


# ---------------------------------------------------------------- 接触（仅 rigid）
def _rigid_world(cfg):
    return World(cfg, backend="rigid")


def test_rigid_ground_stops_fall(cfg):
    """从 1.5 m 自由落体，落到地面后不穿地、也不用弹起来反复跳。"""
    with _rigid_world(cfg) as w:
        q = w.add_drone("uav0", (0, 0, 1.5))
        q.set_wrench(0.0, np.zeros(3))       # 电机停转
        for _ in range(int(4.0 / cfg.sim.control_dt)):
            w.advance(cfg.sim.control_dt)
        st = q.state()
        assert st.pos[2] >= cfg.drone.collision_radius - 1e-3
        assert abs(st.vel[2]) < 0.5


def test_rigid_drones_separate(cfg):
    """两架机放到几乎同一个点，一步之后不能还重叠。"""
    with _rigid_world(cfg) as w:
        a = w.add_drone("a", (0, 0, 1.0))
        b = w.add_drone("b", (0.01, 0, 1.0))
        w.advance(cfg.sim.physics_dt)
        r = 2 * cfg.drone.collision_radius
        assert float(np.linalg.norm(a.state().pos - b.state().pos)) >= r - 1e-3


def test_rigid_box_blocks(world, cfg):
    """正前方挡一堵墙，满推力朝墙飞 2 秒，机体会停在墙前而不是穿过去。"""
    if getattr(world, "backend", "") != "rigid":
        pytest.skip("该用例只验证自研后端的接触解算")
    q = world.add_drone("uav0", (0, 0, 1.0))
    world.add_box((0.05, 2.0, 2.0), (1.0, 0.0, 1.0))
    q.set_wrench(0.0, np.zeros(3))
    q.pos[0] = 0.5
    for _ in range(int(2.0 / cfg.sim.control_dt)):
        world.advance(cfg.sim.control_dt)
        # 一直往 +x 推，模拟撞墙后仍被指令推着
        q.vel[0] = 1.0
    assert q.pos[0] <= 1.0 - 0.05 - cfg.drone.collision_radius + 1e-3
