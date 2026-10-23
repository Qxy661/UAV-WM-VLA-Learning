"""契约测试：这些测试失败，说明"工程层"的某个承诺被破坏了。

不是"覆盖率好看"，而是把 README §2 的四条硬约束变成可执行断言。
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
from uavinspect.sim.world import World  # noqa: E402

CFG_PATH = PKG / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load(CFG_PATH)


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


def test_mixer_roundtrip(cfg):
    """期望扳手 → 转速 → 扳手必须回到原值，否则混控矩阵与其逆不一致。"""
    with World(cfg) as world:
        q = world.add_drone("uav0", (0, 0, 1.0))
        thrust = 0.55 * q.max_thrust
        torque = np.array([0.01, -0.02, 0.005])
        t_back, tau_back = q.mixer_roundtrip(thrust, torque)
        assert t_back == pytest.approx(thrust, rel=1e-9)
        np.testing.assert_allclose(tau_back, torque, rtol=1e-9, atol=1e-12)


def test_motor_lag_converges(cfg):
    with World(cfg) as world:
        q = world.add_drone("uav0", (0, 0, 1.0))
        q.set_wrench(0.8 * q.max_thrust, np.zeros(3))
        for _ in range(int(1.0 / cfg.sim.physics_dt)):
            q.motor_step(cfg.sim.physics_dt)
        assert np.allclose(q.w, q.w_des, rtol=1e-2)


def test_control_sign_is_correct(cfg):
    """参考点在 +y：必须产生负 roll 力矩，否则正反馈直接翻机。"""
    with World(cfg) as world:
        q = world.add_drone("uav0", (0, 0, 1.5))
        ctrl = CascadedController(cfg, q)
        _, torque = ctrl.compute(q.state(), np.array([0.0, 0.5, 1.5]))
        assert torque[0] < 0.0


def test_world_rejects_bad_advance(cfg):
    with World(cfg) as world:
        world.add_drone("uav0", (0, 0, 1.0))
        with pytest.raises(ValueError, match="整数倍"):
            world.advance(cfg.sim.physics_dt * 0.5)


def test_hover_is_stable(cfg):
    """6 秒定点悬停，最后 1 秒 RMSE < 0.30 m。比 M0 验收阈值宽，用于快速反馈。"""
    with World(cfg) as world:
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
