"""T1 评测：让 π0.5 直接出 (thrust, torque)，按 9.6 Hz 分块下发给 48 Hz 的机体。

为什么是这个闭环，而不是让策略直接以 48 Hz 跑：
- 训练数据是按 9.6 Hz 录的（每 5 个控制步一帧），动作就是当时串级控制器给出的
  `[thrust_N, torque_Nm_x, torque_Nm_y, torque_Nm_z]`。策略学的是"看到这帧图，接下来
  这一小段时间该给什么推力与力矩"。
- 因此评测必须同频：每 1/9.6 s 问一次策略，把预测块里的动作逐个保持 5 个控制步
  （1/48 s × 5 = 1/9.6 s）。用 48 Hz 询问会立刻进入训练分布之外。

指标与 `uav-inspect` 的 M6 协议一致（成功率、覆盖率、最小间距、覆盖用时），
这样 scripted / BC / π0.5 三组数可以直接放在一张表里。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[2]        # projects/
sys.path.insert(0, str(PKG_ROOT / "uav-inspect"))

from uavinspect.config import load as load_config        # noqa: E402
from uavinspect.eval.harness import SPAWN, TaskSpec, EpisodeResult   # noqa: E402
from uavinspect.sim.camera import in_view, render        # noqa: E402
from uavinspect.sim.scene import build_warehouse, visible_viewpoints  # noqa: E402
from uavinspect.sim.world import World                   # noqa: E402

ACTION_HZ = 9.6
CONTROL_HZ = 48
IMG_SIZE = 96
HFOV = 70.0


def substeps() -> int:
    """一个预测动作占几个控制周期。

    训练目标是**逐控制步**的动作：obs.npz 里的 action 是 48 Hz 逐步记录的，
    BC 学的是 image(t) -> action(t..t+7) 这 8 个连续控制步。所以执行时必须
    一步一个地用掉，`VLA_HOLD=1` 是唯一正确的默认值。
    早前默认 5（48/9.6）等于把整段速度放慢 5 倍，策略永远飞不到位。
    """
    return int(os.environ.get("VLA_HOLD", "1"))


@dataclass
class StepOut:
    thrust: float
    torque: np.ndarray


def state_vector(st) -> np.ndarray:
    """与训练数据一致：pos(3) + vel(3) + quat(4) = 10 维。"""
    return np.concatenate([st.pos, st.vel, st.quat]).astype(np.float32)


class ChunkDriver:
    """把策略输出的动作块摊平成逐个控制周期可用的动作。"""

    def __init__(self, actions: np.ndarray, *, hold: int):
        self.actions = np.asarray(actions, dtype=np.float64)
        self.hold = hold
        self.i = 0

    def next(self) -> StepOut:
        idx = min(self.i // self.hold, len(self.actions) - 1)
        a = self.actions[idx]
        self.i += 1
        return StepOut(float(a[0]), np.asarray(a[1:4], dtype=np.float64))

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.actions) * self.hold


def run_vla_episode(policy, seed: int, instruction: str, *,
                    duration: float = 20.0,
                    cover_max_dist: float = 0.7,
                    img_size: int = IMG_SIZE) -> tuple[EpisodeResult, dict]:
    """跑一集；policy 需实现 `infer(image_uint8, state10, prompt) -> (H,4)`。"""
    cfg = load_config()
    hold = substeps()
    with World(cfg, backend="rigid") as world:
        scene = build_warehouse(world, seed=seed, with_target=True)
        quad = world.add_drone("uav", SPAWN)
        boxes = scene.boxes
        view = [vw for vw, ok in zip(scene.viewpoints, visible_viewpoints(scene)) if ok]
        covered = {lbl: False for _p, _y, lbl in view}

        t = 0.0
        path = 0.0
        energy = 0.0
        min_clear = float("inf")
        t_cover = float("inf")
        prev = quad.state().pos.copy()
        n_chunks = 0
        driver: ChunkDriver | None = None

        while t < duration:
            st = quad.state()
            if driver is None or driver.exhausted:
                img, _depth = render(st.pos, yaw=st.yaw, width=img_size, height=img_size,
                                     hfov_deg=HFOV, boxes=boxes)
                acts = policy.infer(img, state_vector(st), instruction)
                driver = ChunkDriver(acts, hold=hold)
                n_chunks += 1

            cmd = driver.next()
            quad.set_wrench(cmd.thrust, cmd.torque)
            world.advance(1.0 / CONTROL_HZ)
            t += 1.0 / CONTROL_HZ

            st = quad.state()
            path += float(np.linalg.norm(st.pos - prev))
            prev = st.pos.copy()
            energy += abs(cmd.thrust) * (1.0 / CONTROL_HZ)

            for _vp, _yw, lbl in view:
                if covered[lbl]:
                    continue
                if (float(np.linalg.norm(st.pos - _vp)) <= cover_max_dist
                        and in_view(st.pos, scene.target_pos, yaw=st.yaw, hfov_deg=HFOV, boxes=boxes)):
                    covered[lbl] = True
            if view and all(covered.values()) and not np.isfinite(t_cover):
                t_cover = t

            for b in boxes:
                d = np.linalg.norm(np.maximum(np.abs(st.pos - b.center) - b.half, 0.0))
                min_clear = min(min_clear, float(d) - cfg.drone.collision_radius)

        crashed = min_clear < 0.05
        coverage = sum(covered.values()) / len(view) if view else 0.0
        res = EpisodeResult(
            seed=seed,
            success=bool(coverage >= 1.0 and not crashed and st.pos[2] >= 0.05),
            coverage=float(coverage),
            min_clearance=float(min_clear),
            time_to_coverage=float(t_cover),
            path_length=float(path),
            energy=float(energy),
            steps=int(round(t / (1.0 / CONTROL_HZ))),
            crashed=bool(crashed),
        )
        return res, {"chunks": n_chunks, "instruction": instruction,
                     "task": TaskSpec(name="t1-inspect", duration=duration).name}
