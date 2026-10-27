"""巡检回合：一条指令、一个视点、一段数据。

从 `scripts/m2_inspect.py` 抽出来的，因为它有两个使用方——M2 的演示脚本和 T1 的数据生成器。
两边必须飞同一条轨迹，否则「训练数据」与「验收数据」不是同一个分布。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from uavinspect.control.cascaded import CascadedController
from uavinspect.data.instructions import make_instruction
from uavinspect.data.recorder import EpisodeRecorder
from uavinspect.sim.camera import in_view, render
from uavinspect.sim.scene import build_warehouse, visible_viewpoints
from uavinspect.sim.world import World
from uavinspect.util.latency import StageTimer

IMG_SIZE = 64
HFOV = 70.0
CAPTURE_EVERY = 5
DWELL_S = 2.5
WP_TIMEOUT_S = 14.0
WP_TOL = 0.16
CENTER_TOL = 0.12     # 目标在画面水平方向的归一化偏移上限，见 eval/protocol.py

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _marker_u(pos, yaw: float, target, hfov_deg: float) -> float:
    """目标在画面里的归一化水平偏移；目标在身后时返回一个远超阈值的大数。"""
    d = np.asarray(target[:2]) - np.asarray(pos[:2])
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([-np.sin(yaw), np.cos(yaw)])
    x = float(d @ fwd)
    if x <= 1e-6:
        return 9.0
    return float(d @ right) / x / np.tan(np.deg2rad(hfov_deg) / 2.0)


def _label(policy, thrust, torque, *, ctrl=None, st=None, ref=None, yaw_ref=0.0):
    """这一拍记什么动作：策略自己给的，或者现问一次专家（DAgger-lite 用）。

    注意必须**现问**：策略每 `policy_every` 拍才被调用一次，若沿用
    `last_label`，中间几拍记下的会是好几个控制步之前的陈旧专家标签，
    图文对应直接错位。早前就吃了这个亏（DAgger 数据把看得见的 BC 打到 0.000）。
    """
    fn = getattr(policy, "label", None)
    if fn is not None and ctrl is not None and st is not None:
        got = fn(st, ref, yaw_ref)
        if got is not None:
            return got
    lbl = getattr(policy, "last_label", None)
    if lbl is None:
        return thrust, torque
    return float(lbl[0]), lbl[1]


def run_episode(cfg, seed: int, out_dir: Path, backend: str, *, img_size: int = IMG_SIZE,
                capture_every: int = CAPTURE_EVERY, only_label: str | None = None,
                held_out: bool = False, instruction: str | None = None,
                policy=None, policy_every: int = 8, start_z: float | None = None,
                perturb: bool = False, perturb_scale: float = 1.0,
                action_space: str = "torque") -> dict:
    dt = cfg.sim.control_dt
    timer = StageTimer()
    rec = EpisodeRecorder(out_dir)

    with World(cfg, backend=backend) as world:
        backend_used = getattr(world, "backend", backend)
        quad = world.add_drone("uav", (0.0, 0.0, start_z if start_z is not None else 0.25))
        scene = build_warehouse(world, seed=seed)
        seeable = visible_viewpoints(scene)
        ctrl = CascadedController(cfg, quad)

        def _act(thrust, torque):
            """动作记录口径。omega 表示记 [thrust, omega_ref]，把角速率环留给低层。"""
            if action_space == "omega" and getattr(ctrl, "omega_ref", None) is not None:
                return np.concatenate([[thrust], np.asarray(ctrl.omega_ref)])
            return np.concatenate([[thrust], torque])

        if perturb:
            # 恢复演示：从偏高/偏侧、带初速、带恒定风的状态起步。种子决定全部扰动量，
            # 同一 seed 可逐位复现。起飞段因此被跳过（起点已经在悬停高度附近）。
            _rng = np.random.default_rng(seed * 101 + 7)
            _dz = float(_rng.uniform(-0.40, 0.40)) * perturb_scale
            _dxy = _rng.uniform(-0.50, 0.50, 2) * perturb_scale
            _v0 = _rng.uniform(-0.50, 0.50, 3) * perturb_scale
            _wind = _rng.uniform(-0.30, 0.30, 3) * perturb_scale
            quad.reset(np.array([_dxy[0], _dxy[1], cfg.task.hover_height + _dz]))
            quad.set_perturbation(vel=_v0, wind=_wind)

        # 步数令牌必须在外层先绑定：drive() 是闭包，而 step 在下面的视点循环里
        # 才第一次赋值，Python 3.11 会把它判成"enclosing 里尚未绑定的自由变量"。
        step = 0

        def drive(st, ref, yaw_ref, image=None, use_policy=True):
            """专家用串级控制器；传入 policy 时由策略出 (thrust, torque)。

            policy 拿到的是一个 ctx 字典而不是位置参数：闭环评测需要图像、
            场景（判可见）和步数，签名一改就得到处改调用点。
            """
            if policy is None or not use_policy:
                return ctrl.compute(st, ref, yaw_ref=yaw_ref)
            return policy({"st": st, "ref": ref, "yaw_ref": yaw_ref, "image": image,
                           "scene": scene, "img_size": img_size, "t": t, "step": step,
                           "ctrl": ctrl})

        # 只飞"从悬停位真的看得见目标"的视点；被货架挡住的视点不飞，
        # 但要在 meta 里如实记下来——覆盖率的分母是可行视点，不是全部视点。
        waypoints = [(pos, yaw, label) for (pos, yaw, label), ok in zip(scene.viewpoints, seeable) if ok]
        if only_label is not None:
            waypoints = [w for w in waypoints if w[2] == only_label]
            if not waypoints:
                raise ValueError(f"seed {seed} 的视点 {only_label} 不可见，无法按该指令执行")
        rack_id = int(np.argmin(np.abs(np.array(scene.rack_x) - scene.target_pos[0]))) + 1
        side = "oblique" if only_label == "oblique" else "front"
        if instruction is None:
            import random as _random
            instruction = make_instruction(rack_id, side, _random.Random(seed), held_out=held_out)
        occluded = [label for (_pos, _yaw, label), ok in zip(scene.viewpoints, seeable) if not ok]
        home = np.array([0.0, 0.0, cfg.task.hover_height])

        ep = {"coverage": 0.0, "covered": [], "route_time_s": None, "action_space": action_space,
              "reach": [], "frames": 0}
        covered: list[str] = []
        reach: list[bool] = []
        t = 0.0
        t_route_start = None
        t_route_end = None

        # ---------- 起飞（给了 start_z 就跳过，让策略自己从低空爬升）----------
        while start_z is None and not perturb and t < 6.0:
            st = quad.state()
            ref = np.array([0.0, 0.0, cfg.task.hover_height])
            thrust, torque = drive(st, ref, 0.0, use_policy=False)
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt
            if np.linalg.norm(st.pos - ref) < 0.20:
                break
            rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                    action=_act(thrust, torque), t=t, phase=0)

        t_route_start = t

        # ---------- 跑视点 ----------
        for wi, (wp, yaw_wp, label) in enumerate(waypoints):
            wp = np.asarray(wp, dtype=float)
            yaw0 = quad.state().yaw
            t_wp0 = t
            arrived = False
            step = 0
            while t - t_wp0 < WP_TIMEOUT_S:
                st = quad.state()
                # yaw 用剩余时间线性接近，避免一开始就原地大转
                remain = max(WP_TIMEOUT_S - (t - t_wp0), 0.1)
                dyaw = wrap_pi(yaw_wp - yaw0)
                yaw_ref = yaw0 + dyaw * min(1.0, (t - t_wp0) / max(remain * 0.5, 0.5))
                policy_image = None
                if policy is not None and step % policy_every == 0:
                    t0 = time.perf_counter()
                    policy_image, _ = render(st.pos, scene.boxes, yaw=yaw_ref, pitch=-0.05,
                                              width=img_size, height=img_size, hfov_deg=HFOV,
                                              with_depth=False)
                    timer.record("render", time.perf_counter() - t0)
                t0 = time.perf_counter()
                thrust, torque = drive(st, wp, yaw_ref, policy_image)
                timer.record("control", time.perf_counter() - t0)
                quad.set_wrench(thrust, torque)
                world.advance(dt)
                t += dt
                step += 1

                st = quad.state()
                if np.linalg.norm(st.pos - wp) < WP_TOL and abs(st.vel).max() < 0.35:
                    arrived = True
                    break
                # 接近段也拍：悬停帧全是从同一个位姿拍的，世界模型会学到"零动作 → 同一张图"
                image = None
                if step % capture_every == 0:
                    t0 = time.perf_counter()
                    image, _ = render(st.pos, scene.boxes, yaw=yaw_ref, pitch=-0.05,
                                      width=img_size, height=img_size, hfov_deg=HFOV,
                                      with_depth=False)
                    timer.record("render", time.perf_counter() - t0)
                rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                        action=_act(*_label(policy, thrust, torque, ctrl=ctrl, st=st, ref=wp,
                                            yaw_ref=yaw_ref)), t=t, phase=1 + wi,
                        image=image)

            reach.append(arrived)
            if not arrived:
                continue

            # ---------- 悬停拍摄 ----------
            t_dwell0 = t
            step = 0
            while t - t_dwell0 < DWELL_S:
                st = quad.state()
                policy_image = None
                if policy is not None and step % policy_every == 0:
                    t0 = time.perf_counter()
                    policy_image, _ = render(st.pos, scene.boxes, yaw=yaw_wp, pitch=-0.05,
                                              width=img_size, height=img_size, hfov_deg=HFOV,
                                              with_depth=False)
                    timer.record("render", time.perf_counter() - t0)
                thrust, torque = drive(st, wp, yaw_wp, policy_image)
                quad.set_wrench(thrust, torque)
                world.advance(dt)
                t += dt
                step += 1
                image = None
                if step % capture_every == 0:
                    t0 = time.perf_counter()
                    image, _ = render(st.pos, scene.boxes, yaw=yaw_wp, pitch=-0.05,
                                      width=img_size, height=img_size, hfov_deg=HFOV,
                                      with_depth=False)
                    timer.record("render", time.perf_counter() - t0)
                    rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                            action=_act(*_label(policy, thrust, torque, ctrl=ctrl, st=st, ref=wp,
                                                yaw_ref=yaw_wp)), t=t, phase=10 + wi,
                            image=image)
                    if label not in covered and in_view(
                            st.pos, scene.target_pos, yaw=st.yaw, hfov_deg=HFOV,
                            boxes=scene.boxes) and abs(_marker_u(
                                st.pos, st.yaw, scene.target_pos, HFOV)) <= CENTER_TOL:
                        covered.append(label)
                else:
                    rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                            action=_act(*_label(policy, thrust, torque, ctrl=ctrl, st=st, ref=wp,
                                                yaw_ref=yaw_wp)), t=t, phase=10 + wi)
            t_route_end = t

        # ---------- 返回并降落 ----------
        while t < 90.0:
            st = quad.state()
            thrust, torque = drive(st, home, 0.0)
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt
            rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                    action=_act(thrust, torque), t=t, phase=20)
            if np.linalg.norm(st.pos - home) < 0.15 and abs(st.vel).max() < 0.2:
                break
        sim_t = world.t

    ep["coverage"] = round(len(covered) / len(waypoints), 4) if waypoints else 0.0
    ep["covered"] = covered
    ep["reach"] = reach
    ep["feasible_viewpoints"] = [w[2] for w in waypoints]
    ep["occluded_viewpoints"] = occluded
    ep["route_time_s"] = (round(t_route_end - t_route_start, 3)
                          if t_route_start is not None and t_route_end is not None else None)
    ep["viewpoints"] = [
        {"label": label, "pos": np.round(pos, 3).tolist(), "visible_from_standoff": bool(ok)}
        for (pos, _y, label), ok in zip(scene.viewpoints, seeable)
    ]
    ep["target_pos"] = np.round(scene.target_pos, 3).tolist()
    ep["backend"] = backend_used
    ep["latency"] = timer.summary()
    ep["sim_time_s"] = sim_t
    ep["seed"] = seed
    ep["instruction"] = instruction
    ep["rack"] = rack_id
    ep["side"] = side
    ep["img_size"] = img_size
    ep["capture_every"] = capture_every
    meta = rec.close(ep)
    return meta
