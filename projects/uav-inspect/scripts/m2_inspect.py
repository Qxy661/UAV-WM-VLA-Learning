"""M2：脚本专家飞巡检路线，录制可训练的数据集。

一条 episode 的流程：
    起飞 → 依次飞向三个巡检视点（每个视点悬停 1.5 s、以 10 Hz 拍图）
    → 回到通道中点 → 降落。全程 48 Hz 记状态与动作，10 Hz 记图像。

指标（写入 metrics.json 与 episode meta）：
    coverage          三个视点里，目标真正落进视场且未被遮挡的比例
    route_time_s      从起飞到最后一个视点悬停结束的耗时
    frames            本次录到的图像帧数
    reach_ratio       按时到达的视点数 / 总数（超时不算到达）

产物：
    runs/m2_dataset/episode_XXXX/{obs.npz, meta.json}
    runs/m2_dataset/{manifest.json, stats.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                                    # noqa: E402
from uavinspect.control.cascaded import CascadedController             # noqa: E402
from uavinspect.data.recorder import EpisodeRecorder, compute_stats, write_manifest  # noqa: E402
from uavinspect.sim.camera import in_view, render                      # noqa: E402
from uavinspect.sim.scene import build_warehouse, visible_viewpoints   # noqa: E402
from uavinspect.sim.world import World                                 # noqa: E402
from uavinspect.util.latency import StageTimer                          # noqa: E402

IMG_SIZE = 64
HFOV = 70.0
CAPTURE_EVERY = 5          # 48 Hz / 5 ≈ 9.6 Hz
DWELL_S = 2.5
WP_TIMEOUT_S = 14.0
WP_TOL = 0.16


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def run_episode(cfg, seed: int, out_dir: Path, backend: str) -> dict:
    dt = cfg.sim.control_dt
    timer = StageTimer()
    rec = EpisodeRecorder(out_dir)

    with World(cfg, backend=backend) as world:
        backend_used = getattr(world, "backend", backend)
        quad = world.add_drone("uav", (0.0, 0.0, 0.25))
        scene = build_warehouse(world, seed=seed)
        seeable = visible_viewpoints(scene)
        ctrl = CascadedController(cfg, quad)

        # 只飞"从悬停位真的看得见目标"的视点；被货架挡住的视点不飞，
        # 但要在 meta 里如实记下来——覆盖率的分母是可行视点，不是全部视点。
        waypoints = [(pos, yaw, label) for (pos, yaw, label), ok in zip(scene.viewpoints, seeable) if ok]
        occluded = [label for (_pos, _yaw, label), ok in zip(scene.viewpoints, seeable) if not ok]
        home = np.array([0.0, 0.0, cfg.task.hover_height])

        ep = {"coverage": 0.0, "covered": [], "route_time_s": None,
              "reach": [], "frames": 0}
        covered: list[str] = []
        reach: list[bool] = []
        t = 0.0
        t_route_start = None
        t_route_end = None

        # ---------- 起飞 ----------
        while t < 6.0:
            st = quad.state()
            ref = np.array([0.0, 0.0, cfg.task.hover_height])
            thrust, torque = ctrl.compute(st, ref, yaw_ref=0.0)
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt
            if np.linalg.norm(st.pos - ref) < 0.20:
                break
            rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                    action=np.concatenate([[thrust], torque]), t=t, phase=0)

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
                t0 = time.perf_counter()
                thrust, torque = ctrl.compute(st, wp, yaw_ref=yaw_ref)
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
                if step % CAPTURE_EVERY == 0:
                    t0 = time.perf_counter()
                    image, _ = render(st.pos, scene.boxes, yaw=yaw_ref, pitch=-0.05,
                                      width=IMG_SIZE, height=IMG_SIZE, hfov_deg=HFOV,
                                      with_depth=False)
                    timer.record("render", time.perf_counter() - t0)
                rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                        action=np.concatenate([[thrust], torque]), t=t, phase=1 + wi,
                        image=image)

            reach.append(arrived)
            if not arrived:
                continue

            # ---------- 悬停拍摄 ----------
            t_dwell0 = t
            step = 0
            while t - t_dwell0 < DWELL_S:
                st = quad.state()
                thrust, torque = ctrl.compute(st, wp, yaw_ref=yaw_wp)
                quad.set_wrench(thrust, torque)
                world.advance(dt)
                t += dt
                step += 1
                image = None
                if step % CAPTURE_EVERY == 0:
                    t0 = time.perf_counter()
                    image, _ = render(st.pos, scene.boxes, yaw=yaw_wp, pitch=-0.05,
                                      width=IMG_SIZE, height=IMG_SIZE, hfov_deg=HFOV,
                                      with_depth=False)
                    timer.record("render", time.perf_counter() - t0)
                    rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                            action=np.concatenate([[thrust], torque]), t=t, phase=10 + wi,
                            image=image)
                    if label not in covered and in_view(
                            st.pos, scene.target_pos, yaw=yaw_wp, hfov_deg=HFOV,
                            boxes=scene.boxes):
                        covered.append(label)
                else:
                    rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                            action=np.concatenate([[thrust], torque]), t=t, phase=10 + wi)
            t_route_end = t

        # ---------- 返回并降落 ----------
        while t < 90.0:
            st = quad.state()
            thrust, torque = ctrl.compute(st, home, yaw_ref=0.0)
            quad.set_wrench(thrust, torque)
            world.advance(dt)
            t += dt
            rec.add(pos=st.pos, vel=st.vel, quat=st.quat,
                    action=np.concatenate([[thrust], torque]), t=t, phase=20)
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
    meta = rec.close(ep)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 巡检录制")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "m2_dataset"))
    ap.add_argument("--backend", default="auto", choices=["auto", "rigid", "pybullet"])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps: list[dict] = []
    wall0 = time.perf_counter()
    for i in range(args.episodes):
        seed = args.seed0 + i
        ep_id = f"episode_{seed:04d}"
        t0 = time.perf_counter()
        meta = run_episode(cfg, seed, out / ep_id, args.backend)
        meta["id"] = ep_id
        meta["dir"] = ep_id
        meta["wall_time_s"] = round(time.perf_counter() - t0, 2)
        eps.append(meta)
        print(f"{ep_id}  seed={seed}  覆盖 {meta['coverage'] * 100:5.1f}%  "
              f"{meta['frames']:3d} 帧  {meta['wall_time_s']:.1f} s  "
              f"已覆盖={meta['covered']}")

    manifest = write_manifest(out, eps)
    stats = compute_stats(out, manifest)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    train = [e for e in manifest["episodes"] if e["split"] == "train"]
    val = [e for e in manifest["episodes"] if e["split"] == "val"]
    cov = float(np.mean([e["coverage"] for e in eps]))
    metrics = {
        "task": "m2_dataset",
        "episodes": len(eps),
        "split": {"train": len(train), "val": len(val)},
        "coverage_mean": round(cov, 4),
        "coverage_all": [round(e["coverage"], 4) for e in eps],
        "frames_total": int(sum(e["frames"] for e in eps)),
        "wall_time_s": round(time.perf_counter() - wall0, 2),
        "seeds": [e["seed"] for e in eps],
        "backend": eps[0]["backend"] if eps else None,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print()
    print(f"数据集：{len(eps)} 集（train {len(train)} / val {len(val)}），"
          f"共 {metrics['frames_total']} 帧，平均覆盖率 {cov * 100:.1f}%")
    print(f"产物：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
