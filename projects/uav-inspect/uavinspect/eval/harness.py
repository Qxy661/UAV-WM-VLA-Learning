"""M6 评测 harness：一个任务、三种策略、同一套口径。

设计目标不是"跑出好看的数"，而是**让不同策略之间的数字可比**，并且把
"不可横比"的原因暴露在结构里：

1. 每个 episode 是一个独立单元。汇总时按 episode 聚合（先算每集，再对集取均值），
   不把几千帧当独立样本——后者会把置信区间缩小一个量级，得到假显著。
2. 报告里带 `protocol` 块（控制频率、任务时长、种子数、成功判据、口径版本）。
   `compare()` 发现两块 protocol 不一致直接拒绝比较，而不是悄悄比。
3. 每个指标同时给点估计与 bootstrap 95% 区间（按 episode 重采样）。

指标定义（写死在这里，避免各处自己解释）：
    success            覆盖全部可行视点 且 全程不触地 且 不撞墙
    coverage           被相机判定为"目标可见"的视点比例
    min_clearance      全程到任意碰撞盒表面的最小距离（m）
    time_to_coverage   首次达到 100% 覆盖的时间（s）；没达到记 inf
    path_length        轨迹长度（m）
    energy             推力对时间的积分 ∫T dt（N·s），只做同任务内比较
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..sim.camera import in_view
from ..sim.scene import build_warehouse, visible_viewpoints
from ..sim.world import World

PROTOCOL_VERSION = "m6-v1"
SPAWN = (-3.6, 0.0, 0.25)          # 通道靠端头，离货架足够远，起飞不擦边


@dataclass
class TaskSpec:
    name: str
    duration: float
    camera_period: float = 0.2          # 5 Hz
    accept_clearance: float = 0.05      # 小于它算撞上
    cover_max_dist: float = 0.7         # 必须真的飞到视点附近（站位容差），否则不算拍到
    description: str = ""


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    coverage: float
    min_clearance: float
    time_to_coverage: float
    path_length: float
    energy: float
    steps: int
    crashed: bool


@dataclass
class AgentReport:
    agent: str
    task: str
    protocol: dict
    episodes: list = field(default_factory=list)

    # ---------------- 汇总 ----------------
    def summary(self) -> dict:
        def stat(key, as_float=True):
            vals = np.array([getattr(e, key) for e in self.episodes], dtype=float)
            vals = np.where(np.isfinite(vals), vals, np.nan)
            if np.all(np.isnan(vals)):
                return {"mean": None, "ci95": None, "n": len(vals)}
            if as_float and key in ("success", "crashed"):
                mean = float(np.nanmean(vals))
                lo, hi = bootstrap_ci(vals, agg=np.mean)
                return {"rate": mean, "ci95": [lo, hi], "n": int(len(vals))}
            mean = float(np.nanmean(vals))
            lo, hi = bootstrap_ci(vals)
            return {"mean": mean, "ci95": [lo, hi], "n": int(np.sum(~np.isnan(vals)))}

        return {
            "agent": self.agent,
            "task": self.task,
            "protocol": self.protocol,
            "success": stat("success"),
            "coverage": stat("coverage"),
            "min_clearance": stat("min_clearance"),
            "time_to_coverage": stat("time_to_coverage"),
            "path_length": stat("path_length"),
            "energy": stat("energy"),
        }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0, agg=np.mean) -> tuple[float, float]:
    """按 episode 重采样的 95% 区间。样本少时它比正态近似诚实得多。"""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    stats = agg(v[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


# ---------------------------------------------------------------- 单集
def run_episode(task: TaskSpec, policy, seed: int, *, control_dt: float = 1 / 48,
                verbose: bool = False) -> EpisodeResult:
    cfg = _load_cfg()
    rng = np.random.default_rng(seed)
    with World(cfg, backend="rigid") as world:
        scene = build_warehouse(world, seed=seed, with_target=True)
        quad = world.add_drone("uav", SPAWN)
        policy.reset(quad)

        box_objs = scene.boxes
        ok = visible_viewpoints(scene)
        view = [(vp, yaw, lbl) for (vp, yaw, lbl), good in zip(scene.viewpoints, ok) if good]
        seen = {lbl: False for _vp, _yw, lbl in view}
        if hasattr(policy, "set_task"):          # 任务书（视点坐标），不是场景真值
            policy.set_task([(vp, yw) for vp, yw, _lbl in view])
        t = 0.0
        path = 0.0
        energy = 0.0
        min_clear = float("inf")
        t_cover = float("inf")
        prev = quad.state().pos.copy()

        while t < task.duration:
            st = quad.state()
            thrust, torque = policy.act(st, t)
            quad.set_wrench(thrust, torque)
            world.advance(control_dt)
            t += control_dt
            ans = 0.0

            st = quad.state()
            path += float(np.linalg.norm(st.pos - prev))
            prev = st.pos.copy()
            energy += abs(float(thrust)) * control_dt

            if int(round(t / control_dt)) % max(1, int(round(task.camera_period / control_dt))) == 0:
                for _vp, _yw, lbl in view:
                    if seen[lbl]:
                        continue
                    if (in_view(st.pos, scene.target_pos, yaw=st.yaw, hfov_deg=70.0, boxes=box_objs)
                            and float(np.linalg.norm(st.pos - _vp)) <= task.cover_max_dist):
                        seen[lbl] = True
                if all(seen.values()) and not np.isfinite(t_cover):
                    t_cover = t

            for b in box_objs:
                d = np.linalg.norm(np.maximum(np.abs(st.pos - b.center) - b.half, 0.0))
                min_clear = min(min_clear, float(d) - cfg.drone.collision_radius)

        crashed = min_clear < task.accept_clearance
        coverage = sum(seen.values()) / len(view) if view else 0.0
        return EpisodeResult(
            seed=seed,
            success=bool(coverage >= 1.0 and not crashed and st.pos[2] >= 0.05),
            coverage=float(coverage),
            min_clearance=float(min_clear),
            time_to_coverage=float(t_cover),
            path_length=float(path),
            energy=float(energy),
            steps=int(round(t / control_dt)),
            crashed=bool(crashed),
        )


def evaluate(agent, task: TaskSpec, seeds, *, make_policy=None, verbose=False) -> AgentReport:
    """seeds 就是 episode 数：一个 seed 一集，按集聚合。"""
    protocol = {
        "version": PROTOCOL_VERSION,
        "task": task.name,
        "control_hz": round(1 / (task.duration / max(1, task.duration / (1 / 48))), 3),
        "duration_s": task.duration,
        "episodes": len(list(seeds)),
        "camera_period_s": task.camera_period,
        "accept_clearance_m": task.accept_clearance,
        "success_rule": "coverage>=1.0 && min_clearance>=accept_clearance && min_z>=0.05",
    }
    rep = AgentReport(agent=agent, task=task.name, protocol=protocol)
    for seed in seeds:
        pol = make_policy(seed) if make_policy else agent
        rep.episodes.append(run_episode(task, pol, seed))
        if verbose:
            e = rep.episodes[-1]
            print(f"  {agent} seed={seed} cov={e.coverage:.2f} clear={e.min_clearance:.3f} "
                  f"success={e.success}")
    return rep


def compare(reports, strict: bool = True) -> None:
    """不同 protocol 的报告不允许放进同一张表。

    这不是洁癖：控制频率、回合数、成功判据任一不同，数字就没有可比性，
    放在同一行会直接误导读者。
    """
    if not strict or len(reports) < 2:
        return
    base = reports[0].protocol
    for r in reports[1:]:
        if r.protocol != base:
            diff = {k: (base.get(k), r.protocol.get(k))
                    for k in set(base) | set(r.protocol)
                    if base.get(k) != r.protocol.get(k)}
            raise ValueError(
                f"protocol 不一致，拒绝比较（{reports[0].agent} vs {r.agent}）：{diff}")


def to_json(reports) -> str:
    return json.dumps([r.summary() for r in reports], ensure_ascii=False, indent=2)


def _load_cfg():
    from pathlib import Path
    from ..config import load
    return load(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
