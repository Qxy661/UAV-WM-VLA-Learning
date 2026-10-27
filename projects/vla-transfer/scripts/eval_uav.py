"""闭环评测：把训练好的动作块策略装回 uav-inspect 的巡检任务。

与开环验证的区别：这里图是现渲染的，动作喂回动力学，飞歪了没人纠正。
所以 nll/MSE 低不代表成功率，两件事分开报。

用法（开发机）：
    python projects/vla-transfer/scripts/eval_uav.py \
        --ckpt /root/work/runs/bc/model.pt --data /root/work/data/uav96 \
        --episodes 40 --seeds 8000 8100 --out /root/work/runs/bc_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "uav-inspect"))
sys.path.insert(0, str(ROOT / "projects" / "vla-transfer"))

from bc.data import HOVER_N, base_action
from bc.model import ACTION_DIM, HORIZON, STATE_DIM, BCNet, text_features
from uavinspect.config import load as load_cfg
from uavinspect.pipeline.inspect import run_episode
from uavinspect.sim.scene import build_warehouse
from uavinspect.sim.world import World
from eval.protocol import m6_coverage, marker_uv, yaw_from_quat

# 目标必须落在画面水平方向中心附近。放宽到 0.35 时盲策略靠「正对货架」
# 能蒙中约四成 seed；收紧到 0.12 后只剩 |dx| < 6 cm 的少数 seed 漏过去。
CENTER_TOL = 0.12


class ChunkPolicy:
    """一次推理出 8 步动作，逐步弹出；弹空才重新看图。

    这就是 PLAN.md 里 N ≥ ceil(f_control/f_policy) 的落地：策略 10 Hz、控制 48 Hz，
    不补这层插值，中间 7 个控制周期就得冻结或重复同一帧动作。
    """

    def __init__(self, ckpt: Path, instruction: str, device: str = "cpu",
                 action_space: str = "torque", residual: bool = False, blind: bool = False):
        # action_space=omega 时 a[1:4] 是期望角速率而非力矩，转换交给 OmegaWrapper
        self.action_space = action_space
        self.residual = residual
        # blind=True：把图像置零再用。用来判断策略到底有没有用视觉，
        # 若准确率不掉，说明它其实是个"状态策略 + 文本"。
        self.blind = blind
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        # 动作块长度由训练时的 --horizon 决定，从 ckpt 的 args 读，别用模块常量覆盖
        self.horizon = int(obj.get("args", {}).get("horizon", HORIZON))
        self.model = BCNet(horizon=self.horizon, action_dim=ACTION_DIM, state_dim=STATE_DIM).to(device)
        self.model.load_state_dict(obj["state_dict"])
        self.model.eval()
        self.mean = np.asarray(obj["mean"], np.float32)
        self.std = np.asarray(obj["std"], np.float32)
        self.text = torch.from_numpy(text_features(instruction))[None].to(device)
        self.device = device
        self.buf: list[np.ndarray] = []
        self.replans = 0
        # 没图时的兜底推力。残差模型的 mean 恒为 0（见 bc/data.py 的 fit_stats），
        # 直接用它当推力等于 0 N，起飞后第一帧没图就会掉下去——这是 r2/r3/r4 三次
        # 「飞起来又沉底」的直接原因。残差模式必须用保高前馈的悬停项。
        self.hover = HOVER_N if residual else float(self.mean[0])

    def __call__(self, ctx: dict):
        if not self.buf:
            image = ctx["image"]
            if image is None:                     # 还没到拍摄周期，按悬停推力兜底
                return self.hover, np.zeros(3)
            arr = np.asarray(image, np.float32) / 255.0
            if self.blind:
                arr = np.zeros_like(arr)
            img = torch.from_numpy(arr)
            img = img.permute(2, 0, 1)[None].to(self.device)
            st = ctx["st"]
            state = np.concatenate([st.pos, st.vel, st.quat]).astype(np.float32)
            with torch.no_grad():
                pred = self.model(img, torch.from_numpy(state)[None].to(self.device), self.text)
            act = pred[0].cpu().numpy() * self.std + self.mean
            if self.residual:
                # chunk 内的保高前馈用当前状态近似：8 步只有 0.17 s，高度变化不超过厘米级
                zc = float(ctx["st"].pos[2]); vzc = float(ctx["st"].vel[2])
                act = act + base_action(np.full(act.shape[0], zc), np.full(act.shape[0], vzc))
            self.buf = [act[i] for i in range(act.shape[0])]
            self.replans += 1
        a = self.buf.pop(0)
        return float(a[0]), np.asarray(a[1:], dtype=float)


class OmegaWrapper:
    """把策略输出的 omega_ref 换成力矩：torque = I(rate_kp(w_ref-w)) + w×Iw。"""

    def __init__(self, inner, adapter) -> None:
        self.inner, self.adapter = inner, adapter
        self.replans = 0

    def __call__(self, ctx: dict):
        thrust, omega_ref = self.inner(ctx)
        self.replans = getattr(self.inner, "replans", 0)
        return thrust, self.adapter(omega_ref, ctx["st"].omega)


class SubgoalPolicy:
    """模型只出「目标在哪」，怎么飞交给底层 P-D 控制器。

    接口动机见 `projects/vla-transfer/PLAN-v3-分层动作接口.md`：逐控制步学推力+力矩
    等于让 30 万参数的 CNN 同时学"去哪"和"怎么稳住"，而后者已有调好的串级 P-D。
    换成子目标后，视觉误差直接映射成 |u| 误差（|u| ≈ dx / 0.805），视觉据此成为必要条件。
    """

    STAND = np.array([0.0, -1.15, 0.10])      # 沿货架法向退 1.15 m，抬高 0.10 m

    def __init__(self, ckpt: Path | None, instruction: str, device: str = "cpu",
                 blind: bool = False, every: int = 5, predict=None) -> None:
        self.predict = predict                     # 外部预测器（π0.5 服务），给了就不加载 ckpt
        self.model = None
        if predict is None:
            obj = torch.load(ckpt, map_location=device, weights_only=False)
            self.model = BCNet(horizon=1, action_dim=3, state_dim=STATE_DIM).to(device)
            self.model.load_state_dict(obj["state_dict"])
            self.model.eval()
            self.mu = np.asarray(obj["goal_mean"], np.float32)
            self.sd = np.asarray(obj["goal_std"], np.float32)
        self.text = torch.from_numpy(text_features(instruction))[None].to(device)
        self.device, self.blind, self.every = device, blind, every
        self.est = None
        self.since = 10 ** 9
        self.replans = 0

    def _predict(self, ctx: dict) -> np.ndarray:
        st = ctx["st"]
        if self.predict is not None:
            # 外部预测器直接给"目标相对我的位移"（米），加回站位即为世界坐标。
            v = self.predict(ctx)
            if v is None:
                return None
            return np.asarray(st.pos, np.float32) + np.asarray(v, dtype=np.float32).reshape(3)
        arr = np.asarray(ctx["image"], np.float32) / 255.0
        if self.blind:
            arr = np.zeros_like(arr)
        img = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device)
        state = np.concatenate([st.pos, st.vel, st.quat]).astype(np.float32)
        with torch.no_grad():
            pred = self.model(img, torch.from_numpy(state)[None].to(self.device), self.text)
        # 模型输出的是"目标相对我的位移"，加回当前站位得到世界坐标。
        rel = pred[0, 0].cpu().numpy() * self.sd + self.mu
        return np.asarray(st.pos, np.float32) + rel

    def __call__(self, ctx: dict):
        self.since += 1
        if ctx["image"] is not None and (self.est is None or self.since >= self.every):
            new = self._predict(ctx)
            if new is not None:
                self.est = new
                self.since = 0
                self.replans += 1
        if self.est is None:
            return 9.81, np.zeros(3)
        wp = self.est + self.STAND
        yaw = float(np.arctan2(self.est[1] - wp[1], self.est[0] - wp[0]))
        return ctx["ctrl"].compute(ctx["st"], wp, yaw_ref=yaw)


class SmoothPolicy:
    """动作低通：a_t = alpha * a_{t-1} + (1 - alpha) * a_t。

    用来单独检验「误差累积」这条假设：如果平滑能把成功率抬起来，
    说明问题在单步抖动被闭环放大，而不是策略不知道往哪飞。
    纯推理期技巧，不需要重训。
    """

    def __init__(self, inner, alpha: float = 0.0) -> None:
        self.inner, self.alpha = inner, float(alpha)
        self.prev = None
        self.replans = 0

    def __call__(self, ctx: dict):
        thrust, torque = self.inner(ctx)
        self.replans = getattr(self.inner, "replans", 0)
        a = np.concatenate([[thrust], np.asarray(torque, dtype=float)])
        if self.prev is not None and self.alpha > 0.0:
            a = self.alpha * self.prev + (1.0 - self.alpha) * a
        self.prev = a
        return float(a[0]), a[1:]


class BiasPolicy:
    """诊断包装：给推力加常数偏置，用来区分「塌到均值」与「接线错误」。"""
    def __init__(self, inner, bias: float) -> None:
        self.inner, self.bias = inner, float(bias)
        self.replans = 0

    def __call__(self, ctx: dict):
        thrust, torque = self.inner(ctx)
        self.replans = getattr(self.inner, "replans", 0)
        return thrust + self.bias, torque


def side_of(meta_path: Path) -> str:
    return json.loads(meta_path.read_text(encoding="utf-8")).get("viewpoint", "front")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="BC 本地 ckpt；用 --server 时可省")
    ap.add_argument("--data", required=True, help="npz 数据集根（用来读指令与场景 seed 列表）")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scripted", action="store_true", help="专家脚本跑同一批 episode，作为上界")
    ap.add_argument("--perturb", action="store_true", help="从扰动状态起步（恢复能力测试）")
    ap.add_argument("--perturb-scale", type=float, default=0.5)
    ap.add_argument("--instruction-override", default="",
                    help="把每条 episode 的指令换成这一句，用来测指令是否被使用")
    ap.add_argument("--blind", action="store_true",
                    help="推理时把图像置零，判断策略是否真的依赖视觉")
    ap.add_argument("--start-z", type=float, default=None, help="从指定高度起飞，交给策略爬升（恢复能力测试）")
    ap.add_argument("--policy-every", type=int, default=0, help="每隔几个控制步重规划一次，0=每块一次")
    ap.add_argument("--thrust-bias", type=float, default=0.0, help="诊断用：BC 输出推力加偏置")
    ap.add_argument("--smooth", type=float, default=0.0, help="动作低通系数 alpha（0=不平滑）")
    ap.add_argument("--subgoal", action="store_true",
                    help="ckpt 出的是目标点（3 维），用底层控制器跟踪")
    ap.add_argument("--subgoal-every", type=int, default=5, help="每几个控制步重问一次子目标")
    ap.add_argument("--residual", action="store_true", help="ckpt 学的是相对保高前馈的残差")
    ap.add_argument("--action-space", default="torque", choices=["torque", "omega"],
                    help="ckpt 训练时的动作口径；omega 表示输出期望角速率")
    ap.add_argument("--server", default=None, help="host:port；给了就评 π0.5 服务而不是本地 ckpt")
    ap.add_argument("--layout", default="a", choices=["a", "b"])
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval_uav"))
    a = ap.parse_args()
    os.environ["UAV_LAYOUT"] = a.layout
    if not (a.server or a.ckpt or a.scripted):
        raise SystemExit("要么给 --ckpt，要么给 --server host:port，要么 --scripted")

    cfg = load_cfg(ROOT / "projects" / "uav-inspect" / "configs" / "default.yaml")
    root = Path(a.data)
    metas = sorted(root.glob("shard_*/episode_*/meta.json"))
    test = [m for m in metas
            if json.loads(m.read_text(encoding="utf-8")).get("split") == "test"]
    if a.seeds:
        seeds = []
        for s in a.seeds:
            seeds.extend([x for x in test if json.loads(x.read_text(encoding="utf-8"))["seed"] == s])
    else:
        seeds = test
    picked: dict[int, Path] = {}
    for m in seeds:
        meta = json.loads(m.read_text(encoding="utf-8"))
        picked.setdefault(int(meta["seed"]), m)
    chosen = [(s, picked[s]) for s in sorted(picked)][:a.episodes]
    if not chosen:
        raise SystemExit("没有可用的 test episode，先看数据集的 split 字段")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    for k, (seed, meta_path) in enumerate(chosen, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instruction = meta["instruction"]
        if a.instruction_override:
            instruction = a.instruction_override
        side = meta.get("viewpoint", "front")
        if a.scripted:
            policy = None
        elif a.server:
            sys.path.insert(0, str(ROOT / "projects" / "vla-transfer" / "policy"))
            from pi05_policy import Pi05Policy
            host, _, port = a.server.partition(":")
            inner = Pi05Policy(host=host, port=int(port or 8000), instruction=instruction,
                               action_dim=3 if a.subgoal else 4)

            def _predict(ctx, _inner=inner):
                v = _inner.infer_vector(ctx)
                if v is None:                       # 还没到拍摄周期，保持上一次估计
                    return None
                return np.asarray(v, dtype=np.float32).reshape(3)

            policy = (SubgoalPolicy(None, instruction, device=a.device, blind=a.blind,
                                    every=a.subgoal_every, predict=_predict)
                      if a.subgoal else inner)
        else:
            if a.subgoal:
                policy = SubgoalPolicy(Path(a.ckpt), instruction, device=a.device,
                                       blind=a.blind, every=a.subgoal_every)
            else:
                policy = ChunkPolicy(Path(a.ckpt), instruction, device=a.device,
                                     action_space=a.action_space, residual=a.residual,
                                     blind=a.blind)
            if a.action_space == "omega":
                sys.path.insert(0, str(ROOT / "projects" / "vla-transfer"))
                from policy.omega_adapter import OmegaToTorque
                policy = OmegaWrapper(policy, OmegaToTorque(cfg))
            if a.thrust_bias:
                policy = BiasPolicy(policy, a.thrust_bias)
            if a.smooth and policy is not None:
                policy = SmoothPolicy(policy, a.smooth)
        ep = run_episode(cfg, seed, out / f"ep_{seed}", "rigid", img_size=a.img_size,
                         only_label=side, instruction=instruction, policy=policy,
                         policy_every=(a.policy_every or HORIZON), start_z=a.start_z, perturb=a.perturb, perturb_scale=a.perturb_scale)
        # 两套判据都记下来，别混着报：
        # reach 是数据生成器的"到达"口径（≤0.16 m），coverage_m6 是 M6 的"拍到"口径
        # （≤0.7 m 且目标在视场内未遮挡）。成功率以 M6 口径为准，因为它与
        # scripted / hover-only / random 三个基线同协议。
        obs = np.load(out / f"ep_{seed}" / "obs.npz")
        scene = build_warehouse(World(cfg, "rigid"), seed)
        # 一条指令只点一个侧向（front 或 oblique），所以"可行视点"必须按指令收窄。
        # 若把整场两个可行视点都算进分母，成功率恒 ≤0.5，而且实际读数只反映
        # "这一集是不是恰好只有一个视点可见"——那是场景几何，不是策略能力。
        feas = [lab == side for _p, _y, lab in scene.viewpoints]
        covered_n, feasible_n, covered_labels = m6_coverage(
            np.asarray(obs["pos"]), yaw_from_quat(np.asarray(obs["quat"])), scene,
            center_tol=CENTER_TOL, feasible=feas)
        cov_m6 = covered_n / feasible_n if feasible_n else 0.0
        aim = np.asarray(scene.viewpoints[feas.index(True)][0], float) \
            if any(feas) else np.asarray(scene.target_pos, float)
        # 二值成功率对"学到一半"的策略没有分辨力：阈值卡在 0.12 时，
        # 看得见与盲都可能 0/60。所以再记一列连续量——全程里目标离画面中心最近那一步的 |u|。
        yaws = yaw_from_quat(np.asarray(obs["quat"]))
        uv = np.array([marker_uv(pp, yy, scene.target_pos)
                       for pp, yy in zip(np.asarray(obs["pos"]), yaws)])
        # 只在"够得着"的步上取 |u|：距**本集要去的那个视点** <= 0.7 m 的步。
        # 这里不能用"距目标 <= 0.7 m"：专家的站位本身就在目标前 1.15 m，
        # 用目标距离做门限会把专家的每一集都判成 9.0（整段没有一步够得着）。
        dist = np.linalg.norm(np.asarray(obs["pos"]) - aim[None, :], axis=1)
        near = dist <= 0.7
        absu = np.abs(uv[near, 0]) if near.any() else np.array([9.0])
        absu = absu[absu < 8.0]          # 9.0 表示目标在身后，不计
        min_abs_u = float(absu.min()) if absu.size else 9.0
        cov_at = {}
        for tol in (0.12, 0.20, 0.30):
            cn, fn, _ = m6_coverage(np.asarray(obs["pos"]), yaws, scene,
                                    center_tol=tol, feasible=feas)
            cov_at[str(tol)] = round(cn / fn if fn else 0.0, 3)
        ok = cov_m6 >= 1.0 and float(obs["pos"][-1, 2]) >= 0.05
        rows.append({"seed": seed, "side": side, "instruction": instruction,
                     "success": ok, "reach": bool(ep["reach"][0]),
                     "coverage": ep["coverage"], "coverage_m6": round(cov_m6, 3),
                     "covered_labels": covered_labels,
                     "z_end": round(float(obs["pos"][-1, 2]), 2),
                     "min_abs_u": round(min_abs_u, 4), "cov_at": cov_at,
                     "replans": getattr(policy, "replans", 0),
                     "frames": ep["frames"]})
        if k % 5 == 0 or k == len(chosen):
            rate = np.mean([r["success"] for r in rows])
            print(f"[{k}/{len(chosen)}] success={rate:.3f} elapsed={time.time()-t_start:.0f}s",
                  flush=True)

    rate = float(np.mean([r["success"] for r in rows]))
    summary = {"ckpt": str(a.ckpt), "episodes": len(rows), "success_rate": rate,
               "protocol": "m6-v1",
               "reached_rate": float(np.mean([r["reach"] for r in rows])),
               "coverage_mean": float(np.mean([r["coverage_m6"] for r in rows])),
               "coverage_arrival_mean": float(np.mean([r["coverage"] for r in rows])),
               "min_abs_u_median": float(np.median([r["min_abs_u"] for r in rows])),
                "min_abs_u_mean": float(np.mean([r["min_abs_u"] for r in rows])),
                "cov_at_mean": {t: float(np.mean([r["cov_at"][t] for r in rows]))
                                for t in ("0.12", "0.2", "0.3")},
                "secs": round(time.time() - t_start, 1), "rows": rows}
    # 成败是 0/1，区间用 Wilson 下界报告，避免 3/40 也写上"7.5%"却看不到不确定度
    p, n = rate, len(rows)
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    summary["wilson95"] = [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"success {rate:.3f}  95% Wilson {summary['wilson95']}  n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
