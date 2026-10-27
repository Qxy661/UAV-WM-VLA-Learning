"""DAgger-lite：让**策略自己飞**，在它飞到的地方问专家"这里该怎么飞"，把标签记下来。

和 `gen_dataset.py` 的区别只有一处：那条轨迹是策略产生的，动作标签是专家的。
这正是 DAgger 想修的东西——专家数据只覆盖专家会走的状态，策略一旦偏离就没人告诉它
该怎么回来（RESULTS.md 第二层失败）。这里不做「聚合多轮」，只跑一轮，所以叫 lite。

用法（开发机，从 /root/work/repo 起跑）：

    python projects/vla-transfer/scripts/dagger_lite.py \\
        --ckpt runs/bc_r5/ckpt.pt --episodes 600 --residual \\
        --out /root/work/data/uav96d --policy-every 4

产物与 gen_dataset 同格式（obs.npz + meta.json + manifest.json），可以直接喂
`npz_to_lerobot_manual.py` 与 `train_bc.py`，split 记成 train。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "projects" / "uav-inspect"))
sys.path.insert(0, str(REPO / "projects" / "vla-transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # eval_uav.py 在同目录，不是包

from uavinspect.config import load as load_config            # noqa: E402
from uavinspect.data.instructions import make_instruction   # noqa: E402
from uavinspect.data.recorder import compute_stats, write_manifest   # noqa: E402
from uavinspect.pipeline.inspect import run_episode          # noqa: E402
from uavinspect.sim.scene import build_warehouse    # noqa: E402
from uavinspect.sim.world import World                       # noqa: E402

SEED_BASE = 10000          # 和 gen_dataset 的 0..7999 隔开，避免重号


class DaggerPolicy:
    """策略出动作去飞，同时把专家的动作挂到 last_label 上。

    `run_episode` 记录时优先用 `label()`（现问一次专家，见 pipeline/inspect.py 的 _label），
    所以同一个 episode 里：轨迹是策略的，动作标签是专家的。
    早前版本只在策略被调用的那几拍更新 `last_label`，中间几拍记下的是陈旧标签，
    图文错位，DAgger 数据因此把看得见的 BC 打到 0.000。
    """

    def __init__(self, inner, residual: bool = False):
        self.inner = inner
        self.residual = residual
        self.last_label = None
        self.labels = 0
        self.ctrl = None

    def __call__(self, ctx):
        self.ctrl = ctx.get("ctrl")
        thrust, torque = self.inner(ctx)
        ex_t, ex_tau = self.ctrl.compute(ctx["st"], ctx["ref"], yaw_ref=ctx["yaw_ref"])
        self.last_label = (float(ex_t), np.asarray(ex_tau, dtype=float))
        return thrust, torque

    def label(self, st, ref, yaw_ref=0.0):
        """记录时现问专家：这一拍在这个状态上该怎么飞。"""
        if self.ctrl is None:
            return self.last_label
        t, tau = self.ctrl.compute(st, ref, yaw_ref=yaw_ref)
        self.labels += 1
        return float(t), np.asarray(tau, dtype=float)

    def latency_summary(self):
        return self.inner.latency_summary() if hasattr(self.inner, "latency_summary") else {}


def build_policy(args, instruction: str):
    from eval_uav import ChunkPolicy     # 复用同一条推理路径，保证与评测一致

    ckpt = Path(args.ckpt).resolve()
    return ChunkPolicy(ckpt, instruction, device=args.device,
                       action_space=args.action_space, residual=args.residual)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DAgger-lite：策略飞行 + 专家标签")
    ap.add_argument("--config", default=str(REPO / "projects" / "uav-inspect" / "configs" / "default.yaml"))
    ap.add_argument("--ckpt", required=True, help="当前策略的 ckpt（BC 或残差 BC）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--capture-every", type=int, default=5)
    ap.add_argument("--policy-every", type=int, default=4, help="每几个控制步重规划一次")
    ap.add_argument("--backend", default="rigid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--action-space", default="torque", choices=["torque", "omega"])
    ap.add_argument("--perturb", action="store_true", help="从扰动状态起步，让策略先偏离再被标注")
    ap.add_argument("--perturb-scale", type=float, default=0.5)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    eps: list[dict] = []
    skipped = 0
    t_all = time.perf_counter()
    for i in range(args.episodes):
        seed = args.seed_base + i
        side = "front" if seed % 2 == 0 else "oblique"
        ep_id = f"episode_{seed:05d}"
        # 指令要"先于飞行"确定：策略构造时就要知道自己在执行哪一句。
        # 只建一次场景（不推进物理）拿目标所在货架号，比空跑一集快两个数量级。
        with World(cfg, "rigid") as w:
            scene0 = build_warehouse(w, seed=seed)
            rack = int(np.argmin(np.abs(np.asarray(scene0.rack_x) - scene0.target_pos[0]))) + 1
        instruction = make_instruction(rack, side, random.Random(seed * 7919 + 13))
        try:
            inner = build_policy(args, instruction)
            pol = DaggerPolicy(inner, residual=args.residual)
            meta = run_episode(cfg, seed, out / ep_id, args.backend,
                               img_size=args.img_size, capture_every=args.capture_every,
                               only_label=side, instruction=instruction,
                               policy=pol, policy_every=args.policy_every,
                               perturb=args.perturb, perturb_scale=args.perturb_scale,
                               action_space=args.action_space)
        except ValueError:
            skipped += 1
            continue

        meta["id"] = ep_id
        meta["dir"] = ep_id
        meta["split"] = "train"        # DAgger 数据只进训练集
        meta["viewpoint"] = side
        meta["instruction"] = instruction
        meta["source"] = "dagger-lite"
        meta["rack"] = rack
        meta["expert_labels"] = int(getattr(pol, "labels", 0))
        (out / ep_id / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        eps.append(meta)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{args.episodes}] 集，帧 {sum(e['frames'] for e in eps)}，"
                  f"用时 {time.perf_counter() - t_all:.0f}s", flush=True)

    manifest = write_manifest(out, eps, val_every=1 << 30)
    for e in manifest["episodes"]:
        e["split"] = "train"
    stats = compute_stats(out, manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": len(eps), "skipped": skipped,
                      "frames": int(sum(e["frames"] for e in eps)),
                      "wall_time_s": round(time.perf_counter() - t_all, 1)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
