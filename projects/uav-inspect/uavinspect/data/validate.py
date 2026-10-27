"""数据集校验：坏数据在入库时被拒绝，而不是训练完才发现。

十二条规则，每条都能失败。`--strict` 时任何一条失败都返回非零退出码。

设计取舍：
- 只读 npz 的头几个数组，不做全量解码；一百集里坏一集能被抓到，不需要把
  3 GB 图像全读一遍。
- 划分检查在数据集层做（跨 episode），单集检查只管单集。
- 「测试集指令模板」按特征短语判断，不靠正则穷举：模板是有限的，维护在
  `instructions.py` 里，这里只查在训练集里不该出现的那些片段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = ("images", "pos", "vel", "quat", "action", "t", "frame_t", "phase")
IMAGE_SHAPE = (96, 96, 3)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class Report:
    root: str
    episodes: int = 0
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    def as_dict(self) -> dict[str, Any]:
        failed = [c.as_dict() for c in self.checks if not c.ok]
        return {
            "root": self.root,
            "episodes": self.episodes,
            "ok": self.ok,
            "n_checks": len(self.checks),
            "n_failed": len(failed),
            "failed": failed,
        }


def _quat_norm_ok(q: np.ndarray) -> bool:
    n = np.linalg.norm(q.reshape(-1, 4), axis=1)
    return bool(np.all(np.abs(n - 1.0) < 0.01))


def validate_episode(ep_dir: Path) -> list[Check]:
    out: list[Check] = []
    npz_path = ep_dir / "obs.npz"
    meta_path = ep_dir / "meta.json"
    out.append(Check("obs.npz 存在", npz_path.is_file(), str(npz_path)))
    out.append(Check("meta.json 存在", meta_path.is_file(), str(meta_path)))
    if not (npz_path.is_file() and meta_path.is_file()):
        return out

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z]
        out.append(Check("字段齐全", not missing, f"缺 {missing}" if missing else ""))
        if missing:
            return out

        images = z["images"]
        t = z["t"]
        frame_t = z["frame_t"]
        action = z["action"]
        quat = z["quat"]
        pos = z["pos"]

        out.append(Check("帧数与图像数一致", len(frame_t) == len(images),
                         f"frame_t={len(frame_t)} images={len(images)}"))
        out.append(Check("图像 shape", tuple(images.shape[1:]) == IMAGE_SHAPE,
                         f"{tuple(images.shape[1:])} != {IMAGE_SHAPE}"))
        out.append(Check("图像 dtype", images.dtype == np.uint8, str(images.dtype)))
        out.append(Check("时间严格单调",
                         bool(np.all(np.diff(t) > 0)) and bool(np.all(np.diff(frame_t) > 0)),
                         "存在回退或重复"))
        # 动作是物理量：第 0 维推力（N），后三维力矩（N·m）。不是归一化到 [-1,1] 的。
        # 这条规则抓的是「忘了换算单位」——比如推力写成归一化值 0.9 或写成 m*g*a。
        thrust = action[:, 0]
        torque = action[:, 1:]
        out.append(Check("推力在 [0,30] N", bool(np.all(thrust >= 0.0) and np.all(thrust <= 30.0)),
                         f"[{thrust.min():.2f}, {thrust.max():.2f}] N"))
        # 力矩上限按这套仿真的惯例给 8 N·m：级联的角速率环 gain 较大，偏航猛转时
        # 力矩确实能到 5 N·m 量级，这不是单位错。
        out.append(Check("力矩在 [-8,8] N·m",
                         bool(np.all(np.abs(torque) <= 8.0)),
                         f"max |tau| = {np.abs(torque).max():.3f}"))
        # 推力中位数必须落在悬停附近：m*g = 9.81 N 的 0.5–2.0 倍。
        # 这条专门抓「少乘/多乘了一个量纲」——[0,30] N 的区间太宽，抓不到。
        med = float(np.median(thrust))
        out.append(Check("推力中位数接近悬停", 0.5 * 9.81 <= med <= 2.0 * 9.81,
                         f"median = {med:.2f} N"))
        out.append(Check("四元数模长为 1", _quat_norm_ok(quat), ""))
        out.append(Check("无 NaN", not any(np.isnan(a).any() for a in (pos, quat, action, t)),
                         ""))
        out.append(Check("帧时间在控制时间范围内",
                         bool(frame_t[0] >= t[0] - 1e-6 and frame_t[-1] <= t[-1] + 1e-6),
                         f"frame [{frame_t[0]:.3f},{frame_t[-1]:.3f}] t [{t[0]:.3f},{t[-1]:.3f}]"))
        cov = meta.get("coverage")
        out.append(Check("meta.coverage 存在且在 [0,1]",
                         isinstance(cov, (int, float)) and 0.0 <= float(cov) <= 1.0, str(cov)))
        out.append(Check("meta.instruction 非空", bool(str(meta.get("instruction", "")).strip()), ""))
        out.append(Check("meta.split 合法", meta.get("split") in ("train", "val", "test"),
                         str(meta.get("split"))))
        # 抽查前 8 帧是否有完全重复（同一帧贴两次是最常见的录制 bug）
        dup = 0
        head = images[: min(8, len(images))]
        for i in range(len(head)):
            for j in range(i + 1, len(head)):
                if np.array_equal(head[i], head[j]):
                    dup += 1
        out.append(Check("前 8 帧无整帧重复", dup == 0, f"{dup} 对重复"))
    return out


def validate_dataset(root: str | Path, *, max_episodes: int | None = None) -> Report:
    root = Path(root)
    rep = Report(root=str(root))
    eps = sorted(root.glob("shard_*/episode_*")) or sorted(root.glob("episode_*"))
    if max_episodes:
        eps = eps[:max_episodes]
    rep.episodes = len(eps)
    rep.add("找到 episode", rep.episodes > 0, f"{rep.episodes} 集")

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    tasks: dict[str, set[str]] = {"train": set(), "test": set()}
    for ep in eps:
        checks = validate_episode(ep)
        bad = [c for c in checks if not c.ok]
        rep.add(f"{ep.name} 单集", not bad,
                "; ".join(f"{c.name}: {c.detail}" for c in bad[:3]))
        meta_path = ep / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sp = meta.get("split")
        seed = meta.get("seed", meta.get("episode", -1))
        if sp in splits:
            splits[sp].append(int(seed))
        if sp in tasks:
            tasks[sp].add(str(meta.get("instruction", "")))

    overlap_tv = set(splits["train"]) & set(splits["val"])
    overlap_tt = set(splits["train"]) & set(splits["test"])
    rep.add("train/val seed 不重叠", not overlap_tv, f"{len(overlap_tv)} 个")
    rep.add("train/test seed 不重叠", not overlap_tt, f"{len(overlap_tt)} 个")
    if rep.episodes >= 50:
        rep.add("三个划分都非空", all(splits[k] for k in ("train", "val", "test")),
                {k: len(v) for k, v in splits.items()})

    held_out_markers = ("绕到", "藏在", "先把机头对准", "那边有东西", "沿通道去")
    leaked = [t for t in tasks["train"] if any(m in t for m in held_out_markers)]
    rep.add("训练集不含测试指令模板", not leaked, f"{len(leaked)} 条泄漏")
    return rep
