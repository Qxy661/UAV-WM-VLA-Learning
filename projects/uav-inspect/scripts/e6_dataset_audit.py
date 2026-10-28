"""E6 数据审计：划分泄漏、重复场景、指令覆盖率、可用率。

四个问题都是「数据看起来齐了但实际不可用」的常见形态，且都不需要训练就能查：
1. 同一个 seed 出现在两个划分里（泄漏）；
2. 同一 (layout, target_pos) 被多条 episode 复用（重复场景，验证集虚高）；
3. 指令模板覆盖率（若全是同一句，语言条件等于没有）；
4. coverage < 0.5 的截断轨迹占比（专家自己没走完）。

用法：
    py -3.9 projects/uav-inspect/scripts/e6_dataset_audit.py --root /root/work/data/uav96c \
        --root /root/work/data/uav96b --out projects/uav-inspect/runs/e6_audit
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def audit_root(root: Path) -> dict:
    metas = sorted(root.glob("shard_*/episode_*/meta.json"))
    seeds: dict[str, list[str]] = collections.defaultdict(list)
    scenes: dict[str, list[str]] = collections.defaultdict(list)
    instr: collections.Counter[str] = collections.Counter()
    cov_ok = cov_bad = 0
    per_split: collections.Counter[str] = collections.Counter()
    frames = 0
    for mp in metas:
        m = json.loads(mp.read_text(encoding="utf-8"))
        split = m.get("split", "?")
        per_split[split] += 1
        seeds[str(m.get("seed", mp.parent.name))].append(split)
        key = f"{m.get('layout', 'a')}|{m.get('target_pos')}"
        scenes[key].append(mp.parent.name)
        instr[m.get("instruction", "")] += 1
        c = float(m.get("coverage", 0.0))
        if c >= 0.5:
            cov_ok += 1
        else:
            cov_bad += 1
        frames += int(m.get("frames", 0))

    leaked = {s: v for s, v in seeds.items() if len(set(v)) > 1}
    dup_scene = {k: v for k, v in scenes.items() if len(v) > 1}
    n = len(metas)
    return {
        "root": str(root),
        "episodes": n,
        "frames": frames,
        "splits": dict(per_split),
        "seed_leak": len(leaked),
        "seed_leak_examples": dict(list(leaked.items())[:3]),
        "duplicate_scene_groups": len(dup_scene),
        "duplicate_scene_examples": {k: v[:3] for k, v in list(dup_scene.items())[:3]},
        "distinct_instructions": len(instr),
        "instruction_top": instr.most_common(3),
        "coverage_ok": cov_ok,
        "coverage_bad": cov_bad,
        "coverage_ok_rate": round(cov_ok / n, 4) if n else 0.0,
    }


def digest(rows: list[dict]) -> str:
    """把审计结果压成一个短哈希，用来判断两批数据是不是同一份。"""
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据审计：泄漏/重复/指令覆盖/可用率")
    ap.add_argument("--root", action="append", required=True, help="数据集根（可多次给）")
    ap.add_argument("--out", default="", help="写 metrics.json 的目录")
    a = ap.parse_args(argv)

    rows = [audit_root(Path(r)) for r in a.root]
    print("| 数据集 | episode | 帧 | 划分 | seed 泄漏 | 重复场景组 | 指令种类 | 可用率 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {Path(r['root']).name} | {r['episodes']} | {r['frames']} | "
              f"{','.join(f'{k}:{v}' for k, v in sorted(r['splits'].items()))} | "
              f"{r['seed_leak']} | {r['duplicate_scene_groups']} | "
              f"{r['distinct_instructions']} | {r['coverage_ok_rate']:.3f} |")

    verdict = {
        "leak_free": all(r["seed_leak"] == 0 for r in rows),
        "instruction_diverse": all(r["distinct_instructions"] > 1 for r in rows),
        "usable": all(r["coverage_ok_rate"] >= 0.5 for r in rows),
    }
    digest16 = digest(rows)
    print(f"\n判定：{verdict}\n数据指纹：{digest16}")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            json.dumps({"roots": rows, "verdict": verdict, "digest": digest16},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {out/'metrics.json'}")
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
