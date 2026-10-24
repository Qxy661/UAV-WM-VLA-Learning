"""M6：三个基线、一个任务、一套口径，出一张可比的表。

用法：
    py -3.9 projects/uav-inspect/scripts/m6_eval.py
    py -3.9 projects/uav-inspect/scripts/m6_eval.py --episodes 20 --seeds 3
    py -3.9 projects/uav-inspect/scripts/m6_eval.py --demo-mismatch   # 演示"不可横比"的拒绝

产物：runs/m6_eval/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.config import load                                  # noqa: E402
from uavinspect.eval.harness import (TaskSpec, compare, evaluate,   # noqa: E402
                                     to_json)
from uavinspect.eval.policies import (HoverPolicy, RandomPolicy,    # noqa: E402
                                      WaypointPolicy)


def fmt_rate(s: dict) -> str:
    if s.get("rate") is None:
        return "   —   "
    lo, hi = s["ci95"]
    return f"{s['rate'] * 100:5.1f}% [{lo * 100:4.1f},{hi * 100:4.1f}]"


def fmt_mean(s: dict, unit: str = "", digits: int = 3) -> str:
    if s.get("mean") is None:
        return "   —   "
    lo, hi = s["ci95"]
    return f"{s['mean']:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]{unit}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M6 三基线统一口径评测")
    ap.add_argument("--config", default=str(PKG_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(PKG_ROOT / "runs" / "m6_eval"))
    ap.add_argument("--episodes", type=int, default=20, help="每个 seed 的回合数")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--demo-mismatch", action="store_true")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    task = TaskSpec(name="inspection", duration=args.duration,
                    description="仓储通道巡检：飞到视点、目标可见即覆盖")

    makers = {
        "hover-only": lambda s: HoverPolicy(cfg),
        "random": lambda s: RandomPolicy(cfg, seed=1000 + s),
        "scripted-waypoints": lambda s: WaypointPolicy(cfg),
    }
    reports = []
    for name, make in makers.items():
        seeds = list(range(args.seeds * args.episodes))
        rep = evaluate(name, task, seeds, make_policy=make)
        reports.append(rep)
        print(f"跑完 {name}：{len(rep.episodes)} 集")

    # 先比 protocol，再比数：口径不一致就拒绝，避免把不可比的数放进一张表
    compare(reports)

    print()
    hdr = f"{'策略':<20}{'成功率':>22}{'覆盖率':>24}{'最小间距 m':>22}{'覆盖用时 s':>22}"
    print(hdr)
    print("-" * len(hdr.encode('gbk', 'ignore').decode('gbk')))
    s = {r.agent: r.summary() for r in reports}
    for name in makers:
        d = s[name]
        print(f"{name:<20}{fmt_rate(d['success']):>22}{fmt_mean(d['coverage']):>24}"
              f"{fmt_mean(d['min_clearance']):>22}{fmt_mean(d['time_to_coverage'], digits=1):>22}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task.name,
        "protocol": reports[0].protocol,
        "agents": [r.summary() for r in reports],
        "note": "汇总按 episode 聚合，区间为按集重采样的 95% bootstrap；"
                "coverage 的分母只含相机几何上看得见的视点。",
    }
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n产物：{out / 'metrics.json'}")

    if args.demo_mismatch:
        shorter = TaskSpec(name="inspection", duration=args.duration / 2)
        rep2 = evaluate("scripted-waypoints", shorter, [0, 1], make_policy=makers["scripted-waypoints"])
        try:
            compare([reports[2], rep2])
        except ValueError as exc:
            print(f"\n拒绝比较（这就是要的行为）：\n  {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
