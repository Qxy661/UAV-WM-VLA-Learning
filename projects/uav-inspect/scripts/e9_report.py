"""E9 报告：把 runs/ 下的结果登记进注册表，并按指标排表。

用法：
    py -3.9 projects/uav-inspect/scripts/e9_report.py --register      # 扫描并登记
    py -3.9 projects/uav-inspect/scripts/e9_report.py                 # 只读注册表排表
    py -3.9 projects/uav-inspect/scripts/e9_report.py --best success_rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(PKG_ROOT))

from uavinspect.mlops import drift, registry  # noqa: E402

METRICS_OF_INTEREST = ("success_rate", "coverage", "min_sep_m", "passed", "roll_mse", "vis_auc")


def scan_runs(runs_root: Path) -> list[Path]:
    return sorted(p.parent for p in runs_root.glob("*/metrics.json"))


def register_all(runs_root: Path) -> int:
    n = 0
    for d in scan_runs(runs_root):
        registry.register(d)
        n += 1
    return n


def render(rows: list[dict]) -> str:
    if not rows:
        return "（注册表为空：先跑 --register）"
    head = ["run_id", "kind", "git", "data_hash", "success_rate", "其他指标"]
    out = [" | ".join(head), "|".join(["---"] * len(head))]
    for r in rows:
        m = r.get("metrics") or {}
        sr = m.get("success_rate", m.get("passed", "—"))
        extra = ", ".join(f"{k}={m[k]}" for k in METRICS_OF_INTEREST[1:] if k in m)
        out.append(" | ".join([str(r.get("run_id")), str(r.get("kind")), str(r.get("git")),
                               str(r.get("data_hash")), str(sr), extra or "—"]))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E9 模型运维报告")
    ap.add_argument("--runs", default=str(PKG_ROOT / "runs"))
    ap.add_argument("--register", action="store_true", help="扫描 runs/ 并把结果写进注册表")
    ap.add_argument("--best", default="", help="按这个指标取最优 run")
    ap.add_argument("--drift-ref", default="", help="参考数据根（算亮度 PSI）")
    ap.add_argument("--drift-cur", default="", help="待检数据根")
    a = ap.parse_args(argv)

    if a.register:
        n = register_all(Path(a.runs))
        print(f"登记 {n} 个 run → {registry.REGISTRY}")
    if a.drift_ref and a.drift_cur:
        res = drift.check(drift.reference_from_episodes(a.drift_ref),
                          drift.reference_from_episodes(a.drift_cur))
        print(json.dumps({"drift": res}, ensure_ascii=False, indent=2))
    rows = registry.load()
    print(render(rows))
    if a.best:
        b = registry.best(a.best)
        print(f"\n最优 {a.best}: {b['run_id']} = {(b.get('metrics') or {}).get(a.best)}" if b else "\n没有可比记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
