"""从 runs/*/metrics.json 直接生成表格，避免手抄数字。

三个口径错误都是「手抄/心算」引入的（协议、判据、归一化），所以表格也要机器生成：
成功率、Wilson 区间、逐集配对检验都由脚本算，正文只引用输出。

用法：
    py -3.9 projects/vla-transfer/scripts/make_tables.py runs/a_eval runs/b_eval
    py -3.9 projects/vla-transfer/scripts/make_tables.py --pair runs/a_eval runs/b_eval
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def load(run_dir: Path) -> dict:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    m["_dir"] = run_dir
    return m


def per_episode(m: dict) -> dict[int, int]:
    """把 rows 里每个 seed 的成功与否抽出来（成功=coverage 达到 1 或 success 字段）。"""
    out: dict[int, int] = {}
    for r in m.get("rows") or []:
        seed = r.get("seed")
        if seed is None:
            continue
        ok = r.get("success")
        if ok is None:
            ok = r.get("coverage", 0.0) >= 0.999
        out[int(seed)] = 1 if ok else 0
    return out


def table(ms: list[dict]) -> str:
    head = "| 策略 | 成功率 | 95% Wilson | n | 平均覆盖 | min\\|u\\| 中位 | 用时 s |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for m in ms:
        n = int(m.get("episodes", 0))
        p = float(m.get("success_rate", float("nan")))
        k = int(round(p * n)) if n else 0
        lo, hi = wilson(k, n)
        lines.append(
            f"| {m['_dir'].name} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {n} | "
            f"{float(m.get('coverage_mean', float('nan'))):.3f} | "
            f"{float(m.get('min_abs_u_median', float('nan'))):.3f} | "
            f"{float(m.get('secs', 0)):.0f} |")
    return "\n".join(lines)


def mcnemar(a: dict[int, int], b: dict[int, int]) -> dict:
    seeds = sorted(set(a) & set(b))
    only_a = sum(1 for s in seeds if a[s] == 1 and b[s] == 0)
    only_b = sum(1 for s in seeds if a[s] == 0 and b[s] == 1)
    n = only_a + only_b
    if n == 0:
        return {"seeds": len(seeds), "only_a": 0, "only_b": 0, "p": 1.0}
    # 精确二项检验（双侧）
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"seeds": len(seeds), "only_a": only_a, "only_b": only_b,
            "p": min(1.0, 2 * tail)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从 metrics.json 生成表格")
    ap.add_argument("runs", nargs="+", help="run 目录（含 metrics.json）")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="两两配对 McNemar")
    a = ap.parse_args(argv)
    ms = [load(Path(r)) for r in a.runs]
    print(table(ms))
    if a.pair:
        ma, mb = load(Path(a.pair[0])), load(Path(a.pair[1]))
        ea, eb = per_episode(ma), per_episode(mb)
        if not ea or not eb:
            print("\n配对检验需要 metrics.json 里的 rows[].seed（本份没有）")
        else:
            r = mcnemar(ea, eb)
            print(f"\n配对（{len(set(ea) & set(eb))} 集）："
                  f"{Path(a.pair[0]).name} 独赢 {r['only_a']}，"
                  f"{Path(a.pair[1]).name} 独赢 {r['only_b']}，精确 p = {r['p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
