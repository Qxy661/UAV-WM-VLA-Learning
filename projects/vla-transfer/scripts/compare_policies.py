"""把几条策略的评测结果并成一张表，并给出配对差值的区间。

为什么不用均值的置信区间直接比：同一批 seed 跑出来的成功率是配对的，
按 seed 对齐做差值重采样，才能回答「A 比 B 高多少」而不是「A 和 B 各自是多少」。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["seed"]): int(bool(r["success"])) for r in data["rows"]}


def paired_diff(a: dict[int, int], b: dict[int, int], n_boot: int = 5000,
                seed: int = 0) -> tuple[float, float, float, int]:
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return 0.0, 0.0, 0.0, 0
    diffs = [a[s] - b[s] for s in seeds]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return point, lo, hi, len(seeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="包含各策略 metrics.json 的目录")
    ap.add_argument("--names", nargs="+", default=["scripted_eval", "bc_eval", "a3_eval"])
    ap.add_argument("--out", default="runs/compare.json")
    a = ap.parse_args()
    root = Path(a.runs)

    loaded: dict[str, dict[int, int]] = {}
    print(f"{'策略':<16}{'n':>4}{'成功率':>9}{'95% Wilson':>18}")
    for name in a.names:
        path = root / name / "metrics.json"
        if not path.exists():
            print(f"{name:<16} 缺失")
            continue
        rows = load(path)
        loaded[name] = rows
        p = sum(rows.values()) / len(rows)
        lo, hi = wilson(p, len(rows))
        print(f"{name:<16}{len(rows):>4}{p:>8.1%}  [{lo:.3f}, {hi:.3f}]")

    pairs = []
    names = [n for n in a.names if n in loaded]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d, lo, hi, n = paired_diff(loaded[names[i]], loaded[names[j]])
            sig = "不重叠" if (lo > 0 or hi < 0) else "含 0"
            pairs.append({"a": names[i], "b": names[j], "diff": d, "lo": lo, "hi": hi,
                          "n": n, "significant": bool(lo > 0 or hi < 0)})
            print(f"\n{names[i]} - {names[j]}: {d:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  "
                  f"n={n}  {sig}")

    Path(a.out).write_text(json.dumps({"table": {k: v for k, v in loaded.items()},
                                       "pairs": pairs}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
