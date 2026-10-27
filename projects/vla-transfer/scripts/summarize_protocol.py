"""把 run_protocol.sh 的多个种子结果汇成一张可报告的表。

用法：
    python projects/vla-transfer/scripts/summarize_protocol.py runs/p_subgoal_s0_eval runs/p_subgoal_s1_eval
    python ... --baseline runs/sg_rel2_eval        # 加一行对照

输出三块：
1. 逐种子明细（success、n、合并 Wilson、min|u| 中位）
2. 跨种子汇总（mean ± std；合并回合的 Wilson 区间）
3. 与基线的配对检验（只在两边回合集合完全一致时做 McNemar；否则只列差值）
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def load(d: Path) -> dict:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    rows = m.get("rows") or []
    n = int(m.get("n") or len(rows))
    succ = m.get("success")
    if succ is None:
        succ = (sum(1 for r in rows if r.get("success")) / len(rows)) if rows else 0.0
    us = [r.get("min_abs_u") for r in rows if r.get("min_abs_u") is not None]
    return {"dir": d.name, "n": n, "success": float(succ),
            "k": int(round(float(succ) * n)), "u_med": st.median(us) if us else float("nan"),
            "ep": {r.get("seed"): bool(r.get("success")) for r in rows if r.get("seed") is not None}}


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确检验：b、c 是两个策略分歧的回合数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()

    rs = [load(Path(r)) for r in a.runs]
    print("| 种子 | n | success | 95% Wilson | min\\|u\\| 中位 |")
    print("|---|---|---|---|---|")
    for r in rs:
        lo, hi = wilson(r["k"], r["n"])
        print(f"| {r['dir']} | {r['n']} | {r['success']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['u_med']:.3f} |")

    ss = [r["success"] for r in rs]
    k_all = sum(r["k"] for r in rs)
    n_all = sum(r["n"] for r in rs)
    lo, hi = wilson(k_all, n_all)
    mean = sum(ss) / len(ss)
    sd = st.stdev(ss) if len(ss) > 1 else 0.0
    print()
    print(f"**跨 {len(rs)} 个种子：success {mean:.3f} ± {sd:.3f}；合并回合 {k_all}/{n_all} = "
          f"{k_all / n_all:.3f}，95% Wilson [{lo:.3f}, {hi:.3f}]**")
    if len(rs) > 1:
        print(f"（跨种子 std 衡量训练随机性；合并 Wilson 衡量评测样本量，两者不可互替）")

    if a.baseline:
        b = load(Path(a.baseline))
        keys = set(b["ep"])
        paired = all(keys and set(r["ep"]) == keys for r in rs)
        if paired:
            print(f"\n配对检验（每个种子各自与基线比，{len(keys)} 个共同回合）：")
            print("| 种子 | 本方案胜 | 基线胜 | 平 | McNemar p |")
            print("|---|---|---|---|---|")
            for r in rs:
                win_a = sum(1 for k in keys if r["ep"][k] and not b["ep"][k])
                win_b = sum(1 for k in keys if b["ep"][k] and not r["ep"][k])
                tie = len(keys) - win_a - win_b
                print(f"| {r['dir']} | {win_a} | {win_b} | {tie} | {mcnemar_exact(win_a, win_b):.4f} |")
            print("\n注：多个种子各自检验会放大第一类错误；只把最强的那个种子当结论、"
                  "其余作为重复性证据，比把三个 p 值挑最小更诚实。")
        else:
            print(f"\n基线 {b['dir']} 的回合集合与本方案不一致，只列差值："
                  f"{sum(ss) / len(ss) - b['success']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
