"""E10 多机分配报告：4 机 6 点，丢包 0/15/30%，与精确最优比。

    python projects/uav-inspect/scripts/e10_report.py --seeds 20

四张表：
1. 分配器 × 丢包 → makespan / 最优（顺序拍卖 vs 全局贪心）
2. 掉线重分配：通信耗时（按 20 ms 单向延迟估）与重排轮次
3. 消息量与出价丢失率
4. DP 最优的交叉验证（小规模穷举）
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects" / "uav-inspect"))

from uavinspect.fleet.allocation import (  # noqa: E402
    Link, all_assignments, auction_allocate, cost_matrix, greedy_global,
    optimal_makespan, random_problem, reassign_after_failure,
)


def brute_force(c: np.ndarray) -> float:
    n_d, n_t = c.shape
    best = float("inf")
    for combo in all_assignments(n_t, n_d):
        loads = [0.0] * n_d
        for tid, did in enumerate(combo):
            loads[did] += c[did][tid]
        best = min(best, max(loads))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--drones", type=int, default=4)
    ap.add_argument("--delay-ms", type=float, default=20.0)
    a = ap.parse_args()

    checks = []
    for s in range(5):
        tasks, drones = random_problem(a.tasks, a.drones, seed=1000 + s)
        c = cost_matrix(tasks, drones)
        checks.append(abs(optimal_makespan(c) - brute_force(c)))
    dp_ok = bool(max(checks) < 1e-9)

    delay = a.delay_ms / 1000.0
    rows = []
    for loss in (0.0, 0.15, 0.30):
        auc_gap, grd_gap, reas_ms, msgs, lost = [], [], [], [], []
        for s in range(a.seeds):
            tasks, drones = random_problem(a.tasks, a.drones, seed=s)
            c = cost_matrix(tasks, drones)
            opt = optimal_makespan(c)

            asg = auction_allocate(tasks, drones, link=Link(loss=loss, delay_s=delay, seed=s))
            auc_gap.append(asg.makespan(c, tasks) / opt)
            msgs.append(asg.messages)
            lost.append(asg.bids_lost)

            grd = greedy_global(tasks, drones, link=Link(loss=loss, delay_s=delay, seed=s))
            grd_gap.append(grd.makespan(c, tasks) / opt)

            _, t_comm = reassign_after_failure(tasks, drones, drones[0].did, assignment=asg,
                                               link=Link(loss=loss, delay_s=delay, seed=s))
            reas_ms.append(t_comm * 1000.0)

        rows.append({
            "loss": loss,
            "auction_gap_mean": round(st.mean(auc_gap), 4),
            "auction_gap_p90": round(float(np.percentile(auc_gap, 90)), 4),
            "auction_gap_worst": round(max(auc_gap), 4),
            "greedy_gap_mean": round(st.mean(grd_gap), 4),
            "reassign_comm_ms": round(st.mean(reas_ms), 2),
            "messages_per_seed": int(st.mean(msgs)),
            "bid_loss_rate": round(st.mean(lost) / max(st.mean(msgs), 1), 4),
        })

    print(json.dumps({"dp_matches_bruteforce": dp_ok, "tasks": a.tasks, "drones": a.drones,
                      "seeds": a.seeds, "delay_ms": a.delay_ms, "rows": rows},
                     ensure_ascii=False, indent=2))
    print()
    print("| 丢包 | 顺序拍卖 gap | 全局贪心 gap | 重分配通信 ms | 出价丢失率 |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['loss']:.0%} | {r['auction_gap_mean']:.3f} "
              f"(worst {r['auction_gap_worst']:.3f}) | {r['greedy_gap_mean']:.3f} "
              f"| {r['reassign_comm_ms']:.1f} | {r['bid_loss_rate']:.1%} |")
    return 0 if dp_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
