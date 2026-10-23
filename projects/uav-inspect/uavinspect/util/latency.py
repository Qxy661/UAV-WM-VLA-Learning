"""分阶段耗时统计。

只报 P50/P99，不报均值（ADR-005）：均值会把尾部掩盖掉，而部署决策看的是 P99——
"平均 8 ms"在 1% 的帧里变成 80 ms，就是一次炸机。
"""

from __future__ import annotations

import numpy as np


class StageTimer:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, stage: str, seconds: float) -> None:
        self._samples.setdefault(stage, []).append(float(seconds))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, xs in self._samples.items():
            a = np.asarray(xs, dtype=float) * 1000.0   # 统一毫秒
            out[stage] = {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "p50_ms": float(np.percentile(a, 50)),
                "p90_ms": float(np.percentile(a, 90)),
                "p99_ms": float(np.percentile(a, 99)),
                "max_ms": float(a.max()),
            }
        return out

    def table(self) -> str:
        s = self.summary()
        if not s:
            return "(无采样)"
        head = f"{'阶段':<10}{'n':>7}{'P50/ms':>10}{'P90/ms':>10}{'P99/ms':>10}{'max/ms':>10}"
        lines = [head, "-" * len(head.encode('gbk', errors='replace'))]
        for stage in sorted(s):
            v = s[stage]
            lines.append(
                f"{stage:<10}{v['n']:>7d}{v['p50_ms']:>10.3f}{v['p90_ms']:>10.3f}"
                f"{v['p99_ms']:>10.3f}{v['max_ms']:>10.3f}"
            )
        return "\n".join(lines)
