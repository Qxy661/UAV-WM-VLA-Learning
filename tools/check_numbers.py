"""核对 demo 打印的关键数字与仓库里存的基线是否一致。

每个参与核对的脚本用 `common.metrics.emit(...)` 打一行 `[metrics] k=v ...`，
这里把那一行抓出来，与 `code/expected/metrics.json` 逐项比。

    python tools/check_numbers.py            # 比对，退出码 1 表示有漂移
    python tools/check_numbers.py --write    # 重写基线（改动过实验后才有意义）
    python tools/check_numbers.py --only a_  # 只跑名字里带 a_ 的脚本

为什么要有它：文档里引用的数字是从这些脚本跑出来的。脚本一改，文档就可能失真，
而人不会每次都记得去改。让 CI 记住这件事，比让人记住便宜。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
BASELINE = CODE / "expected" / "metrics.json"

# 参与核对的脚本：跑得快、数字稳（不依赖随机初始化的收敛细节）
SCRIPTS = [
    "e_gsc_safety.py",
    "c_action_heads.py",
    "a_worldmodel_rssm.py",
    "t_wm_diagnostics.py",
    "u_jepa_probe.py",
    "x_hallucination_metric.py",
]

LINE = re.compile(r"^\[metrics\]\s+(.*)$", re.MULTILINE)
TOL_REL = 1e-3
TOL_ABS = 1e-6


def run_script(name: str) -> dict:
    """跑一个脚本，返回它 emit 出来的键值。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(CODE / name)],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=1800,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
        raise SystemExit(f"{name} 跑失败（退出码 {proc.returncode}）：\n{tail}")

    got = {}
    for line in LINE.findall(proc.stdout):
        for token in line.split():
            if "=" not in token:
                continue
            key, _, raw = token.partition("=")
            try:
                got[key] = float(raw)
            except ValueError:
                got[key] = raw
    if not got:
        raise SystemExit(f"{name} 没有输出 [metrics] 行，检查是否导入了 common.metrics")
    return got


def close(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= max(TOL_ABS, TOL_REL * max(abs(a), abs(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="把当前结果写成新的基线")
    ap.add_argument("--only", default="", help="只跑文件名含该子串的脚本")
    args = ap.parse_args()

    scripts = [s for s in SCRIPTS if args.only in s]
    current, measured_in = {}, {}
    for name in scripts:
        got = run_script(name)
        for k, v in got.items():
            if k in current:
                print(f"  [重复键] {k} 同时出现在 {measured_in[k]} 与 {name}")
            current[k] = v
            measured_in[k] = name
        print(f"  {name:<32} {len(got)} 项")

    if args.write:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"\n已写入基线 {BASELINE.relative_to(ROOT)}（{len(current)} 项）")
        return 0

    if not BASELINE.exists():
        print("没有基线文件，先跑一次 --write", file=sys.stderr)
        return 1
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    bad = []
    for k, want in baseline.items():
        if k not in current:
            bad.append((k, want, "<缺失>", measured_in.get(k, "?")))
        elif not close(current[k], want):
            bad.append((k, want, current[k], measured_in.get(k, "?")))
    for k in current:
        if k not in baseline:
            bad.append((k, "<新增>", current[k], measured_in[k]))

    if not bad:
        print(f"\n{len(baseline)} 项全部一致（相对容差 {TOL_REL:g}）")
        return 0

    print(f"\n{len(bad)} 项与基线不一致：", file=sys.stderr)
    print(f"{'键':<26}{'基线':>14}{'本次':>14}  来源", file=sys.stderr)
    for k, want, got, src in bad:
        w = f"{want:>14.6g}" if isinstance(want, (int, float)) else f"{want:>14}"
        g = f"{got:>14.6g}" if isinstance(got, (int, float)) else f"{got:>14}"
        print(f"{k:<26}{w}{g}  {src}", file=sys.stderr)
    print("\n若这一改动是有意的，跑 --write 更新基线，并同步文档里引用的数字。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
