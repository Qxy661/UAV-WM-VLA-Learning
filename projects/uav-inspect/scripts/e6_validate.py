"""E6 数据平台：一条命令判断一批数据能不能进训练集。

用法：
    python scripts/e6_validate.py --root /root/work/data/uav96
    python scripts/e6_validate.py --root /root/work/data/uav96 --strict   # 有失败就退出 1
    python scripts/e6_validate.py --root ... --json runs/e6/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavinspect.data.validate import validate_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rep = validate_dataset(a.root, max_episodes=a.max_episodes)
    body = rep.as_dict()
    print(f"数据集 {body['root']}：{body['episodes']} 集，检查 {body['n_checks']} 条，"
          f"失败 {body['n_failed']} 条")
    for item in body["failed"]:
        print(f"  ✗ {item['name']}  {item['detail']}")
    if body["ok"]:
        print("  全部通过")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告写入 {out}")

    if a.strict and not body["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
