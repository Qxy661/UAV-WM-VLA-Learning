"""把实验里的关键数字显式打出来，供 `tools/check_numbers.py` 与文档对账。

用法：

    from common.metrics import emit
    emit(convex_off_hit=100.0, convex_on_hit=0.0)

终端会多出一行 `[metrics] convex_off_hit=100 convex_on_hit=0`；
如果设置了环境变量 `UAV_METRICS_OUT`，还会把它们并进那个 JSON 文件。

设计上有意不引第三方库：CI 里跑数字核对时，脚本本身要能独立跑起来。
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _plain(v):
    """把 numpy / torch 的标量转成 JSON 能存的 Python 标量。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return round(v, 6)
    try:                                    # numpy / torch 标量都有 .item()
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


def emit(**kv) -> None:
    """打印一行 `[metrics] k=v ...`，并按需追加到 JSON 文件。"""
    items = {k: _plain(v) for k, v in kv.items()}
    line = " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in items.items())
    print(f"[metrics] {line}")

    out = os.environ.get("UAV_METRICS_OUT")
    if not out:
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(items)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
