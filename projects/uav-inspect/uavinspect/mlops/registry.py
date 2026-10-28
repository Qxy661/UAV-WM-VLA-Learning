"""run 注册表：把「哪个模型、什么数据、什么配置、多少分」绑成一条记录。

为什么要有它：现在 `runs/*/metrics.json` 里只有数字，没有「这份数字是哪份数据、
哪个 commit、哪套配置跑出来的」。三件事一旦分开存放，就会出现本文档开头警告过的
情形——把两次不同口径的跑分放在一张表里比较。注册表把三者写进同一条 JSONL 记录，
report 只读注册表，不猜。

一条记录长这样（JSONL 一行）：
    {"run_id": "m1_multi-20261028T1145", "kind": "m1", "git": "0a564d7",
     "data_hash": "sha256:...", "config": {...}, "metrics": {...}, "ckpt": null}
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "runs" / "registry.jsonl"


def _git_sha(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True,
            stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def data_hash(root: str | Path | None, *, limit: int = 4000) -> str | None:
    """数据指纹：路径里每个文件的大小求和再哈希。

    不读文件内容——几十 GB 的数据集读一遍要几分钟，而「同名文件大小变了」
    已经足够发现换数据；真要防篡改，用 `--full` 走内容哈希。
    """
    if root is None:
        return None
    root = Path(root)
    if not root.exists():
        return None
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(root)}\t{p.stat().st_size}\n".encode())
            n += 1
            if n >= limit:
                break
    return "sha256:" + h.hexdigest()[:16] if n else None


def register(run_dir: str | Path, *, kind: str | None = None, data_root=None,
             repo_root: str | Path | None = None, metrics: dict | None = None,
             config: dict | None = None, ckpt: str | None = None) -> dict:
    run_dir = Path(run_dir)
    m = metrics if metrics is not None else json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = config
    if cfg is None and (run_dir / "env.json").exists():
        env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
        cfg = env.get("config") if isinstance(env, dict) else None
    rec = {
        "run_id": f"{run_dir.name}-{time.strftime('%Y%m%dT%H%M')}",
        "run_dir": str(run_dir),
        "kind": kind or run_dir.name.split("_")[0],
        "git": _git_sha(Path(repo_root) if repo_root else None),
        "data_hash": data_hash(data_root),
        "config": cfg,
        "metrics": m,
        "ckpt": ckpt,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(x) for x in REGISTRY.read_text(encoding="utf-8").splitlines() if x.strip()]


def best(metric: str = "success_rate") -> dict | None:
    rows = [(r["metrics"].get(metric), r) for r in load()]
    rows = [(v, r) for v, r in rows if isinstance(v, (int, float))]
    return max(rows, key=lambda x: x[0])[1] if rows else None
