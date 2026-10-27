"""把巡检 npz 转成 LeRobot v2.1 格式（π0 的数据加载器只吃这个格式）。

为什么要转：openpi 的数据管线（动作分块、归一化、统计量、shuffle）全建在 `LeRobotDataset` 上，
绕开它等于重写训练管线，而且和官方代码分叉后没法对齐。

四个取舍：
- 图像存逐帧 PNG（`dtype: "image"`），不编码视频：pi0 环境里没有 torchcodec，96² 的合成图
  视频压缩收益也小。代价是文件多（60 万个小 PNG）。
- 状态 10 维 = pos(3)+vel(3)+quat(4)，不存角速率：动作里已含力矩，角速率是内部量。
- 时间戳用采集时的绝对时间，不按 frame_index/fps 重算：采集是 9.6 Hz，整数 fps 会漂。
- 三个划分各建一个数据集目录（train/val/test），不做 `splits` 字段：划分按 episode，
  测试集的场景种子与指令表述都是留出的。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "image": {"dtype": "image", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (10,), "names": None},
    "actions": {"dtype": "float32", "shape": (4,), "names": None},
}


def state_at(z: dict, k: int) -> np.ndarray:
    return np.concatenate([z["pos"][k], z["vel"][k], z["quat"][k]]).astype(np.float32)


def load_episodes(src: Path, split: str) -> list[Path]:
    dirs = sorted(src.glob("shard_*/episode_*")) or sorted(src.glob("episode_*"))
    picked = []
    for d in dirs:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("split") == split:
            picked.append(d)
    return picked


def convert(src: Path, dst: Path, split: str, img_size: int, fps: int,
            limit: int | None = None) -> dict:
    eps = load_episodes(src, split)
    if limit:
        eps = eps[:limit]
    if not eps:
        raise SystemExit(f"{split} 划分下没有 episode：{src}")

    root = dst / split
    if root.exists():
        shutil.rmtree(root)
    feats = json.loads(json.dumps(FEATURES))
    feats["image"]["shape"] = (img_size, img_size, 3)
    ds = LeRobotDataset.create(repo_id=f"uav-inspect-{split}", fps=fps,
                               root=root, features=feats, use_videos=False)

    frames = 0
    for d in eps:
        z = np.load(d / "obs.npz")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        task = meta["instruction"]
        t = np.asarray(z["t"], dtype=np.float64)
        ft = np.asarray(z["frame_t"], dtype=np.float64)
        if len(ft) == 0:
            continue
        # frame_t 是 t 的子集：找每一帧落在哪个控制步，图像与动作、状态按此对齐
        idx = np.clip(np.searchsorted(t, ft), 0, len(t) - 1)
        t0 = float(t[idx[0]])          # 每个 episode 从 0 开始，LeRobot 的 fps/时间戳校验才过
        for j, k in enumerate(idx):
            ds.add_frame({
                "image": np.asarray(z["images"][j], dtype=np.uint8),
                "state": state_at(z, int(k)),
                "actions": np.asarray(z["action"][int(k)], dtype=np.float32),
                "task": task,
                "timestamp": np.asarray(t[int(k)] - t0, dtype=np.float32),
            })
        ds.save_episode()
        frames += len(idx)

    meta = {
        "split": split,
        "episodes": len(eps),
        "frames": frames,
        "root": str(root),
        "episodes_list": [d.name for d in eps],
    }
    (dst / f"convert_{split}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(convert(Path(a.src), Path(a.dst), a.split, a.img_size, a.fps, a.limit),
                     ensure_ascii=False))
