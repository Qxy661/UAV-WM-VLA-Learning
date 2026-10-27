"""把巡检 npz 直接写成 LeRobot v2.1 数据集（不经过 LeRobotDataset.create）。

为什么手写：lerobot 0.1.0 的 `validate_feature_numpy_array` 拿 np.ndarray 的
`shape`（tuple）和 `create_empty_dataset_info` 里 JSON 化后的 `shape`（list）做
`!=` 比较，`(4,) != [4]` 恒真，所以官方 create/add_frame 这条路在本机必报错。
手写只需要产出 meta/info.json、meta/tasks.jsonl、meta/episodes.jsonl、
data/chunk-000/episode_XXXXXX.parquet 与 images/image/episode_XXXXXX/frame_XXXXXX.png，
再由 `LeRobotDataset(root=...)` 读回验证。

约定（与官方 LIBERO 数据一致的三条）：
- 一行一个**图像帧**；动作/状态取该帧时间戳最近的控制步。
- 状态 10 维 = 位置 3 + 速度 3 + 四元数 4。
- 图像存逐帧 PNG，parquet 里存相对路径（`use_videos=False`，不碰 torchcodec）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "projects" / "uav-inspect"))

IMAGE_KEY = "image"
STATE_DIM = 10
ACTION_DIM = 4


def features_of(img_hw: int, action_dim: int = ACTION_DIM) -> dict:
    # shape 用 list，JSON 化后保持不变，读回时与写入一致
    return {
        IMAGE_KEY: {"dtype": "image", "shape": [3, img_hw, img_hw],
                    "names": ["channels", "height", "width"]},
        "state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
        "actions": {"dtype": "float32", "shape": [action_dim], "names": None},
    }


def iter_episodes(root: Path):
    for meta_path in sorted(root.glob("shard_*/episode_*/meta.json")):
        ep_dir = meta_path.parent
        obs_path = ep_dir / "obs.npz"
        if not obs_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        yield ep_dir, obs_path, meta


def state_at(z, k: int) -> np.ndarray:
    return np.concatenate([z["pos"][k], z["vel"][k], z["quat"][k]]).astype(np.float32)


def nearest_step(t: np.ndarray, frame_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t, frame_t)
    idx = np.clip(idx, 1, len(t) - 1)
    left, right = t[idx - 1], t[idx]
    return np.where(np.abs(frame_t - left) <= np.abs(right - frame_t), idx - 1, idx)


def convert(src: Path, dst: Path, split: str, limit: int | None, img_size: int, flat: bool = False,
            require_coverage: float = 0.0, repo_id: str | None = None,
            action_mode: str = "torque") -> dict:
    import pandas as pd
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION  # noqa: F401
    from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.common.datasets.utils import create_empty_dataset_info, serialize_dict, write_json, write_stats

    if not flat:
        dst = dst / split
    if dst.exists():
        raise SystemExit(f"{dst} 已存在，先删掉再转（避免 info.json 与 parquet 对不上）")
    (dst / "meta").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "images" / IMAGE_KEY / "chunk-000").mkdir(parents=True)

    items = [it for it in iter_episodes(src) if it[2].get("split", "train") == split]
    if require_coverage > 0.0:
        # 扰动数据里有一部分 episode 专家自己也没走完（覆盖率 0），这些轨迹是截断的，
        # 收进数据集只会教模型"停在半路"，所以按覆盖率过滤。
        before = len(items)
        items = [it for it in items if float(it[2].get("coverage") or 0.0) >= require_coverage]
        print(f"[{split}] 覆盖率过滤 {before} -> {len(items)}")
    if limit is not None:
        items = items[:limit]
    if not items:
        raise SystemExit(f"{split} 划分没有数据")

    # fps 必须与真实帧间隔一致，否则 LeRobot 的 timestamp 连续性检查会报错
    # （相机每 5 个控制步采一帧，48 Hz 控制 -> 9.6 Hz，不是整齐的 10 Hz）。
    z0 = np.load(items[0][1])
    ft0 = z0["frame_t"].astype(np.float64)
    fps = round(1.0 / float(np.median(np.diff(ft0))), 3)

    a_dim = 4 if action_mode == "torque" else 3
    feats = features_of(img_size, a_dim)
    info = create_empty_dataset_info(CODEBASE_VERSION, fps, "uav", feats, use_videos=False)
    info["splits"] = {split: f"0:{len(items)}"}
    write_json(info, dst / "meta" / "info.json")

    tasks, task_index_of = [], {}
    episodes, episodes_stats, raw_stats, total_frames, global_index = [], [], [], 0, 0
    for ep_idx, (ep_dir, obs_path, meta) in enumerate(items):
        z = np.load(obs_path)
        t = z["t"].astype(np.float64)
        frame_t = z["frame_t"].astype(np.float64)
        steps = nearest_step(t, frame_t)
        images = z["images"]

        task = str(meta.get("instruction", "inspect the target"))
        if task not in task_index_of:
            task_index_of[task] = len(tasks)
            tasks.append(task)
        task_index = task_index_of[task]

        ep_images = dst / "images" / IMAGE_KEY / "chunk-000" / f"episode_{ep_idx:06d}"
        ep_images.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        tgt = np.asarray(meta.get("target_pos", [np.nan] * 3), dtype=np.float64)
        rows = {"image": [], "state": [], "actions": [], "timestamp": [],
                "frame_index": [], "episode_index": [], "index": [], "task_index": []}
        for fi, (img, tr, k) in enumerate(zip(images, frame_t, steps)):
            rel = f"images/{IMAGE_KEY}/chunk-000/episode_{ep_idx:06d}/frame_{fi:06d}.png"
            Image.fromarray(img).save(dst / rel)
            rows["image"].append(rel)
            st = state_at(z, int(k))
            rows["state"].append(st)
            if action_mode == "subgoal":
                # 相对位移子目标：网络只回答"目标相对我多少米"，保高与对准交给下层控制器。
                # 不裁剪：房间对角约 8 m，裁到 ±3 m 会把远端的 x 分量压成常数。
                rows["actions"].append((tgt - st[:3]).astype(np.float32))
            else:
                rows["actions"].append(z["action"][int(k)].astype(np.float32))
            rows["timestamp"].append(fi / fps)
            rows["frame_index"].append(fi)
            rows["episode_index"].append(ep_idx)
            rows["index"].append(global_index)
            rows["task_index"].append(task_index)
            global_index += 1
        df = pd.DataFrame(rows)
        df.to_parquet(dst / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet", index=False)

        # v2.1 要求每条 episode 自带 stats（meta/episodes_stats.jsonl），
        # 没有它 LeRobotDatasetMetadata.load_metadata() 会抛异常并转去 hub 拉取。
        ep_stats = compute_episode_stats(
            {IMAGE_KEY: [str(dst / rel) for rel in rows["image"]],
             "state": np.stack(rows["state"]), "actions": np.stack(rows["actions"])}, feats)
        raw_stats.append(ep_stats)
        episodes_stats.append({"episode_index": ep_idx, "stats": serialize_dict(ep_stats)})
        episodes.append({"episode_index": ep_idx, "tasks": [task], "length": len(rows["image"])})
        total_frames += len(rows["image"])

    # chunks_size 取「不小于 episode 总数」，这样所有 episode 都落在 chunk-000 里，
    # 与下面实际写入的 data/chunk-000/ 路径一致。lerobot 的读取端会按
    # data/chunk-{episode_index // chunks_size:03d}/ 拼路径；若 chunks_size 仍是默认 1000
    # 而文件全写在 chunk-000，第 1000 集之后会判为缺失并尝试联网下载（离线机器上直接报错）。
    info.update({"total_episodes": len(episodes), "total_frames": total_frames,
                 "total_tasks": len(tasks), "total_chunks": 1,
                 "chunks_size": max(len(episodes), 1000)})
    if repo_id:
        info["repo_id"] = repo_id
    write_json(info, dst / "meta" / "info.json")
    write_stats(aggregate_stats(raw_stats), dst)

    for name, rows_ in (("tasks", [{"task_index": i, "task": t} for i, t in enumerate(tasks)]),
                        ("episodes", episodes),
                        ("episodes_stats", episodes_stats)):
        with open(dst / "meta" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = {"split": split, "episodes": len(episodes), "frames": total_frames,
           "tasks": len(tasks), "root": str(dst)}
    (dst / f"convert_{split}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/work/data/uav96")
    ap.add_argument("--dst", default="/root/work/data/lerobot")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--flat", action="store_true", help="直接写到 --dst（训练时 repo_id 需要一个完整数据集根）")
    ap.add_argument("--require-coverage", type=float, default=0.0)
    ap.add_argument("--repo-id", default=None, help="写进 info.json 的 repo_id，默认用 --dst 的相对路径")
    ap.add_argument("--action-mode", choices=("torque", "subgoal"), default="torque",
                    help="torque: 4 维力矩；subgoal: 3 维相对位移子目标（目标世界坐标 - 当前位置）")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        out = convert(Path(args.src), Path(args.dst), split, args.limit, args.img_size, args.flat,
                      require_coverage=args.require_coverage, repo_id=args.repo_id,
                      action_mode=args.action_mode)
        print(f"[{split}] episodes={out['episodes']} frames={out['frames']} tasks={out['tasks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
