"""读 uav-inspect 的 npz 分片，切成 (图像, 状态, 指令) → 动作块 的样本。

为什么不走 LeRobot：BC 基线只要张量，而 LeRobot 的 PNG 解码是给 π0.5 那条链路用的。
两边读同一批 episode，样本集合一致，比较才成立；这一点写在 PLAN.md 的「同数据」约束里。

image 用 uint8 存，float 化与归一化在 `__getitem__` 里做，省一半内存。
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bc.model import ACTION_DIM, HORIZON, STATE_DIM, text_features

MEAN = np.array([0.0] * ACTION_DIM, dtype=np.float32)   # 由 fit_stats 覆盖
STD = np.array([1.0] * ACTION_DIM, dtype=np.float32)


class EpisodeCache:
    """小 LRU：一个 episode 解压后约 2 MB，缓存 256 个够把 IO 摊平。"""

    def __init__(self, cap: int = 256):
        self.cap = cap
        self.d = OrderedDict()

    def get(self, path: Path) -> dict:
        if path in self.d:
            self.d.move_to_end(path)
            return self.d[path]
        z = dict(np.load(path))
        self.d[path] = z
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return z


def list_episodes(root: Path, splits: tuple[str, ...],
                  require_coverage: float = 0.0) -> list[Path]:
    out = []
    for meta_path in sorted(root.glob("shard_*/episode_*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("split") not in splits:
            continue
        # 扰动数据里专家自己没走完的轨迹是截断的，学它等于学"停在半路"。
        if require_coverage > 0.0 and float(meta.get("coverage") or 0.0) < require_coverage:
            continue
        out.append(meta_path.parent)
    return out


# 注意量纲：录下来的 action[0] 是**牛顿**（专家控制器直接给 f_des·b3），
# 不是 0.5 那种归一化推力。专家高度环是 u0 = 0.5 + 0.10(h−z) − 0.30·vz，
# 而 thrust_N = 2·m·g·u0（quad_sim 里 act[:,0]*g*2 − g），m=1 kg，于是：
HOVER_N = 9.81            # 2·m·g·0.5
KP_Z, KD_Z = 1.962, 5.886  # 2·m·g·0.10, 2·m·g·0.30
# 残差动作的归一化尺度：推力 2 N、三轴力矩 0.5 N·m（约等于数据里动作标准差的量级，
# 但不受悬停帧多寡影响）。改这里等于改"1 个输出单位代表多少物理量"。
RESIDUAL_SCALE = (2.0, 0.5, 0.5, 0.5)


def base_action(z: np.ndarray, vz: np.ndarray, hover: float = 1.5) -> np.ndarray:
    """不依赖航点的"保高"前馈：只用地形常识与当前状态。

    这是 residual 动作空间的基底：策略只学"相对保高要多给多少"，避免把
    9.81 N 这种几乎常数的量也交给网络回归——偏 0.3 N 就会一路沉到地面。
    """
    u0 = HOVER_N + KP_Z * (hover - np.asarray(z)) - KD_Z * np.asarray(vz)
    out = np.zeros((np.atleast_1d(u0).shape[0], 4), dtype=np.float32)
    out[:, 0] = np.atleast_1d(u0)
    return out


def fit_stats(root: Path, limit: int = 400, require_coverage: float = 0.0,
              max_episodes: int = 0, residual: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """动作归一化参数只在 train 划分上估，避免测试集信息漏进训练。

    residual 模式不走数据方差，而是用**物理尺度**做归一化（推力 2 N、力矩 0.5 N·m）。
    理由是残差的方差本身就小、且随"专家当时在干什么"变化；按方差会被悬停段压到
    接近 0，训练目标被放大几十倍（实测 val_mse 25 量级、闭环 0/20）。按基底尺度缩放
    等价于"1 个单位的输出 = 用满 2 N 的推力余量"，与数据里出现多少悬停帧无关。
    """
    if residual:
        return np.zeros(4, dtype=np.float32), np.asarray(RESIDUAL_SCALE, np.float32)
    eps = list_episodes(root, ("train",), require_coverage)[:limit]
    if max_episodes:
        eps = eps[:max_episodes]
    chunks = []
    for p in eps:
        z = np.load(p / "obs.npz")
        a = np.asarray(z["action"], np.float32)
        if residual:
            a = a - base_action(np.asarray(z["pos"])[:, 2], np.asarray(z["vel"])[:, 2])
        chunks.append(a)
    acts = np.concatenate(chunks, axis=0)
    mean = acts.mean(axis=0).astype(np.float32)
    std = acts.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    # 试过给残差通道的标准差设下限 0.05：val_mse 仍 25 量级、闭环 0/20，
    # 而且高度保持退回到 0.13 m。结论是这种"把尺度拉回来"的做法不解决问题，
    # 残差动作空间的归一化要换一套（按基底尺度缩放，而不是按残差自身方差）。
    return mean, std


class ChunkDataset(Dataset):
    def __init__(self, root: Path, splits: tuple[str, ...], mean=None, std=None,
                 horizon: int = HORIZON, every: int = 1, require_coverage: float = 0.0,
                 max_episodes: int = 0, residual: bool = False):
        self.root = Path(root)
        self.eps = list_episodes(self.root, splits, require_coverage)
        if max_episodes:
            # 数据规模曲线用：只取前 N 条。list_episodes 按路径排序、seed 单调，所以取前 N 是确定性的。
            self.eps = self.eps[:max_episodes]
        if not self.eps:
            raise SystemExit(f"{splits} 下没有 episode：{root}")
        self.cache = EpisodeCache()
        self.horizon = horizon
        self.residual = residual
        self.mean = np.zeros(ACTION_DIM, np.float32) if mean is None else np.asarray(mean, np.float32)
        self.std = np.ones(ACTION_DIM, np.float32) if std is None else np.asarray(std, np.float32)
        self.text_cache: dict[str, np.ndarray] = {}

        # 索引：每个 episode 的每一帧都是一个样本起点
        self.index: list[tuple[int, int]] = []
        for ei, ep in enumerate(self.eps):
            meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
            n = int(meta.get("frames") or 0)
            for j in range(0, n, every):
                self.index.append((ei, j))

    def __len__(self) -> int:
        return len(self.index)

    def _text(self, s: str) -> np.ndarray:
        if s not in self.text_cache:
            self.text_cache[s] = text_features(s)
        return self.text_cache[s]

    def __getitem__(self, i: int):
        ei, j = self.index[i]
        ep = self.eps[ei]
        z = self.cache.get(ep / "obs.npz")
        meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        img = z["images"][j].astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))
        t = np.asarray(z["t"], np.float64)
        ft = np.asarray(z["frame_t"], np.float64)
        k = int(np.clip(np.searchsorted(t, ft[j]), 0, len(t) - 1))
        state = np.concatenate([z["pos"][k], z["vel"][k], z["quat"][k]]).astype(np.float32)
        acts = np.asarray(z["action"], np.float32)[k:k + self.horizon]
        if len(acts) < self.horizon:                      # 末尾补齐：重复最后一步
            pad = np.repeat(acts[-1:], self.horizon - len(acts), axis=0)
            acts = np.concatenate([acts, pad], axis=0)
        if self.residual:
            acts = acts - base_action(z["pos"][k:k + self.horizon, 2],
                                      z["vel"][k:k + self.horizon, 2])
        norm = (acts - self.mean) / self.std
        return (img, torch.from_numpy(state), torch.from_numpy(self._text(meta["instruction"])),
                torch.from_numpy(norm.astype(np.float32)))
