"""仓储场景：墙、两排货架、一条通道，一个随机位置的"故障标记"。

选仓储而不是空旷场地，是因为要测的东西必须有遮挡：如果目标永远可见，
覆盖率、遮挡补全、世界模型的"想象被挡住的部分"都无从谈起。

目标位置在通道左侧货架的格口里随机，三个巡检视点分别在前、侧、后，
至少有一个视点会被货架或墙挡住——这是故意的，M3 的遮挡预测需要这种样本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import target_visible


@dataclass
class Scene:
    boxes: list = field(default_factory=list)          # 与 engine.boxes 同对象
    target_pos: np.ndarray | None = None
    target_idx: int = -1
    viewpoints: list = field(default_factory=list)     # [(pos(3,), yaw, label)]
    standoff_m: float = 1.3


ROOM = dict(x=9.0, y=5.0, h=2.6, wall=0.12)
RACK_X = [-3.0, -1.2, 0.6, 2.4]
RACK_Y = 1.35
RACK = np.array([0.45, 0.18, 0.75])                    # 半长
TARGET = np.array([0.13, 0.07, 0.13])                  # 半长，贴在货架朝通道的一面


def _box(engine, half, pos, rgba, emissive=0.0):
    engine.add_box(tuple(half), tuple(pos), rgba=rgba, emissive=emissive)
    return engine.boxes[-1]


def build_warehouse(engine, seed: int = 0, *, with_target: bool = True) -> Scene:
    rng = np.random.default_rng(seed)
    scene = Scene()
    W, H, T = ROOM["wall"], ROOM["h"], ROOM["wall"]

    # 四面墙（只做通道侧的两面 + 两端，留出"远处还有空间"的暗示）
    _box(engine, (ROOM["x"] / 2, T / 2, H / 2), (0.0, -ROOM["y"] / 2, H / 2), (0.62, 0.63, 0.66, 1))
    _box(engine, (ROOM["x"] / 2, T / 2, H / 2), (0.0, ROOM["y"] / 2, H / 2), (0.62, 0.63, 0.66, 1))
    _box(engine, (T / 2, ROOM["y"] / 2, H / 2), (-ROOM["x"] / 2, 0.0, H / 2), (0.62, 0.63, 0.66, 1))
    _box(engine, (T / 2, ROOM["y"] / 2, H / 2), (ROOM["x"] / 2, 0.0, H / 2), (0.62, 0.63, 0.66, 1))

    # 两排货架：朝通道的一排（y<0）放目标，另一排只做遮挡
    for y in (-RACK_Y, RACK_Y):
        for x in RACK_X:
            _box(engine, RACK, (x, y, RACK[2]), (0.55, 0.42, 0.30, 1))

    if not with_target:
        scene.boxes = engine.boxes
        return scene

    # 槽位按 seed 取模而不是随机抽：连续 seed 才能保证覆盖到不同格口。
    # 用 rng.integers 抽的话，seed 0/2/3 会抽到同一个槽，四集里三集是同一份数据。
    slot = int(seed % len(RACK_X))
    x = RACK_X[slot]
    # 站位距离小幅抖动：让同一格口的多次巡检不是逐像素相同的图
    stand = 1.0 + 0.22 * float(rng.random())
    # 贴在朝通道（-y）的面上
    target_pos = np.array([x, RACK_Y - RACK[1] - TARGET[1], 1.05])
    tgt = _box(engine, TARGET, target_pos, (0.95, 0.16, 0.12, 1), emissive=0.55)
    scene.target_pos = target_pos
    scene.target_idx = len(engine.boxes) - 1
    scene.boxes = engine.boxes

    # 三个视点：正前、斜前、侧后。yaw 指向目标
    cand = [
        (target_pos + np.array([0.0, -1.15, 0.10]) * stand, "front"),
        (target_pos + np.array([-0.95, -0.85, 0.35]) * stand, "oblique"),
        (target_pos + np.array([0.40, 0.75, 0.35]), "behind-rack"),
    ]
    for pos, label in cand:
        pos = np.asarray(pos, dtype=float)
        d = target_pos - pos
        yaw = float(np.arctan2(d[1], d[0]))
        scene.viewpoints.append((pos, yaw, label))
    return scene


def visible_viewpoints(scene: Scene) -> list[bool]:
    """逐视点判定目标是否真的看得见（射线不被货架挡住）。"""
    out = []
    for pos, _yaw, _label in scene.viewpoints:
        out.append(target_visible(pos, scene.target_pos, boxes=scene.boxes))
    return out
