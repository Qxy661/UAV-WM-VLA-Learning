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
    rack: int = 0                                      # 目标所在货架，1..4
    target_color: str = "red"
    distractor_pos: np.ndarray | None = None           # 干扰目标（异色），仅作视觉线索
    layout: str = "a"                                  # 场景布局名，见 LAYOUTS
    rack_x: list = field(default_factory=list)         # 当前布局的货架 x 坐标


ROOM = dict(x=9.0, y=5.0, h=2.6, wall=0.12)
RACK_X = [-3.0, -1.2, 0.6, 2.4]
RACK_Y = 1.35
RACK = np.array([0.45, 0.18, 0.75])                    # 半长
TARGET = np.array([0.13, 0.07, 0.13])                  # 半长，贴在货架朝通道的一面

# 两套布局。a 是训练用的仓储；b 换成三排更宽的货架、更深更矮的房间，
# 货架数量（4→3）、间距、房间尺寸、可达边界全变，用来量"换成没见过的仓库还能不能干"。
# 布局只从参数或环境变量 UAV_LAYOUT 进来，默认 a，保证既有数据/评测的几何逐位不变。
LAYOUTS = {
    "a": dict(room=ROOM, rack_x=RACK_X, rack_y=RACK_Y),
    "b": dict(room=dict(x=7.6, y=6.2, h=2.9, wall=0.12),
              rack_x=[-2.5, -0.6, 1.5], rack_y=1.75),
}


def resolve_layout(name: str | None = None) -> tuple[str, dict]:
    """按名字或环境变量取布局；未知名字直接报错，不静默回退到 a。"""
    if name is None:
        import os

        name = os.environ.get("UAV_LAYOUT", "a")
    if name not in LAYOUTS:
        raise ValueError(f"未知布局 {name!r}，可选 {sorted(LAYOUTS)}")
    return name, LAYOUTS[name]


def _box(engine, half, pos, rgba, emissive=0.0):
    engine.add_box(tuple(half), tuple(pos), rgba=rgba, emissive=emissive)
    return engine.boxes[-1]


def build_warehouse(engine, seed: int = 0, *, with_target: bool = True,
                    with_distractor: bool = True, layout: str | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    name, lay = resolve_layout(layout)
    room, rack_x, rack_y = lay["room"], lay["rack_x"], lay["rack_y"]
    scene = Scene()
    scene.layout = name
    scene.rack_x = list(rack_x)
    W, H, T = room["wall"], room["h"], room["wall"]

    # 四面墙（只做通道侧的两面 + 两端，留出"远处还有空间"的暗示）
    _box(engine, (room["x"] / 2, T / 2, H / 2), (0.0, -room["y"] / 2, H / 2), (0.62, 0.63, 0.66, 1))
    _box(engine, (room["x"] / 2, T / 2, H / 2), (0.0, room["y"] / 2, H / 2), (0.62, 0.63, 0.66, 1))
    _box(engine, (T / 2, room["y"] / 2, H / 2), (-room["x"] / 2, 0.0, H / 2), (0.62, 0.63, 0.66, 1))
    _box(engine, (T / 2, room["y"] / 2, H / 2), (room["x"] / 2, 0.0, H / 2), (0.62, 0.63, 0.66, 1))

    # 两排货架：朝通道的一排（y<0）放目标，另一排只做遮挡
    for y in (-rack_y, rack_y):
        for x in rack_x:
            _box(engine, RACK, (x, y, RACK[2]), (0.55, 0.42, 0.30, 1))

    if not with_target:
        scene.boxes = engine.boxes
        return scene

    # 槽位按 seed 取模而不是随机抽：连续 seed 才能保证覆盖到不同格口。
    # 用 rng.integers 抽的话，seed 0/2/3 会抽到同一个槽，四集里三集是同一份数据。
    slot = int(seed % len(rack_x))
    x = rack_x[slot]
    # 站位距离小幅抖动：让同一格口的多次巡检不是逐像素相同的图
    stand = 1.0 + 0.22 * float(rng.random())
    # 贴在朝通道（-y）的面上
    # 高度也抖一档：三档让"固定高度悬停"这条捷径失效
    # 目标在货架面内再抖一次：横向 ±0.40 m、竖向 ±0.20 m。这一抖是任务设计的关键——
    # 抖过之后，"几号货架 + 哪一侧"不再决定目标在画面里的位置，只有看过图像的
    # 策略才能把镜头对准它。不抖的话盲策略也能靠位置把镜头摆对，视觉就是冗余输入。
    # 目标在货架面内连续抖动：三档离散值时约 1/3 的 seed 恰好 dx=0，
    # 盲策略（朝向固定为正对货架）会白捡三分之一，区分度被稀释；
    # 用 seed 的散列做连续取值，盲策略只有在 dx≈0 的测度零集合上才蒙得中。
    h = (seed * 2654435761) % 1000003
    dx = 0.40 * (2.0 * (h % 1000) / 999.0 - 1.0)
    dz = 0.20 * (2.0 * ((h // 1000) % 1000) / 999.0 - 1.0)
    z = 1.05 + 0.10 * (((seed // len(rack_x)) % 3) - 1) + dz
    target_pos = np.array([x + dx, rack_y - RACK[1] - TARGET[1], z])
    tgt = _box(engine, TARGET, target_pos, (0.95, 0.16, 0.12, 1), emissive=0.55)
    scene.target_pos = target_pos
    scene.target_idx = len(engine.boxes) - 1
    scene.rack = slot + 1

    # 干扰目标：另一排货架上放一个同尺寸的蓝色标记。指令里必须说"红色"，
    # 否则"去几号货架拍照"这句话在有多个标记时没有唯一解。
    if with_distractor:
        other = (slot + 2) % len(rack_x)
        dp = np.array([rack_x[other], rack_y - RACK[1] - TARGET[1], 1.05])
        _box(engine, TARGET, dp, (0.16, 0.30, 0.85, 1), emissive=0.35)
        scene.distractor_pos = dp

    scene.boxes = engine.boxes

    # 三个视点：正前、斜前、侧后。
    # 视点的**位置**锚在货架中心（anchor），不锚在目标上——否则"目标抖动"连站位一起
    # 抖，策略连该飞到哪里都不知道，任务从"对准镜头"变成"盲搜"，而且专家示范也
    # 无法被复现。视点的 **朝向** 仍指向目标真实位置：专家（oracle）知道目标在哪，
    # 只有看过图像的策略才知道该往哪转。
    anchor = np.array([rack_x[slot], rack_y - RACK[1] - TARGET[1], 1.05])
    cand = [
        (anchor + np.array([0.0, -1.15, 0.10]) * stand, "front"),
        (anchor + np.array([-0.95, -0.85, 0.35]) * stand, "oblique"),
        (anchor + np.array([0.40, 0.75, 0.35]), "behind-rack"),
    ]
    # 视点必须落在可达空间里：贴着墙的"视点"机子飞不到，任务本身不可行。
    # 留 0.55 m 余量（机体半径 0.08 + 控制超调），否则机体会被墙推着擦出碰撞。
    for pos, label in cand:
        pos = np.asarray(pos, dtype=float).copy()
        lim_x = room["x"] / 2 - room["wall"] - 0.55
        lim_y = room["y"] / 2 - room["wall"] - 0.55
        pos[0] = float(np.clip(pos[0], -lim_x, lim_x))
        pos[1] = float(np.clip(pos[1], -lim_y, lim_y))
        pos[2] = float(np.clip(pos[2], 0.60, room["h"] - 0.40))
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
