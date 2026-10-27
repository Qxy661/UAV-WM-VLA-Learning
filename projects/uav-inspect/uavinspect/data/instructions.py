"""指令生成：把「去几号货架、从哪一侧拍」变成一句自然语言。

设计要点：
- 槽位固定、表述多样。训练集见不到的**表述**要在测试集里出现，否则测的是记忆不是泛化。
- 指令里必须含可达的信息：货架编号（决定 x）、侧向（决定朝向）、颜色（同场还有异色
  干扰标记时，颜色是唯一的区分线索）。场景里没有的线索不许写，写了模型只能瞎猜。
"""

from __future__ import annotations

import random

SIDES = ("front", "oblique")
SIDE_ZH = {"front": "正面", "oblique": "斜侧"}
RACKS = (1, 2, 3, 4)

# 训练用表述。占位：{rack} {side} {side_en}
TEMPLATES_TRAIN = [
    "去 {rack} 号货架，从{side}拍下红色目标",
    "飞到 {rack} 号货架{side}，把红色标记拍清楚",
    "{rack} 号货架{side}有红色标记，过去拍一张",
    "检查 {rack} 号货架{side}的红色目标",
    "请前往 {rack} 号货架，从{side}拍下红色目标",
    "到 {rack} 号货架{side}，确认红色目标是否可见",
]

# 测试用表述：训练集里一句都不出现。模型要读懂"绕到…后侧""对着…拍"这类没见过的说法。
TEMPLATES_HELD_OUT = [
    "绕到 {rack} 号货架的{side}，给红色标记来一张",
    "红色目标藏在 {rack} 号货架，从{side}对着它拍",
    "先把机头对准 {rack} 号货架{side}，再拍红色那个",
    "{rack} 号货架那边有个红色标记，过去看一眼并拍下来",
    "沿通道去 {rack} 号货架，{side}视角拍一张红色标记",
]

FILLER = [
    "",
    "，注意别撞到货架",
    "，保持一米以上高度",
    "，慢一点",
]


def make_instruction(rack: int, side: str, rng: random.Random, *, held_out: bool = False,
                     with_filler: bool = True) -> str:
    pool = TEMPLATES_HELD_OUT if held_out else TEMPLATES_TRAIN
    text = rng.choice(pool).format(rack=rack, side=SIDE_ZH[side], side_en=side)
    if with_filler and rng.random() < 0.35:
        text += rng.choice(FILLER)
    return text


def split_of(seed: int, side: str, *, n_train: int = 6400, n_val: int = 800) -> str:
    """按 episode 序号划分，不按帧。测试集另用没见过的场景 seed 段。"""
    if seed < n_train:
        return "train"
    if seed < n_train + n_val:
        return "val"
    return "test"
