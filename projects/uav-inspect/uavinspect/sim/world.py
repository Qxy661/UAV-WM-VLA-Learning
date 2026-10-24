"""仿真世界：物理步进、多机、障碍。

两种后端共用同一个对外接口（add_drone / add_box / advance / state_vector / t / steps），
调用方不需要知道背后是 PyBullet 还是自研积分：

    world = World(cfg)                  # 有 pybullet 就用它，否则用 rigid
    world = World(cfg, backend="rigid") # 明确指定

`advance(dt)` 只按物理步长推进，并且在**每个物理子步**里更新电机与外力。
把"控制一次、物理多步"做对，才谈得上频率分离（ADR-003）；否则所谓 240 Hz 物理
只是把同一个力重复施加 5 次，等于隐式改写了推力曲线。
"""

from __future__ import annotations

from ..config import Config


def pybullet_available() -> bool:
    try:
        import pybullet  # noqa: F401
    except Exception:
        return False
    return True


def World(cfg: Config, backend: str = "auto"):
    """按后端返回一个 world 对象；接口见模块 docstring。"""
    if backend == "auto":
        backend = "pybullet" if pybullet_available() else "rigid"
    if backend == "rigid":
        from .rigid import RigidEngine

        return RigidEngine(cfg)
    if backend == "pybullet":
        if not pybullet_available():
            raise RuntimeError(
                "指定了 pybullet 后端，但当前解释器导不进 pybullet。"
                "换 backend='rigid'，或先在本环境装好 pybullet。")
        from .world_pybullet import PyBulletWorld

        return PyBulletWorld(cfg)
    raise ValueError(f"未知后端：{backend!r}（可选 auto / rigid / pybullet）")
