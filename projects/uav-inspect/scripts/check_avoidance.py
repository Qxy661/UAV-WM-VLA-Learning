"""互惠斥力避让的单元级检查（不依赖世界）。

用法：py -3.9 projects/uav-inspect/scripts/check_avoidance.py
两条断言：
  1. 正面对穿：两机对飞，斥力必须让它们错开（不顶住）。
  2. 远离时斥力为 0，不能凭空产生速度。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from uavinspect.planning.avoidance import repulsive_velocity  # noqa: E402


def main() -> int:
    # ① 正面对穿：i 在原点 +x 飞，j 在 +2 m 处 -x 飞
    pos = np.array([[0.0, 0.0, 1.0], [1.2, 0.0, 1.0]])
    vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    u = repulsive_velocity(pos, vel, d_safe=0.6, d_influence=1.5, k_rep=1.0, v_max=2.0)
    # 切向分量必须存在，否则两机只会沿 +x/-x 互相顶
    lat = max(abs(u[0, 1]), abs(u[1, 1]))
    assert lat > 0.05, f"切向让让量太小，正面对穿会僵持：{u}"
    assert u[0, 0] < 0 and u[1, 0] > 0, f"径向方向应为相互推开：{u}"

    # ② 远离：超出影响半径，斥力为 0
    far = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    z = repulsive_velocity(far, np.zeros((2, 3)), d_safe=0.6, d_influence=1.5,
                           k_rep=1.0, v_max=2.0)
    assert np.allclose(z, 0.0), f"超出影响半径仍有斥力：{z}"

    # ③ 上限：两机几乎重合时斥力不能超过 v_max
    near = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]])
    u2 = repulsive_velocity(near, np.zeros((2, 3)), d_safe=0.6, d_influence=1.5,
                            k_rep=1.0, v_max=1.0)
    assert np.max(np.linalg.norm(u2, axis=1)) <= 1.0 + 1e-9, u2

    print("避让单元检查通过：正面对穿有切向让让、远处无斥力、速度有上限")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
