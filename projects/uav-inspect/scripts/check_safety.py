"""安全层契约检查（不依赖仿真，纯接口级）。

用法：py -3.9 projects/uav-inspect/scripts/check_safety.py

七条断言，都是"实现改坏了会立刻红"的那种：
  1. 3 个周期才升级（防抖），第 2 个周期不动
  2. 连续 30 个正常周期才降级
  3. 陈旧指令不被执行（HOLD 用的是上一条好指令）
  4. 不合理指令被拒绝并记录（0.5 倍悬停推力在限幅内，但必须拦）
  5. 持续断流按 HOLD→HOVER 阶梯上升，不跳级
  6. 出围栏直接要求 RTL
  7. 推力/力矩的硬限幅生效
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from uavinspect.safety.guard import Command, GuardConfig, Mode, SafetyGuard  # noqa: E402

FENCE = (-3.5, 3.5, -2.5, 2.5)
HOME = np.array([0.0, 0.0, 1.5])


def guard(cfg: GuardConfig | None = None) -> SafetyGuard:
    return SafetyGuard(cfg or GuardConfig(), FENCE)


def feed(g, t, cmds, pos=None, vel=None):
    pos = np.array([0.0, 0.0, 1.5]) if pos is None else pos
    vel = np.zeros(3) if vel is None else vel
    return g.filter(t, cmds, pos, vel, HOME)


def good(t, thrust=9.8, torque=(0.0, 0.0, 0.0), priority=0, source="mission") -> Command:
    return Command(thrust, np.array(torque, float), t, priority, source)


def main() -> int:
    # ① 防抖：3 周期才升级
    g = guard()
    feed(g, 0.00, [good(0.00)])
    assert g.st.mode == Mode.NORMAL
    feed(g, 0.02, [])                 # 第 1 个缺失周期
    feed(g, 0.04, [])                 # 第 2 个
    assert g.st.mode == Mode.NORMAL, "第 2 个周期就升级了，说明没有防抖"
    feed(g, 0.06, [])                 # 第 3 个
    assert g.st.mode == Mode.HOLD, f"3 周期后应进 HOLD，实际 {g.st.mode}"

    # ② 降级要 30 个正常周期
    for i in range(29):
        feed(g, 0.08 + i * 0.02, [good(0.08 + i * 0.02)])
    assert g.st.mode == Mode.HOLD, "29 个正常周期就降级了，退出太早"
    feed(g, 0.08 + 29 * 0.02, [good(0.08 + 29 * 0.02)])
    assert g.st.mode == Mode.NORMAL, f"30 周期后应回 NORMAL，实际 {g.st.mode}"

    # ③ 陈旧指令不被执行：HOLD 时用上一条好指令（9.8 N），而不是 2 N 的坏值
    g = guard()
    feed(g, 0.0, [good(0.0, thrust=9.8)])
    for i in range(1, 4):                      # 时间要越过 0.2 s 心跳窗口
        feed(g, 0.2 + i * 0.02, [good(0.0, thrust=2.0)])
    thrust, _, mode = feed(g, 0.29, [good(0.0, thrust=2.0)])
    assert mode == Mode.HOLD, f"3 个陈旧周期后应进 HOLD，实际 {mode}"
    assert thrust == 9.8, f"HOLD 期间应用上一条好指令 9.8 N，实际 {thrust:.2f}"

    # ④ 不合理指令（限幅内但物理上错）必须被拒绝
    g = guard()
    feed(g, 0.0, [good(0.0)])
    thrust, _, mode = feed(g, 0.5, [good(0.5, thrust=5.0)])   # 0.51 倍悬停推力
    assert any("拒绝不合理" in e["what"] for e in g.st.events), "0.5 倍悬停推力没有被拦"
    assert thrust > 9.0, f"拒绝后应回到悬停推力附近，实际 {thrust:.2f}"

    # ⑤ 持续断流：HOLD → HOVER，不跳级
    cfg = GuardConfig(hold_timeout_s=0.10, hover_timeout_s=0.10)
    g = guard(cfg)
    feed(g, 0.0, [good(0.0)])
    seen = []
    for i in range(1, 60):
        t = i * 0.02
        _, _, mode = feed(g, t, [])
        if not seen or seen[-1] != mode:
            seen.append(mode)
    assert seen[:3] == [Mode.NORMAL, Mode.HOLD, Mode.HOVER], f"阶梯不对：{seen}"
    assert seen == sorted(seen), f"阶梯出现回退：{seen}"

    # ⑥ 出围栏 → RTL
    g = guard()
    pos = np.array([4.2, 0.0, 1.5])            # x 超出 3.5 - 0.3
    for i in range(3):
        _ = feed(g, i * 0.02, [good(i * 0.02)], pos=pos)
    assert g.st.mode == Mode.RTL, f"出围栏应 RTL，实际 {g.st.mode}"

    # ⑦ 硬限幅
    cfg = GuardConfig()
    g = guard(cfg)
    thrust, torque, _ = feed(g, 0.0, [good(0.0, thrust=999.0, torque=(9.0, -9.0, 9.0))])
    assert thrust <= cfg.thrust_abs_max + 1e-9, thrust
    assert float(np.abs(torque).max()) <= cfg.torque_abs_max + 1e-9, torque

    print("安全层契约检查通过：防抖 / 迟滞 / 陈旧 / 不合理指令 / 阶梯 / 围栏 / 限幅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
