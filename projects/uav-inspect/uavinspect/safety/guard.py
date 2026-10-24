"""安全层：独立于策略的一票否决，带迟滞与降级阶梯。

三条硬规矩（对应 docs/10-横切判据/08-安全层与降级.md）：

1. **独立**：它不读策略的内部状态，只看"最终指令 + 时间戳 + 当前位姿"。
   策略挂了、卡了、发疯发大指令，安全层都还能工作。
2. **只做减法**：安全层只能把指令往安全方向改（截断、限速、替换成悬停），
   不能凭空生成新的飞行意图。
3. **迟滞**：进入保护快（3 个控制周期）、退出保护慢（30 个周期 + 三个条件），
   否则会在阈值附近来回切，机身跟着抖。

降级阶梯（只升不降，落地才清零）：
    NORMAL → HOLD（守住上一条好指令）→ HOVER（原地定高）→ RTL（回起飞点）→ LAND

为什么不是"一超时就悬停"：链路抖动是常态，1 次丢包就悬停会让任务永远做不完。
所以先 HOLD（几百毫秒内指令仍然有效），持续丢才升级。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Mode(IntEnum):
    NORMAL = 0
    HOLD = 1
    HOVER = 2
    RTL = 3
    LAND = 4


@dataclass
class GuardConfig:
    heartbeat_timeout_s: float = 0.20      # 超过这么久没有新指令 → 进 HOLD
    hold_timeout_s: float = 0.60           # HOLD 持续这么久 → HOVER
    hover_timeout_s: float = 3.0           # HOVER 持续这么久 → RTL
    rtl_timeout_s: float = 12.0            # RTL 走这么久还没到 → LAND
    enter_cycles: int = 3                  # 连续 N 个周期才升级（防抖）
    exit_cycles: int = 30                  # 连续 N 个周期正常才降级
    thrust_abs_max: float = 21.0           # 总推力上限（N），≈ 4 桨满转
    torque_abs_max: float = 0.5            # N·m，约 45 rad/s^2 的角加速度
    slew_limit: float = 3.0                # 每周期推力变化上限（N），≈ 0.3 g/周期
    fence_margin: float = 0.30             # 离围栏边界留多少米
    hover_height: float = 1.5
    mass: float = 1.0
    gravity: float = 9.81
    hover_kp: float = 2.0                  # 定高 P（N/m）
    hover_kd: float = 1.5                  # 定高 D（N·s/m）
    rtl_kp: float = 0.08                   # 回航位置误差 → 力矩
    plausible_thrust_band: tuple = (0.65, 1.35)   # 相对悬停推力的可信区间
    plausible_torque: float = 0.15          # N·m，超过这个量级就当成故障


@dataclass
class Command:
    thrust: float
    torque: np.ndarray
    stamp: float                            # 产生这条指令的时间（秒）
    priority: int = 0                       # 数字小 = 优先级高；同周期到达时裁决用
    source: str = "policy"


@dataclass
class GuardState:
    mode: Mode = Mode.NORMAL
    since: float = 0.0
    last_good: Command | None = None
    last_mode_change: float = 0.0
    events: list = field(default_factory=list)

    def log(self, t: float, what: str) -> None:
        self.events.append({"t": round(t, 3), "what": what, "mode": self.mode.name})


class SafetyGuard:
    """用法：每控制周期调用 `guard.filter(t, fresh_cmd, pos, vel, home)`。"""

    def __init__(self, cfg: GuardConfig, fence: tuple[float, float, float, float]):
        self.cfg = cfg
        self.fence = fence                  # (xmin, xmax, ymin, ymax)
        self.st = GuardState()
        self._enter_count = 0
        self._exit_count = 0

    # ---------------- 对外唯一入口 ----------------
    def filter(self, t: float, cmds, pos: np.ndarray, vel: np.ndarray,
               home: np.ndarray) -> tuple[float, np.ndarray, Mode]:
        """cmds 是本周期到达的全部指令（可能 0 条、1 条或冲突时的多条）。

        同周期多路指令按 priority 裁决（数字小者胜），落败方不会被执行——
        否则两路会以控制周期为节奏互相覆盖，表现为机身高频抖动。
        """
        cfg = self.cfg
        cmd = self._arbitrate(cmds)
        out = self._pick(t, cmd, pos, vel, home)
        thrust, torque = self._clip(out)
        thrust = self._slew(thrust)
        self._update_mode(t, cmd, pos)
        return thrust, torque, self.st.mode

    def _arbitrate(self, cmds) -> Command | None:
        if not cmds:
            return None
        cmds = list(cmds)
        best = min(cmds, key=lambda c: (c.priority, c.stamp))
        if len(cmds) > 1:
            losers = [c.source for c in cmds if c is not best]
            self.st.log(best.stamp, f"裁决：{best.source} 胜，落败 {losers}")
        return best

    # ---------------- 指令来源 ----------------
    def _pick(self, t: float, cmd: Command | None, pos: np.ndarray, vel: np.ndarray,
              home: np.ndarray) -> Command:
        cfg = self.cfg
        stale = cmd is None or (t - cmd.stamp) > cfg.heartbeat_timeout_s
        if cmd is not None and not stale and not self._plausible(cmd):
            # 指令"新鲜"但物理上不合理（推力远低于悬停、力矩超一个量级）：
            # 当成一次故障处理，而不是老实执行一条会把机子摔下去的命令。
            self.st.log(t, f"拒绝不合理指令 thrust={cmd.thrust:.2f} "
                           f"torque={float(np.abs(cmd.torque).max()):.2f}")
            stale = True
        if self.st.mode == Mode.NORMAL:
            if not stale:
                self.st.last_good = Command(cmd.thrust, np.array(cmd.torque, float), cmd.stamp)
                return cmd
            return self._hold_or_hover(t, pos, vel)
        if self.st.mode == Mode.HOLD:
            return self._hold_or_hover(t, pos, vel)
        if self.st.mode == Mode.HOVER:
            return self._hover_cmd(pos, vel)
        if self.st.mode == Mode.RTL:
            return self._goto(pos, home)
        return Command(0.0, np.zeros(3), t)          # LAND：切推力

    def _plausible(self, cmd: Command) -> bool:
        """指令的物理可信度检查：推力在悬停推力附近，力矩不超过一个量级。

        这一条比"限幅"重要：限幅只保证不超硬件极限，但一条 0.5 倍悬停推力的指令
        完全在限幅内，却会让机子直接掉下去。安全层要拦的是这种"合理范围内但明显错"的命令。
        """
        c = self.cfg
        hover = c.mass * c.gravity
        lo, hi = c.plausible_thrust_band
        if not (lo * hover <= cmd.thrust <= hi * hover):
            return False
        return float(np.abs(cmd.torque).max()) <= c.plausible_torque

    def _hold_or_hover(self, t: float, pos: np.ndarray, vel: np.ndarray) -> Command:
        hold_age = t - (self.st.last_good.stamp if self.st.last_good else 0.0)
        if self.st.last_good is not None and hold_age <= self.cfg.hold_timeout_s:
            return self.st.last_good
        return self._hover_cmd(pos, vel)

    def _hover_cmd(self, pos: np.ndarray, vel: np.ndarray) -> Command:
        """定高：推力补重力与高度误差，力矩把角速度拉平（N 与 N·m 的量纲）。"""
        c = self.cfg
        thrust = c.mass * c.gravity + c.hover_kp * (c.hover_height - pos[2]) \
            - c.hover_kd * vel[2]
        return Command(float(thrust), np.zeros(3), 0.0)

    def _goto(self, pos: np.ndarray, target: np.ndarray) -> Command:
        c = self.cfg
        d = np.asarray(target, float) - np.asarray(pos, float)
        thrust = c.mass * c.gravity + c.hover_kp * d[2]
        # 位置误差 → 期望倾角 → 力矩（只用 P，够回得去）
        torque = np.array([-c.rtl_kp * d[1], c.rtl_kp * d[0], 0.0])
        return Command(float(thrust), torque, 0.0)

    # ---------------- 限幅与限速 ----------------
    def _clip(self, cmd: Command) -> tuple[float, np.ndarray]:
        cfg = self.cfg
        thrust = float(np.clip(cmd.thrust, 0.0, cfg.thrust_abs_max))
        torque = np.clip(np.asarray(cmd.torque, float), -cfg.torque_abs_max, cfg.torque_abs_max)
        return thrust, torque

    def _slew(self, thrust: float) -> float:
        prev = self.st.last_good.thrust if self.st.last_good else thrust
        d = np.clip(thrust - prev, -self.cfg.slew_limit, self.cfg.slew_limit)
        return float(prev + d) if self.st.mode != Mode.NORMAL else thrust

    # ---------------- 状态机 ----------------
    def _update_mode(self, t: float, cmd: Command | None, pos: np.ndarray) -> None:
        cfg = self.cfg
        stale = cmd is None or (t - cmd.stamp) > cfg.heartbeat_timeout_s
        out_of_fence = not (self.fence[0] + cfg.fence_margin <= pos[0] <= self.fence[1] - cfg.fence_margin
                            and self.fence[2] + cfg.fence_margin <= pos[1] <= self.fence[3] - cfg.fence_margin)

        want = None
        if out_of_fence:
            want = Mode.RTL
        elif stale:
            want = Mode.HOLD

        # 已在保护态时按"在该态待了多久"继续升级（阶梯只升不降，除了回 NORMAL）
        if stale and self.st.mode >= Mode.HOLD:
            age = t - self.st.last_mode_change
            nxt = {Mode.HOLD: (cfg.hold_timeout_s, Mode.HOVER),
                   Mode.HOVER: (cfg.hover_timeout_s, Mode.RTL),
                   Mode.RTL: (cfg.rtl_timeout_s, Mode.LAND)}.get(self.st.mode)
            if nxt is not None and age > nxt[0]:
                self._switch(t, nxt[1])
            return

        if want is not None and want > self.st.mode:
            self._enter_count += 1
            self._exit_count = 0
            if self._enter_count >= cfg.enter_cycles:
                self._switch(t, want)
                self._enter_count = 0
            return

        if want is None and self.st.mode != Mode.NORMAL:
            self._exit_count += 1
            self._enter_count = 0
            if self._exit_count >= cfg.exit_cycles:
                self._switch(t, Mode.NORMAL)
                self._exit_count = 0
            return
        self._enter_count = 0

    def _switch(self, t: float, mode: Mode) -> None:
        if mode != self.st.mode:
            self.st.log(t, f"{self.st.mode.name}->{mode.name}")
            self.st.mode = mode
            self.st.last_mode_change = t
