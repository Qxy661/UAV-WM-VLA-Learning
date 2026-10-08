"""
四旋翼仿真环境（torch 批量实现）

A/B/E/F/G/K/P/Q/R/S 十个 demo 共用的最小仿真器：不追求气动保真，只要求
  - 状态/动作的物理含义清楚
  - 有姿态内环，控制器不会因为倾角饱和而自激
  - 可批量并行，CPU 上也能几秒跑完

状态 12 维: [p(3), v(3), rpy(3), omega(3)]
    p     世界系位置 (m)
    v     世界系速度 (m/s)
    rpy   横滚/俯仰/偏航 (rad)
    omega 机体角速率 (rad/s)

动作 4 维: [thrust, wx, wy, wz] ∈ [-1, 1]
    thrust  归一化总距，0.5 ≈ 悬停（因为 acc_z = thrust*2g - g）
    wx/wy/wz 机体角速率指令 (rad/s)，内部按 OMEGA_MAX 缩放

坐标约定（纯约定，为的是让"哪个轴修正哪个误差"一目了然）:
    正 roll(rpy[0])  -> +x 方向加速
    正 pitch(rpy[1]) -> +y 方向加速
真实四旋翼的机体系定义与此不同，这里换成对角形式是为了让
Demo B 里"侧向轴交叉耦合"这个 bug 一眼能看出来。
"""

import torch

# 角速率指令的满量程 (rad/s)
OMEGA_MAX = 5.0
# 倾角限幅 (rad) —— 超过约 69° 后 tan() 会发散，必须限
TILT_LIMIT = 1.2
# 速度阻尼系数，模拟气动阻力
DRAG = 0.1


class QuadSim:
    """批量四旋翼仿真器。

    典型用法::

        env = QuadSim(256)
        obs = env.reset()
        for t in range(T):
            obs = env.step(action)   # action: (256, 4)
    """

    def __init__(self, n, dt=0.02, g=9.81):
        self.n, self.dt, self.g = n, dt, g
        self.reset()

    def reset(self, rand=False, pos_std=0.5, vel_std=0.3):
        """重置。rand=True 时在初始位置/速度上加高斯扰动，用于制造分布漂移。"""
        if rand:
            self.p = pos_std * torch.randn(self.n, 3)
            self.v = vel_std * torch.randn(self.n, 3)
        else:
            self.p = torch.zeros(self.n, 3)
            self.v = torch.zeros(self.n, 3)
        self.rpy = torch.zeros(self.n, 3)
        self.om = torch.zeros(self.n, 3)
        return self.obs()

    def obs(self):
        return torch.cat([self.p, self.v, self.rpy, self.om], dim=-1)

    def step(self, a):
        """推进一步。a: (n, 4)，超出 [-1,1] 会被截断（模拟执行器饱和）。"""
        a = a.clamp(-1.0, 1.0)
        acc = torch.zeros(self.n, 3)
        # 总距 -> 垂向加速度：thrust=0.5 时恰好抵消重力
        acc[:, 2] = a[:, 0] * self.g * 2 - self.g
        # 倾角 -> 水平加速度（小角度下 tan(θ) ≈ θ）
        acc[:, :2] = torch.tanh(self.rpy[:, :2]) * self.g
        acc = acc - DRAG * self.v

        self.p = self.p + self.v * self.dt
        self.v = self.v + acc * self.dt
        # 角速率指令直接驱动姿态（一阶响应），倾角限幅
        self.om = a[:, 1:] * OMEGA_MAX
        self.rpy = torch.clamp(self.rpy + self.om * self.dt, -TILT_LIMIT, TILT_LIMIT)
        return self.obs()


def expert(env, goal, kp=0.30, kd=0.70, ka=3.0, tilt_max=0.6):
    """串级控制器，充当"脚本专家"。

    级联结构（这是无人机姿态控制的常规做法）::

        位置环 P-D  ->  期望倾角 tilt_des
        姿态环 P    ->  角速率指令
        角速率      ->  直接下发给 QuadSim

    关键点：角速率指令是**开环**的，没有姿态反馈就会一直积分到限幅自激。
    Demo B 里"增益调高反而更差"的根因就在这里。

    返回: (n, 4) 动作
    """
    # 位置环：把 x/y 误差转成期望倾角
    tilt_des = torch.stack([
        torch.clamp(kp * (goal[:, 0] - env.p[:, 0]) - kd * env.v[:, 0], -tilt_max, tilt_max),
        torch.clamp(kp * (goal[:, 1] - env.p[:, 1]) - kd * env.v[:, 1], -tilt_max, tilt_max),
    ], dim=-1)
    a = torch.zeros(env.n, 4)
    # 高度环：thrust 基线 0.5 悬停，叠加 P-D 修正
    a[:, 0] = 0.5 + 0.10 * (goal[:, 2] - env.p[:, 2]) - 0.30 * env.v[:, 2]
    # 姿态内环：把倾角误差转成角速率（缺了这一环，位置环增益一高就振荡）
    a[:, 1:3] = ka * (tilt_des - env.rpy[:, :2])
    return a.clamp(-1, 1)


def reward(obs, goal):
    """教学用奖励：离目标越近越好，速度越小越好。"""
    return -torch.norm(obs[:, :3] - goal, dim=-1) - 0.1 * torch.norm(obs[:, 3:6], dim=-1)
