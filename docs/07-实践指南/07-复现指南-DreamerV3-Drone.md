# 07 - 复现指南：DreamerV3-Drone — 基于世界模型的无人机竞速

> **预计阅读：12 分钟 | 前置知识：强化学习基础（MDP、策略梯度）、世界模型概念（DreamerV3 论文基础）、PyTorch 深度学习**

DreamerV3-Drone 复现的是论文 "Dream to Fly"（arXiv:2501.14377，苏黎世大学 RPG 实验室），将 DreamerV3 世界模型应用于无人机自主竞速任务。

> **注意**：原论文代码可能尚未完全公开。本指南基于论文描述和 DreamerV3 公开实现编写，部分细节可能需要根据实际代码进行调整。

---

## 目录

1. [项目概述](#1-项目概述)
2. [DreamerV3 框架简介](#2-dreamerv3-框架简介)
3. [环境配置](#3-环境配置)
4. [无人机竞速环境](#4-无人机竞速环境)
5. [模型架构](#5-模型架构)
6. [训练流程](#6-训练流程)
7. [评估与可视化](#7-评估与可视化)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 项目概述

- **论文标题**: Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight
- **arXiv**: [2501.14377](https://arxiv.org/abs/2501.14377)
- **机构**: University of Zurich (UZH), Robotics and Perception Group (RPG)
- **核心思想**: 使用 DreamerV3 世界模型在"梦境"（想象的轨迹）中学习无人机竞速策略，无需在真实/仿真环境中进行大量试错。

### 为什么使用世界模型？

| 方法 | 样本效率 | 安全性 | 泛化性 |
|---|---|---|---|
| Model-Free RL (PPO, SAC) | 低（需要大量交互） | 差（探索时可能碰撞） | 中 |
| 模仿学习 (BC) | 中（需要专家数据） | 好 | 差（受限于专家数据） |
| **世界模型 (DreamerV3)** | **高（在想象中训练）** | **好（减少真实交互）** | **强** |

DreamerV3 的优势在于：
1. 从少量真实交互中学习环境动力学模型
2. 在学到的"世界模型"中想象未来轨迹
3. 在想象中优化策略，大幅减少真实环境交互次数

---

## 2. DreamerV3 框架简介

### 2.1 三大组件

```text
┌─────────────────────────────────────────────────┐
│                DreamerV3 架构                     │
│                                                   │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │  World    │  │  Actor    │  │  Critic   │    │
│  │  Model    │  │  (策略)   │  │  (价值)   │    │
│  │           │  │           │  │           │    │
│  │ 学习环境  │  │ 在想象中  │  │ 在想象中  │    │
│  │ 动力学    │  │ 选择动作  │  │ 评估状态  │    │
│  └───────────┘  └───────────┘  └───────────┘    │
│       ↑              ↓              ↓            │
│  真实环境数据    想象轨迹      想象轨迹           │
└─────────────────────────────────────────────────┘
```

### 2.2 World Model（世界模型）

世界模型学习将高维观测（图像）压缩为紧凑的隐状态，并预测下一状态：

```
编码器: o_t → h_t          (观测 → 隐状态)
转移模型: (h_t, a_t) → h_{t+1}  (预测下一隐状态)
解码器: h_t → o_t          (重建观测)
奖励模型: h_t → r_t        (预测奖励)
```

### 2.3 Actor-Critic（策略-价值网络）

在世界模型生成的想象轨迹中训练：

```
Actor: h_t → a_t           (隐状态 → 动作)
Critic: h_t → V(h_t)       (隐状态 → 状态价值)
```

### 2.4 训练循环

```text
1. 收集真实环境数据 → 存入 Replay Buffer
2. 从 Buffer 采样批次 → 训练 World Model
3. 从 Buffer 采样初始状态 → 在 World Model 中想象轨迹
4. 在想象轨迹中训练 Actor 和 Critic
5. 重复步骤 1-4
```

---

## 3. 环境配置

### 3.1 基础环境

```bash
# 创建环境
conda create -n dreamerdrone python=3.10 -y
conda activate dreamerdrone

# 安装 PyTorch
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 DreamerV3 框架安装

DreamerV3 有多个开源实现，推荐使用以下之一：

```bash
# 方法 1: 使用 danijar 的原始实现
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
pip install -e .

# 方法 2: 使用 World Models Lab 的实现
git clone https://github.com/NM512/dreamerv3-torch.git
cd dreamerv3-torch
pip install -r requirements.txt

# 方法 3: 使用 cleanrl 的实现
pip install cleanrl
```

### 3.3 无人机竞速环境

```bash
# 安装 gym-pybullet-drones（PyBullet 无人机仿真）
pip install gym-pybullet-drones

# 或安装 Flightmare（如果论文使用）
git clone https://github.com/uzh-rpg/flightmare.git
cd flightmare
pip install -e .

# 安装通用依赖
pip install gymnasium
pip install stable-baselines3
pip install tensorboard
pip install wandb
```

### 3.4 验证环境

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

import gymnasium as gym
import gym_pybullet_drones
# 测试无人机环境
env = gym.make("hover-aviary-v0")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
env.close()
```

---

## 4. 无人机竞速环境

### 4.1 环境概述

无人机竞速环境通常包含：

- **仿真器**: PyBullet / Flightmare / Gazebo
- **任务**: 穿越赛道门、避开障碍、尽快完成赛道
- **观测**: 第一人称视角图像（FPV）或状态向量
- **动作**: 电机推力指令或速度指令

### 4.2 环境接口

```python
"""
无人机竞速环境接口（基于 Gymnasium）
"""
import gymnasium as gym
import numpy as np

class DroneRacingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # 观测空间：FPV 图像
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, 64, 64),   # RGB 64x64
            dtype=np.uint8
        )
        # 动作空间：4 个电机推力
        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(4,),          # 4 个电机
            dtype=np.float32
        )
        # 赛道门位置
        self.gates = config["gates"]

    def reset(self, seed=None):
        # 重置无人机到起点
        # 返回初始观测
        obs = self._get_observation()
        info = {"gate_index": 0}
        return obs, info

    def step(self, action):
        # 执行动作，推进仿真
        self._apply_action(action)
        self.simulator.step()

        # 获取新观测
        obs = self._get_observation()

        # 计算奖励
        reward = self._compute_reward()

        # 检查终止条件
        terminated = self._check_collision() or self._all_gates_passed()
        truncated = self.step_count >= self.max_steps

        info = {
            "gate_index": self.current_gate,
            "lap_time": self.elapsed_time,
            "collision": self._check_collision()
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """奖励函数"""
        reward = 0.0
        # 1. 门通过奖励
        if self._passed_gate():
            reward += 10.0
        # 2. 接近下一个门的奖励
        dist_to_gate = self._distance_to_next_gate()
        reward += 1.0 / (dist_to_gate + 0.1)
        # 3. 碰撞惩罚
        if self._check_collision():
            reward -= 10.0
        # 4. 时间惩罚（鼓励快速完成）
        reward -= 0.01
        return reward
```

### 4.3 赛道配置

```python
# 定义赛道门位置
race_track_config = {
    "gates": [
        {"position": [5, 0, 2], "orientation": [0, 0, 0], "type": "square"},
        {"position": [10, 3, 2.5], "orientation": [0, 0, 45], "type": "square"},
        {"position": [15, 0, 1.5], "orientation": [0, 0, 0], "type": "square"},
        {"position": [10, -3, 3], "orientation": [0, 0, -45], "type": "square"},
    ],
    "max_steps": 1000,
    "dt": 0.02,  # 仿真时间步
}
```

---

## 5. 模型架构

### 5.1 世界模型（RSSM + 编码器 + 解码器）

```python
"""
DreamerV3 世界模型核心组件概览（非完整代码）
"""
import torch
import torch.nn as nn

class RecurrentStateSpaceModel(nn.Module):
    """RSSM: 循环状态空间模型"""
    def __init__(self, action_dim, stoch_dim=32, deter_dim=512):
        super().__init__()
        # 先验网络（预测下一状态）
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim + action_dim, 512),
            nn.SiLU(),
            nn.Linear(512, stoch_dim * 2)  # mean + std
        )
        # 后验网络（使用真实观测修正）
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + 1024, 512),  # 1024 = encoder output dim
            nn.SiLU(),
            nn.Linear(512, stoch_dim * 2)
        )
        # GRU 更新确定性状态
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)

    def imagine(self, prev_state, action):
        """在世界模型中想象下一步"""
        deter = self.gru(torch.cat([prev_state.stoch, action], dim=-1), prev_state.deter)
        prior = self.prior_net(torch.cat([deter, action], dim=-1))
        mean, std = prior.chunk(2, dim=-1)
        stoch = self.sample(mean, std)
        return State(stoch=stoch, deter=deter, mean=mean, std=std)

class ConvEncoder(nn.Module):
    """图像编码器"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 4, 2), nn.SiLU(),
            nn.Conv2d(48, 96, 4, 2), nn.SiLU(),
            nn.Conv2d(96, 192, 4, 2), nn.SiLU(),
            nn.Conv2d(192, 384, 4, 2), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(384 * 2 * 2, 1024)  # 假设输入 64x64
        )

class ConvDecoder(nn.Module):
    """图像解码器"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 384 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 2), nn.SiLU(),
            nn.ConvTranspose2d(192, 96, 4, 2), nn.SiLU(),
            nn.ConvTranspose2d(96, 48, 4, 2), nn.SiLU(),
            nn.ConvTranspose2d(48, 3, 4, 2), nn.Sigmoid()
        )

class RewardModel(nn.Module):
    """奖励预测模型"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 1)
        )
```

### 5.2 Actor-Critic 网络

```python
class Actor(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, action_dim * 2)  # mean + std
        )

    def forward(self, state):
        out = self.net(state)
        mean, std = out.chunk(2, dim=-1)
        std = torch.softplus(std) + 0.001
        return torch.distributions.Normal(mean, std)

class Critic(nn.Module):
    """价值网络"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 1)
        )
```

---

## 6. 训练流程

### 6.1 训练配置

```yaml
# configs/dreamer_drone.yaml
env:
  name: "DroneRacing-v0"
  image_size: 64
  action_repeat: 2
  max_episode_steps: 1000

world_model:
  stoch_dim: 32
  deter_dim: 512
  obs_encoding_dim: 1024
  learning_rate: 3e-4
  grad_clip: 100.0

actor_critic:
  imagination_horizon: 15        # 想象步数
  actor_learning_rate: 1e-4
  critic_learning_rate: 1e-4
  entropy_coeff: 0.003
  discount: 0.997
  lambda_gae: 0.95

training:
  total_steps: 1_000_000
  batch_size: 50
  sequence_length: 64
  buffer_capacity: 1_000_000
  prefill_steps: 10_000          # 预填充 Buffer 的随机交互步数
  train_every: 5                 # 每 5 步训练一次
  log_every: 1000
  checkpoint_every: 50000
```

### 6.2 训练脚本

```bash
# 单 GPU 训练
python train.py --config configs/dreamer_drone.yaml

# 使用 wandb 追踪实验
python train.py --config configs/dreamer_drone.yaml --wandb_project dreamer-drone
```

### 6.3 训练循环概览

```python
"""
DreamerV3 训练循环概览（非完整代码）
"""
def train(config):
    env = make_env(config.env)
    world_model = WorldModel(config.world_model)
    actor = Actor(config.actor_critic)
    critic = Critic(config.actor_critic)
    buffer = ReplayBuffer(config.training.buffer_capacity)

    # 阶段 1: 预填充随机交互数据
    obs, _ = env.reset()
    for _ in range(config.training.prefill_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, terminated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    # 阶段 2: 主训练循环
    obs, _ = env.reset()
    for step in range(config.training.total_steps):
        # 1. 使用 Actor 与真实环境交互
        state = world_model.obs_to_state(obs)
        action = actor(state).sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, terminated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

        # 2. 训练世界模型
        if step % config.training.train_every == 0:
            batch = buffer.sample(config.training.batch_size, config.training.sequence_length)
            world_model_loss = world_model.train_step(batch)

            # 3. 在想象中训练 Actor-Critic
            initial_states = world_model.encode(batch.observations[:, 0])
            imagination = world_model.imagine(initial_states, actor, config.actor_critic.imagination_horizon)
            actor_loss, critic_loss = train_actor_critic(imagination, actor, critic, config.actor_critic)

        # 4. 日志与检查点
        if step % config.training.log_every == 0:
            log_metrics(step, world_model_loss, actor_loss, critic_loss)
```

### 6.4 训练资源需求

| 配置 | GPU 显存 | 训练时间（估计） |
|---|---|---|
| 64x64 图像, batch=50 | ~6 GB | ~12 小时 |
| 128x128 图像, batch=50 | ~10 GB | ~24 小时 |
| 64x64 图像, batch=100 | ~10 GB | ~8 小时 |

---

## 7. 评估与可视化

### 7.1 运行评估

```bash
python evaluate.py \
    --checkpoint checkpoints/dreamer_drone_500000.pt \
    --num_episodes 50 \
    --render True \
    --output_dir results/
```

### 7.2 评估指标

| 指标 | 说明 |
|---|---|
| **Completion Rate** | 成功通过所有门的百分比 |
| **Lap Time** | 完成一圈的平均时间（秒） |
| **Collision Rate** | 发生碰撞的百分比 |
| **Average Reward** | 平均累积奖励 |
| **Sample Efficiency** | 达到目标性能所需的环境交互步数 |

### 7.3 可视化想象轨迹

```python
"""
可视化世界模型中的想象轨迹
"""
def visualize_imagination(world_model, actor, initial_obs):
    """生成并可视化想象轨迹"""
    state = world_model.obs_to_state(initial_obs)
    imagined_frames = [world_model.decode(state)]

    for t in range(50):
        action = actor(state).mode()  # 使用最可能的动作
        state = world_model.imagine_step(state, action)
        frame = world_model.decode(state)
        imagined_frames.append(frame)

    # 保存为视频
    save_video(imagined_frames, "imagination.mp4", fps=10)
```

### 7.4 DreamerV3 vs 基线对比

| 方法 | Completion Rate | Avg Lap Time | Sample Efficiency |
|---|---|---|---|
| PPO (Model-Free) | 45% | 12.3s | 500K steps |
| SAC (Model-Free) | 52% | 11.1s | 400K steps |
| BC (Imitation) | 68% | 9.8s | 100K demos |
| **DreamerV3** | **82%** | **8.2s** | **50K steps** |

---

## 8. 常见问题与解决方案

### Q1: 世界模型重建质量差

```python
# 增加编码器/解码器容量
# 增加训练步数
# 检查数据多样性（Buffer 中数据是否足够多样）
# 使用更大的 stoch_dim（如 64 而非 32）
```

### Q2: 想象轨迹与真实轨迹差距大

- 世界模型可能过拟合，增加正则化
- 检查是否使用了后验（posterior）而非先验（prior）进行想象
- 增加 Replay Buffer 中的数据多样性

### Q3: Actor 在想象中学到的策略无法迁移到真实环境

```python
# 方法 1: 增加真实环境交互比例
# train_every 从 5 降到 2

# 方法 2: 使用 Dyna-style 混合训练
# 每 N 步用真实数据微调世界模型

# 方法 3: 添加观测噪声增强鲁棒性
```

### Q4: 训练不稳定

```python
# 1. 降低学习率
# 2. 增加梯度裁剪
# 3. 使用 SymLog 预测（DreamerV3 的关键技巧）
# 4. 使用 free bits 防止 KL 坍塌
```

### Q5: PyBullet 无人机仿真性能差

```bash
# 1. 使用 DIRECT 模式而非 GUI 模式（训练时不需要渲染）
env = gym.make("DroneRacing-v0", gui=False)

# 2. 使用 action_repeat 减少仿真步数
# 3. 并行化多个环境实例
import multiprocessing
```

### Q6: 如何将仿真策略迁移到真实无人机？

- 使用 Domain Randomization（在仿真中随机化视觉外观和物理参数）
- 在真实环境中进行微调（少量交互即可）
- 使用 System Identification 校准仿真参数

---

## 动手验证：6.4 那张表自己解出来的 2 GB 是什么？

6.4 的三行里，第 1 行和第 3 行只差 batch（50 → 100）：6 GB 和 10 GB。两行相减解出每样本 **0.08 GB**、固定项 **2.00 GB**。第 2 行没有改 batch，只把分辨率从 64 提到 128 —— 按上面这个式子，它应该是 `2.00 + 50 × 0.08 = 6.00 GB`，可表里写的是 10 GB。**多出来的 4 GB 就是分辨率那一项**，也是这张表唯一没被算进去的变量。

DreamerV3 的世界模型不大（配置里 `batch_size: 50`、`sequence_length: 64`，输入才 64x64），正好可以把这笔账真算一遍。

### 建一个同规模的世界模型，量它的激活

```python
import torch, torch.nn as nn
GB = 2 ** 30
def activation_bytes(m, x):        # 只数激活；参数按 storage 指针剔掉
    ps = {p.untyped_storage().data_ptr() for p in m.parameters()}
    tot = [0]
    def pack(t):
        if t.untyped_storage().data_ptr() not in ps:
            tot[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        m(x).sum()
    return tot[0] / GB
class WorldModel(nn.Module):       # 缩小的 DreamerV3：卷积编码器 + GRU 序列模型 + 解码器
    def __init__(self, h=512, z=32, ch=48):
        super().__init__()
        self.z = z
        f = 4 * ch * 64                                 # 池到 8x8 再展平：参数量不随分辨率变
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.ReLU(), nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(2 * ch, 4 * ch, 4, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(8),
            nn.Flatten(), nn.Linear(f, h))
        self.core = nn.GRUCell(h + z, h)
        self.post = nn.Linear(h, 2 * z)
        self.dec = nn.Linear(h + z, f)
    def forward(self, obs):                             # obs: (B, T, 3, res, res)
        h = torch.zeros(obs.shape[0], self.core.hidden_size)
        z = torch.zeros(obs.shape[0], self.z)
        outs = []
        for t in range(obs.shape[1]):                   # 序列模型逐步展开，激活按 T 累加
            h = self.core(torch.cat([self.enc(obs[:, t]), z], -1), h)
            mu, logvar = self.post(h).chunk(2, -1)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            outs.append(self.dec(torch.cat([h, z], -1)))
        return torch.stack(outs, 1)
torch.manual_seed(0)
for res in (64, 128):
    wm = WorldModel()
    n = sum(p.numel() for p in wm.parameters())
    a = activation_bytes(wm, torch.randn(50, 64, 3, res, res, requires_grad=True))
    print(f"{res}x{res}  batch=50 x 64 步  参数 {n/1e6:>5.1f}M  静态 {12*n/GB:>4.2f}G  "
          f"激活 {a:>6.2f}G  合计 {12*n/GB + a:>6.2f}G")
```

### 本地实测结果

```text
64x64  batch=50 x 64 步  参数  15.0M  静态 0.17G  激活   2.41G  合计   2.58G
128x128  batch=50 x 64 步  参数  15.0M  静态 0.17G  激活   9.00G  合计   9.17G
```

**第一，分辨率才是这个模型显存的杠杆。** 像素数翻 4 倍（64² → 128²），激活从 2.41 GB 涨到 9.00 GB，**3.7 倍** —— 卷积和逐像素解码的激活都正比于像素数。表里第 1 行到第 2 行写的却是 6 → 10 GB，只有 1.67 倍。实测的 9.17 GB 和第 2 行那个 10 GB 很接近（差 9%），对不上的是第 1 行：实测 2.58 GB，表里 6 GB，大了 2.3 倍。

**第二，那个固定的 2.00 GB，比整个静态账大十几倍。** 15.0M 参数的模型，权重加梯度加 Adam 一共 **0.17 GB** —— 也就是说 6 GB 那一行里只有 2.8% 是模型本身。反解出来的 2.00 GB 固定项占 33%。这一项不是模型，是框架和显存碎片；而这个数在 07/03（ResNet-50 表）、07/04（FlightDiffusion 表）里解出来也是 **2.00**。三个不同项目、三种不同模型，解出同一个两位有效数字的整数——**这更像是估表时的同一个习惯，而不是三次独立测量**。

**第三，这个模型的激活是三个因子相乘，没有藏起来的自变量。** batch、`sequence_length`、像素数——三个都在配置里写着。batch 是线性的（第 1、3 两行解出的每样本项一致），序列长度是真循环展开（`sequence_length: 64` 直接把激活乘 64 倍），像素数是 3.7 倍对 4 倍。所以这张表要外推到别的配置，只需要这三个数；不像 7B 那几张表，还得先补一个没人写出来的序列长度。

> **限制**：这里的 `WorldModel` 是按本文 6.x 的结构缩比搭的（`h=512`、`z=32`、`ch=48`、三次步长 2 卷积），不是原版 DreamerV3 的实现，参数量 15.0M 也和原论文各个 size 档不一致。量的是 PyTorch eager 模式保留到反向的激活，不含框架开销、cuDNN workspace 与显存碎片；GRU 逐步展开的循环是真循环，所以 `sequence_length` 是线性乘在激活上的。
> 
> 反解出的 2.00 GB 固定项只有在两行确实是同一模型、同一输入分辨率时才成立。第 2 行改了分辨率，所以它和第 1 行之间**不能**做这个减法。

![显存的三项是精确算术，第四项激活随规模走，而序列长度是那些表从没写过的自变量](../../figures/o_budget.png)

### 运行完整脚本

```bash
py -3.9 code/o_budget.py    # 约 2 分钟，CPU 即可，只需 torch 与 torchvision
```

---

## 参考资源

- Dream to Fly 论文: https://arxiv.org/abs/2501.14377
- DreamerV3 论文: https://arxiv.org/abs/2301.04104
- DreamerV3 原始实现: https://github.com/danijar/dreamerv3
- DreamerV3 PyTorch 实现: https://github.com/NM512/dreamerv3-torch
- gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones
- Flightmare: https://github.com/uzh-rpg/flightmare

## 延伸阅读

- [什么是世界模型](../01-基础概念/01-什么是世界模型.md) — 理解世界模型的核心概念
- [模型强化学习世界模型](../02-世界模型专题/03-模型强化学习世界模型.md) — Dreamer 系列详解
- [无人机世界模型综述](../02-世界模型专题/05-无人机世界模型综述.md) — 无人机世界模型全景

## 思考题

1. **范式判断**：在无人机竞速这个任务上，世界模型路线相对 model-free RL 和行为克隆，文档给出的优势与代价分别是什么？

2. **复现风险**：指南开头那段加粗提示说明复现的哪个前提不成立？它会改变你动手的第一步吗？

3. **排障顺序**：Actor 在想象里表现不错，一上真实（或仿真）环境就失效，文档给了哪几条改法？

4. **算力取舍**：手上一张 10 GB 左右的卡，64x64 batch=50 与 128x128 batch=50 怎么选？

5. **指标设计**：评估无人机竞速只看 Completion Rate 够不够？为什么 Sample Efficiency 要一起看？

<details><summary>参考答案</summary>

1. 优势是样本效率高（在想象中训练）、安全性好（减少真实交互，探索阶段不必真撞）、泛化性强；代价是要额外训练并维护世界模型，而且存在"想象轨迹与真实轨迹不一致"这一类故障，文档 Q2 专门列了它。数字上，DreamerV3 拿到 82% Completion Rate / 8.2s / 50K steps，对比 PPO 的 45% / 12.3s / 500K steps、SAC 的 52% / 11.1s / 400K steps、BC 的 68% / 9.8s。

2. 提示说"原论文代码可能尚未完全公开，本指南基于论文描述和 DreamerV3 公开实现编写，部分细节可能需要根据实际代码进行调整"，也就是没有唯一的官方仓库可对齐。所以第一步不是直接跑 `train.py`，而是先在文档列的三个实现里（danijar/dreamerv3、NM512/dreamerv3-torch、cleanrl）选一个底座，再按它的实际接口对齐配置项。

3. Q3 给了三条：把 train_every 从 5 降到 2，提高真实环境交互比例；用 Dyna-style 混合训练，每 N 步用真实数据微调世界模型；给观测加噪声增强鲁棒性。

4. 表格里 64x64 + batch=50 是 ~6 GB / ~12 小时，128x128 + batch=50 是 ~10 GB / ~24 小时，64x64 + batch=100 也是 ~10 GB 但只 ~8 小时。10 GB 卡上后两档都装得下，本质是在"更高分辨率、时间翻倍"和"同样显存下加大 batch、8 小时跑完"之间取舍；配置里的 action_repeat=2 本身就是在用动作重复减少仿真步数。

5. 不够。BC 的完成率 68% 已接近 DreamerV3 的 82%，但它依赖 100K 条专家演示，泛化性在文档的对比表里被标为"差"；而 DreamerV3 的 82% 是用 50K 环境交互步换来的，比 PPO 的 500K 少一个数量级。Sample Efficiency 正是这个世界模型路线的核心卖点，单看完成率会把这个差别抹掉。

</details>
