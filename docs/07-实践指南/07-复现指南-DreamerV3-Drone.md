# 07 - 复现指南：DreamerV3-Drone — 基于世界模型的无人机竞速

> **预计阅读：12 分钟 | 前置知识：强化学习基础（MDP、策略梯度）、世界模型概念（DreamerV3 论文基础）、PyTorch 深度学习**

DreamerV3-Drone 复现的是论文 "Dream to Fly"（arXiv:2501.14377，苏黎世大学 RPG 实验室），将 DreamerV3 世界模型应用于无人机自主竞速任务。

> **注意**：论文只点名了它用的 DreamerV3 底座（PyTorch 版 `dreamerv3-torch`），**没有放出自己那套无人机竞速环境的代码**（Flightmare + Habitat 的接法、赛道生成、奖励实现都不在仓库里）。本指南据论文描述与 DreamerV3 公开实现编写，凡是没有出处的地方都在下文标注了。

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
- **核心思想**: 用 DreamerV3 从像素直接学到无人机竞速策略，不依赖中间表征，也不用模仿学习做引导（bootstrapping）。真机只做硬件在环（HIL）验证。

### 为什么使用世界模型？

论文给的对照不是一张评分表，而是一句样本效率判断加一句代价：

| 路线 | 论文里的说法 |
|---|---|
| Model-Free RL（PPO、SAC） | 在 2000 万次环境交互内**学不出可用机动**，奖励始终很低；两条基线都得先做中间表征或模仿学习引导才能训起来 |
| 模仿学习（IL）引导 + RL 微调 | 前人工作（论文 [10]）的做法：先把视觉输入压成只含赛道门的二值掩码，再用 IL 引导、再 RL 微调。本文的目的就是绕开这套 |
| **世界模型（DreamerV3）** | 从像素直接学到策略，三条赛道都收敛，既不要中间表征也不要 IL 引导。**代价是训练时间显著更长——论文自述约 240 小时** |

DreamerV3 的三步是这样接起来的：
1. 在**仿真环境**里采集交互数据，学到环境动力学
2. 在学到的「世界模型」里想象未来轨迹
3. 在想象里训练 Actor-Critic，于是策略梯度不必再花真实交互

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
# 创建环境（dreamerv3-torch 的 README 要求 python 3.11）
conda create -n dreamerdrone python=3.11 -y
conda activate dreamerdrone

# 安装 PyTorch（对齐 dreamerv3-torch 的 requirements：torch==2.4.1）
pip install torch==2.4.1 torchvision
```

### 3.2 DreamerV3 框架安装

论文点名的底座是 PyTorch 版 DreamerV3：

```bash
git clone https://github.com/NM512/dreamerv3-torch.git
cd dreamerv3-torch
pip install -r requirements.txt
```

### 3.3 无人机竞速环境

```bash
# 动力学与赛道生成：Flightmare（论文用的那一层）
git clone https://github.com/uzh-rpg/flightmare.git
cd flightmare
pip install -e .

# 渲染器在环：Habitat（论文把它的图像流直接接进训练循环）
# 平台参数与真机平台来自 Agilicious：https://github.com/uzh-rpg/agilicious

# 训练侧依赖
pip install stable-baselines3
pip install tensorboard
```

### 3.4 验证环境

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

```bash
# 先跑通底座自带的最小任务，确认入口能起来（上游 README 的冒烟测试就是这条）
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/smoke
```

> 无人机竞速任务要自己把 Flightmare + Habitat 接成环境再注册 config，没有现成的 pip 包可装。`hover-aviary-v0` 属于 `gym-pybullet-drones`，与本文栈无关。

---

## 4. 无人机竞速环境

### 4.1 环境概述

论文这套环境由三层拼起来：

- **动力学与赛道**: Flightmare 提供四旋翼动力学与赛道生成，物理参数取自 Agilicious——质量 `m = 0.6 kg`、转动惯量 `J = diag([0.002410, 0.001800, 0.003759]) kg·m²`、旋翼力矩常数 `κ = 0.022`、臂长 `0.14 m`，最大旋翼推力限到 `4.0 N`（推重比 2.7）
- **渲染器**: Habitat，在环跑 *"several thousand frames per second"*，训练时直接拿到图像流
- **观测**: 相机图像缩放为 **64×64 RGB**（论文原话 *"Input images from the simulated camera were resized to 64×64 RGB pixels"*）
- **动作**: **集体推力 + 机体角速度**（原文 *"collective thrust and bodyrate commands"*），不是四个电机的推力
- **赛道**: Circle / Kidney / Figure 8 三条，每条跑 **5 个随机种子**
- **奖励**: 主项是进度项 `b₁(‖g_k − p_{k−1}‖ − ‖g_k − p_k‖)`（`g` 是目标门中心、`p` 是无人机位置），另含机体角速度项；`b₂` 走课程学习，初始为 `0.0`，总奖励超过 `50.0` 后逐渐加大

### 4.2 环境接口

> **下面这段只是接口示意，不是论文或仓库里的代码。** 真实的观测、动作与奖励如 4.1 所列；`gym.Env` 五元组 `(obs, reward, terminated, truncated, info)` 是新版 Gymnasium 的接口，dreamerv3-torch 用的是老 API（`gym==0.22.0`）的四元组。

```python
"""
无人机竞速环境接口示意（非论文/仓库代码）
"""
import gymnasium as gym
import numpy as np

class DroneRacingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # 观测空间：相机图像 64x64 RGB
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, 64, 64),   # RGB 64x64
            dtype=np.uint8
        )
        # 动作空间：集体推力 + 三轴机体角速度
        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(4,),          # [thrust, wx, wy, wz]
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

```yaml
# 论文用的三条赛道；门位姿由 Flightmare 的赛道生成器给出
track: circle      # 另两条：kidney、figure8
time_limit: 1000   # 单回合计步上限
seed: 0            # 每条赛道跑 5 个种子：0..4
```

---

## 5. 模型架构

> **本节的代码是结构示意，不是论文或仓库的实现。** 论文对自己那套网络的交代只有一句：*"we adopted the hyperparameters detailed in [13] and used the Large (L) network configuration described in Appendix B of [41]. This configuration consists of four-layer MLPs with 768 units for the decoder, predictors, actor, and critic networks, and 2048 recurrent units for the world model's recurrent component."* 也就是 §5.2 里的 actor / critic 各是**四层 768 单元**（不是下面写的 512），世界模型的循环部分是 **2048 单元**。想象时域 `T = 16`，这是论文自己设的，与 dreamerv3-torch 默认的 `imag_horizon: 15` 不同。

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
# 上游 configs.yaml 的 defaults 段（下面是其中的真实键名与默认值）
task: drone_racing     # 无人机任务需自己接 Flightmare + Habitat 后注册
size: [64, 64]         # 观测缩放到 64x64 RGB
action_repeat: 2
time_limit: 1000
seed: 0
steps: 1e6             # 论文跑到 2e7 量级，见 6.4
prefill: 2500          # 随机交互预热步数
batch_size: 16
batch_length: 64       # 序列长度
train_ratio: 512
dyn_hidden: 512
dyn_deter: 512         # 确定性状态维度
dyn_stoch: 32          # 随机状态维度
dyn_discrete: 32
units: 512             # MLP 单元数（论文用的是 Large 配置的 768）
imag_horizon: 15       # 想象时域（论文自己设 T = 16）
actor: {layers: 2, lr: 3e-5, entropy: 3e-4, grad_clip: 100.0}
critic: {layers: 2, lr: 3e-5, grad_clip: 100.0, slow_target: True}
kl_free: 1.0
eval_every: 1e4
eval_episode_num: 10
device: 'cuda:0'
precision: 32
```

### 6.2 训练脚本

```bash
# 单卡训练。入口是 dreamer.py，不是 train.py
python3 dreamer.py --logdir ./logdir/drone_racing --seed 0 --steps 2e7 --device cuda:0

# 日志直接看 tensorboard
tensorboard --logdir ./logdir
```

### 6.3 训练循环概览

```python
"""
DreamerV3 训练循环概览（非完整代码）
"""
def train(config):
    env = make_env(config.task)
    world_model = WorldModel(config)
    actor = Actor(config)
    critic = Critic(config)
    buffer = ReplayBuffer()

    # 阶段 1: 预填充随机交互数据
    obs = env.reset()
    for _ in range(config.prefill):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, done)
        obs = next_obs if not done else env.reset()

    # 阶段 2: 主训练循环
    obs = env.reset()
    for step in range(int(config.steps)):
        # 1. 使用 Actor 与仿真环境交互
        state = world_model.obs_to_state(obs)
        action = actor(state).sample()
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, done)
        obs = next_obs if not done else env.reset()

        # 2. 训练世界模型
        if step % config.train_ratio == 0:
            batch = buffer.sample(config.batch_size, config.batch_length)
            world_model_loss = world_model.train_step(batch)

            # 3. 在想象中训练 Actor-Critic
            initial_states = world_model.encode(batch.observations[:, 0])
            imagination = world_model.imagine(initial_states, actor, config.imag_horizon)
            actor_loss, critic_loss = train_actor_critic(imagination, actor, critic, config)

        # 4. 日志与评估
        if step % config.log_every == 0:
            log_metrics(step, world_model_loss, actor_loss, critic_loss)
```

### 6.4 训练资源需求

论文只交代了硬件与总时长，没有按配置分行的显存表：

| 项目 | 论文里的数 |
|---|---|
| 硬件 | 单张 **NVIDIA Quadro RTX 8000**（原文 *"All experiments were performed on a single Quadro RTX 8000"*） |
| 训练时长 | **约 240 小时**收敛（原文 *"requiring approximately 240 hours to converge"*） |
| 输入分辨率 | 64×64 RGB |
| 并行种子 | 三条赛道各 5 个随机种子 |
| 基线规模 | PPO / SAC 各跑到 **2000 万次**环境交互，仍学不出可用机动 |

---

## 7. 评估与可视化

### 7.1 运行评估

```bash
# 评估挂在训练循环里，没有独立的评估脚本
python3 dreamer.py --logdir ./logdir/drone_racing --evaldir ./evaldir/drone_racing \
    --eval_every 1e4 --eval_episode_num 10
```

### 7.2 评估指标

| 指标 | 论文里怎么用 |
|---|---|
| **Episode Reward** | 主结果：三条赛道 × 5 个种子的奖励曲线（论文 Fig. 3） |
| **重建观测** | 世界模型质量：0.4M / 1M / 10M 步时的重建对比真值观测（论文 Fig. 4） |
| **真机-仿真轨迹与速度剖面** | 迁移质量：同一条 Figure 8 赛道上仿真与真机的轨迹、速度剖面与相机指向对比（论文 Fig. 6） |
| **峰值速度** | 真机 HIL 部署达到 *"speeds of up to 9 m/s"* |

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

### 7.4 与基线的真实对比

论文的对比只有一张奖励曲线图（Fig. 3）：三条赛道（Circle、Kidney、Figure 8），每条 **5 个随机种子**，画平均奖励与标准差。结论是 DreamerV3 三条赛道都收敛，PPO 与 SAC **在 2000 万次环境交互内没有任何可观的训练进展**——原文 *"none PPO nor SAC are able to achieve any considerable training in 20 million environment interactions, while DreamerV3 is able to train to convergence for the three tracks"*，并且两者 *"fail to execute any meaningful flight maneuvers, resulting in consistently low reward values"*。两条模型无关基线用的是 CNN + 四层 768 单元的 MLP（actor 与 critic 各一套）。

论文**没有把行为克隆（BC）当基线跑过**，BC 与那套「二值掩码 + 模仿学习引导」的做法只在 Related Work 里作为前人路线出现。

代价要一起看：DreamerV3 在 **10M 步量级**收敛（Fig. 4 的重建对比分别取 0.4M / 1M / 10M 步），训练约 **240 小时**，原文自述这比基线 *"significantly longer"*。样本效率上的优势没有「一个数量级」那么夸张——是 DreamerV3 在 10M 步收敛、基线跑到 20M 步仍学不出东西，**2 倍量级的步数差加基线完全失败**。

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

### Q5: 训练吞吐上不去

瓶颈几乎总在渲染而不是动力学——论文让渲染器在环，整条流水线跑到 *"several thousand frames per second"* 才够用。可查的方向：

```yaml
action_repeat: 2        # 用动作重复减少仿真步数（上游默认就是 2）
size: [64, 64]          # 观测缩放；分辨率是激活与渲染成本的主要杠杆
```

```bash
# 并行环境实例（上游有 parallel 开关）
--parallel True
```

### Q6: 如何将仿真策略迁移到真实无人机？

- 使用 Domain Randomization（在仿真中随机化视觉外观和物理参数）
- 在真实环境中进行微调（少量交互即可）
- 使用 System Identification 校准仿真参数

---

## 动手验证：从一张显存表里解出的 2 GB 是什么？

假设有这么一张三行的显存表。第 1 行和第 3 行只差 batch（50 → 100）：6 GB 和 10 GB。两行相减解出每样本 **0.08 GB**、固定项 **2.00 GB**。第 2 行没有改 batch，只把分辨率从 64 提到 128 —— 按上面这个式子，它应该是 `2.00 + 50 × 0.08 = 6.00 GB`，可表里写的是 10 GB。**多出来的 4 GB 就是分辨率那一项**，也是这张表唯一没被算进去的变量。

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
- DreamerV3 JAX 实现（论文未用）: https://github.com/danijar/dreamerv3
- DreamerV3 PyTorch 实现（论文用的底座）: https://github.com/NM512/dreamerv3-torch
- 同作者的新 PyTorch 版: https://github.com/NM512/r2dreamer
- Flightmare（动力学与赛道生成）: https://github.com/uzh-rpg/flightmare
- Agilicious（平台与物理参数）: https://github.com/uzh-rpg/agilicious

## 延伸阅读

- [什么是世界模型](../01-基础概念/01-什么是世界模型.md) — 理解世界模型的核心概念
- [模型强化学习世界模型](../02-世界模型专题/03-模型强化学习世界模型.md) — Dreamer 系列详解
- [无人机世界模型综述](../02-世界模型专题/05-无人机世界模型综述.md) — 无人机世界模型全景

## 思考题

1. **范式判断**：在无人机竞速这个任务上，论文给出的世界模型路线的优势与代价分别是什么？这些说法各自踩在哪一条证据上？

2. **复现风险**：指南开头那段加粗提示说明复现的哪个前提不成立？它会改变你动手的第一步吗？

3. **排障顺序**：Actor 在想象里表现不错，一上真实（或仿真）环境就失效，文档给了哪几条改法？

4. **算力取舍**：手上一张 10 GB 左右的卡，训练时该先定分辨率还是先定 batch？依据是什么？

5. **指标设计**：论文用什么结果说明 DreamerV3 比 PPO/SAC 强？只看其中一个够不够？

<details><summary>参考答案</summary>

1. 优势只有一条能落到证据上：样本效率。论文的对照是奖励曲线——PPO 与 SAC 在 2000 万次环境交互内学不出可用机动，DreamerV3 三条赛道都收敛（Fig. 3，每条 5 个随机种子）。代价有两条，一是训练时间显著更长（论文自述约 240 小时），二是要额外训练并维护世界模型，随之带来「想象轨迹与真实轨迹不一致」这类故障，Q2 列了它。论文没有对安全性、泛化性做对比评分，也没把行为克隆当基线跑过。

2. 开头那段提示说的是：论文点名了底座（PyTorch 版 `dreamerv3-torch`），但**没有放出自己那套无人机环境的代码**——Flightmare + Habitat 怎么接、赛道怎么生成、奖励怎么实现，都要自己补。所以第一步不是跑训练脚本，而是先把环境接起来并把任务注册成上游的一个 config；在那之前可以拿上游自带的 `dmc_vision` 任务做冒烟测试，确认底座能跑通。可选底座只有 `dreamerv3-torch` 与论文一致：`cleanrl` 里没有本项目可用的实现，`danijar/dreamerv3` 是 JAX 版。

3. Q3 给了三条：把 train_every 从 5 降到 2，提高真实环境交互比例；用 Dyna-style 混合训练，每 N 步用真实数据微调世界模型；给观测加噪声增强鲁棒性。

4. 选卡时不要照搬成表的显存数字。可算的结论来自动手验证：**分辨率是显存的杠杆**——像素数从 64² 到 128² 翻 4 倍，激活实测从 2.41 GB 涨到 9.00 GB（3.7 倍）；batch 则是线性的。10 GB 卡上先定分辨率（论文自己就用 64×64），再按剩余预算加 batch。`action_repeat: 2` 本身也是在减少仿真步数。

5. 这个问题本身要改：论文里根本没有 Completion Rate 这个指标，也没有圈速，全文这两个词零命中。它的主结果是奖励曲线，配合重建图（Fig. 4）看世界模型质量、真机轨迹与速度剖面（Fig. 6）看迁移。真要比，可比的是**同样的交互预算下基线能不能学出东西**：论文给的是 PPO/SAC 跑到 2000 万步仍学不出可用机动，DreamerV3 在 10M 步量级收敛。这才是世界模型路线的卖点，也必须和约 240 小时的训练时间一起看。

</details>
