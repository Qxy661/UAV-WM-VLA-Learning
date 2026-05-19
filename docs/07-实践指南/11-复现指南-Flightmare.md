# 11 - 复现指南：Flightmare — 高保真无人机仿真器

> **预计阅读：12 分钟 | 前置知识：Python 基础、Gymnasium 接口概念、无人机基础**

本指南帮助你安装和使用 Flightmare 仿真器。Flightmare 是苏黎世大学 RPG 实验室开发的高保真四旋翼仿真器，支持 Unity 渲染和 Gymnasium 接口，被大量无人机 RL 论文引用。

---

## 项目背景

| 项目 | 信息 |
|------|------|
| **GitHub** | [uzh-rpg/flightmare](https://github.com/uzh-rpg/flightmare) |
| **Stars** | ~1,356 |
| **开发者** | 苏黎世大学 Robotics & Perception Group (RPG) |
| **核心特点** | Unity 渲染 + Gymnasium 接口 + 高保真物理 |
| **与本项目关系** | Dream to Fly（ICRA 2026）的基础仿真环境 |

---

## 环境要求

- **操作系统**：Ubuntu 20.04/22.04（推荐），Windows 需 WSL2
- **Python**：3.8+
- **GPU**：基础仿真纯 CPU，Unity 渲染需 GPU（推荐 NVIDIA RTX）
- **依赖**：Eigen3, PyBind11, CMake
- **可选**：Unity 2020+（用于高保真渲染）

---

## 安装步骤

```bash
# 1. 克隆仓库（递归，含子模块）
git clone --recursive https://github.com/uzh-rpg/flightmare.git
cd flightmare

# 2. 安装系统依赖
sudo apt-get update
sudo apt-get install -y libeigen3-dev cmake build-essential

# 3. 编译 flightlib
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..

# 4. 安装 Python 绑定
cd flightlib
pip install -e .
cd ..

# 5. 安装 Gymnasium 接口
cd flightenvs
pip install -e .
cd ..
```

---

## 运行方式

### 基础悬停 Demo

```python
import gymnasium as gym
import flightenvs  # 注册 Flightmare 环境

# 创建环境
env = gym.make("Quadrotor-v0", render=True)
obs, info = env.reset()

for step in range(1000):
    # 随机动作（悬停）
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 用 PPO 训练导航策略

```python
from stable_baselines3 import PPO
import gymnasium as gym
import flightenvs

env = gym.make("Quadrotor-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_quadrotor")
```

---

## 核心架构

```
Flightmare 架构
├── flightlib/          # C++ 核心库（物理引擎 + 动力学）
│   ├── Quadrotor       # 四旋翼动力学模型
│   ├── Camera          # 相机模型
│   └── Obstacle        # 障碍物
├── flightenvs/         # Gymnasium 环境封装
│   ├── QuadrotorEnv    # 悬停/导航环境
│   └── RacingEnv       # 竞速环境
└── flightrender/       # Unity 渲染器（可选）
```

---

## Gymnasium 环境说明

| 环境名 | 任务 | 观测空间 | 动作空间 |
|--------|------|---------|---------|
| `Quadrotor-v0` | 悬停 | 位置 + 速度 + 姿态 | 4D 推力 |
| `Quadrotor-v1` | 航点追踪 | + 目标位置 | 4D 推力 |
| `Racing-v0` | 竞速 | + 门位置 | 4D 推力 |

---

## 复现 Checklist

- [ ] 克隆仓库并编译 flightlib
- [ ] 运行基础悬停 demo（无渲染）
- [ ] 安装 Stable-Baselines3，运行 PPO 训练
- [ ] 观察训练曲线和策略效果
- [ ] （可选）启用 Unity 渲染，观察视觉效果
- [ ] （可选）对比不同 RL 算法（PPO vs SAC）

---

## 与其他仿真器的对比

| 维度 | Flightmare | gym-pybullet-drones | AirSim | Isaac Sim |
|------|-----------|---------------------|--------|-----------|
| **渲染** | Unity（高保真） | PyBullet（基础） | Unreal（高保真） | Omniverse（最高） |
| **物理** | 自研 Rust 引擎 | PyBullet | PhysX | PhysX 5 |
| **GPU 需求** | 可选 | 不需要 | 需要 | 需要 RTX |
| **Gymnasium** | ✅ | ✅ | 需封装 | 需封装 |
| **引用量** | 高 | 高 | 最高 | 中 |
| **适合场景** | RL 训练 | RL 入门 | sim-to-real | 工业级仿真 |

---

## 参考资源

- Flightmare GitHub: https://github.com/uzh-rpg/flightmare
- Flightmare 文档: https://flightmare.readthedocs.io
- Dream to Fly 论文: https://arxiv.org/abs/2501.14377
- Gymnasium 文档: https://gymnasium.farama.org

## 延伸阅读

- [什么是世界模型](../01-基础概念/01-什么是世界模型.md) — 理解世界模型基础
- [模型强化学习世界模型](../02-世界模型专题/03-模型强化学习世界模型.md) — Dreamer 系列详解
- [可复现项目候选清单](./08-可复现项目候选清单.md) — 更多可复现项目
