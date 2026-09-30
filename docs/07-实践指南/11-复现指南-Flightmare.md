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
| **物理** | 自研 C++ 引擎（flightlib） | PyBullet | PhysX | PhysX 5 |
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

## 思考题

1. **选型判断**：要在 Flightmare、gym-pybullet-drones、AirSim、Isaac Sim 里选一个，怎么判断？各自的分水岭在哪？

2. **平台匹配**：在 Windows 上能不能跑？Unity 渲染是必装项吗？

3. **复现卡点**：克隆与编译环节分别最容易漏掉什么？

4. **接口对比**：Quadrotor-v0、Quadrotor-v1、Racing-v0 三个环境差在哪？动作空间变了吗？

5. **路线衔接**：已经在 Quadrotor-v0 上跑通 PPO，想换成世界模型做竞速，仿真器层要不要一起换？

<details><summary>参考答案</summary>

1. 四者的分水岭是渲染方案、GPU 需求和接口封装。要 RL 训练又想要高保真视觉，选 Flightmare（Unity 渲染 + 自研 C++ 引擎 flightlib，GPU 可选）；只想快速入门 RL、不想碰 Unity 和编译，选 gym-pybullet-drones（PyBullet，不需要 GPU）；目标是 sim-to-real，AirSim 引用量最高但 Gymnasium 需要自己封装；要工业级仿真且手上有 RTX，选 Isaac Sim（Omniverse + PhysX 5）。文档的对比表里，只有 Flightmare 与 gym-pybullet-drones 自带 Gymnasium 接口。

2. Windows 不是不能跑，但要 WSL2——环境要求写的是"Ubuntu 20.04/22.04（推荐），Windows 需 WSL2"。Unity 渲染是可选项：基础仿真纯 CPU，只有 Unity 渲染才需要 GPU（推荐 NVIDIA RTX），Checklist 里"启用 Unity 渲染，观察视觉效果"也标了可选，而 flightrender 在架构图里本身就写着"（可选）"。

3. 两处：克隆必须带 `--recursive`（要一并拉子模块）；编译 flightlib 前要先装系统依赖 `libeigen3-dev cmake build-essential`，缺 Eigen3 会直接编译失败。Python 侧是两步，先在 flightlib 里 `pip install -e .` 装绑定，再进 flightenvs 装 Gymnasium 接口，后者是 `gym.make("Quadrotor-v0")` 能注册的前提（demo 里专门 `import flightenvs` 做注册）。

4. 动作空间三者一样，都是 4D 推力；差别在观测。Quadrotor-v0 是悬停（位置 + 速度 + 姿态），Quadrotor-v1 航点追踪在此基础上加目标位置，Racing-v0 竞速再加门位置。也就是说从悬停换到竞速，策略网络要吃的输入变多了，动作接口不用改。

5. 不用换。文档"与本项目关系"一栏写明 Flightmare 是 Dream to Fly（ICRA 2026）的基础仿真环境，参考资源里对应的就是 arXiv:2501.14377，而竞速任务对应的 Racing-v0 已包含在 flightenvs 的环境表中。要换的是训练算法那一侧，仿真器的编译与 Gymnasium 接口流程不变。

</details>
