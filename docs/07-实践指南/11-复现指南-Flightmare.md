# 11 - 复现指南：Flightmare — 高保真无人机仿真器

> **预计阅读：12 分钟 | 前置知识：Python 基础、强化学习基本概念、无人机动力学基础**

Flightmare 是苏黎世大学 RPG 实验室开发的模块化四旋翼仿真器（CoRL 2020），由两个**完全解耦**的部分组成：一个基于 Unity 的可配置渲染引擎，和一个自研的 C++ 物理引擎。被大量无人机 RL 论文引用——包括本仓 07 篇讲的 Dream to Fly。

---

## 项目背景

| 项目 | 信息 |
|------|------|
| **GitHub** | [uzh-rpg/flightmare](https://github.com/uzh-rpg/flightmare) |
| **Stars** | ~1,431（2026-10 查） |
| **开发者** | 苏黎世大学 Robotics & Perception Group (RPG) |
| **论文** | "Flightmare: A Flexible Quadrotor Simulator"（CoRL 2020，Spotlight） |
| **核心特点** | Unity 渲染与 C++ 物理引擎完全解耦；可并行仿真数百架四旋翼；带三维点云接口与 VR 头显集成 |
| **与本项目关系** | Dream to Fly（ICRA 2026，arXiv 2501.14377）的基础仿真环境 |

> **勘误（2026-10）**：本节早先写「核心特点：Unity 渲染 + **Gymnasium 接口** + 高保真物理」。**Flightmare 没有 Gymnasium 接口**——它的 RL 栈钉在旧版 Gym 上（`rpg_baselines` 的 `install_requires` 逐字是 `gym==0.11`），环境对象由 pybind11 模块 `flightgym` 直接构造，**不走 `gym.make()` 注册**。它的三个卖点按 README 原文是：*"(i) a large multi-modal sensor suite, including an interface to extract the 3D point-cloud of the scene; (ii) an API for reinforcement learning which can simulate hundreds of quadrotors in parallel; and (iii) an integration with a virtual-reality headset"*。上表已按原文重写。

---

## 环境要求

- **操作系统**：Linux（官方文档与 Dockerfile 都用 Ubuntu；Ubuntu 18.04 是 Dockerfile 的基底，ROS 那条路线注明 Ubuntu 20.04）。**Windows 未在官方文档里出现**——本仓没有实测过，按 WSL2 处理属于推测，不是仓库的说法
- **Python**：**3.6**（`Install-with-pip` 给的命令逐字是 `conda create --name ENVNAME python=3.6`）
- **GPU**：基础仿真纯 CPU；只有 Unity 渲染需要 GPU
- **系统依赖**：`build-essential`、`cmake`、`libzmqpp-dev`、`libopencv-dev`（外加 python3/python3-dev/python3-pip）
- **Python 依赖**：`tensorflow-gpu==1.14`（无 GPU 用 `tensorflow==1.14`，**TF1**）、`scikit-build`；`rpg_baselines` 自带 `gym==0.11` + `stable_baselines==2.10.1`
- **环境变量**：`FLIGHTMARE_PATH` 指向仓库根目录，训练脚本按它去找配置

> **勘误（2026-10）**：本节早先写「依赖：**Eigen3**, PyBind11, CMake」「Python 3.8+」「Unity 2020+」。三处都不对：**系统依赖里没有 Eigen3**，官方列的是 `build-essential / cmake / libzmqpp-dev / libopencv-dev`（Dockerfile 与 wiki 一致）；Python 版本要求是**向下钉到 3.6**，因为整条 RL 栈吃 TF1；**Unity 版本没有被指定**，渲染器不是用 Unity 编辑器打开的工程，而是从 Releases 下载编译好的二进制。另外早先完全没提 `FLIGHTMARE_PATH` 和 TF1 依赖——而这两条是跑不起来的第一、第二原因。

---

## 安装步骤

官方推荐 RL 任务走 pip 路线（`Install with pip, A Quick Start Guide`）。**编译与安装是同一个动作**——`flightlib` 的 `setup.py` 自己调 CMake，不需要手工 `mkdir build && cmake ..`。

```bash
# 0. 系统依赖
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   build-essential cmake libzmqpp-dev libopencv-dev

# 1. Python 环境（官方用的 3.6）
conda create --name ENVNAME python=3.6
conda activate ENVNAME

# 2. 克隆（普通克隆即可，没有子模块）
cd ~/Desktop
git clone https://github.com/uzh-rpg/flightmare.git

# 3. 注册 FLIGHTMARE_PATH（训练脚本靠它读 configs/vec_env.yaml）
echo "export FLIGHTMARE_PATH=~/Desktop/flightmare" >> ~/.bashrc
source ~/.bashrc

# 4. TF1 与构建工具
pip install tensorflow-gpu==1.14    # 无 GPU 改成 tensorflow==1.14
pip install scikit-build

# 5. 编译并安装 flightlib（C++ 核心 + pybind11 绑定，一条命令搞定）
cd flightmare/flightlib
pip install .

# 6. 安装 RL baselines
cd ../flightrl
pip install .
```

Unity 渲染是**独立的一步**，不装也不影响纯物理仿真：从 GitHub **Releases** 下载渲染器二进制，解压进 `flightrender/` 目录，需要可视化时先手动双击里面的可执行文件把渲染器跑起来（wiki 把它写作 `RPG_Flightmare.x84-64`，看名字是 `x86-64` 的笔误）。

> **勘误（2026-10）**：本节早先的六步全是编的：`git clone --recursive`（**没有子模块**，官方就是普通 clone）、`apt-get install libeigen3-dev`（Eigen3 不在依赖里）、手工 `mkdir build && cmake .. && make -j$(nproc)`（**没有这一步**，`pip install .` 内部会编译）、`cd flightlib && pip install -e .`（是 `pip install .`，**不是 `-e` 可编辑安装**）、以及最关键的 `cd flightenvs && pip install -e .`——**`flightenvs` 这个目录根本不存在**，仓库顶层是 `flightlib / flightrl / flightrender / flightros`，RL 侧叫 `flightrl`。整套步骤已按 wiki 的 `Prerequisites` / `Install-with-pip` / `Basic-Usage-with-Python` 三页重写。

---

## 运行方式

### 官方示例：PPO 训练 / 测试

仓库自带的就是一个 PPO 控制器示例，不用自己写训练循环：

```bash
cd flightrl/examples

# 训练（默认 2500 万步，权重存到 examples/saved/quadrotor_env.zip）
python3 run_drone_control.py --train 1

# 测试已有权重
python3 run_drone_control.py --train 0

# 开 Unity 可视化测试（要先手动把 flightrender 里的渲染器跑起来）
python3 run_drone_control.py --train 0 --render 1
```

### 手动搭一个环境

环境对象由 `flightgym` 这个 pybind11 模块直接构造（照 `flightrl/examples/run_drone_control.py` 的写法），配置从 `$FLIGHTMARE_PATH` 下的 YAML 读：

```python
import os
import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper

from rpg_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1

cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                       "/flightlib/configs/vec_env.yaml", 'r'))
cfg["env"]["num_envs"] = 1        # 单环境；训练时设大（默认走多线程并行）
cfg["env"]["num_threads"] = 1
cfg["env"]["render"] = "no"       # "yes" 时连 Unity

env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
# 这是个 VecEnv：观测/奖励都带一维环境编号，num_envs=1 时该维长度就是 1

obs = env.reset()
for _ in range(1000):
    action = np.zeros(env.num_acts, dtype=np.float32)  # 动作空间是 Box(-1, 1)，不是物理推力
    obs, reward, done, info = env.step(action)         # 老 Gym 四元组，没有 truncated

env.close()
```

> **勘误（2026-10）**：本节早先的两段代码都是编的。第一段用 `import flightenvs` + `gym.make("Quadrotor-v0", render=True)`——**`flightenvs` 模块不存在**，也**没有 `gym.make` 注册**这套机制，环境是从 `flightgym` 直接 `QuadrotorEnv_v1(...)` 构造的；`obs, info = env.reset()` 和 `terminated/truncated` 五元组是 Gymnasium 的接口，而 Flightmare 的 `EnvWrapper.step()` 逐字返回 `self.observation, self.reward, self.done, [dict(...)]`——**四元组**（`gym==0.11` 的时代没有 truncated）。第二段用 `from stable_baselines3 import PPO`——**仓库没用 SB3**，它自带一份 PPO2 实现（`rpg_baselines/ppo/ppo2.py`，从 stable-baselines 2.10.1 fork 而来）并依赖 TF1；`model.learn(total_timesteps=100_000)` 的步数也是编的，示例脚本里是 `int(25000000)`。

---

## 核心架构

```
Flightmare
├── flightlib/       # C++ 核心库：物理引擎 + 动力学 + 传感器 + 环境
│                    #   子目录：dynamics / sensors / objects / envs /
│                    #           bridges / common / json
│                    #   经 pybind11 导出为 Python 模块 flightgym
├── flightrl/        # RL baselines（纯 Python）
│   ├── rpg_baselines/{common, envs, ppo}
│   └── examples/run_drone_control.py   # PPO 训练/测试入口
├── flightrender/    # Unity 渲染器（二进制，从 Releases 下载后放进来）
└── flightros/       # ROS 封装（另一条安装路线）
```

渲染引擎与物理引擎**完全解耦**：`flightlib` 单独就能跑纯物理仿真，`flightrender` 只有需要图像时才连上。

> **勘误（2026-10）**：本节早先的架构树里写了 `flightenvs/`（Gymnasium 环境封装，含 `QuadrotorEnv` 与 `RacingEnv`）。**这个目录不存在**，也没有 `RacingEnv`——RL 侧是 `flightrl/`，而且它是一个纯 Python 包（`rpg_baselines`），C++ 环境类住在 `flightlib/src/envs/` 里。早先的树还漏掉了 `flightros/`（ROS 那条安装路线）与 `flightlib/src/bridges/`（连 Unity 的桥）。

---

## 环境与接口说明

Python 侧可用的环境类只有两个，来自 `flightgym` 模块（`flightlib/src/wrapper/pybind_wrapper.cpp` 里逐字导出）：

| 导出名 | 对应 C++ 类 | 观测空间 | 动作空间 |
|---|---|---|---|
| `QuadrotorEnv_v1` | `VecEnv<QuadrotorEnv>` | 12 维：位置(3) + 姿态(3) + 线速度(3) + 角速度(3) | 4 维归一化值，`Box(-1, 1)` |
| `TestEnv_v0` | `TestEnv<QuadrotorEnv>` | 同上（用于跑固定场景的测试） | 同上 |

两个要点容易踩错：

- **不是 `gym.make()`**。环境对象由构造函数直接建，参数是一个 YAML 字符串和一个 bool：`QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)`。包装器是 `rpg_baselines.envs.vec_env_wrapper.FlightEnvVec`。
- **动作不是物理推力**。`quadrotor_env.cpp` 里 `quad_act_ = act.cwiseProduct(act_std_) + act_mean_`，`act_mean_ = -mass*g/4`、`act_std_ = -mass*2g/4`，也就是把 `[-1, 1]` 的四个数映射到四个电机的推力；上层的 `Box(-1, 1)` 只负责把量纲归一化。

真值与场景配置都在 YAML 里（`flightlib/configs/`）：

```yaml
# quadrotor_env.yaml
quadrotor_dynamics:
  mass: 0.73            # kg
  arm_l: 0.17           # m
  motor_omega_max: 3000.0   # rpm
  kappa: 0.016          # 旋翼阻力系数
  omega_max: [6.0, 6.0, 6.0]  # 机体角速度约束
# vec_env.yaml
env:
  scene_id: 0           # 0 warehouse, 1 garage, 3 natureforest
  num_envs: 100         # 默认 100 个环境并行
  num_threads: 10
  render: no
```

> **勘误（2026-10）**：早先这张表列的是 `Quadrotor-v0`（悬停）、`Quadrotor-v1`（航点追踪）、`Racing-v0`（竞速）三个环境名，全是编的——`pybind_wrapper.cpp` 里只导出 `QuadrotorEnv_v1` 与 `TestEnv_v0`，**没有 `Racing-v0`，也没有 `Quadrotor-v0`/`Quadrotor-v1` 这种带连字符的注册名**（`QuadrotorEnv_v1` 末尾的 `_v1` 是版本号，不是环境序号）。「航点追踪 / 竞速」这两种任务划分在仓库里找不到对应实现。观测与动作维度按 `quadrotor_env.hpp` 的 `enum Ctl`（`kNObs = 12`，`kNAct = 4`）与 `quadrotor_env.cpp` 的动作缩放重写。

---

## 复现 Checklist

- [ ] 装系统依赖与 conda 3.6 环境
- [ ] 设好 `FLIGHTMARE_PATH`（漏了会在读 `vec_env.yaml` 时直接报错）
- [ ] 装 `tensorflow==1.14` 与 `scikit-build`（TF2 直接不兼容）
- [ ] `cd flightlib && pip install .`，确认 `from flightgym import QuadrotorEnv_v1` 能导入
- [ ] `cd flightrl && pip install .`
- [ ] 先用 `run_drone_control.py --train 0` 跑预训练权重确认环境通了（比直接开训快得多）
- [ ] 跑 `--train 1` 训练，观察 TensorBoard 曲线
- [ ] （可选）下载 Releases 里的 Unity 渲染器，`--render 1` 看视觉效果

---

## 与其他仿真器的对比

| 维度 | Flightmare | gym-pybullet-drones | AirSim | Isaac Sim |
|------|-----------|---------------------|--------|-----------|
| **渲染** | Unity（与物理解耦） | PyBullet（基础） | Unreal（高保真） | Omniverse |
| **物理** | 自研 C++ 引擎（flightlib） | PyBullet | PhysX | PhysX |
| **GPU 需求** | 只有 Unity 渲染需要 | 不需要 | 需要 | 需要 RTX |
| **Gym 接口** | ❌（钉在 `gym==0.11`，靠 `flightgym` 直接构造） | ✅ gymnasium + SB3 2.0 | 需自行封装 | 需自行封装 |
| **GitHub Stars**（2026-10） | 1,431 | 2,154 | 18,539 | 4,231（`isaac-sim/IsaacSim`） |
| **适合场景** | RL 训练 | RL 入门 | sim-to-real | 工业级仿真 |

> **勘误（2026-10）**：本节早先有一行「**Gymnasium**：Flightmare ✅ / gym-pybullet-drones ✅」。**Flightmare 那一格是错的**——它用的是旧版 Gym，`rpg_baselines` 的依赖逐字钉在 `gym==0.11`，接口是四元组，也没有 `gym.make` 注册。早先还有一行「引用量：高/高/最高/中」，四个都是印象值，已换成 2026-10 实测的 GitHub Stars。

---

## 参考资源

- Flightmare GitHub: https://github.com/uzh-rpg/flightmare
- 安装与使用文档（wiki，本指南的权威出处）: https://github.com/uzh-rpg/flightmare/wiki
- 在线文档: https://flightmare.readthedocs.io
- 项目主页: https://uzh-rpg.github.io/flightmare/
- 论文（CoRL 2020 Spotlight）: [Flightmare: A Flexible Quadrotor Simulator](http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)
- Dream to Fly 论文（ICRA 2026）: https://arxiv.org/abs/2501.14377
- Unity 渲染器二进制: https://github.com/uzh-rpg/flightmare/releases

> **勘误（2026-10）**：本节早先列的「Gymnasium 文档」与本项目已无关（Flightmare 不用 Gymnasium）；同时漏掉了 wiki——而**安装步骤的唯一权威出处就在 wiki 上**，README 只有一句话把读者指过去（*"Installation instructions can be found in our Wiki"*）。

## 延伸阅读

- [什么是世界模型](../01-基础概念/01-什么是世界模型.md) — 理解世界模型基础
- [模型强化学习世界模型](../02-世界模型专题/03-模型强化学习世界模型.md) — Dreamer 系列详解
- [可复现项目候选清单](./08-可复现项目候选清单.md) — 更多可复现项目

## 思考题

1. **选型判断**：要在 Flightmare、gym-pybullet-drones、AirSim、Isaac Sim 里选一个，怎么判断？各自的分水岭在哪？

2. **平台匹配**：在 Windows 上能不能跑？Unity 渲染是必装项吗？

3. **复现卡点**：这套安装流程里，最容易漏掉的是哪几步？

4. **接口对比**：`QuadrotorEnv_v1` 与 `TestEnv_v0` 差在哪？观测和动作各是多少维，动作值是什么物理量？

5. **路线衔接**：已经在 `QuadrotorEnv_v1` 上跑通 PPO，想换成世界模型做竞速，仿真器这一层能直接复用吗？

<details><summary>参考答案</summary>

1. 四者的分水岭是渲染方案、GPU 需求、Gym 接口和编译成本。要 RL 训练又想要高保真视觉，选 Flightmare（Unity 渲染与 C++ 引擎解耦，GPU 只在开渲染时需要，但要自己编译）；只想快速入门 RL、不想碰编译，选 gym-pybullet-drones（PyBullet，不需要 GPU，而且它是四者里唯一明确兼容 **gymnasium + stable-baselines3 2.0** 的）；目标是 sim-to-real，AirSim 社区最大（18.5k stars）但 Gym 接口要自己封装；要工业级仿真且手上有 RTX，选 Isaac Sim。注意 Flightmare 虽然开源最早，但它的 RL 栈停在 TF1 时代——**不要因为"它常被论文引用"就默认它接口现代**。

2. Windows 能不能跑，**官方文档没有回答**——wiki 的安装页只有 Linux 的 apt 命令，Dockerfile 用 Ubuntu 18.04，ROS 那条路线提到 Ubuntu 20.04，全篇没有 Windows/WSL 字样。所以"按 WSL2 处理"是我们自己的判断，不是仓库的承诺，本仓也没实测过。Unity 渲染可以确定是可选项：基础仿真纯 CPU，`flightrender/` 里的渲染器是**单独的二进制**，要专门去 Releases 下载解压，不装也能把 PPO 训完；只有想看视觉效果时才需要它（先把渲染器跑起来，再 `--render 1`）。

3. 三处，且都不是传统意义上的"编译环节"——因为编译已经被 `pip install .` 吞掉了。最容易漏的是**`FLIGHTMARE_PATH`**：`run_drone_control.py` 用它拼 `flightlib/configs/vec_env.yaml` 的路径，没设环境变量会在读配置时直接抛错。其次是**TF1**：依赖是 `tensorflow-gpu==1.14`（无 GPU 用 `tensorflow==1.14`），代码里还有 `tf.set_random_seed` 这种 TF1 专有 API，装成 TF2 会大面积报错。第三是系统依赖里那几个 C++ 库（`build-essential cmake libzmqpp-dev libopencv-dev`），`libzmqpp-dev` 缺了会在连 Unity 的桥上编译失败。另外要注意 `flightlib` 与 `flightrl` 是**两次**独立的 `pip install .`，少装后者就没有 `rpg_baselines`。

4. 两个导出类共用同一个 C++ 环境（`QuadrotorEnv`），差别只在包装方式：`QuadrotorEnv_v1` 是 `VecEnv<QuadrotorEnv>`（向量化、并行用），`TestEnv_v0` 是 `TestEnv<QuadrotorEnv>`（跑固定场景的测试）。观测都是 **12 维**：位置(3) + 姿态(3) + 线速度(3) + 角速度(3)（`enum Ctl` 里 `kNObs = 12`）。动作都是 **4 维**，但不是物理推力——`quadrotor_env.cpp` 里 `quad_act_ = act.cwiseProduct(act_std_) + act_mean_`，`act_mean_ = -mass*g/4`、`act_std_ = -mass*2g/4`，也就是外面的 `Box(-1, 1)` 先被线性映射成四个电机的推力再送进动力学。所以换任务时动作接口不动，要动的是奖励系数（`quadrotor_env.yaml` 里的 `pos_coeff` / `ori_coeff` / `lin_vel_coeff` / `ang_vel_coeff` / `act_coeff`）。

5. **不能直接复用**。Flightmare 提供的是两样东西：四旋翼动力学（`flightlib` 的 dynamics）和赛道生成，所以"仿真器"这一层是有用的；但把这两样接成**竞速任务**的那一整条链路——门位姿、奖励设计、以及把 Habitat 接进来做渲染——**论文没有开源**（见本仓 [07-复现指南：DreamerV3-Drone](./07-复现指南-DreamerV3-Drone.md) 的说明：论文只点名了它用的 DreamerV3 底座，环境代码没放出来）。而且仓库里也**没有现成的竞速环境**：`pybind_wrapper.cpp` 只导出 `QuadrotorEnv_v1` 与 `TestEnv_v0`，两个都是位置/姿态控制任务，不带赛道门。要换的不是"训练算法那一侧"那么轻——环境这一侧得自己写。

</details>

> **勘误（2026-10）**：参考答案 1–5 早先全部建立在编造的环境名（`Quadrotor-v0` / `Quadrotor-v1` / `Racing-v0`）、编造的"自带 Gymnasium 接口"和编造的安装步骤（`--recursive`、Eigen3、手工 cmake、`flightenvs`）之上，因此五条答案一并重写。其中第 5 题的结论**方向反转**了：早先答"不用换"，实际是环境这一层必须自己接。
