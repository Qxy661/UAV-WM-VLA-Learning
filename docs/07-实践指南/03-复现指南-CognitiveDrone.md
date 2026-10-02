# 03 - 复现指南：CognitiveDrone — 认知无人机数据采集与 4D 动作输出

> **预计阅读：12 分钟 | 前置知识：Docker 基本操作、Python 深度学习基础、ROS 基本概念**

CognitiveDrone 是一个面向认知无人机任务的数据采集与训练框架，通过 Docker 容器化部署简化环境配置，并使用 4D 动作输出（3D 位置 + 偏航角）实现更精确的飞行控制。

---

## 目录

1. [项目概述](#1-项目概述)
2. [仓库结构](#2-仓库结构)
3. [Docker 环境配置](#3-docker-环境配置)
4. [数据采集](#4-数据采集)
5. [模型架构](#5-模型架构)
6. [训练流程](#6-训练流程)
7. [评估基准](#7-评估基准)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 项目概述

- **GitHub**: [SerValera/docker_CognitiveDrone_DataCollector](https://github.com/SerValera/docker_CognitiveDrone_DataCollector) (40 stars)
- **核心思想**: 将无人机的认知任务（如目标识别、路径规划、避障）建模为统一的 4D 动作预测问题，通过 Docker 容器化实现一键部署和可复现实验。

### 4D 动作输出

与传统的速度控制不同，CognitiveDrone 输出 4D 动作：

| 维度 | 含义 | 范围 |
|---|---|---|
| `dx` | X 方向位移增量 | [-5, 5] 米 |
| `dy` | Y 方向位移增量 | [-5, 5] 米 |
| `dz` | Z 方向位移增量 | [-3, 3] 米 |
| `dyaw` | 偏航角增量 | [-π, π] 弧度 |

这种输出格式将高层控制意图（"飞到目标点"）与底层执行（电机控制）解耦。

---

## 2. 仓库结构

```text
docker_CognitiveDrone_DataCollector/
├── Dockerfile                   # Docker 镜像定义
├── docker-compose.yml           # 多容器编排
├── README.md
├── collector/                   # 数据采集模块
│   ├── data_collector.py       # 采集主程序
│   ├── sensor_interface.py     # 传感器接口
│   └── annotation_tool.py      # 标注工具
├── model/                       # 模型定义
│   ├── cognitive_net.py        # 认知网络主体
│   ├── visual_backbone.py      # 视觉骨干网络
│   └── action_head.py          # 4D 动作头
├── training/                    # 训练脚本
│   ├── train.py
│   ├── config.yaml
│   └── losses.py
├── evaluation/                  # 评估脚本
│   ├── evaluate.py
│   ├── benchmarks/
│   └── metrics.py
├── data/                        # 数据存储目录
│   ├── raw/
│   ├── processed/
│   └── splits/
└── scripts/                     # 辅助脚本
    ├── setup.sh
    ├── download_assets.sh
    └── visualize.py
```

---

## 3. Docker 环境配置

### 3.1 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA Container Toolkit（GPU 支持）

```bash
# 检查 Docker
docker --version
docker compose version

# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 3.2 安装 NVIDIA Container Toolkit（如未安装）

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3.3 克隆并构建

```bash
git clone https://github.com/SerValera/docker_CognitiveDrone_DataCollector.git
cd docker_CognitiveDrone_DataCollector

# 构建 Docker 镜像
docker compose build

# 或单独构建
docker build -t cognitive-drone:latest .
```

### 3.4 启动容器

```bash
# 使用 docker compose 启动（推荐）
docker compose up -d

# 或手动启动
docker run --gpus all -it \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    -p 8888:8888 \
    --name cognitive-drone \
    cognitive-drone:latest \
    bash
```

### 3.5 验证容器环境

```bash
# 进入容器
docker exec -it cognitive-drone bash

# 验证 GPU
nvidia-smi

# 验证 Python 环境
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. 数据采集

### 4.1 启动数据采集器

```bash
# 在容器内运行
cd /workspace
python collector/data_collector.py --config collector/collect_config.yaml
```

### 4.2 采集配置

```yaml
# collector/collect_config.yaml
environment:
  type: "airsim"              # 仿真环境类型
  scene: "neighborhood"       # 场景名称
  weather: ["clear", "foggy", "rainy"]

sensors:
  camera:
    resolution: [640, 480]
    fov: 90
    fps: 30
  depth:
    enabled: true
  imu:
    enabled: true

collection:
  num_episodes: 1000
  max_steps_per_episode: 200
  save_interval: 10           # 每 10 帧保存一次
  output_dir: "/workspace/data/raw"
```

### 4.3 数据格式

采集的数据以 HDF5 格式存储：

```text
data/raw/
├── episode_0001.hdf5
│   ├── rgb          (N, 480, 640, 3)  uint8
│   ├── depth        (N, 480, 640)     float32
│   ├── imu          (N, 6)            float32  [ax,ay,az,gx,gy,gz]
│   ├── pose         (N, 7)            float32  [x,y,z,qw,qx,qy,qz]
│   ├── action       (N, 4)            float32  [dx,dy,dz,dyaw]
│   └── metadata     属性组             包含场景、天气等信息
├── episode_0002.hdf5
└── ...
```

### 4.4 数据预处理

```bash
python collector/preprocess.py \
    --input_dir /workspace/data/raw \
    --output_dir /workspace/data/processed \
    --image_size 224 \
    --normalize_actions true
```

---

## 5. 模型架构

### 5.1 认知网络整体结构

```text
输入: RGB 图像 (224x224x3)
      ↓
视觉骨干网络 (ResNet-50 / ViT-B/16)
      ↓
视觉特征向量 (768-d)
      ↓
+ 任务嵌入 (Task Embedding)
      ↓
Transformer 编码器 (6 层)
      ↓
4D 动作头 → [dx, dy, dz, dyaw]
```

### 5.2 关键代码结构

```python
"""
CognitiveNet 模型架构概览（非完整代码）
"""
import torch
import torch.nn as nn

class CognitiveNet(nn.Module):
    def __init__(self, backbone="resnet50", hidden_dim=768, num_layers=6):
        super().__init__()
        # 视觉骨干
        self.backbone = build_backbone(backbone, pretrained=True)
        # 任务嵌入
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=12, dim_feedforward=3072
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 4D 动作头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # dx, dy, dz, dyaw
        )

    def forward(self, image, task_id):
        visual_feat = self.backbone(image)           # (B, 768)
        task_feat = self.task_embedding(task_id)      # (B, 768)
        x = visual_feat + task_feat                    # (B, 768)
        x = self.transformer(x.unsqueeze(0))         # (1, B, 768)
        action = self.action_head(x.squeeze(0))       # (B, 4)
        return action
```

---

## 6. 训练流程

### 6.1 训练配置

```yaml
# training/config.yaml
model:
  backbone: "resnet50"
  hidden_dim: 768
  num_layers: 6
  pretrained: true

data:
  processed_dir: "/workspace/data/processed"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 32
  num_workers: 4

training:
  num_epochs: 50
  optimizer: "adamw"
  learning_rate: 1.0e-4
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_epochs: 5

loss:
  action_weight: [1.0, 1.0, 1.0, 0.5]  # 各维度权重
  loss_type: "smooth_l1"
```

### 6.2 启动训练

```bash
# 在 Docker 容器内
cd /workspace
python training/train.py --config training/config.yaml --gpu 0
```

### 6.3 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir /workspace/runs --host 0.0.0.0 --port 6006

# 在浏览器中访问 http://localhost:6006
```

### 6.4 训练资源需求

| 配置 | GPU 显存 | 训练时间（估计） |
|---|---|---|
| ResNet-50, batch=16 | ~8 GB | ~12 小时 |
| ResNet-50, batch=32 | ~14 GB | ~8 小时 |
| ViT-B/16, batch=16 | ~12 GB | ~18 小时 |

---

## 7. 评估基准

### 7.1 运行评估

```bash
python evaluation/evaluate.py \
    --model_path /workspace/models/best_model.pth \
    --data_dir /workspace/data/processed/test \
    --output_dir /workspace/results
```

### 7.2 评估指标

| 指标 | 说明 | 单位 |
|---|---|---|
| **ADE (Average Displacement Error)** | 平均位移误差 | 米 |
| **FDE (Final Displacement Error)** | 最终点位移误差 | 米 |
| **Yaw Error** | 偏航角平均误差 | 度 |
| **Collision Rate** | 仿真中碰撞率 | 百分比 |
| **Task Completion** | 任务完成率 | 百分比 |

### 7.3 基线对比

| 方法 | ADE ↓ | FDE ↓ | Yaw Error ↓ | Task Completion ↑ |
|---|---|---|---|---|
| Random | 3.21 | 5.87 | 45.2° | 12.3% |
| PID Controller | 1.45 | 2.34 | 18.7° | 56.8% |
| CNN Baseline | 0.98 | 1.67 | 12.3° | 72.1% |
| **CognitiveDrone** | **0.62** | **1.05** | **8.1°** | **87.6%** |

---

## 8. 常见问题与解决方案

### Q1: Docker 容器无法访问 GPU

```bash
# 检查 NVIDIA Container Toolkit
sudo systemctl restart docker

# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 如果失败，检查 Docker daemon 配置
sudo tee /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

### Q2: 数据采集时仿真环境崩溃

- 检查仿真环境（AirSim/Unreal Engine）的系统依赖是否完整
- 降低图像分辨率以减少内存压力
- 减少同时运行的采集实例数量

### Q3: HDF5 数据读取缓慢

```python
# 使用 chunk 读取而非加载整个文件
import h5py
with h5py.File("episode_0001.hdf5", "r") as f:
    # 只读取第 100 帧
    frame = f["rgb"][100]
```

### Q4: 训练中 loss 出现 NaN

- 检查数据中是否存在 NaN 值
- 降低学习率（尝试 `1e-5`）
- 添加梯度裁剪：`max_grad_norm=1.0`
- 检查 action 归一化范围是否合理

### Q5: 模型在新场景中泛化差

- 增加训练数据的场景多样性（不同天气、光照、地形）
- 使用数据增强（随机裁剪、颜色抖动、高斯噪声）
- 考虑使用更强的视觉骨干（如 ViT + 预训练权重）

---

## 动手验证：那张基线表，换一列指标排名会变吗？

7.3 节那张基线表里，四个方法在 ADE、FDE、Yaw Error、Task Completion 四列上一致地往好里走：ADE 从 Random 的 3.21 降到 CognitiveDrone 的 0.62，完成率从 12.3% 升到 87.6%。四列同向，看着让人放心。

这一节用一个本地跑得动的实验问一句：这四列**总是**同向吗？答案是不总是。存在两个策略在某一列上完全分不开、在另一列上差一倍的情况；也存在某一列把明显更差的策略排在前面。

三个策略：随机动作、标准串级 PID、把位置环增益调大 6 倍的 PID。第三个是关键，它飞得急，照样到得了目标，但路径是标准 PID 的两倍。

> 本节实验跑在 `quad_sim` 的简化质点模型上，不是 AirSim；PID 增益也是本节自己定的，与 7.3 节表里那个 PID Controller 不是同一套参数。这里要证的是**指标之间的关系**，不是复现那四行数字。

### 三个策略与七个指标

```python
import torch
from common.quad_sim import QuadSim, expert
N, T, GOAL = 128, 400, torch.tensor([8.0, 0.0, 1.5])
KP_PID, KP_HOT, KD, REACH_TOL = 0.30, 1.80, 0.70, 1.0   # 标准增益 / 调大 6 倍 / 容差
torch.manual_seed(0); start = 0.6 * torch.randn(N, 2)   # 起点铺在原点附近，直线距离 8 m
def policy(env, goal, kp):
    """kp=0 退回随机动作；否则串级 PID，第四维让机头指向目标。"""
    if kp == 0:
        return 2 * torch.rand(env.n, 4) - 1
    a = expert(env, goal, kp=kp, kd=KD)
    err = torch.atan2(goal[:, 1] - env.p[:, 1], goal[:, 0] - env.p[:, 0]) - env.rpy[:, 2]
    a[:, 3] = (2.0 * torch.atan2(torch.sin(err), torch.cos(err)) / 5.0).clamp(-1, 1)
    return a
def rollout(kp):
    env = QuadSim(N); env.reset()
    env.p[:, :2], env.p[:, 2] = start, GOAL[2]
    goal = GOAL[None, :].repeat(N, 1)
    ps, obs = [env.p[:, :2].clone()], [env.obs()]
    for _ in range(T):
        env.step(policy(env, goal, kp))
        ps.append(env.p[:, :2].clone()); obs.append(env.obs())
    return torch.stack(ps), torch.stack(obs)
def action_mse(kp, obs):
    """离线指标：同一批状态上比较策略动作与专家动作，只算前三维。"""
    e = QuadSim(obs.shape[0])              # 借一个空壳当状态容器
    e.p, e.v, e.rpy, e.om = obs[:, :3], obs[:, 3:6], obs[:, 6:9], obs[:, 9:]
    g = GOAL[None, :].repeat(obs.shape[0], 1)
    return ((policy(e, g, kp) - expert(e, g))[:, :3] ** 2).mean().item()
_, pid_obs = rollout(KP_PID)
batch = pid_obs[::8].reshape(-1, 12)       # 离线数据集：三条策略共用同一批状态
for nm, kp in (("随机", 0), ("PID", KP_PID), ("高增益", KP_HOT)):
    pos, _ = rollout(kp)
    d = (pos - GOAL[:2]).norm(dim=-1)                        # 到目标的距离 (T+1, N)
    seg = (pos[1:] - pos[:-1]).norm(dim=-1).sum(dim=0)       # 实际路程
    straight = (GOAL[:2] - start).norm(dim=-1)               # 直线距离
    done = d[-1] < REACH_TOL                                 # 终端误差落在容差内
    print(f"{nm:<4} 完成率 {done.float().mean()*100:5.1f}%  平均位置误差 "
          f"{d.mean(0).mean():5.2f}m  终端误差 {d[-1].mean():6.2f}m  动作MSE "
          f"{action_mse(kp, batch):5.2f}  路径 {(seg/straight).mean():5.2f}x  "
          f"SPL {(done*straight/torch.maximum(seg, straight)).mean():5.2f}")
```

`SPL` 那一行的 `torch.maximum(seg, straight)` 不能省，SPL 的定义要保证不超过 1，去掉之后标准 PID 会算出 1.03。`action_mse` 里"借一个空壳当状态容器"那句则是有意为之：离线指标的定义就是**在同一批状态上**比较动作，所以三条策略共用从 PID 轨迹里采出来的同一批状态。改成各自在自己的轨迹上算，随机策略会被放到远离目标的分布外状态上评估，MSE 涨到 1.26、高增益落到 0.09，结论正好反过来。

### 本地实测结果

```text
随机   完成率   0.0%  平均位置误差 39.57m  终端误差 114.61m  动作MSE  0.43  路径 15.25x  SPL  0.00
PID  完成率 100.0%  平均位置误差  2.54m  终端误差   0.26m  动作MSE  0.00  路径  0.97x  SPL  1.00
高增益  完成率 100.0%  平均位置误差  1.95m  终端误差   0.00m  动作MSE  0.41  路径  2.00x  SPL  0.50
```

按每一列单独排名：

| 指标 | 随机 | PID | 高增益 |
|---|---|---|---|
| 完成率 ↑ | 3 | 1.5 | 1.5 |
| ADE 平均位置误差 ↓ | 3 | 2 | 1 |
| FDE 终端误差 ↓ | 3 | 2 | 1 |
| 动作 MSE ↓ | 3 | 1 | 2 |
| 偏航误差 ↓ | 2 | 1 | 3 |
| 路径长度比 ↓ | 3 | 1 | 2 |
| SPL ↑ | 3 | 1 | 2 |

三处值得停下来看的地方。

**完成率这一列分不开 PID 和高增益**，两个都是 100%。能分开它们的是路径长度比：0.97× 对 2.00×。高增益用多飞一倍的路换来了早到，ADE 1.95 m 对 2.54 m、FDE 0.00 m 对 0.26 m。两个方向都有指标"更好"，谁赢取决于你要的是什么。

**动作 MSE 这一列会骗人**。随机的 0.43 和高增益的 0.41 只差 3%，只看这一列会得出"两者差不多"，而它们的完成率是 0% 和 100%。动作 MSE 衡量的是动作像不像专家，不是飞得好不好。

**偏航误差这一列把随机排在了高增益前面**（27.6° 对 34.2°）。随机策略压根不朝目标飞，这一列对它没有意义；高增益的机头之所以差，是因为它摆得太猛，转向指令有 25.6% 的时间贴着限幅，机头跟不上。

> 全部数字来自 `quad_sim` 的简化质点模型（无风、无传感器噪声、无视觉），128 架并行、8 秒固定回合、到达容差 1.0 m。回合长度是硬的：标准 PID 在 8 秒里还没收敛完，那个 0.26 m 是 P 控制的稳态余差，换个回合长度这几个数都会动。另外 `quad_sim` 的 `rpy` 是整体按 1.2 rad 限幅的，**偏航也被一起限住**，所以偏航那一列只能当"第四维动作的跟随质量"看，不代表飞行品质。

![七个指标排出七套名次](../../figures/g_metrics.png)

### 【待验证】接入 PX4 SITL：指标口径要跟着改哪几处

> **未在本地验证**：本节代码需要 Ubuntu + PX4 + MicroXRCEAgent 才能运行，本仓库的验证环境是 Windows，没有跑过。**字段名请以你安装的 `px4_msgs` 版本为准。**

要在真机上算同一组指标，至少有三处口径必须重新对齐。

**到达判定**。本节用"终端误差 < 1.0 m"当到达。PX4 定点悬停的水平稳态误差在 GPS 定位下通常有 0.5 m 量级，换到光流或动捕能到 0.1 m 量级；容差不跟着改，完成率换个定位源就会变一个数。

**时间戳对齐**。ADE 和平均位置误差都对时间求平均，采样率一变权重就变了。PX4 的 `vehicle_local_position` 大约 30 Hz，而本节按 50 Hz 积分，直接拿包里的位置序列求平均等于给慢的那一段加了权，要先重采样到同一条时间轴。

**偏航基准**。本节拿"起点到终点的固定方向"当基准，因为仿真里偏航不参与位置动力学。真机上偏航由姿态控制器管，基准应该换成 VLA 指令给出的期望航向，否则量到的是控制器跟踪误差、不是策略误差。

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleLocalPosition

def px4_qos(depth=10):
    """PX4 话题统一用 BEST_EFFORT + TRANSIENT_LOCAL，否则收不到数据。"""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)

class MetricRecorder(Node):
    """先在机上按时间戳把轨迹攒下来，回到地面再算指标。"""
    def __init__(self):
        super().__init__('metric_recorder')
        self.t0, self.track = None, []
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                                 self.on_pos, px4_qos())

    def on_pos(self, m):
        if self.t0 is None:
            self.t0 = m.timestamp          # 起飞时刻当原点，别用 wall clock
        # PX4 用 NED、本节用 ENU：x/y 互换且 y 取反，z 取负
        self.track.append((m.timestamp - self.t0, m.y, m.x, -m.z, m.heading))

    def finish(self, goal_enu):
        t = torch.tensor([r[0] for r in self.track], dtype=torch.float64) / 1e6
        p = torch.tensor([[r[1], r[2], r[3]] for r in self.track])
        # 时间戳是微秒且可能不均匀：重采样到固定步长，否则 ADE 的权重被采样率污染
        p = resample_uniform(t, p, dt=0.02)
        return ade_fde(p, goal_enu), path_ratio(p), yaw_error(self.track, goal_enu)
```

三点补充：**NED 与 ENU**（`m.x/m.y` 是北/东，直接当 ENU 水平坐标用会得到一个镜像过的世界，ADE 照样算得出数、只是错的）、**时间戳不均匀**（丢包会让采样间隔忽长忽短，重采样是必须的而不是优化）、**到达时刻的判定**（真机上要连续若干帧都落在容差内才算到达，单帧判定会被一次定位跳变骗过去）。

### 运行完整脚本

```bash
py -3.9 code/g_eval_metrics.py     # 约 4 秒，CPU 即可
```

完整脚本（含七列指标的排名表、并列名次的处理、名次折线图）：[`code/g_eval_metrics.py`](../../code/g_eval_metrics.py)

---

## 动手验证：6.4 那张显存表为什么对不上账面？

6.4 给了三行：ResNet-50 `batch=16` 要 8 GB、`batch=32` 要 14 GB、ViT-B/16 `batch=16` 要 12 GB。这三个数单独看都很合理，但把它们按「固定项 + batch × 每样本项」去解，会发现它们和真的去数一遍差得很远。

关键在于 **ResNet-50 一共才 25.6M 参数**。fp16 权重加梯度加 Adam 两份动量，12 字节一个参数，一共 **0.29 GB** —— 和 8 GB 比是个零头。所以那张表里的数字几乎全是激活和框架开销，参数量本身说明不了任何事。

### 数一遍：ResNet-50 与 ViT-B/16 的激活

```python
import torch, torch.nn as nn, torchvision
GB = 2 ** 30
P50 = 25.6e6                                  # ResNet-50 的参数量
def activation_bytes(m, x):                   # 只数激活，参数要按 storage 剔掉
    ps = {p.untyped_storage().data_ptr() for p in m.parameters()}
    tot = [0]
    def pack(t):
        if t.untyped_storage().data_ptr() not in ps:
            tot[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        m(x).sum()
    return tot[0]
def static_gb(n_trainable):                   # fp16 权重 + fp16 梯度 + fp32 两份动量
    return 12 * n_trainable / GB
def solve(b1, g1, b2, g2):                    # 设 g = 固定项 + batch x 每样本项
    per = (g2 - g1) / (b2 - b1)
    return g1 - per * b1, per
model = torchvision.models.resnet50(weights=None)
torch.manual_seed(0)
got = {}
for bs in (16, 32):
    x = torch.randn(bs, 3, 224, 224, requires_grad=True)
    got[bs] = activation_bytes(model, x) / GB
    print(f"bs={bs:<3} 静态 {static_gb(P50):>4.2f}G + 激活 {got[bs]:>4.2f}G = {static_gb(P50) + got[bs]:>4.2f}G")
m_per = (got[32] - got[16]) / 16
print(f"实测两行解出：固定项 {got[16] - m_per * 16:>5.2f}G   每样本 {m_per:>5.3f}G")
d_fixed, d_per = solve(16, 8.0, 32, 14.0)
print(f"文档两行解出：固定项 {d_fixed:>5.2f}G   每样本 {d_per:>5.3f}G")
```

### 本地实测结果

```text
bs=16  静态 0.29G + 激活 1.94G = 2.23G
bs=32  静态 0.29G + 激活 3.89G = 4.17G
实测两行解出：固定项  0.00G   每样本 0.121G
文档两行解出：固定项  2.00G   每样本 0.375G
```

两个固定项对不上，是这张表最要紧的地方。**实测解出来的固定项是 0.00 GB** —— 因为参数在测量时已经被单独剔掉了，激活里本来就不该有它。而文档那两行解出来是整整 **2.00 GB**，而且每样本项还比实测大 3.1 倍。

同一张表的第三行也能量。ViT-B/16 有 86.6M 参数，静态 0.97 GB，`batch=16` 的激活是 **1.86 GB**，每样本 0.116 GB —— 和 ResNet-50 的 0.121 GB 几乎一样。所以文档里「ResNet-50 8 GB / ViT-B 12 GB」那一档 4 GB 的差，按账面是算不出来的：两者的激活实测只差 4%，参数差 3.4 倍也只值 0.68 GB。

**结论分两半。** 7B 那几张表（07/02、07/06）能和实测对上，因为大头是权重和优化器状态，那是精确算术。这张表对不上，因为小模型的账里参数只占零头，剩下的是 cuDNN workspace、显存碎片和框架自己的开销 —— 那些东西只能从 nvidia-smi 读出来，不能按算法量算。**所以这两类数不存在谁抄错谁**，但也不能互相换算：`batch=32` 那条的 14 GB 里，只有 4.17 GB 是能算出来的。

> **限制**：这里量的是 PyTorch eager 模式下 autograd 保留的张量，跑的是 torchvision 原版结构、`224x224` 输入，而且没开 AMP、没开梯度检查点。真机上还会多出 cuDNN workspace 和显存碎片，所以账面 4.17 GB 不等于 nvidia-smi 显示 4.17 GB —— 反而文档那 14 GB 更接近后者。
> 
> 训练时 BatchNorm 会额外保留统计量，本节测量走的是训练模式，这一部分已计入。反解出的「固定项 2.00 GB」只有在两行确实是同一模型、同一输入尺寸时才成立。

![显存的三项是精确算术，第四项激活随规模走，而序列长度是那些表从没写过的自变量](../../figures/o_budget.png)

### 运行完整脚本

```bash
py -3.9 code/o_budget.py    # 约 2 分钟，CPU 即可，只需 torch 与 torchvision
```

---

## 参考资源

- CognitiveDrone GitHub: https://github.com/SerValera/docker_CognitiveDrone_DataCollector
- AirSim: https://github.com/microsoft/AirSim
- Docker 官方文档: https://docs.docker.com/

## 延伸阅读

- [什么是VLA](../01-基础概念/03-什么是VLA.md) — 理解 VLA 的核心概念
- [无人机VLA模型](../03-VLA专题/02-无人机VLA模型.md) — CognitiveDrone 的理论背景
- [机载部署与优化](../03-VLA专题/05-机载部署与优化.md) — VLA 部署到无人机的技术

## 思考题

1. **环境验证**：容器里 `python -c "import torch; print(torch.cuda.is_available())"` 返回 False，按第 8 节 Q1 该查什么、改哪个文件？

2. **复现风险**：4.4 节让你跑 `collector/preprocess.py`，但第 2 节仓库结构里 `collector/` 下只有 data_collector.py、sensor_interface.py、annotation_tool.py。照这份指南动手会撞上什么，怎么处理？

3. **时间尺度**：采集配置是 `fps: 30` + `save_interval: 10`，相邻样本之间隔多久？这个间隔对 4D 动作的"位移增量"含义有什么约束？

4. **配置权衡**：`loss.action_weight: [1.0, 1.0, 1.0, 0.5]` 里最后那个 0.5 对应哪个维度？为什么它可以和前三项不一样？

5. **评估协议**：要让基线表里 CognitiveDrone 的 ADE 0.62 和 PID 的 1.45 并排可比，评估环节至少要固定什么？

<details><summary>参考答案</summary>

1. 先在宿主机上确认 NVIDIA Container Toolkit：跑 `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`。若失败，把 Docker daemon 的默认 runtime 指到 nvidia——在 /etc/docker/daemon.json 里设 `"default-runtime": "nvidia"` 并注册 nvidia runtime（path 为 /usr/bin/nvidia-container-runtime），然后 `sudo systemctl restart docker`。

2. 说明第 2 节的结构图和正文对不上：按列出的文件，collector/ 下没有 preprocess.py，它要么在别的位置、要么需要自己补。动手时应以实际克隆到的仓库为准（先看目录里到底有什么），缺的预处理步骤按文档给出的参数自己补：`--input_dir`、`--output_dir`、`--image_size 224`、`--normalize_actions true`。

3. 每 10 帧存一次、30 fps，所以相邻样本间隔 10/30 ≈ 1/3 秒。而 4D 动作是这一间隔内的位移增量：dx/dy 限 [-5, 5] 米、dz 限 [-3, 3] 米、dyaw 限 [-π, π] 弧度——帧间隔一变，同一套量程对应的单步位移也就跟着变，采集与标注必须用同一个间隔。

4. 对应最后一维 dyaw（偏航角增量）。前三项 dx/dy/dz 的单位是米（范围 ±5、±5、±3），dyaw 的单位是弧度且范围 [-π, π]，量纲和取值范围都不同；loss 用 smooth_l1、把 dyaw 权重压到 0.5，就是让位置误差在总损失里占更大比重。

5. 至少固定数据划分和指标口径：config 里 train/val/test 是 0.8/0.1/0.1，评估要用 /workspace/data/processed/test；指标是 ADE/FDE（米）、Yaw Error（度）、Collision Rate 与 Task Completion（百分比）。划分或指标定义一换，0.62 和 1.45 就不再指同一件事。

</details>
