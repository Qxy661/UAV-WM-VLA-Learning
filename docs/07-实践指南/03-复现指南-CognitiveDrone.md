# 03 - 复现指南：CognitiveDrone — 认知无人机数据采集与 4D 动作输出

> **预计阅读：12 分钟 | 前置知识：Docker 基本操作、Python 深度学习基础、ROS 基本概念**

本指南帮助你复现 CognitiveDrone 项目。CognitiveDrone 是一个面向认知无人机任务的数据采集与训练框架，通过 Docker 容器化部署简化环境配置，并使用 4D 动作输出（3D 位置 + 偏航角）实现更精确的飞行控制。

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

## 参考资源

- CognitiveDrone GitHub: https://github.com/SerValera/docker_CognitiveDrone_DataCollector
- AirSim: https://github.com/microsoft/AirSim
- Docker 官方文档: https://docs.docker.com/

## 延伸阅读

- [什么是VLA](../01-基础概念/03-什么是VLA.md) — 理解 VLA 的核心概念
- [无人机VLA模型](../03-VLA专题/02-无人机VLA模型.md) — CognitiveDrone 的理论背景
- [机载部署与优化](../03-VLA专题/05-机载部署与优化.md) — VLA 部署到无人机的技术
