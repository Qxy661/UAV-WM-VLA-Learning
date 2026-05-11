# 02 - 复现指南：UAV-Flow — 基于 VLA 的无人机视觉语言动作控制

> **预计阅读：12 分钟 | 前置知识：Python 深度学习基础、PyTorch 使用经验、Hugging Face Transformers 基本操作**

本指南帮助你从零开始复现 UAV-Flow 项目。UAV-Flow 是由北航 CoLA Lab 提出的基于视觉语言动作（VLA）模型的无人机端到端控制方案，通过微调 OpenVLA 模型实现在自然语言指令引导下的无人机飞行控制。

---

## 目录

1. [项目概述](#1-项目概述)
2. [仓库结构](#2-仓库结构)
3. [环境配置](#3-环境配置)
4. [数据集下载](#4-数据集下载)
5. [模型推理](#5-模型推理)
6. [模型训练/微调](#6-模型训练微调)
7. [仿真评估环境](#7-仿真评估环境)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 项目概述

- **论文标题**: UAV-Flow: Learning Dense Optical Flow for UAVs with a Vision-Language-Action Model
- **GitHub**: [buaa-colalab/UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) (124 stars)
- **Hugging Face 模型**: wangxiangyu0814/UAV-Flow
- **Hugging Face 数据集**: wangxiangyu0814/UAV-Flow
- **核心思想**: 基于 OpenVLA（Open Vision-Language-Action）架构，将视觉观测和自然语言指令作为输入，直接输出无人机的飞行动作（如速度指令），实现端到端的视觉语言动作控制。

### 技术路线

```
视觉观测（RGB图像） + 语言指令（"飞向红色建筑"）
        ↓
   OpenVLA 视觉编码器 + 语言编码器
        ↓
   跨模态融合 Transformer
        ↓
   动作解码器 → 无人机速度指令 (vx, vy, vz, yaw_rate)
```

---

## 2. 仓库结构

```text
UAV-Flow/
├── README.md
├── requirements.txt
├── setup.py
├── configs/                    # 训练配置文件
│   ├── train_openvla.yaml
│   └── eval_unrealzoo.yaml
├── data/                       # 数据加载相关
│   ├── dataset.py
│   └── transforms.py
├── models/                     # 模型定义
│   ├── openvla_uav.py         # OpenVLA-UAV 模型
│   ├── vision_encoder.py      # 视觉编码器
│   └── action_decoder.py      # 动作解码头
├── scripts/                    # 工具脚本
│   ├── download_data.sh       # 数据下载
│   ├── train.py               # 训练入口
│   ├── eval.py                # 评估入口
│   └── inference.py           # 推理脚本
├── envs/                       # 仿真环境
│   └── unrealzoo_gym/         # UnrealZoo Gym 接口
└── utils/                      # 工具函数
    ├── metrics.py             # 评估指标
    └── visualization.py       # 可视化工具
```

---

## 3. 环境配置

### 3.1 克隆仓库

```bash
git clone https://github.com/buaa-colalab/UAV-Flow.git
cd UAV-Flow
```

### 3.2 创建 Python 环境

```bash
# 使用 conda
conda create -n uavflow python=3.10 -y
conda activate uavflow

# 安装 PyTorch（根据你的 CUDA 版本调整）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

### 3.3 关键依赖说明

`requirements.txt` 中主要包含以下包：

```text
transformers>=4.36.0        # Hugging Face Transformers
accelerate>=0.25.0          # 分布式训练
peft>=0.7.0                 # 参数高效微调
datasets                    # 数据集加载
pillow                      # 图像处理
opencv-python               # 视频/图像处理
numpy
pyyaml                      # 配置文件解析
wandb                       # 实验追踪
gymnasium                   # 仿真环境接口
pyquaternion                # 四元数运算
```

### 3.4 安装 OpenVLA 基础模型

UAV-Flow 基于 OpenVLA 架构，需要先确保 OpenVLA 的依赖可用：

```bash
# OpenVLA 的核心依赖通常已在 requirements.txt 中包含
# 如需单独安装：
pip install prismatic-vla
```

---

## 4. 数据集下载

### 4.1 从 Hugging Face 下载

UAV-Flow 数据集托管在 Hugging Face 上，包含训练和评估数据。

```bash
# 方法 1：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download --repo-type dataset wangxiangyu0814/UAV-Flow --local-dir ./data/uavflow_dataset

# 方法 2：使用 Python API
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="wangxiangyu0814/UAV-Flow",
    repo_type="dataset",
    local_dir="./data/uavflow_dataset"
)
```

### 4.2 数据集内容说明

```text
uavflow_dataset/
├── train/
│   ├── images/               # 训练图像（RGB）
│   ├── trajectories/         # 对应的飞行轨迹数据
│   └── instructions.jsonl    # 语言指令文件
├── val/
│   ├── images/
│   ├── trajectories/
│   └── instructions.jsonl
└── test/
    ├── images/
    ├── trajectories/
    └── instructions.jsonl
```

每条数据样本的格式：

```json
{
    "image_path": "train/images/000001.png",
    "instruction": "Fly forward towards the tall building",
    "action": [0.32, 0.01, -0.05, 0.02],
    "action_format": "vx_vy_vz_yawrate"
}
```

其中 `action` 为 `[vx, vy, vz, yaw_rate]`，分别表示前向速度、横向速度、垂直速度和偏航角速度。

### 4.3 下载 OpenVLA 预训练权重

```bash
# 下载 OpenVLA 基础模型
huggingface-cli download openvla/openvla-7b --local-dir ./models/openvla-7b
```

---

## 5. 模型推理

### 5.1 下载 UAV-Flow 微调权重

```bash
huggingface-cli download wangxiangyu0814/UAV-Flow --local-dir ./models/uavflow
```

### 5.2 运行推理脚本

```bash
# 单张图像推理
python scripts/inference.py \
    --model_path ./models/uavflow \
    --image_path ./data/uavflow_dataset/test/images/000001.png \
    --instruction "Fly forward and turn left" \
    --device cuda
```

### 5.3 推理代码逻辑

```python
"""
UAV-Flow 推理逻辑演示（非完整代码，仅展示关键步骤）
"""
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# 加载模型和处理器
model = AutoModelForVision2Seq.from_pretrained(
    "./models/uavflow",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./models/uavflow", trust_remote_code=True)

# 准备输入
image = Image.open("test_image.png")
instruction = "Fly towards the red building"

# 构建 prompt
prompt = f"In: What action should the drone take to {instruction}\nOut:"

# 编码输入
inputs = processor(prompt, image).to(model.device, dtype=torch.float16)

# 生成动作
action = model.predict_action(**inputs, unnorm_key="uavflow")
print(f"Predicted action: {action}")
# 输出: [vx, vy, vz, yaw_rate]
```

---

## 6. 模型训练/微调

### 6.1 训练配置

编辑 `configs/train_openvla.yaml`：

```yaml
# 基础配置
model:
  name: "openvla/openvla-7b"
  dtype: "bfloat16"

# 数据配置
data:
  dataset_path: "./data/uavflow_dataset"
  image_size: 224
  batch_size: 8            # 根据 GPU 显存调整

# 训练超参数
training:
  num_epochs: 10
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 100
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0

# LoRA 配置（参数高效微调）
lora:
  enabled: true
  r: 32
  alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "out_proj"]

# 日志与保存
logging:
  use_wandb: true
  wandb_project: "uavflow"
  save_every: 500
  eval_every: 500
```

### 6.2 启动训练

```bash
# 单 GPU 训练
python scripts/train.py --config configs/train_openvla.yaml

# 多 GPU 训练（使用 accelerate）
accelerate launch --multi_gpu --num_processes 4 scripts/train.py \
    --config configs/train_openvla.yaml
```

### 6.3 训练资源需求

| 配置 | GPU 数量 | 显存占用 | 训练时间（估计） |
|---|---|---|---|
| batch_size=4, LoRA, fp16 | 1x RTX 4090 | ~18 GB | ~24 小时 |
| batch_size=8, LoRA, bf16 | 1x A100 80GB | ~40 GB | ~12 小时 |
| batch_size=8, LoRA, bf16 | 4x A100 80GB | ~40 GB/卡 | ~3 小时 |

---

## 7. 仿真评估环境

### 7.1 UnrealZoo Gym 简介

UAV-Flow 使用 UnrealZoo Gym 作为评估环境。UnrealZoo 是基于 Unreal Engine 的无人机仿真平台，提供 Gymnasium 兼容接口。

### 7.2 安装 UnrealZoo Gym

```bash
# 克隆 UnrealZoo Gym
git clone https://github.com/UnrealZoo/UnrealZoo_Gym.git
cd UnrealZoo_Gym
pip install -e .

# 下载仿真环境可执行文件
# 详见 UnrealZoo_Gym 仓库的 README
```

### 7.3 运行评估

```bash
python scripts/eval.py \
    --model_path ./models/uavflow \
    --env_config configs/eval_unrealzoo.yaml \
    --num_episodes 100 \
    --output_dir ./results
```

### 7.4 评估指标

| 指标 | 说明 |
|---|---|
| **Task Success Rate** | 成功到达目标的百分比 |
| **Position Error** | 最终位置与目标位置的欧氏距离（米） |
| **Path Efficiency** | 实际路径长度 / 最短路径长度 |
| **Collision Rate** | 发生碰撞的百分比 |
| **Action MSE** | 预测动作与专家动作的均方误差 |

---

## 8. 常见问题与解决方案

### Q1: Hugging Face 下载超时

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download wangxiangyu0814/UAV-Flow --local-dir ./data/uavflow_dataset
```

### Q2: GPU 显存不足（OOM）

- 减小 `batch_size`（如从 8 降到 2）
- 启用 LoRA 微调而非全参数微调
- 使用 `gradient_checkpointing`
- 使用 `bf16` 或 `fp16` 混合精度训练

```yaml
# 在配置文件中添加
training:
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

### Q3: OpenVLA 模型加载报错

```bash
# 确保 transformers 版本足够新
pip install transformers>=4.36.0

# 如果报 trust_remote_code 错误，在加载时添加参数
model = AutoModelForVision2Seq.from_pretrained(
    model_path, trust_remote_code=True
)
```

### Q4: UnrealZoo Gym 环境无法启动

- 确保系统已安装 Unreal Engine 依赖的运行时库
- 检查 GPU 驱动版本是否兼容
- WSL2 用户需使用 WSLg 或配置 X11 转发

### Q5: 训练损失不收敛

- 检查学习率是否过大（建议从 `2e-5` 开始）
- 确认数据集加载是否正确（打印几个样本检查）
- 检查 action 归一化是否与训练配置一致
- 确保预训练权重加载正确

---

## 参考资源

- UAV-Flow GitHub: https://github.com/buaa-colalab/UAV-Flow
- OpenVLA 论文: https://arxiv.org/abs/2406.09246
- UnrealZoo Gym: https://github.com/UnrealZoo/UnrealZoo_Gym

## 延伸阅读

- [什么是VLA](../01-基础概念/03-什么是VLA.md) — 理解 VLA 的核心概念
- [语言条件飞行控制](../03-VLA专题/03-语言条件飞行控制.md) — UAV-Flow 的理论背景
- [VLA架构演进](../03-VLA专题/01-VLA架构演进.md) — OpenVLA 在 VLA 发展中的位置
