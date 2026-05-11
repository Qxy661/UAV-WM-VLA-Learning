# 04 - 复现指南：FlightDiffusion — 基于扩散模型的飞行动作生成

> **预计阅读：12 分钟 | 前置知识：扩散模型基本原理（DDPM/DDIM）、PyTorch 深度学习、FPV 无人机飞行概念**

本指南帮助你理解和复现 FlightDiffusion 项目。FlightDiffusion 将去噪扩散模型（Denoising Diffusion Model）应用于无人机飞行动作生成，通过条件扩散过程从 FPV 视频观测生成未来飞行轨迹。

> **注意**：截至本文撰写时，FlightDiffusion 的完整代码可能尚未完全开源。本指南基于论文描述和公开信息编写，部分内容可能需要根据实际代码发布进行调整。

---

## 目录

1. [项目概述](#1-项目概述)
2. [方法原理](#2-方法原理)
3. [环境配置](#3-环境配置)
4. [数据准备](#4-数据准备)
5. [模型架构](#5-模型架构)
6. [训练流程](#6-训练流程)
7. [推理与评估](#7-推理与评估)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 项目概述

- **核心思想**: 将无人机轨迹预测建模为条件扩散过程——给定当前视觉观测和历史轨迹，通过迭代去噪生成未来的飞行动作序列。
- **关键创新**:
  - 将扩散模型的去噪过程与飞行动作空间对齐
  - 支持多模态轨迹生成（同一观测下可以生成多条合理轨迹）
  - 利用 FPV 视频的时序信息提升预测精度

### 与其他方法的对比

| 方法 | 范式 | 确定性/随机性 | 多模态能力 |
|---|---|---|---|
| BC (行为克隆) | 直接回归 | 确定性 | 无 |
| CVAE | 变分推断 | 随机 | 有限 |
| **FlightDiffusion** | 扩散去噪 | 随机 | 强 |

---

## 2. 方法原理

### 2.1 扩散模型基础

扩散模型包含两个过程：

**前向过程（加噪）**：逐步向数据添加高斯噪声

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
```

**反向过程（去噪）**：学习从噪声中恢复数据

```
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_t_bar) * eps_theta(x_t, t, c))
```

### 2.2 FlightDiffusion 的条件扩散

在飞行动作生成场景中：

- **数据 x**: 未来飞行轨迹片段 `a_{t:t+H}`，H 为预测步长
- **条件 c**: 当前视觉观测 `I_t` + 历史轨迹 `a_{t-K:t}`
- **网络 eps_theta**: 条件 U-Net 或 Transformer，预测噪声

```
训练目标：min E[||eps - eps_theta(x_t, t, c)||^2]
```

### 2.3 推理过程

```
1. 采样纯噪声 x_T ~ N(0, I)
2. for t = T, T-1, ..., 1:
3.     预测噪声 eps_theta(x_t, t, c)
4.     执行去噪步骤得到 x_{t-1}
5. 输出 x_0 = 预测的飞行轨迹
```

---

## 3. 环境配置

### 3.1 基础环境

```bash
# 创建环境
conda create -n flightdiff python=3.10 -y
conda activate flightdiff

# 安装 PyTorch
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装扩散模型相关包
pip install diffusers>=0.25.0
pip install transformers
pip install accelerate
```

### 3.2 项目特定依赖

```bash
# 克隆仓库（如果已开源）
git clone https://github.com/<author>/FlightDiffusion.git
cd FlightDiffusion
pip install -r requirements.txt

# 主要依赖可能包括：
# - diffusers: 扩散模型框架
# - einops: 张量操作
# - rotary-embedding-torch: 旋转位置编码
# - ema-pytorch: 指数移动平均
```

### 3.3 数据处理依赖

```bash
pip install av              # 视频解码
pip install opencv-python   # 图像处理
pip install albumentations  # 数据增强
pip install h5py            # 数据存储
```

---

## 4. 数据准备

### 4.1 FPV 飞行视频数据

FlightDiffusion 需要带有轨迹标注的 FPV 飞行视频数据。数据来源可能包括：

- 自采集的 FPV 飞行数据（通过仿真或真实飞行）
- 公开的 FPV 竞速数据集
- AirSim / PX4 仿真生成的数据

### 4.2 数据格式

```text
data/
├── videos/
│   ├── flight_001.mp4       # FPV 视频
│   ├── flight_002.mp4
│   └── ...
├── trajectories/
│   ├── flight_001.hdf5      # 对应轨迹（位姿序列）
│   ├── flight_002.hdf5
│   └── ...
└── metadata.json            # 元数据（场景、天气等）
```

轨迹数据格式（每个时间步）：

```python
{
    "position": [x, y, z],        # 世界坐标系位置（米）
    "orientation": [qw, qx, qy, qz],  # 四元数姿态
    "velocity": [vx, vy, vz],     # 速度（米/秒）
    "angular_velocity": [wx, wy, wz],  # 角速度（弧度/秒）
    "timestamp": 0.033            # 时间戳（秒）
}
```

### 4.3 数据预处理脚本

```bash
# 从视频和轨迹文件生成训练数据
python scripts/preprocess_data.py \
    --video_dir data/videos \
    --traj_dir data/trajectories \
    --output_dir data/processed \
    --context_frames 5 \
    --prediction_horizon 10 \
    --image_size 224
```

预处理后的数据格式：

```text
data/processed/
├── train/
│   ├── samples_000000.hdf5
│   └── ...
└── val/
    ├── samples_000000.hdf5
    └── ...
```

每个样本包含：

- `context_images`: 上下文帧 (5, 3, 224, 224)
- `context_actions`: 历史动作 (5, 4)
- `target_actions`: 未来动作 (10, 4) — 训练目标
- `condition_embedding`: 条件嵌入向量

---

## 5. 模型架构

### 5.1 条件 U-Net 扩散网络

```python
"""
FlightDiffusion 条件 U-Net 架构概览（非完整代码）
"""
import torch
import torch.nn as nn

class ConditionalUNet(nn.Module):
    def __init__(self, action_dim=4, horizon=10, hidden_dim=256):
        super().__init__()
        # 时间步嵌入
        self.time_embed = SinusoidalPosEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # 条件编码器（视觉 + 历史动作）
        self.vision_encoder = VisionEncoder(output_dim=hidden_dim)
        self.action_encoder = nn.Linear(action_dim * 5, hidden_dim)  # 5 帧历史

        # 条件融合
        self.condition_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # U-Net 去噪网络
        self.down_blocks = nn.ModuleList([
            DownBlock(hidden_dim, hidden_dim * 2),
            DownBlock(hidden_dim * 2, hidden_dim * 4),
        ])
        self.mid_block = MidBlock(hidden_dim * 4)
        self.up_blocks = nn.ModuleList([
            UpBlock(hidden_dim * 4, hidden_dim * 2),
            UpBlock(hidden_dim * 2, hidden_dim),
        ])

        # 输出头：预测噪声
        self.output = nn.Linear(hidden_dim, action_dim * horizon)

    def forward(self, x_t, t, context_images, context_actions):
        """
        x_t: (B, horizon, action_dim) 带噪轨迹
        t: (B,) 时间步
        context_images: (B, 5, 3, 224, 224) 上下文帧
        context_actions: (B, 5, action_dim) 历史动作
        """
        # 编码条件
        t_emb = self.time_mlp(self.time_embed(t))
        v_feat = self.vision_encoder(context_images)
        a_feat = self.action_encoder(context_actions.flatten(1))
        condition = self.condition_proj(torch.cat([t_emb, v_feat, a_feat], dim=-1))

        # U-Net 前向
        h = x_t
        for down in self.down_blocks:
            h = down(h, condition)
        h = self.mid_block(h, condition)
        for up in self.up_blocks:
            h = up(h, condition)

        # 预测噪声
        eps_pred = self.output(h)
        return eps_pred.view(-1, 10, 4)
```

### 5.2 视觉编码器

使用预训练的 ResNet-50 或 CLIP 视觉编码器提取特征：

```python
class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, output_dim)
        # 对多帧取平均池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, images):
        # images: (B, 5, 3, 224, 224)
        B, T = images.shape[:2]
        x = images.flatten(0, 1)  # (B*5, 3, 224, 224)
        feat = self.backbone(x)   # (B*5, 256)
        feat = feat.view(B, T, -1)  # (B, 5, 256)
        feat = self.temporal_pool(feat.transpose(1, 2)).squeeze(-1)  # (B, 256)
        return feat
```

---

## 6. 训练流程

### 6.1 训练配置

```yaml
# configs/train_diffusion.yaml
model:
  action_dim: 4
  prediction_horizon: 10
  context_frames: 5
  hidden_dim: 256

diffusion:
  num_timesteps: 1000           # 扩散步数
  beta_schedule: "cosine"       # 噪声调度
  prediction_type: "epsilon"    # 预测噪声还是 x_0

training:
  num_epochs: 200
  batch_size: 64
  learning_rate: 1.0e-4
  ema_decay: 0.9999
  gradient_clip: 1.0
```

### 6.2 启动训练

```bash
python train.py --config configs/train_diffusion.yaml \
    --data_dir data/processed \
    --output_dir checkpoints \
    --wandb_project flightdiffusion
```

### 6.3 训练关键步骤

```python
"""
训练循环概览（非完整代码）
"""
def train_step(model, batch, noise_scheduler):
    context_images = batch["context_images"]
    context_actions = batch["context_actions"]
    target_actions = batch["target_actions"]

    # 1. 采样随机时间步
    t = torch.randint(0, noise_scheduler.num_timesteps, (B,))

    # 2. 向目标轨迹添加噪声
    noise = torch.randn_like(target_actions)
    x_t = noise_scheduler.add_noise(target_actions, noise, t)

    # 3. 模型预测噪声
    noise_pred = model(x_t, t, context_images, context_actions)

    # 4. 计算损失
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 6.4 训练资源需求

| 配置 | GPU 显存 | 训练时间（估计） |
|---|---|---|
| hidden=256, batch=32 | ~10 GB | ~24 小时 |
| hidden=256, batch=64 | ~18 GB | ~12 小时 |
| hidden=512, batch=32 | ~20 GB | ~36 小时 |

---

## 7. 推理与评估

### 7.1 推理

```bash
python inference.py \
    --model_path checkpoints/best_model.pt \
    --video_path test_flight.mp4 \
    --output_path results/predicted_trajectory.json \
    --num_samples 10 \
    --ddim_steps 50
```

推理参数说明：

| 参数 | 说明 |
|---|---|
| `num_samples` | 从不同随机噪声采样生成的轨迹数量 |
| `ddim_steps` | DDIM 加速采样步数（越少越快，质量可能下降） |
| `guidance_scale` | 条件引导强度（类似 Classifier-Free Guidance） |

### 7.2 评估指标

| 指标 | 说明 |
|---|---|
| **Position Error (PE)** | 预测轨迹位置的平均欧氏误差（米） |
| **Orientation Error (OE)** | 预测姿态的角度误差（度） |
| **ADE** | 平均位移误差 |
| **FDE** | 最终点位移误差 |
| **Diversity** | 多条采样轨迹的多样性（米） |
| **Collision Rate** | 仿真中的碰撞率 |

### 7.3 评估脚本

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/processed/val \
    --output_dir results/evaluation \
    --num_samples 20
```

---

## 8. 常见问题与解决方案

### Q1: 训练初期 loss 不下降

扩散模型的训练损失（MSE）在初期可能波动较大，这是正常现象。如果持续不下降：

- 检查数据归一化是否正确（动作值应在合理范围内）
- 确认时间步采样是否均匀
- 尝试使用 `linear` beta schedule 而非 `cosine`

### Q2: 推理速度太慢

```bash
# 方法 1: 减少 DDIM 步数
python inference.py --ddim_steps 20  # 从 1000 降到 20

# 方法 2: 使用 DDIM 而非 DDPM
# 方法 3: 使用 DPM-Solver 加速
```

### Q3: 生成轨迹不平滑

- 在推理后应用简单平滑滤波（如 Savitzky-Golay 滤波）
- 增加扩散步数以提高生成质量
- 检查训练数据中轨迹是否存在突变

### Q4: 多模态生成的轨迹过于相似

- 增大 `guidance_scale` 以增强条件引导
- 使用 Classifier-Free Guidance 训练
- 增加采样数量以探索更多模态

### Q5: 显存不足

```bash
# 使用 gradient checkpointing
# 使用 mixed precision (fp16/bf16)
# 减小 batch_size
# 使用 DeepSpeed ZeRO Stage 2
```

---

## 参考资源

- DDPM 论文: https://arxiv.org/abs/2006.11239
- DDIM 论文: https://arxiv.org/abs/2010.02502
- Diffusers 文档: https://huggingface.co/docs/diffusers
- Diffuser (Planning with Diffusion): https://arxiv.org/abs/2205.09991

## 延伸阅读

- [什么是世界模型](../01-基础概念/01-什么是世界模型.md) — 理解世界模型的核心概念
- [生成式世界模型](../02-世界模型专题/02-生成式世界模型.md) — FlightDiffusion 的理论背景
- [无人机世界模型综述](../02-世界模型专题/05-无人机世界模型综述.md) — 无人机世界模型全景
