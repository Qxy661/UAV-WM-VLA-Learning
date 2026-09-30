# 04 - 复现指南：FlightDiffusion — 基于扩散模型的飞行动作生成

> **预计阅读：12 分钟 | 前置知识：扩散模型基本原理（DDPM/DDIM）、PyTorch 深度学习、FPV 无人机飞行概念**

本指南帮助你理解和复现 FlightDiffusion 项目。FlightDiffusion 将去噪扩散模型（Denoising Diffusion Model）应用于无人机飞行动作生成，通过条件扩散过程从 FPV 视频观测生成未来飞行轨迹。

> **注意**：截至本文撰写时，FlightDiffusion 的完整代码可能尚未完全开源。本指南基于论文描述和公开信息编写，部分内容可能需要根据实际代码发布进行调整。

> **更正说明（2026-09 核对该论文原文后补）**：本指南正文第 2、5 节给出的是一套**通用的条件扩散策略**实现——条件 U-Net 直接对动作序列去噪——**这不是 FlightDiffusion 论文的方法**。论文原文用扩散模型生成的是**视频**，动作空间是从生成的视频里反解出来的，详见文末 [第 9 节](#9-论文原文的做法视频生成而非动作生成)。正文保持原样，作为"扩散策略怎么写"的入门实现仍然成立，但**不要把它当成该论文的复现**。

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
9. [论文原文的做法](#9-论文原文的做法视频生成而非动作生成)

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
# 由当前噪声样本估计 x_0
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t, c)) / sqrt(alpha_bar_t)

# DDIM 确定性更新
x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * eps_theta(x_t, t, c)
```

**注意**：上式中的 `alpha_bar_t = prod(alpha_1 ... alpha_t)` 是**累积**乘积，不是单步的 `alpha_t`。把两者混用是扩散模型实现中最常见的错误之一，会让采样结果完全失真——排错时可优先检查这里。

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
        self.action_dim = action_dim   # 供 forward 里 reshape 使用，避免硬编码
        self.horizon = horizon

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
        return eps_pred.view(-1, self.horizon, self.action_dim)
```

### 5.2 视觉编码器

使用预训练的 ResNet-50 或 CLIP 视觉编码器提取特征：

```python
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        # 新版 torchvision 用 weights= 传预训练权重；
        # 旧的 pretrained=True 已废弃，会在未来版本移除
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
import torch.nn.functional as F

def train_step(model, batch, noise_scheduler):
    context_images = batch["context_images"]
    context_actions = batch["context_actions"]
    target_actions = batch["target_actions"]
    B = target_actions.shape[0]        # 批次大小从数据本身取，不要写成外部变量

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

## 9. 论文原文的做法：视频生成而非动作生成

> 本节依 [arXiv:2509.14082](https://arxiv.org/abs/2509.14082) v2 全文补写，用于校正前面第 1、2、5 节对论文方法的描述。论文作者来自 Skolkovo 的 Intelligent Space Robotics Laboratory。

### 9.1 真实的管线：三阶段，输入是**一帧**

论文的输入是机载相机拍下的**单张 RGB 图像**，不是视频序列：

| 阶段 | 用什么 | 产出 |
|---|---|---|
| ① 视觉推理与任务规划 | **Gemini 2.5 Flash**（Google DeepMind 的多模态 VLM）+ 结构化自然语言提示 | 一段高层指令文本 |
| ② 视频生成 | **Wan 2.2 I2V Fast**（图生视频） | 一段 FPV 视频，单条耗时约 20–60 s |
| ③ 轨迹重建 | **ORB-SLAM3**（单目模式）做视觉里程计 | 3D 轨迹、速度、位姿、加速度 |

重建出的轨迹是一串 SE(3) 位姿，转成**速度指令**下发给飞控；同时这些 state-action 对构成监督数据集，用来训下游策略。长时任务按"推理 → 视频生成 → 路径规划 → 执行"分段推进。

**与正文的关键差别**：扩散模型在这里是**视频生成器**，不是动作去噪器。论文自己的说法是 "video diffusion model generates future frames and their corresponding action spaces"——动作空间是生成视频的**副产品**，由视觉里程计反解得到，而不是在动作空间上做去噪。

### 9.2 论文报告的量化结果

| 指标 | 数值 |
|---|---|
| 位置误差 | 均值 **0.25 m**（RMSE 0.28 m） |
| 朝向误差 | 均值 **0.19 rad**（RMSE 0.24 rad） |
| 长时任务（真机） | **5 分 20 秒**完成，均速约 **1 m/s**；仿真 4 分钟；单条视频生成约 30 s |
| 仿真 vs 真机 | 成功率 M = 0.628 (SD 0.162) vs M = 0.617 (SD 0.177)，双向 ANOVA **F(1,16) = 0.394, p = 0.541**，无显著差异 |

仿真在 Gazebo 的 small city world 里做；每类机动（前飞、隧道穿行、避障、降落）在仿真与真机各执行 20 次。注意最后一行的正确读法是"**没测出差异**"，不是"真机表现更好"——p = 0.541 意味着这个样本量下两者无法区分。

### 9.3 论文自己承认的局限

- 生成帧里会**出现真实环境中不存在的物体**（hallucination），论文称这是扩散生成模型的预期局限
- 与 Vicon 动捕真值比对存在小的朝向偏差 Δθ
- 长时任务会**累积误差**（compounding errors），需要按子任务分段规划
- 单目 ORB-SLAM3 提供轨迹可靠，但完整建图与重定位能力受限

### 9.4 那正文第 2–7 节还有用吗

有用，但用途要改口径：

| 本文正文 | 论文原文 |
|---|---|
| 条件 U-Net 对**动作序列**去噪 | 图生视频模型生成**视频帧** |
| 输入 5 帧上下文 + 历史动作 | 输入**单帧** + 文本提示 |
| 输出 `(B, horizon, action_dim)` 动作块 | 输出视频 → SLAM 反解轨迹 |
| 自己训（有数据就能跑、不依赖外部 API） | 依赖 Gemini / Wan 2.2 等外部大模型 API |

也就是说：正文第 2–7 节是一套**学扩散策略的完整练习**（自己写模型、自己训、自己采样），而论文是**用现成大模型搭系统**（它并不训练自己的扩散模型）。两条路线都值得学，但重点完全不同——想复现论文，功夫花在 VLM 提示工程和 SLAM 接口调通上，而不是训条件 U-Net。

---

## 参考资源

- FlightDiffusion 论文（本指南对应的论文）: https://arxiv.org/abs/2509.14082
- DDPM 论文: https://arxiv.org/abs/2006.11239
- DDIM 论文: https://arxiv.org/abs/2010.02502
- Diffusers 文档: https://huggingface.co/docs/diffusers
- Diffuser (Planning with Diffusion): https://arxiv.org/abs/2205.09991

## 延伸阅读

- [什么是世界模型](../01-基础概念/01-什么是世界模型.md) — 理解世界模型的核心概念
- [生成式世界模型](../02-世界模型专题/02-生成式世界模型.md) — FlightDiffusion 的理论背景
- [无人机世界模型综述](../02-世界模型专题/05-无人机世界模型综述.md) — 无人机世界模型全景

## 思考题

1. **复现起点**：把开头的"注意"和 3.2 节的克隆命令放在一起看，照这份指南动手，最早会在哪一步卡住？

2. **排错优先级**：采样出来的轨迹完全失真，文档明确说应该优先检查哪里？为什么？

3. **形状契约**：把 `preprocess_data.py` 的 `--context_frames` 从 5 改成 8，模型里哪一处必须同步改？为什么？

4. **显存预算**：6.4 节三档配置在 24 GB 的卡上都放得下，该按什么选？真 OOM 时文档又给了哪几条降配手段？

5. **评估口径**：文档只给了 PE/OE/ADE/FDE/Diversity/Collision Rate 六个指标和 evaluate.py，想拿自己的数字去对齐原论文，缺的是什么？

<details><summary>参考答案</summary>

1. 卡在 3.2 节克隆仓库这一步：命令里写的是 `git clone https://github.com/<author>/FlightDiffusion.git`，`<author>` 是占位符，没有真实地址；开头也注明代码"可能尚未完全开源"，本指南基于论文描述和公开信息编写。可行的起点是先按第 3 节把环境搭起来（conda python=3.10、torch 2.1.2/torchvision 0.16.2 cu121、diffusers>=0.25.0、einops/rotary-embedding-torch/ema-pytorch），再按第 4、5 节的格式自己实现数据管线与条件 U-Net。

2. 优先检查 `alpha_bar_t` 是不是被写成了单步的 `alpha_t`。文档说明 `alpha_bar_t = prod(alpha_1 ... alpha_t)` 是累积乘积，把两者混用是扩散模型实现中最常见的错误之一，会让采样结果完全失真；x0_pred 公式和 DDIM 更新式里各有一个 alpha_bar（对应 t 与 t-1），都要用累积量。

3. 要改 `ConditionalUNet` 里的 `self.action_encoder = nn.Linear(action_dim * 5, hidden_dim)`（注释写明"5 帧历史"）。forward 里 `context_actions` 按 (B, 5, action_dim) 传入并 `flatten(1)` 成 action_dim*5 维，帧数改成 8 而不改这层，输入维度就对不上；视觉侧同样写死了 5 帧（context_images 是 (B, 5, 3, 224, 224)，VisionEncoder 按 T 还原后做时序池化）。

4. 三档是 hidden=256/batch=32（~10 GB、~24 小时）、hidden=256/batch=64（~18 GB、~12 小时）、hidden=512/batch=32（~20 GB、~36 小时）；在放得下的前提下 batch=64 那档显存和时间都更划算，hidden=512 只有在判定模型容量不够时才值得多花一倍多时间。真 OOM 时按 Q5：gradient checkpointing、mixed precision（fp16/bf16）、减小 batch_size、DeepSpeed ZeRO Stage 2。

5. 缺的是可比口径。第 9 节给出了原论文报告的数值（位置误差 0.25 m、朝向 0.19 rad、真机长时任务 5 分 20 秒、仿真与真机成功率 F(1,16)=0.394, p=0.541），但这些是**论文那套系统**的端到端成绩，不是某个动作预测模型的指标；本文第 7 节的 PE/OE/ADE/FDE 是另一套口径，两者不能直接比。况且论文没有公开测试集划分和官方评估脚本（数据和代码都待发布/自备：FPV 数据要自采或用公开竞速数据集、AirSim/PX4 仿真生成），也没有与其他方法的对比数字。自己跑 evaluate.py 时用哪个 split（文档示例是 data/processed/val）和多少条采样（--num_samples 20）都要与原论文一致才谈得上对齐。

</details>
