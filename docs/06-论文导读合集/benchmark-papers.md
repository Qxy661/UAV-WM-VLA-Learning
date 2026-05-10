# 基准与数据集论文卡片

> 本文收录无人机VLA/VLM/世界模型领域的基准和数据集论文

---

## 1. MotionScape — 大规模无人机视频数据集

**一句话总结**：30+小时4K无人机视频，4.5M+帧，6-DoF轨迹+语言描述

| 项目 | 内容 |
|------|------|
| **标题** | MotionScape: A Large-Scale Real-World Highly Dynamic UAV Video Dataset for World Models |
| **arXiv** | [2604.07991](https://arxiv.org/abs/2604.07991) |
| **GitHub** | [Thelegendzz/MotionScape](https://github.com/Thelegendzz/MotionScape) |
| **年份** | 2026 |
| **推荐度** | ★★★ |

**数据规模**：
- 30+小时4K视频
- 4.5M+帧
- 准确的6-DoF相机轨迹
- 细粒度自然语言描述

**数据管线**：
1. CLIP相关性过滤
2. 时间分割
3. 鲁棒视觉SLAM轨迹恢复
4. LLM驱动语义标注

**关键发现**：对齐的标注有效提升现有世界模型模拟复杂3D动态和处理大视角变化的能力

---

## 2. AeroVerse — 航空航天世界模型基准

**一句话总结**：最全面的无人机世界模型基准套件

| 项目 | 内容 |
|------|------|
| **标题** | AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models |
| **arXiv** | [2408.15511](https://arxiv.org/abs/2408.15511) |
| **机构** | 哈尔滨工业大学 |
| **年份** | 2024 |
| **推荐度** | ★★☆ |

**五大任务**：
1. 场景认知
2. 导航规划
3. 目标跟踪
4. 避障控制
5. 降落评估

**数据内容**：
- 预训练数据集
- 微调数据集
- 评估协议

---

## 3. CARLA-Air — 统一空地仿真平台

**一句话总结**：在CARLA世界中飞无人机，统一空中和地面智能体

| 项目 | 内容 |
|------|------|
| **标题** | CARLA-Air: Fly Drones Inside a CARLA World |
| **arXiv** | [2603.28032](https://arxiv.org/abs/2603.28032) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心特点**：
- 统一空中和地面智能体
- 支持视觉语言动作任务
- 高保真仿真环境
- 空地协同工作负载

---

## 4. UAVBench / UAVIT-1M — 无人机VLM基准

**一句话总结**：966K样本的无人机视觉语言基准

| 项目 | 内容 |
|------|------|
| **标题** | UAVBench and UAVIT-1M |
| **arXiv** | [2603.14336](https://arxiv.org/abs/2603.14336) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**数据规模**：
- 43个测试单元
- 966K样本
- 1.24M指令调优数据集

**任务类型**：低空无人机视觉语言任务

---

## 5. BEDI — 无人机具身智能基准

**一句话总结**：评估无人机具身智能的六大核心子技能

| 项目 | 内容 |
|------|------|
| **标题** | BEDI: A Comprehensive Benchmark for Evaluating Embodied Agents on UAVs |
| **arXiv** | [2505.18229](https://arxiv.org/abs/2505.18229) |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**六大子技能**：
1. 语义感知
2. 空间感知
3. 运动控制
4. 任务理解
5. 规划推理
6. 安全约束

---

## 6. Embodied4C — 具身VLN基准

**一句话总结**：跨车辆、无人机、操作器的闭环VLN评估

| 项目 | 内容 |
|------|------|
| **标题** | Embodied4C: Measuring What Matters for Embodied Vision-Language Navigation |
| **arXiv** | [2512.18028](https://arxiv.org/abs/2512.18028) |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**评估内容**：
- 约1100个推理问题
- 58个导航任务
- 四种推理能力（空间、时间、语义、物理）

**关键发现**：跨模态对齐和指令调优比模型规模更重要

---

## 7. OS-W2S — 开放集航拍目标检测

**一句话总结**：用VLM自动标注航拍图像，163K图像2M描述对

| 项目 | 内容 |
|------|------|
| **标题** | OS-W2S: An Automatic Labeling Engine for Language-Guided Open-Set Aerial Object Detection |
| **arXiv** | [2505.03334](https://arxiv.org/abs/2505.03334) |
| **年份** | 2025 |
| **推荐度** | ★☆☆ |

**数据规模**：
- 163K图像
- 2M描述对
- 开放集标注

---

## 8. Nano-drone Benchmark — 纳米无人机系统辨识

**一句话总结**：Crazyflie 2.1的75k真实样本系统辨识基准

| 项目 | 内容 |
|------|------|
| **标题** | Nonlinear System Identification Nano-drone Benchmark |
| **arXiv** | [2512.14450](https://arxiv.org/abs/2512.14450) |
| **年份** | 2025 |
| **推荐度** | ★☆☆ |

**数据规模**：75k真实世界样本

**平台**：Crazyflie 2.1纳米四旋翼

---

## 9. UAV-Flow — 语言条件无人机控制基准

**一句话总结**："Flying-on-a-Word"任务基准

| 项目 | 内容 |
|------|------|
| **标题** | UAV-Flow Colosseo |
| **arXiv** | [2505.15725](https://arxiv.org/abs/2505.15725) |
| **GitHub** | [buaa-colalab/UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) |
| **HuggingFace** | wangxiangyu0814/UAV-Flow, UAV-Flow-Sim |
| **年份** | 2025 |
| **推荐度** | ★★★ |

**数据内容**：
- 真实世界无人机轨迹
- 仿真无人机轨迹
- OpenVLA-UAV适配模型
- 评估环境

---

## 基准对比表

| 基准 | 类型 | 规模 | 任务 | 开源 |
|------|------|------|------|------|
| MotionScape | 视频数据 | 30+小时 | 世界模型训练 | ✓ |
| AeroVerse | 基准套件 | 5任务 | 多种 | ✓ |
| CARLA-Air | 仿真平台 | 无限 | 多种 | ✓ |
| UAVBench | VLM基准 | 966K | 视觉语言 | ✓ |
| BEDI | 具身基准 | 6技能 | 多种 | ✓ |
| Embodied4C | VLN基准 | 1100题 | 导航 | ✓ |
| UAV-Flow | 控制基准 | 多数据集 | 语言控制 | ✓ |
| OS-W2S | 检测数据 | 163K | 目标检测 | ✓ |
| Nano-drone | 系统辨识 | 75K | 动力学 | ✓ |

---

## 延伸阅读

- [关键数据集与基准](../02-世界模型专题/06-关键数据集与基准.md) — 数据集详解
- [完整论文列表](../../references/paper-list.md) — 50+篇论文分类汇总
