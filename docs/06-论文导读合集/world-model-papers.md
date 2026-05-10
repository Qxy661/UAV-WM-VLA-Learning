# 世界模型论文卡片

> 本文收录15+篇世界模型关键论文，每篇含一句话总结、核心贡献、方法分析和链接

---

## 论文卡片格式

每篇论文包含：
- **一句话总结**：快速了解论文核心
- **核心贡献**：论文的主要创新点
- **方法要点**：技术方法的关键细节
- **结果亮点**：最重要的实验结果
- **链接**：论文和代码

---

## 1. ANWM — 航空导航世界模型

**一句话总结**：用FFP模块为无人机提供几何先验，实现长距离视觉预测

| 项目 | 内容 |
|------|------|
| **标题** | Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space |
| **arXiv** | [2512.21887](https://arxiv.org/abs/2512.21887) |
| **年份** | 2025 |
| **推荐度** | ★★★ |

**核心贡献**：
- 提出Future Frame Projection (FFP)模块，物理启发的几何先验
- 预测未来视觉观测，条件化于4-DoF无人机动作
- 按语义合理性和导航效用排序候选轨迹

**方法要点**：
- 将过去帧投影到未来相机视角
- 缓解长距离视觉生成中的表示不确定性
- 结合高层语义和低层导航

**结果亮点**：显著超越现有世界模型的长距离视觉预测能力

---

## 2. AirScape — 航空生成式世界模型

**一句话总结**：可控制相机运动的航空视角世界模型

| 项目 | 内容 |
|------|------|
| **标题** | AirScape: An Aerial Generative World Model with Motion Controllability |
| **会议** | ACM 2025 |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 专门为航空视角设计的生成式世界模型
- 支持可控的相机运动参数
- 生成逼真的航空视频序列

**方法要点**：
- 条件视频生成
- 相机运动参数作为条件
- 航空视角的特殊处理

---

## 3. FlightDiffusion — 扩散模型生成FPV视频

**一句话总结**：用扩散模型从单帧生成多样FPV视频，用于无人机策略训练

| 项目 | 内容 |
|------|------|
| **标题** | FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video |
| **arXiv** | [2509.14082](https://arxiv.org/abs/2509.14082) |
| **年份** | 2025 |
| **推荐度** | ★★★ |

**核心贡献**：
- 从单帧生成多样FPV视频轨迹
- 同时生成对应的action space
- 合成数据用于策略学习

**方法要点**：
- 条件扩散模型
- 单帧条件化
- 状态-动作对合成

**结果亮点**：
- 位置误差：0.25m
- 仿真和真实无显著差异 (p=0.541)
- 成功率：约62%

---

## 4. MotionScape — 大规模无人机视频数据集

**一句话总结**：30+小时4K无人机视频数据集，配6-DoF轨迹和语言描述

| 项目 | 内容 |
|------|------|
| **标题** | MotionScape: A Large-Scale Real-World Highly Dynamic UAV Video Dataset for World Models |
| **arXiv** | [2604.07991](https://arxiv.org/abs/2604.07991) |
| **GitHub** | [Thelegendzz/MotionScape](https://github.com/Thelegendzz/MotionScape) |
| **年份** | 2026 |
| **推荐度** | ★★★ |

**核心贡献**：
- 最大规模的真实无人机视频数据集
- 30+小时4K视频，4.5M+帧
- 准确的6-DoF相机轨迹
- 细粒度自然语言描述

**方法要点**：
- CLIP相关性过滤
- 时间分割
- 鲁棒视觉SLAM轨迹恢复
- LLM驱动语义标注

**结果亮点**：对齐的标注有效提升现有世界模型模拟复杂3D动态的能力

---

## 5. AeroVerse — 航空航天世界模型基准

**一句话总结**：最全面的无人机世界模型基准套件，定义五大下游任务

| 项目 | 内容 |
|------|------|
| **标题** | AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models |
| **arXiv** | [2408.15511](https://arxiv.org/abs/2408.15511) |
| **年份** | 2024 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 定义"航空航天世界模型"概念
- 五大下游任务基准
- 预训练和微调数据集

**五大任务**：
1. 场景认知
2. 导航规划
3. 目标跟踪
4. 避障控制
5. 降落评估

---

## 6. Dream to Fly — DreamerV3用于无人机竞速

**一句话总结**：首次将DreamerV3用于无人机竞速，涌现感知意识行为

| 项目 | 内容 |
|------|------|
| **标题** | Dream to fly: Model-based reinforcement learning for vision-based drone flight |
| **arXiv** | [2501.14377](https://arxiv.org/abs/2501.14377) |
| **会议** | ICRA 2026 |
| **机构** | UZH, RPG |
| **推荐度** | ★★★ |

**核心贡献**：
- DreamerV3首次用于无人机竞速
- 从像素学习，无需手工特征
- 涌现的感知意识行为
- 真实四旋翼9m/s飞行

**方法要点**：
- 隐空间世界模型
- 想象训练
- 无需设计奖励项

**结果亮点**：模型方法在样本效率上远超无模型方法 (PPO, SAC)

---

## 7. HDVIO — 混合动力学VIO

**一句话总结**：结合解析模型和学习组件的混合动力学状态估计

| 项目 | 内容 |
|------|------|
| **标题** | HDVIO: Improving Localization and Disturbance Estimation with Hybrid Dynamics VIO |
| **arXiv** | [2306.11429](https://arxiv.org/abs/2306.11429) |
| **机构** | UZH, RPG |
| **年份** | 2023 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 混合动力学模型：解析+学习
- 学习组件捕获残余空气动力学效应
- 改进四旋翼状态估计

**方法要点**：
- 点质量车辆模型 + 学习残差
- 捕获难以解析建模的效应
- 电机动力学学习

---

## 8. Physics-guided Learning on SE(3)

**一句话总结**：将哈密顿方程结构注入神经ODE网络用于四旋翼控制

| 项目 | 内容 |
|------|------|
| **标题** | Physics-guided Learning-based Adaptive Control on the SE(3) Manifold |
| **arXiv** | [2201.04339](https://arxiv.org/abs/2201.04339) |
| **年份** | 2022 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 物理启发的神经ODE
- SE(3)流形上的控制
- 保持物理结构的学习动力学

---

## 9. Learning MPC for Quadrotors

**一句话总结**：利用过去成功任务迭代改进四旋翼控制性能

| 项目 | 内容 |
|------|------|
| **标题** | Learning Model Predictive Control for Quadrotors |
| **arXiv** | [2202.07716](https://arxiv.org/abs/2202.07716) |
| **年份** | 2022 |
| **推荐度** | ★☆☆ |

**核心贡献**：
- 学习MPC方法
- 利用历史数据改进性能
- 尊重系统动力学约束

---

## 10. Wireless Dreamer — 无线边缘智能世界模型

**一句话总结**：世界模型用于无线边缘智能优化，含天气感知无人机轨迹规划

| 项目 | 内容 |
|------|------|
| **标题** | World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks |
| **arXiv** | [2506.00417](https://arxiv.org/abs/2506.00417) |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- "Wireless Dreamer"框架
- 世界模型用于边缘智能
- 天气感知无人机轨迹规划案例

---

## 11. Contrastive World Model for Drone Navigation

**一句话总结**：用对比学习在世界模型框架中学习无人机导航视觉表示

| 项目 | 内容 |
|------|------|
| **标题** | Learning visual representation for autonomous drone navigation via a contrastive world model |
| **期刊** | IEEE Transactions |
| **年份** | 2023 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 对比学习+世界模型
- 学习鲁棒的航空视角场景理解
- 用于无人机导航

---

## 12. Nonlinear System ID Nano-drone Benchmark

**一句话总结**：Crazyflie 2.1纳米四旋翼的75k真实样本系统辨识基准

| 项目 | 内容 |
|------|------|
| **标题** | Nonlinear System Identification Nano-drone Benchmark |
| **arXiv** | [2512.14450](https://arxiv.org/abs/2512.14450) |
| **年份** | 2025 |
| **推荐度** | ★☆☆ |

**核心贡献**：
- 75k真实世界样本
- Crazyflie 2.1平台
- 系统辨识和动力学模型学习

---

## 13. PINN for Multirotor Slung Load

**一句话总结**：用物理信息神经网络学习多旋翼-悬挂负载系统的端到端模型

| 项目 | 内容 |
|------|------|
| **标题** | Physics-Informed Neural Network for Multirotor Slung Load Systems |
| **arXiv** | [2405.09428](https://arxiv.org/abs/2405.09428) |
| **年份** | 2024 |
| **推荐度** | ★☆☆ |

**核心贡献**：
- PINN用于复杂耦合系统
- 端到端学习
- 物理约束保证合理性

---

## 论文推荐优先级

| 优先级 | 论文 | 理由 |
|--------|------|------|
| ★★★ | ANWM | 无人机专属世界模型，FFP创新 |
| ★★★ | FlightDiffusion | 扩散模型+FPV视频生成 |
| ★★★ | Dream to Fly | DreamerV3用于无人机，ICRA 2026 |
| ★★★ | MotionScape | 最大无人机视频数据集 |
| ★★☆ | AirScape | 可控航空世界模型 |
| ★★☆ | AeroVerse | 全面的基准套件 |
| ★★☆ | HDVIO | 混合动力学，UZH RPG |
| ★★☆ | Wireless Dreamer | 边缘智能应用 |
| ★★☆ | Contrastive WM | 对比学习世界模型 |
| ★☆☆ | 其他 | 了解即可 |

---

## 延伸阅读

- [VLA论文卡片](vla-papers.md) — VLA领域论文导读
- [VLM论文卡片](vlm-papers.md) — VLM领域论文导读
- [基准与数据集论文](benchmark-papers.md) — 基准论文导读
- [完整论文列表](../../references/paper-list.md) — 69篇论文分类汇总
