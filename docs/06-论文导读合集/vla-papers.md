# VLA 论文卡片

> 本文收录10+篇VLA关键论文，每篇含一句话总结、核心贡献、方法分析和链接

---

## 1. VLA-AN — 机载VLA框架

**一句话总结**：端到端VLA框架，98.1%成功率，8.3x推理加速

| 项目 | 内容 |
|------|------|
| **标题** | VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments |
| **arXiv** | [2512.15258](https://arxiv.org/abs/2512.15258) |
| **年份** | 2025 |
| **推荐度** | ★★★ |

**核心贡献**：
- 3D Gaussian Splatting高保真数据生成
- 三阶段渐进训练（场景理解→飞行技能→复杂导航）
- 轻量动作模块+几何安全校正
- 深度推理优化，机载部署

**方法要点**：
- 3DGS缩小sim-to-real差距
- 替换随机生成策略为确定性安全策略
- 推理优化实现8.3x加速

**结果亮点**：
- 98.1%最大单任务成功率
- 8.3x推理吞吐提升
- 实时机载执行

---

## 2. CognitiveDrone — 认知无人机VLA

**一句话总结**：VLA模型处理认知无人机任务：人类识别、符号理解、推理

| 项目 | 内容 |
|------|------|
| **标题** | CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs |
| **arXiv** | [2503.01378](https://arxiv.org/abs/2503.01378) |
| **项目页** | [cognitivedrone.github.io](https://cognitivedrone.github.io) |
| **GitHub** | [SerValera/docker_CognitiveDrone_DataCollector](https://github.com/SerValera/docker_CognitiveDrone_DataCollector) |
| **年份** | 2025 |
| **推荐度** | ★★★ |

**核心贡献**：
- 8000+模拟飞行轨迹训练
- 实时4D动作输出
- CognitiveDrone-R1：VLM推理模块分解复杂指令
- 认知任务基准

**方法要点**：
- 第一人称视觉输入+文本指令
- VLM推理模块（R1版本）
- 高频控制输出

**结果亮点**：
- CognitiveDrone: 59.6%成功率 (vs 31.3%基线)
- CognitiveDrone-R1: 77.2%成功率

---

## 3. UAV-TrackVLA — 无人机跟踪VLA

**一句话总结**：基于π₀.₅的无人机跟踪VLA，时序压缩+双分支解码

| 项目 | 内容 |
|------|------|
| **标题** | UAV-Track VLA: Embodied Aerial Tracking via VLA Models |
| **arXiv** | [2604.02241](https://arxiv.org/abs/2604.02241) |
| **GitHub** | [RobotFlow-Labs/project_uav_trackvla](https://github.com/RobotFlow-Labs/project_uav_trackvla) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 基于π₀.₅ VLA架构
- 时序压缩网络处理帧间动态
- 并行双分支解码器
- 空间感知辅助定位头+流匹配动作专家

**方法要点**：
- 890K+帧，176任务，85目标（CARLA）
- 33.4%推理延迟降低
- 强零样本泛化

**结果亮点**：
- 61.76%成功率
- 269.65平均跟踪帧
- 0.0571s/步推理延迟

---

## 4. UAV-Flow — 语言条件无人机控制

**一句话总结**："Flying-on-a-Word"任务，VLA优于VLN用于细粒度无人机控制

| 项目 | 内容 |
|------|------|
| **标题** | UAV-Flow Colosseo: Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning |
| **arXiv** | [2505.15725](https://arxiv.org/abs/2505.15725) |
| **GitHub** | [buaa-colalab/UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) (124 stars) |
| **HuggingFace** | wangxiangyu0814/UAV-Flow, OpenVLA-UAV |
| **年份** | 2025 |
| **推荐度** | ★★★ |

**核心贡献**：
- 形式化"Flying-on-a-Word"任务
- OpenVLA-UAV适配模型
- 真实和仿真数据集
- VLA优于VLN的关键发现

**方法要点**：
- 模仿学习：模仿专家飞行员轨迹
- 原子语言指令配对
- UnrealZoo Gym评估环境

**结果亮点**：VLA模型在细粒度无人机控制上显著优于VLN基线

---

## 5. VLN-Pilot — VLM作为室内无人机操作员

**一句话总结**：大型VLM替代人类飞行员，在GPS拒止环境中操作室内无人机

| 项目 | 内容 |
|------|------|
| **标题** | VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator |
| **arXiv** | [2602.05552](https://arxiv.org/abs/2602.05552) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心贡献**：
- VLM作为人类飞行员替代
- 自由形式自然语言指令解释
- GPS拒止环境导航

**应用场景**：巡检、搜救、设施监控

---

## 6. CoDrone — 云边端基础模型无人机导航

**一句话总结**：首个云边端协同框架，将基础模型集成到无人机巡航中

| 项目 | 内容 |
|------|------|
| **标题** | CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models |
| **arXiv** | [2512.19083](https://arxiv.org/abs/2512.19083) |
| **期刊** | IEEE Internet of Things Journal |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 云边端协同架构
- 灰度图像减少计算
- 边缘辅助深度估计（Depth Anything V2）
- DRL神经调度器

**结果亮点**：
- 40%平均飞行距离提升
- 5%导航质量改进

---

## 7. FM-Planner — 基础模型路径规划

**一句话总结**：系统评估8种LLM/VLM方法用于无人机路径规划

| 项目 | 内容 |
|------|------|
| **标题** | FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation |
| **arXiv** | [2505.20783](https://arxiv.org/abs/2505.20783) |
| **GitHub** | [NTU-ICG/FM-Planner](https://github.com/NTU-ICG/FM-Planner) |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 8种LLM/VLM方法对比
- LLM-Vision集成规划器
- 语义推理+视觉感知结合
- 真实世界验证

---

## 8. NavFoM — 跨形态导航基础模型

**一句话总结**：跨四足、无人机、轮式、车辆的统一导航基础模型

| 项目 | 内容 |
|------|------|
| **标题** | NavFoM: Embodied Navigation Foundation Model |
| **arXiv** | [2509.12129](https://arxiv.org/abs/2509.12129) |
| **项目页** | [pku-epic.github.io/NavFoM-Web](https://pku-epic.github.io/NavFoM-Web/) |
| **机构** | 北京大学 |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 8M样本跨形态训练
- 标识符token嵌入相机视角和时间上下文
- 动态token采样策略
- 无需任务特定微调

**结果亮点**：跨多形态达到SOTA或竞争性能

---

## 9. CityNavAgent — 城市航空VLN

**一句话总结**：LLM驱动的城市航空VLN代理，层次化语义规划+全局记忆

| 项目 | 内容 |
|------|------|
| **标题** | CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory |
| **会议** | ACL 2025 |
| **arXiv** | [2505.05622](https://arxiv.org/abs/2505.05622) |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 层次化语义规划
- 拓扑记忆图
- 长期任务分解

---

## 10. UAV-VLRR — 视觉语言NMPC搜救

**一句话总结**：VLM+ChatGPT-4o场景解释+NMPC控制，搜救响应时间提升33.75%

| 项目 | 内容 |
|------|------|
| **标题** | UAV-VLRR: Vision-Language Informed NMPC for Rapid Response in UAV Search and Rescue |
| **arXiv** | [2503.02465](https://arxiv.org/abs/2503.02465) |
| **机构** | Skolkovo |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**核心贡献**：
- VLM+ChatGPT-4o场景解释
- NMPC控制
- 搜救响应时间提升33.75%

---

## 11. AutoFly — 野外自主导航VLA

**一句话总结**：端到端VLA用于野外自主无人机导航，ICLR 2026

| 项目 | 内容 |
|------|------|
| **标题** | AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild |
| **会议** | ICLR 2026 |
| **arXiv** | [2602.09657](https://arxiv.org/abs/2602.09657) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 伪深度编码器
- 渐进两阶段训练
- 3.9%成功率提升

---

## 论文推荐优先级

| 优先级 | 论文 | 理由 |
|--------|------|------|
| ★★★ | VLA-AN | 最佳机载VLA，98.1%成功率 |
| ★★★ | CognitiveDrone | 认知任务，R1推理 |
| ★★★ | UAV-Flow | 语言条件控制基准，开源 |
| ★★☆ | UAV-TrackVLA | 跟踪VLA，开源 |
| ★★☆ | VLN-Pilot | VLM作为飞行员 |
| ★★☆ | CoDrone | 云边端架构 |
| ★★☆ | FM-Planner | 8种方法对比 |
| ★★☆ | NavFoM | 跨形态导航 |
| ★★☆ | AutoFly | ICLR 2026 |

---

## 延伸阅读

- [世界模型论文卡片](world-model-papers.md) — 世界模型论文导读
- [VLM论文卡片](vlm-papers.md) — VLM论文导读
- [完整论文列表](../../references/paper-list.md) — 50+篇论文分类汇总
