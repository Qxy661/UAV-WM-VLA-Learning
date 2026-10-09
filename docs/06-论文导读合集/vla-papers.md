# VLA 论文卡片

> 预计阅读：20 分钟 | 前置知识：了解VLA基础概念（见[什么是VLA](../01-基础概念/03-什么是VLA.md)）
>
> 本文收录 11 篇 VLA 关键论文。每张卡片由「标题行 → 徽章行 → 一句话总结 → 核心贡献 / 方法要点 / 结果亮点」组成，题名取 arXiv API 返回的完整题名

---

## 1. VLA-AN — 机载VLA框架

**一句话总结**：把 VLA 压到能上机载的端到端框架

**标题**：*VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments*

[![arXiv](https://img.shields.io/badge/arXiv-2512.15258-b31b1b.svg)](https://arxiv.org/abs/2512.15258) · 2025 · 推荐度 ★★★

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

**标题**：*CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs*

[![arXiv](https://img.shields.io/badge/arXiv-2503.01378-b31b1b.svg)](https://arxiv.org/abs/2503.01378) [![GitHub](https://img.shields.io/badge/GitHub-docker__CognitiveDrone__DataCollector-181717.svg?logo=github)](https://github.com/SerValera/docker_CognitiveDrone_DataCollector) [![Website](https://img.shields.io/badge/Website-cognitivedrone.github.io-0A66C2.svg)](https://cognitivedrone.github.io) · 2025 · 推荐度 ★★★

**核心贡献**：
- 8000+模拟飞行轨迹训练
- 实时4D动作输出
- CognitiveDrone-R1：VLM推理模块分解复杂指令
- 认知任务基准

**方法要点**：第一人称视觉输入 + 文本指令，高频控制输出。

**结果亮点**：
- CognitiveDrone: 59.6%成功率 (vs 31.3%基线)
- CognitiveDrone-R1: 77.2%成功率

---

## 3. UAV-TrackVLA — 无人机跟踪VLA

**一句话总结**：基于π₀.₅的无人机跟踪VLA，时序压缩+双分支解码

**标题**：*UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models*

[![arXiv](https://img.shields.io/badge/arXiv-2604.02241-b31b1b.svg)](https://arxiv.org/abs/2604.02241) [![GitHub](https://img.shields.io/badge/GitHub-UAV--Track__VLA-181717.svg?logo=github)](https://github.com/Hub-Tian/UAV-Track_VLA) · 2026 · 推荐度 ★★☆

**核心贡献**：
- 基于π₀.₅ VLA架构
- 时序压缩网络处理帧间动态
- 并行双分支解码器
- 空间感知辅助定位头+流匹配动作专家

**训练数据**：890K+ 帧，176 任务，85 目标（CARLA）。

**结果亮点**：
- 61.76%成功率
- 269.65平均跟踪帧
- 0.0571s/步推理延迟

---

## 4. UAV-Flow — 语言条件无人机控制

**一句话总结**：提出细粒度语言条件飞行任务 "Flying-on-a-Word"，并开源数据与模型

**标题**：*UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning*

[![arXiv](https://img.shields.io/badge/arXiv-2505.15725-b31b1b.svg)](https://arxiv.org/abs/2505.15725) [![GitHub](https://img.shields.io/badge/GitHub-UAV--Flow-181717.svg?logo=github)](https://github.com/buaa-colalab/UAV-Flow) · 约 170 stars · 2025 · HuggingFace wangxiangyu0814/UAV-Flow, OpenVLA-UAV · 推荐度 ★★★

**核心贡献**：
- OpenVLA-UAV适配模型
- 真实和仿真数据集

**方法要点**：
- 模仿学习：模仿专家飞行员轨迹
- 原子语言指令配对
- UnrealCV 仿真环境

**结果亮点**：VLA模型在细粒度无人机控制上显著优于VLN基线

> **勘误（2026-10）**：本卡片早先写「UnrealZoo Gym评估环境」。arXiv:2505.15725 原文用的是 *"We utilize UnrealCV as the simulation environment for the UAV"*，全文无 "gym" 命中。

---

## 5. VLN-Pilot — VLM作为室内无人机操作员

**一句话总结**：大型VLM替代人类飞行员，在GPS拒止环境中操作室内无人机

**标题**：*VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator*

[![arXiv](https://img.shields.io/badge/arXiv-2602.05552-b31b1b.svg)](https://arxiv.org/abs/2602.05552) · 2026 · 推荐度 ★★☆

**核心贡献**：
- 自由形式自然语言指令解释
- GPS拒止环境导航

**应用场景**：巡检、搜救、设施监控

---

## 6. CoDrone — 云边端基础模型无人机导航

**一句话总结**：首个云边端协同框架，将基础模型集成到无人机巡航中

**标题**：*CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models*

[![arXiv](https://img.shields.io/badge/arXiv-2512.19083-b31b1b.svg)](https://arxiv.org/abs/2512.19083) · IEEE Internet of Things Journal · 2025 · 推荐度 ★★☆

**核心贡献**：
- 灰度图像减少计算
- 边缘辅助深度估计（Depth Anything V2）
- DRL神经调度器

**结果亮点**：
- 40%平均飞行距离提升
- 5%导航质量改进

---

## 7. FM-Planner — 基础模型路径规划

**一句话总结**：系统评估8种LLM/VLM方法用于无人机路径规划

**标题**：*FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation*

[![arXiv](https://img.shields.io/badge/arXiv-2505.20783-b31b1b.svg)](https://arxiv.org/abs/2505.20783) [![GitHub](https://img.shields.io/badge/GitHub-FM--Planner-181717.svg?logo=github)](https://github.com/NTU-ICG/FM-Planner) · 2025 · 推荐度 ★★☆

**核心贡献**：
- LLM-Vision集成规划器
- 语义推理+视觉感知结合
- 真实世界验证

---

## 8. NavFoM — 跨形态导航基础模型

**一句话总结**：跨四足、无人机、轮式、车辆的统一导航基础模型

**标题**：*Embodied Navigation Foundation Model（系统名 NavFoM）*

[![arXiv](https://img.shields.io/badge/arXiv-2509.12129-b31b1b.svg)](https://arxiv.org/abs/2509.12129) [![Website](https://img.shields.io/badge/Website-pku--epic.github.io-0A66C2.svg)](https://pku-epic.github.io/NavFoM-Web/) · 2025 · 北京大学 · 推荐度 ★★☆

**核心贡献**：
- 8M样本跨形态训练
- 标识符token嵌入相机视角和时间上下文
- 动态token采样策略
- 无需任务特定微调

**结果亮点**：跨多形态达到SOTA或竞争性能

---

## 9. CityNavAgent — 城市航空VLN

**一句话总结**：LLM驱动的城市航空VLN代理，层次化语义规划+全局记忆

**标题**：*CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory*

[![arXiv](https://img.shields.io/badge/arXiv-2505.05622-b31b1b.svg)](https://arxiv.org/abs/2505.05622) · ACL 2025 · 2025 · 推荐度 ★★☆

**核心贡献**：
- 拓扑记忆图
- 长期任务分解

---

## 10. UAV-VLRR — 视觉语言NMPC搜救

**一句话总结**：VLM+ChatGPT-4o场景解释+NMPC控制，搜救响应时间提升33.75%

**标题**：*UAV-VLRR: Vision-Language Informed NMPC for Rapid Response in UAV Search and Rescue*

[![arXiv](https://img.shields.io/badge/arXiv-2503.02465-b31b1b.svg)](https://arxiv.org/abs/2503.02465) · 2025 · Skolkovo · 推荐度 ★★☆

**核心贡献**：
- NMPC控制

---

## 11. AutoFly — 野外自主导航VLA

**一句话总结**：端到端VLA用于野外自主无人机导航，ICLR 2026

**标题**：*AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild*

[![arXiv](https://img.shields.io/badge/arXiv-2602.09657-b31b1b.svg)](https://arxiv.org/abs/2602.09657) · ICLR 2026 · 2026 · 推荐度 ★★☆

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
- [完整论文列表](../../references/paper-list.md) — 195 篇论文分类汇总

## 思考题

1. **对比分析**：VLA-AN 的 98.1% 和 CognitiveDrone-R1 的 77.2% 都叫"成功率"，能直接横向比较吗？为什么？

2. **架构选型**：UAV-TrackVLA 走"时序压缩 + 并行双分支解码"，AutoFly 走"伪深度编码器 + 渐进两阶段训练"。面向算力受限的机载平台，你更该借鉴哪一套，依据是什么？

3. **方案权衡**：CoDrone 选择云边端协同，VLA-AN 选择纯机载部署。换成 GPS 拒止的室内任务，你会怎么取舍？

4. **范式辨析**：UAV-Flow 的结论是"VLA 在细粒度控制上优于 VLN"，NavFoM 则主张一个模型跨四足/无人机/轮式/车辆、无需任务特定微调。这两种立场能同时成立吗？

5. **方法迁移**：CityNavAgent 的"层次化语义规划 + 拓扑记忆图"是为城市长期 VLN 设计的。搬去解 UAV-TrackVLA 的目标跟踪任务，哪一环最不匹配？

<details><summary>参考答案</summary>

1. 不能。两个数字来自不同任务与不同评测：VLA-AN 报的是复杂环境空中导航的最大单任务成功率，其系统由 3D Gaussian Splatting 数据生成、三阶段渐进训练（场景理解→飞行技能→复杂导航）和几何安全校正支撑；CognitiveDrone 的基线是 31.3%、本体 59.6%、加上 R1 推理模块后 77.2%，评的是人类识别、符号理解、推理这类认知任务基准。任务定义、对手和环境都不同，数字只反映各自设定下的相对提升。

2. 优先借鉴 UAV-TrackVLA。它以降低推理开销为明确设计目标：时序压缩网络加并行双分支解码器换来 33.4% 的推理延迟降低，实现 0.0571s/步，支撑 CARLA 上 890K+ 帧、176 任务、85 目标的数据规模，并且代码开源。AutoFly 的卖点是伪深度编码器与渐进两阶段训练带来的 3.9% 成功率提升，卡片没有给出任何延迟方面的收益，对算力受限平台的参考价值弱一些。

3. 关键取舍是"能否容忍对外部链路的依赖"。若室内能部署稳定的边缘节点，CoDrone 式的分工可以把大模型放在边缘，并用灰度图像压低算力、用边缘端深度估计（Depth Anything V2）补感知，换来 40% 平均飞行距离提升和 5% 导航质量改进；若要求完全自主或链路不可靠，就以 VLA-AN 的机载路线为基准，它的 8.3x 推理加速和轻量动作模块正是为纯机载设计的。VLN-Pilot 证明了大型 VLM 可以当室内飞行员，但没有回答算力问题。

4. 可以并存，因为两者回答的不是同一个问题，但存在边界。UAV-Flow 比较的是同一类任务下 VLA 与 VLN 两种建模范式，结论落在细粒度动作级控制；NavFoM 比的是跨形态迁移，靠 8M 样本、标识符 token 编码相机视角与时间上下文、动态 token 采样做到无需任务特定微调。张力在于：如果跨形态基础模型真能不微调直接用于无人机，专用无人机 VLA 的必要性就要重新论证；而卡片只说了 NavFoM 跨多形态达到 SOTA 或竞争性能，没有给出细粒度机动的证据，这一层尚未验证。

5. 最不匹配的是"长期任务分解 + 记忆"这条主线。CityNavAgent 的层次化语义规划和拓扑记忆图解决的是城市环境下长时间、跨地标的导航（ACL 2025）；UAV-TrackVLA 的任务是对单一目标持续跟踪，难点在帧间动态与时序信息压缩，靠时序压缩网络、空间感知辅助定位头和流匹配动作专家解决，不需要跨地标的记忆。可迁移的只是"高层选目标、低层出动作"的分层思想，语义规划层和拓扑记忆图基本用不上。

</details>
