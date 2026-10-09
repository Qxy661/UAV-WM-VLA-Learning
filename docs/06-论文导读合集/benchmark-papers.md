# 基准与数据集论文卡片

> 预计阅读：15 分钟 | 前置知识：了解世界模型、VLA、VLM基础概念
>
> 本文收录 9 篇无人机 VLA/VLM/世界模型领域的基准与数据集论文。每张卡片由「标题行 → 徽章行 → 一句话总结 → 核心贡献 / 方法要点 / 结果亮点」组成，题名取 arXiv API 返回的完整题名

---

## 1. MotionScape — 大规模无人机视频数据集

**一句话总结**：按运动强度分层的无人机视频基准，228 段 / 62,700 帧，配天气、光照、场景与相机运动标注

**标题**：*MotionScape: A Motion-Stratified UAV Video Benchmark for World Modeling and Future Video Generation*

[![arXiv](https://img.shields.io/badge/arXiv-2604.07991-b31b1b.svg)](https://arxiv.org/abs/2604.07991) [![GitHub](https://img.shields.io/badge/GitHub-MotionScape-181717.svg?logo=github)](https://github.com/Thelegendzz/MotionScape) · 2026 · 推荐度 ★★★

**数据规模**：
- 228 段高分辨率视频，共 62,700 帧（约 35 分钟）
- 标注天气与光照条件、场景环境、相机视角运动
- 按相机运动强度分层（低/中/高运动层）

**数据管线**：
1. 人工选源
2. CLIP 辅助筛查
3. 人工校验

**关键发现**：对齐的标注有效提升现有世界模型模拟复杂3D动态和处理大视角变化的能力

---

## 2. AeroVerse — 航空航天世界模型基准

**一句话总结**：把世界模型的预训练、微调和评估收进同一套基准

**标题**：*AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models*

[![arXiv](https://img.shields.io/badge/arXiv-2408.15511-b31b1b.svg)](https://arxiv.org/abs/2408.15511) · 2024 · 中国科学院空天信息创新研究院 · 推荐度 ★★☆

**五大任务**：
1. 场景认知（aerospace embodied scene awareness）
2. 空间推理（spatial reasoning）
3. 导航探索（navigational exploration）
4. 任务规划（task planning）
5. 运动决策（motion decision）

---

## 3. CARLA-Air — 统一空地仿真平台

**一句话总结**：在CARLA世界中飞无人机，统一空中和地面智能体

**标题**：*CARLA-Air: Fly Drones Inside a CARLA World — A Unified Infrastructure for Air-Ground Embodied Intelligence*

[![arXiv](https://img.shields.io/badge/arXiv-2603.28032-b31b1b.svg)](https://arxiv.org/abs/2603.28032) · 2026 · 推荐度 ★★☆

**核心特点**：
- 支持视觉语言动作任务
- 高保真仿真环境
- 空地协同工作负载

---

## 4. UAVBench / UAVIT-1M — 无人机VLM基准

**一句话总结**：面向低空无人机的视觉语言基准

**标题**：*UAVBench and UAVIT-1M: Benchmarking and Enhancing MLLMs for Low-Altitude UAV Vision-Language Understanding*

[![arXiv](https://img.shields.io/badge/arXiv-2603.14336-b31b1b.svg)](https://arxiv.org/abs/2603.14336) · 2026 · 推荐度 ★★☆

**数据规模**：
- 43个测试单元
- 966K样本
- 1.24M指令调优数据集

---

## 5. BEDI — 无人机具身智能基准

**一句话总结**：评估无人机具身智能的六大核心子技能

**标题**：*BEDI: A Comprehensive Benchmark for Evaluating Embodied Agents on UAVs*

[![arXiv](https://img.shields.io/badge/arXiv-2505.18229-b31b1b.svg)](https://arxiv.org/abs/2505.18229) · 2025 · 推荐度 ★★☆

**六大子技能**：
1. 语义感知（semantic perception）
2. 空间感知（spatial perception）
3. 运动控制（motion control）
4. 工具使用（tool utilization）
5. 任务规划（task planning）
6. 动作生成（action generation）

---

## 6. Embodied4C — 具身VLN基准

**一句话总结**：跨车辆、无人机、操作器的闭环VLN评估

**标题**：*Embodied4C: Measuring What Matters for Embodied Vision-Language Navigation*

[![arXiv](https://img.shields.io/badge/arXiv-2512.18028-b31b1b.svg)](https://arxiv.org/abs/2512.18028) · 2025 · 推荐度 ★★☆

**评估内容**：
- 约1100个推理问题
- 58个导航任务
- 四种推理能力（空间、时间、语义、物理）

**关键发现**：跨模态对齐和指令调优比模型规模更重要

---

## 7. OS-W2S — 开放集航拍目标检测

**一句话总结**：用 VLM 自动生成航拍图像的开放集标注

**标题**：*OS-W2S: An Automatic Labeling Engine for Language-Guided Open-Set Aerial Object Detection*

[![arXiv](https://img.shields.io/badge/arXiv-2505.03334-b31b1b.svg)](https://arxiv.org/abs/2505.03334) · 2025 · 推荐度 ★☆☆

**数据规模**：163K 图像，2M 描述对。

---

## 8. Nano-drone Benchmark — 纳米无人机系统辨识

**一句话总结**：Crazyflie 2.1的75k真实样本系统辨识基准

**标题**：*Nonlinear System Identification Nano-drone Benchmark*

[![arXiv](https://img.shields.io/badge/arXiv-2512.14450-b31b1b.svg)](https://arxiv.org/abs/2512.14450) · 2025 · 推荐度 ★☆☆

---

## 9. UAV-Flow — 语言条件无人机控制基准

**一句话总结**：细粒度语言条件飞行控制的评测基准

**标题**：*UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning*

[![arXiv](https://img.shields.io/badge/arXiv-2505.15725-b31b1b.svg)](https://arxiv.org/abs/2505.15725) [![GitHub](https://img.shields.io/badge/GitHub-UAV--Flow-181717.svg?logo=github)](https://github.com/buaa-colalab/UAV-Flow) · 2025 · HuggingFace wangxiangyu0814/UAV-Flow, UAV-Flow-Sim · 推荐度 ★★★

**数据内容**：
- 真实世界无人机轨迹
- 仿真无人机轨迹
- OpenVLA-UAV适配模型
- 评估环境

---

## 基准对比表

| 基准 | 类型 | 规模 | 任务 |
|------|------|------|------|
| MotionScape | 视频数据 | 30+小时 | 世界模型训练 |
| AeroVerse | 基准套件 | 5任务 | 多种 |
| CARLA-Air | 仿真平台 | 无限 | 多种 |
| UAVBench | VLM基准 | 966K | 视觉语言 |
| BEDI | 具身基准 | 6技能 | 多种 |
| Embodied4C | VLN基准 | 1100题 | 导航 |
| UAV-Flow | 控制基准 | 多数据集 | 语言控制 |
| OS-W2S | 检测数据 | 163K | 目标检测 |
| Nano-drone | 系统辨识 | 75K | 动力学 |

---

## 延伸阅读

- [关键数据集与基准](../02-世界模型专题/06-关键数据集与基准.md) — 数据集详解
- [完整论文列表](../../references/paper-list.md) — 195 篇论文分类汇总

## 思考题

1. **选型判断**：如果目标是训练一个无人机世界模型，本文九项资源里哪一项最直接可用？哪一项名字里也带"基准"、却完全不是同一类东西？

2. **可比性批判**：BEDI、Embodied4C、AeroVerse 都在评估无人机具身智能，它们的成绩能否直接横向比较？为什么？

3. **方法迁移**：OS-W2S 用 VLM 自动标注出 163K 图像、2M 描述对。把这套"自动标注引擎"搬到 MotionScape 这类带轨迹的视频数据上，最先卡住的是哪一环？

4. **实验设计**：你要做"无人机听懂原子语言指令并完成细粒度机动"的闭环实验，UAV-Flow 和 UAVBench 分别能给你什么、缺什么？

5. **批判思考**：Embodied4C 的关键发现是"跨模态对齐和指令调优比模型规模更重要"。如果这条结论成立，UAVBench 堆到 966K 样本、1.24M 指令调优数据还有价值吗？

<details><summary>参考答案</summary>

1. 最直接的是 MotionScape：30+ 小时 4K 视频、4.5M+ 帧、准确的 6-DoF 相机轨迹加细粒度自然语言描述，卡片明确说对齐的标注能提升现有世界模型模拟复杂 3D 动态和处理大视角变化的能力。Nano-drone Benchmark 虽然也叫基准，但做的是 Crazyflie 2.1 纳米四旋翼的 75k 真实样本系统辨识，产出的是动力学模型而不是视觉世界模型数据。

2. 不能。三者的切分维度不同：BEDI 按语义感知、空间感知、运动控制、任务理解、规划推理、安全约束六大子技能组织；Embodied4C 按空间、时间、语义、物理四种推理能力组织，且只针对闭环 VLN（约 1100 个推理问题、58 个导航任务）；AeroVerse 按场景认知、导航规划、目标跟踪、避障控制、降落评估五大任务组织。任务集合、评测协议和指标都不一样，分数只在各自框架内可比。

3. 卡在时序对齐与相机位姿这一环。OS-W2S 面向的是开放集航拍目标检测的静态图文对（163K 图像、2M 描述对），本质是单帧图像生成描述；MotionScape 要求 6-DoF 相机轨迹与细粒度描述在时间上对齐，其数据管线里的"鲁棒视觉 SLAM 轨迹恢复"和"时间分割"不是 VLM 标注能替代的。直接套用只会得到一批描述正确、但轨迹缺失且时序错位的样本。

4. UAV-Flow 给的是闭环控制需要的东西：形式化的 "Flying-on-a-Word" 任务、真实与仿真无人机轨迹、OpenVLA-UAV 适配模型和评估环境，可以直接拿来做模仿学习。UAVBench/UAVIT-1M 给的是 43 个测试单元、966K 样本的视觉语言评测集和 1.24M 指令调优数据，缺的是动作轨迹与闭环评估环境，只能用来做感知与指令理解，训不出控制策略。

5. 仍有价值，但价值方向变了。Embodied4C 的发现指向"对齐质量与调优方式比参数规模更关键"，而 UAVBench 的 1.24M 指令调优数据和 43 个测试单元，正好是改进跨模态对齐、检验调优效果的素材和标尺。被这条结论否定的是"只把模型堆大就能变好"的思路，而不是大规模指令数据与评测集本身。

</details>
