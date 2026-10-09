# VLM 论文卡片

> 预计阅读：20 分钟 | 前置知识：了解VLM基础概念（见[什么是VLM](../01-基础概念/02-什么是VLM.md)）
>
> 本文收录 12 篇 VLM 关键论文。每张卡片由「标题行 → 徽章行 → 一句话总结 → 核心贡献 / 方法要点 / 结果亮点」组成，题名取 arXiv API 返回的完整题名

---

## 1. GeoChat — 首个遥感VLM

**一句话总结**：首个grounded大型遥感VLM，支持多轮对话和视觉定位

**标题**：*GeoChat: Grounded Large Vision-Language Model for Remote Sensing*

[![arXiv](https://img.shields.io/badge/arXiv-2311.15826-b31b1b.svg)](https://arxiv.org/abs/2311.15826) [![GitHub](https://img.shields.io/badge/GitHub-GeoChat-181717.svg?logo=github)](https://github.com/mbzuai-oryx/GeoChat) · 约 760 stars · CVPR 2024 · 数据集 MBZUAI/GeoChat_Instruct (HuggingFace) · 推荐度 ★★★

**方法要点**：微调自 318K 图像-指令对，支持多轮对话、VQA、图像描述和指代目标检测（输出边界框）。

**应用场景**：遥感图像理解、目标检测、场景描述

---

## 2. RSGPT — 遥感VLM

**一句话总结**：模型和评测基准成对给出，覆盖遥感图像描述、VQA 和视觉定位

**标题**：*RSGPT: A Remote Sensing Vision Language Model and Benchmark*

[![arXiv](https://img.shields.io/badge/arXiv-2307.15266-b31b1b.svg)](https://arxiv.org/abs/2307.15266) [![GitHub](https://img.shields.io/badge/GitHub-RSGPT-181717.svg?logo=github)](https://github.com/Lavender105/RSGPT) · 推荐度 ★★☆

---

## 3. SkySenseGPT — 细粒度遥感理解

**一句话总结**：细粒度遥感指令调优，支持多粒度感知、详细描述生成，以及视觉定位和指代表达理解

**标题**：*SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding*

[![arXiv](https://img.shields.io/badge/arXiv-2406.10100-b31b1b.svg)](https://arxiv.org/abs/2406.10100) [![GitHub](https://img.shields.io/badge/GitHub-SkySenseGPT-181717.svg?logo=github)](https://github.com/Luo-Z13/SkySenseGPT) · 武汉大学 · 推荐度 ★★☆

> **勘误（2026-10）**：本卡片早先写的机构是「深圳大学」。arXiv:2406.10100 正文署名为 Wuhan University（武汉大学）。

---

## 4. EarthGPT — 多传感器遥感理解

**一句话总结**：通用多模态 LLM，把光学、SAR 等多种遥感传感器图像放进同一个模型理解

**标题**：*EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain*

[![arXiv](https://img.shields.io/badge/arXiv-2401.16822-b31b1b.svg)](https://arxiv.org/abs/2401.16822) [![GitHub](https://img.shields.io/badge/GitHub-EarthGPT-181717.svg?logo=github)](https://github.com/wivizhang/EarthGPT) · IEEE TGRS · 推荐度 ★★☆

---

## 5. RS-LLaVA — 遥感LLaVA适配

**一句话总结**：把 LLaVA 在遥感数据上微调，适配遥感领域

**标题**：*RS-LLaVA: A Large Vision-Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery*

[![GitHub](https://img.shields.io/badge/GitHub-RS--LLaVA-181717.svg?logo=github)](https://github.com/BigData-KSU/RS-LLaVA) · MDPI Remote Sensing · 推荐度 ★☆☆

---

## 6. ChangeChat — 双时相变化分析

**一句话总结**：用 LLM 分析双时相遥感图像的变化，并用自然语言描述

**标题**：*ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning*

[![arXiv](https://img.shields.io/badge/arXiv-2409.08582-b31b1b.svg)](https://arxiv.org/abs/2409.08582) [![GitHub](https://img.shields.io/badge/GitHub-ChangeChat-181717.svg?logo=github)](https://github.com/hanlinwu/ChangeChat) · 推荐度 ★☆☆

---

## 7. LHRS-Bot — 高分辨率遥感VLM

**一句话总结**：面向高分辨率遥感图像，用大规模数据训练

**标题**：*LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model*

[![arXiv](https://img.shields.io/badge/arXiv-2402.02544-b31b1b.svg)](https://arxiv.org/abs/2402.02544) [![GitHub](https://img.shields.io/badge/GitHub-LHRS--Bot-181717.svg?logo=github)](https://github.com/NJU-LHRS/LHRS-Bot) · ECCV 2024 · 推荐度 ★☆☆

---

## 8. UAVBench — 无人机VLM基准

**一句话总结**：面向低空无人机的视觉语言基准，含配套指令调优数据

**标题**：*UAVBench and UAVIT-1M: Benchmarking and Enhancing MLLMs for Low-Altitude UAV Vision-Language Understanding*

[![arXiv](https://img.shields.io/badge/arXiv-2603.14336-b31b1b.svg)](https://arxiv.org/abs/2603.14336) · 2026 · 推荐度 ★★☆

**数据规模**：966K 样本，1.24M 指令调优数据，43 个测试单元。

---

## 9. BEDI — 无人机具身智能基准

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

> **勘误（2026-10）**：本卡片早先的六项是「语义感知 / 空间感知 / 运动控制 / 任务理解 / 规划推理 / 安全约束」，其中**后三项原文里没有**。arXiv:2505.18229 原文写的是 *"six core sub-skills: semantic perception, spatial perception, motion control, tool utilization, task planning and action generation"*，已按原文改正。

---

## 10. Embodied4C — 具身VLN基准

**一句话总结**：跨车辆、无人机、操作器的闭环VLN评估基准

**标题**：*Embodied4C: Measuring What Matters for Embodied Vision-Language Navigation*

[![arXiv](https://img.shields.io/badge/arXiv-2512.18028-b31b1b.svg)](https://arxiv.org/abs/2512.18028) · 2025 · 推荐度 ★★☆

**四种推理**：
1. 空间推理
2. 时间推理
3. 语义推理
4. 物理推理

**关键发现**：跨模态对齐和指令调优比模型规模更重要

---

## 11. VLN for UAVs Survey — 无人机VLN综述

**一句话总结**：无人机视觉语言导航的进展、挑战和研究路线图

**标题**：*Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap*

[![arXiv](https://img.shields.io/badge/arXiv-2604.13654-b31b1b.svg)](https://arxiv.org/abs/2604.13654) · 2026 · 推荐度 ★★☆

**核心贡献**：
- VLA架构整合
- 世界模型与VLA的融合趋势

---

## 12. VLN for Aerial Robots Survey — 航空VLN综述

**一句话总结**：航空机器人VLN方法综述，LLM/VLM整合分析

**标题**：*Vision-Language Navigation for Aerial Robots: Towards the Era of Large Language Models*

[![arXiv](https://img.shields.io/badge/arXiv-2604.07705-b31b1b.svg)](https://arxiv.org/abs/2604.07705) · 2026 · 推荐度 ★★☆

**核心贡献**：
- 航空VLN方法分类
- 端到端LLM/VLM方法

---

## 论文推荐优先级

| 优先级 | 论文 | 理由 |
|--------|------|------|
| ★★★ | GeoChat | CVPR 2024，首个遥感VLM，开源 |
| ★★☆ | UAVBench | 无人机VLM基准 |
| ★★☆ | BEDI | 六大子技能评估 |
| ★★☆ | Embodied4C | 跨形态VLN基准 |
| ★★☆ | SkySenseGPT | 细粒度遥感理解 |
| ★★☆ | EarthGPT | 多传感器支持 |
| ★★☆ | VLN Surveys | 综述类，了解全貌 |
| ★☆☆ | RSGPT, RS-LLaVA等 | 了解即可 |

---

## 延伸阅读

- [世界模型论文卡片](world-model-papers.md) — 世界模型论文导读
- [VLA论文卡片](vla-papers.md) — VLA论文导读
- [遥感VLM](../04-VLM专题/01-遥感VLM.md) — 遥感VLM详解

## 思考题

1. **选型判断**：给低空无人机做场景理解，GeoChat 和 UAVBench 该怎么分工？

2. **对比分析**：GeoChat 是 ★★★，RS-LLaVA 是 ★☆☆，两者都在做"把通用 VLM 适配到遥感"，差距出在哪里？

3. **信息一致性**：UAVBench、BEDI、Embodied4C 在本文和 [基准与数据集论文卡片](benchmark-papers.md) 里各有一张卡片。如果两处描述将来出现冲突，你会怎么处理？

4. **能力补全**：本文 12 张卡片里，哪一张是真正处理"时间维度"的？它的时间性和导航需要的时间推理差别在哪？

5. **研究定位**：第 11、12 篇都是 ★★☆ 的 VLN 综述。它们在选题流程里能提供什么、又不能替你做掉什么？

<details><summary>参考答案</summary>

1. 分工取决于你要的是"能力"还是"标尺"。GeoChat 提供的是可借鉴的任务格式与训练范式：微调自 318K 图像-指令对，是首个 grounded 遥感 VLM，支持多轮对话、VQA、图像描述、输出边界框的指代目标检测和区域级任务。UAVBench 则是低空无人机视觉语言的评测与调优资源（43 个测试单元、966K 样本、1.24M 指令调优数据）。一种做法是借 GeoChat 的 grounded 输出格式补上区域级定位能力，再回到 UAVBench 上评测。

2. 差别在贡献的新颖度与可验证性。GeoChat 是首个 grounded 遥感 VLM，CVPR 2024，开源且社区热度高（759 stars，2026-10 查），有明确的训练数据规模（318K 图像-指令对）和多种任务格式（对话、VQA、描述、指代检测、区域级任务）。RS-LLaVA 的卡片只写了"LLaVA 遥感适配 + 遥感数据微调"，属于常规的领域移植，既没有新任务能力也没有新数据，因此只到"了解即可"。

3. 先核验再定源，不要在两个副本之间来回改。目前两份卡片的口径是一致的（UAVBench 966K 样本 / 1.24M 指令调优数据 / 43 个测试单元，BEDI 六大子技能，Embodied4C 约 1100 个推理问题、58 个导航任务），但同一个事实存在两处拷贝本身就是隐患。稳妥做法是以论文原文（arXiv 2603.14336、2505.18229、2512.18028）核验，并在仓库里指定唯一的元数据出处（如 references/paper-list.md），其余位置只链接、不复制。

4. 是 ChangeChat，它做双时相遥感图像的变化分析并用自然语言描述变化，是唯一显式面向"两个时刻之间发生了什么"的卡片。它和导航需要的时间推理不同：ChangeChat 比对的是两期静态影像的差异，属于变化检测；Embodied4C 的四种推理能力里虽然包含时间推理，但服务的是闭环导航中对时序关系与动作后果的判断。一个是被动比对，一个是主动决策的支撑。

5. 能提供的是地图：VLN for UAVs Survey 给出无人机 VLN 的进展、挑战和研究路线图，并指出 VLA 架构整合以及世界模型与 VLA 的融合趋势；VLN for Aerial Robots Survey 给出航空 VLN 的方法分类、LLM/VLM 整合分析和端到端 LLM/VLM 方法。它们替代不了的是证据——两篇卡片都没有任何量化结果或代码，选题时仍要回到具体论文卡片（如 GeoChat 的 318K 指令对、Embodied4C 的 1100 个推理问题）去确认哪条路线真有数据支撑。

</details>
