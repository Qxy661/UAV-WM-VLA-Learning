# VLM 论文卡片

> 本文收录10+篇VLM关键论文，每篇含一句话总结、核心贡献、方法分析和链接

---

## 1. GeoChat — 首个遥感VLM

**一句话总结**：首个grounded大型遥感VLM，支持多轮对话和视觉定位

| 项目 | 内容 |
|------|------|
| **标题** | GeoChat: Grounded Large Vision-Language Model for Remote Sensing |
| **会议** | CVPR 2024 |
| **arXiv** | [2311.15826](https://arxiv.org/abs/2311.15826) |
| **GitHub** | [mbzuai-oryx/GeoChat](https://github.com/mbzuai-oryx/GeoChat) (713 stars) |
| **数据集** | MBZUAI/GeoChat_Instruct (HuggingFace) |
| **推荐度** | ★★★ |

**核心贡献**：
- 首个grounded遥感VLM
- 多轮对话、VQA、图像描述
- 指代目标检测（输出边界框）
- 区域级任务

**方法要点**：
- 微调自318K图像-指令对
- 支持多种任务格式
- 边界框输出能力

**应用场景**：遥感图像理解、目标检测、场景描述

---

## 2. RSGPT — 遥感VLM

**一句话总结**：遥感图像描述、VQA和视觉定位

| 项目 | 内容 |
|------|------|
| **标题** | RSGPT: A Remote Sensing Vision Language Model and Benchmark |
| **GitHub** | [dirk-niu/RSGPT](https://github.com/dirk-niu/RSGPT) |
| **推荐度** | ★★☆ |

**核心贡献**：
- 遥感图像描述
- 视觉问答
- 视觉定位

---

## 3. SkySenseGPT — 细粒度遥感理解

**一句话总结**：细粒度遥感指令调优，多粒度感知和详细描述生成

| 项目 | 内容 |
|------|------|
| **标题** | SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding |
| **GitHub** | [CVI-SZU/SkySenseGPT](https://github.com/CVI-SZU/SkySenseGPT) |
| **机构** | 深圳大学 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 细粒度遥感理解
- 多粒度感知
- 视觉定位和指代表达理解
- 区域级描述生成

---

## 4. EarthGPT — 多传感器遥感理解

**一句话总结**：通用多模态LLM，理解多种遥感传感器图像

| 项目 | 内容 |
|------|------|
| **标题** | EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain |
| **期刊** | IEEE TGRS |
| **GitHub** | [limanling/EarthGPT](https://github.com/limanling/EarthGPT) |
| **推荐度** | ★★☆ |

**核心贡献**：
- 多传感器支持（光学、SAR等）
- VQA和图像描述
- 跨传感器理解

---

## 5. RS-LLaVA — 遥感LLaVA适配

**一句话总结**：将LLaVA适配到遥感领域

| 项目 | 内容 |
|------|------|
| **GitHub** | [LUOYUKAI/RS-LLaVA](https://github.com/LUOYUKAI/RS-LLaVA) |
| **推荐度** | ★☆☆ |

**核心贡献**：
- LLaVA遥感适配
- 遥感数据微调

---

## 6. ChangeChat — 双时相变化分析

**一句话总结**：用LLM进行双时相遥感图像变化分析

| 项目 | 内容 |
|------|------|
| **GitHub** | [Chen-Yijun/ChangeChat](https://github.com/Chen-Yijun/ChangeChat) |
| **推荐度** | ★☆☆ |

**核心贡献**：
- 双时相变化检测
- 自然语言描述变化

---

## 7. LHRS-Bot — 大规模高分辨率遥感VLM

**一句话总结**：大规模高分辨率遥感VLM

| 项目 | 内容 |
|------|------|
| **GitHub** | [NJU-LHRS/LHRS-Bot](https://github.com/NJU-LHRS/LHRS-Bot) |
| **推荐度** | ★☆☆ |

**核心贡献**：
- 高分辨率遥感理解
- 大规模训练

---

## 8. UAVBench — 无人机VLM基准

**一句话总结**：966K样本的无人机视觉语言基准，1.24M指令调优数据

| 项目 | 内容 |
|------|------|
| **标题** | UAVBench and UAVIT-1M |
| **arXiv** | [2603.14336](https://arxiv.org/abs/2603.14336) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 43个测试单元
- 966K样本
- 1.24M指令调优数据集
- 低空无人机视觉语言任务

---

## 9. BEDI — 无人机具身智能基准

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

## 10. Embodied4C — 具身VLN基准

**一句话总结**：跨车辆、无人机、操作器的闭环VLN评估基准

| 项目 | 内容 |
|------|------|
| **标题** | Embodied4C: Measuring What Matters for Embodied Vision-Language Navigation |
| **arXiv** | [2512.18028](https://arxiv.org/abs/2512.18028) |
| **年份** | 2025 |
| **推荐度** | ★★☆ |

**四种推理**：
1. 空间推理
2. 时间推理
3. 语义推理
4. 物理推理

**关键发现**：跨模态对齐和指令调优比模型规模更重要

---

## 11. VLN for UAVs Survey — 无人机VLN综述

**一句话总结**：无人机视觉语言导航的进展、挑战和研究路线图

| 项目 | 内容 |
|------|------|
| **标题** | Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap |
| **arXiv** | [2604.13654](https://arxiv.org/abs/2604.13654) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 无人机VLN全面综述
- VLA架构整合
- 世界模型与VLA的融合趋势

---

## 12. VLN for Aerial Robots Survey — 航空VLN综述

**一句话总结**：航空机器人VLN方法综述，LLM/VLM整合分析

| 项目 | 内容 |
|------|------|
| **标题** | Vision-Language Navigation for Aerial Robots: Towards the Era of Large Language Models |
| **arXiv** | [2604.07705](https://arxiv.org/abs/2604.07705) |
| **年份** | 2026 |
| **推荐度** | ★★☆ |

**核心贡献**：
- 航空VLN方法分类
- LLM/VLM整合分析
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
