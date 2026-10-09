# 世界模型论文卡片

> 预计阅读：25 分钟 | 前置知识：了解世界模型基础概念（见[什么是世界模型](../01-基础概念/01-什么是世界模型.md)）
>
> 本文收录 13 篇世界模型关键论文。每张卡片由「标题行 → 徽章行 → 一句话总结 → 核心贡献 / 方法要点 / 结果亮点」组成，题名取 arXiv API 返回的完整题名

---

## 论文卡片格式

每张卡片的字段固定为：

- **标题行**：`**标题**：*英文完整题名*`。题名取 arXiv API 返回的完整题名——系统名（如 ANWM）是简称，不进标题位。
- **徽章行**：arXiv 号、代码仓、项目页三类链接，徽章颜色分别是 `b31b1b` / `181717` / `0A66C2`；行尾是年份、会议或期刊、机构与推荐度。
- **一句话总结 / 核心贡献 / 方法要点 / 结果亮点**：按论文实际给出的范围写。论文没有的部分留空，不补。

同一篇论文可能在两篇合集里各出现一次（例如 MotionScape 既是世界模型论文也是基准），这是交叉列出，不是重复录入。

---

## 1. ANWM — 航空导航世界模型

**一句话总结**：用FFP模块为无人机提供几何先验，实现长距离视觉预测

**标题**：*Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space*

[![arXiv](https://img.shields.io/badge/arXiv-2512.21887-b31b1b.svg)](https://arxiv.org/abs/2512.21887) · 2025 · 推荐度 ★★★

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

**标题**：*AirScape: An Aerial Generative World Model with Motion Controllability*

[![arXiv](https://img.shields.io/badge/arXiv-2507.08885-b31b1b.svg)](https://arxiv.org/abs/2507.08885) [![Website](https://img.shields.io/badge/Website-embodiedcity.github.io-0A66C2.svg)](https://embodiedcity.github.io/AirScape) · ACM MM 2025 · 2025 · 推荐度 ★★☆

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

**标题**：*FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video*

[![arXiv](https://img.shields.io/badge/arXiv-2509.14082-b31b1b.svg)](https://arxiv.org/abs/2509.14082) · 2025 · 推荐度 ★★★

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

**一句话总结**：按运动强度分层的无人机视频基准，228 段 / 62,700 帧，配天气、光照、场景与相机运动标注

**标题**：*MotionScape: A Motion-Stratified UAV Video Benchmark for World Modeling and Future Video Generation*

[![arXiv](https://img.shields.io/badge/arXiv-2604.07991-b31b1b.svg)](https://arxiv.org/abs/2604.07991) [![GitHub](https://img.shields.io/badge/GitHub-MotionScape-181717.svg?logo=github)](https://github.com/Thelegendzz/MotionScape) · 2026 · 推荐度 ★★★

**核心贡献**：
- 真实无人机视角视频基准：**228 段高分辨率视频、共 62,700 帧**（约 35 分钟）
- 标注天气与光照条件、场景环境、相机视角运动
- 按相机运动强度分层（低/中/高运动层），供"运动分层"式评测

**方法要点**：
- 人工选源
- CLIP 辅助筛查
- 人工校验（三步，非自动化管线）

**结果亮点**：对齐的标注有效提升现有世界模型模拟复杂 3D 动态的能力

---

## 5. AeroVerse — 航空航天世界模型基准

**一句话总结**：最全面的无人机世界模型基准套件，定义五大下游任务

**标题**：*AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models*

[![arXiv](https://img.shields.io/badge/arXiv-2408.15511-b31b1b.svg)](https://arxiv.org/abs/2408.15511) · 2024 · 推荐度 ★★☆

**核心贡献**：
- 定义"航空航天世界模型"概念
- 五大下游任务基准
- 预训练和微调数据集

**五大任务**：
1. 场景认知（aerospace embodied scene awareness）
2. 空间推理（spatial reasoning）
3. 导航探索（navigational exploration）
4. 任务规划（task planning）
5. 运动决策（motion decision）

---

## 6. Dream to Fly — DreamerV3用于无人机竞速

**一句话总结**：首次将DreamerV3用于无人机竞速，涌现感知意识行为

**标题**：*Dream to fly: Model-based reinforcement learning for vision-based drone flight*

[![arXiv](https://img.shields.io/badge/arXiv-2501.14377-b31b1b.svg)](https://arxiv.org/abs/2501.14377) · ICRA 2026 · UZH, RPG · 推荐度 ★★★

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

**标题**：*HDVIO: Improving Localization and Disturbance Estimation with Hybrid Dynamics VIO*

[![arXiv](https://img.shields.io/badge/arXiv-2306.11429-b31b1b.svg)](https://arxiv.org/abs/2306.11429) · 2023 · UZH, RPG · 推荐度 ★★☆

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

**标题**：*Physics-guided Learning-based Adaptive Control on the SE(3) Manifold*

[![arXiv](https://img.shields.io/badge/arXiv-2201.04339-b31b1b.svg)](https://arxiv.org/abs/2201.04339) · 2022 · 推荐度 ★★☆

**核心贡献**：
- 物理启发的神经ODE
- SE(3)流形上的控制
- 保持物理结构的学习动力学

---

## 9. Learning MPC for Quadrotors

**一句话总结**：利用过去成功任务迭代改进四旋翼控制性能

**标题**：*Learning Model Predictive Control for Quadrotors*

[![arXiv](https://img.shields.io/badge/arXiv-2202.07716-b31b1b.svg)](https://arxiv.org/abs/2202.07716) · 2022 · 推荐度 ★☆☆

**核心贡献**：
- 学习MPC方法
- 利用历史数据改进性能
- 尊重系统动力学约束

---

## 10. Wireless Dreamer — 无线边缘智能世界模型

**一句话总结**：世界模型用于无线边缘智能优化，含天气感知无人机轨迹规划

**标题**：*World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks*

[![arXiv](https://img.shields.io/badge/arXiv-2506.00417-b31b1b.svg)](https://arxiv.org/abs/2506.00417) · 2025 · 推荐度 ★★☆

**核心贡献**：
- "Wireless Dreamer"框架
- 世界模型用于边缘智能
- 天气感知无人机轨迹规划案例

---

## 11. Contrastive World Model for Drone Navigation

**一句话总结**：用对比学习在世界模型框架中学习无人机导航视觉表示

**标题**：*Learning visual representation for autonomous drone navigation via a contrastive world model*

 · IEEE Transactions · 2023 · 推荐度 ★★☆

**核心贡献**：
- 对比学习+世界模型
- 学习鲁棒的航空视角场景理解
- 用于无人机导航

---

## 12. Nonlinear System ID Nano-drone Benchmark

**一句话总结**：Crazyflie 2.1纳米四旋翼的75k真实样本系统辨识基准

**标题**：*Nonlinear System Identification Nano-drone Benchmark*

[![arXiv](https://img.shields.io/badge/arXiv-2512.14450-b31b1b.svg)](https://arxiv.org/abs/2512.14450) · 2025 · 推荐度 ★☆☆

**核心贡献**：
- 75k真实世界样本
- Crazyflie 2.1平台
- 系统辨识和动力学模型学习

---

## 13. PINN for Multirotor Slung Load

**一句话总结**：用物理信息神经网络学习多旋翼-悬挂负载系统的端到端模型

**标题**：*Physics-Informed Neural Network for Multirotor Slung Load Systems Modeling*

[![arXiv](https://img.shields.io/badge/arXiv-2405.09428-b31b1b.svg)](https://arxiv.org/abs/2405.09428) · 2024 · 推荐度 ★☆☆

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
| ★★★ | MotionScape | 运动分层的无人机视频基准 |
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
- [完整论文列表](../../references/paper-list.md) — 195 篇论文分类汇总

## 思考题

1. **选型判断**：数据稀缺的无人机竞速项目里，FlightDiffusion 和 Dream to Fly 各解决什么问题？应该先上哪一个？

2. **批判思考**：FlightDiffusion 用 p=0.541 说明"仿真和真实无显著差异"。这个结论最多能支撑到什么程度？

3. **方法迁移**：把 ANWM 的 FFP 和 HDVIO 的混合动力学拼到一起做导航，拼接的接口在哪、风险在哪？

4. **归纳**：HDVIO、SE(3) 物理引导学习、Learning MPC、PINN 这四张卡片严格说都不是"世界模型"。它们被收进本文的共同价值是什么？

5. **选型标准**：让你给"无人机世界模型"的一个新场景挑论文，你会用哪几条标准？用本文的卡片检验一下。

<details><summary>参考答案</summary>

1. 两者定位不同。Dream to Fly 解决的是"没有足够真实交互，怎么学出能飞的控制策略"：用 DreamerV3 在隐空间做想象训练，从像素直接学、无需手工特征和额外奖励项设计，在真实四旋翼上飞到 9m/s，样本效率远超 PPO、SAC。FlightDiffusion 解决的是"数据不够怎么造"：用条件扩散从单帧生成多样的 FPV 视频轨迹，并同时生成对应的 action space。要先把策略跑通，先上 Dream to Fly；FlightDiffusion 适合作为数据扩容手段叠加，它报告位置误差 0.25m、成功率约 62%。

2. 最多支撑"在作者设定的检验下、在作者设定的指标上，未拒绝仿真与真实同分布的原假设"。卡片只给了一个 p 值，没有说明检验的是位置误差还是成功率、样本量多少、真实飞行做了多少架次，因此不能推广成"仿真数据可以替代真实飞行"。同一张卡片里并列的 0.25m 位置误差和约 62% 成功率也只是点估计，没有置信区间；要支撑替代性结论，至少还需要跨场地、跨机型的重复实验。

3. 接口在"相机位姿"这一环。FFP 的做法是把过去帧投影到未来相机视角，为长距离视觉生成提供几何一致性先验；HDVIO 提供的是四旋翼的定位与扰动估计（点质量车辆模型加学习残差，还学习电机动力学），正好可以充当 FFP 投影所需的位姿来源。风险是误差互相注入：HDVIO 的定位漂移会让投影错位，进而污染生成结果；反过来若把生成帧当作观测回灌进状态估计，闭环会更不稳定。卡片里两者没有联合评测，这套拼法属于待验证。

4. 共同价值是示范"物理先验怎么和学习组件结合"。HDVIO 用点质量车辆模型打底，让网络只学解析建模不了的残余空气动力学效应；SE(3) 那张把哈密顿方程的结构注入神经 ODE，在 SE(3) 流形上做控制；Learning MPC 用过去成功任务迭代改进并显式尊重动力学约束；PINN 用物理约束保证多旋翼-悬挂负载这类强耦合系统的合理性。四者的共同点——物理负责保底、学习负责补残差——正是无人机世界模型抑制长时漂移的关键手段，ANWM 的 FFP 是同一思路的视觉版本。

5. 我会用三条标准，本文的推荐度排序基本也能用它们解释。第一，是否面向航空平台且做的是视觉或动力学建模：ANWM、Dream to Fly、FlightDiffusion、MotionScape 都符合，故列 ★★★；HDVIO、SE(3)、Learning MPC、PINN 虽同为四旋翼，但目标是控制与状态估计，不是世界模型。第二，有没有量化结果：FlightDiffusion 给了 0.25m、p=0.541、约 62%，Dream to Fly 给了 9m/s 和与 PPO/SAC 的样本效率对比，而 AirScape、Wireless Dreamer、对比学习世界模型等卡片只有定性描述，整体停在 ★★☆ 一档。第三，有没有可复用的数据或代码：MotionScape 挂出了 GitHub 仓库，而本文多数卡片既无代码链接也无 arXiv 信息（如 AirScape、对比学习世界模型），按这一条筛，它们只能作为思路参考。

</details>
