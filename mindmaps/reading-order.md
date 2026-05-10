# 论文阅读路线图

> 本文为不同学习目标的读者提供结构化的论文阅读建议。按难度分级、按主题分组，配合前置知识依赖关系，帮助你高效地进入世界模型与 VLA 领域。

---

## 一、总体阅读路线图

```mermaid
mindmap
  root((阅读路线))
    Level 0 基础
      Transformer
      多模态基础
      机器人学基础
    Level 1 入门
      VLM 入门
      World Model 概念
      VLA 概念
    Level 2 核心
      经典 WM 论文
      经典 VLA 论文
      开源实现
    Level 3 进阶
      前沿 WM 架构
      VLA 改进
      无人机适配
    Level 4 专题
      无人机 WM
      无人机 VLA
      集群与安全
```

---

## 二、按目标的阅读路径

### 路径 A：理解基础概念（2-3 周）

```mermaid
graph TD
    A1["1. Attention Is All You Need<br/>(Vaswani 2017)"] --> A2["2. ViT: An Image is Worth 16x16 Words<br/>(Dosovitskiy 2020)"]
    A2 --> A3["3. CLIP<br/>(Radford 2021)"]
    A3 --> A4["4. LLaVA<br/>(Liu 2023)"]
    A4 --> A5["5. RT-2<br/>(Brohan 2023)"]
    A5 --> A6["6. DreamerV3<br/>(Hafner 2023)"]
    A6 --> A7["7. OpenVLA<br/>(Kim 2024)"]

    style A1 fill:#e8f5e9
    style A3 fill:#e3f2fd
    style A5 fill:#fff3e0
    style A7 fill:#fce4ec
```

| 序号 | 论文 | 年份 | 阅读目的 | 预计时间 |
|:---:|:---|:---:|:---|:---:|
| 1 | Attention Is All You Need | 2017 | 理解 Transformer 基础 | 3h |
| 2 | ViT | 2020 | 视觉 Transformer 入门 | 2h |
| 3 | CLIP | 2021 | 视觉-语言对齐 | 2h |
| 4 | LLaVA | 2023 | 理解 VLM 基本架构 | 3h |
| 5 | RT-2 | 2023 | VLA 的开创性工作 | 3h |
| 6 | DreamerV3 | 2023 | 世界模型经典方法 | 4h |
| 7 | OpenVLA | 2024 | 开源 VLA 基线 | 3h |

### 路径 B：深入世界模型（3-4 周）

```mermaid
graph TD
    B0["前置: 路径 A 的 1-6"] --> B1

    B1["8. Ha & Schmidhuber 2018<br/>World Models"] --> B2["9. Dreamer (v1)<br/>(Hafner 2020)"]
    B2 --> B3["10. DreamerV2<br/>(Hafner 2021)"]
    B3 --> B4["11. DreamerV3<br/>(Hafner 2023)"]

    B4 --> B5["12. UniSim<br/>(Yang 2023)"]
    B4 --> B6["13. IRIS<br/>(Micheli 2023)"]

    B5 --> B7["14. Genie<br/>(Bruce 2024)"]
    B6 --> B8["15. DIAMOND<br/>(Alonso 2024)"]

    B7 --> B9["16. Cosmos<br/>(NVIDIA 2025)"]
    B8 --> B10["17. GAIA-2<br/>(Wayve 2025)"]

    B9 --> B11["18. 无人机 WM 方向<br/>(选读)"]
    B10 --> B11

    style B0 fill:#ffcdd2
    style B11 fill:#c8e6c9
```

**世界模型必读清单：**

| 序号 | 论文 | 核心贡献 | 难度 |
|:---:|:---|:---|:---:|
| 8 | World Models (Ha 2018) | 首个神经网络世界模型 | ★★ |
| 9 | Dreamer (2020) | 隐空间 + RL | ★★★ |
| 10 | DreamerV2 (2021) | 离散隐变量 | ★★★ |
| 11 | DreamerV3 (2023) | 通用隐空间世界模型 | ★★★ |
| 12 | UniSim (2023) | 统一模拟器概念 | ★★★ |
| 13 | IRIS (2023) | Transformer + IDM | ★★★★ |
| 14 | Genie (2024) | 11B 交互世界模型 | ★★★★ |
| 15 | DIAMOND (2024) | 扩散世界模型 | ★★★★ |
| 16 | Cosmos (2025) | 物理世界基础模型 | ★★★★★ |
| 17 | GAIA-2 (2025) | 自动驾驶世界模型 | ★★★★★ |

### 路径 C：深入 VLA（3-4 周）

```mermaid
graph TD
    C0["前置: 路径 A 的 1-7"] --> C1

    C1["19. PaLM-E<br/>(Driess 2023)"] --> C2["20. RT-2<br/>(Brohan 2023)"]
    C2 --> C3["21. Octo<br/>(Team 2024)"]
    C3 --> C4["22. OpenVLA<br/>(Kim 2024)"]

    C4 --> C5["23. RoboVLM<br/>(Jin 2024)"]
    C4 --> C6["24. π₀<br/>(Black 2024)"]

    C5 --> C7["25. π₀.₅<br/>(Black 2025)"]
    C6 --> C7

    C7 --> C8["26. VLA-AN<br/>(2025)"]
    C7 --> C9["27. CognitiveDrone<br/>(2025)"]
    C7 --> C10["28. UAV-TrackVLA<br/>(2025)"]

    style C0 fill:#ffcdd2
    style C8 fill:#c8e6c9
    style C9 fill:#c8e6c9
    style C10 fill:#c8e6c9
```

**VLA 必读清单：**

| 序号 | 论文 | 核心贡献 | 难度 |
|:---:|:---|:---|:---:|
| 19 | PaLM-E (2023) | 首个具身多模态大模型 | ★★★ |
| 20 | RT-2 (2023) | 动作 token 化 | ★★★ |
| 21 | Octo (2024) | 开源通用机器人策略 | ★★★ |
| 22 | OpenVLA (2024) | 开源 VLA 基线 | ★★★ |
| 23 | RoboVLM (2024) | VLA 架构系统研究 | ★★★★ |
| 24 | π₀ (2024) | 流匹配 VLA | ★★★★ |
| 25 | π₀.₅ (2025) | 世界模型增强 VLA | ★★★★★ |
| 26 | VLA-AN (2025) | 无人机专用 VLA | ★★★★ |
| 27 | CognitiveDrone (2025) | 认知无人机 | ★★★★★ |
| 28 | UAV-TrackVLA (2025) | 跟踪任务 VLA | ★★★★ |

### 路径 D：实用项目导向（4-8 周）

```mermaid
graph TD
    D0["前置基础"] --> D1

    D1["选定仿真环境<br/>Habitat / AirSim / Gazebo"] --> D2["学习 RL 框架<br/>Stable-Baselines3 / CleanRL"]
    D2 --> D3["复现 DreamerV3<br/>在简单环境"]
    D3 --> D4["复现 OpenVLA<br/>或 Octo"]
    D4 --> D5["迁移至无人机环境"]

    D5 --> D6["方向 A: 微调 VLA"]
    D5 --> D7["方向 B: 构建 WM"]
    D5 --> D8["方向 C: 端到端"]

    D6 --> D9["部署到边缘设备<br/>Jetson / Khadas"]
    D7 --> D9
    D8 --> D9

    style D0 fill:#ffcdd2
    style D9 fill:#c8e6c9
```

**实践项目建议：**

| 阶段 | 任务 | 产出 |
|:---:|:---|:---|
| 1 | 搭建仿真环境（AirSim / Gazebo + PX4） | 可运行的无人机仿真 |
| 2 | 训练基础策略（PPO/SAC） | 简单导航策略 |
| 3 | 集成 VLM 做场景理解 | 语言指令 -> 场景描述 |
| 4 | 复现 OpenVLA/π₀ 在仿真中 | VLA 在仿真中工作 |
| 5 | 添加世界模型辅助 | 提升策略性能 |
| 6 | 部署到真实/边缘设备 | 端侧推理演示 |

---

## 三、按主题的详细阅读清单

### 3.1 世界模型主题

```mermaid
graph TB
    subgraph 基础["基础 (必须读)"]
        WM1["Ha 2018: World Models"]
        WM2["Dreamer 系列 (v1/v2/v3)"]
    end

    subgraph 架构["架构创新"]
        WM3["IRIS: IDM + Transformer"]
        WM4["DIAMOND: 扩散 WM"]
        WM5["TD-MPC2: 隐空间 MPC"]
    end

    subgraph 规模["规模化"]
        WM6["Genie: 11B 参数"]
        WM7["Cosmos: 物理世界基础"]
        WM8["GAIA-2: 自动驾驶"]
    end

    subgraph 应用["应用"]
        WM9["UniSim: 统一模拟器"]
        WM10["World Labs: 3D 世界"]
        WM11["GameNGen: 游戏生成"]
    end

    基础 --> 架构
    架构 --> 规模
    架构 --> 应用
```

### 3.2 VLA 主题

```mermaid
graph TB
    subgraph 奠基["奠基 (必须读)"]
        VLA1["RT-2: 动作 token"]
        VLA2["OpenVLA: 开源基线"]
    end

    subgraph 改进["架构改进"]
        VLA3["π₀: 流匹配"]
        VLA4["RoboVLM: 系统研究"]
        VLA5["Octo: 通用策略"]
    end

    subgraph 增强["能力增强"]
        VLA6["π₀.₅: WM 增强"]
        VLA7["OpenVLA-7B: 扩展"]
    end

    subgraph 无人机["无人机 VLA"]
        VLA8["VLA-AN"]
        VLA9["CognitiveDrone"]
        VLA10["UAV-TrackVLA"]
    end

    奠基 --> 改进
    改进 --> 增强
    增强 --> 无人机
```

### 3.3 VLM 主题

```mermaid
graph TB
    subgraph 通用["通用 VLM"]
        VLM1["LLaVA 系列"]
        VLM2["Qwen-VL 系列"]
        VLM3["InternVL 系列"]
    end

    subgraph 遥感["遥感 VLM"]
        VLM4["GeoChat"]
        VLM5["RSGPT"]
        VLM6["LHRS-Bot"]
        VLM7["ChangeChat"]
    end

    subgraph 无人机["无人机 VLM"]
        VLM8["DroneVLM"]
        VLM9["UAV-VL 系列"]
    end

    通用 --> 遥感
    遥感 --> 无人机
```

---

## 四、按难度分级

### Level 0：前置知识（1-2 周）

| 主题 | 推荐资源 | 备注 |
|:---|:---|:---|
| Transformer 原理 | Attention Is All You Need | 必读 |
| 视觉 Transformer | ViT, DeiT | 必读 |
| 多模态基础 | CLIP, SigLIP | 必读 |
| 扩散模型基础 | DDPM, DDIM | 世界模型方向必读 |
| 强化学习基础 | Spinning Up in Deep RL | WM+RL 方向必读 |
| 机器人学基础 | 任意 ROS2 教程 | 实践方向必读 |

### Level 1：入门论文（2-3 周）

```mermaid
graph LR
    L1A["LLaVA<br/>VLM 入门"] --> L1B["RT-2<br/>VLA 入门"]
    L1B --> L1C["DreamerV3<br/>WM 入门"]
    L1C --> L1D["OpenVLA<br/>开源 VLA"]

    style L1A fill:#e8f5e9
    style L1B fill:#e3f2fd
    style L1C fill:#fff3e0
    style L1D fill:#fce4ec
```

### Level 2：核心论文（3-4 周）

| 方向 | 必读论文 | 推荐论文 |
|:---|:---|:---|
| WM 架构 | IRIS, DIAMOND | TD-MPC2, GameNGen |
| WM 规模化 | Genie, Cosmos | GAIA-2, World Labs |
| VLA 架构 | π₀, Octo | RoboVLM |
| VLM 遥感 | GeoChat, RSGPT | LHRS-Bot, ChangeChat |

### Level 3：前沿论文（2-4 周）

| 方向 | 论文 | 难度 |
|:---|:---|:---:|
| WM + VLA 融合 | π₀.₅ | ★★★★★ |
| 无人机 VLA | VLA-AN, CognitiveDrone | ★★★★ |
| 无人机 WM | Neural Flyer 相关 | ★★★★ |
| 3D 世界模型 | World Labs | ★★★★★ |
| 集群 VLA | 多智能体方向 | ★★★★★ |

### Level 4：专题深入（按需）

```mermaid
mindmap
  root((专题方向))
    安全与可靠性
      安全 RL
      约束优化
      形式化验证
      碰撞避免
    高效部署
      模型蒸馏
      量化压缩
      边缘推理
      硬件协同
    多智能体
      集群协作
      通信机制
      分布式决策
    域适应
      Sim2Real
      域随机化
      视觉增广
    长程任务
      层次规划
      记忆机制
      任务分解
```

---

## 五、论文阅读技巧

### 5.1 三遍阅读法

```mermaid
graph TD
    P1["第一遍 (15min)<br/>标题+摘要+图表+结论"] --> |"值得深入?"| P2["第二遍 (1h)<br/>全文通读,标记关键"]
    P2 --> |"核心论文?"| P3["第三遍 (3h+)<br/>逐行精读,复现关键部分"]

    P1 --> |"不相关"| Skip["跳过"]
    P2 --> |"一般论文"| Note["做笔记后停止"]

    style P1 fill:#e8f5e9
    style P2 fill:#fff3e0
    style P3 fill:#fce4ec
```

### 5.2 阅读笔记模板

```markdown
## [论文标题]

**一句话总结:** ...

**核心问题:** 这篇论文试图解决什么问题？

**关键方法:**
- 创新点 1: ...
- 创新点 2: ...

**实验结果:**
- 在 XX 数据集上，XX 指标提升了 XX%
- 与 YY 方法相比，优势在于...

**对无人机方向的启示:**
- 这个方法如何应用到无人机场景？
- 需要做哪些适配？

**相关论文:** [列出相关/引用论文]
```

### 5.3 建立知识图谱

```mermaid
graph TB
    subgraph 你的知识图谱
        WM_K["世界模型知识"]
        VLA_K["VLA 知识"]
        VLM_K["VLM 知识"]
        UAV_K["无人机知识"]
    end

    WM_K -- "世界模型增强VLA" --> VLA_K
    VLM_K -- "VLM作为VLA骨干" --> VLA_K
    UAV_K -- "无人机适配" --> VLA_K
    UAV_K -- "无人机世界模型" --> WM_K
    VLM_K -- "遥感VLM" --> UAV_K
```

---

## 六、推荐学习周历

### 周历总览（16 周计划）

```mermaid
gantt
    title 16 周学习计划
    dateFormat  YYYY-MM-DD
    axisFormat  %m/%d

    section 基础阶段
    Transformer/ViT/CLIP     :a1, 2026-05-11, 5d
    VLM 入门 (LLaVA)         :a2, after a1, 4d
    RL 基础                  :a3, after a1, 4d

    section 世界模型
    Ha 2018 + Dreamer 系列   :b1, after a2, 7d
    IRIS + DIAMOND           :b2, after b1, 5d
    Genie + Cosmos           :b3, after b2, 5d

    section VLA
    RT-2 + OpenVLA           :c1, after a2, 5d
    π₀ + π₀.₅               :c2, after c1, 7d
    无人机 VLA               :c3, after c2, 5d

    section 实践
    仿真环境搭建             :d1, after b1, 7d
    模型复现/微调            :d2, after d1, 14d
    部署演示                 :d3, after d2, 7d
```

### 每周建议

| 周 | 主题 | 论文/任务 | 产出 |
|:---:|:---|:---|:---|
| 1 | 基础 | Transformer, ViT, CLIP | 笔记总结 |
| 2 | VLM 入门 | LLaVA, SigLIP | 概念笔记 |
| 3 | WM 入门 | Ha 2018, DreamerV3 | 对比笔记 |
| 4 | VLA 入门 | RT-2, OpenVLA | 架构分析 |
| 5 | WM 进阶 | IRIS, DIAMOND | 深度笔记 |
| 6 | WM 规模化 | Genie, Cosmos | 趋势分析 |
| 7 | VLA 进阶 | π₀, Octo | 技术分析 |
| 8 | VLA 增强 | π₀.₅, RoboVLM | 创新点总结 |
| 9 | 无人机 VLA | VLA-AN, CognitiveDrone | 应用分析 |
| 10 | 仿真搭建 | AirSim/PX4 环境 | 可运行环境 |
| 11 | WM 复现 | 简单环境 WM | 代码仓库 |
| 12 | VLA 复现 | 微调 OpenVLA/π₀ | 微调模型 |
| 13 | 无人机适配 | 无人机仿真集成 | 初步结果 |
| 14 | 优化调试 | 性能调优 | 改进结果 |
| 15 | 部署准备 | 模型压缩/量化 | 部署就绪 |
| 16 | 演示总结 | 整理成果 | 演示视频 |

---

## 七、学习资源推荐

### 7.1 在线课程

| 课程 | 平台 | 相关度 | 备注 |
|:---|:---|:---:|:---|
| CS231n | Stanford | ★★★ | 视觉基础 |
| CS224n | Stanford | ★★★ | NLP 基础 |
| Deep RL Course | HuggingFace | ★★★★ | RL 实践 |
| Embodied AI Survey | arXiv 综述 | ★★★★★ | 领域全景 |

### 7.2 GitHub 仓库

| 仓库 | 内容 | 推荐度 |
|:---|:---|:---:|
| octo-models/octo | Octo 开源实现 | ★★★★ |
| openvla/openvla | OpenVLA 开源 | ★★★★★ |
| NVlabs/cosmos | NVIDIA Cosmos | ★★★★ |
| worldmodels/worldmodels | 经典 WM 教程 | ★★★ |
| NTUMARS/Awesome-World-Model | WM 论文列表 | ★★★★★ |

### 7.3 综述论文

| 综述 | 主题 | 推荐度 |
|:---|:---|:---:|
| "A Survey on Vision-Language-Action Models" | VLA 综述 | ★★★★★ |
| "World Models for Autonomous Driving" | WM + 自动驾驶 | ★★★★ |
| "Foundation Models for Robotics" | 机器人基础模型 | ★★★★ |
| "A Survey of Embodied AI" | 具身 AI 综述 | ★★★★ |

---

## 八、常见问题

**Q: 我没有机器人学背景，能学 VLA 吗？**
> 可以。VLA 更偏重深度学习和多模态，机器人学基础可以通过 ROS2 教程快速补充。建议先走路径 A 建立基础。

**Q: 我主要关注无人机，需要读所有论文吗？**
> 不需要。建议先走路径 A，然后直接跳到无人机专用论文（路径 C 的后半部分）。世界模型可以选读 DreamerV3 和 Cosmos。

**Q: 理论 vs 实践如何平衡？**
> 建议 40% 理论 + 60% 实践。每读 2-3 篇论文就尝试复现或修改代码。路径 D 的实践导向最适合。

**Q: 需要什么硬件？**
> - 学习理论：普通笔记本即可
> - 模型训练：至少 1x RTX 4090 (24GB)
> - 无人机仿真：RTX 3060 以上 + 16GB RAM
> - 边缘部署：Jetson Orin Nano / Khadas VIM4

---

*本文件为 UAV-WM-VLA-Learning 项目的一部分，最后更新：2026-05-10。*
