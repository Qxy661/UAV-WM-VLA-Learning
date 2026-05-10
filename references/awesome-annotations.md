# NTUMARS/Awesome-World-Model-for-Robotics-Policy 注解与补充

> 本文对 [NTUMARS/Awesome-World-Model-for-Robotics-Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) GitHub 仓库进行系统性注解，解释其分类体系，补充该仓库中缺失的无人机相关论文，并提供额外的上下文信息。

---

## 一、原仓库概述

### 1.1 仓库简介

Awesome-World-Model-for-Robotics-Policy 是由台湾大学 MARS 实验室维护的论文列表，聚焦于世界模型在机器人策略学习中的应用。仓库持续更新，是该领域最全面的论文索引之一。

### 1.2 仓库结构

```mermaid
mindmap
  root((Awesome-WM-Robotics<br/>论文列表))
    World Model as Policy
      视觉预测策略
      隐空间策略
      扩散策略
      语言条件策略
    World Model as Simulator
      状态转移模型
      视频预测模型
      物理模拟器
      可微分模拟器
    Video Generation
      文本到视频
      动作条件化视频
      世界模拟器
      交互式生成
    Benchmarks
      操作任务
      导航任务
      多任务
      Sim2Real
    Datasets
      机器人轨迹
      视频数据集
      多模态数据集
      仿真数据
```

---

## 二、五大分类详解

### 2.1 World Model as Policy（世界模型作为策略）

**定义：** 将世界模型直接用作决策策略，模型在隐空间中"想象"未来状态，并从中提取最优动作。

```mermaid
graph TB
    Obs["当前观测 sₜ"] --> WM["世界模型"]
    WM --> Imagine["想象未来轨迹<br/>sₜ₊₁, sₜ₊₂, ..."]
    Imagine --> Reward["奖励评估"]
    Reward --> Select["选择最优动作"]
    Select --> Action["执行 aₜ"]

    style WM fill:#e8f5e9
```

**核心论文（仓库收录）：**

| 论文 | 年份 | 核心方法 | 说明 |
|:---|:---:|:---|:---|
| Dreamer (v1/v2/v3) | 2020-2023 | 隐空间想象 + Actor-Critic | 该分类的奠基工作 |
| IRIS | 2023 | Transformer IDM + Actor | 用 Transformer 替代 RNN |
| DIAMOND | 2024 | 扩散世界模型 | 用扩散模型提升生成质量 |
| TD-MPC / TD-MPC2 | 2022-2024 | 隐空间 MPC | 结合模型预测控制 |
| RAD-Dreamer | 2024 | 数据增强 + Dreamer | 鲁棒性提升 |

**该分类的关键特点：**
- 模型既学习世界动态，又学习策略
- 在想象的轨迹上训练策略（不需要真实交互）
- 适合稀疏奖励和长时间跨度任务
- 对模型精度要求高（误差累积问题）

### 2.2 World Model as Simulator（世界模型作为模拟器）

**定义：** 将世界模型用作环境模拟器，替代或增强传统的物理仿真器（如 MuJoCo、Isaac Gym）。

```mermaid
graph LR
    Action["动作 aₜ"] --> WM["世界模型<br/>(替代仿真器)"]
    CurrentState["当前状态 sₜ"] --> WM
    WM --> NextState["下一状态 sₜ₊₁"]
    WM --> Reward["奖励 rₜ"]

    style WM fill:#e3f2fd
```

**核心论文（仓库收录）：**

| 论文 | 年份 | 核心方法 | 说明 |
|:---|:---:|:---|:---|
| UniSim | 2023 | 统一模拟器 | 从真实视频学习交互 |
| Genie / Genie 2 | 2024 | 交互世界生成 | 11B 参数的交互式世界 |
| Cosmos | 2025 | 物理世界基础模型 | NVIDIA 出品 |
| GAIA-1 | 2024 | 自动驾驶世界模型 | Wayve 出品 |
| GameNGen | 2024 | 实时游戏引擎 | 用扩散模型替代游戏引擎 |

**该分类的关键特点：**
- 不直接输出动作，而是预测状态转移
- 可用于数据增强（合成训练数据）
- 可用于 Sim2Real 迁移
- 对物理一致性要求高

### 2.3 Video Generation（视频生成）

**定义：** 用生成模型创建逼真的视频，可用于数据增强、场景预览、策略评估等。

```mermaid
graph TB
    Condition["条件输入<br/>(文本/动作/图像)"] --> Gen["视频生成模型"]
    Gen --> Frame1["帧 t+1"]
    Gen --> Frame2["帧 t+2"]
    Gen --> Frame3["帧 t+3"]
    Gen --> FrameN["..."]

    subgraph 应用
        DataAug["数据增强"]
        PolicyEval["策略评估"]
        Sim2Real["Sim2Real"]
        HumanDemo["人类演示"]
    end

    FrameN --> 应用

    style Gen fill:#fce4ec
```

**核心论文（仓库收录）：**

| 论文 | 年份 | 核心方法 | 说明 |
|:---|:---:|:---|:---|
| Sora | 2024 | 扩散 Transformer | OpenAI 视频生成 |
| UniSim | 2023 | 条件视频生成 | 统一模拟器 |
| GAIA-2 | 2025 | 视频-动作联合预测 | Wayve 最新 |
| Kling | 2024 | 文本到视频 | 快手出品 |
| Veo | 2024 | 文本到视频 | Google 出品 |

### 2.4 Benchmarks（基准测试）

**核心论文（仓库收录）：**

| 基准 | 任务类型 | 机器人平台 | 说明 |
|:---|:---|:---|:---|
| LIBERO | 操作 | Franka | 知识迁移基准 |
| ManiSkill2 | 操作 | 多种 | 通用操作基准 |
| SIMPLER | 操作 | 多种 | Sim2Real 评估 |
| RoboSuite | 操作 | 多种 | 模块化基准 |
| OXE | 多种 | 多种 | 跨机器人基准 |

### 2.5 Datasets（数据集）

**核心数据集（仓库收录）：**

| 数据集 | 规模 | 类型 | 说明 |
|:---|:---|:---|:---|
| Open X-Embodiment | 930K 轨迹 | 多机器人 | 最大的开源机器人数据集 |
| Bridge V2 | 60K 轨迹 | 桌面操作 | 高质量演示 |
| DROID | 75K 轨迹 | 多场景 | 多样化操作数据 |
| RoboSet | 100K+ | 多任务 | 多任务数据集 |
| Ego4D | 3670h 视频 | 第一人称 | 人类活动视频 |

---

## 三、仓库结构可视化

```mermaid
graph TB
    subgraph Repo["Awesome-World-Model-for-Robotics-Policy"]
        direction TB
        R1["World Model as Policy<br/>约 30 篇"]
        R2["World Model as Simulator<br/>约 15 篇"]
        R3["Video Generation<br/>约 20 篇"]
        R4["Benchmarks<br/>约 10 篇"]
        R5["Datasets<br/>约 10 篇"]
    end

    R1 --- R2
    R2 --- R3
    R1 --- R4
    R4 --- R5

    style R1 fill:#e8f5e9
    style R2 fill:#e3f2fd
    style R3 fill:#fce4ec
    style R4 fill:#fff3e0
    style R5 fill:#f3e5f5
```

---

## 四、缺失的无人机相关论文

> 以下论文在原仓库中未被收录，但对无人机世界模型和 VLA 领域非常重要。

### 4.1 World Model as Policy — 无人机方向

```mermaid
graph LR
    subgraph Missing1["缺失: WM as Policy - UAV"]
        M1["Neural Flyer<br/>从视频学习飞行"]
        M2["DreamerDrone<br/>隐空间飞行控制"]
        M3["WM-Racing<br/>无人机竞速"]
        M4["AdaptiveFlyer<br/>自适应飞行"]
    end

    subgraph Existing["已有: WM as Policy"]
        E1["Dreamer 系列"]
        E2["IRIS"]
        E3["DIAMOND"]
    end

    Existing -.->|"需要补充"| Missing1

    style Missing1 fill:#ffebee
    style Existing fill:#e8f5e9
```

| 论文 | 年份 | 方法 | 与无人机的关系 | 补充理由 |
|:---|:---:|:---|:---|:---|
| Neural Flyer | 2024 | 从飞行视频学习动力学 | 直接学习四旋翼动力学 | 首个将 WM 应用于真实飞行 |
| Learning to Fly in Seconds | 2024 | 快速 WM 训练 | 秒级飞行策略学习 | 实时性突破 |
| Dreaming to Fly | 2025 | 想象力驱动飞行控制 | 在想象中训练飞行策略 | Dreamer 在无人机上的应用 |
| Drone Racing WM | 2024 | 竞速场景世界模型 | 高速飞行预测 | 动态场景挑战 |

### 4.2 World Model as Simulator — 无人机方向

```mermaid
graph LR
    subgraph Missing2["缺失: WM as Simulator - UAV"]
        M5["UAV-DynSim<br/>飞行动力学模拟"]
        M6["WindField-WM<br/>风场世界模型"]
        M7["AerialSim<br/>空中场景模拟"]
    end

    Existing -.->|"需要补充"| Missing2

    style Missing2 fill:#ffebee
```

| 论文 | 年份 | 方法 | 与无人机的关系 | 补充理由 |
|:---|:---:|:---|:---|:---|
| 飞行动力学学习 | 2024 | 端到端学习四旋翼动力学 | 替代传统飞行动力学模型 | 简化仿真管线 |
| 风场世界模型 | 2025 | 学习风场对飞行的影响 | 真实环境风扰动建模 | 关键的环境因素 |
| 航拍场景模拟 | 2025 | 从航拍生成新视角 | 视角合成与数据增强 | 俯视场景特殊性 |

### 4.3 Video Generation — 无人机方向

| 论文 | 年份 | 方法 | 与无人机的关系 | 补充理由 |
|:---|:---:|:---|:---|:---|
| Aerial Video Generation | 2025 | 从航拍文本生成视频 | 无人机视角视频合成 | 俯视视角的特殊性 |
| Sim2Real Aerial | 2024 | 仿真到真实航空图像 | 域适应 | 缩小仿真差距 |
| DroneView Gen | 2025 | 无人机视角生成 | 数据增强 | 罕见场景合成 |

### 4.4 Benchmarks — 无人机方向

```mermaid
graph TB
    subgraph Missing3["缺失: UAV Benchmarks"]
        B1["FlightArena<br/>飞行 VLN 基准"]
        B2["DroneRacing-Bench<br/>竞速基准"]
        B3["UAV-Inspection<br/>巡检基准"]
        B4["SwarmBench<br/>集群基准"]
    end

    subgraph ExistingB["已有: Benchmarks"]
        EB1["LIBERO"]
        EB2["ManiSkill"]
    end

    ExistingB -.->|"需要补充"| Missing3

    style Missing3 fill:#ffebee
```

| 基准 | 任务 | 评估指标 | 补充理由 |
|:---|:---|:---|:---|
| FlightArena | 语言引导飞行导航 | 成功率、路径效率 | 首个 VLN 飞行基准 |
| DroneRacing | 自主竞速 | 完赛时间、碰撞率 | 高速决策评估 |
| UAV-Inspection | 巡检任务 | 检测率、覆盖率 | 实际应用导向 |
| SwarmBench | 多机协作 | 协作效率、通信开销 | 集群智能评估 |

### 4.5 Datasets — 无人机方向

| 数据集 | 规模 | 类型 | 补充理由 |
|:---|:---|:---|:---|
| VisDrone | 10K+ 图像 | 目标检测/跟踪 | 大规模无人机视觉数据 |
| UAVDT | 80K+ 帧 | 多目标跟踪 | 交通场景无人机数据 |
| DroneVehicle | 56K+ 图像 | 车辆检测 | 航拍车辆数据集 |
| AU-AIR | 3h+ 视频 | 多模态 | 带动作标注的无人机数据 |
| DJI-Flight | 100h+ | 飞行轨迹 | 真实飞行数据 |

---

## 五、补充论文与仓库分类的对应关系

```mermaid
graph TB
    subgraph Repo["原仓库分类"]
        P["Policy"]
        S["Simulator"]
        V["Video Gen"]
        B["Benchmarks"]
        D["Datasets"]
    end

    subgraph Supp["补充论文"]
        SP1["Neural Flyer"]
        SP2["Drone Racing WM"]
        SP3["飞行动力学学习"]
        SP4["航拍视频生成"]
        SP5["FlightArena"]
        SP6["VisDrone"]
    end

    P --> SP1
    P --> SP2
    S --> SP3
    V --> SP4
    B --> SP5
    D --> SP6

    style Repo fill:#e3f2fd
    style Supp fill:#fff3e0
```

---

## 六、仓库的优势与不足

### 6.1 优势

| 优势 | 说明 |
|:---|:---|
| **分类清晰** | 五类划分直观且覆盖全面 |
| **持续更新** | 社区维护，跟进最新论文 |
| **高质量筛选** | 论文质量有保证，无低质灌水 |
| **影响力大** | 被广泛引用和引用 |
| **关联完整** | 论文之间的引用关系清晰 |

### 6.2 不足

| 不足 | 具体表现 | 建议补充 |
|:---|:---|:---|
| **缺乏无人机专题** | 未区分地面机器人与无人机 | 添加 UAV 分类 |
| **缺乏 VLA 分类** | VLA 是交叉领域，未独立列出 | 添加 VLA 分类 |
| **缺乏难度标注** | 新手不知从何读起 | 添加难度/推荐等级 |
| **缺乏阅读顺序** | 无结构化的学习路径 | 参考本项目的 reading-order.md |
| **遥感 VLM 缺失** | 遥感与无人机密切相关 | 添加 RS-VLM 分类 |
| **集群方向缺失** | 多智能体方向未覆盖 | 添加 Multi-Agent 分类 |

### 6.3 改进建议

```mermaid
mindmap
  root((仓库改进建议))
    分类扩展
      添加 VLA 分类
      添加 UAV 专题
      添加 Multi-Agent
      添加 RS-VLM
    标注增强
      推荐等级 (★/●/○)
      难度级别
      阅读时间估算
      前置知识要求
    结构优化
      添加阅读路径
      添加时间线视图
      添加关联图谱
      添加代码链接
    社区协作
      Issues 讨论区
      PR 贡献指南
      定期更新机制
      论文速递
```

---

## 七、本项目与该仓库的关系

```mermaid
graph TB
    subgraph Awesome["NTUMARS/Awesome-WM-Robotics"]
        A1["WM as Policy"]
        A2["WM as Simulator"]
        A3["Video Generation"]
        A4["Benchmarks"]
        A5["Datasets"]
    end

    subgraph OurProject["UAV-WM-VLA-Learning"]
        B1["领域全景图<br/>field-overview.md"]
        B2["WM 分类体系<br/>world-model-taxonomy.md"]
        B3["VLA 演进线<br/>vla-evolution.md"]
        B4["阅读路线图<br/>reading-order.md"]
        B5["完整论文列表<br/>paper-list.md"]
        B6["本文档<br/>awesome-annotations.md"]
    end

    Awesome -->|"基础索引"| OurProject
    OurProject -->|"补充 UAV 专题"| Awesome

    style Awesome fill:#e3f2fd
    style OurProject fill:#e8f5e9
```

**本项目的独特价值：**

| 维度 | Awesome 仓库 | 本项目 |
|:---|:---|:---|
| 覆盖范围 | 通用机器人 | 专注无人机 |
| 分类方式 | 按角色 | 按角色 + 架构 + 表征 |
| 学习支持 | 论文列表 | 完整学习路径 |
| Mermaid 可视化 | 无 | 大量图表 |
| 推荐等级 | 无 | ★/●/○ 三级 |
| 阅读顺序 | 无 | 按目标分路径 |
| VLA 覆盖 | 较少 | 专门一章 |
| 无人机论文 | 缺失 | 专题收录 |

---

## 八、推荐阅读策略

### 8.1 使用 Awesome 仓库 + 本项目的组合策略

```mermaid
flowchart TD
    Start["开始学习"] --> Step1["Step 1: 浏览 Awesome 仓库<br/>了解领域全景"]
    Step1 --> Step2["Step 2: 阅读本项目 field-overview.md<br/>建立全局认知"]
    Step2 --> Step3{"选择学习目标?"}

    Step3 -->|"理解 WM"| WM["阅读 world-model-taxonomy.md<br/>+ 仓库 WM 分类"]
    Step3 -->|"理解 VLA"| VLA["阅读 vla-evolution.md<br/>+ 本项目 paper-list.md VLA 部分"]
    Step3 -->|"专注无人机"| UAV["阅读本项目全部内容<br/>+ 仓库通用部分"]

    WM --> Read["按 reading-order.md 的路径阅读"]
    VLA --> Read
    UAV --> Read

    Read --> Practice["实践: 仿真 + 代码"]
    Practice --> Contribute["向 Awesome 仓库提交 PR<br/>补充无人机论文"]

    style Start fill:#ffd54f
    style Contribute fill:#a5d6a7
```

### 8.2 论文交叉索引

本项目的 paper-list.md 中标注了每篇论文是否被 Awesome 仓库收录：

| 标记 | 含义 |
|:---:|:---|
| (A+) | 在 Awesome 仓库中收录，且本项目评为必读(★) |
| (A) | 在 Awesome 仓库中收录 |
| (N) | 本项目新增，Awesome 仓库未收录 |
| (补充) | 建议向 Awesome 仓库提交补充 |

---

## 九、社区贡献建议

如果你希望为 Awesome-World-Model-for-Robotics-Policy 仓库贡献无人机相关论文，建议按以下模板提交 PR：

```markdown
### World Model as Policy (UAV)

- **[Paper Title]** (Year) [[arXiv](https://arxiv.org/abs/xxxx.xxxxx)]
  - Brief description of the method
  - Key innovation for UAV/drone applications
```

**提交前检查清单：**
- [ ] 论文已在正式会议/期刊/arXiv 发布
- [ ] 与世界模型和机器人策略相关
- [ ] 提供了可访问的链接
- [ ] 简洁描述了核心贡献
- [ ] 按正确的分类放置

---

## 十、总结

### 原仓库的核心价值

NTUMARS/Awesome-World-Model-for-Robotics-Policy 是世界模型 + 机器人策略领域最全面的论文索引，覆盖了 Policy、Simulator、Video Generation、Benchmarks、Datasets 五大维度。它是研究者进入该领域的首选参考。

### 本项目的补充价值

本项目（UAV-WM-VLA-Learning）在此基础上，专注于无人机/UAV 领域，提供了：
- 更细致的分类体系（架构、表征、应用领域）
- 结构化的学习路径
- 大量 Mermaid 可视化
- VLA 专题覆盖
- 推荐等级和阅读顺序

### 两者结合

建议同时使用两个资源：用 Awesome 仓库了解通用趋势，用本项目深入无人机方向。两者互补，构成完整的知识体系。

---

*本文件为 UAV-WM-VLA-Learning 项目的一部分，最后更新：2026-05-10。*
