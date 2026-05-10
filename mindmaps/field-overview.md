# World Models / VLA / VLM 领域全景图

> 本文以 Mermaid 思维导图与结构化文字，梳理世界模型（World Model）、视觉-语言-动作模型（VLA）、视觉-语言模型（VLM）在无人机/UAV 领域的整体关系。
> 适合初学者建立全局认知，也适合研究者快速定位自己所在的位置。

---

## 一、核心技术族谱总览

```mermaid
mindmap
  root((具身智能<br/>Embodied AI))
    感知 Perception
      视觉 Vision
        单目/双目
        深度估计
        光流
      语言 Language
        指令理解
        场景描述
      多模态融合
        VLM
        CLIP / SigLIP
    认知 Cognition
      世界模型 World Model
        基于像素
        基于隐空间
        视频生成
      预测 Prediction
        状态预测
        动作预测
        不确定性建模
      推理 Reasoning
        空间推理
        因果推理
        长程规划
    决策 Decision
      VLA 模型
        端到端动作
        分层控制
      强化学习
        Model-based
        Model-free
      经典规划
        A* / RRT
        MPC
    执行 Execution
      机器人 Manipulator
      自动驾驶 AV
      无人机 UAV
        多旋翼
        固定翼
        VTOL
```

---

## 二、三大模型之间的关系

```mermaid
graph TB
    subgraph VLM["视觉-语言模型 VLM"]
        LLaVA["LLaVA 系列"]
        QwenVL["Qwen-VL 系列"]
        InternVL["InternVL 系列"]
        GPT4V["GPT-4V / GPT-4o"]
        Cambrian["Cambrian"]
    end

    subgraph WM["世界模型 World Model"]
        UniSim["UniSim (2023)"]
        Genie["Genie (2024)"]
        DIAMOND["DIAMOND (2024)"]
        Cosmos["Cosmos (2025)"]
        GAIA["GAIA-2 (2025)"]
        GameNGen["GameNGen"]
    end

    subgraph VLA["视觉-语言-动作模型 VLA"]
        RT2["RT-2 (2023)"]
        Octo["Octo (2024)"]
        OpenVLA["OpenVLA (2024)"]
        Pi0["π₀ (2024)"]
        Pi05["π₀.₅ (2025)"]
        DroneVLA["VLA-AN / CognitiveDrone"]
    end

    subgraph UAV["无人机 UAV"]
        训练仿真["仿真训练"]
        部署推理["端侧部署"]
        集群协作["集群协作"]
    end

    VLM -- "视觉-语言理解骨干" --> VLA
    VLM -- "场景描述/标注" --> WM
    WM -- "想象/数据增强" --> VLA
    WM -- "预测模型" --> UAV
    VLA -- "端到端控制" --> UAV
    UAV -- "真实数据回流" --> WM
    UAV -- "任务反馈" --> VLA
```

---

## 三、技术路线分类

### 3.1 世界模型的三大角色

| 角色 | 核心功能 | 代表工作 | 与无人机的关系 |
|:---:|:---:|:---:|:---|
| **策略模型** | 直接输出动作 | IRIS、DIAMOND | 可用于无人机导航策略学习 |
| **模拟器** | 预测下一状态 | UniSim、Cosmos | 替代/增强飞行动力学仿真器 |
| **视频生成** | 生成逼真视频帧 | Sora、GAIA-2 | 合成训练数据、场景预览 |

### 3.2 VLA 模型的架构分代

| 代次 | 时间 | 核心思路 | 代表模型 | 关键突破 |
|:---:|:---:|:---|:---|:---|
| 第一代 | 2023 | VLM + 动作 token 化 | RT-2 | 证明大模型可以直接输出动作 |
| 第二代 | 2024 初 | 开源 + 高效训练 | OpenVLA, Octo | 降低门槛，社区可复现 |
| 第三代 | 2024 末 | 流匹配 + 扩散 | π₀ | 连续动作输出，表达力更强 |
| 第四代 | 2025 | 推理增强 + 世界模型 | π₀.₅, VLA-AN | 融入世界模型进行"想象" |
| 第五代 | 2025+ | 具身基础模型 | CognitiveDrone | 无人机专用，长程推理 |

### 3.3 VLM 在遥感/无人机中的演进

```mermaid
timeline
    title VLM 在无人机与遥感中的发展
    2023-06 : LLaVA-1.5
             : 通用 VLM 基线
    2023-10 : GeoChat
             : 首个遥感 VLM
    2024-01 : RSGPT
             : 遥感图文对话
    2024-06 : LHRS-Bot
             : 多分辨率遥感
    2024-09 : ChangeChat
             : 变化检测对话
    2025-01 : Qwen2.5-VL
             : 通用 VLM 能力飞跃
    2025-03 : DroneVLM
             : 无人机专用 VLM
```

---

## 四、无人机场景下的技术栈

```mermaid
graph LR
    subgraph 数据层
        A1["飞行遥感图像"]
        A2["LiDAR 点云"]
        A3["IMU / GPS"]
        A4["仿真环境"]
    end

    subgraph 模型层
        B1["VLM<br/>场景理解"]
        B2["World Model<br/>状态预测"]
        B3["VLA<br/>动作生成"]
        B4["检测/跟踪<br/>目标感知"]
    end

    subgraph 任务层
        C1["自主导航"]
        C2["目标跟踪"]
        C3["搜索与救援"]
        C4["巡检与测绘"]
        C5["集群协同"]
    end

    A1 --> B1
    A1 --> B2
    A2 --> B4
    A3 --> B3
    A4 --> B2
    B1 --> B3
    B2 --> B3
    B1 --> C2
    B2 --> C1
    B3 --> C1
    B3 --> C5
    B4 --> C2
    B1 --> C3
    B2 --> C4
```

---

## 五、关键问题与研究前沿

### 5.1 核心挑战

| 挑战 | 描述 | 相关研究方向 |
|:---|:---|:---|
| **sim-to-real gap** | 仿真与真实飞行环境差异大 | 域随机化、世界模型作为自适应模拟器 |
| **实时性** | VLA/VLM 推理延迟 vs 飞控 50-200Hz | 模型蒸馏、量化、边缘部署 |
| **安全性** | 无人机坠毁代价高 | 安全 RL、约束优化、形式化验证 |
| **长程决策** | 电池续航有限，需要高效规划 | 层次化 VLA、世界模型 + 搜索 |
| **数据稀缺** | 真实飞行数据收集成本高 | 仿真器合成、世界模型生成、少样本学习 |
| **集群可扩展** | 多机通信与协作策略 | 多智能体 VLA、分布式世界模型 |

### 5.2 前沿方向

```mermaid
mindmap
  root((前沿方向))
    世界模型 + 无人机
      4D 时空预测
        未来帧预测
        深度/语义预测
        物理规律学习
      替代飞行动力学仿真
        端到端学习动力学
        自适应模拟器
      数据引擎
        合成罕见场景
        对抗样本生成
    VLA + 无人机
      端到端自主飞行
        视觉输入到电机输出
        取代传统 PX4/ArduPilot
      多模态指令
        语音控制
        手势控制
        自然语言目标
      集群协作 VLA
        分布式决策
        异构编队
    VLM + 无人机
      空中场景理解
        实时地物分类
        变化检测
        异常检测
      人机交互
        自然语言任务描述
        视觉问答
        报告自动生成
```

---

## 六、典型应用案例

### 6.1 从语言指令到飞行动作

```mermaid
sequenceDiagram
    participant Operator as 操作员
    participant VLM as VLM (Qwen2.5-VL)
    participant WM as World Model
    participant VLA as VLA (π₀.₅)
    participant FC as 飞控 (PX4)

    Operator->>VLM: "飞到红色建筑旁边<br/>检查窗户是否有裂缝"
    VLM->>VLM: 场景理解<br/>定位红色建筑
    VLM->>WM: 提供目标语义描述
    WM->>WM: 预测飞行路径上的障碍
    WM->>VLA: 提供想象的未来帧
    VLA->>VLA: 融合视觉+语言+预测
    VLA->>FC: 输出动作序列<br/>(速度/航向/高度)
    FC->>FC: 执行飞行控制
    FC-->>VLM: 实时图像反馈
```

### 6.2 检查巡检场景

| 阶段 | 技术 | 说明 |
|:---|:---|:---|
| 起飞规划 | World Model | 预测天气、风场对飞行的影响 |
| 航线跟踪 | VLA | 根据视觉实时调整航迹 |
| 异常检测 | VLM | 理解"正常 vs 异常"语义 |
| 报告生成 | VLM | 自然语言描述发现的问题 |
| 返航决策 | World Model + VLA | 预测电量、规划最优返航路径 |

---

## 七、与其他领域的交叉

```mermaid
graph TB
    UAV_Drone["无人机 UAV"] --- WM["世界模型"]
    UAV_Drone --- VLA["VLA"]
    UAV_Drone --- VLM["VLM"]

    WM --- AV["自动驾驶"]
    WM --- Game["游戏 AI"]
    WM --- Sim2Real["Sim2Real"]

    VLA --- Manip["机械臂操作"]
    VLA --- Humanoid["人形机器人"]
    VLA --- Legged["四足机器人"]

    VLM --- RS["遥感解译"]
    VLM --- Medical["医学影像"]
    VLM --- Embodied["具身智能"]

    style UAV_Drone fill:#f9f,stroke:#333,stroke-width:3px
    style WM fill:#bbf,stroke:#333
    style VLA fill:#bfb,stroke:#333
    style VLM fill:#fbb,stroke:#333
```

---

## 八、学习路线建议

| 阶段 | 目标 | 建议时长 |
|:---:|:---|:---:|
| **入门** | 理解 VLM 基础、Transformer、多模态 | 2-3 周 |
| **基础** | 掌握 World Model 核心概念（表征、预测） | 2-3 周 |
| **进阶** | 深入 VLA 架构（RT-2, OpenVLA, π₀） | 3-4 周 |
| **专题** | 无人机 VLA/WM 前沿论文 | 2-4 周 |
| **实践** | 仿真环境搭建 + 模型微调/部署 | 4-8 周 |

> 详细的论文阅读顺序请参考 [reading-order.md](reading-order.md)。
> 完整论文列表请参考 [../references/paper-list.md](../references/paper-list.md)。

---

*本文件为 UAV-WM-VLA-Learning 项目的一部分，最后更新：2026-05-10。*
