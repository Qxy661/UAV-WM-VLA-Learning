# 世界模型分类体系

> 本文从 **角色定位、架构设计、表征方式、无人机专用** 四个维度，系统梳理世界模型（World Model）的分类体系。配合 Mermaid 思维导图与分类表格，帮助读者建立清晰的认知地图。

---

## 一、总分类图谱

```mermaid
mindmap
  root((世界模型<br/>World Model))
    按角色分类
      策略模型 Policy
      模拟器 Simulator
      视频生成 Video Gen
    按架构分类
      IDM 逆动力学
      单骨干 Single-backbone
      MoE 混合专家
      统一表征 Unified
      隐空间 Latent
    按表征分类
      像素 Pixel
      隐空间 Latent
      语义 Semantic
      3D 体素 Voxel
    按应用领域
      机器人 Manipulation
      自动驾驶 AV
      游戏 Game
      无人机 UAV
    按训练范式
      自监督 Self-supervised
      预测式 Predictive
      生成式 Generative
      对比式 Contrastive
```

---

## 二、按角色分类

### 2.1 角色定义

```mermaid
graph LR
    Obs["观测 sₜ"] --> WM["世界模型"]
    Action["动作 aₜ"] --> WM
    WM --> Role1["作为策略<br/>输出 aₜ₊₁"]
    WM --> Role2["作为模拟器<br/>输出 sₜ₊₁"]
    WM --> Role3["作为生成器<br/>输出视频帧"]
```

| 角色 | 输入 | 输出 | 典型损失函数 | 代表模型 |
|:---:|:---|:---|:---|:---|
| **策略模型** | 观测序列 | 动作序列 | 行为克隆 / RL 奖励 | IRIS (2023), DIAMOND (2024) |
| **模拟器** | 当前状态 + 动作 | 下一状态 | 状态重建 / 预测误差 | UniSim (2023), Cosmos (2025) |
| **视频生成** | 条件（文本/动作） | 视频帧序列 | 扩散去噪 / GAN | Sora (2024), GAIA-2 (2025) |

### 2.2 三种角色的协作关系

```mermaid
graph TB
    subgraph PolicyRole["策略模型 (Policy)"]
        IRIS["IRIS"]
        DIAMOND["DIAMOND"]
    end

    subgraph SimRole["模拟器 (Simulator)"]
        UniSim["UniSim"]
        Cosmos["NVIDIA Cosmos"]
        GAIA2["Wayve GAIA-2"]
    end

    subgraph VideoRole["视频生成 (Video Gen)"]
        Sora["Sora"]
        Veo["Google Veo"]
        Kling["快手 Kling"]
    end

    SimRole -- "提供想象轨迹" --> PolicyRole
    VideoRole -- "数据增强" --> SimRole
    PolicyRole -- "提供动作条件" --> VideoRole
    SimRole -- "条件化生成" --> VideoRole

    style PolicyRole fill:#e8f5e9
    style SimRole fill:#e3f2fd
    style VideoRole fill:#fce4ec
```

---

## 三、按架构分类

### 3.1 五大架构范式

```mermaid
mindmap
  root((架构分类))
    IDM 逆动力学模型
      原理: 从 (sₜ, sₜ₊₁) 推断 aₜ
      优势: 无需动作标签即可训练正向模型
      代表: VIPER, IRIS
      无人机应用: 从飞行视频学习动作
    单骨干模型
      原理: 一个模型同时预测状态和动作
      优势: 简单统一
      代表: DreamerV3, TD-MPC2
      无人机应用: 端到端飞行动作预测
    MoE 混合专家
      原理: 多个专家网络 + 路由机制
      优势: 处理多模态/多任务
      代表: MoE World Model (2024)
      无人机应用: 不同飞行模式切换
    统一表征
      原理: 语言+视觉+动作共享表征
      优势: 知识迁移
      代表: Unified World Model (2024)
      无人机应用: 多任务无人机智能体
    隐空间模型
      原理: 在压缩的隐空间做预测
      优势: 计算效率高
      代表: Dreamer系列, TD-MPC
      无人机应用: 嵌入式部署
```

### 3.2 架构对比表

| 架构 | 计算复杂度 | 预测质量 | 动作可控性 | 适合部署 | 代表模型 |
|:---:|:---:|:---:|:---:|:---:|:---|
| IDM | 中 | 中 | 低（需配合正向模型） | 中 | VIPER |
| 单骨干 | 中 | 中-高 | 高 | 中 | DreamerV3 |
| MoE | 高 | 高 | 高 | 低 | MoE-WM |
| 统一表征 | 高 | 高 | 高 | 低 | UWM |
| 隐空间 | 低-中 | 中 | 中 | 高 | TD-MPC2 |

### 3.3 关键模型的架构路线

```mermaid
graph TB
    subgraph 2023["2023"]
        UniSim23["UniSim<br/>统一模拟器"]
        Dreamer3["DreamerV3<br/>隐空间模型"]
        IRIS23["IRIS<br/>IDM + Transformer"]
    end

    subgraph 2024["2024"]
        Genie24["Genie<br/>11B 交互世界"]
        DIAMOND24["DIAMOND<br/>扩散世界模型"]
        GameNGen24["GameNGen<br/>实时游戏生成"]
        GAIA1["GAIA-1<br/>自动驾驶"]
    end

    subgraph 2025["2025"]
        Cosmos25["Cosmos<br/>物理世界基础"]
        GAIA2["GAIA-2<br/>统一视频-动作"]
        WorldLab["World Labs<br/>3D 世界"]
    end

    UniSim23 --> Cosmos25
    Dreamer3 --> DIAMOND24
    IRIS23 --> GAIA1
    GAIA1 --> GAIA2
    Genie24 --> WorldLab
```

---

## 四、按表征分类

### 4.1 四种表征方式

```mermaid
mindmap
  root((表征方式))
    像素表征 Pixel
      特点: 直接预测RGB像素
      优势: 信息无损,视觉直观
      劣势: 计算量大,冗余多
      代表: Sora, GAIA-2
      适用: 视频生成,数据增强
    隐空间表征 Latent
      特点: VAE/VQ-VAE压缩后预测
      优势: 计算效率高
      劣势: 可能丢失细节
      代表: DreamerV3, LWM
      适用: 嵌入式,长程预测
    语义表征 Semantic
      特点: 预测语义分割/特征图
      优势: 高层抽象,泛化强
      劣势: 无法重建精细外观
      代表: GAIA-1 Semantic
      适用: 导航规划
    3D体素表征 Voxel
      特点: 3D空间体素预测
      优势: 空间关系明确
      劣势: 内存消耗大
      代表: World Labs
      适用: 3D场景理解
```

### 4.2 像素 vs 隐空间：详细对比

| 维度 | 像素表征 | 隐空间表征 |
|:---|:---|:---|
| **信息保真度** | 高（无损） | 中-低（有压缩损失） |
| **计算成本** | 高（H×W×3） | 低（h×w×d, h<<H） |
| **训练难度** | 中-高 | 中 |
| **可控性** | 高（可直接编辑像素） | 中（需解码器） |
| **物理一致性** | 中（需大量数据学习） | 低-中 |
| **适合任务** | 视频生成、数据增强 | RL 策略学习、嵌入部署 |
| **典型分辨率** | 256x256 ~ 1024x1024 | 16x16 ~ 64x64 |

### 4.3 隐空间模型的编码器选择

```mermaid
graph LR
    Input["原始帧<br/>256x256x3"] --> Enc1["CNN 编码器"]
    Input --> Enc2["ViT 编码器"]
    Input --> Enc3["VQ-VAE"]
    Input --> Enc4["VAE (SD风格)"]

    Enc1 --> Lat1["隐空间<br/>8x8x512"]
    Enc2 --> Lat2["隐空间<br/>16x16x768"]
    Enc3 --> Lat3["离散码本<br/>16x16xCode"]
    Enc4 --> Lat4["连续隐空间<br/>32x32x4"]

    Lat1 --> WM1["DreamerV3"]
    Lat2 --> WM2["IRIS"]
    Lat3 --> WM3["DIAMOND"]
    Lat4 --> WM4["LWM"]
```

---

## 五、按应用领域分类

### 5.1 四大应用领域

```mermaid
graph TB
    WM["世界模型"] --> Manip["机器人操作"]
    WM --> AV["自动驾驶"]
    WM --> Game["游戏 / 交互"]
    WM --> UAV["无人机"]

    Manip --> Manip1["桌面抓取"]
    Manip --> Manip2["灵巧手"]
    Manip --> Manip3["移动操作"]

    AV --> AV1["城市驾驶"]
    AV --> AV2["高速场景"]
    AV --> AV3["越野场景"]

    Game --> Game1["游戏 NPC"]
    Game --> Game2["虚拟世界"]
    Game --> Game3["具身模拟"]

    UAV --> UAV1["自主导航"]
    UAV --> UAV2["目标跟踪"]
    UAV --> UAV3["集群协作"]
    UAV --> UAV4["巡检测绘"]
```

### 5.2 领域差异对比

| 维度 | 机器人操作 | 自动驾驶 | 游戏 | 无人机 |
|:---|:---|:---|:---|:---|
| **动作空间** | 6-7 DoF 末端 | 2D (转向+速度) | 键鼠/手柄 | 4-6 DoF (速度+姿态) |
| **观测空间** | 手眼相机 | 环视相机+LiDAR | 屏幕像素 | 机载相机+下视 |
| **动力学复杂度** | 中 | 中-高 | 低（规则驱动） | 高（6DoF） |
| **安全要求** | 高 | 极高 | 低 | 高 |
| **数据获取** | 需要机器人 | 大规模路采 | 海量日志 | 成本高 |
| **实时性** | 10-30Hz | 10-30Hz | 60Hz | 50-200Hz |

---

## 六、无人机专用世界模型

### 6.1 无人机世界模型的独特挑战

```mermaid
mindmap
  root((无人机世界模型<br/>挑战))
    观测特殊性
      高度变化大 0-500m
      俯视视角为主
      运动模糊
      光照变化剧烈
    动力学特殊性
      6DoF运动
      风扰动强
      非线性耦合
      电池约束
    数据特殊性
      收集成本高
      安全风险大
      真实数据稀缺
      仿真差距大
    任务特殊性
      长航时规划
      3D导航
      动态目标跟踪
      集群协调
```

### 6.2 无人机世界模型的研究方向

| 方向 | 描述 | 代表/参考 |
|:---|:---|:---|
| **飞行视频预测** | 从无人机视频学习未来帧预测 | 基于 GAIA-2 的无人机适配 |
| **风场/动力学建模** | 隐式学习飞行器动力学与环境交互 | Neural Flyer (2024) |
| **3D 场景世界模型** | 从航拍构建可交互的 3D 世界 | 结合 NeRF/3DGS |
| **Sim2Real 桥梁** | 用世界模型弥补仿真与真实差距 | 域适应 + 视觉增广 |
| **多机协同模拟** | 多无人机交互的世界模型 | 多智能体世界模型 |

### 6.3 无人机世界模型技术栈

```mermaid
graph TB
    subgraph 输入
        FrontCam["前视相机"]
        DownCam["下视相机"]
        IMU["IMU 数据"]
        GPS["GPS/RTK"]
        Cmd["任务指令"]
    end

    subgraph 编码
        VisEnc["视觉编码器<br/>(ViT / SigLIP)"]
        StateEnc["状态编码器<br/>(MLP / TCN)"]
        LangEnc["语言编码器<br/>(LLM / T5)"]
    end

    subgraph 世界模型核心
        PredCore["预测核心<br/>(Transformer / SSM)"]
        ActionCond["动作条件化"]
        Uncertainty["不确定性建模"]
    end

    subgraph 输出
        NextObs["下一观测预测"]
        Reward["奖励预测"]
        Done["终止预测"]
        Value["价值估计"]
    end

    FrontCam --> VisEnc
    DownCam --> VisEnc
    IMU --> StateEnc
    GPS --> StateEnc
    Cmd --> LangEnc

    VisEnc --> PredCore
    StateEnc --> PredCore
    LangEnc --> PredCore
    ActionCond --> PredCore

    PredCore --> NextObs
    PredCore --> Reward
    PredCore --> Done
    PredCore --> Value
    PredCore --> Uncertainty
```

---

## 七、训练范式分类

### 7.1 四种训练范式

```mermaid
graph TB
    subgraph SelfSup["自监督学习"]
        SSL1["对比学习"]
        SSL2["掩码预测"]
        SSL3["重建学习"]
    end

    subgraph Pred["预测式学习"]
        Pred1["前向预测 sₜ₊₁ = f(sₜ, aₜ)"]
        Pred2["逆向预测 aₜ = f(sₜ, sₜ₊₁)"]
        Pred3["多步预测"]
    end

    subgraph Gen["生成式学习"]
        Gen1["扩散模型"]
        Gen2["自回归"]
        Gen3["GAN"]
    end

    subgraph Cont["对比式学习"]
        Cont1["正负样本对比"]
        Cont2["时序对比"]
        Cont3["跨模态对比"]
    end

    SelfSup --> Pred
    Pred --> Gen
```

### 7.2 范式对比

| 范式 | 数据需求 | 训练稳定性 | 生成质量 | 适合场景 |
|:---:|:---:|:---:|:---:|:---|
| 自监督 | 大量无标注 | 高 | 中 | 表征预训练 |
| 预测式 | 带动作的轨迹 | 高 | 中-高 | RL 环境模型 |
| 生成式 | 海量视频 | 中-低 | 极高 | 视频/数据生成 |
| 对比式 | 多视角/时序对 | 高 | 低（非生成） | 表征学习 |

---

## 八、从经典到前沿的演进

```mermaid
timeline
    title 世界模型关键里程碑
    2018 : Ha & Schmidhuber
           : 首个神经网络世界模型
    2020 : Dreamer (v1)
           : 隐空间 + RL
    2021 : DreamerV2
           : 离散隐变量
    2022 : DreamerV3
           : 通用隐空间模型
    2023 : UniSim
           : 统一模拟器
    2023 : IRIS
           : Transformer + IDM
    2024 : Genie
           : 11B 参数交互世界
    2024 : DIAMOND
           : 扩散世界模型
    2024 : GAIA-1
           : 自动驾驶世界模型
    2025 : Cosmos
           : 物理世界基础模型
    2025 : GAIA-2
           : 统一视频-动作预测
    2025 : World Labs
           : 3D 交互世界
```

---

## 九、选择指南：哪种世界模型适合你的场景？

```mermaid
flowchart TD
    Start["你的场景是什么?"] --> Q1{"需要生成视频?"}
    Q1 -->|是| VideoGen["视频生成模型<br/>Sora / GAIA-2"]
    Q1 -->|否| Q2{"需要替代仿真器?"}
    Q2 -->|是| Sim["模拟器类<br/>UniSim / Cosmos"]
    Q2 -->|否| Q3{"用于 RL 策略学习?"}
    Q3 -->|是| Q4{"计算资源受限?"}
    Q4 -->|是| Latent["隐空间模型<br/>DreamerV3 / TD-MPC"]
    Q4 -->|否| Pixel["像素预测模型<br/>DIAMOND"]
    Q3 -->|否| Q5{"需要预测动作?"}
    Q5 -->|是| IDM["IDM 模型<br/>VIPER / IRIS"]
    Q5 -->|否| Classic["经典世界模型<br/>Ha 2018 原始版本"]

    VideoGen --> DroneCheck1{"部署在无人机上?"}
    Sim --> DroneCheck2{"部署在无人机上?"}
    Latent --> DroneCheck3{"部署在无人机上?"}

    DroneCheck1 -->|是| Drone1["需要模型压缩<br/>(量化/蒸馏)"]
    DroneCheck2 -->|是| Drone2["考虑机载推理<br/>vs 地面站推理"]
    DroneCheck3 -->|是| Drone3["最有可能直接部署<br/>隐空间模型体积小"]

    style Start fill:#ffd54f
    style Drone1 fill:#ef9a9a
    style Drone2 fill:#ef9a9a
    style Drone3 fill:#a5d6a7
```

---

*本文件为 UAV-WM-VLA-Learning 项目的一部分，最后更新：2026-05-10。*
