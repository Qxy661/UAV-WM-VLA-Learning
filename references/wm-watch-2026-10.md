# 世界模型 前沿增量 · 2026-10

> 生成：`py -3.9 tools/watch.py --volume wm --write`｜窗口 2026-06-11 ~ 2026-10-09（120 天）｜词表 24 条｜请求失败 0 条｜新增 346 条

词表按 docs/02-世界模型专题/07–09 三篇 分节。

**读法**（纪律 F 的镜像规则）：TOTAL 是**题摘层面的命中数，不是相关论文数**。一条宽查询命中几百条不含任何信息，所以下表只列**仓库尚未收录**的条目。`NULL` 表示请求失败（429/超时），**不是 0 命中**；`0` 表示确实没有，但第一反应应当是怀疑查询措辞，换个提法复核后再下结论。

arXiv 的 `all:` / `abs:` 只覆盖题名/摘要/作者/注释，**不覆盖全文**。

**去重范围是整仓**：`docs/`、`references/`、`mindmaps/`、`paper/` 下的 `.md`，加 `README.md` / `CONTRIBUTING.md` / `code/README.md`。`paper/` 下的 `_work*/` 取数缓存（json/xml/html/pdf）**不算收录**，故不计入——那里放的是原始查询结果，不是读过的文献。所以某条 ID 不在本表，只能推出**整仓没写过**，推不出"这个专题没有"。
（2026-10-08 修：`paper/` 原先漏扫，加上 ID 正则不吃 `v1` 版本号后缀，两条合起来让 27 条早已在 `paper/` 笔记里读过的论文被标成"新增"。三卷的台账互为已收录，故 `<卷>-watch-*.md` 一律排除。）

## 07 · 世界模型评测与诊断

**`abs:"world model" AND abs:"evaluation"`** — 命中 471，其中新增 21（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09763v1` | Beyond Policy Support: Interaction Constrained Offline Reinforcement Learning for Autonomous Driving |
| 2026-10-07 | `2610.09546v1` | PCDT: A Predictive Cognitive Digital Twin Framework for Intelligent and Autonomous 6G Network Ecosystems |
| 2026-10-07 | `2610.09438v1` | Controllable Crowd Generation through World-Model Planning |
| 2026-10-07 | `2610.09307v1` | vLLM-Omni Technical Report: A Unified Serving Runtime for Omni-Modality Generation |
| 2026-10-07 | `2610.09305v1` | Kuration SDK: Addressing the Virtual2Real Gap via Data Curation |
| 2026-10-06 | `2610.09048v1` | Towards Financial World Modeling |
| 2026-10-06 | `2610.08780v1` | DepthWorld: 3D World Model for Robot Manipulation |
| 2026-10-06 | `2610.08777v1` | CtrlCache: Accelerating Interactive Video World Models with Control-Aware Caching |
| 2026-10-06 | `2610.08640v1` | RIWANav: Recursive World-Action Models with Self-Improvement for Urban Navigation |
| 2026-10-06 | `2610.08469v1` | A Belief-State World Model for Catheter Navigation under Sparse Fluoroscopy: A Planar Proof of Concept |
| 2026-10-06 | `2610.08464v1` | Federated Bayesian Surveillance of Mechanical Thrombectomy Adverse Events: A Population Risk Layer for Surgica |
| 2026-10-06 | `2610.08267v1` | WM4ISAC: World Model for Proactive ISAC Under Dynamic Blockage |
| 2026-10-06 | `2610.08033v1` | Learning in Dreams, Winning in Reality: A Continuous Dyna Loop for a Ten-Hero MOBA |
| 2026-10-06 | `2610.07704v1` | Independent Multi-Agent Reinforcement Learning with Counterfactual Semantic-Social World Models |
| 2026-10-05 | `2610.06814v2` | TAPDreamer: Transferable Adversarial Patches for World Action Models |
| 2026-10-05 | `2610.06349v1` | KineWorld: Action-Induced Transport Fields for Embodied World Modeling |
| 2026-10-05 | `2610.06100v1` | From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation |
| 2026-10-05 | `2610.05912v1` | MiniCorp: The Last Mile of the AI Agent Firm |
| 2026-10-05 | `2610.05861v1` | Imagine to Act: High-Fidelity Data Synthesis via Image Editing World Model for Scalable GUI Agent Training |
| 2026-10-04 | `2610.07028v1` | Identifiable World Models from Pretrained Diffusion Representations |
| 2026-10-04 | `2610.05240v1` | Pythia: Toward Foundation World Models for Multimodal Time Series |

**`abs:"world model" AND abs:"benchmark"`** — 命中 288，其中新增 17（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10274v1` | Sparse Planning in Visual World Models via Cost Gradients |
| 2026-10-07 | `2610.09546v1` | PCDT: A Predictive Cognitive Digital Twin Framework for Intelligent and Autonomous 6G Network Ecosystems |
| 2026-10-07 | `2610.09514v1` | STRIKE: Learning Visual State Transitions for Physical World Modeling |
| 2026-10-07 | `2610.09305v1` | Kuration SDK: Addressing the Virtual2Real Gap via Data Curation |
| 2026-10-06 | `2610.08760v1` | WorldSonus: Bringing Sound to Worlds |
| 2026-10-06 | `2610.08464v1` | Federated Bayesian Surveillance of Mechanical Thrombectomy Adverse Events: A Population Risk Layer for Surgica |
| 2026-10-06 | `2610.07704v1` | Independent Multi-Agent Reinforcement Learning with Counterfactual Semantic-Social World Models |
| 2026-10-05 | `2610.06814v2` | TAPDreamer: Transferable Adversarial Patches for World Action Models |
| 2026-10-05 | `2610.05861v1` | Imagine to Act: High-Fidelity Data Synthesis via Image Editing World Model for Scalable GUI Agent Training |
| 2026-10-03 | `2610.04475v1` | VCLMU: Mechanism-Centric Virtual Cell World Modeling for Perturbation Response |
| 2026-10-03 | `2610.04301v1` | EnvDreamer: Large-Scale Multimodal-to-Environment Generation for Embodied AI |
| 2026-10-03 | `2610.04168v1` | Agentic Cognitive Depth: Operational Criteria for Evaluating LLM Agents |
| 2026-10-02 | `2610.03713v1` | What Should World Models Forget? Stratified Retention for Continual Adaptation |
| 2026-10-02 | `2610.03356v1` | ReFract: Benchmarking Perspective Awareness in Language Model Agents with Text World Models |
| 2026-10-02 | `2610.02957v1` | Understanding Trajectory Heterogeneity in Federated World Model Learning |
| 2026-10-01 | `2610.02368v1` | Rethinking World-Action Model for Compositional and In-Context Robotic Manipulation |
| 2026-10-01 | `2610.02331v1` | World Editing: Intervening on Executable Worlds at Increasing Depth |

**`abs:"rollout" AND abs:"error accumulation"`** — 命中 40，其中新增 29（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-02 | `2610.03313v1` | SDECast: Probabilistic Weather Forecasting in Continuous Time with Neural SDEs |
| 2026-10-02 | `2610.02779v1` | TRAC: Trajectory-aware Reuse and Adaptive Correction for Efficient Autoregressive Video Generation |
| 2026-10-01 | `2610.01765v1` | Physics-Refined Spatiotemporal Forecasting on Open-Boundary Hydrologic Graphs |
| 2026-09-29 | `2609.38114v1` | Self-Aligned Forcing: Streaming Video Diffusion with Differentiable Noisy History |
| 2026-09-28 | `2609.34697v2` | Triangular Resampling for Long-Horizon Motion Generation |
| 2026-09-27 | `2609.33412v1` | Resolving State-Representation Mismatch: State-Space Visual Reasoning for Open-Loop VLA Planning |
| 2026-09-17 | `2609.19674v2` | Conservation Buys Stability and Factoring Buys Counterfactuals in Physical World Models |
| 2026-09-15 | `2609.16621v1` | Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction |
| 2026-09-08 | `2609.09123v1` | Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout |
| 2026-09-03 | `2609.03225v2` | Long-Horizon Consistent and Interaction-Aware World Models for Multi-Style End-to-End Driving |
| 2026-08-23 | `2608.22277v3` | Dynamics-Aware Weighting for Deep Learning Forecasts of Chaotic Systems |
| 2026-08-16 | `2608.15815v1` | KOALA: Koopman Operator Learning for WiFi-Based Anticipatory Hum |
| 2026-08-12 | `2608.12107v1` | Avatar-Forever: Decoupled Parallel Training for High-Quality Real-Time Infinite Avatars |
| 2026-08-12 | `2608.11623v1` | FM-LLM: A frequency-enhanced mixture-of-experts framework for adapting LLMs to time series forecasting |
| 2026-08-07 | `2608.07189v1` | Autoregressive rollout error in latent-space reduced-order models of bluff-body wakes is accumulated phase dri |
| 2026-08-06 | `2608.06241v1` | Timestep-Conditioned Transformers for Global Weather Forecasting |
| 2026-08-06 | `2608.05925v1` | Local-Global Feature Mixer and Trend-Guided Consistent Learning for Remaining Useful Life Prediction of Rotati |
| 2026-08-06 | `2608.05806v1` | Hierarchical Latent Prediction for Language Models |
| 2026-08-02 | `2608.01164v1` | Hybrid Lagrangian-Eulerian Model for Lagrangian Fluid Simulation |
| 2026-07-31 | `2609.20309v1` | Hypernetwork-Parameterized Spatially Adaptive Neural Operators for PDE Learning |
| 2026-07-31 | `2607.29135v1` | HERO: History-Enriched Rollout Training for Long-Horizon Autoregressive Neural Operators |
| 2026-07-31 | `2608.11237v1` | Geometry-aware Incremental Neural Operator for Long-Horizon PDE prediction |
| 2026-07-29 | `2607.27110v2` | FreqForcing: Autoregressive Long Video Generation via Spectral Self-Anchoring |
| 2026-07-29 | `2607.27036v1` | Mitigating Compounding Error via Video Representation Regularization |
| 2026-07-22 | `2607.19719v2` | Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model Imagination |
| 2026-07-20 | `2607.18082v4` | CriPO: Enhancing Rubric-based RL via Self-Distillation |
| 2026-07-17 | `2607.18309v1` | Spatio-Temporal Prediction of Unsteady Airfoil Aerodynamics Using Augmented Graph Neural Ordinary Differential |
| 2026-07-13 | `2607.11971v1` | Uncertainty-Aware Crack Growth Forecasting via Conditional Denoising Diffusion Models for Phase-Field Fracture |
| 2026-07-11 | `2607.10504v1` | SUREFlow: State-space Uncertainty-aware REsidual Flow Matching for Robust Robot Manipulation |

**`abs:"video prediction" AND abs:"evaluation metrics"`** — 命中 0。**复核过：词汇层面的假空白。**同窗口换提法 —— `abs:"video prediction" AND abs:"benchmark"` 命中 **14**、`abs:"video generation" AND abs:"evaluation metrics"` 命中 **6**、`abs:"video prediction" AND abs:"metrics"` 命中 **2**。不限窗口时原提法本身也只有 **3** 条，且全部止于 2021 年。「evaluation metrics」是评测论文的写法，生成类论文写 benchmark / metric（单数）。**不得记录为「视频预测没有评测工作」。**

**`abs:"world model" AND abs:"diagnostic"`** — 命中 55，其中新增 23（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09305v1` | Kuration SDK: Addressing the Virtual2Real Gap via Data Curation |
| 2026-10-06 | `2610.08960v1` | Directed Temporal Representations for Offline Visual Control |
| 2026-10-03 | `2610.04168v1` | Agentic Cognitive Depth: Operational Criteria for Evaluating LLM Agents |
| 2026-09-30 | `2609.40003v1` | DashVMC: Real-Time Discrete World Model Control in Geometry Dash |
| 2026-09-28 | `2609.36333v2` | ATLAS: Aligned Transport of Latent Structure for Reliable World Model Planning |
| 2026-09-28 | `2609.35463v1` | A.D.A.M.O. (Agent for language-Driven Actions with Multimodal Observations): A Visual-Symbolic Framework for V |
| 2026-09-26 | `2609.32679v1` | The GUI Is Not the State: Diagnosing State Aliasing in GUI World Models |
| 2026-09-25 | `2609.30711v1` | Recommendation World Models for Future-State Control |
| 2026-09-25 | `2609.30650v1` | Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation |
| 2026-09-24 | `2609.30264v2` | AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control |
| 2026-09-23 | `2609.27621v2` | SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design |
| 2026-09-14 | `2609.15781v1` | When the World Lies: Backdoor Attacks on Latent World Models for Downstream Control |
| 2026-09-13 | `2609.14854v1` | AutoLab: An Internet-Accessible Experimental Platform for Operational World Models in Wireless Networks |
| 2026-09-07 | `2609.07719v1` | A radiographic world model for clinical reasoning and evidence generation |
| 2026-08-30 | `2608.29998v1` | The Intervention Gap in Latent World Models |
| 2026-08-23 | `2608.22358v1` | Tracing the Unlabeled Storm: Cross-Variable Transfer in a Lagrangian Atmospheric JEPA Framework |
| 2026-08-23 | `2608.22294v2` | Beyond Instance Slots: Semantically Rich World Models for Physical Interaction Planning |
| 2026-08-19 | `2608.19085v2` | DA-WAM: Decision-Aligned Future Latents for Driving World Models |
| 2026-08-17 | `2608.16859v2` | HarnessEval-W: Agentifying the Evaluation of Visual Worlds |
| 2026-08-15 | `2608.15156v4` | Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models |
| 2026-08-13 | `2608.13049v1` | H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models |
| 2026-08-12 | `2609.05461v1` | ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models |
| 2026-08-10 | `2608.10145v1` | The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom |

**`abs:"world model" AND abs:"failure modes"`** — 命中 33，其中新增 26（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10274v1` | Sparse Planning in Visual World Models via Cost Gradients |
| 2026-10-05 | `2610.06582v1` | Mind the Execution Gap: Action-Semantic Mismatch in World-Model Control |
| 2026-09-30 | `2609.39604v1` | Why Do Conventional World Models Fail to Learn Cellular Automata? |
| 2026-09-28 | `2609.35463v1` | A.D.A.M.O. (Agent for language-Driven Actions with Multimodal Observations): A Visual-Symbolic Framework for V |
| 2026-09-27 | `2609.33940v1` | Behavioral Monitoring of JEPA World Models with Jacobian Centroids |
| 2026-09-26 | `2609.32761v1` | From Feed-Forward to Flow: Unifying Reconstruction and Generation Is Easier Than You Think |
| 2026-09-26 | `2609.32679v1` | The GUI Is Not the State: Diagnosing State Aliasing in GUI World Models |
| 2026-09-22 | `2609.26293v2` | Dual-Frontier: When Can an Agent Trust Its World Model? |
| 2026-09-15 | `2609.17325v1` | Intrinsic Motivation in Reinforcement Learning: A Research Agenda for Adaptive Self-Organisation |
| 2026-09-12 | `2609.13845v1` | LePlanner: An Iterative Amortized Controller For World Models |
| 2026-09-05 | `2609.05834v1` | Learning Counterfactual World Models for Embodied Reasoning under Partial Observability |
| 2026-09-02 | `2609.04264v3` | Spectral-Target Physical Latent Structuring for JEPA-Style World Models |
| 2026-08-25 | `2608.26200v2` | GameWAM: A World Action Model for Video Games |
| 2026-08-24 | `2608.23526v1` | Correcting a learned physical invariant improves world-model rollouts |
| 2026-08-23 | `2608.22421v1` | Where World Models Break: Natural-Input Failure Discovery |
| 2026-08-11 | `2608.10618v1` | Toward the Cognitive--Physical Limits of Embodied Intelligence through a World-Model-Centric Autonomous Racing |
| 2026-08-06 | `2608.05720v2` | PhyLatent: Learning Dynamics-Relevant Representations for JEPA World Models |
| 2026-07-29 | `2607.27511v1` | Failure Detection for Surgical Robot Imitation Policies via Flow-Matching World Modeling |
| 2026-07-29 | `2607.26752v1` | CalTwin: Towards Calibrated, Shift-Robust Medical World Models via Fisher-Information Regularisation |
| 2026-07-29 | `2607.26712v2` | ActSWM: Action-Sensitive World Models for Long-Horizon Planning in Open-World Games |
| 2026-07-22 | `2607.19749v1` | The World Model Remembers, the Actor Forgets: Dream Rehearsal for Continual Model-Based RL |
| 2026-07-17 | `2607.15620v1` | AEGIS: Assay-Aware Protocol Validation and Runtime Monitoring for Open-Source Liquid Handling Robots |
| 2026-07-15 | `2607.13681v1` | Towards Spatial Supersensing in the Wild |
| 2026-07-08 | `2607.07763v1` | Unlocking Temporal Generalization in Hamiltonian Video Dynamics Models |
| 2026-07-06 | `2607.05352v2` | Multiplayer Interactive World Models with Representation Autoencoders |
| 2026-06-26 | `2606.27806v3` | Agent vs. Parametric World Models: Hybrid Planning for Reliable Language Agents |

## 08 · 联合嵌入预测与潜空间

**`abs:"JEPA"`** — 命中 266，其中新增 20（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09940v1` | Juno: Taming Predictive Latents for Vision-Language-Action Models |
| 2026-10-06 | `2610.09194v1` | Patient, Place, Prior (P$^3$): What Counts as Personalization in Medical World Models? |
| 2026-10-06 | `2610.09048v1` | Towards Financial World Modeling |
| 2026-10-06 | `2610.08400v1` | Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems |
| 2026-10-06 | `2610.08192v1` | MacJEPA: Missingness-Robust Audio-Visual Recognition from Untrimmed Egocentric Videos |
| 2026-10-05 | `2610.06008v1` | Ultrasound Operator Guidance Using World Modeling and Retrieval Based Action Planning |
| 2026-10-05 | `2610.05996v1` | EpicWorldModel: Exploration-driven Planning with Latent World Models |
| 2026-10-05 | `2610.05731v1` | T-JEPA: A Temporal Joint-Embedding Predictive Architecture for Learning Better Remote Sensing Representations |
| 2026-10-04 | `2610.05409v1` | BeliefGraph-JEPA: Structured Latent World Models for Action-Conditioned Time Series |
| 2026-10-04 | `2610.07025v1` | WiSPER: Pose-Supervised Predictive and Residual Flow Refinement For Multi-Person 3D Pose Estimation With WiFi  |
| 2026-10-04 | `2610.07006v1` | STOCK-JEPA: Prior-Anchored Latent Revision Representation Learning in Equity Markets |
| 2026-10-04 | `2610.05043v1` | CI-JEPA: A Counterfactual Analysis of Latent Representations in Joint-Embedding Predictive Architectures for S |
| 2026-10-03 | `2610.04778v1` | Repeated-Measure Leakage, Distribution Shift, and Reliability under Partial Observation in Patient World Model |
| 2026-10-03 | `2610.06965v1` | ACG-WAM: World-Action Modeling via Action-Conditioned Geometric Latent Prediction |
| 2026-10-02 | `2610.03374v1` | EVEWorld: Physical Evolution Supervision for Embodied World Models |
| 2026-10-02 | `2610.03106v1` | S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales |
| 2026-10-02 | `2610.02864v1` | NeuroLens: Learning Latent Embeddings of Neural Semantics from Chronic Recordings |
| 2026-10-01 | `2610.01947v1` | Latent JEPA: Abstract Future Prediction for Latent Reasoning in Chemistry |
| 2026-10-01 | `2610.00976v2` | Variational Streaming Flow: Probabilistic Forecasting in Physical Time |
| 2026-09-30 | `2610.00722v1` | JEPA-TTT: Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts |

**`abs:"joint embedding" AND abs:"prediction"`** — 命中 152，其中新增 20（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09940v1` | Juno: Taming Predictive Latents for Vision-Language-Action Models |
| 2026-10-06 | `2610.08400v1` | Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems |
| 2026-10-05 | `2610.05996v1` | EpicWorldModel: Exploration-driven Planning with Latent World Models |
| 2026-10-05 | `2610.05731v1` | T-JEPA: A Temporal Joint-Embedding Predictive Architecture for Learning Better Remote Sensing Representations |
| 2026-10-04 | `2610.05543v1` | AngularWM: Wireless World Modeling for AoA Prediction from Multi-Antenna I/Q |
| 2026-10-04 | `2610.05409v1` | BeliefGraph-JEPA: Structured Latent World Models for Action-Conditioned Time Series |
| 2026-10-04 | `2610.05240v1` | Pythia: Toward Foundation World Models for Multimodal Time Series |
| 2026-10-04 | `2610.07006v1` | STOCK-JEPA: Prior-Anchored Latent Revision Representation Learning in Equity Markets |
| 2026-10-04 | `2610.05043v1` | CI-JEPA: A Counterfactual Analysis of Latent Representations in Joint-Embedding Predictive Architectures for S |
| 2026-10-02 | `2610.03106v1` | S2S-JEPA: Predicting the Predictable at Subseasonal-to-Seasonal Timescales |
| 2026-10-02 | `2610.02864v1` | NeuroLens: Learning Latent Embeddings of Neural Semantics from Chronic Recordings |
| 2026-10-01 | `2610.01947v1` | Latent JEPA: Abstract Future Prediction for Latent Reasoning in Chemistry |
| 2026-10-01 | `2610.00976v2` | Variational Streaming Flow: Probabilistic Forecasting in Physical Time |
| 2026-09-30 | `2610.00722v1` | JEPA-TTT: Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts |
| 2026-09-30 | `2609.40129v1` | VR-JEPA: Learning Contrastive-State Latent Guidance for Generation-based Video Reasoning |
| 2026-09-29 | `2609.36952v1` | ER-JEPA: Experience Replay Improves Joint-Embedding Predictive Learning in Language Models |
| 2026-09-28 | `2609.35603v2` | Control-Geometry Straightening for Sampling-Based Latent Planning |
| 2026-09-28 | `2609.34407v1` | Beyond Textual Chain-of-Thought: JEPA-Conditioned Latent Reasoning for Large Audio Language Models |
| 2026-09-28 | `2609.34085v1` | AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving |
| 2026-09-27 | `2609.33698v1` | One Latent, Many Tokens: Jointly Learning Compressed Embeddings for Efficient Language Diffusion |

**`abs:"representation collapse" AND abs:"world model"`** — 命中 6，其中新增 2

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-28 | `2609.36305v1` | Bilinear World Models: Learning Representations with Structured Dynamics for Efficient Control |
| 2026-09-02 | `2609.04264v3` | Spectral-Target Physical Latent Structuring for JEPA-Style World Models |

**`abs:"latent space" AND abs:"world model"`** — 命中 83，其中新增 22（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-05 | `2610.06008v1` | Ultrasound Operator Guidance Using World Modeling and Retrieval Based Action Planning |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-10-01 | `2610.01942v1` | Latent-Foresight: End-to-End Learning Predictable Representations for Latent World Models |
| 2026-10-01 | `2610.01842v1` | On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models |
| 2026-10-01 | `2610.01224v1` | Supervise What Decides Success: Criterion-Aligned Auxiliary Losses for Latent World-Model Planning |
| 2026-09-30 | `2609.38927v1` | World-as-Graph: Relational World Modeling Through Latent Space Graphs |
| 2026-09-29 | `2609.36985v1` | Abductive World Modeling via Causal Representation Learning |
| 2026-09-29 | `2609.36845v1` | DSWM: Decomposed Spatio-Temporal World Model for Demand-Driven UAV Base Station Repositioning |
| 2026-09-28 | `2609.34375v1` | LRC-JEPA: Disentangling Dynamics and Residual Context for Efficient World Models |
| 2026-09-28 | `2609.34300v1` | When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations |
| 2026-09-28 | `2609.34206v1` | WorldGuide: Learning Success-Failure Boundaries in Latent World Models for Vision-Language-Action Policies |
| 2026-09-25 | `2609.31162v1` | WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting |
| 2026-09-24 | `2609.30436v1` | WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving |
| 2026-09-24 | `2609.30214v1` | Underwater C3-JEPA: An Object-Centric Cross-View World Model for ROV Salvage |
| 2026-09-23 | `2609.28414v1` | Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-flow World Model |
| 2026-09-21 | `2609.24048v1` | What Matters in Designing World Action Models: An Empirical Study |
| 2026-09-18 | `2609.22521v2` | Latent Policy Steering: An Efficient and Flexible Framework for Cross-Embodiment Transfer |
| 2026-09-12 | `2609.13845v1` | LePlanner: An Iterative Amortized Controller For World Models |
| 2026-09-11 | `2609.12874v1` | VideoTok4D: A 4D-Aware Video Tokenizer for Compact World Representation |
| 2026-09-11 | `2609.12347v1` | DWMP: Leveraging Dual World Models for Humanoid Obstacle Traversal |
| 2026-09-02 | `2609.04264v3` | Spectral-Target Physical Latent Structuring for JEPA-Style World Models |
| 2026-08-29 | `2608.28995v2` | Hydra: A Navigation World Action Model with Discrete Latent Planning and Continuous Flow-Matching Execution |

**`abs:"energy-based" AND abs:"world model"`** — 命中 0。**复核过：低频真线，只是这 120 天恰好空窗。**不限窗口命中 **8**，且集中在 2026 上半年：`2605.07199`（Three-in-One World Model，能量一致性）、`2602.23058`（GeoWorld）。同族提法 `abs:"EBM" AND abs:"world model"` 不限窗口命中 **1**（`2406.08862` Cognitively Inspired Energy-Based World Models）。对照：`abs:"energy-based model"` 不限窗口 **630**、同窗口 **35**，说明词本身不冷，是「energy-based + world model」这个组合少。**可以写「这一线稀疏」，不得写「没有」。**

**`abs:"self-supervised" AND abs:"world model"`** — 命中 36，其中新增 27（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-04 | `2610.05543v1` | AngularWM: Wireless World Modeling for AoA Prediction from Multi-Antenna I/Q |
| 2026-10-01 | `2610.02515v1` | IGNITE Tokamak World Model Architecture |
| 2026-10-01 | `2610.01201v1` | iSEE: Object Permanence Through Self-Supervision |
| 2026-09-30 | `2610.00722v1` | JEPA-TTT: Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts |
| 2026-09-30 | `2609.39182v1` | MEND: Label-Free Detection, Localisation, and Correction of Latent Hallucination in World Models |
| 2026-09-29 | `2609.38278v1` | Masked Swingers: Harnessing Data Augmentation to Advance Autoencoders for Self-Supervised Learning |
| 2026-09-28 | `2609.34375v1` | LRC-JEPA: Disentangling Dynamics and Residual Context for Efficient World Models |
| 2026-09-28 | `2609.34085v1` | AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving |
| 2026-09-25 | `2609.32013v2` | TriO: Tri-Modal Unsupervised Occupancy World Model for Anything Perception |
| 2026-09-25 | `2609.30667v1` | StarWM: Self-Supervised Trained Attention Routing for Robust World Models |
| 2026-09-23 | `2609.28414v1` | Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-flow World Model |
| 2026-09-23 | `2609.28049v1` | Prompt, Probe, Train, or Annotate? Single-camera sports video understanding in amateur settings |
| 2026-09-22 | `2609.25541v1` | A JEPA Recipe for Tabular Foundation Models |
| 2026-09-18 | `2609.21740v1` | Sandwich-Residuals: Parameter-Efficient Test-time Adaptation of World Models |
| 2026-09-09 | `2609.09627v1` | Seven Sources of Physical AI Capability Formation |
| 2026-08-30 | `2608.29998v1` | The Intervention Gap in Latent World Models |
| 2026-08-11 | `2608.11174v2` | VIScore: Diagnosing Planning-Relevant Quality in Latent World Models |
| 2026-08-11 | `2608.11026v1` | MAJEPPA: Morphing and Assessing in a Unified Piano Performance Space |
| 2026-08-10 | `2608.09771v1` | SLIM-0.5B: Learning Action-Grounded Predictive Latents for Robot Manipulation |
| 2026-08-10 | `2609.26118v1` | GDLAM: Group-Disentangled Latent Action Model for Highly Disentangled Embodied Pretraining |
| 2026-08-07 | `2608.07409v1` | UniJEPA: A Unified Joint-Embedding Predictive Architecture for Task-Agnostic Visual World Modeling |
| 2026-08-06 | `2608.05706v1` | LAWM-3D: Learning 3D-Aware Latent Actions from Human Videos for Generalizable Robot World Models |
| 2026-08-05 | `2608.04378v1` | Helping Music Co-Creation Agents 'Listen' Well: Hierarchical Self-Supervised World Models for Understanding an |
| 2026-07-30 | `2607.28624v1` | PhiZero: A World Model Built Around Physical Language |
| 2026-07-24 | `2607.22000v1` | Music-JEPA: Learning a World Model of Sound from Action |
| 2026-07-05 | `2607.04500v1` | Geographic Diversity Beats Data Volume for Cross-Domain Generalization in Zero-Label JEPA Driving World Models |
| 2026-07-03 | `2607.03198v2` | Reduced-Order Models: The Mother of World Models |

## 09 · 长时程与交互式生成

**`abs:"long-horizon" AND abs:"world model"`** — 命中 179，其中新增 17（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.08640v1` | RIWANav: Recursive World-Action Models with Self-Improvement for Urban Navigation |
| 2026-10-05 | `2610.06637v1` | Long-Horizon Textual World Modeling through Structured Reasoning |
| 2026-10-05 | `2610.06100v1` | From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation |
| 2026-10-04 | `2610.04916v1` | PreAct-Nav: Agentic Reasoning Before Action for Urban Navigation |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-10-01 | `2610.02368v1` | Rethinking World-Action Model for Compositional and In-Context Robotic Manipulation |
| 2026-10-01 | `2610.00976v2` | Variational Streaming Flow: Probabilistic Forecasting in Physical Time |
| 2026-09-30 | `2609.40003v1` | DashVMC: Real-Time Discrete World Model Control in Geometry Dash |
| 2026-09-30 | `2609.39727v1` | OverForge: Reasoning Through Strategies and Tactics Helps Cooperative Lifelong Adaptation |
| 2026-09-30 | `2609.39101v1` | Beyond Prediction: Steering VLM Agents with Retrospective World Modeling |
| 2026-09-30 | `2609.38839v1` | FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation |
| 2026-09-29 | `2609.38562v1` | LongTake: Learning to Sustain Dynamics in Long-Horizon Video Generation |
| 2026-09-29 | `2609.36851v2` | RoXDrive: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving via Action-Faithful Rollouts |
| 2026-09-28 | `2609.36344v1` | DeepRewind: Predicting and Repairing Premature Commitments in Deep Research Agents |
| 2026-09-28 | `2609.35375v1` | From Pixel to Poses: Object-centric Tool Manipulation Learning from Human Demonstrations |
| 2026-09-28 | `2609.35138v2` | FlexiWorld: Learning and Planning via Flexible Action Chunks Across Multiple Time Scales |
| 2026-09-28 | `2609.34470v1` | Precise Editing and Flexible Referencing for Interactable Worlds |

**`abs:"interactive world model"`** — 命中 35，其中新增 21（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09785v1` | UltraWorld: Learning Interactive Ultrasound World Models from Untracked Clinical Videos with Acoustic Sampling |
| 2026-10-06 | `2610.08941v1` | SPW-Nav: A Streaming Panoramic World Model for Language-Guided Navigation |
| 2026-10-04 | `2610.04920v1` | PWM: Personalized World Models with Online Reinforcement Learning |
| 2026-10-01 | `2610.02331v1` | World Editing: Intervening on Executable Worlds at Increasing Depth |
| 2026-09-29 | `2609.38123v1` | HelixWorld: A Real-time Interactive Audio-Visual World Model |
| 2026-09-28 | `2609.34144v1` | CAST: Reconstruction-Coupled Acceleration of Interactive World Models |
| 2026-09-25 | `2609.31893v1` | CyberWorld: World Models for Sample-Efficient Autonomous Cyber Defense |
| 2026-09-07 | `2609.07532v1` | PhysReal: Learning Real-World Deformable Object Physics via Hybrid Constitutive Modeling |
| 2026-09-07 | `2609.07051v1` | TrojanWorld: Backdooring World-Model Agents via Imagination Steering |
| 2026-09-01 | `2609.01560v1` | H3-World: Turning Language Understanding into World Control |
| 2026-08-26 | `2608.25479v3` | 4DStreamCtrl: Interactive Video Generation with Online 4D Control |
| 2026-08-18 | `2608.21439v1` | WorldMind: Decoupled Game World Model for State-Aware NPC Behavior |
| 2026-08-14 | `2608.14022v1` | ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models |
| 2026-08-13 | `2608.13546v2` | Alaya-EVOKE: From Linear-Scaling Supervision to Endless World |
| 2026-08-10 | `2608.09449v2` | Sekai2: From World Exploration to Interactive World Modeling |
| 2026-07-30 | `2607.28624v1` | PhiZero: A World Model Built Around Physical Language |
| 2026-07-30 | `2607.28362v1` | ShadowDancer: Teaching Video World Models Any Action by Learning Unified Dynamics Representations from a Video |
| 2026-07-23 | `2607.21594v1` | Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers |
| 2026-07-21 | `2607.18703v1` | Generative World Renderer at the Speed of Play |
| 2026-07-11 | `2607.10389v1` | Stateful Worlds, Stateless Elasticity: Exact-State Serving for Interactive World Models |
| 2026-07-07 | `2607.06216v2` | MoWorld: A Flash World Model |

**`abs:"action-conditioned" AND abs:"video generation"`** — 命中 24，其中新增 21

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-30 | `2609.40153v1` | Dream4ACT: A Shared Visual Action Interface for Multi-Embodiment Video-Action Modeling |
| 2026-09-30 | `2609.38839v1` | FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation |
| 2026-09-29 | `2609.38059v1` | WorldLine: Action-Driven Visual Simulation for Robotic Manipulation |
| 2026-09-23 | `2609.28811v1` | DeltaWAM: Delta World Action Models for Bimanual Manipulation |
| 2026-08-31 | `2608.30897v1` | CAER: Causal Action Effect Reweighting for World Model Training |
| 2026-08-27 | `2608.27406v2` | CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators |
| 2026-08-25 | `2608.24199v2` | NVIDIA Cosmos-H-Dreams: Real-Time Generative Physics Simulation for Surgical Robotics |
| 2026-08-23 | `2608.22403v1` | LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models |
| 2026-08-18 | `2608.18077v1` | Hydra-0: Action Flow for Generalist World Modeling and Control |
| 2026-08-14 | `2608.14022v1` | ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models |
| 2026-07-31 | `2607.29302v1` | BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning |
| 2026-07-29 | `2607.26579v1` | ContactFlow: A video action conditioning that transfers across embodiments |
| 2026-07-07 | `2607.06018v1` | RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures |
| 2026-06-28 | `2606.29501v1` | Learning Transferable Dynamics Priors from Action to World Modeling |
| 2026-06-24 | `2606.26410v2` | Neural Voxel Dynamics: Learning Volumetric Feature Advection for 3D Physics in V-JEPA Latent Space |
| 2026-06-24 | `2606.25473v1` | Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation i |
| 2026-06-22 | `2606.23296v1` | IOI: Decoupling Kinematics and Physics for Interactive World Models |
| 2026-06-17 | `2606.18610v3` | SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation |
| 2026-06-16 | `2606.18180v1` | EgoCS-400K: An Egocentric Gameplay Dataset for World Models |
| 2026-06-14 | `2606.15768v1` | LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies |
| 2026-06-13 | `2606.15341v1` | CausalDrive: Real-time Causal World Models for Autonomous Driving |

**`abs:"object permanence"`** — 命中 13，其中新增 13

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-02 | `2610.03713v1` | What Should World Models Forget? Stratified Retention for Continual Adaptation |
| 2026-10-01 | `2610.01201v1` | iSEE: Object Permanence Through Self-Supervision |
| 2026-09-28 | `2609.34554v1` | Where Memory Belongs: Ledger, an Object Ledger for Memory-Augmented VLAs |
| 2026-09-23 | `2609.28654v1` | Training Object Permanence in World Models |
| 2026-09-17 | `2609.20819v2` | Can 4D Foundation Models Remember? |
| 2026-08-27 | `2608.26794v1` | Ring Forcing: Towards Precise Long-Term Memory for Autoregressive Video Diffusion |
| 2026-08-03 | `2608.02289v1` | Extended Field of View Analysis for VideoGAN-based Trajectory Generation |
| 2026-08-03 | `2608.01614v1` | Linear Multi-Timescale Retention as a Memory-Efficient Vision-Language Bridge |
| 2026-07-14 | `2607.12231v1` | The GEST-Engine: From Event Graphs to Synthetic Video. A Full Technical Report |
| 2026-07-10 | `2607.09138v1` | BeyondSight: Object Permanence for End-to-End Autonomous Driving |
| 2026-06-29 | `2606.30481v1` | Situation Perception: A Necessary Primitive to Artificial Superintelligence |
| 2026-06-26 | `2606.28455v3` | Event-Conditioned Diagnostics of Kinematic, Contact, and Object-Permanence Structure in Passive Object-State W |
| 2026-06-15 | `2606.20707v1` | GEOPHYS: The Geometry of Physical Plausibility |

**`abs:"autoregressive" AND abs:"video prediction"`** — 命中 4，其中新增 3

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.09090v1` | Tucker Bottleneck Attention for Multi-Dimensional Sequence Modeling |
| 2026-09-16 | `2609.18430v1` | StrucPhysVideo: Learning Physical Dynamics from Structured Captions and Robot Actions |
| 2026-07-14 | `2607.13031v1` | The Seriality Gap in Video Diffusion Models |

**`abs:"memory" AND abs:"long-horizon" AND abs:"navigation"`** — 命中 41，其中新增 26（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-04 | `2610.05024v1` | LightVLN: Efficient Aerial Vision-and-Language Navigation with Compact Memory and History-Guided Local Aggrega |
| 2026-10-04 | `2610.04916v1` | PreAct-Nav: Agentic Reasoning Before Action for Urban Navigation |
| 2026-09-30 | `2609.39915v1` | NavHarness: Adaptive Goals for Agentic Vision-Language Navigation |
| 2026-09-28 | `2609.35431v1` | Memory in the Sky: Low-Altitude Question Answering with Multi-Agent Memory Aggregation |
| 2026-09-28 | `2609.34163v1` | Reliability-Aware Sparse Route Memory for Round-Trip Vision-Language Navigation |
| 2026-09-26 | `2609.32626v1` | World SLAM Model: Joint World Modeling for SLAM and Navigation |
| 2026-09-23 | `2609.27612v1` | RegenHarness: A Robot Agent Harness with Evidence-Gated Recursive Self-Improvement |
| 2026-09-22 | `2609.25666v1` | Deploying Foundation Models for Embodied Navigation |
| 2026-09-20 | `2609.23465v1` | Propose, Verify, Commit: Evidence-Grounded Memory for Long-Horizon Multi-Actor Conversations |
| 2026-09-08 | `2609.08442v1` | AirAnchor: Bridging Local and Global Spatial Information for Zero-Shot Aerial Vision-and-Language Navigation |
| 2026-09-08 | `2609.08159v1` | OmniNav: Robust Long-Horizon Target Navigation in Dynamic Environments |
| 2026-08-31 | `2608.30396v1` | Scaffolding Foundation Models into Physical-World Agents Pushes the Frontier of Long-Horizon Navigation |
| 2026-08-30 | `2608.29596v2` | A Systematic Survey of Agentic Skills: Architecture, Lifecycle, and Security |
| 2026-08-29 | `2608.29114v1` | CGFM-Nav: Cognitive Graph-Field Memory for Semantic-Guided Lifelong Multimodal Embodied Navigation |
| 2026-08-26 | `2609.29555v1` | Visual Representation and History Modeling for Navigation World Models |
| 2026-08-24 | `2608.23383v2` | Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds |
| 2026-08-21 | `2608.21690v1` | Context as an Environment: Programmatic Context Management for Long-Horizon Agents |
| 2026-08-04 | `2608.05013v1` | OneDayAgent: Towards a Long-Horizon Harness for Autonomous Agents |
| 2026-08-02 | `2608.01456v1` | Long-Horizon Embodied Decision-Making via Multimodal Memory Compression |
| 2026-07-31 | `2607.29600v1` | HAM-VLN: Harnessing Hierarchical Agentic Memory for Zero-Shot Vision-and-Language Navigation |
| 2026-07-26 | `2607.23504v1` | MemVLN: Episodic and Procedural Memory for Vision-and-Language Navigation |
| 2026-07-20 | `2607.17599v1` | ConsiSpace: Learning Geometric Consistency Matters for Video Spatial Reasoning |
| 2026-07-19 | `2607.17038v1` | Reward-Driven LLM Agent Workflows: Synthesizing POMDP Routing and Self-Correction for Autonomous Decision-Maki |
| 2026-07-14 | `2607.12370v1` | StratMamba: Strategic and Reactive Stream Partitioning for Path-Efficient LiDAR-Based Obstacle Avoidance |
| 2026-07-13 | `2607.11377v1` | A Glimpse into Long-term Physical Coexistence with Intelligent Robots |
| 2026-07-11 | `2607.10350v3` | ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory |

## uav · 无人机主线

**`abs:"UAV" AND abs:"world model"`** — 命中 17，其中新增 14

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09335v1` | SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models |
| 2026-09-29 | `2609.36845v1` | DSWM: Decomposed Spatio-Temporal World Model for Demand-Driven UAV Base Station Repositioning |
| 2026-09-20 | `2609.23656v1` | WOLF: World Model Guided LiDAR Exploration with Predictive Frontiers |
| 2026-09-16 | `2609.18326v1` | UAVs Meet Embodied Intelligence: Bridging Human Intents and Flying Dynamics Via Harnessing Physical-Digital AI |
| 2026-08-31 | `2609.00106v1` | Deploying and Evaluating a Smart-Agriculture Agentic Engine for Full-Season Soybean Farm Operations |
| 2026-08-20 | `2608.20126v1` | RMWorld: Task-Aware Radio World Models with Value-of-Information Guided Multi-Trial Learning for Multi-UAV Com |
| 2026-08-06 | `2608.05792v1` | When Agentic AI Meets Integrated Sensing and Communication |
| 2026-08-06 | `2608.05597v1` | Uncertainty-Aware World Model for Aerial Image-Goal Navigation |
| 2026-07-30 | `2607.27865v1` | Learning to Understand Body Language from Flight through Robust 3D Avatar Placing |
| 2026-07-28 | `2607.25728v1` | Shared Voxel-Map-Based Cooperative Indoor UAV Guidance with a Multi-Agent Soft Actor-Critic Controller |
| 2026-07-22 | `2607.19719v2` | Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model Imagination |
| 2026-07-10 | `2607.09078v1` | Toward Active Object Detection for UAVs in the Wild: A Large-Scale Dataset, Benchmark and Method |
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |
| 2026-07-05 | `2607.04352v1` | Last-Meter Precision Navigation for UAVs: A Diffusion-Refined Aerial Visual Servoing Approach |

**`abs:"aerial" AND abs:"world model"`** — 命中 18，其中新增 14

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09335v1` | SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models |
| 2026-09-29 | `2609.36845v1` | DSWM: Decomposed Spatio-Temporal World Model for Demand-Driven UAV Base Station Repositioning |
| 2026-09-23 | `2609.27621v2` | SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design |
| 2026-09-20 | `2609.23656v1` | WOLF: World Model Guided LiDAR Exploration with Predictive Frontiers |
| 2026-09-16 | `2609.18326v1` | UAVs Meet Embodied Intelligence: Bridging Human Intents and Flying Dynamics Via Harnessing Physical-Digital AI |
| 2026-08-20 | `2608.20126v1` | RMWorld: Task-Aware Radio World Models with Value-of-Information Guided Multi-Trial Learning for Multi-UAV Com |
| 2026-08-06 | `2608.05792v1` | When Agentic AI Meets Integrated Sensing and Communication |
| 2026-08-06 | `2608.05597v1` | Uncertainty-Aware World Model for Aerial Image-Goal Navigation |
| 2026-08-02 | `2608.01049v1` | FactorJEPA: Factorizing Monolithic Futures into Layout-Agent-Interaction Channels for Crowded and Chaotic Glob |
| 2026-07-30 | `2607.27865v1` | Learning to Understand Body Language from Flight through Robust 3D Avatar Placing |
| 2026-07-10 | `2607.09661v1` | PanoWorld: Real-World Panoramic Generation |
| 2026-07-10 | `2607.09078v1` | Toward Active Object Detection for UAVs in the Wild: A Large-Scale Dataset, Benchmark and Method |
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |
| 2026-07-05 | `2607.04352v1` | Last-Meter Precision Navigation for UAVs: A Diffusion-Refined Aerial Visual Servoing Approach |

**`abs:"drone" AND abs:"world model"`** — 命中 7，其中新增 6

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-08-31 | `2609.00106v1` | Deploying and Evaluating a Smart-Agriculture Agentic Engine for Full-Season Soybean Farm Operations |
| 2026-07-30 | `2607.27865v1` | Learning to Understand Body Language from Flight through Robust 3D Avatar Placing |
| 2026-07-28 | `2607.25728v1` | Shared Voxel-Map-Based Cooperative Indoor UAV Guidance with a Multi-Agent Soft Actor-Critic Controller |
| 2026-07-20 | `2607.17669v2` | Attention from Above: A Multimodal Model for Drone-Based Object Localization |
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |
| 2026-06-23 | `2606.24152v2` | Autonomous Video Generation with Counterfactual Controllability for Self-Evolving World Models |

**`abs:"aerial" AND abs:"video generation"`** — 命中 0。**复核过：纯粹是窗口假象。**不限窗口命中 **6**，最近三条就在窗口外几周：`2605.19728`（Aero-World，2026-05-19）、`2605.15964`（WorldVLN，2026-05-15）、`2604.07991`（MotionScape，2026-04-09）。同窗口换提法 `abs:"drone" AND abs:"video generation"` 命中 **1**（`2606.24152`）、`abs:"UAV" AND abs:"video generation"` 命中 **1**（`2610.02451`）。**这是本轮最该记的一条：窗口起点卡在 6 月 10 日，把 2026 年 4–5 月空中世界模型那一批整批切掉了**——写「近期空白」之前必须先看不限窗口的命中。

**`abs:"UAV" AND abs:"video prediction"`** — 命中 0。**复核过：同上是窗口假象。**不限窗口命中 **2**：`2606.06147`（WorldFly，2026-06-04，比窗口起点早 6 天）、`2512.21710`（RAPTOR，2025-12-25）。同窗口换提法 `abs:"aerial" AND abs:"video prediction"`、`abs:"aerial" AND abs:"future frame prediction"` 也都命中 **0**。**不得写成「无人机不做视频预测」。**

**`abs:"aerial" AND abs:"navigation" AND abs:"world model"`** — 命中 5，其中新增 4

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-16 | `2609.18326v1` | UAVs Meet Embodied Intelligence: Bridging Human Intents and Flying Dynamics Via Harnessing Physical-Digital AI |
| 2026-08-06 | `2608.05597v1` | Uncertainty-Aware World Model for Aerial Image-Goal Navigation |
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |
| 2026-07-05 | `2607.04352v1` | Last-Meter Precision Navigation for UAVs: A Diffusion-Refined Aerial Visual Servoing Approach |

---

*本文件由 `tools/watch.py` 生成，重跑即可刷新。收录进正文前，ID 与标题一律以 arXiv API 为准。*
