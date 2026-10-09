# VLA 前沿增量 · 2026-10

> 生成：`py -3.9 tools/watch.py --volume vla --write`｜窗口 2026-06-11 ~ 2026-10-09（120 天）｜词表 35 条｜请求失败 0 条｜新增 441 条

词表按 docs/03-VLA专题/06–10 五篇 分节。

**读法**（纪律 F 的镜像规则）：TOTAL 是**题摘层面的命中数，不是相关论文数**。一条宽查询命中几百条不含任何信息，所以下表只列**仓库尚未收录**的条目。`NULL` 表示请求失败（429/超时），**不是 0 命中**；`0` 表示确实没有，但第一反应应当是怀疑查询措辞，换个提法复核后再下结论。

arXiv 的 `all:` / `abs:` 只覆盖题名/摘要/作者/注释，**不覆盖全文**。

**去重范围是整仓**：`docs/`、`references/`、`mindmaps/`、`paper/` 下的 `.md`，加 `README.md` / `CONTRIBUTING.md` / `code/README.md`。`paper/` 下的 `_work*/` 取数缓存（json/xml/html/pdf）**不算收录**，故不计入——那里放的是原始查询结果，不是读过的文献。所以某条 ID 不在本表，只能推出**整仓没写过**，推不出"这个专题没有"。
（2026-10-08 修：`paper/` 原先漏扫，加上 ID 正则不吃 `v1` 版本号后缀，两条合起来让 27 条早已在 `paper/` 笔记里读过的论文被标成"新增"。三卷的台账互为已收录，故 `<卷>-watch-*.md` 一律排除。）

## 06 · 动作头与动作分块

**`abs:"action chunking"`** — 命中 232，其中新增 21（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10528v1` | Long-WAM: Scaling the Context of World-Action Models |
| 2026-10-07 | `2610.10367v1` | Temporally Interpretable Differentiable Decision Trees |
| 2026-10-06 | `2610.09016v1` | PAIR: Bridging Perception and Action in Vision-Language-Action Models |
| 2026-10-06 | `2610.08150v1` | ViDAL: A Visual Dynamics-Grounded Action Latent Space for Vision-Language-Action Models |
| 2026-10-06 | `2610.07946v1` | Adapting Vision-Language-Action Models to Unknown Visual Disruptions During Execution |
| 2026-10-06 | `2610.07756v1` | StairVLA: Stage-Aware Hierarchical Action Generation for Vision-Language-Action Models |
| 2026-10-06 | `2610.07752v1` | CoRE: Learning Collaboration-Role Experts for Decentralized Collaborative Manipulation with One Policy |
| 2026-10-06 | `2610.07696v1` | ESP: Energy-Score Policy for One-Step Multimodal Action Generation |
| 2026-10-05 | `2610.07527v1` | Task-Space Imitation Guidance for Efficient Reinforcement Learning |
| 2026-10-05 | `2610.06104v1` | Conditional Trajectory Peaks: Single-Pass Multimodal Policies over Action Chunks |
| 2026-10-05 | `2610.07056v1` | Behavioral Cloning Mystery |
| 2026-10-04 | `2610.05331v1` | VAMPS: Visual and Motor Policies from Sampling-Based Planning |
| 2026-10-03 | `2610.04659v1` | PermVLA: Factorization Order as a Regularizer for VLA Learning |
| 2026-10-03 | `2610.04607v1` | ForeAct3D: Policy-Grounded Future World Modeling for VLA Policies |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-10-02 | `2610.02759v1` | Proprioceptive Sketches as Long-Horizon Intent for Generative Action Policies |
| 2026-10-02 | `2610.02666v1` | CHASE-VLA: Post-Training Quantization Framework for Vision-Language-Action Models with Chunk-Aware Scale Estim |
| 2026-10-02 | `2610.02626v1` | Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination |
| 2026-10-01 | `2610.02323v1` | World-Calibrated Proposal-to-Action Flow for Vision-Language-Action Models |
| 2026-10-01 | `2610.01258v1` | ColoACT: Multi-Cue Action Chunking for Smooth Autonomous Colon Navigation on a Self-Propelled Endoscopic Robot |
| 2026-09-30 | `2609.39970v1` | PhasePlan: Ordered Future-Phase Planning for Robot Brain Models |

**`abs:"action head" AND abs:"vision-language"`** — 命中 43，其中新增 22（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09940v1` | Juno: Taming Predictive Latents for Vision-Language-Action Models |
| 2026-10-07 | `2610.09708v1` | Black-Box Adversarial Patch Attacks on VLAs via Ancestor VLM Exploitation |
| 2026-10-06 | `2610.07756v1` | StairVLA: Stage-Aware Hierarchical Action Generation for Vision-Language-Action Models |
| 2026-10-06 | `2610.07696v1` | ESP: Energy-Score Policy for One-Step Multimodal Action Generation |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-09-28 | `2609.36118v1` | The Layer Mystery of VLA: An Information-Theoretical Analysis of VLA Latent Interface |
| 2026-09-25 | `2609.30913v1` | Causeway: Restoring Task Accessibility for Instruction Switching in VLA Policies |
| 2026-09-19 | `2609.22854v1` | SmoLSTM: A Compact Vision-Language-Action Model with Recurrent Memory that Persists |
| 2026-09-18 | `2609.21461v1` | AtomEgo: Exploring Ego-Robot Integration for Embodied Foundation Model Pretraining |
| 2026-09-16 | `2609.22335v1` | KerColle: Unlocking Fine-Grained GPU Concurrency in Vision-Language-Action Models |
| 2026-09-16 | `2609.18374v2` | Decoupling Vision, Language, and Action for Efficient Multi-Task Robot Policies |
| 2026-09-16 | `2609.18084v2` | Not All Layers Need Tuning: Diagnosing and Directing Adaptation in Vision-Language-Action Models |
| 2026-09-15 | `2609.17210v1` | FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence |
| 2026-09-15 | `2609.17035v1` | SWIM: Vision-Language-Grounded Soft Whole-Body Interactive Manipulation |
| 2026-09-15 | `2609.16641v1` | SAVLA: Symmetry-Aware Vision-Language-Action Models for Robotic Manipulation |
| 2026-08-31 | `2608.30378v3` | PAVE: Predictive Alignment and Value-Guided Evolution for World-Action Policies |
| 2026-08-23 | `2608.22591v3` | WorldToken: Time-First Sequence Modeling for Robotic Imitation Learning |
| 2026-08-07 | `2608.07619v1` | GWM-VLA: Geometry-Aware Latent World Modeling for Vision-Language-Action Learning |
| 2026-08-06 | `2608.06374v1` | DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control for Cross-Embodiment Manipulation |
| 2026-08-05 | `2608.04692v1` | Suppression Sticks, Locality Is Fragile: A Closed-Loop Target-and-Control Audit of Task-Vector Negation in VLA |
| 2026-08-05 | `2608.04510v1` | GUARD: Grounding Uncertainty and Ablation-Based Risk Detection for Diffusion-Based VLAs |
| 2026-08-04 | `2608.03727v1` | Track4Action: Distilling World-Centric 3D Tracker into Vision-Language-Action Policies |

**`abs:"flow matching" AND abs:"robot policy"`** — 命中 18，其中新增 18

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10534v1` | RoboPrompt: Intuitive Robot Policy Steering with Sparse Human Input |
| 2026-10-06 | `2610.08789v1` | QF3: Fast Flow RL with Filtered Q-Gradients |
| 2026-10-06 | `2610.08784v1` | PEARS: Physical-Prior-Guided Efficient Adaptation via Failure Reasoning and Diffusion Steering for Tactile Man |
| 2026-10-06 | `2610.07696v1` | ESP: Energy-Score Policy for One-Step Multimodal Action Generation |
| 2026-10-05 | `2610.06089v1` | Adaptive Mean Flow for Responsive Closed-Loop Robot Control |
| 2026-09-30 | `2609.40165v1` | PrefPI: Preference-Guided Steering into Out-of-Distribution Behaviors |
| 2026-09-29 | `2609.36872v1` | PreferenceFlow: Test-Time Guidance of Flow-Matching Robot Policies from Human Interventions |
| 2026-09-29 | `2609.36413v3` | One from Infinity: Actualizing Futures from Pretrained World Models into Robot Actions |
| 2026-09-28 | `2609.35231v2` | Zero-Shot Reactive Obstacle Avoidance for Generative Robot Policies |
| 2026-09-27 | `2609.33765v1` | Principal Steering Subspaces for Online Adaptation of Frozen Generative Robot Policies |
| 2026-09-26 | `2609.32236v1` | RoboFFT: Finetuning generative robot policy via online reinforcement learning with forward process |
| 2026-09-14 | `2609.15014v1` | Steering Generative Robot Policies with Lexicographic Preferences |
| 2026-09-03 | `2609.03715v1` | MINERVA: How Small Can a Manipulation Policy Be and Still Solve LIBERO? |
| 2026-08-20 | `2608.20208v1` | RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation |
| 2026-07-28 | `2607.25918v1` | DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models |
| 2026-07-09 | `2607.08877v1` | FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space |
| 2026-06-16 | `2606.18043v2` | Uncertainty Quantification for Flow-Based Generalist Robot Policies |
| 2026-06-16 | `2606.17408v1` | Where Should Action Generation Begin? A Learnable Source Prior for Generative Robot Policies |

**`abs:"diffusion policy" AND abs:"manipulation"`** — 命中 77，其中新增 29（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09369v1` | Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment |
| 2026-10-05 | `2610.05755v2` | Demonstration-Calibrated Port-Hamiltonian Retuning for Manipulation Policies |
| 2026-10-04 | `2610.07015v1` | LEAP: Making Privileged Geometry Supervision Effective for Visuomotor Learning |
| 2026-10-03 | `2610.04536v1` | WasserMan: Benchmark for Underwater Manipulation Policy Learning |
| 2026-10-02 | `2610.03333v1` | Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation |
| 2026-10-02 | `2610.02706v1` | AdaTempo: Learning Shared Relative Tempo from Demonstrations for Faster Robot Manipulation |
| 2026-09-30 | `2609.39599v1` | Text-to-3D Policy: Fine-Grained Language-Behavior Alignment for Unseen Specification Generalization |
| 2026-09-29 | `2609.36575v1` | EquivDP3: A SIM(3)-Invariant Point-Cloud Encoder for Data-Efficient Humanoid Loco-Manipulation |
| 2026-09-26 | `2609.33007v1` | CAPEX: Efficiently Distilling Foundation Model Behavior into Deployable Robot Policies through Experience-Adap |
| 2026-09-26 | `2609.32129v2` | Residual Denoising Enables Sample-Efficient Multi-Agent Coordination on Demand |
| 2026-09-24 | `2609.30521v1` | Aerial Manipulation in the Wild with Onboard Perception, Policy Learning, and Whole-Body Control |
| 2026-09-23 | `2609.28818v1` | KeyGen: Unsupervised Keypoint based Object-Centric Representations for Category-Level Policy Generalization |
| 2026-09-22 | `2609.25506v1` | RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation |
| 2026-09-21 | `2609.25322v2` | JAMB: Joint Action-Motion Diffusion for Bimanual Manipulation |
| 2026-09-21 | `2609.24660v2` | Touch2Robot: Robot Touch in the Human Demonstration Loop |
| 2026-09-19 | `2609.22829v2` | Whole-Body UMI: Transferring UMI Manipulation Skills to Humanoid Whole-Body Manipulation via Real-Time Motion  |
| 2026-09-19 | `2609.22730v1` | BEACON: Belief-Enabled Adaptive CONtrol for Imitation Learning under Uncertainty |
| 2026-09-18 | `2609.21621v2` | Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Fo |
| 2026-09-18 | `2609.21448v1` | Robotic Multiphase Interaction: Manipulating Coupled Liquid and Solid Dynamics with a World Model |
| 2026-09-17 | `2609.20669v1` | Learning Foresight without Explicit Trajectories for 3D Diffusion Policies |
| 2026-09-16 | `2609.18930v1` | Learning Holistic Whole-Body Loco-Manipulation with a Bipedal Mobile Manipulator |
| 2026-09-16 | `2609.19200v1` | ULOHA: An Underwater Bimanual Robot System for Robot Learning |
| 2026-09-15 | `2609.17714v1` | Vision-Language Grounded Task-Context-Aware Imitation Learning for Robotic Disassembly |
| 2026-09-14 | `2609.15726v1` | Bench2Dex: Benchmarking Visuo-Tactile Bimanual Dexterous Manipulation Across Dexterous Hands |
| 2026-09-14 | `2609.15162v2` | LieSpline-DP: Lie-Group B-Spline Diffusion Policy for Smooth Robot Manipulation |
| 2026-09-11 | `2609.12634v1` | Online Material Estimation for Conditioned Diffusion Policy in Shaping Deformable Linear Objects |
| 2026-09-10 | `2609.12245v2` | DIA: Denoising Intermediate Advantage for Diffusion Policy Optimization |
| 2026-09-10 | `2609.13318v1` | Attention-DP3: Spatially Object-aware 3D Diffusion Policy via Geometry-aligned Attentional Conditioning |
| 2026-09-10 | `2609.12103v2` | RodForesight: A World Model Enhanced Diffusion Policy for Slender Rod Insertion |

**`abs:"action tokenization"`** — 命中 81，其中新增 29（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.09178v1` | CAP: Codebook-Aligned Prediction for Tokenized Robot Policies |
| 2026-10-06 | `2610.09170v1` | Beyond Reconstruction: What Matters in Action Tokenization for Robot Policies? |
| 2026-10-06 | `2610.09016v1` | PAIR: Bridging Perception and Action in Vision-Language-Action Models |
| 2026-10-06 | `2610.08150v1` | ViDAL: A Visual Dynamics-Grounded Action Latent Space for Vision-Language-Action Models |
| 2026-10-03 | `2610.04805v1` | ExStereo: Lifting 2D Vision-Language-Action Models to 3D with Explicit Stereo Representations |
| 2026-10-02 | `2610.02759v1` | Proprioceptive Sketches as Long-Horizon Intent for Generative Action Policies |
| 2026-10-01 | `2610.00949v1` | PG-SFT: Balancing Capability Acquisition and Retention in Offline Agent Fine-Tuning |
| 2026-10-01 | `2610.00899v1` | TOAST: Stochastic Robot Action Tokenization for Autoregressive Vision-Language-Action Models |
| 2026-09-30 | `2609.40219v2` | Learning Skills from Historical Action Trajectories: Action Experience Dictionary for World Action Models |
| 2026-09-30 | `2610.00437v1` | JevSpawn: Adaptive Agentic Inference through Compositional Action Spaces |
| 2026-09-30 | `2609.39973v1` | EWAM: Emergent Depth-Wise Specialization in a Unified Embodied Model -- From Semantic Understanding through Vi |
| 2026-09-30 | `2609.39526v1` | Discrete Forcing: Infusing Discrete Guidance into Continuous Denoising for Few-Step Action Experts |
| 2026-09-30 | `2609.39518v1` | Referential Uncertainty in Human--AI Collaboration |
| 2026-09-30 | `2609.39324v1` | MotionWeave: Learning Motion-Centered Future Dynamics for Vision-Language-Action Policies |
| 2026-09-30 | `2609.39056v1` | SteerQuant: Steering Quantization Error with Action-Guided Scaling in World-Action Models |
| 2026-09-30 | `2609.38984v1` | Sparse-WAM: Accelerating World Action Models via Action-Guided Sparse Imagination |
| 2026-09-29 | `2609.37522v2` | Graph-Conditioned On-Policy Agent Distillation from Off-the-Shelf Teachers |
| 2026-09-28 | `2609.35749v1` | Towards Communication-Efficient Social Intelligence in Language Agents |
| 2026-09-28 | `2609.35709v1` | Humanoid Loco-Manipulation With Discrete VLA Model |
| 2026-09-28 | `2609.35540v1` | Continuous Context Management |
| 2026-09-28 | `2609.35469v1` | Rethinking Causal Action Tokenization with Conditional Annealing in Flow Matching |
| 2026-09-24 | `2609.28865v1` | Direction-Scale Decomposition in Action Representation: Rethinking What to Tokenize for Vision-Language-Action |
| 2026-09-23 | `2609.27513v1` | Behavior-Aligned Action Tokenization for Robot Policy Learning |
| 2026-09-22 | `2609.25820v1` | Beyond Reconstruction Error: Analytical and Data-Driven Action Tokenization for Autoregressive Vision-Language |
| 2026-09-20 | `2609.23445v1` | BiRoAD: Learning Shared and Role-Adaptive Representations for Bimanual Manipulation |
| 2026-09-17 | `2609.20715v1` | Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL |
| 2026-09-17 | `2609.19894v1` | Learning Reliable Parking Policies via Offline Reinforcement Learning with Quantized Action Representations |
| 2026-09-16 | `2609.18623v1` | FIVE-VLA: Fast and EffectIVE Autonomous Driving with Recurrent Action Memory |
| 2026-09-16 | `2609.18487v1` | ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models |

**`abs:"receding horizon" AND abs:"imitation learning"`** — 命中 0。**复核过：这是词汇层面的假空白。**换提法 `abs:"execution horizon"` 命中 **22**、`abs:"action horizon"` 命中 **11**（同 120 天窗口）。「receding horizon」是控制论的说法，VLA 社区写「action / execution horizon」。**不得记录为「这个方向没人做」。**

## 07 · 数据、预训练与跨具身

**`abs:"cross-embodiment"`** — 命中 89，其中新增 24（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.07650v1` | Silicon Language: A Robot-Native Knowledge Exchange Framework for Heterogeneous Robots |
| 2026-10-06 | `2610.07597v1` | The Robot Is Not Its Description: GaugeBench for Representation Robustness in Morphology-Aware Policies |
| 2026-10-02 | `2610.03861v1` | GOTT: Object-centric Dexterous Manipulation with a Reusable Cross-Embodiment Primitive |
| 2026-10-02 | `2610.03607v1` | World Action Learning via Interaction-Centric Spectral Latent Guidance |
| 2026-10-02 | `2610.03278v1` | DexJoCo-X: Benchmarking Action Representations for Multi-Hand Dexterous Manipulation |
| 2026-10-02 | `2610.03248v1` | EmbPASS: Towards Cross-Embodiment Open Panoramic Segmentation |
| 2026-10-01 | `2610.01742v1` | World Motion Models: Flexible Sequence Modeling of SE(3) Trajectories |
| 2026-09-30 | `2609.40219v2` | Learning Skills from Historical Action Trajectories: Action Experience Dictionary for World Action Models |
| 2026-09-30 | `2609.39973v1` | EWAM: Emergent Depth-Wise Specialization in a Unified Embodied Model -- From Semantic Understanding through Vi |
| 2026-09-30 | `2609.39006v1` | Function beyond Form: Functional Correspondence for Cross-Embodiment Dexterous Grasp Generation |
| 2026-09-29 | `2609.38087v1` | CrossBFM: Distilling a Shared Latent Behavior Space Across Humanoid Embodiments |
| 2026-09-29 | `2609.37519v1` | Video2STL: Grounding VLM-Generated Temporal Specifications for Robot Learning |
| 2026-09-28 | `2609.34414v1` | From World Models to World Action Models: Rethinking Next-State Prediction |
| 2026-09-25 | `2609.31207v1` | Enabling a Unified Cross-Domain Representation for Two-Finger Gripper Manipulation via Interaction-Centric Mod |
| 2026-09-22 | `2609.27095v1` | Intelligence Across Embodiments |
| 2026-09-18 | `2609.22521v2` | Latent Policy Steering: An Efficient and Flexible Framework for Cross-Embodiment Transfer |
| 2026-09-18 | `2609.21983v1` | SkelWAM: A Skeleton-Guided World-Action Model for Zero-Shot Cross-Embodiment Manipulation |
| 2026-09-18 | `2609.21461v1` | AtomEgo: Exploring Ego-Robot Integration for Embodied Foundation Model Pretraining |
| 2026-09-16 | `2609.18504v1` | InterMASH: A Unified Geometric Representation for Grasp Synthesis |
| 2026-09-14 | `2609.15213v1` | X-WBC: A Cross-Embodiment Foundation Model for Humanoid Whole-Body Control |
| 2026-09-10 | `2609.11753v1` | SEED-UMI: Sharing the Exoskeleton between human and robot for onE-to-one Dexterous demonstration |
| 2026-09-05 | `2609.05892v1` | A4A: Cross-Embodiment Transfer of Action-Oriented 4D Affordances from Human Demonstrations |
| 2026-09-04 | `2609.05588v1` | GE-Act 2.0: Pretraining and Scaling a World-Action Model for Robotic Manipulation |
| 2026-09-03 | `2609.03927v1` | Toward Unified Robot Learning: Bridging Representation, Vision-Language-Action, and World Models |

**`abs:"robot dataset" AND abs:"pretraining"`** — 命中 15，其中新增 11

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-30 | `2609.40244v1` | StreamRig: Exploiting Intra-Rig Geometry for Streaming Multi-Camera Odometry |
| 2026-09-22 | `2609.25627v1` | MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence |
| 2026-09-18 | `2609.22085v2` | SeeQ: Training Generalist Value Functions for Long-Horizon Robotic Manipulation |
| 2026-09-11 | `2609.12721v1` | Improving Imitation Learning Efficiency for Manipulation through Geometric Prior Pretraining |
| 2026-08-25 | `2608.24042v1` | Hierarchical Skill Retrieval for Data-Efficient Adaptation of Vision-Language-Action Models |
| 2026-08-23 | `2608.22301v1` | The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction |
| 2026-08-12 | `2608.11739v1` | G0.5: One Autoregressive Stream for Robot Reasoning and Action |
| 2026-07-26 | `2607.23782v1` | $N_0$-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens |
| 2026-07-06 | `2607.04880v1` | PRISM: Personalized Robotic Dataset Generation via Image-based Scene and Motion Synthesis |
| 2026-06-29 | `2606.30988v4` | Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force |
| 2026-06-16 | `2606.18043v2` | Uncertainty Quantification for Flow-Based Generalist Robot Policies |

**`abs:"co-training" AND abs:"robot"`** — 命中 43，其中新增 24（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10384v1` | OpenViTac: Learning and Benchmarking Visuo-Tactile Policies in a Unified Sim-and-Real Framework |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-10-02 | `2610.03710v1` | EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras |
| 2026-10-01 | `2610.02338v1` | SoTa: Soft Tactile Skins for Dexterous Manipulation |
| 2026-10-01 | `2610.02054v2` | UniWAM: Unified World-Action Model |
| 2026-10-01 | `2610.02274v1` | Awomo-SimDataEngine: Agentic Simulation-ReadyWorld Generation |
| 2026-09-26 | `2609.32779v1` | Copper-Policy: Focus on the Representation for Robust Robot Manipulation |
| 2026-09-25 | `2609.30959v1` | VisTacAlign: Co-Training Dexterous Policies on Tactile Human and Robot Demonstrations |
| 2026-09-23 | `2609.28339v1` | Beyond Future Prediction: Denoising as Generative Adaptation for Robot Control |
| 2026-09-22 | `2609.26672v1` | Imperfection for Precision: Upcycling Imperfect Data for High-Precision Robotic Manipulation |
| 2026-09-18 | `2609.21461v1` | AtomEgo: Exploring Ego-Robot Integration for Embodied Foundation Model Pretraining |
| 2026-09-17 | `2609.21045v2` | DEXTERA: From a Single Image to Deployable Dexterous Manipulation via Real-to-Sim-to-Real |
| 2026-09-05 | `2609.06114v2` | Where Success Breaks: Failure-Boundary Learning for Robust Vision-Language-Action Models |
| 2026-09-04 | `2609.05588v1` | GE-Act 2.0: Pretraining and Scaling a World-Action Model for Robotic Manipulation |
| 2026-09-02 | `2609.02546v2` | ZETA: A Controlled Study of Zero-Shot Cross-Embodiment VLA Transfer for Tabletop Manipulation |
| 2026-08-17 | `2608.16885v1` | $τ_0$-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation |
| 2026-08-12 | `2608.12122v1` | HandEdit: A Unified Benchmark for Egocentric Human-to-Robot Dexterous Hand Image Editing |
| 2026-08-02 | `2608.01410v2` | GenTrack: Physical Alignment for Robot-Native Motion Generation and Zero-Shot Humanoid Tracking |
| 2026-07-30 | `2607.28405v1` | QuantWAMs: Calibrating at the Right Granularity for World Action Models |
| 2026-07-28 | `2607.25593v1` | When Does Legacy Data Start to Help? Emergent Transfer in Cross-Configuration Robot Learning |
| 2026-07-27 | `2607.24493v1` | KAI: A Kinematic-Aware Interface for Data-Efficient Articulated Object Manipulation |
| 2026-07-22 | `2607.20061v1` | ReferTrack: Referring Then Tracking for Embodied Visual Tracking |
| 2026-07-22 | `2607.19745v2` | EgoRecovery: Acquiring Failure Recovery Ability Through Human Recovery Demonstration |
| 2026-07-15 | `2607.13429v2` | Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment |

**`abs:"latent action" AND abs:"robot"`** — 命中 32，其中新增 26（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09734v1` | ΔWAM: Distilling Action Tangent Fields into World Action Models |
| 2026-10-02 | `2610.03607v1` | World Action Learning via Interaction-Centric Spectral Latent Guidance |
| 2026-10-02 | `2610.03391v1` | Native Action-Prior Learning from Videos for World Action Models |
| 2026-09-16 | `2609.18174v1` | TacBPM: A Tactile-conditioned Behavior Prior Model for Dexterous Reorientation |
| 2026-09-15 | `2609.17099v1` | GeoLAM: Learning Geometry-Grounded Latent Actions from Unlabeled Human Videos |
| 2026-09-14 | `2609.15870v1` | WLA$^3$: World Latent Action Modeling for Semantics, Dynamics, and Kinematics |
| 2026-09-14 | `2609.15189v1` | Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling |
| 2026-08-29 | `2608.29434v2` | Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations and |
| 2026-08-27 | `2608.27406v2` | CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators |
| 2026-08-25 | `2608.24882v2` | Latent Action as Intention Enables Efficient Future Imagination for World Action Models |
| 2026-08-20 | `2608.19613v2` | What Matters for Latent Actions in Robot Learning |
| 2026-08-12 | `2608.11605v2` | Foresight Without Seeing: Latent Futures for World Action Models |
| 2026-08-10 | `2609.26118v1` | GDLAM: Group-Disentangled Latent Action Model for Highly Disentangled Embodied Pretraining |
| 2026-08-07 | `2608.07619v1` | GWM-VLA: Geometry-Aware Latent World Modeling for Vision-Language-Action Learning |
| 2026-08-06 | `2608.05706v1` | LAWM-3D: Learning 3D-Aware Latent Actions from Human Videos for Generalizable Robot World Models |
| 2026-08-06 | `2608.05674v1` | JoyAI-RA 0.5: Scaling Robot Manipulation Learning via Dual Action Alignment |
| 2026-07-29 | `2607.27138v1` | DLAM: Distributional Latent Actions with Temporal Constraints |
| 2026-07-13 | `2607.11397v2` | WALA Learning Executable Latent Actions from Action-Labeled Demonstrations and Action-Free Videos |
| 2026-07-10 | `2607.09185v2` | Causally Debiased Latent Action Model for Embodied Action-Conditioned World Models |
| 2026-07-06 | `2607.04816v2` | CAC-VLA: Context-Gated Action Conditioning for Vision-Language-Action Models |
| 2026-07-01 | `2607.00678v2` | ABot-M0.5: Unified Mobility-and-Manipulation World Action Model |
| 2026-06-23 | `2606.24669v1` | LaGO: Latent Action Guidance for Online Reinforcement Learning |
| 2026-06-22 | `2606.23420v1` | Flowing With Purpose: Latent Action Guided Flow Matching Policies For Robotic Manipulation |
| 2026-06-19 | `2606.21672v1` | Imitation from Heterogeneous Demonstrations using Grounded Latent-Action World Models |
| 2026-06-19 | `2606.21139v1` | PoLAR: Factorizing Extent and Mode in Latent Actions for Robot Policy Learning |
| 2026-06-17 | `2606.18955v2` | Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos |

**`abs:"Open X-Embodiment"`** — 命中 1，其中新增 1

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-11 | `2609.13053v1` | Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model |

**`abs:"human video" AND abs:"robot policy"`** — 命中 12，其中新增 12

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10288v1` | TouchScale: 500 Hours of Human Vision and Touch for Visual-Tactile Learning |
| 2026-10-07 | `2610.09455v1` | RLHND: Video Foundation Models as Physically Grounded Hand Trackers for Robot Learning |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-09-16 | `2609.19138v1` | In-Context Robot Learning with VLM Agents |
| 2026-09-11 | `2609.12541v3` | Agent as Policy for Robotic Manipulation |
| 2026-08-27 | `2608.27033v1` | Riemann-1.0: An Embodied World Action Model for Physical AI |
| 2026-08-23 | `2608.22301v1` | The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction |
| 2026-07-16 | `2607.15275v1` | RoboTTT: Context Scaling for Robot Policies |
| 2026-07-13 | `2607.11397v2` | WALA Learning Executable Latent Actions from Action-Labeled Demonstrations and Action-Free Videos |
| 2026-07-04 | `2607.03723v1` | OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies |
| 2026-06-28 | `2606.29517v2` | CORE: Common Outcome Regularities from Action-Free Visual Demonstrations for Robot Manipulation |
| 2026-06-23 | `2606.24448v1` | Supervise What Survives: Geometry-Guided VLA Adaptation from Synthetic Robot Videos |

## 08 · 强化学习后训练与自我改进

**`abs:"reinforcement learning" AND abs:"vision-language-action"`** — 命中 101，其中新增 20（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.07594v1` | BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation |
| 2026-10-05 | `2610.05994v1` | How (and How Not) to Use Data Augmentation in VLA Post-Training |
| 2026-10-03 | `2610.06926v1` | SWAP: Stepwise Action Policy Routing for Vision-Language-Action Models |
| 2026-09-30 | `2609.40134v2` | Tactile Curiosity Drives Robot Interaction |
| 2026-09-30 | `2609.39601v1` | GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives |
| 2026-09-30 | `2609.38890v1` | PRICE the Action Chunks: Physical Relational Credit Assignment for Embodied Reinforcement Learning |
| 2026-09-29 | `2609.38641v1` | Vision-Language-Action Autonomous Driving Agent with Language-based Memory |
| 2026-09-29 | `2609.36934v1` | VLALight: A Vision-Language-Action Model for Traffic Signal Control |
| 2026-09-29 | `2609.36588v1` | Cooperative Multi-Agent Vision-Language-Action Models via Reinforced Fine Tuning |
| 2026-09-29 | `2610.00317v1` | DriftOPD: Sequence-Level Reverse-KL Distillation for One-Step VLA Policies |
| 2026-09-28 | `2609.36352v1` | StructRL: Online Structured Reinforcement Learning for Long-Horizon Vision-Language-Action Tasks |
| 2026-09-28 | `2609.35078v1` | RefineDrive: Reliable Failure-Guided Learning for Vision-Language-Action Driving |
| 2026-09-28 | `2609.34688v1` | Unified Trajectory Matching Policy Optimization: Diverse T2I Generation and VLA Generalization |
| 2026-09-28 | `2609.34387v2` | CAR-VLA: Complexity-Aware and Risk-Adaptive Reasoning for Autonomous Driving |
| 2026-09-27 | `2609.33765v1` | Principal Steering Subspaces for Online Adaptation of Frozen Generative Robot Policies |
| 2026-09-27 | `2609.33125v1` | Train Together or Merge Later? Unifying VLA Experts via a Shared Action Interface |
| 2026-09-26 | `2609.32634v1` | PF-RL: Progress Field Reinforcement Learning via Goal-Conditioned Value Geometry for Vision-Language-Action Mo |
| 2026-09-23 | `2609.28838v1` | Uncertainty-Gated Exploration Noise Suppresses Task Collapse in Online RL Fine-Tuning of a Flow-Matching Visio |
| 2026-09-22 | `2609.26467v1` | RouteRLT: Learning When and Which RL Specialist Should Control a Vision-Language-Action Policy |
| 2026-09-21 | `2609.23968v1` | Opt2VLA: Force-Aware Vision-Language-Action for Contact-Rich Humanoid Whole-Body Manipulation |

**`abs:"GRPO" AND abs:"robot"`** — 命中 14，其中新增 9

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-30 | `2609.39601v1` | GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives |
| 2026-09-22 | `2609.25631v1` | DynaForge: Planning-Guided Residual Learning for Dynamic Manipulation Demonstration Generation |
| 2026-08-17 | `2608.17027v2` | FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences |
| 2026-08-15 | `2608.15284v1` | VTInstructor: Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments |
| 2026-08-08 | `2608.08053v1` | PhysX-CoT: Structured Physical Reasoning from a Single Image to Simulation-Ready 3D Assets |
| 2026-07-14 | `2607.12787v2` | Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters? |
| 2026-07-08 | `2607.07001v1` | Ego-Human Motion Prediction with 3D-Aware LLM |
| 2026-07-06 | `2607.05391v2` | LLM-as-a-Verifier: A General-Purpose Verification Framework |
| 2026-06-22 | `2606.23531v4` | BiliVLA: Scene-Aware Vision-Language-Action Model with Reinforcement Learning for Autonomous Biliary Endoscopi |

**`abs:"reward model" AND abs:"robot policy"`** — 命中 10，其中新增 5

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-27 | `2609.33832v1` | Achieve What You Imagined: Learning to Align Actions with Visual Plans |
| 2026-08-19 | `2608.18787v1` | Dream2Reward: Transition-Alignment Reward Models from Positive Demonstrations for Robotic Manipulation |
| 2026-07-14 | `2607.13033v1` | DenseReward: Dense Reward Learning via Failure Synthesis for Robotic Manipulation |
| 2026-07-14 | `2607.12466v1` | Deployable Human Preference Alignment in Robotics: Learning Representative Rewards from Diverse Human Preferen |
| 2026-06-30 | `2606.32027v3` | Freeform Preference Learning for Robotic Manipulation |

**`abs:"self-improvement" AND abs:"robot"`** — 命中 25，其中新增 20

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.09228v1` | Co-Evolving Robot Orchestrators and Policies through Deployment |
| 2026-10-06 | `2610.08995v1` | PhysEvo: Astra Can Act, Let It |
| 2026-10-06 | `2610.08761v1` | VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning |
| 2026-09-30 | `2609.38905v3` | EmbodiRSI: Recursive Self-Improvement for Data-Efficient Robot Adaptation |
| 2026-09-28 | `2609.36012v1` | In-Context Learning for Robots: Methods and Applications |
| 2026-09-28 | `2609.34823v2` | AGRO-SUVIDE: Agentic Robotics for Surgical Viscoelastic Debridement |
| 2026-09-26 | `2609.32862v1` | RoboFoundry: System-as-Policy Evolution for Self-Learning Embodied Agents |
| 2026-09-23 | `2609.27612v1` | RegenHarness: A Robot Agent Harness with Evidence-Gated Recursive Self-Improvement |
| 2026-09-22 | `2609.26499v1` | Generalizing Manipulation Skills with a Local Coding Agent |
| 2026-09-15 | `2609.17372v1` | XPACE: Joint World and Action Modeling from Heterogeneous Experience |
| 2026-09-01 | `2609.01679v1` | A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inferenc |
| 2026-08-07 | `2608.06877v1` | Autonomous Optimization of Complex Oxides for Thermochemical Fuel Production |
| 2026-08-03 | `2608.01851v1` | Weights or Skills? A Survey of Robot-Learning Techniques: from Action-Predicting Weights to Robots that Write  |
| 2026-07-31 | `2607.29640v1` | Bootstrapping Self-Supervised Learning of Binary Classification Using Error Bounds: A Case Study on a Robotic  |
| 2026-07-29 | `2607.26809v1` | Practice Makes Policies: Bootstrapping and Consolidating Robotic Capabilities from Zero Human Demonstrations |
| 2026-07-17 | `2607.15524v1` | Recursive Harness Self-Improvement |
| 2026-07-05 | `2607.04426v1` | ACE-Brain-0.5: A Unified Embodied Foundational Model for Physical Agentic AI |
| 2026-06-19 | `2606.21406v1` | Robot Self-Improvement via Human-Video Dynamics Models |
| 2026-06-17 | `2606.18953v1` | Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement |
| 2026-06-16 | `2606.18247v1` | Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement |

**`abs:"world model" AND abs:"policy improvement"`** — 命中 17，其中新增 11

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09763v1` | Beyond Policy Support: Interaction Constrained Offline Reinforcement Learning for Autonomous Driving |
| 2026-09-29 | `2609.36851v2` | RoXDrive: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving via Action-Faithful Rollouts |
| 2026-09-15 | `2609.17372v1` | XPACE: Joint World and Action Modeling from Heterogeneous Experience |
| 2026-09-10 | `2609.12036v1` | Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence |
| 2026-09-08 | `2609.09155v1` | SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators |
| 2026-08-31 | `2609.00455v2` | Towards a Belief-Based World Model for LLM Agents |
| 2026-08-31 | `2608.30237v2` | Motus2: A Self-Evolving General World Model for Dexterous Manipulation |
| 2026-08-18 | `2608.17959v2` | Towards Zero-Shot Task Transfer with Neurosymbolic World Models |
| 2026-08-06 | `2608.05674v1` | JoyAI-RA 0.5: Scaling Robot Manipulation Learning via Dual Action Alignment |
| 2026-06-17 | `2606.18960v3` | Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation |
| 2026-06-11 | `2606.13672v2` | WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation |

**`abs:"online fine-tuning" AND abs:"robot policy"`** — 命中 1，无新增

## 09 · 评测基准与报告口径

**`abs:"LIBERO"`** — 命中 429，其中新增 21（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10528v1` | Long-WAM: Scaling the Context of World-Action Models |
| 2026-10-07 | `2610.10498v1` | EmbodiedRSI: Active Continual Robot Learning Through Hypothesis-Guided Co-Evolution |
| 2026-10-07 | `2610.09734v1` | ΔWAM: Distilling Action Tangent Fields into World Action Models |
| 2026-10-07 | `2610.09451v1` | TempoBridge: Language-Guided Tempo Control for Vision-Language-Action Policies |
| 2026-10-06 | `2610.09170v1` | Beyond Reconstruction: What Matters in Action Tokenization for Robot Policies? |
| 2026-10-06 | `2610.09144v1` | DIVA: Dual-Space Intent-Aware Visual Attenuation for Vision-Language-Action Policies |
| 2026-10-06 | `2610.09016v1` | PAIR: Bridging Perception and Action in Vision-Language-Action Models |
| 2026-10-06 | `2610.08555v1` | Towards Efficient Robotic Manipulation Models with Self-Recursive Pruning |
| 2026-10-06 | `2610.08444v1` | ActTune: Action-Aware Precision and GPU Operating-Point Adaptation for Energy-Efficient Vision-Language-Action |
| 2026-10-06 | `2610.08183v1` | Compact Robot Policies Need Fine-Grained Visual Representations |
| 2026-10-06 | `2610.08150v1` | ViDAL: A Visual Dynamics-Grounded Action Latent Space for Vision-Language-Action Models |
| 2026-10-06 | `2610.08133v1` | VLA-ACL: Action-Consistent Visual Token Pruning for Efficient Vision-Language-Action Models |
| 2026-10-06 | `2610.07961v1` | IronMan: Information-Constrained Video-Action Learning for Robot Manipulation |
| 2026-10-06 | `2610.07946v1` | Adapting Vision-Language-Action Models to Unknown Visual Disruptions During Execution |
| 2026-10-06 | `2610.07922v1` | OpenWAM: An Open Framework for Composable World-Action Models |
| 2026-10-06 | `2610.07917v1` | PACE: Stage-Consistent Long-Horizon Robot Manipulation via Progress-Aligned Context for Execution |
| 2026-10-06 | `2610.07756v1` | StairVLA: Stage-Aware Hierarchical Action Generation for Vision-Language-Action Models |
| 2026-10-05 | `2610.06843v1` | Recursive Video In-Context Learning for Agentic Robot |
| 2026-10-05 | `2610.06814v2` | TAPDreamer: Transferable Adversarial Patches for World Action Models |
| 2026-10-05 | `2610.06617v1` | RealtimeWAM: One-Step Asynchronous World Action Models |
| 2026-10-05 | `2610.06318v1` | Wiring Matters: Injection Topology and Initialization of Affordance Heads in Vision-Language-Action Policies |

**`abs:"SimplerEnv"`** — 命中 24，其中新增 21

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09940v1` | Juno: Taming Predictive Latents for Vision-Language-Action Models |
| 2026-09-28 | `2609.34206v1` | WorldGuide: Learning Success-Failure Boundaries in Latent World Models for Vision-Language-Action Policies |
| 2026-09-27 | `2609.33378v1` | Recursive Harness Distillation across Agents for Robot Manipulation |
| 2026-09-24 | `2609.28865v1` | Direction-Scale Decomposition in Action Representation: Rethinking What to Tokenize for Vision-Language-Action |
| 2026-09-20 | `2609.23275v1` | SCULPT-VLA: Learning Structured Control through Staged Action Grounding |
| 2026-09-19 | `2609.22895v1` | H-VLA: Hierarchical Vision-Language-Action Model with Key-Action Reasoning and Motion Planning in a Unified Ac |
| 2026-09-16 | `2609.18487v1` | ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models |
| 2026-08-24 | `2608.23478v1` | Act with Intent: Distilling Behavior Intent for Vision-Language-Action Models |
| 2026-08-12 | `2608.11739v1` | G0.5: One Autoregressive Stream for Robot Reasoning and Action |
| 2026-08-11 | `2608.10484v1` | Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models |
| 2026-08-10 | `2608.09448v2` | VANE: Reliable Test-Time Training for Vision-Language-Action Models via Future Visual Representation Predictio |
| 2026-08-06 | `2608.05738v1` | In-Context VLA: Endowing Vision-Language-Action Models with Language via In-Context Post-Training and Agentic  |
| 2026-08-05 | `2608.04510v1` | GUARD: Grounding Uncertainty and Ablation-Based Risk Detection for Diffusion-Based VLAs |
| 2026-08-03 | `2609.26071v1` | StrataVLA: Hierarchical and Efficient 3D Geometric Grounding for Vision-Language-Action Models |
| 2026-08-03 | `2608.02197v1` | Look Where It Matters: Adaptive Visual Refinement for Vision-Language-Action Models |
| 2026-07-29 | `2607.27261v1` | It's Not Just More Demos: Counterfactual Action Sensitivity Coverage for Data-Efficient Robust Robot Imitation |
| 2026-07-08 | `2607.07608v1` | Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation |
| 2026-07-02 | `2607.01586v2` | VLAFlow: A Unified Training Framework for Vision-Language-Action Models via Co-training and Future Latent Alig |
| 2026-06-26 | `2606.27872v1` | S$^2$-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation |
| 2026-06-23 | `2606.25215v1` | Reflective VLA: In-Context Action Consequences Make VLAs Generalize |
| 2026-06-18 | `2606.20246v2` | Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think |

**`abs:"VLA benchmark"`** — 命中 7，其中新增 6

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-07-30 | `2607.28251v2` | Policy-Level Recursive Self-Improvement for Embodied AI with a Criticality World Model |
| 2026-07-09 | `2607.08182v1` | LEEVLA: Seeing What Matters in Latent Environment Evolution for Vision-Language-Action |
| 2026-07-05 | `2607.13056v1` | HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration |
| 2026-06-28 | `2606.29247v1` | SurgVLA-Bench: Towards Evaluating Vision-Language-Action Models for Laparoscopic Surgical Robotics |
| 2026-06-13 | `2606.15165v1` | VLALeaks: Membership Inference Attacks against Vision-Language-Action Models |
| 2026-06-11 | `2606.12859v1` | AIR-VLA+: Decoupling Movement and Manipulation via Cascaded Dual-Action Decoders with Asymmetric MoE for Aeria |

**`abs:"real robot" AND abs:"evaluation protocol"`** — 命中 5，其中新增 4

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-16 | `2609.18374v2` | Decoupling Vision, Language, and Action for Efficient Multi-Task Robot Policies |
| 2026-09-15 | `2609.17210v1` | FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence |
| 2026-07-01 | `2607.00978v2` | Depth-Only Open-Vocabulary 3D Semantic Segmentation For Privacy-Preserving Robotic Applications |
| 2026-06-24 | `2606.26443v1` | WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation |

**`abs:"success rate" AND abs:"robot policy" AND abs:"evaluation"`** — 命中 24，其中新增 19

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.08555v1` | Towards Efficient Robotic Manipulation Models with Self-Recursive Pruning |
| 2026-10-04 | `2610.05492v1` | When Does Retrieval Help? A Study of In-Context Adaptation in Vision-Language-Action Models |
| 2026-09-29 | `2609.36872v1` | PreferenceFlow: Test-Time Guidance of Flow-Matching Robot Policies from Human Interventions |
| 2026-09-27 | `2609.33832v1` | Achieve What You Imagined: Learning to Align Actions with Visual Plans |
| 2026-09-26 | `2609.32239v1` | Federated Subspace Guided Vision-Language-Action Policy Distillation for Non-IID Multi-Robot Manipulation |
| 2026-09-25 | `2609.31606v1` | Learning Robot Policies from Sparse Success Signals via STL-Guided Stein Variational Policy Gradient |
| 2026-09-23 | `2609.28807v1` | An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics |
| 2026-09-23 | `2609.27513v1` | Behavior-Aligned Action Tokenization for Robot Policy Learning |
| 2026-09-17 | `2609.20648v2` | SkipVLA: Skipping VLA Steps with Classical Planning for Fast Robot Manipulation |
| 2026-09-11 | `2609.13053v1` | Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model |
| 2026-08-15 | `2608.15002v1` | NPU Offloading of a Frozen Visual Encoder for Robot Policy Training |
| 2026-08-13 | `2608.13453v1` | UniTexture: Cross-Task Universal Adversarial Textures for Vision-Language-Action Models |
| 2026-08-10 | `2608.09138v2` | SpeedTuning: Speeding Up Policy Execution with Lightweight Reinforcement Learning |
| 2026-07-06 | `2607.04880v1` | PRISM: Personalized Robotic Dataset Generation via Image-based Scene and Motion Synthesis |
| 2026-07-05 | `2607.13056v1` | HRIBench: Benchmarking Interaction-Centric Human-Robot Collaboration |
| 2026-06-29 | `2606.30101v1` | SIR: Structured Image Representations for Explainable Robot Learning |
| 2026-06-28 | `2606.29517v2` | CORE: Common Outcome Regularities from Action-Free Visual Demonstrations for Robot Manipulation |
| 2026-06-26 | `2606.28276v4` | SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation |
| 2026-06-22 | `2607.00033v2` | Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration |

**`abs:"reproducibility" AND abs:"robot learning"`** — 命中 11，其中新增 11

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-08-04 | `2608.03127v1` | DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units |
| 2026-07-18 | `2607.16998v1` | User-Driven Learning from Demonstration: A Trajectory and Impedance Learning Method |
| 2026-07-17 | `2607.16187v1` | Handroid: Bridging Dexterous Hand and Humanoid |
| 2026-07-15 | `2609.18950v1` | Changepoint-Aware World Models: Detecting Dynamics Shifts and Recovering by Forgetting Stale Replay in Model-B |
| 2026-07-07 | `2607.06699v1` | RoboSnap: One-Shot Real-to-Sim Scene Generation for Generalizable Robot Learning and Evaluation |
| 2026-07-06 | `2607.13059v1` | GPUSimBench: Towards Scalable and Reliable GPU-Accelerated Simulators in Embodied AI |
| 2026-06-18 | `2606.20549v1` | Generating Robot Hands from Human Demonstrations |
| 2026-06-16 | `2606.17511v2` | MagicSim: A Unified Infrastructure for Executable Embodied Interaction |
| 2026-06-16 | `2606.17418v1` | DexLink Hand: A Compact, Affordable, 16-DOF Linkage-Driven Hand with Human-Like Dexterity |
| 2026-06-12 | `2606.14561v1` | ORCA: A Platform for Open-Source Dexterity Research |
| 2026-06-11 | `2606.12936v3` | Pipette: An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Rob |

## 10 · 世界模型增强 VLA

**`abs:"world model" AND abs:"vision-language-action"`** — 命中 75，其中新增 19（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09940v1` | Juno: Taming Predictive Latents for Vision-Language-Action Models |
| 2026-10-03 | `2610.04607v1` | ForeAct3D: Policy-Grounded Future World Modeling for VLA Policies |
| 2026-10-02 | `2610.04009v1` | SUAVE: Unified Video-Action Models via Masked Diffusion |
| 2026-09-30 | `2609.39324v1` | MotionWeave: Learning Motion-Centered Future Dynamics for Vision-Language-Action Policies |
| 2026-09-28 | `2609.35003v1` | Learning to Act under Visual Interruptions with Vision-Language-Action Models |
| 2026-09-28 | `2609.34206v1` | WorldGuide: Learning Success-Failure Boundaries in Latent World Models for Vision-Language-Action Policies |
| 2026-09-25 | `2609.31313v1` | Towards VLA-Dreamer: Refining VLA Behavior Using World Models |
| 2026-09-21 | `2609.24682v2` | Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies |
| 2026-09-18 | `2609.21712v2` | ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation |
| 2026-09-18 | `2609.21228v1` | FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models |
| 2026-09-17 | `2609.19600v1` | WorldContact: A Contact-Centric World Model for Scalable Robot Learning |
| 2026-09-15 | `2609.17728v1` | RAF-VLA: Representation Alignment with the Future for End-to-End Autonomous Driving |
| 2026-09-15 | `2609.17210v1` | FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence |
| 2026-09-15 | `2609.16705v1` | The Robot Data Factory |
| 2026-09-08 | `2609.08162v1` | WorldAgen: Unified State-Action Prediction with Test-Time World Model Training |
| 2026-09-07 | `2609.07470v1` | Measuring Language Transfer in Robot Policies: Adding Greek to a Cosmos3 Vision-Language-Action Policy |
| 2026-09-03 | `2609.04193v1` | GIFT: Guided Intermediate Feature Training via Action-Oriented Structural Supervision for Robotic Manipulation |
| 2026-09-02 | `2609.13236v1` | Self-Evolving AI for Humanoids: Mechanisms, Safety, and Evaluation of Post-Deployment Self-Improvement |
| 2026-09-01 | `2609.01215v1` | REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs |

**`abs:"video prediction" AND abs:"robot policy"`** — 命中 4，其中新增 2

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.08150v1` | ViDAL: A Visual Dynamics-Grounded Action Latent Space for Vision-Language-Action Models |
| 2026-07-28 | `2607.25918v1` | DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models |

**`abs:"imagined rollout"`** — 命中 20，其中新增 18

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09427v1` | Event-Aligned Visual Action Reasoning for World Action Models |
| 2026-10-07 | `2610.09335v1` | SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models |
| 2026-10-03 | `2610.04494v1` | DreamTest: World-Model Surrogates for Search-Based Testing of Deep Reinforcement Learning Agents |
| 2026-09-29 | `2609.36845v1` | DSWM: Decomposed Spatio-Temporal World Model for Demand-Driven UAV Base Station Repositioning |
| 2026-09-27 | `2609.33336v1` | Beyond Conservatism: Recoverability-Conditioned Exploration for Model-Based Imitation Learning |
| 2026-09-24 | `2609.30214v1` | Underwater C3-JEPA: An Object-Centric Cross-View World Model for ROV Salvage |
| 2026-09-17 | `2609.20892v1` | WM-VS: Progress-Aligned World Models for Closed-Loop Visual Servoing |
| 2026-09-09 | `2609.09941v1` | HaWMPO: Hallucination-Aware World Model-based Policy Optimization for Generalist Robot Policy |
| 2026-09-03 | `2609.03225v2` | Long-Horizon Consistent and Interaction-Aware World Models for Multi-Style End-to-End Driving |
| 2026-08-19 | `2608.18669v2` | Reinforced Planning with Latent World Models |
| 2026-08-18 | `2608.17739v1` | Offline Multi-Agent Reinforcement Learning with a Physics-Informed World Model for Cooperative Mixed Traffic C |
| 2026-08-17 | `2608.17163v1` | Q-Learning With World Models |
| 2026-08-07 | `2608.07746v1` | LUCID: Latent-Skill Unified Control via Imagined Dynamics for Long-Horizon Humanoid Loco-Manipulation |
| 2026-07-22 | `2607.19749v1` | The World Model Remembers, the Actor Forgets: Dream Rehearsal for Continual Model-Based RL |
| 2026-07-21 | `2607.19343v1` | Masked Visual Actions for Unified World Modeling |
| 2026-07-02 | `2607.02087v2` | SUNTA: Hierarchical Video Prediction with Surprise-based Chunking |
| 2026-06-29 | `2606.30192v1` | Domain Adaptation with Adaptive Imagination for Visual Reinforcement Learning under Limited Target Data |
| 2026-06-21 | `2606.22509v1` | Imagine to Ensure Safety in Hierarchical Reinforcement Learning |

**`abs:"latent dynamics" AND abs:"policy learning"`** — 命中 4，其中新增 4

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-07 | `2609.07002v1` | WM-Craftnet: World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipulation |
| 2026-08-07 | `2608.06799v1` | Is Forward Prediction Enough? Physical State Grounding for JEPA World Models |
| 2026-07-29 | `2607.27138v1` | DLAM: Distributional Latent Actions with Temporal Constraints |
| 2026-07-01 | `2607.00808v1` | Local Motion Matters: A Deconstruct-Recompose Paradigm for Reinforcement Learning Pre-training from Videos |

**`abs:"model-based" AND abs:"vision-language-action"`** — 命中 18，其中新增 12

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.07696v1` | ESP: Energy-Score Policy for One-Step Multimodal Action Generation |
| 2026-09-30 | `2609.38855v1` | Online Evolution Strategy for Flow-Matching VLA Policies via Self-Supervised Trajectory Distribution Optimizat |
| 2026-09-16 | `2609.18111v1` | A Comprehensive Review of Generative Physical Artificial Intelligence |
| 2026-09-05 | `2609.06221v1` | RefGuard: Identity-Aware Language-Guided Robot Manipulation via Joint Target-Anchor-Frame Grounding |
| 2026-08-25 | `2608.24101v3` | TrAct: Bridging Robot Control and Visual Prediction with Visual Tracks |
| 2026-08-20 | `2608.20111v1` | Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms |
| 2026-08-17 | `2608.17163v1` | Q-Learning With World Models |
| 2026-08-12 | `2608.12416v1` | RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills |
| 2026-07-25 | `2607.22999v2` | WCM: World-Cognition Model for Generalizable Human-Robot Interaction |
| 2026-07-22 | `2607.19633v1` | LENS: LLM-guided Environment Simplification for Planning and Control in Clutter |
| 2026-06-17 | `2606.18610v3` | SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation |
| 2026-06-15 | `2606.20698v1` | SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model |

## uav · 无人机主线

**`abs:"aerial" AND abs:"vision-language-action"`** — 命中 10，其中新增 2

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-10 | `2609.11310v1` | Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Lan |
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |

**`abs:"UAV" AND abs:"vision-language-action"`** — 命中 7，其中新增 1

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |

**`abs:"drone" AND abs:"vision-language navigation"`** — 命中 1，其中新增 1

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-08-10 | `2608.09270v1` | GRASP: Granularity-Aware Region Alignment and Semantic Prototype Learning for Fine-Grained Cross-Modal Underst |

**`abs:"aerial" AND abs:"language model" AND abs:"control"`** — 命中 19，其中新增 16

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-02 | `2610.03319v1` | Defense-in-Depth at the Perception-Reasoning Interface of LLM-Centric Agentic UAV Swarms |
| 2026-10-01 | `2610.01569v1` | Managing Context and Communication in Distributed Agentic UAV Swarms |
| 2026-09-21 | `2609.23992v1` | From Ideal Motion to Flight-Executable Communications: LLM-Evolved Multi-UAV Deployment for Cell-Free Massive  |
| 2026-09-20 | `2609.23655v1` | Reassessing Global Gradient-Norm Imbalance in BLIP Fine-Tuning Across Physical Domains |
| 2026-09-17 | `2609.19538v1` | Agentic AI Networking for Heterogeneous Unmanned Aerial Systems in Low-Altitude Wireless Networks |
| 2026-09-05 | `2609.06174v1` | Exploiting LLM Agents for Trustworthy AutoResearch in Wireless Communications |
| 2026-08-11 | `2608.10434v1` | Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Tru |
| 2026-08-08 | `2608.08225v1` | Toward Intelligent Skies: Signal Processing and AI Foundations of Low-Altitude Wireless Networks |
| 2026-08-06 | `2608.05792v1` | When Agentic AI Meets Integrated Sensing and Communication |
| 2026-07-21 | `2607.18604v1` | Intelligent Multi-UAV Navigation in ITNTNs: A Hierarchical LLM Approach |
| 2026-07-09 | `2607.08359v1` | FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation |
| 2026-07-07 | `2607.07350v1` | Towards Reliable Aerial Ground Vehicle Collaboration: An Integrated Planning and Autonomy Framework for Field  |
| 2026-07-04 | `2607.03869v2` | GeoSelect: Spatial-Program Execution for Training-Free Referring Remote Sensing Image Segmentation |
| 2026-07-02 | `2607.02718v1` | Diagnosing Aerial-View Object Detectors with Foundational Image Generative Models |
| 2026-06-30 | `2606.31073v1` | MultiUAV-Plat: An LLM-Oriented Platform, Benchmark and Framework for Multi-UAV Collaborative Task Planning |
| 2026-06-29 | `2606.30151v1` | AERIS: Aerial-Edge Role-Driven Intelligence at Runtime via Orchestrated Language-Model Swarm |

**`abs:"quadrotor" AND abs:"foundation model"`** — 命中 0。**复核过：零落在术语上，不落在领域上。**`aerial` / `UAV` / `drone foundation model`、`aerial foundation models` 四种提法命中全为 **0**（两套词表交集为空，够得上「事实级空白」的判据）；但同窗口 `abs:"aerial" AND abs:"vision-language-action"` 命中 **11**。结论只能写到这一步：**空中领域不用「foundation model」自我描述，而用 VLA/VLN**。**不得写成「空中没有基础模型工作」。**

**`abs:"aerial" AND abs:"embodied" AND abs:"policy"`** — 命中 5，其中新增 2

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.08306v1` | Sensor-Layout-Agnostic Navigation via Geometric Observation Canonicalization |
| 2026-06-30 | `2606.31772v1` | Autonomous UAV Navigation for Individual Wildlife Re-Identification |

---

*本文件由 `tools/watch.py` 生成，重跑即可刷新。收录进正文前，ID 与标题一律以 arXiv API 为准。*
