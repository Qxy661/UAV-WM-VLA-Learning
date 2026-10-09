# VLM 前沿增量 · 2026-10

> 生成：`py -3.9 tools/watch.py --volume vlm --write`｜窗口 2026-06-11 ~ 2026-10-09（120 天）｜词表 24 条｜请求失败 0 条｜新增 449 条

词表按 docs/04-VLM专题/05–07 三篇 分节。

**读法**（纪律 F 的镜像规则）：TOTAL 是**题摘层面的命中数，不是相关论文数**。一条宽查询命中几百条不含任何信息，所以下表只列**仓库尚未收录**的条目。`NULL` 表示请求失败（429/超时），**不是 0 命中**；`0` 表示确实没有，但第一反应应当是怀疑查询措辞，换个提法复核后再下结论。

arXiv 的 `all:` / `abs:` 只覆盖题名/摘要/作者/注释，**不覆盖全文**。

**去重范围是整仓**：`docs/`、`references/`、`mindmaps/`、`paper/` 下的 `.md`，加 `README.md` / `CONTRIBUTING.md` / `code/README.md`。`paper/` 下的 `_work*/` 取数缓存（json/xml/html/pdf）**不算收录**，故不计入——那里放的是原始查询结果，不是读过的文献。所以某条 ID 不在本表，只能推出**整仓没写过**，推不出"这个专题没有"。
（2026-10-08 修：`paper/` 原先漏扫，加上 ID 正则不吃 `v1` 版本号后缀，两条合起来让 27 条早已在 `paper/` 笔记里读过的论文被标成"新增"。三卷的台账互为已收录，故 `<卷>-watch-*.md` 一律排除。）

## 05 · 通用架构与视觉编码器

**`abs:"vision encoder" AND abs:"multimodal large language"`** — 命中 17，其中新增 17

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09440v1` | Mixture of Layers: Dynamic Layer Routing for Visual Reasoning |
| 2026-10-04 | `2610.05413v1` | A Strong Baseline for Evaluating Vision Encoders in Multimodal Large Language Models |
| 2026-09-30 | `2609.40361v1` | Ranking-Aware Prompt Optimization for Multimodal Clinical Diagnosis |
| 2026-09-28 | `2609.34977v1` | SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models |
| 2026-09-07 | `2609.07823v1` | SAFIRE: Safety-Critical Benchmark for Fine-grained Fire and Smoke Understanding in Multimodal LLMs |
| 2026-08-30 | `2608.29802v1` | Foundation and Multimodal Large Language Models for Face Presentation and Morph Attack Detection |
| 2026-08-28 | `2608.28312v1` | AIM: Anchor Identity Features, Then Match for Multimodal Large Language Model Unlearning |
| 2026-07-27 | `2607.24743v2` | ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding |
| 2026-07-20 | `2607.17806v1` | PGN: Design and Implementation of a Vision-Language Navigation System Based on Pangu Multimodal Foundation Mod |
| 2026-07-13 | `2607.12112v2` | Continual Learning with Elastic Regularization and Synthetic Replay for Federated MLLM Fine-Tuning |
| 2026-07-13 | `2607.11562v1` | MonkeyOCRv2: A Visual-Text Foundation Model for Document AI |
| 2026-07-08 | `2607.07673v1` | MedPMC: A Systematic Framework for Scaling High-Fidelity Medical Multimodal Data for Foundation Models |
| 2026-07-07 | `2607.05880v1` | Harrison.Rad 1.5 Technical Report: A radiology foundation model that can draft reports from images, priors and |
| 2026-07-01 | `2607.00416v1` | DroneIQA-VLE: Multi-Task Drone Image Quality Assessment via Vision-Language Ensemble |
| 2026-06-23 | `2606.25084v1` | Are We There Yet? Exploring the Capabilities of MLLMs in Assistive AI Applications |
| 2026-06-22 | `2606.23885v1` | Mind the Heads: Topological Representation Alignment for Multimodal LLMs |
| 2026-06-13 | `2606.15134v1` | Beyond Scalar Distances: Semantic Attribute Gradients from Frozen MLLMs for Visual Embeddings |

**`abs:"connector" AND abs:"multimodal large language"`** — 命中 1，其中新增 1

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-18 | `2609.21675v1` | DRT: Dense Reasoning Trace for Efficient and Grounded Multimodal Reasoning |

**`abs:"SigLIP"`** — 命中 43，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-01 | `2610.02148v1` | Omni-Embed-Mini: Binding Modalities Without Forgetting via Dense Distillation |
| 2026-09-30 | `2609.39625v1` | D-Scope: Decomposing and Steering Diffusion Transformers with Sparse Autoencoders |
| 2026-09-29 | `2609.37685v2` | PAIQ: Patch-Aligned Semantic Injection via Residual Rotation |
| 2026-09-28 | `2609.36101v1` | One Geometry, Different Outcomes: Readout-Dependent Effects of the Modality Gap in Vision-Language Models |
| 2026-09-28 | `2609.35470v1` | Representation Risk in Pretrained Image Encoders |
| 2026-09-27 | `2609.34021v1` | Position Aware Layer Queries for Test Time Training in Vision Language Models |
| 2026-09-27 | `2609.33359v1` | When Does Geometric View Synthesis Help Wine Label Retrieval? A Public One-Shot Benchmark Across Self-Supervis |
| 2026-09-23 | `2609.28857v1` | MEVL-STP: Multi-Encoder and Vision Language Model for Arbitrarily Shaped Scene Text Spotting |
| 2026-09-22 | `2609.27142v1` | MINER: Multi-crop INference-time Enhancement for Rare-Object Retrieval with Frozen Dual Encoders |
| 2026-09-09 | `2609.10224v1` | UOT-Gap: A Variational Principle for the Modality Gap in Vision-Language Models via Unbalanced Optimal Transpo |
| 2026-09-07 | `2609.07937v1` | TDDN: Text-aligned Diffused DINO Network for Puzzle Understanding |
| 2026-09-02 | `2609.03085v2` | Solving the Needle-in-a-Haystack Problem in Mammography Vision-Language Model with Differentiable Subset Sampl |
| 2026-08-30 | `2608.29974v1` | SpanCalib-VLM: Calibrated Hallucination Span Detection in Vision-Language Models |
| 2026-08-29 | `2608.29395v1` | GATE: Reliability-Gated Gaussian Evidence Fusion for Training-Free Test-Time Adaptation of Vision-Language Mod |
| 2026-08-28 | `2608.28316v2` | Conditional Visual Evidence Utility: State-Dependent Rank Reversals in Frozen Vision-Language Encoders |
| 2026-08-25 | `2608.24133v1` | PlaceSeek: Human-Centered Geospatial Retrieval of Urban Outdoor Places via Semantic Grounding and Affective Al |
| 2026-08-24 | `2608.23484v1` | Multi-Modal Semantic Expansion with Constrained LLM Reranking for Conversational Music Recommendation |
| 2026-08-23 | `2608.23634v1` | The Blending Ratio Is Not Where the Performance Is: Diagnosing Prototype Blending for Few-Shot Adaptation of V |
| 2026-08-21 | `2608.20988v1` | Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models |
| 2026-08-19 | `2608.19376v1` | Does Marginal Coverage Guarantee Class-Conditional Safety for Zero-Shot VLMs Under Shift? |
| 2026-08-19 | `2608.21443v1` | Text-Guided Visual Dependency Graph Learning with Cross-Modal Attention Priors |
| 2026-08-19 | `2608.18591v1` | Can a Lightweight Multimodal Model Estimate LLM Reasoning Performance? A Study for Compute-Optimal Document In |
| 2026-08-17 | `2608.16690v2` | AnchorScore: A CLIP-Based Diagnostic of MLLM Annotation Difficulty |
| 2026-08-17 | `2608.16975v1` | Margin-Regularized Structured Semantic Alignment for Brain-Language Correspondence |
| 2026-08-17 | `2608.16263v1` | Seeing Before Answering: Training-Free Visual Layer Profiling for Vision-Language Models |
| 2026-08-17 | `2608.16234v1` | GaussianDWM++: Language-Grounded 3D Gaussian Driving World Model for Unified Scene Understanding, Editing, and |
| 2026-08-17 | `2608.16962v1` | Clinical Pathways Matter for Multimodal Deep Learning in Early Alzheimers Disease Detection |
| 2026-08-11 | `2609.26168v1` | TRACE: Transparent Retrieval for Abstract Concept Evaluation |
| 2026-08-09 | `2608.08477v5` | VectraYX-Vision-1B: A Sub-2B Spanish/LATAM Cybersecurity Vision-Language Model, and What Limits Its Visual Gro |
| 2026-07-28 | `2607.25842v1` | Adversarial Deepfake Generation and an Investigation of Purification-Based Adversarial Detection |

**`abs:"visual token" AND abs:"multimodal"`** — 命中 144，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.07984v1` | Decide Before You Look: Learning Which Retrieved Memories Deserve Pixels |
| 2026-10-06 | `2610.07572v1` | Two Vectors Replace In-Context Demos: Structured Task Adaptation via Embeddings |
| 2026-10-04 | `2610.05097v1` | ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory |
| 2026-10-03 | `2610.04225v1` | FlashGaze: Training-Free Multi-Scale Patch Pruning For Efficient Video Understanding |
| 2026-10-02 | `2610.03400v1` | Beyond Entropy: Self-Diagnostic Multi-Role Token Optimization for Video Reasoning |
| 2026-10-02 | `2610.03822v1` | Beyond Token Accuracy: Prioritizing What Matters for Visual Reconstruction |
| 2026-09-30 | `2610.00623v1` | HAWK: Rethinking Multimodal Drafting for Speculative Decoding |
| 2026-09-30 | `2609.39924v1` | CoVisco: Codec-Native Vision Encoder with Native Token Compression for Unified Image-Video Understanding |
| 2026-09-30 | `2609.38823v1` | DecoMoE: Decoupling Visual Propagation and Expert Computation for Efficient Multimodal MoE Inference |
| 2026-09-29 | `2609.38485v1` | Beyond Layers: Position-Resolved Gradient Conflict and Position-Aware Modulation for Unified Multimodal Models |
| 2026-09-29 | `2609.37225v1` | ResComEmb: Effective and Efficient Multimodal Embedding via Residual Homogeneity Compression |
| 2026-09-29 | `2609.37096v2` | Why MLLMs Struggle to Count: Overcoming Individuation and Aggregation Bottlenecks with ConvStack |
| 2026-09-29 | `2609.37052v1` | OmniRoute: Mapping Temporal Semantic Evidence to Audio-Visual Token Budgets for Efficient Omnimodal Large Lang |
| 2026-09-29 | `2609.36916v2` | Representation Dynamics Reveal Semantic Saliency and Similarity for Visual Token Pruning in MLLMs |
| 2026-09-29 | `2609.36782v1` | Decoding Affective Nuances: Enhancing MLLMs via Hierarchical Emotion Reasoning and Contrastive Discriminative  |
| 2026-09-28 | `2609.36145v1` | From Sharp Eyes to Expert Mind: Internalizing Expert Knowledge in MLLMs for Tampered Text Detection |
| 2026-09-28 | `2609.35457v1` | How Far Are We from Removing the Visual Encoder? Scaling Laws for Encoder-Free Multimodal Pretraining |
| 2026-09-28 | `2609.34977v1` | SPIDER: Multi-Layer Semantic Token Pruning and Adaptive Sub-Layer Skipping in Multimodal Large Language Models |
| 2026-09-28 | `2609.34972v1` | Just MLPs: Efficient Visual State Reconstruction for Multimodal Language Models |
| 2026-09-28 | `2609.34942v1` | Resolution as a First-Class Decision: Task-Conditioned Routing for Efficient Multimodal Large Language Models |
| 2026-09-28 | `2609.34867v1` | P4Q: Co-designing Token Pruning and Quantization for Vision-Language Model Acceleration |
| 2026-09-28 | `2609.34408v1` | Distilling Visual Reasoning into Text Space |
| 2026-09-28 | `2609.34330v2` | MiCo: Mutual Information Coverage Optimization through Semantic Erasure Modeling for Efficient MLLM Inference |
| 2026-09-28 | `2609.34196v1` | ConvCue: Complementary Visual Inductive Biases for Vision-Language Models |
| 2026-09-27 | `2609.33518v1` | SceneScaffold: Active Scene-State Construction for Unified 3D Scene Understanding |
| 2026-09-26 | `2609.32353v2` | Fewer Tokens, More Self-Teaching: On-Policy Self-Distillation for Extreme Visual Token Reduction |
| 2026-09-26 | `2609.32333v1` | Progressive-View On-Policy Distillation for Regional-to-Global Transfer in Multimodal LLMs |
| 2026-09-25 | `2609.30783v1` | Skip the Talk, Re-Focus on Vision: Latent Reasoning for Reasoning Segmentation in Multimodal Large Language Mo |
| 2026-09-24 | `2609.30210v1` | The Alignment Illusion in Multimodal Large Language Models |
| 2026-09-24 | `2609.29940v1` | Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Selective Probability Mass Concentration |

**`abs:"native resolution" AND abs:"vision-language"`** — 命中 5，其中新增 5

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.07729v1` | Foveated Compression: Selective High-Resolution Preservation for Token-Efficient VLMs |
| 2026-10-03 | `2610.04318v1` | Rethinking Long-Video Efficiency: A Joint Allocation Perspective on Frames, Pixels, and Front-End Latency |
| 2026-09-18 | `2609.21712v2` | ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation |
| 2026-09-09 | `2609.13287v1` | LLaDA-UI: Bringing Block-wise Diffusion to Vision-Language GUI Agents |
| 2026-06-18 | `2606.20523v2` | SARLO-80: Worldwide Slant SAR Language Optic Dataset 80cm |

**`abs:"contrastive" AND abs:"vision-language" AND abs:"pretraining"`** — 命中 42，其中新增 29（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-30 | `2609.39924v1` | CoVisco: Codec-Native Vision Encoder with Native Token Compression for Unified Image-Video Understanding |
| 2026-09-30 | `2609.38989v1` | Cue the Flow: Steering Flow-Matching Policies for Open-World Delivery Manipulation |
| 2026-09-28 | `2609.34944v1` | Adjoint Guidance Flow: Amortized Critic Guidance for VLA Policies |
| 2026-09-28 | `2609.34826v1` | WM-VLM: Probing Internal World Models for Interleaved Visual-Textual Reasoning |
| 2026-09-28 | `2609.34479v1` | SentZero: An Enhanced Sentence-Centric Vision-Language Pretraining for Multi-Task Zero-Shot Chest X-Ray Analys |
| 2026-09-28 | `2609.34206v1` | WorldGuide: Learning Success-Failure Boundaries in Latent World Models for Vision-Language-Action Policies |
| 2026-09-26 | `2609.32582v1` | Cool Embeddings: Predicting Galaxy Cluster Cooling Times in IllustrisTNG and TNG-Cluster with AstroCLIP |
| 2026-09-26 | `2609.32108v1` | SAMBAR: Selective Anchoring via Method of Multipliers for Balanced Knowledge Acquisition and Retention in Visi |
| 2026-09-25 | `2609.31985v1` | Does Vision-Language Pretraining Granularity Matter? A Controlled Evaluation of Vision-Language Objectives Acr |
| 2026-09-24 | `2609.29156v1` | Med-AR: Autoregressive Vision-Language Pretraining for Long-Tailed Chest X-Ray Classification and Uncertainty- |
| 2026-09-17 | `2609.20648v2` | SkipVLA: Skipping VLA Steps with Classical Planning for Fast Robot Manipulation |
| 2026-09-15 | `2609.16664v1` | Bridging the Perceptual Gap: Residual-Enhanced Downscaling and Manifold-Aware Perception Alignment Adaptation  |
| 2026-09-11 | `2609.12454v1` | Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Imag |
| 2026-09-10 | `2609.12035v2` | Reading the Whole Heart: Latent-Attention Masked Autoencoders for Multimodal Cardiac Representation Learning |
| 2026-09-04 | `2609.05583v1` | An overview of 3D Vision-Language Models |
| 2026-09-03 | `2609.03391v2` | Exploring the Potential of Contrastive Language-Image Pre-training for Multi-Source Remote Sensing Data |
| 2026-09-02 | `2609.03085v2` | Solving the Needle-in-a-Haystack Problem in Mammography Vision-Language Model with Differentiable Subset Sampl |
| 2026-09-01 | `2609.01757v1` | AlphaRAD: Grounded Zero-Shot Classification in Chest Radiology via $α$-Corrected Binary Cross Entropy and Fact |
| 2026-08-31 | `2609.00411v1` | Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models |
| 2026-08-26 | `2608.25701v1` | Skeleton-based Zero-Shot Spatio-Temporal Action Localization via Weakly-Supervised Pretraining |
| 2026-08-16 | `2608.15456v1` | AlignJEPA: Predictive Vision-Language Alignment for Remote Sensing Foundation Models |
| 2026-08-11 | `2609.26168v1` | TRACE: Transparent Retrieval for Abstract Concept Evaluation |
| 2026-08-06 | `2608.05960v1` | Big, Bright, or Invisible: A Frozen-Feature Benchmark of 3D CT Foundation Models |
| 2026-08-05 | `2608.05389v1` | Text-Guided Refinement of Multi-sequence Glioma Subregion Segmentation with a Vision-Language Foundation Model |
| 2026-07-31 | `2608.00279v1` | Learning to See Locally and Align Clinically with Pathology Semantics for Radiology Report Generation |
| 2026-07-31 | `2608.00231v2` | Learning How Much, Not Just What: Cross-Patient Burden Order for CT Vision-Language Pretraining |
| 2026-07-29 | `2609.17572v1` | Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation  |
| 2026-07-25 | `2607.23271v1` | What CLIP Knows but Cannot Say: Recovering Negation from Frozen Intermediate Features |
| 2026-07-24 | `2607.21970v1` | TextSLIP: Text Self-Supervised CLIP for Medical Report Generation |

## 06 · 指令微调与对齐

**`abs:"visual instruction tuning"`** — 命中 8，其中新增 8

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-05 | `2609.05885v1` | One Rate Is Not Enough: Adaptive Anisotropic Learning Rates for LoRA Fine-Tuning |
| 2026-08-31 | `2608.30209v3` | DICS: Exploring Data Intrinsic Consistency for Visual Instruction Selection |
| 2026-08-27 | `2608.26820v1` | LLaVAFlow: Preserving Latent Alignment Flow for Parameter-Efficient Multimodal Fine-Tuning |
| 2026-08-10 | `2608.09907v1` | DistMoE: Private-data Rehearsal-free Routing in Mixture-of-Experts for Distributed Instruction Tuning |
| 2026-08-03 | `2608.01635v1` | Mitigating Visual Degradation in MLLMs via Spatial-Spectral Visual Anchor Learning |
| 2026-07-29 | `2607.26596v1` | Decoupled Visual Processing: Efficient Multimodal Adaptation via Modality-Specific Transformer Substitution |
| 2026-07-01 | `2607.00465v1` | StochasT: Learning with Stochastic Turn Depth for Visual Instruction Tuning |
| 2026-06-17 | `2606.19100v5` | AMALIA-VL: A Native European Portuguese Open-Source Vision and Language Model |

**`abs:"LoRA" AND abs:"vision-language"`** — 命中 86，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.10163v1` | Beyond Anonymous Captions: Grounding Character Identity in Video Captioning and Question Answering |
| 2026-10-07 | `2610.09940v1` | Juno: Taming Predictive Latents for Vision-Language-Action Models |
| 2026-10-05 | `2610.06184v1` | Arm-wise Compositional Generalization in Dual-Arm Vision-Language-Action Models |
| 2026-10-01 | `2610.02388v1` | Octrees as an Explicit 3D Language |
| 2026-10-01 | `2610.02148v1` | Omni-Embed-Mini: Binding Modalities Without Forgetting via Dense Distillation |
| 2026-09-30 | `2609.39820v1` | Learning from Runtime Feedback through Failure-Bank Self-Evolution for Vision-Language-Action Models |
| 2026-09-29 | `2609.38391v1` | Team MSU GenText-Forensics Challenge 2026 Technical Report |
| 2026-09-29 | `2609.36492v1` | Benchmarking Vision-Language Models on Synapse Detection and Proofreading in Connectomics |
| 2026-09-28 | `2609.34968v1` | RoboFL: Federated Expert Assembly for World Action Models |
| 2026-09-28 | `2609.34561v1` | Brain-Conditioned Action Policies for Neural Motor Decoding |
| 2026-09-27 | `2609.33694v1` | Seeing and Solving Are Not Enough for Vision-Language Models |
| 2026-09-27 | `2609.33158v1` | FOCUS: Benchmarking Retinal Model Generalization from Foundation Vision Encoders to Multimodal LLMs |
| 2026-09-27 | `2609.33125v1` | Train Together or Merge Later? Unifying VLA Experts via a Shared Action Interface |
| 2026-09-27 | `2609.33082v1` | SemReward-VL: Semantic Reward-Guided Video-Language Adaptation for Developmental Behavior Assessment |
| 2026-09-25 | `2609.31450v1` | From Reward Signal to Visual Utility: A Controlled Audit of Medical VLM Post-Training |
| 2026-09-21 | `2609.24334v1` | Multitask Jet Analysis with Vision-Language Models: A Physics-Informed Four-Panel Representation |
| 2026-09-21 | `2609.24198v1` | SKstars at SHROOM: Visions Agreement-Guided Ensembling of Zero-Shot and LoRA-Adapted Vision--Language Models |
| 2026-09-20 | `2609.23655v1` | Reassessing Global Gradient-Norm Imbalance in BLIP Fine-Tuning Across Physical Domains |
| 2026-09-19 | `2609.23012v1` | CrowdCue: Specialist-Cue Conditioning for Vision-Language Crowd Counting |
| 2026-09-19 | `2609.22925v1` | "Dear LLaVA, Please Drive": A Depth-Aware Vision-Language Agent for Closed-Loop Robotic Control |
| 2026-09-16 | `2609.18860v2` | Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Dete |
| 2026-09-16 | `2609.18084v2` | Not All Layers Need Tuning: Diagnosing and Directing Adaptation in Vision-Language-Action Models |
| 2026-09-15 | `2609.16486v1` | VPRef: A Cross-Domain Benchmark for Referring Remote Sensing Image Segmentation |
| 2026-09-14 | `2609.15603v1` | A Unified Vision-Language Model for PSMA PET/CT Report Generation, Visual Question Answering, and Lesion Segme |
| 2026-09-14 | `2609.15229v1` | Pre-PEFT Probing: Weight Statistics and Perturbation Robustness for Layer Selection in VLM Vision Encoders |
| 2026-09-13 | `2609.14523v1` | Selective Tool Use for Agentic Change Visual Question Answering in Remote Sensing |
| 2026-09-13 | `2609.14350v1` | Two-Stage Mixture-of-LoRA for Multi-Task Medical Vision-Language Learning |
| 2026-09-13 | `2609.14284v2` | Vision-Language Models for Criterion-Level Grading of Handwritten Examinations in Outcome-Based Education |
| 2026-09-10 | `2609.11310v1` | Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Lan |
| 2026-09-08 | `2609.08188v1` | Bridging the Semantic-Utility Gap in Multimodal RAG via Generator-in-the-Loop Alignment |

**`abs:"instruction tuning" AND abs:"multimodal"`** — 命中 49，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-29 | `2609.37889v1` | ReCAP: Retrieval-Guided Capability Reuse for Multimodal Continual Instruction Tuning |
| 2026-09-29 | `2609.37283v1` | SAM Meets VLM: Parameter-Decoupled Full-Parameter Training for Unified Medical Reasoning and Segmentation |
| 2026-09-28 | `2609.36121v1` | Render Before Reading: Visual Rendering as a Prompt Injection Defense |
| 2026-09-17 | `2609.20325v1` | AgriScope: Pixel-Grounded Multimodal Understanding for Agricultural Images |
| 2026-09-12 | `2609.13742v1` | Hyper-LLaVA: Hyperbolic Uncertainty-aware Modality-Balanced Routing for Multimodal Continual Instruction Tunin |
| 2026-09-01 | `2609.00654v1` | SciTrue: Reliable Scientific Claim Validation with Frontier and Open Language Models at the NTCIR SciClaimEval |
| 2026-08-30 | `2608.29661v1` | OmniClimate-TC: Physics-Aware Visual Abstractions for Multimedia Reasoning over Tropical Cyclones |
| 2026-08-29 | `2608.29088v2` | HANIA: Planner-Guided Multimodal Graph Evidence Selection for Grounded Question Answering |
| 2026-08-28 | `2608.27867v1` | CoRe-MoE: Compact Reusable MoE for Continual Multimodal Instruction Tuning |
| 2026-08-27 | `2609.25040v1` | BananaVLM: A Domain-Adapted Vision Language Model for Banana Crop Disease Diagnosis |
| 2026-08-27 | `2608.26820v1` | LLaVAFlow: Preserving Latent Alignment Flow for Parameter-Efficient Multimodal Fine-Tuning |
| 2026-08-25 | `2608.25198v1` | Tunable Tool-Call Rates in LLM Agents via Representation Steering |
| 2026-08-16 | `2608.15516v1` | UniFed-VLM: Federated Instruction Tuning for Vision-Language Models with Multiple Heterogeneity |
| 2026-08-14 | `2608.14198v1` | MINT: A Universal Zero-Shot Predictor for Transaction Data |
| 2026-08-10 | `2608.09907v1` | DistMoE: Private-data Rehearsal-free Routing in Mixture-of-Experts for Distributed Instruction Tuning |
| 2026-08-08 | `2608.08107v1` | NeuPAT: Neuron-aware Plasticity Allocation Tuning for Language-Preserving MLLMs |
| 2026-08-06 | `2608.06246v1` | A Six-Dimensional Taxonomy of Post-Training Adaptation Techniques with Applications in AI Governance |
| 2026-08-06 | `2608.05691v1` | SciQNet: Two-Stage Multimodal Adaptation for Scientific Image Quality Assessment |
| 2026-08-03 | `2608.02790v1` | Confident but Unreliable: A Behavioral Safety Audit of Vision-Language Models on Brain MRI |
| 2026-08-03 | `2608.01635v1` | Mitigating Visual Degradation in MLLMs via Spatial-Spectral Visual Anchor Learning |
| 2026-08-02 | `2608.01437v1` | Beyond Routing Saturation: A Long-Horizon Class-Incremental Perspective on Expert Routing in Multimodal Contin |
| 2026-07-29 | `2607.26947v2` | Progressive Multimodal Alignment for Continual Instruction Tuning |
| 2026-07-29 | `2607.26596v1` | Decoupled Visual Processing: Efficient Multimodal Adaptation via Modality-Specific Transformer Substitution |
| 2026-07-29 | `2608.00068v1` | SafeBuild-Bench: A Temporal-Robust Construction Safety Benchmark with Graph-Enhanced Data Mining |
| 2026-07-28 | `2607.25947v1` | A Cost-Effective Multimodal LLM Reasoning Framework for Question Answering over Irregular Clinical Time Series |
| 2026-07-27 | `2607.23917v1` | Gaze-to-text Generation: Beyond Categorical Decoding of Human Attention |
| 2026-07-15 | `2607.13860v1` | Towards Enhancing 3D Spatial Reasoning in Medical Multimodal Large Language Models |
| 2026-07-13 | `2607.11102v1` | CHARM: Charge Calibration and Acoustic Rescue for LLM-based Multimodal Sarcasm Detection |
| 2026-07-11 | `2607.10308v1` | Generalize LMMs to Versatile Visual Modalities via Fabricated Modality Synthesis |
| 2026-07-11 | `2607.10299v1` | Empowering Long-form Omni-modal Understanding with Robust Audio Perception |

**`abs:"parameter-efficient" AND abs:"multimodal"`** — 命中 38，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-04 | `2610.05129v1` | Representation--Behavior Alignment for Explainable Weakly-Supervised Video Anomaly Detection |
| 2026-09-27 | `2609.33640v1` | SafeMol: Dual-Modality Safety Alignment for Molecular Multimodal Models |
| 2026-09-12 | `2609.13654v1` | Multimodal Foundation Models Adaptation based on Domain-Aware Relaxed Orthogonal Subspace for Remote Sensing |
| 2026-09-06 | `2609.06497v1` | Vision-Guided Text Prompt Tuning for Multimodal Sentiment Analysis |
| 2026-09-05 | `2609.06188v1` | MVFA: A Multi-View Text-Guided Multimodal Fusion LLM Adapter for Sentiment Analysis and Emotion Recognition |
| 2026-09-02 | `2609.05539v1` | Dual-Latent Memory Routing for Vision-Language Reasoning |
| 2026-09-02 | `2609.02101v1` | Federated LoRA Adaptation of BiomedCLIP Across Four International Chest X-Ray Cohorts |
| 2026-08-31 | `2608.30712v1` | GUIDE: Guiding Internal Evidence with Language Instructions |
| 2026-08-30 | `2608.29802v1` | Foundation and Multimodal Large Language Models for Face Presentation and Morph Attack Detection |
| 2026-08-29 | `2608.29357v1` | LiteSearch-VL: Small Multimodal Search Agents via Trajectory Distillation and Synthetic Step-DPO |
| 2026-08-28 | `2608.27867v1` | CoRe-MoE: Compact Reusable MoE for Continual Multimodal Instruction Tuning |
| 2026-08-19 | `2608.19355v1` | GRACE: Grounded Reasoning via Adapter Composition and Evidence-Aware Calibration for Educational Visual Questi |
| 2026-08-18 | `2608.18386v1` | TTSD-FAR: Test-Time Self-Distillation with Fisher-Anchored Restoration for Missing-Modality Emotion Recognitio |
| 2026-08-18 | `2608.17514v2` | SE-MoLoRA: Shared-Expert LoRA Adapters for Domain-Specific Photographic Assessment |
| 2026-08-16 | `2608.15516v1` | UniFed-VLM: Federated Instruction Tuning for Vision-Language Models with Multiple Heterogeneity |
| 2026-08-10 | `2609.26182v1` | Modality-Gated Deep Adapters: Adding a Modality to a Frozen Embedding Model with Exact Preservation |
| 2026-08-08 | `2608.08244v2` | FemWear: A Parameter-Efficient Wearable Foundation Model for Women's Health |
| 2026-08-06 | `2608.06246v1` | A Six-Dimensional Taxonomy of Post-Training Adaptation Techniques with Applications in AI Governance |
| 2026-08-04 | `2608.04130v1` | Radar4D-VLM: Proposal-Grounded Temporal 4D Radar Reasoning Across Frozen Language Models |
| 2026-08-01 | `2608.00847v1` | Models as Tools: An Agentic Coordination Framework for Unified Multimodal Visual Tracking |
| 2026-07-29 | `2607.26596v1` | Decoupled Visual Processing: Efficient Multimodal Adaptation via Modality-Specific Transformer Substitution |
| 2026-07-25 | `2607.23245v1` | FedTaste: Topology-Aware Structural Transfer for Multimodal Federated Learning with Missing Modalities |
| 2026-07-21 | `2607.19270v1` | GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models |
| 2026-07-21 | `2607.18716v1` | Continual Video-MLLM Adaptation over Evolving Domains |
| 2026-07-16 | `2607.15241v1` | Beyond the Leaderboard: Design Lessons for Trustworthy Multimodal VQA |
| 2026-07-15 | `2607.13466v1` | PQFA: Parallel Quantum Feature Augmentation of Fused Representations for Multimodal Classification |
| 2026-07-13 | `2607.11839v1` | LoRA-Based Cascaded Multimodal Fusion for Action Recognition in Medical Training Environments |
| 2026-07-10 | `2607.09583v1` | Promptable Concept Segmentation from Above: Evaluating SAM 3's Zero-Shot and One-Shot Capabilities in Remote S |
| 2026-07-07 | `2607.05736v1` | Multimodal Molecular Representation Learning with Graph Neural Networks, Deep & Cross Networks, and SMILES Emb |
| 2026-07-03 | `2607.02930v1` | CL-Anomaly: Layer-Adaptive Mixture-of-Experts with Multimodal Large Language Model for Continual Learning in A |

**`abs:"catastrophic forgetting" AND abs:"multimodal"`** — 命中 21，其中新增 21

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-29 | `2609.37889v1` | ReCAP: Retrieval-Guided Capability Reuse for Multimodal Continual Instruction Tuning |
| 2026-09-22 | `2609.25841v1` | Metric-Bench: Exploring In-context Spatial Metric Reasoning in VLMs for Indoor Scenes |
| 2026-09-21 | `2609.24813v1` | INTCORT: Training-Free Spatial Reasoning Enhancement for Vision-Language Models via Input Transformations and  |
| 2026-09-07 | `2609.07009v1` | NeuCME: Toward Dynamic Multimodal Continual Learning via Neural Combinatorics of Multiple Experts |
| 2026-08-27 | `2608.26820v1` | LLaVAFlow: Preserving Latent Alignment Flow for Parameter-Efficient Multimodal Fine-Tuning |
| 2026-08-26 | `2608.26095v1` | A Visual Dependence-Aware Framework for Multimodal Unsupervised Continual Post-Training |
| 2026-08-24 | `2608.23242v1` | Mind the Couch! Eliciting MLLM Reasoning in Interior Design via Weak-to-Strong Task Vector Injection |
| 2026-08-12 | `2608.11758v1` | AWARe: Mitigating Catastrophic Forgetting via Activation-Weighted Adaptive REtention |
| 2026-08-04 | `2608.03660v1` | Taming the Implicit: Dual-Channel Risk-Aware Reinforcement Fine-Tuning for Continual Multimodal Post-Training |
| 2026-07-30 | `2607.28050v1` | IndustryForge-27B: A Domain-Enhanced Multimodal Foundation Model for Industrial CAD |
| 2026-07-30 | `2607.27665v1` | FedOGL: Combating Catastrophic Forgetting in Federated Open-World Multimodal Graph Learning |
| 2026-07-21 | `2607.18716v1` | Continual Video-MLLM Adaptation over Evolving Domains |
| 2026-07-13 | `2607.12112v2` | Continual Learning with Elastic Regularization and Synthetic Replay for Federated MLLM Fine-Tuning |
| 2026-07-05 | `2607.04364v3` | RL Forgets! Towards Continual Policy Optimization |
| 2026-07-05 | `2607.20511v1` | SiGMA: Sign-Guided Merging and Adaptation for Multimodal Continual Instruction Tuning |
| 2026-07-04 | `2608.26155v1` | VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation |
| 2026-07-01 | `2607.00434v1` | Information-Regularized Attention for Visual-Centric Reasoning |
| 2026-07-01 | `2607.00302v1` | Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs |
| 2026-07-01 | `2607.00293v1` | Rosetta: Composable Native Multimodal Pretraining |
| 2026-06-23 | `2606.24963v1` | Curvature-Guided Mixing for MLLM Adaptation |
| 2026-06-16 | `2606.18472v1` | Domain Generalizable Adaptation of 3D Vision-Language Models via Regularized Fine-Tuning |

**`abs:"preference alignment" AND abs:"vision-language"`** — 命中 8，其中新增 8

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-23 | `2609.31746v1` | VisionPsy-Nano: Improving Accuracy, Efficiency, and Reliability in On-Device Vision-Language Models |
| 2026-09-08 | `2609.08188v1` | Bridging the Semantic-Utility Gap in Multimodal RAG via Generator-in-the-Loop Alignment |
| 2026-09-06 | `2609.13259v1` | TryOnReward: Learning Foveated Consistency for Reinforcement Fine-Tuning of Virtual Try-On |
| 2026-09-04 | `2609.04961v1` | SAM-D2Q: Aligning Multimodal Doc2Query with Search Demand and Conversion for E-commerce |
| 2026-08-26 | `2608.25452v1` | VGA-BenchV2: An Expanded Unified Benchmark and Multi-Model Framework for Evaluating Video Aesthetics and Gener |
| 2026-08-14 | `2608.14260v1` | Personalized Digital Semantic Communication for Image Transmission with Vision-Language Models |
| 2026-07-02 | `2607.01721v1` | CoRe: Combined Rewards with Vision-Language Model Feedback for Preference-Aligned Reinforcement Learning |
| 2026-06-24 | `2606.28401v1` | Vision-driven Preference Synthesis for Mitigating Hallucinations in VLMs |

## 07 · 通用评测与幻觉

**`abs:"hallucination" AND abs:"multimodal large language"`** — 命中 78，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-04 | `2610.05610v2` | SEA-LM: Egocentric Spatial Audio Understanding for Wearable Microphone Arrays |
| 2026-10-03 | `2610.06972v1` | BoT-Feedback: Grounding Multimodal Reasoning in Biomechanical Evidence for Explainable Human Action Feedback |
| 2026-10-02 | `2610.02887v1` | Revealing Epistemic Uncertainty in MLLMs via Causal-Invariant Masking |
| 2026-09-29 | `2609.36562v1` | ThinkingGuard: Decoding Implicit Hazards via Step-by-Step Risk Attribution in Multimodal Large Language Models |
| 2026-09-28 | `2609.35536v1` | Look Before You Judge: Training-Free Region Mining for Grounded and Explainable Deepfake Detection |
| 2026-09-28 | `2609.34330v2` | MiCo: Mutual Information Coverage Optimization through Semantic Erasure Modeling for Efficient MLLM Inference |
| 2026-09-24 | `2609.29940v1` | Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Selective Probability Mass Concentration |
| 2026-09-23 | `2609.28570v1` | DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs |
| 2026-09-18 | `2609.21828v1` | Touvigation: Embodied Adaptive Object Acquisition for Blind and Low-Vision Users in Unfamiliar Indoor Environm |
| 2026-09-18 | `2609.21675v1` | DRT: Dense Reasoning Trace for Efficient and Grounded Multimodal Reasoning |
| 2026-09-15 | `2609.17269v1` | Semantic-Spatial Agreement Verification for Mitigating Object Hallucination in Multimodal Large Language Model |
| 2026-09-15 | `2609.17248v1` | Video-HolmesV2: Can MLLMs Reason with Spatio-Temporal Audio-Visual Evidence in Long Videos? |
| 2026-09-15 | `2609.16601v1` | SAVOR: Self-Aware Visual Grounding via Confidence-Calibrated Reinforcement Learning for Multimodal Hallucinati |
| 2026-09-12 | `2609.13849v1` | UniCAR-RL: Seeing Better before Thinking Deeper in Visual Mathematics |
| 2026-09-10 | `2609.11244v1` | OmniHallu: Unified Hallucination Detection for Cross-Modal Comprehension and Generation in Multimodal Large La |
| 2026-09-10 | `2609.11154v1` | Multi-Faceted Evaluation and Mitigation of Emotion Hallucinations in MLLMs |
| 2026-09-08 | `2609.08300v1` | Human-Centric Image Captioning with Subject-Centered Spatial Understanding |
| 2026-09-05 | `2609.09206v2` | MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads |
| 2026-09-03 | `2609.03414v1` | StrixAE: An Intelligent Agent for Audio Enhancement under Complex Distortion Coupling in Real-World Scenarios |
| 2026-08-31 | `2609.00231v1` | Beyond Language Priors: Diagnosing and Fixing Visual-Origin Hallucinations in Multimodal LLM |
| 2026-08-31 | `2608.30653v1` | Fine-Grained Multi Image Object Hallucination Benchmark |
| 2026-08-28 | `2609.29607v1` | STRAND: Benchmarking and Improving Object-Centric Spatio-Temporal Monitoring in Video Large Language Models |
| 2026-08-27 | `2608.28707v1` | ReVA: A Region-Aware Visual Assistant for Visually Grounded Question Answering |
| 2026-08-26 | `2608.25986v1` | Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs |
| 2026-08-25 | `2608.23974v1` | Boot-and-Feedback Framework for Generalist-Expert Model Collaboration in Breast Ultrasound Diagnosis |
| 2026-08-24 | `2608.23242v1` | Mind the Couch! Eliciting MLLM Reasoning in Interior Design via Weak-to-Strong Task Vector Injection |
| 2026-08-23 | `2608.22367v1` | Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs |
| 2026-08-21 | `2609.27890v1` | RelCheck: Dual-Evidence Spatial Grounding for VLM Hallucination Correction |
| 2026-08-19 | `2608.18586v1` | OmniHandwritingOCR: A Diagnostic Benchmark for Evaluating Multimodal LLMs in Handwritten OCR Scenarios |
| 2026-08-18 | `2608.17895v1` | BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models |

**`abs:"MMMU"`** — 命中 15，其中新增 15

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-05 | `2610.05897v1` | Fitting Vision Adapters at Frontier Scales |
| 2026-09-29 | `2609.38413v1` | VidHarness: Evolving Agent Harnesses for Cost-Efficient Long Video Understanding |
| 2026-09-29 | `2609.36651v1` | FocusVTC: Efficient and High-Performance Visual Text Compression with Adaptive Resolution |
| 2026-09-27 | `2609.33414v1` | TTRSD: Test-Time Reinforcement Learning with Self-Distillation for Vision-Language Models |
| 2026-09-22 | `2609.26333v1` | Disaggregated Quantization: Specializing LLM Prefill and Decode |
| 2026-09-14 | `2609.15051v1` | Not All Prompts Are Equal: Exploration-Guided Prompt Scaffolding for Multimodal Reinforcement Post-Training |
| 2026-09-13 | `2609.14258v1` | SPARK: Representation-Level KV Memory Alignment for Safer Vision-Language Models |
| 2026-08-27 | `2608.26495v2` | Video-FLAIR: Not Whether to Reason, But How |
| 2026-08-24 | `2608.23268v1` | Dual-Grained Agent Memory and Shapley Context Attribution for Multimodal Agentic Learner |
| 2026-08-02 | `2608.01207v2` | It's the Decoding Format, Not the Perturbation: Auditing Consistency-Based Selection for Vision-Language Test- |
| 2026-07-15 | `2607.14333v1` | SD-MAR: Multi-image Analytical Reasoning via Synthetic Data and Reinforcement Learning |
| 2026-07-09 | `2607.08215v1` | On the Limitations of Non-GPU AI Accelerators for Large-Model Inference: A Field Study of MoE and Multimodal S |
| 2026-06-25 | `2606.27376v1` | Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewar |
| 2026-06-16 | `2606.18385v1` | CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework |
| 2026-06-12 | `2606.14629v3` | When Good Verifiers Go Bad: Silent Negative Transfer in Verifier-Guided VLM Training |

**`abs:"MMBench"`** — 命中 9，其中新增 9

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-07 | `2610.09361v1` | From Chunks to Functional Evidence: Function-Aware Retrieval for EDA Documentation QA |
| 2026-09-30 | `2609.40362v1` | Multimodal Flow: Unified Flow Modeling of Language and Vision in Embedding Spaces |
| 2026-09-15 | `2609.16601v1` | SAVOR: Self-Aware Visual Grounding via Confidence-Calibrated Reinforcement Learning for Multimodal Hallucinati |
| 2026-08-27 | `2608.28707v1` | ReVA: A Region-Aware Visual Assistant for Visually Grounded Question Answering |
| 2026-08-03 | `2608.01979v2` | ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs |
| 2026-07-27 | `2607.24424v1` | MAViE: A Multi-scale Adaptive Vision Encoder for Fine-grained Visual Perception and Efficient Multimodal Reaso |
| 2026-07-15 | `2607.14333v1` | SD-MAR: Multi-image Analytical Reasoning via Synthetic Data and Reinforcement Learning |
| 2026-07-02 | `2607.01813v1` | MMBench-Live: A Continuously Evolving Benchmark for Multimodal Models |
| 2026-06-29 | `2606.30084v1` | One Forward Beats Two: InnerZoom for Accurate and Efficient GUI Grounding |

**`abs:"POPE"`** — 命中 28，其中新增 28

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-30 | `2609.40362v1` | Multimodal Flow: Unified Flow Modeling of Language and Vision in Embedding Spaces |
| 2026-09-29 | `2609.38485v1` | Beyond Layers: Position-Resolved Gradient Conflict and Position-Aware Modulation for Unified Multimodal Models |
| 2026-09-15 | `2609.17269v1` | Semantic-Spatial Agreement Verification for Mitigating Object Hallucination in Multimodal Large Language Model |
| 2026-09-15 | `2609.16601v1` | SAVOR: Self-Aware Visual Grounding via Confidence-Calibrated Reinforcement Learning for Multimodal Hallucinati |
| 2026-09-10 | `2609.11910v2` | From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good |
| 2026-09-01 | `2610.00024v1` | Encoded but Disconnected: Decomposing Vision-Language Model Failures under a Patching Null |
| 2026-08-31 | `2609.00231v1` | Beyond Language Priors: Diagnosing and Fixing Visual-Origin Hallucinations in Multimodal LLM |
| 2026-08-30 | `2608.29996v1` | Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models |
| 2026-08-30 | `2608.29924v1` | Hallucination Mitigation for Large Vision-Language Models via Implicit Feature Stabilization |
| 2026-08-29 | `2608.29193v1` | HalluPrism: When Multimodal Uncertainty Should Diagnose, Not Decide |
| 2026-08-29 | `2608.29092v1` | EviAnchor: Mitigating Hallucinations in Large Vision-Language Models via Regional Visual Evidence Compensation |
| 2026-08-27 | `2608.28707v1` | ReVA: A Region-Aware Visual Assistant for Visually Grounded Question Answering |
| 2026-08-26 | `2608.25865v1` | Instability of Böhm's Einstein metrics |
| 2026-08-08 | `2608.08167v1` | Wiener Representation Filtering for VLM Hallucination Suppression |
| 2026-07-29 | `2607.27145v1` | Explainable and Resource-Efficient Spatial Reasoning in Multimodal LLMs for Decision-Critical Applications |
| 2026-07-29 | `2607.26596v1` | Decoupled Visual Processing: Efficient Multimodal Adaptation via Modality-Specific Transformer Substitution |
| 2026-07-28 | `2607.25527v1` | Argus-Unified: Towards A Compact and Economical Unified Model for Image Understanding and Generation |
| 2026-07-18 | `2607.16727v1` | Constraint-Anchored Reasoning Traces |
| 2026-07-14 | `2607.12962v1` | Form, Not Content? A Preregistered, Placebo-Controlled Evaluation of Learned Error-Conditioned Self-Repair Thr |
| 2026-07-13 | `2607.11791v1` | Cosmology: 100 years after A. A. Friedmann |
| 2026-07-09 | `2607.08059v1` | When Thinking Hurts: Epistemic Signals in the Reasoning Chains of Visual Language Models |
| 2026-07-07 | `2607.06726v1` | A Good Initialization is All You Need for Faithful Visual Attribution |
| 2026-07-05 | `2607.04163v1` | SeeMe: Mitigating Hallucinations in Large Vision-Language Models through Effective Visual Token Engineering |
| 2026-07-03 | `2607.03143v1` | Text as Partial Constraint: Core-Residual Alignment for Robust Vision-Language Learning |
| 2026-07-01 | `2607.00784v1` | LeVLJEPA: End-to-End Vision-Language Pretraining Without Negatives |
| 2026-06-28 | `2606.29431v5` | FADE: Mitigating Hallucinations by Reducing Language-Prior Dominance in Large Vision-Language Models |
| 2026-06-27 | `2606.28862v2` | HKVLM: Faithful Query--Region Binding for Frozen-Detector Visual Grounding |
| 2026-06-14 | `2606.15821v2` | The Truth Stays in the Family: Enhancing Contextual Grounding via Inherited Truthful Heads in Model Lineages |

**`abs:"visual question answering" AND abs:"hallucination"`** — 命中 18，其中新增 18

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-29 | `2609.37685v2` | PAIQ: Patch-Aligned Semantic Injection via Residual Rotation |
| 2026-09-16 | `2609.18562v1` | Sim-to-Real Traffic Scene Understanding by Decoupling Semantics from Caption Generation with V-JEPA |
| 2026-09-15 | `2609.16601v1` | SAVOR: Self-Aware Visual Grounding via Confidence-Calibrated Reinforcement Learning for Multimodal Hallucinati |
| 2026-08-31 | `2608.30475v1` | ImageEval 2026: Culturally Grounded Arabic Multimodal Evaluation |
| 2026-08-28 | `2609.29604v1` | PROVE: Proof-guided Regime-aware Operator Verification for Hallucination Detection in Medical Visual Question  |
| 2026-08-27 | `2608.28707v1` | ReVA: A Region-Aware Visual Assistant for Visually Grounded Question Answering |
| 2026-08-18 | `2608.17427v1` | Counterfactual Anatomy-guided Spatial-Temporal Decoding for Annotation-Free Hallucination Mitigation in Medica |
| 2026-08-17 | `2608.16805v1` | Diagnosing Dense Same-Class Attribute Misbinding in Large Vision-Language Models |
| 2026-08-11 | `2608.10964v1` | CARE: Confidence-Aware Reasoning for Reliable Medical VQA |
| 2026-07-30 | `2607.28374v1` | LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning with a Structured Evidence Ledger |
| 2026-07-29 | `2607.26885v1` | SCALPEL: Semantic Cross-modal Alignment via LLM-Powered Encoder Learning for Medical Vision-Language Represent |
| 2026-07-14 | `2607.12319v1` | DM-KG: A Novel Method for Boosting Spatial Cognition of Vision-Language Models in Street View Imagery |
| 2026-07-05 | `2607.04163v1` | SeeMe: Mitigating Hallucinations in Large Vision-Language Models through Effective Visual Token Engineering |
| 2026-06-30 | `2606.32012v1` | CoMet: Context and Multiplicity Decomposition for Multimodal Uncertainty Estimation |
| 2026-06-25 | `2606.27373v1` | Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models |
| 2026-06-23 | `2606.24115v1` | A Benchmark for Hallucination Detection in VLMs for Gastrointestinal Endoscopy |
| 2026-06-22 | `2606.23354v1` | Faithful Grounded Visual Reasoning via Learned Proxy-Tokens |
| 2026-06-17 | `2606.19584v1` | Language-Instructed Vision Embeddings for Controllable and Generalizable Perception |

**`abs:"vision-language model" AND abs:"evaluation" AND abs:"bias"`** — 命中 39，其中新增 29（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-04 | `2610.05023v1` | Look Where You Say You're Looking: Self-Grounded Attention for Visual Reasoning |
| 2026-09-28 | `2609.36101v1` | One Geometry, Different Outcomes: Readout-Dependent Effects of the Modality Gap in Vision-Language Models |
| 2026-09-23 | `2609.27682v2` | Gender Bias in Vision-Language In-Context Learning |
| 2026-09-23 | `2609.31751v1` | EgoTSR++: Egocentric Spatiotemporal Reasoning for Task Progress Understanding |
| 2026-09-21 | `2609.24485v2` | VPRune: Efficient Training-free Pre-LLM Visual Token Pruning |
| 2026-09-15 | `2609.16651v1` | Mechanism-Level Evaluation for Vision-Language Models: Controlled Activation-Replacement Diagnosis of Gender B |
| 2026-09-15 | `2609.16647v1` | ViD: Vision-Dominant Gender Bias Mitigation for Large Vision-Language Models |
| 2026-09-04 | `2609.05761v1` | GeoContext: One Context Ladder, Two Failure Modes in Vision-Language Geolocation: Flat Reliance on User-Provid |
| 2026-09-01 | `2609.01691v1` | FairLens: Benchmarking Fairness in Vision-Language Models for High-Stakes Decision-Making |
| 2026-09-01 | `2609.01318v1` | Reliability Challenges in Diffusion Vision-Language Models |
| 2026-08-30 | `2608.29590v1` | Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models |
| 2026-08-26 | `2608.25375v1` | GGSS: Geodesic-Gated Spherical Steering for Inference-Time Debiasing of Generative Vision-Language Models |
| 2026-08-17 | `2608.16081v2` | SafeGesture: Evaluating Fine-Grained Hand Gesture Understanding in Vision-Language Models through Scenario-Con |
| 2026-08-14 | `2608.13969v1` | PPOM: Marginalizing Patch-Grid Phase for CLIP-Based Generalizable Vision-Language Prompt Tuning |
| 2026-08-13 | `2608.13267v1` | How Do VLMs Behave When Blind or Misled? Behavioral Evaluation of VLMs on Scientific Figures |
| 2026-08-12 | `2608.21415v1` | Mitigating Bias in Large Vision-Language Models via Counterfactual Ensemble Decoding |
| 2026-08-12 | `2609.26210v1` | Same Chart, Different Story: Bias in Vision-Language Chart Interpretation |
| 2026-08-11 | `2608.11074v1` | CapProbe: Evaluating Detailed Image Captions via Full-Scene Dense Question Answering |
| 2026-08-05 | `2608.04510v1` | GUARD: Grounding Uncertainty and Ablation-Based Risk Detection for Diffusion-Based VLAs |
| 2026-08-04 | `2608.03875v1` | Enhancing VLM Reward Models Through Structure-Aware Fine-Tuning |
| 2026-08-03 | `2608.01661v1` | FairForensics: Seeing Expressions and Parsing Demographics via Vision-Language Modeling for Generalizable Fair |
| 2026-07-31 | `2608.00119v1` | Counting the Cost of War Under Satellite Embargo: Zero-Shot Estimation of Impacted Infrastructure |
| 2026-07-30 | `2607.28609v2` | OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models |
| 2026-07-30 | `2607.28211v1` | Scaling Vision-Language Models Is Not Enough to Mitigate Bias |
| 2026-07-29 | `2609.17572v1` | Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation  |
| 2026-07-21 | `2607.18673v3` | MissingBench-Verified: Probing Vision-Language Models' Inability to Detect Missing Object Parts |
| 2026-07-13 | `2607.11228v1` | DeepBias: Adaptive In-depth Probing of Social Biases in LVLMs |
| 2026-07-13 | `2607.11044v2` | RetroHolmes: When Semantic Plausibility Fails Retrospective Physical Process Reasoning |
| 2026-07-04 | `2607.03831v1` | How Do Diffusion Classifiers Decide? A Bias-Centric Evaluation |

## uav · 无人机主线

**`abs:"UAV" AND abs:"vision-language"`** — 命中 44，其中新增 16（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-01 | `2610.02451v1` | A Simulation-Grounded Agentic VLM Framework for Wildfire Monitoring and Reporting |
| 2026-09-24 | `2609.30629v1` | FRESHLATENT: Channel-Aware Latent Adaptation for Resource-Constrained Embodied VLM Perception |
| 2026-09-16 | `2609.18448v1` | From Pixels to Semantics: Edge AI for UAV-Based Critical Infrastructure Inspection |
| 2026-09-16 | `2609.18326v1` | UAVs Meet Embodied Intelligence: Bridging Human Intents and Flying Dynamics Via Harnessing Physical-Digital AI |
| 2026-09-16 | `2609.18210v2` | Understanding Dynamic Scenes at Gigapixel Scale: Wide-Area Spatio-Temporal Perception from UAVs |
| 2026-09-08 | `2609.08402v1` | Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method |
| 2026-09-08 | `2609.08164v1` | Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation |
| 2026-08-24 | `2608.22678v1` | RACO: Reliability-Aware Coarse-Goal Optimization for Inspection-Oriented UAV Vision-Language Navigation |
| 2026-08-13 | `2608.12835v1` | AirForesight: Current-to-Future Spatial Map Imagination with Cross-Space Planning Consistency for UAV-VLN |
| 2026-08-12 | `2608.14721v1` | AeroGround: A Comprehensive Benchmark for Aerial-Ground Collaborative Reasoning |
| 2026-08-10 | `2608.09564v1` | From Semantic Grounding to Decision Optimization: A Unified Framework for Long-Horizon UAV Vision-Language Nav |
| 2026-08-08 | `2608.08045v1` | Lingjing: A Simulation Testbed for Multi-Agent Embodied Tasks in Open-Ended Cities |
| 2026-08-05 | `2608.04825v1` | Deliberate Before You Fly: Vision-Guided Spatial Deliberation for UAV See-and-Reach Navigation |
| 2026-08-03 | `2608.01906v1` | Assessing the Benefits of Combining Advanced Deep Learning Techniques for Post-Disaster Building Damage Assess |
| 2026-08-03 | `2608.01802v1` | CoNav-UAV: Cooperative Dual-Altitude Aerial Navigation via Stackelberg Learning |
| 2026-07-30 | `2607.27597v1` | A Systems Engineering Framework for Vision-Language-Enabled UAV Triage and Disaster Response |

**`abs:"aerial" AND abs:"vision-language model"`** — 命中 30，其中新增 20

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-04 | `2610.05342v1` | IRSTD-Agent: Agentic Infrared Small Target Detection via Zoom-Guided Interaction Learning |
| 2026-09-20 | `2609.23655v1` | Reassessing Global Gradient-Norm Imbalance in BLIP Fine-Tuning Across Physical Domains |
| 2026-09-16 | `2609.18448v1` | From Pixels to Semantics: Edge AI for UAV-Based Critical Infrastructure Inspection |
| 2026-09-16 | `2609.18210v2` | Understanding Dynamic Scenes at Gigapixel Scale: Wide-Area Spatio-Temporal Perception from UAVs |
| 2026-09-10 | `2609.11310v1` | Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Lan |
| 2026-09-08 | `2609.08402v1` | Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method |
| 2026-09-08 | `2609.08164v1` | Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation |
| 2026-09-01 | `2609.00628v1` | Restrict, Don't Retrain: Inference-Time VLM Guidance for Zero-Shot Aerial Segmentation |
| 2026-08-12 | `2608.14721v1` | AeroGround: A Comprehensive Benchmark for Aerial-Ground Collaborative Reasoning |
| 2026-08-04 | `2608.04175v1` | TriCLE: Tri-Modal Vision-Language Reasoning for Edge-Deployed Fine-Grained Clustering |
| 2026-08-03 | `2608.02039v2` | RSVideo: Are Your Vision-Language Models Ready for Remote Sensing Videos? |
| 2026-08-03 | `2608.01906v1` | Assessing the Benefits of Combining Advanced Deep Learning Techniques for Post-Disaster Building Damage Assess |
| 2026-07-21 | `2607.19288v1` | No Training, Better Flights: Test-Time Scaled VLMs for UAV Navigation |
| 2026-07-13 | `2607.12177v1` | The Emerging Paradigm of Geospatial Foundation Models: From Pre-Training to Agentic Reasoning |
| 2026-07-09 | `2607.08359v1` | FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation |
| 2026-07-07 | `2607.07350v1` | Towards Reliable Aerial Ground Vehicle Collaboration: An Integrated Planning and Autonomy Framework for Field  |
| 2026-07-02 | `2607.02718v1` | Diagnosing Aerial-View Object Detectors with Foundational Image Generative Models |
| 2026-07-01 | `2607.00338v1` | DroneFINE: Domain-Aware Parameter-Efficient Fine-Tuning of Vision-Language Detectors for Drone Images |
| 2026-06-19 | `2607.14125v1` | CARPRT: Class-Aware Zero-Shot Prompt Reweighting for Black-Box Vision-Language Models |
| 2026-06-17 | `2606.19277v1` | A Unified Framework for Efficient Remote Sensing Visual Question Answering: Adapting Dual, Hybrid, and Encoder |

**`abs:"drone" AND abs:"vision-language"`** — 命中 9，其中新增 7

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-25 | `2609.31005v1` | TRACKGRAPH: Online Open-Vocabulary 3D Scene Graphs via Image-Space Tracking |
| 2026-09-16 | `2609.18139v1` | Multi-View Mixture-of-Experts with Vision-Language Reranking for Cross-View Object Geo-Localization |
| 2026-09-07 | `2609.07780v2` | DroneGround: Open-Vocabulary Drone Payload Characterization Using Synthetic Data and Grounded Vision-Language  |
| 2026-08-27 | `2608.26722v1` | UniGeo: A Multi-modal Large Language Model for Text-Guided Cross-View Geo-Localization |
| 2026-08-10 | `2608.09270v1` | GRASP: Granularity-Aware Region Alignment and Semantic Prototype Learning for Fine-Grained Cross-Modal Underst |
| 2026-07-07 | `2607.06706v1` | Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review |
| 2026-07-01 | `2607.00338v1` | DroneFINE: Domain-Aware Parameter-Efficient Fine-Tuning of Vision-Language Detectors for Drone Images |

**`abs:"remote sensing" AND abs:"vision-language model"`** — 命中 33，其中新增 30（只取了最近 30 条，新增数是**下界**）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-10-06 | `2610.09085v1` | Anaximander: Interactively Running Geospatial Deep Learning Models on Any Compute Backend |
| 2026-10-05 | `2610.06196v1` | EORestore-Agent: Fidelity-Guided Agentic Restoration of Remote Sensing Images with Composite Degradations |
| 2026-09-29 | `2609.38603v1` | Aperture: Training-Free Multiscale Concept Bottlenecks for Remote Sensing |
| 2026-09-28 | `2610.00302v1` | Decoding the Disaster: Multi-Task Geospatial Reasoning with Vision-Language Models and Crowdsourced Imagery fo |
| 2026-09-26 | `2609.32813v1` | USAI-Quant: A Quantitative Reasoning Benchmark for Vision-Language Models in Built Environments |
| 2026-09-23 | `2609.28230v2` | A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing |
| 2026-09-16 | `2609.18210v2` | Understanding Dynamic Scenes at Gigapixel Scale: Wide-Area Spatio-Temporal Perception from UAVs |
| 2026-09-15 | `2609.16486v1` | VPRef: A Cross-Domain Benchmark for Referring Remote Sensing Image Segmentation |
| 2026-09-13 | `2609.14523v1` | Selective Tool Use for Agentic Change Visual Question Answering in Remote Sensing |
| 2026-09-09 | `2609.09876v1` | From Pixels to Hierarchical Sequences: Quadtree Mask Encoding for Vision-Language Binary Change Detection |
| 2026-09-03 | `2609.03391v2` | Exploring the Potential of Contrastive Language-Image Pre-training for Multi-Source Remote Sensing Data |
| 2026-09-01 | `2609.01289v1` | Agentic Multimodal Models for Environmental Hyperspectral Unmixing |
| 2026-08-26 | `2608.25485v1` | Semi-Supervised Adaptation of Vision-Language Models for Image Classification |
| 2026-08-13 | `2608.13344v1` | LongEarth-R1: Benchmarking and Aligning Vision-Language Models for Long-Horizon Earth Observation Reasoning |
| 2026-08-07 | `2608.06959v1` | Summarize First, Download Later: Onboard VLMs for Bandwidth-Efficient Earth Observation |
| 2026-08-05 | `2608.04791v1` | On the Effectiveness of Adaptation Strategies for VLM-Based Federated Learning in Remote Sensing |
| 2026-08-04 | `2608.03911v1` | UniEvo-RS: Omni-Prompt Unified Remote Sensing Segmentation with Representative Exemplar-Driven Prototype Evolu |
| 2026-08-03 | `2608.02039v2` | RSVideo: Are Your Vision-Language Models Ready for Remote Sensing Videos? |
| 2026-07-31 | `2607.29192v1` | Locally Consistent Transductive Information Maximization for Few-Shot Remote Sensing Scene Classification |
| 2026-07-23 | `2607.21036v1` | GeoThreat: Transferable Targeted Adversarial Attacks on Large Vision-Language Models for Remote Sensing Image  |
| 2026-07-18 | `2607.16819v1` | FUSAR-R1: A Large-Scale Reasoning Model for Intelligent Interpretation of SAR Images |
| 2026-07-17 | `2607.15942v2` | More with Less: a Large Scale Remote Sensing VLM with a Simple Recipe |
| 2026-07-13 | `2607.24810v1` | RRS-10K: A Multitask Vision-Language Model Benchmark for Rare Remote Sensing Image Interpretation |
| 2026-07-11 | `2607.10120v1` | WeaveEarth: Structured Evidence Construction and Reasoning for Training-Free UHR Remote Sensing Understanding |
| 2026-07-07 | `2607.06485v1` | AirflowAttack: Thermal-Airflow Adversarial Perturbations against Infrared Remote-Sensing Vision-Language Model |
| 2026-07-02 | `2607.02718v1` | Diagnosing Aerial-View Object Detectors with Foundational Image Generative Models |
| 2026-06-30 | `2606.31976v1` | TreeAgent: A Generalizable Multi-Agent Framework for Automated Bias Labeling in Forestry via Compiled Expert R |
| 2026-06-26 | `2606.28266v1` | RSICCLLM: A Multimodal Large Language Model for Remote Sensing Image Change Captioning |
| 2026-06-17 | `2606.19277v1` | A Unified Framework for Efficient Remote Sensing Visual Question Answering: Adapting Dual, Hybrid, and Encoder |
| 2026-06-15 | `2606.17246v1` | GeoDisaster: Benchmarking Orchestrated Agents for Operational Disaster Geo-Intelligence |

**`abs:"UAV" AND abs:"multimodal large language model"`** — 命中 10，其中新增 6

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-08-14 | `2608.13980v1` | FIRM: Fine-Grained Intra-Token Representation of Masks for Remote Sensing Reasoning Segmentation |
| 2026-07-22 | `2607.19857v1` | Memory-Augmented Multimodal Large Language Models for Small Object Understanding in Streaming Aerial Videos |
| 2026-07-19 | `2607.17386v1` | SkyVLaM: Multimodal Large Language Model for UAV Video Understanding in Remote Sensing |
| 2026-07-14 | `2607.12477v2` | Self in Space: Benchmarking Self-Awareness and Spatial Cognition in UAV Embodied Intelligence |
| 2026-07-01 | `2607.00416v1` | DroneIQA-VLE: Multi-Task Drone Image Quality Assessment via Vision-Language Ensemble |
| 2026-06-26 | `2606.28049v1` | AirGroundBench: Probing Spatial Intelligence in Multimodal Large Models under Heterogeneous Multi-View Embodie |

**`abs:"aerial" AND abs:"visual question answering"`** — 命中 3，其中新增 2

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-09-16 | `2609.18210v2` | Understanding Dynamic Scenes at Gigapixel Scale: Wide-Area Spatio-Temporal Perception from UAVs |
| 2026-06-17 | `2606.19277v1` | A Unified Framework for Efficient Remote Sensing Visual Question Answering: Adapting Dual, Hybrid, and Encoder |

---

*本文件由 `tools/watch.py` 生成，重跑即可刷新。收录进正文前，ID 与标题一律以 arXiv API 为准。*
