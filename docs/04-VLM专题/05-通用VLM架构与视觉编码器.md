# 通用 VLM 架构与视觉编码器

> **预计阅读：45 分钟 | 前置知识：Transformer、对比学习基础、[什么是VLM](../01-基础概念/02-什么是VLM.md)**

前面四篇讲的都是无人机与遥感侧的 VLM：用什么数据、评测什么能力、怎么部署到机载。但那些模型本身的零件——图像怎么变成向量、向量怎么进语言模型、一张图要花多少 token——在这一卷里一直缺席。这篇补的就是这条主干。

---

## 1. 四个部件

一个通用 VLM 拆开就是这个四个部件。不同工作名字取得再花，落点都在这四处之一：

| 部件 | 干什么 | 换它会改变什么 |
|---|---|---|
| 视觉编码器 | 图像 → 一串视觉特征 | 「能看清什么」：分辨率、细粒度、密集预测能力 |
| 连接器 | 视觉特征 → 语言模型能吃的序列 | 「看的东西以什么格式进去」：长度、压缩方式、对齐方式 |
| 语言模型 | 序列 → 文本 | 「能说多复杂的话」：推理、知识、指令跟随 |
| 训练配方 | 三个部件怎么连起来训、数据是什么 | 「实际分数」：同样的架构换个配方能差出一大截 |

这个拆法的直接来源是 LLaVA 那一支的极简做法：**一个线性投影层就把视觉特征接进了语言模型**，且只训投影层和语言模型、冻结视觉塔（`2304.08485`）。在它之前，接法要复杂得多——Flamingo 用 Perceiver Resampler 压成固定长度的潜查询再加门控交叉注意力，BLIP-2 用 Q-Former 分两阶段桥接（`2204.14198`、`2301.12597`）。LLaVA 证明了**在指令数据足够的情况下，最笨的连接器也能work**，于是复杂度从连接器转移到了数据和训练配方上。

> **判据**：判断一篇 VLM 论文改的是哪个部件，就看它在消融表里动的是哪一列。如果一篇工作的主张集中在「我们的连接器设计更优」，先去看它在多大分辨率、多少视觉 token 下比的——连接器带来的差距经常被分辨率的差距盖过去，见第 4 节。

---

## 2. 编码器演进：从固定分辨率到原生分辨率

### 2.1 起点：CLIP

`2103.00020` 用 4 亿图文对做对比预训练，得到的是一个**零样本可迁移**的图像编码器。它定义了后面三年 VLM 的默认视觉塔：ViT-L/14，输入固定边长（如 224 或 336），输出 1 + N 个 patch token。

固定分辨率这一步埋下了后面所有麻烦。遥感图像里一栋楼可能只占 20 个像素；自然图像预训练时模型从没见过这种尺度分布，而固定分辨率意味着**你不能靠「输入更大图」来补偿**——放大图会同时放大所有东西，patch 数是固定的。

### 2.2 三条并行的改进线

**规模与训练技巧。** `2303.15389`（EVA-CLIP）在 CLIP 的框架上做大规模蒸馏与掩码图像建模，把视觉塔的容量做大。`2304.14108`（DataComp）走的是数据侧：固定模型和算力，比谁的数据筛选策略更好——它的结论是**数据分布本身就能拉开几个点**，不是模型结构。

**自监督、不要文本。** `2304.07193`（DINOv2）完全不用文本监督，得到的是密集特征更好的编码器。这一点在 VLM 里很实用：CLIP 类编码器擅长「这张图整体像什么」，DINOv2 擅长「这个 patch 属于什么」。Cambrian-1 正是据此把多个编码器拼在一起用（`2406.16860`）。

**打破固定分辨率。** `2307.06304`（NaViT）提出 patch n' pack：把不同长宽比的图切成 patch 后打包进同一个序列训练，让 ViT 天然接受任意分辨率。这条路被 Qwen2-VL 走到了产品里：`2409.12191` 用原生动态分辨率，把图按需要切成若干块、每块编成 token，再配 M-RoPE 让位置编码在多模态序列上自洽。Qwen2.5-VL（`2502.13923`）延续并加强了这个设计（文档、视频、坐标定位）。

> **判据**：一篇论文说「支持任意分辨率」时，要问的是**代价在谁身上**。NaViT 的代价在训练（打包与掩码逻辑）；Qwen2-VL 的代价在推理（token 数随图像面积线性增长，见第 5 节）。没有免费的任意分辨率。

---

## 3. 对比对齐预训练：双塔与 sigmoid 损失

视觉塔单独训完（或被冻结）之后，还需要一次「图文对齐」把图像空间和文本空间拉到一起。这一节讲清两件事：损失函数的形式，和它对 batch 的依赖。

### 3.1 softmax 对比损失与它的 batch 依赖

CLIP 的损失是 batch 内的对称 softmax：对一个 batch 里的 N 个图文对，图像 i 与文本 j 的相似度做温度缩放的 softmax，正例在对角线上。**所有非对角元素都是这个样本的负例**，所以 batch size 直接等于判别难度。这也解释了一个常见现象：小规模复现 CLIP 时分数总差一截，不一定是模型不行，可能是 batch 太小——负例太少，任务太容易。

### 3.2 sigmoid 损失：把这个依赖拆开

`2303.15343`（SigLIP）把 softmax 换成逐对的 sigmoid：每一对图文独立判「配 / 不配」，不要求全局归一化。好处很直接：**不需要跨设备收集全部负例**，小 batch 也能训，且性能不比 softmax 差。这一点在算力受限的场景里价值很大，也是 SigLIP 系编码器在这两年被大量 VLM 直接当作视觉塔的原因。

`2502.14786`（SigLIP 2）在它上面加了三件事：多语言训练（面向非英语 VLM）、定位/密集预测能力、以及除对比之外的描述性预训练目标。对无人机场景的意义是多语言之外的两条——**定位与密集特征**。

### 3.3 两条省算力的旁路

- `2111.07991`（LiT）：**锁住图像塔，只训文本塔**。图像塔学到的表征不变，只把文本侧对齐到它上面。省算力，也避免了在小数据上把图像塔训坏。
- `2212.00794`（FLIP）：训练时随机 mask 掉大部分 patch，用剩下的算对比损失。速度换精度，比例可调。

> **判据**：选编码器时要看它是在**哪一步**被训出来的。同为「SigLIP 系」，只在图文对上对齐过的和额外做过密集定位训练的，在下游定位任务上的表现差别会比它们的零样本分类分数差别大得多。零样本 ImageNet 分数不能作为选视觉塔的唯一依据。

---

## 4. 连接器：投影式与查询式

连接器要做的事只有一件：把视觉特征变成语言模型能吃的序列，并决定这个序列有多长。做法分两派。

### 4.1 投影式（可选压缩）

- **线性 / MLP 投影**：`2304.08485` 直接用一个（后续版本改成两层 MLP）投影把每个 patch token 映射到语言模型维度。长度不变，一个 patch 一个 token。
- **带压缩的投影器**：`2312.06742`（Honeybee）用**局部性增强的投影器**（卷积式 / 可变形卷积式的 abstractor）在投影的同时压缩 token 数；`2407.02392`（TokenPacker）用粗到细的点到区域方式压缩视觉 token，论文自报的压缩幅度是 **75%–89%**（即保留约 1/4 到 1/9）。

### 4.2 查询式

- **Q-Former**：`2301.12597`（BLIP-2）用一组可学习的查询向量去「问」视觉特征，输出**固定长度**的表示。可变长度的图像于是有了恒定长度的接口——代价是查询数是个超参，图像细节多的时候会被瓶颈卡住。
- **Perceiver Resampler + 门控交叉注意力**：`2204.14198`（Flamingo）把重采样器和冻结的语言塔用门控交叉注意力接起来，是多模态 few-shot 的代表做法。

### 4.3 一个容易被忽略的点：模态不匹配

`2405.20797`（Ovis）指出的问题不在「压不压缩」，而在**视觉嵌入和文本嵌入的结构本身不一致**——直接投影过去，视觉 token 落在文本嵌入空间里一个「不该有东西」的位置。它的做法是给视觉 token 一个对齐用的嵌入表。这个视角解释了为什么有些模型架构没变、只换了对齐方式，分数就动了。

### 4.4 连接器到底重不重要：两个消融结论不完全一致

- `2403.09611`（MM1）的消融结论是：**视觉编码器和图像分辨率的影响大于连接器的类型**；连接器怎么设计，差距有限。
- `2405.02246`（Idefics2 那篇"什么在构建 VLM 时重要"）的结论里有两条与之互补：**全自回归的架构优于交叉注意力式**，且训练配方与数据的影响很大。

两个结论并不矛盾：它们比的是不同的轴。但连起来读有一个实用推论——**先把分辨率和数据搞定，再考虑连接器**。反过来做（精调连接器、分辨率不动）通常是在一个已经被卡住的上限里抠分。

> **判据**：看到「我们的连接器把 token 压到 1/N，性能几乎不掉」，先确认这个「几乎不掉」是在哪个任务上量的。如果量的是图像描述类任务，见第 5 节——那一类对 token 预算本来就不敏感。

---

## 5. 视觉 token 预算与分辨率

这是本篇最实用的一节，也是第 9 节动手验证的主题。

### 5.1 两个预算不是一回事

- **分辨率**决定「格子有多细」：同样一张图，切得越细，每个 token 覆盖的实际面积越小，小目标才有可能被单独看见。
- **token 数**决定「要付多少算力」：语言模型侧的注意力开销大致随序列长度二次增长，视觉 token 又和文本 token 抢同一个上下文窗口。

固定分辨率 + 全量 patch 的做法下，这两个预算是绑死的：想看清就得加 token。要解开这个绑定，只有两条路——**压缩**（token 变少但分辨率不变）或**变分辨率**（Qwen2-VL 那样按需切块）。

### 5.2 压缩路线的几种做法

| 方法 | 思路 | 留下什么 |
|---|---|---|
| `2210.09461` ToMe | 在 ViT 内部把相似 token 合并 | 冗余 patch 被合掉，显著区域保留 |
| `2403.15388` LLaVA-PruMerge | 按重要性剪枝并合并 | 与语言模型相关的区域优先保留 |
| `2405.17430` Matryoshka MM | 训练成嵌套表示，一个模型多种预算 | 推理时按需截断，不用重训 |
| `2402.03766` MobileVLM V2 | 轻量投影器 + 小语言模型 | 端侧可跑的整体方案 |

### 5.3 一个必须先问清的问题：压掉的是什么能力

token 预算是被压的对象，但**不同能力对它的敏感度差得很远**。这个差别在第 9 节被量化：把 token 预算从 576 压到 4，描述类任务（计数、主导方向）可以完全不掉，而细粒度定位的误差线性涨了 12 倍。

原因不神秘：计数是**对格子占用做整数求和**，切成几格不改变求和结果；定位要求**读出坐标**，而格子边长就是定位精度的下限。同样是「视觉 token」，被两类任务使用的部分不是同一部分。

> **判据**：报告「压到 1/N token，性能保持 95%」时，必须同时报告任务类型构成。以计数、粗分类、整体描述为主的评测集，会系统性地把 token 压缩的代价藏起来。这条判据与第 7 节「对无人机的含义」直接相关——无人机的下游任务恰好以定位为主。

---

## 6. 与遥感、无人机侧的接口

前面五节讲的是通用侧的主干。到遥感与无人机这里，接口上有几点是**必须改**的，不是调参能解决的：

1. **分辨率优先。** 俯视视角下目标像素数天然更少，同一类目标在 10m 和 500m 高度下的像素尺寸差可以到两个数量级。固定分辨率的编码器在这个分布上必然吃紧，这也是遥感侧大量工作自己做编码器（RemoteCLIP 一类）而不是直接用 CLIP 的原因。
2. **密集任务要密集特征。** 遥感的检测、分割、变化检测都要求 patch 级输出，而 CLIP 类编码器是在「整图」粒度上对齐的。DINOv2 一类自监督编码器与混合编码器（Cambrian-1 的做法）在这类任务上更合适。
3. **机载算力是硬约束。** token 预算在机载上不只是速度问题，还牵涉显存与热设计。这一条见 [04-边缘VLM部署](./04-边缘VLM部署.md)。
4. **坐标系要能从 token 里读出来。** 无人机下游常需要「目标在图上的位置」，这就要求 token 保留坐标信息——正好是 token 压缩最先牺牲的那部分（第 5.3 节）。

---

## 7. 对无人机的含义

**这一节把上面的通用结论翻译成无人机侧的取舍。**

**第一，选型时分辨率和编码器的重要性高于连接器，这条结论在无人机侧被放大。** MM1 的消融是在自然图像分辨率下做的，而无人机的目标更小、尺度变化更大，分辨率这一项的边际收益比自然图像场景更高。用一个小分辨率编码器 + 复杂连接器，在无人机上通常不如反过来。

**第二，token 压缩要先在定位类任务上验证。** 机载部署时压缩 token 是省算力最直接的手段，但第 9 节的实测说明：压缩对描述类任务是几乎免费的、对定位类任务是**线性收费**的。无人机任务里「找目标 / 报位置 / 跟随」都是定位类。所以压缩方案的验收必须在定位评测上做，不能只看描述类基准。

**第三，遥感/遥测数据上的对齐方式值得单独考虑。** Ovis 指出的模态不匹配问题在俯视图像上是否更严重（纹理与文本描述的相关性更弱），是一个没有定论的开放问题——这属于可以在小规模上自己量的事，不必等通用侧的结论。

**第四，原生分辨率带来的是「可变成本」。** Qwen2-VL 这一支的做法让 token 数随输入面积线性增长，这在机载上意味着**飞行高度变化会直接改变推理耗时**——同一模型在低空（小视野大目标）与高空（大视野小目标）的时延不是同一个数。做实时性预算时必须按最坏情况算，不能按单张图的平均值算。

---

## 8. 关键论文

> 本节按四个部件分组，收录通用 VLM 方法主干上的代表工作；每条 arXiv 号已通过 arXiv API 核对标题。无人机与遥感侧的应用工作见 [01](./01-遥感VLM.md)、[02](./02-无人机场景理解.md)、[03](./03-LLM驱动的无人机Agent.md)、[04](./04-边缘VLM部署.md)。

#### 视觉编码器与对齐预训练

- **[arXiv'21.02] CLIP** — *Learning Transferable Visual Models From Natural Language Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg)](https://arxiv.org/abs/2103.00020)
  [![GitHub](https://img.shields.io/badge/GitHub-CLIP-181717.svg?logo=github)](https://github.com/openai/CLIP)
  4 亿图文对对比预训练，定义了后续三年 VLM 的默认视觉塔。

- **[arXiv'21.11] LiT** — *LiT: Zero-Shot Transfer with Locked-image text Tuning*
  [![arXiv](https://img.shields.io/badge/arXiv-2111.07991-b31b1b.svg)](https://arxiv.org/abs/2111.07991)
  锁图像塔、只训文本塔，把「对齐」从「联合训练」里拆出来。

- **[arXiv'22.04] Flamingo** — *Flamingo: a Visual Language Model for Few-Shot Learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2204.14198-b31b1b.svg)](https://arxiv.org/abs/2204.14198)
  Perceiver Resampler + 门控交叉注意力，冻结语言塔的多模态 few-shot 代表。

- **[arXiv'22.12] FLIP** — *Scaling Language-Image Pre-training via Masking*
  [![arXiv](https://img.shields.io/badge/arXiv-2212.00794-b31b1b.svg)](https://arxiv.org/abs/2212.00794)
  训练时掩掉大部分 patch，用速度换精度。

- **[arXiv'23.01] BLIP-2** — *Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2301.12597-b31b1b.svg)](https://arxiv.org/abs/2301.12597)
  [![GitHub](https://img.shields.io/badge/GitHub-LAVIS-181717.svg?logo=github)](https://github.com/salesforce/LAVIS)
  Q-Former 把可变长度的视觉特征压成固定长度查询。

- **[arXiv'23.03] EVA-CLIP** — *EVA-CLIP: Improved Training Techniques for CLIP at Scale*
  [![arXiv](https://img.shields.io/badge/arXiv-2303.15389-b31b1b.svg)](https://arxiv.org/abs/2303.15389)
  大规模蒸馏与掩码建模把视觉塔容量做大。

- **[arXiv'23.03] SigLIP** — *Sigmoid Loss for Language Image Pre-Training*
  [![arXiv](https://img.shields.io/badge/arXiv-2303.15343-b31b1b.svg)](https://arxiv.org/abs/2303.15343)
  [![GitHub](https://img.shields.io/badge/GitHub-big__vision-181717.svg?logo=github)](https://github.com/google-research/big_vision)
  逐对 sigmoid 损失解开了对比学习对全局 batch 的依赖。

- **[arXiv'23.04] DataComp** — *DataComp: In search of the next generation of multimodal datasets*
  [![arXiv](https://img.shields.io/badge/arXiv-2304.14108-b31b1b.svg)](https://arxiv.org/abs/2304.14108)
  固定模型与算力、只比数据筛选策略，证明数据分布能拉开几个点。

- **[arXiv'23.04] DINOv2** — *DINOv2: Learning Robust Visual Features without Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg)](https://arxiv.org/abs/2304.07193)
  [![GitHub](https://img.shields.io/badge/GitHub-DINOv2-181717.svg?logo=github)](https://github.com/facebookresearch/dinov2)
  不用文本监督，密集特征质量更好，适合检测分割类下游。

- **[arXiv'23.07] NaViT** — *Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution*
  [![arXiv](https://img.shields.io/badge/arXiv-2307.06304-b31b1b.svg)](https://arxiv.org/abs/2307.06304)
  任意长宽比与分辨率的 ViT 训练方案，原生分辨率的先声。

- **[arXiv'23.12] Honeybee** — *Honeybee: Locality-enhanced Projector for Multimodal LLM*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.06742-b31b1b.svg)](https://arxiv.org/abs/2312.06742)
  投影器加卷积式的局部性增强，证明连接器设计仍有空间。

- **[arXiv'24.09] Qwen2-VL** — *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*
  [![arXiv](https://img.shields.io/badge/arXiv-2409.12191-b31b1b.svg)](https://arxiv.org/abs/2409.12191)
  原生动态分辨率 + M-RoPE，把任意分辨率做到可用。

- **[arXiv'25.02] SigLIP 2** — *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*
  [![arXiv](https://img.shields.io/badge/arXiv-2502.14786-b31b1b.svg)](https://arxiv.org/abs/2502.14786)
  在 sigmoid 损失之上补多语言、定位与密集特征。

#### 连接器与视觉 token 压缩

- **[arXiv'22.10] ToMe** — *Token Merging: Your ViT But Faster*
  [![arXiv](https://img.shields.io/badge/arXiv-2210.09461-b31b1b.svg)](https://arxiv.org/abs/2210.09461)
  在 ViT 内部合并相似 token，无需重训即可加速。

- **[arXiv'24.02] MobileVLM V2** — *MobileVLM V2: Faster and Stronger Baseline for Vision Language Model*
  [![arXiv](https://img.shields.io/badge/arXiv-2402.03766-b31b1b.svg)](https://arxiv.org/abs/2402.03766)
  轻量投影器加小语言模型的端侧整体方案。

- **[arXiv'24.03] LLaVA-PruMerge** — *LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.15388-b31b1b.svg)](https://arxiv.org/abs/2403.15388)
  按重要性剪枝并合并视觉 token。

- **[arXiv'24.05] Matryoshka MM** — *Matryoshka Multimodal Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2405.17430-b31b1b.svg)](https://arxiv.org/abs/2405.17430)
  嵌套视觉表示，一个模型支持多种 token 预算。

- **[arXiv'24.05] Ovis** — *Ovis: Structural Embedding Alignment for Multimodal Large Language Model*
  [![arXiv](https://img.shields.io/badge/arXiv-2405.20797-b31b1b.svg)](https://arxiv.org/abs/2405.20797)
  指出投影式接法的模态结构不匹配，用视觉嵌入表对齐。

- **[arXiv'24.07] TokenPacker** — *TokenPacker: Efficient Visual Projector for Multimodal LLM*
  [![arXiv](https://img.shields.io/badge/arXiv-2407.02392-b31b1b.svg)](https://arxiv.org/abs/2407.02392)
  粗到细的点到区域压缩，论文自报压缩视觉 token **75%–89%**（保留约 1/4 至 1/9）。

#### 训练配方与开源模型

- **[arXiv'24.02] Prismatic VLMs** — *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2402.07865-b31b1b.svg)](https://arxiv.org/abs/2402.07865)
  对视觉条件语言模型设计空间的系统消融，指出数据与配方的权重大于架构细节。

- **[arXiv'24.03] MM1** — *MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.09611-b31b1b.svg)](https://arxiv.org/abs/2403.09611)
  消融结论：编码器与分辨率比连接器类型更重要。

- **[arXiv'24.05] Idefics2** — *What matters when building vision-language models?*
  [![arXiv](https://img.shields.io/badge/arXiv-2405.02246-b31b1b.svg)](https://arxiv.org/abs/2405.02246)
  全自回归架构优于交叉注意力式，训练配方与数据影响很大。

- **[arXiv'24.06] Cambrian-1** — *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs*
  [![arXiv](https://img.shields.io/badge/arXiv-2406.16860-b31b1b.svg)](https://arxiv.org/abs/2406.16860)
  [![GitHub](https://img.shields.io/badge/GitHub-Cambrian-181717.svg?logo=github)](https://github.com/cambrian-mllm/cambrian)
  多编码器混合，把「视觉侧」当作独立变量系统研究。

- **[arXiv'24.08] LLaVA-OneVision** — *LLaVA-OneVision: Easy Visual Task Transfer*
  [![arXiv](https://img.shields.io/badge/arXiv-2408.03326-b31b1b.svg)](https://arxiv.org/abs/2408.03326)
  单图、多图、视频用同一套配方统一处理。

- **[arXiv'24.12] InternVL 2.5** — *Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.05271-b31b1b.svg)](https://arxiv.org/abs/2412.05271)
  [![GitHub](https://img.shields.io/badge/GitHub-InternVL-181717.svg?logo=github)](https://github.com/OpenGVLab/InternVL)
  模型规模、数据规模与测试时扩展三条轴同时推。

- **[arXiv'25.02] Qwen2.5-VL** — *Qwen2.5-VL Technical Report*
  [![arXiv](https://img.shields.io/badge/arXiv-2502.13923-b31b1b.svg)](https://arxiv.org/abs/2502.13923)
  [![GitHub](https://img.shields.io/badge/GitHub-Qwen__VL-181717.svg?logo=github)](https://github.com/QwenLM/Qwen-VL)
  原生分辨率路线的当前代表（仓库已并入 Qwen3-VL）。

---

## 9. 动手验证：视觉 token 预算——描述不怕压，定位怕

第 5.3 节那条判据是本篇最容易被误用的一条，这一节把它跑成数字。

### 9.0 装置：把编码器建模成均匀网格

真实 VLM 的视觉 token 是学出来的特征，混着分辨率、感受野、注意力等多重因素，很难单独归因。这里做一个**只保留分辨率这一个变量**的最小装置：

- 把单位正方形划成 **G × G 的均匀网格**，`G = round(sqrt(B))`，`B` 是可用的视觉 token 数；
- 每个 token 只携带**它所在格子的中心坐标**——这是「有分辨率、没有别的」的极限情形；
- 场景里放 64 个目标，共 400 个场景；坐标随机均匀分布。

于是格子边长 `s = 1/G` 就是全部假设。**不训练任何模型**：下面每条实测结论都有对应的闭式解，两者的差就是「随机采样噪声」本身。

任务分三类，代表下游对 token 的三种用法：**整数统计**（计数、主导象限）、**读坐标**（判断左右、报位置）、**容差命中**（位置误差落在阈值内算对）。

### 9.1 核心代码

```python
import numpy as np

SEED = 13
BUDGETS = [576, 256, 64, 36, 16, 4]        # 视觉 token 预算
N_SCENE, N_OBJ = 400, 64                    # 400 个场景，每场景 64 个目标
TAU = 0.02                                  # 位置容差
E_R_CONST = (np.sqrt(2) + np.log(1 + np.sqrt(2))) / 6      # 单位格平均距离系数
E_R2_CONST = 1.0 / np.sqrt(6)                              # 单位格 RMS 系数


def grid_of(budget):
    """B 个 token -> G x G 均匀网格，G = round(sqrt(B))。"""
    return max(2, int(round(np.sqrt(budget))))


def encode(pts, G):
    """编码：每个点只记它落在哪一格的格心坐标。"""
    s = 1.0 / G
    idx = np.clip((pts / s).astype(int), 0, G - 1)
    return (idx + 0.5) * s


rng = np.random.default_rng(SEED)
for B in BUDGETS:
    G = grid_of(B)
    err_pt, err_cen, hit = [], [], []
    for _ in range(N_SCENE):
        pts = rng.random((N_OBJ, 2))
        rec = encode(pts, G)                       # 格心坐标
        err_pt.append(np.linalg.norm(rec - pts, axis=1))    # 逐目标定位误差
        err_cen.append(np.linalg.norm(rec.mean(0) - pts.mean(0)))   # 质心误差
        hit.append((np.linalg.norm(rec - pts, axis=1) <= TAU).mean())
```

编码只有一行实质内容：`(idx + 0.5) * s`——**把真实坐标替换成格心坐标**。后面所有结论都是这一行替换的后果。完整脚本里还多了计数、主导象限、同列率、左右判断、token 分配这几组统计，写法都是同一套。完整脚本见 [9.3](#93-运行完整脚本)。

### 9.2 本地实测结果

**① 整数统计精确不变；聚合统计按 √N 缩水**

```text
  token B    G        计数正确率          主导象限准确率        质心误差          逐目标平均误差       倍数
      576   24       1.0000           1.0000    1.90e-03           0.0159      8.4
      256   16       1.0000           1.0000    2.85e-03           0.0239      8.4
       64    8       1.0000           1.0000    5.74e-03           0.0477      8.3
       36    6       1.0000           1.0000    7.39e-03           0.0638      8.6
       16    4       1.0000           1.0000    1.08e-02           0.0954       8.8
        4    2       1.0000           1.0000    2.39e-02           0.1908      8.0
```

计数与主导象限在**所有档位**都是 1.0000，包括 B=4（也就是整张图只剩 4 个格子）的那一档。原因是这两类任务都是**对格子占用数做整数求和**：目标落在哪个格子不重要，重要的是「这个格子里有没有目标」——把格子变大不会改变任何格子的占用与否，只会把多个格子合并成一个非空格。

质心误差（所有目标点的平均位置）随预算上涨，但始终比逐目标误差低约 √64 = 8 倍（实测 8.0–8.8）。原因是逐目标量化误差是**零均值**的，64 个一平均就按 √64 缩下去。这就是「聚合」的防御力：任何先平均再用的统计量都自带这一层保护。

**② 读坐标的判读：判断左右只需要一个坐标列**

```text
  token B        同列率(实测)       1/G          左右判断准确率     闭式 1-0.5/G
      576         0.0420    0.0417           0.9799         0.9792
      256         0.0651    0.0625           0.9673         0.9688
       64         0.1256    0.1250           0.9373         0.9375
       36         0.1655    0.1667           0.9169         0.9167
       16         0.2510    0.2500           0.8754         0.8750
        4         0.4999    0.5000           0.7512         0.7500
```

这里有一个容易搞错的地方，值得单独讲。判断「A 在 B 左边」只用到 x 这一个坐标，所以两个点被量化到**同一列**时判断必然出错。而同列的概率是 `1/G`，**不是 `1/B`**：只用掉一个坐标，退化速度就慢一半——因为另一维的量化误差在这道题里根本不参与。

实测同列率与 `1/G` 贴得很紧，左右判断准确率与闭式 `1 - 0.5/G` 也贴得很紧。

**③ 单体定位：误差就是量化误差**

```text
  token B      边长 s       平均误差     闭式 0.3826s       RMS    闭式 s/sqrt6
      576    0.0417     0.0159         0.0159    0.0169        0.0170
      256    0.0625     0.0239         0.0239    0.0255        0.0255
       64    0.1250     0.0477         0.0478    0.0509        0.0510
       36    0.1667     0.0638         0.0638    0.0680        0.0680
       16    0.2500     0.0954         0.0956    0.1018        0.1021
        4    0.5000     0.1908         0.1913        0.2035        0.2041
```

单位正方形里一个随机点到它所在格心的平均距离有闭式解：

```text
E[R] = (s/6)·(sqrt(2) + ln(1 + sqrt(2))) = 0.382598·s        RMS[R] = s/sqrt(6) = 0.408248·s
```

实测贴着闭式，说明在这个装置里「定位误差」就是「量化误差」本身，没有别的成分。

预算从 576 压到 4（B 除以 144），平均误差涨 12.0 倍，正好是 G 之比；同一区间里左右判断的失手率也涨 12 倍。两条曲线在 log-log 下**斜率都是 -0.500**，差的是常数（0.3826 对 0.5，都由格子几何决定）。所以「还能说谁在左边」和「还能指出在哪」不是两种能力，是同一条直线上的两个点。

**④ 容差：命中率与 token 预算成正比**

```text
  token B      边长 s        命中率(实测)    闭式 pi*tau^2/s^2        实测增长
      576    0.0417         0.7295             0.7238           -
      256    0.0625         0.3211             0.3217       2.27x
       64    0.1250         0.0808             0.0804       3.97x
       36    0.1667         0.0446             0.0452       1.81x
       16    0.2500         0.0202             0.0201       2.21x
        4    0.5000         0.0052             0.0050       3.86x
```

误差落在容差 τ 内的概率近似为 `πτ²/s²`（在 τ ≤ s/2 时成立），而 `s² = 1/B`——**命中率与 token 预算成正比**。B 从 64 到 576 是 9 倍，命中率从 0.0808 涨到 0.7295，正好 9.0 倍。

这条结论的实践含义是：**容差一松，所有预算看起来都能用**。把验收容差设成整张图的 2%，B=576 有 73% 命中、B=64 只有 8%；如果把容差放宽到十分之一张图，B=4 的模型也能"过关"。那不是模型变好了，是任务本身不需要细粒度。

**⑤ 分配方向：token 是有方向的取舍**

最后把预算固定成 B=64，比较两种分配：均匀 8×8（每象限 4×4），与把 token 集中给目标象限 6×6、其余三区各 3×3（合计 63 个）。

```text
                      格边长 s         平均定位误差        闭式        误差倍数
           均匀-全图     0.1250         0.0475    0.0478       1.000
          集中-目标区     0.0833         0.0321    0.0319       0.674
         集中-其余三区     0.1667         0.0637    0.0638       1.340
         集中-整体平均          -         0.0558         -       1.174
```

目标区好 1.48 倍，代价是其余三区各差 1.34 倍；按面积加权，**整体平均反而差 1.17 倍**。也就是说「把 token 花在感兴趣区域」这个直觉，在整体指标上是亏的——它的价值必须由任务本身提供（如果下游只关心目标区，那就是赚的）。

### 9.3 运行完整脚本

```bash
py -3.9 code/w_token_budget.py
```

约 3 秒，纯 numpy；输出 `figures/w_token_budget.png`（三栏：预算-误差双对数曲线、同列率与闭式对比、token 分配对比）。

> **限制**：本实验把视觉编码器建模成**均匀网格 + 格心坐标**，只保留分辨率这一个变量，量的是「格子边长 s 与三类处理方式的关系」，不是某个真实 VLM 的分数。已知不能外推的部分：(a) 真实 token 不是格心而是学出来的特征，相邻格有重叠感受野，定位误差会小于这里的量化误差；(b) 真实模型的描述类任务也会随预算退化（注意力被稀释、语言先验接管），这里让它们精确不变，是**上界**；(c) 真实基准的容差由任务定义，不是这里设的 0.02。因此第 9.2 节的 ①②③④ 是「量化误差的下限行为」，⑤ 是「分配取舍的方向」，都不能直接当作某个模型在某个基准上的分数变化量。

---

## 10. 延伸阅读

- [01-遥感VLM](./01-遥感VLM.md) — 遥感侧为什么自己做编码器
- [02-无人机场景理解](./02-无人机场景理解.md) — 评测这些能力的基准长什么样
- [03-LLM驱动的无人机Agent](./03-LLM驱动的无人机Agent.md) — 语言模型一侧的用法
- [04-边缘VLM部署](./04-边缘VLM部署.md) — 把 token 预算换成机载显存与时延
- [06-VLM指令微调与对齐](./06-VLM指令微调与对齐.md) — 本篇讲「怎么接」，那篇讲「怎么训」
- [07-通用VLM评测与幻觉](./07-通用VLM评测与幻觉.md) — 本篇的结论怎么在评测里被验证或掩盖
- [什么是VLM](../01-基础概念/02-什么是VLM.md) — 概念入口
- [三者关系与区别](../01-基础概念/04-三者关系与区别.md) — VLM 在世界模型 / VLA 之间的位置

---

## 11. 思考题

### 题目 1：为什么 sigmoid 损失能让小 batch 训练变得可行？这对算力受限的复现意味着什么？

<details>
<summary>查看参考答案</summary>

CLIP 用的对比损失是 batch 内对称 softmax，分母要对整个 batch 的相似度矩阵做归一化。这意味着**每个样本的负例就是同 batch 的其他样本**，而梯度的正确性依赖「全体负例都参与归一化」。实现上必须跨设备 all-gather 相似度矩阵，batch 越小，负例越少、任务越容易、学到的表征判别性越差。

SigLIP 把损失换成逐对的 sigmoid：每一对图文独立判「配 / 不配」，正负样本对之间没有归一化耦合。于是：

1. 不需要跨设备收集全部负例，单卡小 batch 也能训；
2. 负例的构造从「batch 内免费的」变成「显式可控的」——可以把构造难负例的策略独立设计；
3. 损失对 batch size 的敏感度显著下降。

对算力受限的复现，含义是：**复现 CLIP 的分数需要复现它的 batch 规模**，而复现 SigLIP 的分数主要需要复现它的数据与训练步数。选型做小规模实验时，后者是更可控的起点。

</details>

### 题目 2：MM1 说连接器类型影响不大，Idefics2 说全自回归优于交叉注意力。这两条怎么同时成立？

<details>
<summary>查看参考答案</summary>

两条比的是不同的轴，且结论的作用域不同。

- MM1 比的是**同为投影式的前提下**的连接器变体（线性、MLP、各种压缩器），在这个范围内差异有限；它真正拉开差距的是视觉编码器和图像分辨率。
- Idefics2 比的是**架构范式的选择**——把视觉 token 直接拼进语言模型序列做全自回归，还是在语言模型内部用交叉注意力去读视觉特征。这是两种不同的信息通路，量级上确实会差。

同时成立的前提是：**「连接器」这个词在两篇里指的不是同一个东西**。MM1 的「连接器」是投影器的实现细节，Idefics2 的「架构」是视觉与语言如何交互。把两条结论合起来的实用推论是：

1. 先定交互范式（拼进序列还是交叉注意力）；
2. 再定分辨率和编码器；
3. 最后才调投影器的实现。

反过来做（在投影器上精调、分辨率和交互范式不动）是在一个已被卡住的上限里抠分。

</details>

### 题目 3：第 9 节里计数任务的正确率在所有预算下都是 1.0000。真实 VLM 的计数能力会这么稳吗？

<details>
<summary>查看参考答案</summary>

不会。这个 1.0000 是这个装置的**结构性结论**，不是对真实模型的预测，需要分清它是从哪来的：

在本装置里，计数 = 「数有多少个格子被占用」。编码是 `(idx + 0.5) * s`，格子边长变大只会把多个格子合并，不会把一个有目标的格子变成空的、也不会无中生有。所以对格子占用求和的结果**精确不变**——这是数学上的不变性，不是测出来的巧合。

真实 VLM 的计数会退化，原因都在本装置之外：

1. **编码不是格心替换。** 真实 token 是学出来的特征，粗粒度特征会把相邻目标糊在一起，两个目标可能只贡献一个「有东西」的证据。
2. **计数是在语言模型侧完成的。** token 数变少并不会让计数变容易——恰恰相反，可数的证据变少了。本装置里「计数」是直接读编码结果，跳过了语言模型。
3. **描述类任务也会受注意力稀释影响。** 这一点已写进 9.3 的「限制」(b)：本装置让描述类任务精确不变，是**上界**。

正确的读法是：第 9 节证明了「**即使**计数所用的信息在编码里被完整保留，定位所用的信息也已经按格子边长线性退化了」——它给出的是描述类任务不退化的**最强可能情形**下，定位仍然在退化。这比「两者都会退化」更强地支持了第 5.3 节的判据。

</details>

### 题目 4：如果要在机载部署里压 token 预算，验收该怎么设计？

<details>
<summary>查看参考答案</summary>

要点是「压掉的代价分布不均匀」，所以验收必须按**代价最大的那类任务**设计，而不是按平均值。

1. **按任务类型分层验收，不报总平均。** 至少分三层：整数统计类（计数、分类）、坐标读取类（判断方位、报位置、跟随）、容差命中类（定位到指定阈值内）。第 9 节说明这三类的退化速度差一个量级，总平均会把定位的损失摊薄掉。
2. **容差要在设计任务时就定死，不能事后放宽。** 第 9.2 节④说明命中率与预算成正比：容差一松，压掉的代价立刻看不出来。先定「误差超过多少就算错」，再跑压缩方案。
3. **检查压缩方案保留的是哪部分 token。** 压缩方法（ToMe、PruMerge 一类）按相似度或重要性合并 token，而相似度不等于几何相邻性——需要确认压缩后坐标信息还能读出来。第 9.2 节②提示：判断方位只需要一个坐标列，所以一定要**分别**测水平与垂直方向，别只测二维距离。
4. **把「按区域分配 token」的收益算清楚。** 第 9.2 节⑤说明集中分配对目标区有 1.48 倍的收益、对其余区域有 1.34 倍的代价。只有当任务确实只关心目标区时，这笔账才划算。
5. **时延按最坏情况算。** 若采用原生分辨率路线，token 数随输入面积变化，低空/高空不是同一个时延（见第 7 节第四条）。

</details>

---

> **下一节**：[06-VLM指令微调与对齐](./06-VLM指令微调与对齐.md) — 架构定了之后，从预训练权重到听得懂指令之间还差一段训练
