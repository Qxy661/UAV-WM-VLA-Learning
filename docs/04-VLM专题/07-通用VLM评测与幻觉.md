# 通用 VLM 评测与幻觉

> **预计阅读：45 分钟 | 前置知识：[05-通用VLM架构与视觉编码器](./05-通用VLM架构与视觉编码器.md)、[02-无人机场景理解](./02-无人机场景理解.md)**

前两篇讲的是模型怎么造。这一篇讲的是**怎么知道它行不行**——以及为什么这个问题的答案比看起来更依赖评测集本身。

核心结论先摆出来：**评测口径不是中立的测量工具。** 同一批模型，换一套负样本的抽样方式，名次就会翻。第 9 节用一个可跑的实验把这件事量出来。

---

## 1. 三种评测口径

VLM 的评测方法可以归成三类，各自的失效方式不同：

| 口径 | 例子 | 度量 | 主要失效来源 |
|---|---|---|---|
| 开放式生成 | `2309.17421` Dawn of LMMs、`2402.04788` MLLM-as-a-Judge | 人工打分 / 强模型打分 | 打分者的长度偏好、风格偏好；不可复现 |
| 选择题式 | `2311.16502` MMMU、`2307.06281` MMBench、`2306.13394` MME | 准确率 | 随机猜测基线、选项先验、**不看图也能答对** |
| 判别式/二分类 | `2305.10355` POPE | 准确率 / F1 | **负样本的抽样分布**（本篇主题） |

> **判据**：报一个 VLM 分数前，先问「它的随机猜测基线是多少」。选择题四选一是 25%，二分类是 50%。一个 0.52 的二分类准确率与一个 0.30 的四选一准确率，前者可能什么都不代表。凡是只报准确率、不报基线、不报分档构成的表，可比性都应当被质疑。

---

## 2. 选择题式基准：能力与天花板

### 2.1 主流基准各测什么

- `2311.16502` **MMMU**：大学水平的学科推理（工程、医学、艺术等），题目需要把图与专业知识结合起来。它的贡献是把评测从「日常场景描述」推到「需要专业知识」。
- `2409.02813` **MMMU-Pro**：对 MMMU 的加固版。加固方式里有两条直接指向本篇主题：**把问题从「图 + 文字」改成「只看图」**，以及**增加选项数**。原版 MMMU 可以在只给文字、不给图的情况下答对相当一部分题目——也就是说，原版分数里有一部分来自语言先验，不是来自看图。
- `2307.06281` **MMBench**：把能力拆成多个维度（属性、关系、计数、空间等），并用循环选项的方式降低「选 A 偏置」的影响。
- `2306.13394` **MME**：感知与认知两大类，采用是/否二分类提问。
- `2310.02255` **MathVista**：图与数学推理结合。
- `2403.20330` **MMStar**：专门针对**数据泄漏**与**题目里存在无关视觉信息**这两个问题，对题目做人工筛选。
- `2311.17005` **MVBench**：视频理解，把时间维度纳入评测。
- `2408.13257` **MME-RealWorld**：高分辨率真实场景，题目难度对标人类。

### 2.2 一个反复出现的发现：分数里混着「不看图也能答对」的成分

- MMMU-Pro 的加固动机就是它（`2409.02813`）；
- `2310.19785`（What's "up" with VLMs）发现模型在处理空间关系时，**「上下」类问题上的表现与图像的实际几何关系不匹配**，而与训练数据里物体的常见位置统计更一致——模型答对「杯子在桌子上面」，可能只是因为它见过很多杯子在桌子上；
- `2404.12390` **BLINK** 一组对人类而言几乎是「一眼就会」的任务（相对深度、视错觉、对应关系），模型表现却接近随机；
- `2401.06209` **Eyes Wide Shut / MMVP**：指出 CLIP 类视觉编码器本身对某些视觉模式不敏感，模型「能看见但看不见关键的东西」。

这四篇从不同角度说的是同一件事：**一个看着不错的准确率，可能绝大部分来自语言先验，而不是视觉。** 这与第 3、4、5 节要讲的幻觉评测是同一条线索的两端。

---

## 3. POPE 与幻觉评测

### 3.1 把幻觉变成一个二分类问题

`2305.10355`（POPE, Polling-based Object Probing Evaluation）的做法很直接：问模型「图里有没有 X？」，X 有时在图上（正例），有时不在（负例），用准确率或 F1 度量。

这个设计的好处是**可自动化、成本低、二分类的判定明确**。代价是它把「幻觉」这个宽泛的概念压缩成了一件事：**对物体存在性的过度肯定**。而这件事对负样本怎么选极其敏感——这正是第 4 节和第 9 节的内容。

### 3.2 幻觉的分类与更全的诊断套件

`2402.00253` 的综述把幻觉分成几类，与 POPE 只覆盖的一类相比要宽得多。更全的诊断套件：

| 工作 | 测什么 | 与 POPE 的差别 |
|---|---|---|
| `2310.14566` HallusionBench | 语言幻觉与视觉错觉的**纠缠** | 强调两者难以分开，而不是只测物体存在性 |
| `2311.07397` AMBER | 多维度幻觉，**不依赖外部 LLM 打分** | 去掉打分模型带来的偏差 |
| `2401.06209` MMVP | 视觉编码器看不见的模式 | 诊断在编码器层，不在语言侧 |
| `2404.12390` BLINK | 人类视知觉直觉类任务 | 覆盖感知而非知识 |

### 3.3 缓解方法都在推理期做干预

- `2311.16922` **VCD**（视觉对比解码）：对比「原图」与「加噪/失真图」两侧的输出分布，放大真正依赖视觉的那部分信号。
- `2311.17911` **OPERA**：对过度自信的 token 施加惩罚，并在生成后回溯重分配。
- `2310.16045` **Woodpecker**：先检测幻觉、再纠正，是一条流水线。

> **判据**：这三类方法的共同前提是「**模型其实知道正确的东西，只是被语言先验压住了**」。这个前提对「物体存在性」类幻觉往往成立，对「视觉编码器根本看不见」类幻觉不成立——后者再怎么调解码也调不出来。用 VCD/OPERA 这类方法前，先确认要治的是哪一类，否则会得到「方法无效」的错误结论。

---

## 4. 负样本构造：随机、流行、对抗

这是本篇与第 9 节实验的直接接口。POPE 式评测里的负样本（图上没有的物体 X）可以按三种方式挑：

| 档位 | 怎么挑 X | 测的是什么 | 语言先验模型的命运 |
|---|---|---|---|
| random | 在词表上逐类等概率抽 | 模型会不会对**冷门、明显不该在场**的东西答「有」 | 轻松答对（答「没有」就对了） |
| popular | 按类频抽，多半是常见类 | 模型会不会对**常见的、但这张图恰好没有**的东西答「有」 | 崩（常见类先验说「应该有」） |
| adversarial | 从与图中物体**最相似/最共现**的类里抽 | 模型会不会被共现先验带偏 | 崩 |

三档的难度不可比，这是一定要记住的一点：**同一个模型的三个檔位分数不能横向比较**，只有同一档内不同模型的排名有意义。原因很直白——每档里「答对很容易」的比例不同，random 档是送分题。

另一个容易被忽略的点：三档是**离散的划分**，而真实世界里的共现结构是连续的。一个模型在 popular 档和 adversarial 档之间的表现，未必能按顺序排列。

---

## 5. yes-偏差如何骗过基准

### 5.1 准确率可以被两个成分互相补偿

二分类的准确率可以写成两个成分的平均：

```text
Accuracy = (Recall + Specificity) / 2        Recall = 正例答对的比率
                                              Specificity = 负例答对的比率
```

于是**一个成分的缺陷可以被另一个成分的盈余掩盖**。语言先验模型正是这种情况：它在正例上有极高的 Recall（正例本身就是常见类，先验直接答「有」），而 Specificity 取决于负样本有多「冷门」。

random 档的负样本多是冷门类，语言先验答「没有」就能拿到高 Specificity；换成 popular 或 adversarial 档，Specificity 立刻塌掉。**但只有 random 档时，你会看到一个很高的总准确率。**

### 5.2 两个更早暴露问题的数

1. **分档报告。** 不报总平均，报三档各自的分数。语言先验模型的三档分数会呈现出「一档极高、两档接近 0.5」的形态，一眼可辨。
2. **yes-ratio。** 模型答「有」的比率。全盲的先验模型在难档位上会逼近 1.0——**它在猜「有」**。这个数比准确率更早、更清楚地暴露「模型在看图还是在猜」。

第 9 节的实测把这两点都量化了：全盲的先验模型在 random 档拿到 0.9250 的准确率排第一，而它的 yes-ratio 从 0.5150 一路升到 0.9450。

> **判据**：任何二分类式 VLM 评测结果，必须同时给出**分档分数**与 **yes-ratio**。只给总准确率的表，无法区分「会看图的模型」与「会猜 yes 的模型」——这不是精度问题，是有效性问题。

---

## 6. 从通用基准到遥感、无人机基准

通用基准搬到遥感/无人机场景会同时失效在两处，且失效方向相反：

**第一，目标尺度。** 通用基准的图像分辨率下，「图里有没有一个人」是清晰可见的；无人机高度下，同一个人可能只占几个像素。此时模型的错误来源从「语言先验」变成了「分辨率不足」——而这在通用基准上根本测不出来（[05 篇](./05-通用VLM架构与视觉编码器.md)第 9 节量化了这个退化）。

**第二，类别分布与共现结构。** 「流行类」在自然图像里是桌子、椅子、人；在俯视航拍里是车、屋顶、树。**共现结构也不同**——自然图像里「人」与「椅子」高度共现，航拍里「车」与「道路」高度共现。把这套结构搬到无人机基准上，需要自己统计。

**第三，描述的语言先验更强。** 航拍图像的内容分布比自然图像窄，模板化描述更容易蒙对，这会让「不看图也能答对」的比例更高。

**第四，拒答与不确定性的代价不对称。** 通用基准上答错扣一分；无人机上「说有障碍」与「说没有障碍」的代价完全不同级。评测指标本身要反映这个不对称，而通用基准的准确率不反映。

遥感与无人机侧的基准（UAVBench、BEDI、Embodied4C、OS-W2S 等）见 [02 篇](./02-无人机场景理解.md)。它们补上了第一、二点（尺度与类别分布），但**负样本构造的分布敏感性**这一层，在那批基准里同样没有系统设计——这是本篇可以补进去的一条检查。

---

## 7. 对无人机的含义

**第一，安全相关的判断必须用对抗档负样本验收。** 「前方有没有障碍物」「画面里有没有人」这类二分类，用随机负样本测出来的高准确率没有任何意义——随机负样本在无人机场景里对应的是「问一个显然不可能出现在空域里的东西」。有效的验收必须用**与当前场景高度共现的目标**做负样本：在有道路的画面里问「有没有车」，在有屋顶的画面里问「有没有窗」。这类负样本的构造需要对场景的共现统计，不能用通用词表直接搬。

**第二，假阳性与假阴性的代价不同，单一准确率报不了这件事。** 无人机上：
- 假阳性（说了「有」但实际没有）→ 不必要的规避、任务中断；代价是效率。
- 假阴性（说了「没有」但实际有）→ 碰撞、坠机；代价可能是安全。

这两类错误的权重在无人机上差一个量级，所以应该报**加权代价**或者分别报假阳率/假阴率，而不是一个平衡的平均准确率。POPE 式的平衡集设计（正负各半）恰恰把这两类错误当成了等价。

**第三，机载模型的风险画像要包含「它是不是在猜」。** 第 5.2 节的 yes-ratio 在机载场景里是一个比准确率更实用的监视指标：如果模型在飞行中的 yes-ratio 突然逼近 1.0，说明它当前面对的输入已经超出了它的判别能力，此时**降低它的权限**（改为上报而非自主决策）比继续用它更安全。

**第四，评测集要按飞行高度分层。** 目标像素数随高度剧烈变化，把不同高度的样本混在一个平均分里，会把高度带来的退化摊平。第 6 节的第一条失效只能用分层评测暴露。

---

## 8. 关键论文

> 本节每条 arXiv 号已通过 arXiv API 核对标题。

#### 评测框架与代表性基准

- **[arXiv'23.09] Dawn of LMMs** — *The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)*
  [![arXiv](https://img.shields.io/badge/arXiv-2309.17421-b31b1b.svg)](https://arxiv.org/abs/2309.17421)
  开放式评测的代表性工作，系统列举了 GPT-4V 的能力与失败模式。

- **[arXiv'23.11] MMMU** — *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*
  [![arXiv](https://img.shields.io/badge/arXiv-2311.16502-b31b1b.svg)](https://arxiv.org/abs/2311.16502)
  大学水平的学科推理基准，把评测推到需要专业知识的层级。

- **[arXiv'24.09] MMMU-Pro** — *MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark*
  [![arXiv](https://img.shields.io/badge/arXiv-2409.02813-b31b1b.svg)](https://arxiv.org/abs/2409.02813)
  加固版：去掉纯文字的得分路径，暴露原版分数里来自语言先验的部分。

- **[arXiv'23.07] MMBench** — *MMBench: Is Your Multi-modal Model an All-around Player?*
  [![arXiv](https://img.shields.io/badge/arXiv-2307.06281-b31b1b.svg)](https://arxiv.org/abs/2307.06281)
  能力维度分解 + 循环选项，降低选项偏置。

- **[arXiv'23.06] MME** — *MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2306.13394-b31b1b.svg)](https://arxiv.org/abs/2306.13394)
  感知与认知两大类，是/否二分类提问。

- **[arXiv'23.08] MM-Vet** — *MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities*
  [![arXiv](https://img.shields.io/badge/arXiv-2308.02490-b31b1b.svg)](https://arxiv.org/abs/2308.02490)
  用开放式回答评测能力的组合使用。

- **[arXiv'23.10] MathVista** — *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.02255-b31b1b.svg)](https://arxiv.org/abs/2310.02255)
  图与数学推理结合。

- **[arXiv'24.03] MMStar** — *Are We on the Right Way for Evaluating Large Vision-Language Models?*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.20330-b31b1b.svg)](https://arxiv.org/abs/2403.20330)
  针对数据泄漏与无关视觉信息筛选题目。

- **[arXiv'23.11] MVBench** — *MVBench: A Comprehensive Multi-modal Video Understanding Benchmark*
  [![arXiv](https://img.shields.io/badge/arXiv-2311.17005-b31b1b.svg)](https://arxiv.org/abs/2311.17005)
  视频理解基准，把时间维度纳入评测。

- **[arXiv'24.08] MME-RealWorld** — *MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?*
  [![arXiv](https://img.shields.io/badge/arXiv-2408.13257-b31b1b.svg)](https://arxiv.org/abs/2408.13257)
  [![GitHub](https://img.shields.io/badge/GitHub-MME__RealWorld-181717.svg?logo=github)](https://github.com/MME-Benchmarks/MME-RealWorld)
  高分辨率真实场景，难度对标人类。

- **[arXiv'24.02] MLLM-as-a-Judge** — *MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark*
  [![arXiv](https://img.shields.io/badge/arXiv-2402.04788-b31b1b.svg)](https://arxiv.org/abs/2402.04788)
  用模型做评委的可信度研究。

- **[arXiv'24.10] LLaVA-Critic** — *LLaVA-Critic: Learning to Evaluate Multimodal Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2410.02712-b31b1b.svg)](https://arxiv.org/abs/2410.02712)
  把「当评委」训进模型里，替代外部打分器。

- **[arXiv'23.10] What's "up" with VLMs** — *What's "up" with vision-language models? Investigating their struggle with spatial reasoning*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.19785-b31b1b.svg)](https://arxiv.org/abs/2310.19785)
  空间关系上的成功来自位置先验，而不是几何理解。

- **[arXiv'24.04] BLINK** — *BLINK: Multimodal Large Language Models Can See but Not Perceive*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.12390-b31b1b.svg)](https://arxiv.org/abs/2404.12390)
  人类直觉级的视知觉任务，模型接近随机。

- **[arXiv'24.01] MMVP / Eyes Wide Shut** — *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs*
  [![arXiv](https://img.shields.io/badge/arXiv-2401.06209-b31b1b.svg)](https://arxiv.org/abs/2401.06209)
  把视觉侧短板追溯到编码器对特定模式的盲区。

#### 幻觉评测

- **[arXiv'23.05] POPE** — *Evaluating Object Hallucination in Large Vision-Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2305.10355-b31b1b.svg)](https://arxiv.org/abs/2305.10355)
  [![GitHub](https://img.shields.io/badge/GitHub-POPE-181717.svg?logo=github)](https://github.com/AoiDragon/POPE)
  把幻觉压缩成物体存在性的二分类问题，负样本分随机/流行/对抗三档。

- **[arXiv'23.10] HallusionBench** — *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.14566-b31b1b.svg)](https://arxiv.org/abs/2310.14566)
  语言幻觉与视觉错觉纠缠时的诊断。

- **[arXiv'23.11] AMBER** — *AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation*
  [![arXiv](https://img.shields.io/badge/arXiv-2311.07397-b31b1b.svg)](https://arxiv.org/abs/2311.07397)
  多维度幻觉评测，去掉外部 LLM 打分带来的偏差。

- **[arXiv'24.02] A Survey on Hallucination in LVLM** — *A Survey on Hallucination in Large Vision-Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2402.00253-b31b1b.svg)](https://arxiv.org/abs/2402.00253)
  幻觉的分类学、成因与缓解方法的系统梳理。

#### 幻觉缓解

- **[arXiv'23.11] VCD** — *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*
  [![arXiv](https://img.shields.io/badge/arXiv-2311.16922-b31b1b.svg)](https://arxiv.org/abs/2311.16922)
  对比原图与失真图的输出分布，放大真正依赖视觉的信号。

- **[arXiv'23.11] OPERA** — *OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation*
  [![arXiv](https://img.shields.io/badge/arXiv-2311.17911-b31b1b.svg)](https://arxiv.org/abs/2311.17911)
  对过度自信的 token 施加惩罚并回溯重分配。

- **[arXiv'23.10] Woodpecker** — *Woodpecker: Hallucination Correction for Multimodal Large Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.16045-b31b1b.svg)](https://arxiv.org/abs/2310.16045)
  先检测幻觉、再纠正的流水线。

#### 工具

- **[GitHub] VLMEvalKit** — 开源多模态评测工具箱，覆盖本节多数基准
  [![GitHub](https://img.shields.io/badge/GitHub-VLMEvalKit-181717.svg?logo=github)](https://github.com/open-compass/VLMEvalKit)

- **[GitHub] lmms-eval** — 另一套多模态评测框架，支持自定义基准接入
  [![GitHub](https://img.shields.io/badge/GitHub-lmms__eval-181717.svg?logo=github)](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---

## 9. 动手验证：负样本怎么选，决定了谁排第一

第 4、5 节的两条判断在这里被量出来。

### 9.0 装置：三个模型 × 三档负样本

不训练任何模型，用**决策规则的概率模型**把「图像盲靠先验」与「有视觉接地」的差别写成两条确定的规则：

| 模型 | 规则 |
|---|---|
| `M_prior` | 图像盲：只看类序号，类序号小于阈值 100 就答 yes——**一眼都不看图** |
| `M_mixed` | 一半回合退回先验捷径、一半靠视觉接地的混合体 |
| `M_vision` | 有视觉接地但判别力有限：正例按固定判正率 0.85 答对，负例按档位难度答对 |

题面构造：词表 1000 类，类频取 Zipf(α=1)；每档 500 道正例（图上有该物体）+ 500 道负例（图上没有，且**X 不在图上**）。三档的区别只在负样本怎么挑：

- `random`：在词表上**逐类等概率**抽 → 抽到的大多是冷门类；
- `popular`：按类频的 1.5 次方抽 → 多半是常见类，恰巧这张图没有；
- `adversarial`：从与图上物体最相似的 50 个类里、再按类频 1.5 次方抽 → 先验说「应该有」、图上没有。

正例与常见类负样本都按类频的 1.5 次方抽：标注数据里常见类本来就占主导。

### 9.1 核心代码

```python
import numpy as np

SEED = 21
VOCAB, ALPHA, BETA = 1000, 1.0, 1.5     # 词表 / 类频指数 / 正题与常见负样本的集中度
N_PER_CLASS = 500                       # 每档 500 正 + 500 负
RANK_THRESHOLD = 100                    # M_prior 的阈值：类序号 < 100 就答 yes
P_VIS_RECALL = 0.85                     # M_vision 判"在"的概率（假设输入）
TN_BY_SPLIT = {"random": 0.90, "popular": 0.80, "adversarial": 0.62}
SPLITS = ["random", "popular", "adversarial"]


def zipf(alpha=ALPHA):
    w = 1.0 / np.arange(1, VOCAB + 1) ** alpha
    return w / w.sum()


def answer_prior(q):
    """图像盲：只看类序号，常见类答 yes。"""
    return q < RANK_THRESHOLD


def build_questions(rng, beta=BETA):
    """三档题库。区别只在负样本怎么抽。"""
    p = zipf()
    w = p ** beta
    cum_b = np.cumsum(w / w.sum())
    present = np.searchsorted(cum_b, rng.random(N_PER_CLASS))

    emb = rng.standard_normal((VOCAB, 8))                 # 类嵌入：用来定义"相似"
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    topk = np.argpartition(-(emb @ emb[present].T), 50, axis=0)[:50]     # 最相似的 50 类
    ww = p[topk] ** beta
    ww /= ww.sum(axis=0, keepdims=True)                   # 池内再按类频加权
    u = rng.random(N_PER_CLASS)[None, :]
    adv = topk[(np.cumsum(ww, axis=0) > u).argmax(axis=0), np.arange(N_PER_CLASS)]

    absent = {
        "random":      rng.integers(0, VOCAB, N_PER_CLASS),          # 逐类等概率
        "popular":     np.searchsorted(cum_b, rng.random(N_PER_CLASS)),
        "adversarial": adv,                                          # 最相似类内再按类频抽
    }
    truth = np.concatenate([np.ones(N_PER_CLASS, bool), np.zeros(N_PER_CLASS, bool)])
    return {s: (np.concatenate([present, absent[s]]), truth) for s in SPLITS}
```

`M_vision` 的三个判负率（`TN_BY_SPLIT`）是**本模型的假设输入，不是从真实 VLM 上测出来的**——它代表「这一档的负样本有多难分辨」。第 9.3 节的限制条款里会再强调一次。

### 9.2 本地实测结果

**① 三档负样本上的五个指标**

```text
[random]  负样本 = 全词表逐类等概率（多是冷门类）
        模型   Accuracy       F1  Precision   Recall  yes-ratio
   M_prior     0.9250   0.9261     0.9126   0.9400     0.5150
   M_mixed     0.9000   0.8986     0.9115   0.8860     0.4860
  M_vision     0.8840   0.8802     0.9103   0.8520     0.4680
   名次（按 Accuracy）: M_prior > M_mixed > M_vision

[popular]  负样本 = 按类频 p^1.5（常见类）
   M_prior     0.4970   0.6514     0.4984   0.9400     0.9430
   M_mixed     0.6480   0.7166     0.5997   0.8900     0.7420
  M_vision     0.8270   0.8319     0.8091   0.8560     0.5290
   名次（按 Accuracy）: M_vision > M_mixed > M_prior

[adversarial]  负样本 = 最相似 50 类内按 p^1.5
   M_prior     0.4950   0.6505     0.4974   0.9400     0.9450
   M_mixed     0.5950   0.6819     0.5614   0.8680     0.7730
  M_vision     0.7570   0.7785     0.7152   0.8540     0.5970
   名次（按 Accuracy）: M_vision > M_mixed > M_prior
```

**② 名次在 random 与 popular 之间翻转**

```text
         负样本抽样   M_prior   M_mixed  M_vision        第一   M_prior yes-ratio
        random    0.9250    0.9000    0.8840   M_prior              0.5150
       popular    0.4970    0.6480    0.8270  M_vision              0.9430
   adversarial    0.4950    0.5950    0.7570  M_vision              0.9450
```

图像盲的 `M_prior` 在 random 档拿到 0.9250 并排第一，换成 popular 负样本掉到 0.4970——**差 0.4280**。同一套词表、同一个阈值、同一个模型，差别只在负样本怎么抽。它的 Recall 在三档上都是 0.9400：正例靠先验就答完了，视觉接地在正例上完全没体现。

`M_mixed` 对 `M_vision` 的对比更细致：random 档 `+0.0160`（混合体赢），popular 档 `-0.1790`（接地模型赢）——**名次翻转**。

**③ 翻转依赖什么：正题的类频集中度**

这是本节最值得单列的一栏。上面那个翻转是**有条件的**，条件就是「图上的物体有多集中在常见类上」：

```text
  beta   M_prior(R)   M_vision(R)      random 谁第一   M_prior(P)   M_vision(P)     popular 谁第一
  1.00       0.8260        0.8650        M_vision       0.5190        0.8420        M_vision
  1.25       0.8810        0.8650         M_prior       0.5090        0.8420        M_vision
  1.50       0.9250        0.8840         M_prior       0.4970        0.8270        M_vision
  1.75       0.9340        0.8650         M_prior       0.5040        0.8420        M_vision
  2.00       0.9520        0.8650         M_prior       0.5020        0.8420        M_vision
  2.50       0.9290        0.8650         M_prior       0.5010        0.8420        M_vision
```

β = 1.00（正题按类频本身抽）时，random 档 M_prior 只有 0.8260，**排第二**，排序与 β = 2.50 时相反；翻转从 β ≥ 1.25 开始出现。β = 1.50 那一行就是 ① 的主实验（不重抽，避免同一实验因重抽样差出噪声）。

`M_vision` 那一列近乎水平，因为它不看类频先验，β 只通过取样噪声影响它——这正是「视觉接地的模型不依赖先验」在数字上的样子。

所以「random 档会高估先验模型」这条结论**有条件**：条件是真实标注数据里正题类的类频集中程度。那个数要从真实数据统计，本 demo 不能替代。不依赖 β 的那一条是：**popular 档下 M_prior 一律垫底**（0.4970–0.5190），且 `M_vision` 一律第一。

**④ yes-ratio 是更早暴露问题的那个数**

```text
        模型          random         popular     adversarial
   M_prior          0.5150          0.9430          0.9450
   M_mixed          0.4860          0.7420          0.7730
  M_vision          0.4680          0.5290          0.5970
```

`M_prior` 的 yes-ratio 从 random 档的 0.5150 升到 popular 档的 0.9430，逼近全答 yes。此时的 Accuracy 是 0.4970——落在 0.5 附近，看起来像「随机猜」；yes-ratio 却明确告诉你：**它不是在随机猜，它在系统性地答「有」。** 这是第 5.2 节那条判据的直接证据。

### 9.3 运行完整脚本

```bash
py -3.9 code/x_hallucination_metric.py
```

约 2 秒，纯 numpy；输出 `figures/x_hallucination_metric.png`（三栏：三档准确率对比、翻转随 β 的变化、yes-ratio）。

> **限制**：本实验是**决策规则的概率模型**：类频、相似类结构、`M_vision` 的三个判负率都是设定的，不是从真实 VLM 上测出来的。它量的是「**评测协议对负样本分布的敏感性**」，不是某个模型的幻觉水平。结论里**不依赖**那些设定值的部分是：(a) popular 档下图像盲模型一律垫底；(b) 同一协议下，任何掺先验的模型都会被「能靠先验答对的负样本」高估。已知不能外推的部分：(c) 真实模型的 yes-偏置随问题类型变，不是常数；(d) 真实共现结构来自数据统计，这里用随机嵌入近似；(e) 真实基准的正题分布比 p^1.5 更复杂，翻转的具体位置取决于它（见 9.2 ③）；(f) 500 题的二项标准误约为 1.6 个百分点，档间小于 2 个百分点的比较不可靠——表中 ② 的 `+0.0160` 就落在这一带，不应作为强结论。

---

## 10. 延伸阅读

- [02-无人机场景理解](./02-无人机场景理解.md) — 无人机侧的评测基准全景
- [05-通用VLM架构与视觉编码器](./05-通用VLM架构与视觉编码器.md) — 分辨率不足会以另一种方式污染分数
- [06-VLM指令微调与对齐](./06-VLM指令微调与对齐.md) — 肯定偏差从哪来
- [01-遥感VLM](./01-遥感VLM.md) — 遥感侧的评测与定位能力
- [论文批判性阅读](../08-研究前沿与开放问题/03-论文批判性阅读.md) — 怎么读一张只报总分的评测表
- [VLM 专题自测](../09-专题自测与考察/03-VLM专题自测.md) — 三层考察，含本篇内容

---

## 11. 思考题

### 题目 1：为什么 POPE 的三档负样本分数不能横向比较，但同一档内的排名有意义？

<details>
<summary>查看参考答案</summary>

因为三档的**基线难度**不同，而基线难度主要由抽样分布决定，与模型无关。

- random 档在词表上逐类等概率抽，抽到的绝大多数是冷门类。对任何模型来说，「答没有」几乎都是对的，所以这一档的准确率天花板很高、基线也很高。
- popular 档按类频抽，抽到的多半是常见类。此时「答没有」不再安全，模型的先验知识反而成为负担。
- adversarial 档更难：候选池就是与图上物体最相似的类，即使模型在判别上做得不错，也更容易被带偏。

三档的**正例是完全相同的**（都是同一批采样），差别只在负例的抽样分布。所以「同一个模型在 random 档 0.92、在 popular 档 0.49」这个跨档比较，量的是**这两档的难度差**，不是模型在这两种场景下的能力差。

同一档内不同模型的排名有意义，是因为此时正负样本的分布对参与比较的所有模型完全相同，难度差异被抵消掉了，剩下的才是模型之间的差别。

**实践含义**：报告结果时应当报「三档 × 各模型」的完整表，而不是把三档平均成一个数。平均会把「一档极高、两档垫底」这种形态磨平成「中等」，正好把问题藏起来。

</details>

### 题目 2：准确率 0.5 的二分类模型，可能是「完全不会」（随机猜），也可能是「全答 yes」。怎么区分？

<details>
<summary>查看参考答案</summary>

区分的方法是看**混淆矩阵的分解**，而不是看准确率本身。

一个平衡集（正负各半）上：
- **全答 yes 的模型**：Recall = 1.00，Specificity = 0.00，Accuracy = 0.50，F1 = 0.667（Precision = 0.5, Recall = 1.0），yes-ratio = 1.00。
- **随机猜的模型**：Recall ≈ 0.50，Specificity ≈ 0.50，Accuracy = 0.50，F1 = 0.50，yes-ratio ≈ 0.50。

两者的准确率完全相同，但：

1. **yes-ratio 直接分开它们**：1.00 对 0.50。
2. **F1 也能分开**：0.667 对 0.50。全答 yes 的 F1 反而更高——这正是「只看 F1」也会被误导的例子。
3. **两者失效的方式完全不同**，对应的修复手段也不同：随机猜的模型是没有学会任务，全答 yes 的模型是学会了「答有最保险」，后者更接近语言先验问题，与 [06 篇](./06-VLM指令微调与对齐.md)第 2.2 节的肯定偏差同源。

本节实测里 `M_prior` 在 popular 档的表现就是这个情形的实例：Accuracy = 0.4970、Recall = 0.9400、yes-ratio = 0.9430。它离「全答 yes」已经很近，而 Accuracy 单独看会让人以为它只是「不会」。

**由此得出一条更一般的判据**：任何二分类评测，Accuracy、F1、yes-ratio 三个数必须一起报。缺一个，就无法区分这几种截然不同的失效模式。

</details>

### 题目 3：第 9 节的结论依赖 β 这个设定值。这对「用 demo 支持论文结论」这种做法意味着什么？

<details>
<summary>查看参考答案</summary>

意味着**任何用一个可调设定值支撑的结论，都必须同时报告它的敏感性**，否则这个结论只是把设定值当成了结论。

第 9.2 节③ 就是这次做的处理：

1. **把设定值当自变量扫一遍**，而不是固定成主实验用的那个数。β 从 1.0 扫到 2.5，看翻转在什么位置出现、什么位置消失。
2. **明确报告「翻转从 β ≥ 1.25 开始」**。这句话比「翻转出现了」信息量大得多——它给出了结论成立的条件。
3. **把不依赖设定值的那部分单独挑出来**。popular 档下 M_prior 一律垫底（0.4970–0.5190），在全部 β 下都成立，这是真正稳的那一条。
4. **把设定值的真实来源标出来**。β 是「图上物体的类频集中度」，它在真实数据里有一个真实的值，但本 demo 测不出来——必须由读者用真实标注统计去补。

对照第 9.3 节的限制条款：还额外说明了 `TN_BY_SPLIT` 的三个数是**假设输入**（其中 0.90/0.80/0.62 代表三档的难度差异），以及 500 题规模带来的 ±1.6 个百分点噪声——这使得表中 `+0.0160` 这个差值落在噪声带里，不该作为结论。

**一般化的做法**：把一个解析式或蒙特卡洛 demo 用来支持结论时，至少要做三件事——列出所有**人为设定的常数**、对每一个做敏感性扫描、把「仅在设定值取某范围时成立」的结论与「无条件成立」的结论分开写。做不到第三条的，通常是因为这个 demo 只在某一个设定点上跑过。

</details>

---

> **读完了自测**：[VLM 专题自测](../09-专题自测与考察/03-VLM专题自测.md) — 三层考察加无人机专场 12 问，答不上来的顺着指针回读
