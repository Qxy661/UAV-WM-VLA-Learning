"""
watchlists.py — 前沿追踪的词表，按卷分。

`watch.py` 只用这里的词表，取数、去重、纪律文案都在内核里。分开的理由：词表是
**会随专题长大的内容**（每加一篇文档就加一节），而内核不该跟着改。

三个卷：

  - `vla` — 对应 `docs/03-VLA专题/06`–`10` 五篇
  - `wm`  — 对应 `docs/02-世界模型专题/07`–`09` 三篇
  - `vlm` — 对应 `docs/04-VLM专题/05`–`07` 三篇

每卷的 `sections` 是有序 dict：`节 id -> (中文名, [查询, ...])`。节 id 就是文档编号，
`uav` 是全场主线、不单独成篇。查询一律用 `abs:`，避开作者名与注释里的偶然命中。

`VERIFIED_ZEROS` 是**跨卷共用**的：同一句查询不管从哪一卷跑出来，复核结论都一样。
"""

# 词表。检查过的 0 命中写在 VERIFIED_ZEROS 里，别在这里注释——
# 复核结论必须随生成物一起写出去，否则它只活在对话里，下次重跑又变成一个裸的 0。
VOLUMES = {
    "vla": {
        "title": "VLA",
        "slug": "vla",
        "about": "docs/03-VLA专题/06–10 五篇",
        "sections": {
            "06": ("动作头与动作分块", [
                'abs:"action chunking"',
                'abs:"action head" AND abs:"vision-language"',
                'abs:"flow matching" AND abs:"robot policy"',
                'abs:"diffusion policy" AND abs:"manipulation"',
                'abs:"action tokenization"',
                'abs:"receding horizon" AND abs:"imitation learning"',
            ]),
            "07": ("数据、预训练与跨具身", [
                'abs:"cross-embodiment"',
                'abs:"robot dataset" AND abs:"pretraining"',
                'abs:"co-training" AND abs:"robot"',
                'abs:"latent action" AND abs:"robot"',
                'abs:"Open X-Embodiment"',
                'abs:"human video" AND abs:"robot policy"',
            ]),
            "08": ("强化学习后训练与自我改进", [
                'abs:"reinforcement learning" AND abs:"vision-language-action"',
                'abs:"GRPO" AND abs:"robot"',
                'abs:"reward model" AND abs:"robot policy"',
                'abs:"self-improvement" AND abs:"robot"',
                'abs:"world model" AND abs:"policy improvement"',
                'abs:"online fine-tuning" AND abs:"robot policy"',
            ]),
            "09": ("评测基准与报告口径", [
                'abs:"LIBERO"',
                'abs:"SimplerEnv"',
                'abs:"VLA benchmark"',
                'abs:"real robot" AND abs:"evaluation protocol"',
                'abs:"success rate" AND abs:"robot policy" AND abs:"evaluation"',
                'abs:"reproducibility" AND abs:"robot learning"',
            ]),
            "10": ("世界模型增强 VLA", [
                'abs:"world model" AND abs:"vision-language-action"',
                'abs:"video prediction" AND abs:"robot policy"',
                'abs:"imagined rollout"',
                'abs:"latent dynamics" AND abs:"policy learning"',
                'abs:"model-based" AND abs:"vision-language-action"',
            ]),
            "uav": ("无人机主线", [
                'abs:"aerial" AND abs:"vision-language-action"',
                'abs:"UAV" AND abs:"vision-language-action"',
                'abs:"drone" AND abs:"vision-language navigation"',
                'abs:"aerial" AND abs:"language model" AND abs:"control"',
                'abs:"quadrotor" AND abs:"foundation model"',
                'abs:"aerial" AND abs:"embodied" AND abs:"policy"',
            ]),
        },
    },

    "wm": {
        "title": "世界模型",
        "slug": "wm",
        "about": "docs/02-世界模型专题/07–09 三篇",
        "sections": {
            "07": ("世界模型评测与诊断", [
                'abs:"world model" AND abs:"evaluation"',
                'abs:"world model" AND abs:"benchmark"',
                'abs:"rollout" AND abs:"error accumulation"',
                'abs:"video prediction" AND abs:"evaluation metrics"',
                'abs:"world model" AND abs:"diagnostic"',
                'abs:"world model" AND abs:"failure modes"',
            ]),
            "08": ("联合嵌入预测与潜空间", [
                'abs:"JEPA"',
                'abs:"joint embedding" AND abs:"prediction"',
                'abs:"representation collapse" AND abs:"world model"',
                'abs:"latent space" AND abs:"world model"',
                'abs:"energy-based" AND abs:"world model"',
                'abs:"self-supervised" AND abs:"world model"',
            ]),
            "09": ("长时程与交互式生成", [
                'abs:"long-horizon" AND abs:"world model"',
                'abs:"interactive world model"',
                'abs:"action-conditioned" AND abs:"video generation"',
                'abs:"object permanence"',
                'abs:"autoregressive" AND abs:"video prediction"',
                'abs:"memory" AND abs:"long-horizon" AND abs:"navigation"',
            ]),
            "uav": ("无人机主线", [
                'abs:"UAV" AND abs:"world model"',
                'abs:"aerial" AND abs:"world model"',
                'abs:"drone" AND abs:"world model"',
                'abs:"aerial" AND abs:"video generation"',
                'abs:"UAV" AND abs:"video prediction"',
                'abs:"aerial" AND abs:"navigation" AND abs:"world model"',
            ]),
        },
    },

    "vlm": {
        "title": "VLM",
        "slug": "vlm",
        "about": "docs/04-VLM专题/05–07 三篇",
        "sections": {
            "05": ("通用架构与视觉编码器", [
                'abs:"vision encoder" AND abs:"multimodal large language"',
                'abs:"connector" AND abs:"multimodal large language"',
                'abs:"SigLIP"',
                'abs:"visual token" AND abs:"multimodal"',
                'abs:"native resolution" AND abs:"vision-language"',
                'abs:"contrastive" AND abs:"vision-language" AND abs:"pretraining"',
            ]),
            "06": ("指令微调与对齐", [
                'abs:"visual instruction tuning"',
                'abs:"LoRA" AND abs:"vision-language"',
                'abs:"instruction tuning" AND abs:"multimodal"',
                'abs:"parameter-efficient" AND abs:"multimodal"',
                'abs:"catastrophic forgetting" AND abs:"multimodal"',
                'abs:"preference alignment" AND abs:"vision-language"',
            ]),
            "07": ("通用评测与幻觉", [
                'abs:"hallucination" AND abs:"multimodal large language"',
                'abs:"MMMU"',
                'abs:"MMBench"',
                'abs:"POPE"',
                'abs:"visual question answering" AND abs:"hallucination"',
                'abs:"vision-language model" AND abs:"evaluation" AND abs:"bias"',
            ]),
            "uav": ("无人机主线", [
                'abs:"UAV" AND abs:"vision-language"',
                'abs:"aerial" AND abs:"vision-language model"',
                'abs:"drone" AND abs:"vision-language"',
                'abs:"remote sensing" AND abs:"vision-language model"',
                'abs:"UAV" AND abs:"multimodal large language model"',
                'abs:"aerial" AND abs:"visual question answering"',
            ]),
        },
    },
}


# 命中 0 的查询，按纪律 F 复核过之后把结论钉在这里，随生成一起写出去。
# 不写这个表的话，复核结论只活在对话里，下次重跑又变成一个裸的 0。
VERIFIED_ZEROS = {
    'abs:"receding horizon" AND abs:"imitation learning"':
        "**复核过：这是词汇层面的假空白。**换提法 `abs:\"execution horizon\"` 命中 **22**、"
        "`abs:\"action horizon\"` 命中 **11**（同 120 天窗口）。"
        "「receding horizon」是控制论的说法，VLA 社区写「action / execution horizon」。"
        "**不得记录为「这个方向没人做」。**",
    'abs:"quadrotor" AND abs:"foundation model"':
        "**复核过：零落在术语上，不落在领域上。**"
        "`aerial` / `UAV` / `drone foundation model`、`aerial foundation models` "
        "四种提法命中全为 **0**（两套词表交集为空，够得上「事实级空白」的判据）；"
        "但同窗口 `abs:\"aerial\" AND abs:\"vision-language-action\"` 命中 **11**。"
        "结论只能写到这一步：**空中领域不用「foundation model」自我描述，而用 VLA/VLN**。"
        "**不得写成「空中没有基础模型工作」。**",

    # ---- 以下四条是 2026-10-08 跑 wm 卷时出现的 0 命中，逐条换提法复核过 ----

    'abs:"video prediction" AND abs:"evaluation metrics"':
        "**复核过：词汇层面的假空白。**同窗口换提法 —— "
        "`abs:\"video prediction\" AND abs:\"benchmark\"` 命中 **14**、"
        "`abs:\"video generation\" AND abs:\"evaluation metrics\"` 命中 **6**、"
        "`abs:\"video prediction\" AND abs:\"metrics\"` 命中 **2**。"
        "不限窗口时原提法本身也只有 **3** 条，且全部止于 2021 年。"
        "「evaluation metrics」是评测论文的写法，生成类论文写 benchmark / metric（单数）。"
        "**不得记录为「视频预测没有评测工作」。**",

    'abs:"energy-based" AND abs:"world model"':
        "**复核过：低频真线，只是这 120 天恰好空窗。**不限窗口命中 **8**，"
        "且集中在 2026 上半年：`2605.07199`（Three-in-One World Model，能量一致性）、"
        "`2602.23058`（GeoWorld）。同族提法 `abs:\"EBM\" AND abs:\"world model\"` "
        "不限窗口命中 **1**（`2406.08862` Cognitively Inspired Energy-Based World Models）。"
        "对照：`abs:\"energy-based model\"` 不限窗口 **630**、同窗口 **35**，"
        "说明词本身不冷，是「energy-based + world model」这个组合少。"
        "**可以写「这一线稀疏」，不得写「没有」。**",

    'abs:"aerial" AND abs:"video generation"':
        "**复核过：纯粹是窗口假象。**不限窗口命中 **6**，最近三条就在窗口外几周："
        "`2605.19728`（Aero-World，2026-05-19）、`2605.15964`（WorldVLN，2026-05-15）、"
        "`2604.07991`（MotionScape，2026-04-09）。同窗口换提法 "
        "`abs:\"drone\" AND abs:\"video generation\"` 命中 **1**（`2606.24152`）、"
        "`abs:\"UAV\" AND abs:\"video generation\"` 命中 **1**（`2610.02451`）。"
        "**这是本轮最该记的一条：窗口起点卡在 6 月 10 日，"
        "把 2026 年 4–5 月空中世界模型那一批整批切掉了**——"
        "写「近期空白」之前必须先看不限窗口的命中。",

    'abs:"UAV" AND abs:"video prediction"':
        "**复核过：同上是窗口假象。**不限窗口命中 **2**："
        "`2606.06147`（WorldFly，2026-06-04，比窗口起点早 6 天）、"
        "`2512.21710`（RAPTOR，2025-12-25）。同窗口换提法 "
        "`abs:\"aerial\" AND abs:\"video prediction\"`、"
        "`abs:\"aerial\" AND abs:\"future frame prediction\"` 也都命中 **0**。"
        "**不得写成「无人机不做视频预测」。**",
}
