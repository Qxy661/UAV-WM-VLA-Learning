# 05 - 复现指南：MotionScape — 无人机视角的未来视频生成评测基准

> **预计阅读：16 分钟 | 前置知识：Python 与 ffmpeg 使用、光流基本概念**

MotionScape 是**真实拍摄**的无人机第一人称视频基准，用途是**评测**世界模型与未来视频生成——不是训练用的数据集。它给 228 段固定长度的片段、每段配语义标注与运动强度分级，再用一套统一协议去量各种生成模型的输出质量。

> **本指南的可复现边界**
> - **能跑通的**：仓库克隆、源视频重建、运动分层、语义标注。这几步只要有 Python、ffmpeg 和网络就能做，**不需要 GPU**。
> - **跑不动的**：基线推理与 FVD。五个基线里最小的是 Cosmos-Predict2.5-**2B**，最大的是 MAGI-1-**24B**，加一个独立的 TensorFlow 环境算 I3D-FVD。**8 GB 显存的机器到运动分层为止。**
> - **一个绕不开的限制**：原作者**不分发原视频**（README 原文：*"The original audiovisual content is not redistributed"*）。你拿到的是元数据与标注，视频要按 manifest 里的 URL 自己去抓。源被下架，对应样本就重建不出来。

---

## 目录

1. [基准概述](#1-基准概述)
2. [仓库与数据获取](#2-仓库与数据获取)
3. [数据格式详解](#3-数据格式详解)
4. [重建与运动分层](#4-重建与运动分层)
5. [语义标注与生成提示词](#5-语义标注与生成提示词)
6. [基线推理与评测](#6-基线推理与评测)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. 基准概述

- **GitHub**：[Thelegendzz/MotionScape](https://github.com/Thelegendzz/MotionScape)（8 stars）
- **数据集（元数据与标注）**：[HF `thelegendzz/MotionScape`](https://huggingface.co/datasets/thelegendzz/MotionScape) · [Zenodo DOI 10.5281/zenodo.21954342](https://doi.org/10.5281/zenodo.21954342)
- **论文**：arXiv:2604.07991
- **定位**（README 原句）：*"a real-world first-person UAV-view benchmark for evaluating world models and future video generation under different conditioning settings and visual-motion intensities"*

### 1.1 关键数字

| 属性 | 数值 | 出处 |
|---|---|---|
| 片段数 | **228** | 论文 Table 1 / manifest |
| 每段帧数 | **275**（固定，无例外） | manifest 全部 228 项的 `frame_count` |
| 总帧数 | **62,700** | 228 × 275 |
| 总时长 | **约 34.9 分钟** | 论文 |
| 每段时长 | **约 9.2 秒** | 论文（275 帧 @ 29.97 FPS） |
| 统一帧率 | **30000/1001 ≈ 29.97 FPS** | manifest `benchmark_specification` |
| 分段 | 观测前缀 **1–200 帧** + 未来目标 **201–275 帧** | 同上 |
| 前缀/目标总帧数 | 45,600 / 17,100 | 论文 |
| 源视频 | **67 个**，平均每源取 3.4 段 | 论文 |
| 源分辨率 | **≥ 1080p**；manifest 里 170/228 是 3840×2160 | 论文 Table 1 + manifest |
| 位姿真值 | **无** | 论文与发布内容 |
| 运动分级 | Low / Medium / High = **75 / 75 / 78** | 论文 + `dynamicity_buckets.json` |

### 1.2 它是「评测」不是「训练」

这个区分决定了整份指南的用法。228 段、34.9 分钟的体量**不足以训练一个世界模型**，它的角色是**考卷**：每段都切好了「看 200 帧、预测后 75 帧」，条件可以是文本、图像或视频，评测时用同一套协议跑所有模型。

它也**不给位姿**。没有相机轨迹、没有 GPS/IMU 真值，能评的是**生成得像不像、动力学对不对**，不能评定位精度——这一点在设计下游任务时会先撞上。

---

## 2. 仓库与数据获取

### 2.1 仓库结构

真实顶层（每个目录都有用）：

```text
MotionScape/
├── Dockerfile
├── LICENSE                       # MIT（evaluate_video_metrics.py 保留 NVIDIA 的 Apache-2.0）
├── README.md
├── requirements.txt
├── reconstruction/               # 源视频下载 + 基准重建
│   ├── download_sources.py
│   └── reconstruct_benchmark.py
├── motion_stratification/        # 光流运动分层
│   ├── compute_motion_strata.py
│   └── motion_scoring.py
├── annotation/                   # 语义标注生成
│   └── generate_semantic_annotations.py
├── evaluation/                   # 最终评测实现
│   ├── evaluate_video_metrics.py
│   ├── i3d_fvd.py
│   ├── regroup_metrics_by_motion.py
│   ├── requirements-frame-metrics.txt
│   └── requirements-i3d-fvd.txt
├── inference/                    # 五个基线的适配器 + 权重下载器
│   ├── cosmos/  cogvideox/  wan/  longcat/  magi/
│   ├── docker/                   # 统一基线容器
│   ├── download_model_weights.py
│   └── requirements-download.txt
└── prompts/                      # 语义标注与四套生成提示词
    ├── semantic_annotation.txt
    ├── text2world.txt  image2world.txt
    ├── video2world_task_level.txt  video2world_clip_level.txt
    └── clip_assisted_inspection.txt
```

### 2.2 环境准备

```bash
git clone https://github.com/Thelegendzz/MotionScape.git
cd MotionScape

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

README 推荐 **Python 3.10 或更新**。系统侧还需要 **`ffmpeg` 与 `ffprobe`** 两个可执行文件——重建脚本靠它们做时间重采样与抽帧，缺了会在第一步就报错。下载侧固定 `yt-dlp==2026.03.17`（manifest 里记录的就是这个版本）。

### 2.3 拿到的是标注，不是视频

Hugging Face 与 Zenodo 上发布的是**元数据与标注**：`source_manifest.json`、`annotations/`、`dynamicity_buckets.json`。原视频**不分发**。

所以要还原基准，路径只有一条：

```bash
# 第 1 步：按 manifest 抓源视频（yt-dlp 抓 YouTube，保留 manifest 里的文件名）
python reconstruction/download_sources.py \
  --manifest  /path/to/MotionScape/source_manifest.json \
  --output-dir /path/to/MotionScape_raw
```

> **下载**：仓库里没有 `scripts/download_dataset.sh`；`huggingface-cli download --repo-type dataset Thelegendzz/MotionScape` 下到的也只是标注，**没有视频**——发布方明确不分发视听内容，版权与可用性仍归源平台与版权方。

`download_sources.py` 按 manifest 记录的**分辨率、标称帧率、编码**去选流，并且会在记录的流不可用时**直接报错，而不是悄悄换一个编码**。可选参数里比较有用的是 `--proxy`、`--cookies-from-browser`、`--limit`、`--dry-run`。**已存在的源文件不会被覆盖。**

---

## 3. 数据格式详解

### 3.1 `source_manifest.json`

顶层四个键：`source_acquisition` / `benchmark_specification` / `items` / `summary`。

```jsonc
{
  "source_acquisition": {
    "tool": "yt-dlp",
    "format_selector": "bv[height=2160]+ba/bv+ba",
    "merge_output_format": "mp4",
    "extractor_args": "youtube:player_client=android",
    "yt_dlp_version": "2026.03.17"
  },
  "benchmark_specification": { "fps": "30000/1001", "num_frames": 275,
                               "observed_frames": 200, "future_frames": 75 },
  "summary": { "source_videos": 67, "benchmark_samples": 228 },
  "items": [ /* 228 项 */ ]
}
```

`items[]` 每项七个字段：`sample_id`、`video`、`video_index`、`segment_index`、`url`、`start_s`、`frame_count`、`resolution`、`fps`、`codec`。

**实测分布**（228 项逐条统计，可复算）：

| 字段 | 分布 |
|---|---|
| `frame_count` | `275` × 228（**无例外**） |
| `codec` | vp9 **169** · h264 **39** · av1 **20** |
| `resolution` | 3840×2160 **170** · 1920×1080 **43** · 2560×1440 **12** · 3840×1634 2 · 7680×3268 1 |
| `fps`（源） | 30/1 **92** · 30000/1001 **50** · 50/1 **40** · 60/1 **34** · 60000/1001 **12** |

这张表解释了两件事。**一是编码不只有 H.264**：H.264 只占 39/228，主力是 VP9。**二是源帧率很杂**（30 / 29.97 / 50 / 60 / 59.94），统一到 29.97 是重建阶段做的重采样，不是源本身就整齐。

`start_s` 是**每段在源视频时间轴上的定点**（例如 `44.0`、`15.047`），不是随机取。

### 3.2 `annotations/`

每个样本一个扁平 JSON，只有四个字段：

```json
{ "sample_id": "...", "weather": "Overcast", "environment": "Urban parking structure",
  "caption": "..." }
```

`weather` 与 `environment` 是**自由文本**，不是枚举——**写下游代码时不要按固定类别去 match**。`caption` 描述的是相机视角的运动，提示词里特意用 "the camera viewpoint" 或 "the onboard camera"，**不暗示能看到无人机本体**。

### 3.3 `dynamicity_buckets.json`

给每个样本的运动分数与 Low / Medium / High 归属。**这个分数是在缩放后的图像空间里量的，是视觉动态性/视角运动强度的代理量，不是物理相机速度**——README 专门写了这一句，引用数字时别把它当速度用。

### 3.4 重建产物

`reconstruct_benchmark.py` 为每个 `sample_id` 产出四份东西：

```text
MotionScape_reconstructed/
├── full_frames/<sample_id>/          # 275 帧图像序列
├── full_videos/<sample_id>.mp4       # 275 帧整段
├── observed_prefix_videos/<sample_id>.mp4   # 1–200 帧（条件输入）
└── last_75_videos/<sample_id>.mp4    # 201–275 帧（预测目标 / GT）
```

最后一步会**校验帧数正是 275 / 200 / 75**，对不上就报错。评测时 `--gt-dir` 指的就是 `last_75_videos`。

---

## 4. 重建与运动分层

### 4.1 重建

```bash
python reconstruction/reconstruct_benchmark.py \
  --input-json /path/to/MotionScape/source_manifest.json \
  --video-root /path/to/MotionScape_raw \
  --output-dir /path/to/MotionScape_reconstructed
```

对每个 `sample_id`，脚本依次做七件事：

1. 定位到源时间轴上的 `start_s`；
2. 时间重采样到 **30000/1001 FPS**；
3. 取**前 275 个连续输出帧**；
4. 写出 275 帧图像序列与整段视频；
5. 写出 1–200 帧的观测前缀视频；
6. 写出 201–275 帧的未来目标视频；
7. 校验 275 / 200 / 75 三个帧数。

两个容易踩的点：**公开的扁平 manifest 模式下 `end_s` 不作为抽取边界**——边界就是「起点 + 275 帧」；**已完整的样本默认跳过**，要覆盖必须显式加 `--overwrite`。

### 4.2 运动分层：把每一步的参数都对一遍

论文的最终协议**只用未来目标段 201–275 帧**来算运动分数（因为分层要刻画的是「未来有多难预测」）。整条链路的参数都是可核的：

| 环节 | 设置 |
|---|---|
| 源帧率 | 30000/1001 |
| 时间降采样 | 每 3 帧取 1（得 25 帧、24 个相邻帧对） |
| 缩放 | 854×480，area 插值 |
| 光流 | 灰度 Farnebäck 稠密光流，OpenCV 参数 `(0.5, 3, 15, 3, 5, 1.2, 0)` |
| 帧对统计量 | 光流幅值的**空间 75 分位** |
| 片段分数 | 24 个帧对统计量的**均值** |
| 分档 | 全局 **33 / 66 分位**切成 Low / Medium / High |

```bash
python -m motion_stratification.compute_motion_strata \
  --frames-root /path/to/MotionScape_reconstructed/full_frames \
  --manifest-json /path/to/MotionScape/source_manifest.json \
  --output-json /path/to/dynamicity_buckets_detailed.json
```

最终 228 段的分档是 **Low 75 / Medium 75 / High 78**。

> 一个反直觉的地方：降采样到「每 3 帧取 1」后，等效帧率只有约 9.99 FPS。**分层的分辨率与帧率都比评测时低得多**——这是刻意的，运动强度只需要一个相对排序，不需要精细的流场。

---

## 5. 语义标注与生成提示词

### 5.1 标注怎么生成

标注由 `prompts/semantic_annotation.txt` 驱动，经 OpenAI 兼容的 API 生成：

```bash
export ANNOTATION_API_KEY='...'
python annotation/generate_semantic_annotations.py \
  --input-path /path/to/MotionScape_reconstructed/full_frames \
  --mode prediction \
  --dynamicity-json /path/to/MotionScape/dynamicity_buckets.json \
  --prompt-file prompts/semantic_annotation.txt \
  --output-dir /path/to/new_annotations \
  --response-json
```

两个必须照做的细节：**时序不能乱**——条件帧 196–200 先发，随后才是抽样的未来帧 201–275；**温度固定 `0.1`**。196–200 这五帧**只作时序上下文**，提示词要求模型只描述 201–275。

### 5.2 四套生成提示词

| 文件 | 条件 |
|---|---|
| `prompts/text2world.txt` | 纯文本 |
| `prompts/image2world.txt` | 单帧图像 |
| `prompts/video2world_task_level.txt` | 只看**固定任务提示词 + 观测视频**，不读任何逐样本标注 |
| `prompts/video2world_clip_level.txt` | `weather + environment + caption` 按此顺序拼接 |

`clip_assisted_inspection.txt` 记的是**预处理阶段的负面提示词**（字幕文字、标题/logo/片头、转场特效、第三人称无人机画面四类），用来在选片阶段剔掉非目标片段，**不喂给生成模型**。

评测里还有一个口径要分清：**CLIPSim 只用 `caption`**，因为它是评测输入而不是生成提示词。

---

## 6. 基线推理与评测

### 6.1 五个基线

| 基线 | 入口 |
|---|---|
| Cosmos-Predict2.5-2B | `inference/cosmos/` 的补丁与资产 |
| CogVideoX1.5-5B / I2V | `inference/cogvideox/t2v.py` · `i2v.py` |
| Wan2.2 T2V / I2V（A14B） | `inference/wan/t2v.py` · `i2v.py` |
| LongCat-Video | `inference/longcat/v2w.py` |
| MAGI-1-24B | `inference/magi/v2w.py` |

各模型用的条件帧数与原生帧率都不一样，论文的评测按各自原生时序配置跑：

| 模型 / 任务 | 条件帧数 | 原生 FPS | 输出帧数 |
|---|---:|---:|---:|
| Cosmos-Predict2.5-2B（T2W / I2W / V2W） | 0 / 1 / 5 | 30 | 93 / 92 / 88 |
| CogVideoX1.5-5B（T2W / I2W） | 0 / 1 | 16 | 41 |
| Wan2.2-A14B（T2W / I2W） | 0 / 1 | 16 | 40 |
| LongCat-Video（V2W） | 13 | 15 | 40 |
| MAGI-1-24B（V2W） | 32 | 16 | 40 |

权重要自己下（仓库不分发第三方权重）：

```bash
python -m venv .venv-download && source .venv-download/bin/activate
pip install -r inference/requirements-download.txt
python inference/download_model_weights.py --output-root /data/MotionScape_models --model all --dry-run
```

Cosmos 权重是 **gated** 的，得先在 Hugging Face 上接受 NVIDIA 许可再登录。

> **那条 `huggingface-hub==0.36.0` 的钉版本只属于下载环境**，不要拿它去替换任何基线的推理环境——README 专门警告过。

### 6.2 指标与预处理

最终评测器是 `evaluation/evaluate_video_metrics.py`：

```bash
python evaluation/evaluate_video_metrics.py \
  --gt-dir /path/to/MotionScape_reconstructed/last_75_videos \
  --pred-dir /path/to/model_outputs \
  --dynamicity-buckets-json /path/to/MotionScape/dynamicity_buckets.json \
  --clipsim-caption-dir /path/to/MotionScape/annotations \
  --clipsim-caption-field caption \
  --clipsim-model openai/clip-vit-base-patch32 \
  --output-json /path/to/metrics_per_clip.json \
  --output-dynamicity-json /path/to/metrics_by_dynamicity.json \
  --frame-alignment timestamp --frame-metric-size 704x1280 \
  --fvd-mode Video2World --model-name MODEL_NAME \
  --i3d-python .venv-i3d/bin/python
```

六项指标的口径**不是随便取的**，每一条都会影响可比性：

- **PSNR / SSIM / LPIPS / Warping Error** 在各模型**原生帧率**上做时间戳对齐后按片段算；GT 与预测的 RGB 帧**各自独立**缩放到 **1280×704**；PSNR/SSIM/Warping Error 用 `[0,1]` 区间，**LPIPS 用 `[-1,1]`**。
- **Warping Error** 先用相邻 GT 帧估计光流，把前一张预测帧 warp 过来，再对有效像素取平均。
- **CLIPSim** 固定取最多 75 个连续可用预测帧。**运动分层只用于聚合，不改动 CLIPSim 的帧选择。**
- **FVD 是分布级的**：在每个运动档内，把所有片段的 I3D 嵌入**合起来比一次**，得到 Low / Medium / High 各一个 FVD——**不是**片段级 FVD 的平均。FVD 在 GT/预测对的公共时长上均匀采 75 个时间戳，转成 224×224 与 `[-1,1]`，每段抽一个 DeepMind I3D Mean 嵌入。

FVD 需要**独立的 TensorFlow 环境**（官方 TF-Hub I3D 实现与主环境冲突）：

```bash
python -m venv .venv-i3d
.venv-i3d/bin/pip install -r evaluation/requirements-i3d-fvd.txt
```

### 6.3 环境别互相污染

| 环境 | PyTorch | Transformers | Diffusers |
|---|---|---|---|
| Cosmos-Predict2.5-2B | 2.7.1+cu128 | 4.57.1 | 0.35.2 |
| CogVideoX1.5-5B / I2V | 2.8.0+cu128 | 5.13.1 | 0.39.0 |
| Wan2.2 T2V / I2V | 2.8.0+cu128 | 4.51.3 | 0.39.0 |
| LongCat-Video | 2.8.0+cu128 | 4.41.0 | 0.35.1 |
| 帧指标 / CLIPSim | 2.8.0+cu128 | 5.13.1 | N/A |

**MAGI 自带一整套独立运行时**（Python 3.10.20、PyTorch 2.11.0+cu130、FlashAttention/FlashInfer、CUDA 13.0/Blackwell），README 明确要求**不要与上面任何一个环境合并**。统一的基线容器说明在 `inference/docker/README.md`。

> 对 8 GB 显存：这五个基线一个都放不下。**本篇能在这类机器上跑完的只有 §2–§4**（重建 + 分层），语义标注走 API 不需要本地 GPU。生成质量部分只能引论文数字，不能自己出数。

---

## 7. 常见问题与解决方案

### Q1: 重建时某些样本的源视频抓不下来

源被下架、换区、或变成会员可见都会导致 yt-dlp 失败。先 `--dry-run` 看清单，再用 `--cookies-from-browser` 或 `--proxy` 重试，`--limit` 可以先小批量试通。**发布方不分发视频，这个风险无法绕过**——抓不到的样本就重建不出来，评测时要按实际重建成功的样本数报告。

### Q2: 下载器报「记录的流不可用」

这是**设计如此**：`download_sources.py` 按 manifest 的 `resolution / fps / codec` 精确选流，找不到就报错，**不会静默换一个编码**。因为编码变了会让重建结果的画质基线漂移。要放宽只能改 manifest 或接受缺样本，不要绕过去改脚本。

### Q3: `ffmpeg: command not found`

重建依赖系统级的 `ffmpeg` 与 `ffprobe`，pip 装的 Python 包不能替代：

```bash
ffmpeg -version && ffprobe -version    # 两个都要有
```

### Q4: FVD 装不上 / 与主环境冲突

官方 TF-Hub I3D 实现自带一套 TensorFlow 依赖，**必须单独建环境**（`evaluation/requirements-i3d-fvd.txt`），并用 `--i3d-python` 把那个解释器路径传给评测脚本。官方权重不是 HF checkpoint，而是从 TF-Hub 解析 `deepmind/i3d-kinetics-400/1`，评测器会校验固定的聚合 SHA-256。

### Q5: 想用 8 GB 显存跑一个基线

跑不动。最小的是 Cosmos-Predict2.5-**2B**（且显存需求远大于参数量本身，还要扛 93 帧的时间维），最大的是 MAGI-**24B**。可行的替代是：用你自己的小模型生成预测视频，**只要帧数对得上、能落到 `--pred-dir`**，就能接进这套评测协议——这也是把 MotionScape 当基准用的意义：**协议与模型解耦**。

### Q6: CLIPSim 用哪段文本？

只用 `caption`，不用 `weather + environment`。生成侧的 clip-level Video2World 提示词要拼 `weather + environment + caption`，但 **CLIPSim 是评测输入，口径就窄一档**。两者别混。

---

## 参考资源

- MotionScape GitHub: https://github.com/Thelegendzz/MotionScape
- 数据集（HF）: https://huggingface.co/datasets/thelegendzz/MotionScape
- Zenodo 记录: https://doi.org/10.5281/zenodo.21954342
- 论文: https://arxiv.org/abs/2604.07991
- yt-dlp: https://github.com/yt-dlp/yt-dlp
- TensorFlow Hub I3D (Kinetics-400): https://tfhub.dev/deepmind/i3d-kinetics-400/1

## 延伸阅读

- [关键数据集与基准](../02-世界模型专题/06-关键数据集与基准.md) — MotionScape 的理论背景
- [生成式世界模型](../02-世界模型专题/02-生成式世界模型.md) — 视频生成世界模型详解
- [可复现项目候选清单](./08-可复现项目候选清单.md) — 更多可复现项目

## 思考题

1. **定位判断**：228 段、34.9 分钟、无位姿真值。如果给你一个自研的无人机世界模型，MotionScape 能用来做什么、不能用来做什么？分别说出理由。

2. **分层的分辨率**：运动分层把帧缩到 854×480、每 3 帧取 1，等效约 9.99 FPS。评测时的图像处理却是 1280×704、各模型原生帧率。**为什么两处的分辨率差这么多还能用同一套分层结果？**

3. **步骤归属**：仓库里没有 `scripts/preprocess.py`。按上面给出的真实仓库结构，降低分辨率这个需求应该落在哪一步、由谁负责？

4. **FVD 的口径**：为什么 FVD 要在每个运动档内把片段嵌入**合起来比一次**，而不是算每个片段的 FVD 再平均？这两种算法在什么情况下会给出一致的结论？

5. **条件帧的差异**：LongCat 用 13 个条件帧、MAGI 用 32 个，Cosmos 的 V2W 只用 5 个。如果想让三者在同一条件下比，需要改动什么？改了之后还算不算「各自原生配置的报告」？

<details><summary>参考答案</summary>

1. **能做的**：当**评测集**。它切好了「200 帧观测 + 75 帧未来」的固定格式，条件可文本/图像/视频，还有统一的指标实现与运动分层聚合——这正是「考卷」的三要素：题目固定、条件可控、评分口径统一。**不能做的**：当**训练集**。228 段 34.9 分钟的量级不足以训一个世界模型；也**不能评定位/轨迹精度**，因为没有位姿真值，只有视频帧与语义标注。所以「用 MotionScape 训练一个视频预测模型，再用它评定位误差」这条路两头都走不通。

2. **因为分层的产物是一个相对排序，不是一个精确量。** 运动分档只需要「哪段比哪段更动」，用低分辨率、低帧率算出来的 75 分位光流均值足以稳定地排出这个次序——相邻帧的运动趋势在缩到 854×480 后依然保留。而评测阶段要的是**像素级可比性**，所以必须按各模型原生帧率做时间对齐、按统一尺寸 1280×704 缩放。**两者用的是分层结果的「档位标签」，不是「分数数值」**，所以口径不一致不影响。反过来说：如果下游有人直接把 `dynamicity_score` 当物理速度用，那就错了——README 明确说它是图像空间里的代理量。

3. 落在**重建脚本里，由源分辨率与评测侧缩放共同决定，不存在一个独立的「预处理降分辨率」步骤**。真实链路是：`download_sources.py` 按 manifest 记录的原始分辨率（多为 3840×2160）抓源视频，`reconstruct_benchmark.py` 只做**时间**上的重采样与切帧（不降空间分辨率，输出仍是源分辨率），空间缩放发生在**用到的时候**——运动分层缩到 854×480，帧指标缩到 1280×704（GT 与预测各自独立缩）。所以「720p 副本」这个需求在本仓库里没有对应的入口；真要省存储，是在抓源视频那一步用 `--extra-arg` 之类的手段换流，代价是让重建结果偏离 manifest 记录的口径。

4. 因为 **FVD 是分布间的距离**（这里用 Fréchet 距离），不是逐样本的误差。它比较的是**两组嵌入的均值与协方差**，单看一个片段根本算不出 FVD——片段级根本没有「分布」。在一个档位内把 75 段左右的嵌入合起来比一次，得到的是「这个模型在这个运动强度档上的预测分布与真实分布有多远」，这正是想报的量。两种算法只有在**每个片段自成一个档**的退化情形下才会趋于一致，或者当所有片段的嵌入分布足够接近时，平均值与合并值约等于——实务上没人会这么用。

5. LongCat 与 MAGI 用的是 Video2World 的**逐帧条件**（每 2 帧取 1，得 13 / 32 帧），Cosmos 的 V2W 只给 5 帧连续条件，三者原生口径本来就不同。要对齐，得**从同一段 200 帧前缀里按统一规则抽条件帧**（比如统一抽最后 5 帧或等间隔 13 帧），再把抽好的条件喂给每个适配器——这是改 `inference/*/v2w.py` 的输入，不是改评测器。**改完就不再是「各自原生配置」的报告了**：论文那张表的意义正是「大家都按自己最擅长的配置跑，看谁在同一个考卷上分高」；统一条件后报的是「同一条件下谁更强」，两个结论都成立，但**必须在报告里写清是哪一种**，混着比就不可比。

</details>
