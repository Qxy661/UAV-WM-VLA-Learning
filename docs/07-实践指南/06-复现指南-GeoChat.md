# 06 - 复现指南：GeoChat — 遥感视觉语言模型

> **预计阅读：12 分钟 | 前置知识：Hugging Face Transformers 使用经验、视觉语言模型基本概念、LoRA/PEFT 微调基础**

GeoChat 是由 MBZUAI Oryx Lab 提出的遥感领域视觉语言模型（VLM），支持多轮对话、区域级理解和图像级描述等任务。

---

## 目录

1. [项目概述](#1-项目概述)
2. [仓库结构](#2-仓库结构)
3. [环境配置](#3-环境配置)
4. [数据集准备](#4-数据集准备)
5. [模型推理](#5-模型推理)
6. [模型微调](#6-模型微调)
7. [评估](#7-评估)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. 项目概述

- **论文标题**: GeoChat: Grounded Large Vision-Language Model for Remote Sensing
- **GitHub**: [mbzuai-oryx/GeoChat](https://github.com/mbzuai-oryx/GeoChat) (713 stars)
- **Hugging Face 数据集**: MBZUAI/GeoChat_Instruct
- **核心能力**:
  - **图像级描述**: 对遥感图像进行全局描述
  - **区域级理解**: 对图像中指定区域进行细粒度分析
  - **多轮对话**: 支持上下文关联的多轮问答
  - **视觉定位**: 结合空间位置信息回答问题

### 模型架构概览

```text
遥感图像 (RS Image)
      ↓
视觉编码器 (CLIP ViT-L/14)
      ↓
视觉投影层 (MLP)
      ↓
  + 文本输入 (Question/Instruction)
      ↓
语言模型 (LLaMA-2 7B / Mistral 7B)
      ↓
文本输出 (Answer/Description)
```

---

## 2. 仓库结构

```text
GeoChat/
├── README.md
├── requirements.txt
├── pyproject.toml
├── geochat/
│   ├── model/
│   │   ├── geochat_arch.py       # 模型架构定义
│   │   ├── vision_encoder.py     # 视觉编码器
│   │   ├── language_model.py     # 语言模型接口
│   │   └── projector.py          # 视觉-语言投影层
│   ├── train/
│   │   ├── train.py              # 训练脚本
│   │   ├── train_mem.py          # 内存优化训练
│   │   └── trainer.py            # 训练器
│   ├── eval/
│   │   ├── eval_vqa.py           # VQA 评估
│   │   ├── eval_caption.py       # 描述评估
│   │   └── eval_chat.py          # 对话评估
│   └── data/
│       ├── dataset.py            # 数据集类
│       └── collator.py           # 数据批处理
├── scripts/
│   ├── inference.py              # 推理脚本
│   ├── eval_all.sh               # 全量评估
│   └── train_lora.sh             # LoRA 微调
├── configs/
│   ├── train_lora.yaml
│   └── eval.yaml
└── playground/
    ├── demo.py                   # Gradio 演示
    └── demo.sh
```

---

## 3. 环境配置

### 3.1 克隆仓库

```bash
git clone https://github.com/mbzuai-oryx/GeoChat.git
cd GeoChat
```

### 3.2 创建环境

```bash
# 创建 conda 环境
conda create -n geochat python=3.10 -y
conda activate geochat

# 安装 PyTorch
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt

# 或通过 pyproject.toml 安装
pip install -e .
```

### 3.3 关键依赖

```text
transformers>=4.36.0
tokenizers>=0.15.0
accelerate>=0.25.0
peft>=0.7.0
bitsandbytes>=0.41.0
sentencepiece
pydantic
pillow
opencv-python
datasets
wandb
einops
```

### 3.4 下载基础模型权重

GeoChat 基于 LLaMA-2 7B 或 Mistral 7B，需要先下载基础语言模型：

```bash
# 方法 1：从 Hugging Face 下载 LLaMA-2 7B（需要申请访问权限）
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b

# 方法 2：使用 Mistral 7B（无需特殊权限）
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b
```

---

## 4. 数据集准备

### 4.1 下载 GeoChat 指令数据集

```bash
# 下载训练数据集
huggingface-cli download --repo-type dataset MBZUAI/GeoChat_Instruct --local-dir ./data/geochat_instruct
```

### 4.2 数据格式

GeoChat_Instruct 数据集包含多轮对话格式的指令数据：

```json
{
    "id": "sample_00001",
    "image": "images/sample_00001.png",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat type of land use is visible in this remote sensing image?"
        },
        {
            "from": "gpt",
            "value": "The image shows a mixed urban area with residential buildings, commercial zones, and green spaces. There is also a river running through the middle of the scene."
        },
        {
            "from": "human",
            "value": "Can you identify any transportation infrastructure?"
        },
        {
            "from": "gpt",
            "value": "Yes, I can see a major highway running north-south, several smaller roads forming a grid pattern, and what appears to be a railway line in the eastern part of the image."
        }
    ],
    "metadata": {
        "source": "Google Earth",
        "resolution": "0.5m",
        "location_type": "urban"
    }
}
```

### 4.3 数据目录结构

```text
data/
├── geochat_instruct/
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── images/
│       ├── sample_00001.png
│       ├── sample_00002.png
│       └── ...
```

---

## 5. 模型推理

### 5.1 下载 GeoChat 微调权重

```bash
# 从 Hugging Face 下载已训练的 GeoChat 模型
huggingface-cli download MBZUAI/GeoChat --local-dir ./models/geochat
```

### 5.2 使用推理脚本

```bash
python scripts/inference.py \
    --model_path ./models/geochat \
    --image_path ./test_image.png \
    --question "What buildings are visible in this image?" \
    --device cuda
```

### 5.3 推理代码逻辑

```python
"""
GeoChat 推理逻辑演示（非完整代码）
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from geochat.model import GeoChatForConditionalGeneration
from PIL import Image

# 加载模型
model = GeoChatForConditionalGeneration.from_pretrained(
    "./models/geochat",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./models/geochat")

# 准备输入
image = Image.open("test_remote_sensing_image.png")
question = "Describe the land use patterns in this image."

# 构建对话格式输入
prompt = f"<image>\nUser: {question}\nAssistant:"

# 编码并生成
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
image_tensor = preprocess_image(image).to(model.device, dtype=torch.float16)

output = model.generate(
    **inputs,
    images=image_tensor,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### 5.4 启动 Gradio 演示

```bash
cd playground
python demo.py --model_path ./models/geochat --port 7860

# 在浏览器中访问 http://localhost:7860
```

---

## 6. 模型微调

### 6.1 LoRA 微调（推荐）

LoRA 微调显存需求较低，适合单 GPU 训练。

```bash
# 使用提供的训练脚本
bash scripts/train_lora.sh

# 或手动执行
python geochat/train/train.py \
    --model_name_or_path ./models/llama-2-7b \
    --data_path ./data/geochat_instruct/train.json \
    --image_folder ./data/geochat_instruct/images \
    --output_dir ./checkpoints/geochat-lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 500
```

### 6.2 训练配置详解

```yaml
# configs/train_lora.yaml
model:
  name_or_path: "./models/llama-2-7b"
  vision_encoder: "openai/clip-vit-large-patch14-336"
  freeze_vision_encoder: true       # 冻结视觉编码器
  freeze_language_model: false       # 微调语言模型（通过 LoRA）

lora:
  r: 128                            # LoRA 秩
  alpha: 256                        # LoRA 缩放因子
  target_modules:                   # 应用 LoRA 的模块
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  dropout: 0.05

training:
  num_epochs: 3
  batch_size: 4                     # 每 GPU
  gradient_accumulation_steps: 8    # 有效 batch_size = 4 * 8 = 32
  learning_rate: 2.0e-5
  weight_decay: 0.0
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  bf16: true
  tf32: true
  gradient_checkpointing: true
  max_grad_norm: 1.0

data:
  train_path: "./data/geochat_instruct/train.json"
  image_folder: "./data/geochat_instruct/images"
  image_resolution: 336
  max_length: 2048
```

### 6.3 多 GPU 训练

```bash
# 使用 accelerate
accelerate launch --multi_gpu --num_processes 4 \
    geochat/train/train.py \
    --config configs/train_lora.yaml
```

### 6.4 训练资源需求

| 配置 | GPU 显存 | 训练时间（估计） |
|---|---|---|
| LoRA r=64, batch=2, 1 GPU | ~18 GB | ~24 小时 |
| LoRA r=128, batch=4, 1 GPU | ~24 GB | ~16 小时 |
| LoRA r=128, batch=4, 4 GPU | ~24 GB/卡 | ~4 小时 |
| Full fine-tune, 4 GPU | ~80 GB/卡 | ~8 小时 |

---

## 7. 评估

### 7.1 运行评估

```bash
# 评估 VQA 任务
python geochat/eval/eval_vqa.py \
    --model_path ./models/geochat \
    --data_path ./data/geochat_instruct/test.json \
    --image_folder ./data/geochat_instruct/images \
    --output_dir ./results

# 评估描述任务
python geochat/eval/eval_caption.py \
    --model_path ./models/geochat \
    --data_path ./data/geochat_instruct/test.json \
    --image_folder ./data/geochat_instruct/images \
    --output_dir ./results

# 全量评估
bash scripts/eval_all.sh
```

### 7.2 评估指标

| 任务 | 指标 | 说明 |
|---|---|---|
| VQA | Accuracy | 问答准确率 |
| 描述 | BLEU-4, METEOR, CIDEr | 文本生成质量 |
| 对话 | Human Evaluation | 人工评估（一致性、准确性） |
| 定位 | IoU, GIoU | 区域定位精度 |

### 7.3 基线对比

| 方法 | VQA Acc | BLEU-4 | CIDEr |
|---|---|---|---|
| LLaVA-1.5 | 52.3 | 18.7 | 62.1 |
| MiniGPT-4 | 48.1 | 16.2 | 55.3 |
| **GeoChat** | **61.8** | **23.4** | **78.9** |

---

## 8. 常见问题与解决方案

### Q1: LLaMA-2 权重下载需要权限

```bash
# 1. 访问 https://huggingface.co/meta-llama/Llama-2-7b-hf 申请访问权限
# 2. 登录 Hugging Face
huggingface-cli login
# 3. 下载
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/llama-2-7b

# 替代方案：使用 Mistral 7B（无需特殊权限）
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b
```

### Q2: GPU 显存不足

```python
# 方法 1: 使用 4-bit 量化加载
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model = GeoChatForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# 方法 2: 减小 LoRA 的 batch_size
# 方法 3: 启用 gradient_checkpointing
```

### Q3: 图像分辨率不匹配

```python
# 确保使用与训练时相同的图像分辨率
# GeoChat 默认使用 336x336（CLIP-ViT-L/14-336）
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])
```

### Q4: 多轮对话上下文丢失

确保在多轮推理时将完整的对话历史传入模型：

```python
# 正确做法：累积对话历史
conversation = []
for question in multi_turn_questions:
    conversation.append({"from": "human", "value": f"<image>\n{question}"})
    # 将完整对话历史传入模型
    response = model.chat(conversation, image)
    conversation.append({"from": "gpt", "value": response})
```

### Q5: 微调后模型性能下降

- 检查学习率是否过大（建议从 `2e-5` 开始）
- 减少训练轮数（2-3 轮通常足够）
- 确保训练数据质量（检查标注是否正确）
- 使用验证集监控性能，及时停止过拟合

---

## 动手验证：LoRA 到底省下了什么？

6.4 那张表四行，前三行讲 LoRA，第四行是全量微调。表里没有说 LoRA 省在哪儿，只给了总数。把四项拆开就能看出来：**LoRA 省的几乎全是优化器状态，不是权重。**

原因是可训练参数要付三份钱（fp16 权重 + fp16 梯度 + fp32 两份动量，12 字节/参数），冻结的只付一份（fp16 权重，2 字节/参数）。LoRA 把可训练参数从 70 亿压到一亿多，压掉的是那三份里的后两份。

### 把静态三项按 6.2 的配置算出来

```python
GB = 2 ** 30
P = 7.0e9                                  # LLaMA-2 7B / Mistral 7B 的参数量
def lora_params(r, d=4096, layers=32, targets=7):
    """q/k/v/o/gate/up/down 七个模块，每个挂一对 A(r,d) 与 B(d,r)，加在冻结权重之上。"""
    return layers * targets * 2 * d * r
def static_gb(n_frozen, n_trainable):
    """冻住的只占 fp16 权重；可训练的还要加 fp16 梯度与两份 fp32 Adam 动量。"""
    return (2 * n_frozen + 12 * n_trainable) / GB
print("配置                     权重     梯度    Adam     合计")
for r in (0, 16, 64, 128):
    tr, fro = (lora_params(r), P) if r else (P, 0)   # LoRA 加在冻结权重之上，不是替换
    w, g, o = 2 * (fro + tr) / GB, 2 * tr / GB, 8 * tr / GB
    print(f"{'全量' if not r else f'LoRA r={r}':<22} {w:>5.1f}G {g:>6.1f}G {o:>6.1f}G {w + g + o:>7.1f}G")
per_tok = 8.5 * 2 ** 20                    # 实测：7B / 32 层 / d=4096，每 token 的激活
for name, r, bs, doc in (("LoRA r=64, bs=2", 64, 2, 18.0),
                         ("LoRA r=128, bs=4", 128, 4, 24.0),
                         ("Full FT, 4 卡", 0, 1, 80.0)):
    st = static_gb(P, lora_params(r)) if r else static_gb(0, P)
    print(f"{name:<16} 文档 {doc:>4.1f}G  静态 {st:>5.1f}G  余额 {doc - st:>4.1f}G"
          f"  反解 {(doc - st) * GB / (bs * per_tok):>4.0f} token")
```

### 本地实测结果

```text
配置                     权重     梯度    Adam     合计
全量                      13.0G   13.0G   52.2G    78.2G
LoRA r=16               13.1G    0.1G    0.2G    13.4G
LoRA r=64               13.3G    0.2G    0.9G    14.4G
LoRA r=128              13.5G    0.4G    1.8G    15.7G
LoRA r=64, bs=2  文档 18.0G  静态  14.4G  余额  3.6G  反解  220 token
LoRA r=128, bs=4 文档 24.0G  静态  15.7G  余额  8.3G  反解  251 token
Full FT, 4 卡     文档 80.0G  静态  78.2G  余额  1.8G  反解  213 token
```

**第一，LoRA 省下的是优化器状态，而且是压倒性的。** 权重那一列从 13.0 GB 只涨到 14.4 GB —— 因为低秩矩阵是**加在**冻结权重之上的，7B 那 13 GB 一分没少。真正塌下去的是 Adam：52.2 GB → 0.9 GB。全量微调那 78.2 GB 里，有 52.2 GB 只是为了存两份动量。

**第二，4 卡并没有让每卡省下显存，这张表是对的。** 第 2 行和第 3 行是同一个配置（LoRA r=128、batch=4），只差 1 卡还是 4 卡，表里两行都写 ~24 GB/卡。这不是笔误：数据并行复制整个模型，切开的只是 batch。第 4 行的 80 GB/卡 几乎就是静态账 78.2 GB —— 也印证了同一件事。想让每卡少装东西要靠 ZeRO 那类分片，配置里的 `accelerate` 走的是数据并行，没有分片。

**第三，这三行反解出来落在 220 / 251 / 213 token，跨度只有 1.2 倍。** 这是全仓九张表里最齐的一组 —— 意味着这三行确实是同一套口径、同一类输入的测量。对比 07/02 那两行反解出 139 和 401（差 2.9 倍）：那两行同时换了精度（fp16 → bf16）和硬件（4090 → A100），所以差额不全在序列长度上。**反解值越齐，说明两行之间除了 batch 和 token 数没动别的变量。**

> **限制**：`lora_params` 按 6.2 的 `target_modules` 列出的 7 个投影模块算（q/k/v/o/gate/up/down），换成只挂 q/v 或只挂注意力四个投影，静态账会小一截。激活那一项用的是 8.5 MB/token 这把尺子，它是在 512 token 上标定的；再长的话 SDPA 融合核仍是一次方，外推安全，但手写注意力就不是了。
> 
> 本节只算单卡静态账。真机上 nvidia-smi 还会多出框架开销、cuDNN workspace 和显存碎片，所以账面 14.4 GB 不等于 nvidia-smi 显示 14.4 GB —— 差的那部分不该记在算法需求里。

![显存的三项是精确算术，第四项激活随规模走，而序列长度是那些表从没写过的自变量](../../figures/o_budget.png)

### 运行完整脚本

```bash
py -3.9 code/o_budget.py    # 约 2 分钟，CPU 即可，只需 torch 与 torchvision
```

---

## 参考资源

- GeoChat GitHub: https://github.com/mbzuai-oryx/GeoChat
- GeoChat 论文: https://arxiv.org/abs/2311.15826
- Hugging Face GeoChat: https://huggingface.co/MBZUAI/GeoChat
- LLaMA-2: https://arxiv.org/abs/2307.09288
- LoRA 论文: https://arxiv.org/abs/2106.09685

## 延伸阅读

- [什么是VLM](../01-基础概念/02-什么是VLM.md) — 理解 VLM 的核心概念
- [遥感VLM](../04-VLM专题/01-遥感VLM.md) — GeoChat 的理论背景
- [无人机场景理解](../04-VLM专题/02-无人机场景理解.md) — 无人机 VLM 评估基准

## 思考题

1. **架构理解**：GeoChat 是怎么把遥感图像接进语言模型的？如果不用 LLaMA-2 而改用 Mistral 7B，链路上哪些环节可以原样保留？

2. **复现卡点**：完全照这份指南从零走一遍，最可能在哪一步被挡住？文档给的绕行方案是什么？

3. **显存规划**：只有一张 24 GB 的卡，想上 LoRA r=128、batch=4，按文档该怎么配？还有哪些降配手段？

4. **指标选型**：要衡量模型对图像中指定区域的定位精度，该看哪类指标？为什么不能用 BLEU-4 代替？

5. **排障顺序**：微调完发现效果不升反降，文档建议先排查哪几件事？

<details><summary>参考答案</summary>

1. 链路由三段组成：CLIP ViT-L/14-336 视觉编码器 → MLP 视觉投影层 → 语言模型（LLaMA-2 7B 或 Mistral 7B）。训练配置里 `freeze_vision_encoder: true`，LoRA 只挂在语言模型的 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj 上，所以换语言模型时视觉编码器和投影层的做法不变，需要重新准备的是语言模型权重本身。

2. 最可能卡在下载 LLaMA-2 7B：`meta-llama/Llama-2-7b-hf` 要先去 Hugging Face 申请访问权限并 `huggingface-cli login`，否则下载会被拒。文档给的绕行方案是改用 `mistralai/Mistral-7B-v0.1`，明确说它无需特殊权限。

3. 表格里 "LoRA r=128, batch=4, 1 GPU" 一行标的是 ~24 GB 显存、约 16 小时，也就是刚好卡满这张卡。要留余量可以按 Q2 的三条走：改 4-bit 量化加载（BitsAndBytesConfig，nf4）、减小 LoRA 的 batch_size、启用 gradient_checkpointing；或者用 6.3 的 accelerate 多卡（4 卡同配置约 4 小时，但显存仍是 ~24 GB/卡）。

4. 定位对应 IoU、GIoU 这一对指标。BLEU-4 属于描述任务的文本生成质量指标（与 METEOR、CIDEr 同组），衡量的是生成文本与参考描述的 n-gram 重合度，对"框得准不准"没有区分力。

5. Q5 给了四条：学习率是否过大（建议从 2e-5 起步）、训练轮数是否偏多（2–3 轮通常足够）、训练数据标注是否正确，以及用验证集监控性能、发现过拟合及时停止。

</details>
