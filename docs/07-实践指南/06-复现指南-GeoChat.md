# 06 - 复现指南：GeoChat — 遥感视觉语言模型

> **预计阅读：12 分钟 | 前置知识：Hugging Face Transformers 使用经验、视觉语言模型基本概念、LoRA/PEFT 微调基础**

本指南帮助你从零开始复现 GeoChat 项目。GeoChat 是由 MBZUAI Oryx Lab 提出的遥感领域视觉语言模型（VLM），支持多轮对话、区域级理解和图像级描述等任务。

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
