# 10 - 复现指南：RemoteCLIP — 遥感视觉语言基础模型

> **预计阅读：12 分钟 | 前置知识：PyTorch 基础、CLIP 模型概念、遥感图像基础**

本指南帮助你复现 RemoteCLIP 项目。RemoteCLIP 是将 CLIP 针对遥感/航拍图像做领域适配的视觉语言基础模型，支持零样本分类、跨模态检索等任务。

---

## 项目背景

| 项目 | 信息 |
|------|------|
| **GitHub** | [ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) |
| **Stars** | ~550 |
| **论文** | "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing"（IEEE TGRS） |
| **核心思路** | CLIP 遥感适配 → 零样本航拍理解 |
| **与本项目关系** | 无人机视觉感知的预训练骨干，与 GeoChat 形成互补 |

---

## 环境要求

- **操作系统**：Ubuntu 20.04/22.04 或 Windows
- **Python**：3.9+
- **PyTorch**：2.0+
- **GPU**：推荐 8GB+ VRAM（推理），16GB+（微调）
- **依赖**：transformers, timm, open_clip

---

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/ChenDelong1999/RemoteCLIP.git
cd RemoteCLIP

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载预训练模型
# 模型权重托管在 Hugging Face，运行时会自动下载
# 或手动下载: https://huggingface.co/ChenDelong1999/RemoteCLIP
```

---

## 预训练模型

| 模型 | 参数量 | 推荐场景 |
|------|--------|---------|
| RemoteCLIP-ViT-B/32 | ~150M | 轻量推理，CPU 可跑 |
| RemoteCLIP-ViT-B/14 | ~300M | 平衡性能和速度 |
| RemoteCLIP-ViT-L/14 | ~600M | 最佳性能，需 GPU 8GB+ |

---

## 运行推理

### 零样本图像分类

```python
import torch
from PIL import Image
from models.remoteclip import RemoteCLIP

# 加载模型
model = RemoteCLIP.from_pretrained("ChenDelong1999/RemoteCLIP-ViT-B-32")
model.eval()

# 加载图像
image = Image.open("your_aerial_image.jpg")

# 定义候选标签
labels = ["farmland", "urban area", "water body", "forest", "airport"]

# 零样本分类
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(labels)
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    predicted_label = labels[similarity.argmax()]
    print(f"预测类别: {predicted_label}")
```

### 跨模态检索

```python
# 用文本检索图像
query = "a runway with airplanes"
# 编码查询文本
text_features = model.encode_text([query])
# 与图像库计算相似度
similarities = text_features @ all_image_features.T
top_k_indices = similarities.topk(5).indices
```

---

## 复现 Checklist

- [ ] 克隆仓库并安装依赖
- [ ] 下载预训练模型（ViT-B/32 先试）
- [ ] 运行零样本分类 demo：在航拍图片上测试
- [ ] 测试跨模态检索：用文本描述检索航拍图片
- [ ] 对比 RemoteCLIP vs 原始 CLIP 在航拍上的效果差异
- [ ] （可选）在自己的无人机数据上微调

---

## 与 GeoChat 的对比

| 维度 | RemoteCLIP | GeoChat |
|------|-----------|---------|
| **任务** | 分类、检索 | 对话、VQA、描述 |
| **架构** | CLIP（对比学习） | LLaVA（生成式） |
| **推理速度** | 快（毫秒级） | 慢（秒级） |
| **适用场景** | 批量标注、检索 | 交互式理解 |
| **推荐用途** | 数据预处理、特征提取 | 任务推理、报告生成 |

---

## 参考资源

- RemoteCLIP GitHub: https://github.com/ChenDelong1999/RemoteCLIP
- 论文: https://arxiv.org/abs/2306.07location
- HuggingFace 模型: https://huggingface.co/ChenDelong1999/RemoteCLIP

## 延伸阅读

- [什么是VLM](../01-基础概念/02-什么是VLM.md) — 理解视觉语言模型基础
- [遥感VLM](../04-VLM专题/01-遥感VLM.md) — GeoChat 等遥感 VLM 详解
- [无人机场景理解](../04-VLM专题/02-无人机场景理解.md) — 无人机 VLM 评估基准
