# 10 - 复现指南：RemoteCLIP — 遥感视觉语言基础模型

> **预计阅读：12 分钟 | 前置知识：PyTorch 基础、CLIP 模型概念、遥感图像基础**

RemoteCLIP 是将 CLIP 针对遥感/航拍图像做领域适配的视觉语言基础模型，支持零样本分类、跨模态检索等任务。

---

## 项目背景

| 项目 | 信息 |
|------|------|
| **GitHub** | [ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) |
| **Stars** | ~599 |
| **论文** | "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing"（IEEE TGRS，2024-04 接收） |
| **核心思路** | CLIP 遥感适配 → 零样本航拍理解 |
| **与本项目关系** | 无人机视觉感知的预训练骨干，与 GeoChat 形成互补 |

---

## 环境要求

- **操作系统**：Ubuntu 20.04/22.04 或 Windows
- **Python**：3.9+
- **PyTorch**：2.0+
- **GPU**：推荐 8GB+ VRAM（推理），16GB+（微调）
- **依赖**：核心只有 **`open-clip-torch`**；下权重的 `huggingface_hub` 通常随它一起装上

---

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/ChenDelong1999/RemoteCLIP.git
cd RemoteCLIP

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖（README 给的唯一一条）
pip install open-clip-torch

# 4. 下载预训练模型权重
```

```python
from huggingface_hub import hf_hub_download

for model_name in ['RN50', 'ViT-B-32', 'ViT-L-14']:
    path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt",
                           cache_dir='checkpoints')
    print(f'{model_name} -> {path}')
```

---

## 预训练模型

三个权重都在 `chendelong/RemoteCLIP` 这一个 HF 仓库里，参数量按 checkpoint 文件大小（fp32）反推：

| 权重文件 | 基础架构 | 参数量 | 推荐场景 |
|---|---|---|---|
| `RemoteCLIP-RN50.pt` | ResNet-50 | **102M**（408 MB） | 最轻，CPU 也能跑 |
| `RemoteCLIP-ViT-B-32.pt` | ViT-B/32 | **151M**（605 MB） | 平衡，第一次跑通用它 |
| `RemoteCLIP-ViT-L-14.pt` | ViT-L/14 | **428M**（1.71 GB） | 性能最好，fp16 约 0.9 GB |

---

## 运行推理

### 零样本图像分类

```python
import torch, open_clip
from PIL import Image

model_name = 'ViT-B-32'          # 'RN50' 或 'ViT-B-32' 或 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

# 载入 RemoteCLIP 权重
ckpt = torch.load(f"checkpoints/RemoteCLIP-{model_name}.pt", map_location="cpu")
print(model.load_state_dict(ckpt))
model = model.cuda().eval()

# 定义候选标签，走同一套 tokenizer
labels = ["farmland", "urban area", "water body", "forest", "airport"]
text = tokenizer(labels)
image = preprocess(Image.open("your_aerial_image.jpg")).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image.cuda())
    text_features = model.encode_text(text.cuda())
    # CLIP 的相似度要先各自归一化，再乘 100 这个 logit scale
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

print(labels[int(probs.argmax())], float(probs.max()))
```

### 跨模态检索

```python
# 文本到图像的检索：图像库一侧的特征可以离线算好
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image_batch.cuda())
    image_features /= image_features.norm(dim=-1, keepdim=True)   # 归一化后缓存

query = ["a runway with airplanes"]
with torch.no_grad():
    text_features = model.encode_text(tokenizer(query).cuda())
    text_features /= text_features.norm(dim=-1, keepdim=True)

similarities = (text_features @ image_features.T)   # 两边都已归一化，就是余弦相似度
top_k = similarities.topk(5).indices
```

---

## 复现 Checklist

- [ ] 克隆仓库并安装依赖
- [ ] 下载预训练模型（`ViT-B-32` 先试）
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
- 论文: https://arxiv.org/abs/2306.11029
- HuggingFace 权重: https://huggingface.co/chendelong/RemoteCLIP
- HuggingFace 数据集（训练用的遥感图文对）: https://huggingface.co/datasets/gzqy1026/RemoteCLIP
- 仓库内的检索脚本: [retrieval.py](https://github.com/ChenDelong1999/RemoteCLIP/blob/main/retrieval.py)

## 延伸阅读

- [什么是VLM](../01-基础概念/02-什么是VLM.md) — 理解视觉语言模型基础
- [遥感VLM](../04-VLM专题/01-遥感VLM.md) — GeoChat 等遥感 VLM 详解
- [无人机场景理解](../04-VLM专题/02-无人机场景理解.md) — 无人机 VLM 评估基准

## 思考题

1. **选型判断**：RN50、ViT-B-32、ViT-L-14 三个权重分别适合什么情况？第一次跑通该先拿哪个？

2. **显存规划**：只有 8 GB 显存，想先跑批量航拍图检索、之后再微调，这个计划成立吗？

3. **架构对比**：RemoteCLIP 与 GeoChat 的架构、任务和推理速度差在哪？为什么说二者互补而不是二选一？

4. **实验设计**：Checklist 里"对比 RemoteCLIP vs 原始 CLIP 在航拍上的效果差异"，怎么设计才算公平？

5. **检索实现**：用文本检索图像库时，哪部分计算可以提前离线做掉？

<details><summary>参考答案</summary>

1. 按参数量和场景选：RN50 约 102M，最轻，CPU 也能跑；ViT-B-32 约 151M，兼顾性能与速度，是默认选择；ViT-L-14 约 428M，性能最好，但在 8 GB 卡上要转 fp16 才宽裕。Checklist 也建议第一步先用 `ViT-B-32` 试。（注意是 **B-32 而不是 B/14**——`ViT-B/14` 这个权重不存在。）

2. 要拆开看：文档写的是"推荐 8 GB+ VRAM（推理），16 GB+（微调）"，所以 8 GB 卡上做推理在推荐范围内——哪怕 ViT-L-14（fp32 权重 1.71 GB，fp16 约 0.9 GB）也放得下，只是批量要开小些；微调则低于推荐配置。折中做法是先用 `ViT-B-32` 或 `RN50` 把推理与检索跑通，微调留到 16 GB+ 的机器上——Checklist 本来也把"在自己的无人机数据上微调"列成可选项。

3. 架构上 RemoteCLIP 是 CLIP 式对比学习，GeoChat 是 LLaVA 式生成式；任务上前者做分类与检索，后者做对话、VQA、描述。推理速度差一个量级（毫秒级 vs 秒级），所以典型分工是 RemoteCLIP 承担数据预处理与特征提取、GeoChat 承担交互式理解与报告生成，一个管批量、一个管交互。

4. 固定除权重以外的一切：同一批航拍图、同一组候选标签（如 farmland / urban area / water body / forest / airport）、同一套 `preprocess` 与 `tokenizer`，走同一段零样本分类流程——`encode_image` 出图像特征、`encode_text` 出标签特征，两边各自 L2 归一化后乘 100 的 logit scale，softmax 后取 argmax——只替换 `load_state_dict` 载入的权重。这样差异才能归因到遥感适配，而不是标签集或流程的变化。特别要注意两个模型必须用**各自的** `create_model_and_transforms(model_name)` 得到的预处理与 tokenizer：三个权重的输入分辨率不同，混用预处理会直接把对比污染掉。

5. 图像库那一侧的编码可以离线算好并缓存：文档的检索代码只在线编码查询文本，再与已缓存的图像特征矩阵做矩阵乘并 `topk(5)`。把整个图像库的 `encode_image` 结果（**归一化之后**）预先算成矩阵，查询时只编码一条文本，在线成本就只剩一次矩阵乘。归一化必须放在缓存之前，否则缓存下来的是未归一化特征，余弦相似度会算错。

</details>
