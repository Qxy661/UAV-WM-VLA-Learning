# 05 - 复现指南：MotionScape — 运动场景数据集使用指南

> **预计阅读：12 分钟 | 前置知识：Python 数据处理基础、视频处理基础、3D 坐标系与变换基本概念**

本指南帮助你下载、理解和使用 MotionScape 数据集。MotionScape 是一个面向无人机世界模型训练的大规模运动场景数据集，包含 4K 视频、6-DoF 轨迹和语言描述。

---

## 目录

1. [数据集概述](#1-数据集概述)
2. [仓库与下载](#2-仓库与下载)
3. [数据格式详解](#3-数据格式详解)
4. [数据加载与预处理](#4-数据加载与预处理)
5. [数据可视化](#5-数据可视化)
6. [与世界模型训练管线集成](#6-与世界模型训练管线集成)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. 数据集概述

- **GitHub**: [Thelegendzz/MotionScape](https://github.com/Thelegendzz/MotionScape)
- **核心特点**:
  - **4K 分辨率视频**: 高质量视觉观测
  - **6-DoF 轨迹**: 完整的位置和姿态信息
  - **语言描述**: 自然语言描述的飞行意图和场景语义
  - **多场景覆盖**: 城市、郊区、森林、沙漠等多种地形

### 数据集统计

| 属性 | 数值 |
|---|---|
| 总视频时长 | 数十小时 |
| 视频分辨率 | 3840 x 2160 (4K) |
| 帧率 | 30 FPS |
| 轨迹精度 | 6-DoF（位置 + 姿态） |
| 语言描述 | 每段视频配有场景描述 |
| 场景类型 | 城市、郊区、森林、沙漠、水上等 |
| 光照条件 | 白天、黄昏、夜晚 |

---

## 2. 仓库与下载

### 2.1 克隆仓库

```bash
git clone https://github.com/Thelegendzz/MotionScape.git
cd MotionScape
```

### 2.2 下载数据集

MotionScape 数据集较大，请确保磁盘空间充足（建议 500 GB+）。

```bash
# 方法 1：通过仓库提供的下载脚本
bash scripts/download_dataset.sh --output_dir ./data --split all

# 方法 2：分批下载（如数据集托管在 Hugging Face）
pip install huggingface_hub
huggingface-cli download --repo-type dataset Thelegendzz/MotionScape --local-dir ./data

# 方法 3：仅下载训练集
bash scripts/download_dataset.sh --output_dir ./data --split train
```

### 2.3 数据集目录结构

```text
MotionScape/
├── README.md
├── scripts/
│   ├── download_dataset.sh
│   ├── preprocess.py
│   ├── visualize.py
│   └── evaluate_trajectory.py
├── src/
│   ├── dataset.py            # PyTorch Dataset 类
│   ├── transforms.py         # 数据变换
│   └── utils.py
├── configs/
│   └── default.yaml
└── examples/
    ├── load_data.py
    └── visualize_sample.py
```

### 2.4 解压与校验

```bash
# 下载完成后解压（如为压缩格式）
cd data
for f in *.tar.gz; do tar -xzf "$f"; done

# 校验文件完整性
python ../scripts/verify_checksums.py --data_dir .
```

---

## 3. 数据格式详解

### 3.1 视频文件

```text
data/
├── videos/
│   ├── scene_001/
│   │   ├── flight_01.mp4
│   │   ├── flight_02.mp4
│   │   └── ...
│   ├── scene_002/
│   └── ...
```

视频参数：
- 编码格式：H.264
- 分辨率：3840 x 2160（可降至 1080p 或 720p 使用）
- 帧率：30 FPS
- 每段飞行视频长度：30 秒 - 5 分钟不等

### 3.2 轨迹文件

```text
data/
├── trajectories/
│   ├── scene_001/
│   │   ├── flight_01.npz     # NumPy 压缩格式
│   │   ├── flight_02.npz
│   │   └── ...
│   └── ...
```

轨迹数据结构：

```python
import numpy as np

traj = np.load("data/trajectories/scene_001/flight_01.npz")
print(traj.keys())

# 'position': (N, 3)     — 世界坐标系位置 [x, y, z]，单位：米
# 'orientation': (N, 4)   — 四元数姿态 [qw, qx, qy, qz]
# 'velocity': (N, 3)      — 线速度 [vx, vy, vz]，单位：米/秒
# 'angular_velocity': (N, 3)  — 角速度 [wx, wy, wz]，单位：弧度/秒
# 'timestamp': (N,)       — 时间戳，单位：秒
# 'frame_indices': (N,)   — 对应的视频帧索引
```

### 3.3 语言描述文件

```text
data/
├── descriptions/
│   ├── scene_001/
│   │   ├── flight_01.json
│   │   └── flight_02.json
│   └── ...
```

描述文件格式：

```json
{
    "scene_description": "Urban environment with tall buildings and wide roads",
    "flight_intent": "Navigate through the city streets, following the main road",
    "segment_descriptions": [
        {
            "start_frame": 0,
            "end_frame": 300,
            "description": "Taking off from the rooftop and ascending to 50 meters"
        },
        {
            "start_frame": 300,
            "end_frame": 900,
            "description": "Flying north along the main avenue, passing the central park"
        }
    ],
    "weather": "clear",
    "time_of_day": "afternoon"
}
```

---

## 4. 数据加载与预处理

### 4.1 使用仓库自带的 Dataset 类

```python
"""
MotionScape 数据加载示例（基于仓库提供的 Dataset 类）
"""
from src.dataset import MotionScapeDataset
from torchvision import transforms

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 创建数据集
dataset = MotionScapeDataset(
    data_dir="./data",
    split="train",
    context_length=5,           # 上下文帧数
    prediction_length=10,       # 预测帧数
    image_transform=transform,
    resolution="720p",          # 可选: "4k", "1080p", "720p"
    include_language=True       # 包含语言描述
)

# 创建 DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# 读取一个批次
batch = next(iter(loader))
print(f"Images: {batch['images'].shape}")          # (16, 5, 3, 224, 224)
print(f"Positions: {batch['positions'].shape}")    # (16, 15, 3)
print(f"Orientations: {batch['orientations'].shape}")  # (16, 15, 4)
print(f"Descriptions: {len(batch['descriptions'])}")   # 16 个字符串
```

### 4.2 自定义数据加载

如果不使用仓库自带的 Dataset，可以自行编写加载逻辑：

```python
import numpy as np
import cv2
import json

class MotionScapeLoader:
    def __init__(self, data_dir, resolution="720p"):
        self.data_dir = data_dir
        self.resolution = resolution
        self.samples = self._index_samples()

    def _index_samples(self):
        """索引所有可用样本"""
        samples = []
        # 遍历所有场景和飞行段
        # ... 省略遍历逻辑
        return samples

    def load_video_frames(self, video_path, frame_indices):
        """加载指定帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        return np.stack(frames)

    def load_trajectory(self, traj_path, frame_indices):
        """加载对应轨迹"""
        data = np.load(traj_path)
        positions = data["position"][frame_indices]
        orientations = data["orientation"][frame_indices]
        return positions, orientations

    def load_description(self, desc_path):
        """加载语言描述"""
        with open(desc_path, "r") as f:
            return json.load(f)
```

### 4.3 数据预处理脚本

```bash
# 降低视频分辨率（节省存储和计算）
python scripts/preprocess.py \
    --input_dir data/videos \
    --output_dir data/videos_720p \
    --target_resolution 1280x720 \
    --target_fps 15

# 提取轨迹特征
python scripts/extract_features.py \
    --data_dir data \
    --output_dir data/features \
    --feature_type "optical_flow"
```

---

## 5. 数据可视化

### 5.1 轨迹 3D 可视化

```python
"""
3D 轨迹可视化脚本
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory_3d(traj_path):
    data = np.load(traj_path)
    pos = data["position"]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制轨迹
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=1, label="Trajectory")

    # 标记起点和终点
    ax.scatter(*pos[0], c="green", s=100, marker="o", label="Start")
    ax.scatter(*pos[-1], c="red", s=100, marker="^", label="End")

    # 每隔 N 个时间步绘制坐标系（表示姿态）
    N = 50
    for i in range(0, len(pos), N):
        orientation = data["orientation"][i]
        # 将四元数转换为旋转矩阵，绘制坐标轴
        # ... 省略旋转矩阵转换和坐标轴绘制

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.title("MotionScape Trajectory Visualization")
    plt.show()

# 使用
plot_trajectory_3d("data/trajectories/scene_001/flight_01.npz")
```

### 5.2 视频与轨迹叠加可视化

```bash
# 使用仓库提供的可视化脚本
python scripts/visualize.py \
    --video data/videos/scene_001/flight_01.mp4 \
    --trajectory data/trajectories/scene_001/flight_01.npz \
    --output visualization_output.mp4 \
    --overlay_trajectory True
```

### 5.3 使用仓库示例

```bash
cd examples
python visualize_sample.py --scene scene_001 --flight flight_01
```

---

## 6. 与世界模型训练管线集成

### 6.1 用于视频预测模型

MotionScape 可直接用于训练视频预测（Video Prediction）世界模型：

```python
"""
将 MotionScape 数据适配到视频预测管线
"""
class MotionScapeForVideoPrediction:
    def __init__(self, dataset, pred_length=10):
        self.dataset = dataset
        self.pred_length = pred_length

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # 输入：当前帧 + 历史帧
        input_frames = sample["images"][:5]     # (5, 3, H, W)
        # 目标：未来帧
        target_frames = sample["images"][5:15]  # (10, 3, H, W)
        # 条件：动作序列
        actions = sample["actions"][5:15]        # (10, 4)

        return {
            "input_frames": input_frames,
            "target_frames": target_frames,
            "actions": actions,
            "description": sample["description"]
        }
```

### 6.2 用于轨迹预测模型

```python
class MotionScapeForTrajectoryPrediction:
    def __init__(self, dataset, pred_length=10):
        self.dataset = dataset
        self.pred_length = pred_length

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # 输入：视觉观测 + 历史轨迹
        context_images = sample["images"][:5]
        context_traj = sample["positions"][:5]  # (5, 3)
        # 目标：未来轨迹
        target_traj = sample["positions"][5:15]  # (10, 3)
        # 语言条件
        description = sample["description"]

        return {
            "context_images": context_images,
            "context_trajectory": context_traj,
            "target_trajectory": target_traj,
            "language_condition": description
        }
```

### 6.3 与 DreamerV3 等世界模型的集成

MotionScape 的数据格式可以转换为 DreamerV3 所需的 replay buffer 格式：

```python
def convert_to_dreamer_format(sample):
    """将 MotionScape 样本转换为 DreamerV3 兼容格式"""
    return {
        "observation": sample["images"],       # (T, 3, H, W)
        "action": sample["actions"],           # (T, 4)
        "reward": compute_reward(sample),      # 自定义奖励函数
        "done": False,
        "info": {"description": sample["description"]}
    }
```

---

## 7. 常见问题与解决方案

### Q1: 数据集下载中断

```bash
# 使用支持断点续传的工具
wget -c <download_url>

# 或使用 huggingface-cli（自动支持断点续传）
huggingface-cli download --resume-download Thelegendzz/MotionScape --local-dir ./data
```

### Q2: 4K 视频占用空间过大

```bash
# 预处理降低分辨率
python scripts/preprocess.py \
    --input_dir data/videos \
    --output_dir data/videos_720p \
    --target_resolution 1280x720
```

通常 720p 分辨率对于模型训练已经足够，4K 主要用于高质量可视化和展示。

### Q3: 视频帧与轨迹对齐问题

```python
# 使用 frame_indices 字段确保对齐
traj = np.load(traj_path)
frame_indices = traj["frame_indices"]
# frame_indices[i] 对应 videos 中的第 frame_indices[i] 帧
# 确保加载视频帧时使用 frame_indices 而非连续索引
```

### Q4: 四元数转换错误

```python
# 使用 scipy 或 pyquaternion 进行正确的四元数转换
from scipy.spatial.transform import Rotation

quat = [qw, qx, qy, qz]  # 注意四元数顺序
# scipy 使用 [x, y, z, w] 顺序
rot = Rotation.from_quat([qx, qy, qz, qw])
euler = rot.as_euler("xyz", degrees=True)  # 转换为欧拉角
```

### Q5: 内存不足加载整个数据集

```python
# 使用流式加载，不一次性加载所有数据
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2  # 预加载 2 个批次
)
```

---

## 参考资源

- MotionScape GitHub: https://github.com/Thelegendzz/MotionScape
- 四元数与姿态表示: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
- PyTorch DataLoader 文档: https://pytorch.org/docs/stable/data.html

## 延伸阅读

- [关键数据集与基准](../02-世界模型专题/06-关键数据集与基准.md) — MotionScape 的理论背景
- [生成式世界模型](../02-世界模型专题/02-生成式世界模型.md) — 视频生成世界模型详解
- [可复现项目候选清单](./08-可复现项目候选清单.md) — 更多可复现项目
