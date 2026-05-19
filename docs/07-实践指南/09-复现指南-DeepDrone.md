# 09 - 复现指南：DeepDrone — LLM 自然语言控制无人机

> **预计阅读：10 分钟 | 前置知识：Python 基础、LLM 基本概念、API 调用经验**

本指南帮助你复现 DeepDrone 项目。DeepDrone 是一个用大语言模型（LLM）通过自然语言指令控制无人机的框架，是"ChatGPT 控制无人机"最直观的开源实现。

---

## 项目背景

| 项目 | 信息 |
|------|------|
| **GitHub** | [evangelosmeklis/deepdrone](https://github.com/evangelosmeklis/deepdrone) |
| **Stars** | ~189 |
| **核心思路** | LLM → 函数调用 → 飞行动作 |
| **关联论文** | 无（工程驱动项目） |
| **与本项目关系** | 展示 LLM→无人机的映射链路，与 UAV-Flow（VLA）形成方法论对比 |

---

## 环境要求

- **操作系统**：Ubuntu 20.04/22.04 或 Windows
- **Python**：3.9+
- **GPU**：不需要（使用云端 LLM API）或任意 GPU（使用本地 LLM）
- **LLM 后端**：OpenAI GPT-4 / 本地 Ollama（Llama、Qwen 等）

---

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/evangelosmeklis/deepdrone.git
cd deepdrone

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 运行方式

### 方式一：使用 OpenAI API

```bash
# 设置 API Key
export OPENAI_API_KEY="your-api-key"

# 运行
python main.py
```

### 方式二：使用本地 LLM（Ollama）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull llama3

# 修改配置指向本地 Ollama
# 在 config 中设置 base_url = "http://localhost:11434"

python main.py
```

---

## 核心架构

```
用户自然语言指令 ("飞到 10 米高度然后悬停")
        ↓
    LLM 理解意图
        ↓
    函数调用 (Function Calling)
        ↓
    飞行动作序列
        ↓
    无人机执行 (仿真/真实)
```

**关键文件**：
- `main.py` — 主入口，初始化 LLM 和无人机
- `prompts/` — Prompt 模板，定义 LLM 可调用的飞行动作
- `drone/` — 无人机控制接口（仿真/真实）

---

## 复现 Checklist

- [ ] 克隆仓库并安装依赖
- [ ] 配置 LLM API（OpenAI 或本地 Ollama）
- [ ] 运行基础 demo：让 LLM 控制无人机起飞/悬停/降落
- [ ] 修改 prompt 模板，观察行为变化
- [ ] （可选）对比不同 LLM（GPT-4 vs Llama3 vs Qwen）的效果差异

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| API 调用失败 | 检查 API Key 和网络连接 |
| LLM 输出格式错误 | 检查 prompt 模板中的函数定义是否清晰 |
| 本地 LLM 响应慢 | 使用更小的模型（如 qwen2:7b） |

---

## 与其他项目的关联

| 项目 | 区别 |
|------|------|
| **UAV-Flow** | UAV-Flow 用 VLA 端到端控制，DeepDrone 用 LLM 函数调用 |
| **Tello-LLM-ROS** | Tello-LLM-ROS 控制真实无人机，DeepDrone 偏仿真 |
| **CoDrone** | CoDrone 用云边端架构，DeepDrone 用单 LLM |

---

## 参考资源

- DeepDrone GitHub: https://github.com/evangelosmeklis/deepdrone
- OpenAI Function Calling 文档: https://platform.openai.com/docs/guides/function-calling
- Ollama 官网: https://ollama.com

## 延伸阅读

- [什么是VLA](../01-基础概念/03-什么是VLA.md) — 理解 VLA 与 LLM 控制的区别
- [语言条件飞行控制](../03-VLA专题/03-语言条件飞行控制.md) — UAV-Flow 的理论背景
- [LLM驱动的无人机Agent](../04-VLM专题/03-LLM驱动的无人机Agent.md) — LLM Agent 在无人机中的应用
