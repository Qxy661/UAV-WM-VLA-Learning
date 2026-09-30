# 09 - 复现指南：DeepDrone — LLM 自然语言控制无人机

> **预计阅读：10 分钟 | 前置知识：Python 基础、LLM 基本概念、API 调用经验**

DeepDrone 是一个用大语言模型（LLM）通过自然语言指令控制无人机的框架，是"ChatGPT 控制无人机"最直观的开源实现。

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
| **UAV-Flow** | VLA 端到端控制；DeepDrone 用 LLM 函数调用 |
| **Tello-LLM-ROS** | 控制真实无人机；DeepDrone 偏仿真 |
| **CoDrone** | 云边端架构；DeepDrone 用单 LLM |

---

## 参考资源

- DeepDrone GitHub: https://github.com/evangelosmeklis/deepdrone
- OpenAI Function Calling 文档: https://platform.openai.com/docs/guides/function-calling
- Ollama 官网: https://ollama.com

## 延伸阅读

- [什么是VLA](../01-基础概念/03-什么是VLA.md) — 理解 VLA 与 LLM 控制的区别
- [语言条件飞行控制](../03-VLA专题/03-语言条件飞行控制.md) — UAV-Flow 的理论背景
- [LLM驱动的无人机Agent](../04-VLM专题/03-LLM驱动的无人机Agent.md) — LLM Agent 在无人机中的应用

## 思考题

1. **范式对比**：DeepDrone 和 UAV-Flow 在控制链路上的根本区别是什么？这个区别对硬件要求意味着什么？

2. **环境选型**：两种运行方式各自的准备步骤是什么？完全不依赖云端该走哪条路？

3. **复现卡点**：如果 LLM 回的动作解析不出来（输出格式错误），文档让你先查哪里？为什么这一环最脆？

4. **验证顺序**：Checklist 把"运行基础 demo（起飞/悬停/降落）"排在"修改 prompt 模板"之前，这个次序有道理吗？

5. **定位判断**：文档把 DeepDrone 标成"工程驱动项目、无关联论文"，这对你的复现目标意味着什么？

<details><summary>参考答案</summary>

1. DeepDrone 走的是"用户自然语言指令 → LLM 理解意图 → 函数调用（Function Calling）→ 飞行动作序列 → 无人机执行"，UAV-Flow 则是 VLA 端到端控制。前者的硬件要求可以压到零 GPU——文档写的是"GPU：不需要（使用云端 LLM API）或任意 GPU（使用本地 LLM）"，本地 Ollama 也只是换后端，不强制显卡。

2. 方式一是 OpenAI API：设 `export OPENAI_API_KEY`，然后 `python main.py`；方式二是本地 Ollama：装好 Ollama 后 `ollama pull llama3`，在 config 里把 `base_url` 指到 `http://localhost:11434` 再运行。两条路的安装步骤相同，都是 `python -m venv venv` 加 `pip install -r requirements.txt`。

3. 文档指向 prompt 模板，让检查其中的函数定义是否清晰。因为整条链路靠 Function Calling 把 LLM 输出解析成飞行动作，而 `prompts/` 正是定义 LLM 可调用飞行动作的地方；函数定义含糊，模型就更容易给出解析不了的动作描述。

4. 有道理。这一步同时压住了 FAQ 里排前两位的失败点——"API 调用失败"（Key 与网络）和"LLM 输出格式错误"。先用最简动作集把 API 与函数调用两环打通，之后改 prompt 时行为变化才能归因到提示词，而不是被配置问题污染。

5. 意味着没有官方论文指标可对齐，复现目标是"跑通链路"而不是复现某个数，验收标准是行为观察（改 prompt 看行为变化、可选地横向比 GPT-4 / Llama3 / Qwen），而不是 benchmark 分数。从"与其他项目的关联"表看，它的定位是 LLM→无人机映射链路的工程样例，真机那一侧要另找 Tello-LLM-ROS 这类项目。

</details>
