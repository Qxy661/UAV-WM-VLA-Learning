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
- **Python**：3.8+（README 逐字是 *"Python 3.8+"*）
- **GPU**：不需要（走云端 LLM API）或任意 GPU（跑本地 LLM）
- **LLM 后端**：OpenAI / Anthropic / Google（统一走 **LiteLLM**），或本地 **Ollama**（直连，不走 LiteLLM）
- **仿真/真机接口**：内置仿真器（`tools/simulator.py`）+ DroneKit-Python（MAVLink `tcp:127.0.0.1:5760`）；另支持 Webots 的 UDP 直控

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

入口只有一条：`run.py`，它同时拉起仿真器和 Web 界面。

```bash
# 启动仿真器 + Web UI（界面在 http://localhost:8000）
python3 run.py

# 也可以走 shell 包装脚本
bash tools/start.sh
```

启动后在 Web 界面的 **Settings** 里选 AI 提供商，再选无人机的连接方式：

| 连接方式 | 地址 | 说明 |
|---|---|---|
| DroneKit / MAVLink 仿真 | `tcp:127.0.0.1:5760` | 内置仿真器走这条 |
| Webots（C 控制器） | `webots` 或 `udp:127.0.0.1:9000` | UDP 直控，20–50 Hz，格式 `roll pitch yaw throttle` |

### 云端提供商（OpenAI / Anthropic / Google）

这三家统一经 **LiteLLM** 转发，Key 从环境变量取（`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_APPLICATION_CREDENTIALS`）。模型在 `drone/config.py` 的 `models` 字典里预先登记，例如 `claude-3-sonnet` 的 `provider="anthropic"`。

### 本地 Ollama

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型（config.py 里预置的是 llama3.1）
ollama pull llama3.1
```

`drone/config.py` 里 `llama3.1` 这条的 `provider="ollama"`、`base_url="http://localhost:11434"`，Ollama 这条链路**不走 LiteLLM**（`drone/llm_interface.py` 里对 `provider == "ollama"` 单独走直连客户端）。

> **勘误（2026-10）**：本节早先写了两条「方式」——`export OPENAI_API_KEY` 后 `python main.py`，以及改 config 里的 `base_url` 后 `python main.py`。**仓库里没有 `main.py`**，入口是 `run.py`，而且并不是「先设环境变量再跑一个 CLI」，而是**跑起 Web 界面后在里面选提供商与机型**（`python3 run.py` → http://localhost:8000 → Settings）。早先只提了 OpenAI 与 Ollama 两家，漏掉了 Anthropic 与 Google（三家都经 LiteLLM）；`ollama pull` 的模型名也与 `config.py` 预置的不一致（预置的是 `llama3.1`，不是 `llama3`）。

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

**关键文件**（照仓库 README 的 *Repository Layout*）：
- `run.py` — 主入口（仿真器 + Web UI）
- `drone/` — 无人机控制、LLM 接入与 API 服务；可调用的飞行动作定义在 `drone/drone_tools.py` 与 `drone/function_tools.py`，机型与 Key 在 `drone/config.py`
- `static/` — Web UI（HTML/CSS/JS）
- `tools/` — 辅助脚本（启动器、仿真器、Webots 测试）
- `misc/` — 遗留脚本

> **勘误（2026-10）**：本节早先的关键文件里写了 `main.py`（主入口）与 `prompts/`（Prompt 模板，定义 LLM 可调用的飞行动作）。**这两个路径都不存在**：仓库顶层只有 `run.py`，没有 `prompts/` 目录，也没有单独的 prompt 模板文件——工具定义在 `drone/function_tools.py` 与 `drone/drone_tools.py` 里。那张清单是照「一个 LLM Agent 项目大概长什么样」写的。

---

## 复现 Checklist

- [ ] 克隆仓库并安装依赖
- [ ] 在 Web 界面里配置提供商（OpenAI / Anthropic / Google，或本地 Ollama）
- [ ] 运行基础 demo：让 LLM 控制无人机起飞/悬停/降落
- [ ] 换一个 provider 或机型，观察行为变化（机型在 `drone/config.py` 的 `models` 字典里）
- [ ] （可选）对比不同 LLM（如 `gpt-4` / `claude-3-sonnet` / `llama3.1`）的效果差异

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| API 调用失败 | 检查对应提供商的环境变量与网络（`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_APPLICATION_CREDENTIALS`） |
| LLM 输出格式错误 | 检查 `drone/function_tools.py` 与 `drone/drone_tools.py` 里的工具定义是否清晰，模型能不能对上山 |
| 本地 LLM 响应慢 | 换更小的本地模型：在 `drone/config.py` 的 `models` 里加一条 `ModelConfig`，再 `ollama pull` 对应模型 |

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
- LiteLLM（云端提供商统一入口）: https://github.com/BerriAI/litellm
- DroneKit-Python（MAVLink 控制）: https://github.com/dronekit/dronekit-python
- OpenAI Function Calling 文档: https://platform.openai.com/docs/guides/function-calling
- Ollama 官网: https://ollama.com

## 延伸阅读

- [什么是VLA](../01-基础概念/03-什么是VLA.md) — 理解 VLA 与 LLM 控制的区别
- [语言条件飞行控制](../03-VLA专题/03-语言条件飞行控制.md) — UAV-Flow 的理论背景
- [LLM驱动的无人机Agent](../04-VLM专题/03-LLM驱动的无人机Agent.md) — LLM Agent 在无人机中的应用

## 思考题

1. **范式对比**：DeepDrone 和 UAV-Flow 在控制链路上的根本区别是什么？这个区别对硬件要求意味着什么？

2. **环境选型**：云端提供商与本地 Ollama 两条路各自的准备步骤是什么？入口是同一个吗？完全不依赖云端该走哪条？

3. **复现卡点**：如果 LLM 回的动作解析不出来（输出格式错误），文档让你先查哪里？为什么这一环最脆？

4. **验证顺序**：Checklist 把"运行基础 demo（起飞/悬停/降落）"排在"修改 prompt 模板"之前，这个次序有道理吗？

5. **定位判断**：文档把 DeepDrone 标成"工程驱动项目、无关联论文"，这对你的复现目标意味着什么？

<details><summary>参考答案</summary>

1. DeepDrone 走的是"用户自然语言指令 → LLM 理解意图 → 函数调用（Function Calling）→ 飞行动作序列 → 无人机执行"，UAV-Flow 则是 VLA 端到端控制。前者的硬件要求可以压到零 GPU——文档写的是"GPU：不需要（使用云端 LLM API）或任意 GPU（使用本地 LLM）"，本地 Ollama 也只是换后端，不强制显卡。

2. 两条路共用一个入口：都是装完依赖跑 `python3 run.py`，然后**在 Web 界面（http://localhost:8000）的 Settings 里选提供商**。云端那条选 OpenAI / Anthropic / Google，Key 从环境变量来，统一经 LiteLLM 转发；本地那条选 Ollama，模型在 `drone/config.py` 里登记（`base_url="http://localhost:11434"`，预置模型 `llama3.1`），这条链路不走 LiteLLM。完全不依赖云端走本地 Ollama 那条。安装步骤两条路相同：`pip3 install -r requirements.txt`。

3. 文档指向工具定义，让检查其中的函数描述是否清晰（真实位置是 `drone/function_tools.py` 与 `drone/drone_tools.py`，不是早先写的 `prompts/`）。因为整条链路靠 Function Calling 把 LLM 输出解析成飞行动作，工具定义含糊，模型就更容易给出解析不了的动作描述。

4. 有道理。这一步同时压住了 FAQ 里排前两位的失败点——"API 调用失败"（Key 与网络）和"LLM 输出格式错误"（工具定义与模型对不上）。先用最简动作集把提供商接入与函数调用两环打通，之后换 provider 或机型时，行为变化才能归因到模型，而不是被配置问题污染。

5. 意味着没有官方论文指标可对齐，复现目标是"跑通链路"而不是复现某个数，验收标准是行为观察（换 provider 或机型看行为变化、可选地横向比 `gpt-4` / `claude-3-sonnet` / `llama3.1`），而不是 benchmark 分数。从"与其他项目的关联"表看，它的定位是 LLM→无人机映射链路的工程样例，真机那一侧要另找 Tello-LLM-ROS 这类项目（DeepDrone 自己接的是 DroneKit/MAVLink 仿真与 Webots，README 里也提了 DroneKit 真机控制）。

</details>
