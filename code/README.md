# 可运行代码 Demo

四个小 demo，用来给文档里的结论提供可复现的证据 —— 不是伪代码，是能跑出数、
能画成图、能改参数看变化的脚本。

其中 **A/B/C 是真实 PyTorch 实现**（真 `nn.Module`、真训练循环），用来验证正文里的技术结论；
**D 是纯标准库的检索小工具**，不依赖 torch，用来验证"怎么追一个团队"这条方法论。

## 目录结构

```
code/
├── common/
│   ├── quad_sim.py       # 四旋翼仿真（torch 批量）+ 串级控制器
│   └── plotting.py       # 出图：Agg 后端 + 中文字体
├── a_worldmodel_rssm.py  # Demo A: RSSM 世界模型与潜空间想象
├── b_lang_cond_policy.py # Demo B: 语言条件策略与行为克隆的分布漂移
├── c_action_heads.py     # Demo C: 回归动作头 vs 扩散动作头
├── d_track_papers.py     # Demo D: 从 arXiv 号解析年月，画团队研究脉络
└── requirements.txt
```

## 运行

```bash
pip install -r code/requirements.txt

py -3.9 code/a_worldmodel_rssm.py    # ~30 秒
py -3.9 code/b_lang_cond_policy.py   # ~40 秒
py -3.9 code/c_action_heads.py       # ~60 秒
py -3.9 code/d_track_papers.py       # ~2 秒（纯标准库，不装 torch 也能跑）
```

纯 CPU 即可，A/B/C 三个合计两三分钟；D 几乎不耗时。每个脚本会打印指标表，
并把图存到仓库根的 `figures/`。

D 额外有一个可选开关：`--verify` 会联网去 arXiv API 核对内置的 51 个 ID
（走 `https://export.arxiv.org`，**HTTP 会失败**），首次约 10 秒。网络不通时
脚本会打印提示并正常结束，离线出的图和统计不受影响。

## 四个 demo 分别验证什么

| Demo | 对应文档 | 验证的结论 | 产出图 |
|---|---|---|---|
| A | [模型强化学习世界模型](../docs/02-世界模型专题/03-模型强化学习世界模型.md) | 只用先验推演的"想象"能多准地预测回报——这是 Dreamer 用世界模型替代真实环境的依据 | `wm_loss_curve.png`、`wm_imagination.png` |
| B | [语言条件飞行控制](../docs/03-VLA专题/03-语言条件飞行控制.md) | 模仿损失低 ≠ 闭环飞得好：给专家数据加噪声，模仿损失涨 1000 倍，闭环误差反而降 30% | `bc_distribution_shift.png`、`bc_trajectory.png` |
| C | [VLA架构演进](../docs/03-VLA专题/01-VLA架构演进.md) | 多模态动作下，MSE 回归头塌到模态平均，扩散头能采样出两个模态 | `c_multimodal.png` |
| D | [最新进展与团队追踪](../docs/08-研究前沿与开放问题/04-最新进展与团队追踪.md) | 论文产出按团队、按方向怎么看趋势；以及"先核对元数据"能抓出多少引用错误 | `track_two_labs.png` |

## 仿真的约定

A/B/C 三个 demo 共用一个极简四旋翼模型（`common/quad_sim.py`）。D 用不到它。

```
状态 12 维:  [p(3), v(3), rpy(3), omega(3)]     位置、速度、姿态角、角速率
动作  4 维:  [thrust, wx, wy, wz]              归一化总距 + 三轴角速率
dt = 0.02 s（50 Hz）
```

坐标轴是**对角约定**：正 roll → +x 加速，正 pitch → +y 加速。真实四旋翼不是这么定义的，
这里换成对角形式，是为了让"哪个轴修正哪个误差"一眼能看出来 —— Demo B 里那个
"侧向轴交叉耦合"的 bug 正是靠这个约定才暴露出来的。

**它不是气动仿真器**，只是"够用的被控对象"。要接真机请走 PX4 那条路，见下。

## 实测环境

```
Python 3.9.12 | torch 2.6.0+cu124 | numpy 2.0.2 | scipy 1.13.1 | matplotlib 3.9.4
```

Windows / Linux 均可；matplotlib 用 Agg 后端，无显示器也能存图。
中文字体按 `Microsoft YaHei → SimHei → Noto Sans CJK SC → …` 的顺序自动挑，
一个都没有时会退回英文而不是画一屏方块。

## 关于真机

这里的 demo **只做仿真训练**。要接到真实无人机或 PX4 SITL，请看各文档
`## 动手验证` 一节里的 **【待验证】PX4 SITL 接入** 段 —— 那部分是 ROS2 + `px4_msgs`
的 offboard 骨架，**未在本仓库本地验证**（需要 Ubuntu + PX4 + MicroXRCEAgent 才能跑）。
