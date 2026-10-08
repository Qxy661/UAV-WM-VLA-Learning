# 可运行代码 Demo

十九个 demo，用来给文档里的结论提供可复现的证据 —— 不是伪代码，是能跑出数、
能画成图、能改参数看变化的脚本。

除 **D 是纯标准库的检索小工具**（不依赖 torch）外，其余十八个都是真实 PyTorch 实现
（真 `nn.Module`、真训练循环、真闭环推演），用来验证正文里的技术结论。

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
├── e_gsc_safety.py       # Demo E: 几何安全校正（GSC）在闭环里管用到什么程度
├── f_control_loop.py     # Demo F: 量化退化 + 外环延迟的闭环容忍边界
├── g_eval_metrics.py     # Demo G: 七个导航指标的排名分歧
├── h_symlog.py           # Demo H: symlog 是否真的统一了不同尺度的优化难度
├── i_plucker.py          # Demo I: Plucker 坐标对相机运动的线性响应
├── j_render_cost.py      # Demo J: 体渲染与 alpha 溅射，同公式不同结果
├── k_physics_prior.py    # Demo K: 物理残差先验管用在哪一路、失效在哪一路
├── l_seq_metrics.py      # Demo L: ATE / RPE / SSIM / FID，纯 numpy 实现与自检
├── m_exposure_bias.py    # Demo M: 像素自回归的误差累积与加噪训练的代价
├── n_quant_bits.py       # Demo N: 位宽不是均匀打折，断崖在哪
├── o_budget.py           # Demo O: 显存的四项开销与九张表的对账
├── p_chunk_horizon.py    # Demo P: 动作分块长度与闭环精度的取舍
├── q_rl_posttrain.py     # Demo Q: 奖励加权的后训练压得动多少坏数据
├── r_eval_protocol.py    # Demo R: 同一批回合，三种报法给出三种名次
├── s_wm_data_source.py   # Demo S: 世界模型当数据源，收益与想象步长
└── requirements.txt
```

## 运行

```bash
pip install -r code/requirements.txt

py -3.9 code/a_worldmodel_rssm.py    # ~30 秒
py -3.9 code/b_lang_cond_policy.py   # ~40 秒
py -3.9 code/c_action_heads.py       # ~60 秒
py -3.9 code/d_track_papers.py       # ~2 秒（纯标准库，不装 torch 也能跑）
py -3.9 code/e_gsc_safety.py         # ~5 秒
py -3.9 code/f_control_loop.py       # ~12 秒
py -3.9 code/g_eval_metrics.py       # ~4 秒
py -3.9 code/h_symlog.py             # ~30 秒
py -3.9 code/i_plucker.py            # ~1 秒
py -3.9 code/j_render_cost.py        # ~5 秒
py -3.9 code/k_physics_prior.py      # ~40 秒
py -3.9 code/l_seq_metrics.py        # ~1 秒
py -3.9 code/m_exposure_bias.py      # ~90 秒
py -3.9 code/n_quant_bits.py         # ~2 分钟
py -3.9 code/o_budget.py             # ~1.5 分钟
py -3.9 code/p_chunk_horizon.py      # ~10 秒
py -3.9 code/q_rl_posttrain.py       # ~2 分钟
py -3.9 code/r_eval_protocol.py      # ~2 分钟
py -3.9 code/s_wm_data_source.py     # ~4 分钟
```

纯 CPU 即可，全部跑完十七分钟左右；D、I、L 几乎不耗时。每个脚本会打印指标表，
并把图存到仓库根的 `figures/`。

D 额外有一个可选开关：`--verify` 会联网去 arXiv API 核对内置的 51 个 ID
（走 `https://export.arxiv.org`，**HTTP 会失败**），首次约 10 秒。网络不通时
脚本会打印提示并正常结束，离线出的图和统计不受影响。

## 十九个 demo 分别验证什么

| Demo | 对应文档 | 验证的结论 | 产出图 |
|---|---|---|---|
| A | [模型强化学习世界模型](../docs/02-世界模型专题/03-模型强化学习世界模型.md) | 只用先验推演的"想象"能多准地预测回报——这是 Dreamer 用世界模型替代真实环境的依据 | `wm_loss_curve.png`、`wm_imagination.png` |
| B | [语言条件飞行控制](../docs/03-VLA专题/03-语言条件飞行控制.md) | 模仿损失低 ≠ 闭环飞得好：给专家数据加噪声，模仿损失涨 1000 倍，闭环误差反而降 30% | `bc_distribution_shift.png`、`bc_trajectory.png` |
| C | [VLA架构演进](../docs/03-VLA专题/01-VLA架构演进.md) | 多模态动作下，MSE 回归头塌到模态平均，扩散头能采样出两个模态 | `c_multimodal.png` |
| D | [最新进展与团队追踪](../docs/08-研究前沿与开放问题/04-最新进展与团队追踪.md) | 论文产出按团队、按方向怎么看趋势；以及"先核对元数据"能抓出多少引用错误 | `track_two_labs.png` |
| E | [无人机VLA模型](../docs/03-VLA专题/02-无人机VLA模型.md) | 几何安全校正只看得见最近的那个障碍物：凸形场景把碰撞率从 100% 压到 0，凹形场景反而把它从 51.6% 抬到 77.3% | `e_gsc_scene.png` |
| F | [机载部署与优化](../docs/03-VLA专题/05-机载部署与优化.md)、[LLM驱动的无人机Agent](../docs/04-VLM专题/03-LLM驱动的无人机Agent.md) | 外环延迟收进闭环：连续轨迹指令下 τ=50 ms 误差 16 cm，τ=1 s 就是 1.62 m；换成离散航点后基准抬高、增长却平缓得多 | `f_latency.png` |
| G | [复现指南-UAV-Flow](../docs/07-实践指南/02-复现指南-UAV-Flow.md)、[复现指南-CognitiveDrone](../docs/07-实践指南/03-复现指南-CognitiveDrone.md) | 三个策略跑同一任务算七个指标，换一列指标就换一套名次：动作 MSE 上随机策略排第三（0.426），与高增益的 0.412 几乎并列；换到偏航误差，随机策略升到第二（27.62°），仅次于 PID | `g_metrics.png` |
| H | [世界模型发展史](../docs/02-世界模型专题/01-世界模型发展史.md) | 不加 symlog 时总体 MSE 看着更小，那是被 x1000 那组独占的假象；symlog 把四组相对误差从 289.9 倍收到 3.6 倍，但代价是总体 MSE 变大 | `h_symlog.png` |
| I | [生成式世界模型](../docs/02-世界模型专题/02-生成式世界模型.md) | 纯前向平移下中心射线的 Plucker 矩严格不变，边缘射线的位移正比于像半径（位移/半径恒为 0.1111）——透视扩张全由边缘贡献 | `i_plucker.png` |
| J | [3D场景世界模型](../docs/02-世界模型专题/04-3D场景世界模型.md) | 体渲染与 alpha 溅射共用同一个前向合成公式，但公式相同不等于结果相同：残差 0.0787 不随采样密度下降，把深度拉开才收敛到 0.00012 | `j_render_cost.png` |
| K | [无人机世界模型综述](../docs/02-世界模型专题/05-无人机世界模型综述.md) | 物理残差只对它写死的那几路管用：位置通道分布外误差降 56 倍，速度通道纹丝不动；rollout 到 1 秒被纯数据模型反超——单步精度不等于长期精度 | `k_physics_prior.png` |
| L | [关键数据集与基准](../docs/02-世界模型专题/06-关键数据集与基准.md) | 全局平移偏差在配准后对两个指标都近乎归零（0.442 m 的偏移残下 ATE 8.25e-16、RPE 3.14e-16）；真正拉开差距的是尺度漂移，2% 的尺度误差配准后仍在，ATE 量到 0.0318 m、RPE 只有 6.57e-04 | `l_seq_metrics.png` |
| M | [视频生成世界模型](../docs/05-综述论文精读/04-视频生成世界模型.md) | teacher forcing 跑 60 步误差不涨，换成喂自己的输出后第 60 步是第 1 步的 1084 倍；加噪训练买到了抗扰动，却先把单步误差赔掉 8.5 倍 | `m_exposure_bias.png` |
| N | [边缘VLM部署](../docs/04-VLM专题/04-边缘VLM部署.md)、[机载部署与优化](../docs/03-VLA专题/05-机载部署与优化.md) | 位宽不是均匀打折：网络落到哪个盆地本身就带来 93% 的基准波动，INT8 在 10 颗种子间从 -9% 摆到 +56%，能确定的只有断崖存在 | `n_quant_bits.png` |
| O | [环境搭建](../docs/07-实践指南/01-环境搭建.md) 等 Part 7 七篇、[机载部署与优化](../docs/03-VLA专题/05-机载部署与优化.md)、[边缘VLM部署](../docs/04-VLM专题/04-边缘VLM部署.md) | 显存四项里三项是精确算术；LoRA 省的是优化器状态（52.2 → 0.9 GB）而不是权重；九张表的差额能反解出一个从没写出来的自变量——序列长度 | `o_budget.png` |
| P | [动作头与动作分块](../docs/03-VLA专题/06-动作头与动作分块.md) | 块长从 1 拉到 16，策略调用掉到 1/15（120 → 8 次），终点误差只涨 14%（1.3593 → 1.5539 m）：省下来的是调用次数，赔进去的是对策略看不见的阵风的反应 | `ph6_chunk_horizon.png` |
| Q | [强化学习后训练与自我改进](../docs/03-VLA专题/08-强化学习后训练与自我改进.md) | 坏架次占一半时，同一份数据、同一个网络，只换损失里的权重：均匀 BC 3.8593 m，奖励加权 2.5375 m（降 34%）；但坏架次仍拿走 19% 的权重，离全正常架次的 1.3724 m 还差 1.1651 m——软加权压不干净 | `ph8_reward_weighting.png` |
| R | [评测基准与报告口径](../docs/03-VLA专题/09-评测基准与报告口径.md) | 同一批回合三种报法给出三种名次：强风下均值说标准增益好、中位数说高增益好；成功率差 8.6 个百分点的两个策略，64 回合（半宽 11.4 pp）分辨不出、约 120 回合才行；按模型算的区间是按回合算的 2.78 倍 | `ph9_eval_protocol.png` |
| S | [世界模型增强VLA](../docs/03-VLA专题/10-世界模型增强VLA.md) | 单步准不代表多步可用：训练集上一步误差只有 0.0058 m 的世界模型，开环 60 步误差放大 16.1 倍（0.0045 → 0.0719 m），起点稍离训练分布第一步就是 45 倍；想象数据把 8 个种子里训崩的那个救回来（最差 2.7312 → 1.7807 m），离散度收到 34%，但想象步长的最优值量不出来 | `ph10_wm_data_source.png` |

## 仿真的约定

A/B/E/F/G/K/P/Q/R/S 十个 demo 共用一个极简四旋翼模型（`common/quad_sim.py`）。其余 demo 用不到它。

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

上面那张运行时间表是在 CPU（单机四线程）上量的；换机器会有出入，量级不变。

## 关于真机

这里的 demo **只做仿真训练**。要接到真实无人机或 PX4 SITL，请看各文档
`## 动手验证` 一节里的 **【待验证】PX4 SITL 接入** 段 —— 那部分是 ROS2 + `px4_msgs`
的 offboard 骨架，**未在本仓库本地验证**（需要 Ubuntu + PX4 + MicroXRCEAgent 才能跑）。
只有涉及飞控闭环的 demo 才写这一段。
