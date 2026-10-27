# vla-transfer：把 π0.5 从机械臂搬到四旋翼上

这个目录回答一个问题：**一个在大规模机械臂数据上预训练的视觉-语言-动作模型，换到
四旋翼的四维动作空间、换到仓储巡检这个任务上，还剩多少能力？** 以及，如果掉了，
掉在视觉域、动作表示，还是指令分布上。

它不是又跑一遍官方 benchmark，也不是把 π0.5 包一层界面。判据写在
[`PLAN.md`](PLAN.md) 里，跑之前定死：同一条评测协议、同一批 episode、同一套统计口径，
另加一条**行为克隆基线**——没有它，任何"预训练有用"的结论都可能只是"数据教会了它"。

## 三条线

| 线 | 内容 | 状态 |
|---|---|---|
| T1 | 无人机 π0.5 适配：4 维动作、10 维状态、96×96 输入 | 🟡 A3 训练中（6000 步 / 6400 集） |
| T2 | LIBERO 外部锚点：先复现官方 π0.5-LIBERO 分数，再谈迁移 | ⬜ 未开始 |
| T3 | OpenVLA 对照（机械臂侧，条件启动） | ⬜ 未开始 |

## 数据从哪来

`projects/uav-inspect` 的 M2 管线现场生成，不下载：

```
py -3.9 projects/uav-inspect/scripts/gen_dataset.py --total 8000 --shard 0 --shards 12 \
    --img-size 96 --out /root/work/data/uav96 --backend rigid
```

8000 集（train 6400 / val 800 / test 800），每集 96×96、约 9.6 Hz、含指令文本、位姿、
速度、四元数与四维动作（推力 + 三轴力矩）。测试集的指令表述在训练集里**一句都不出现**
（`uavinspect/data/instructions.py` 里的 `TEMPLATES_HELD_OUT`），测的是泛化不是记忆。

转成 LeRobot v2.1 用手写转换器（lerobot 0.1.0 的 `LeRobotDataset.create()` 有
`tuple != list` 的形状校验问题，绕开比改库便宜）：

```bash
python projects/vla-transfer/scripts/npz_to_lerobot_manual.py \
    --src /root/work/data/uav96 --dst /root/work/hf_lerobot/uav/inspect96 \
    --splits train --limit 6400 --flat
```

## 现在能跑出什么数

同一套 harness（`uav-inspect` 的 `run_episode` + `TaskSpec`），同一批 test episode：

| 策略 | 成功 | 覆盖率 | 说明 |
|---|---|---|---|
| scripted 专家 | 20/20 = 100% [83.9%, 100%] | 1.000 | 上界，说明任务本身可达 |
| 行为克隆（315k 参数，3 epoch） | 0/20 = 0% [0%, 16.1%] | 0.000 | 见下面的诊断 |
| π0.5 A3（6000 步 LoRA） | 待测 | — | 训练完成后同一批 episode 评测 |
| hover-only / random | 0% / 0%（M6） | 0.000 | 早前 M6 三基线 |

**行为克隆的 0% 值得单独说**，因为它同时暴露了数据集的问题：

- 离线对齐后的单步误差很小：推力 RMSE 0.027 N，三轴力矩 RMSE 0.008 / 0.065 / 0.042 N·m。
  只看这个数会以为策略学会了。
- 闭环 20 集全败，且把"每步重规划"打开（`--policy-every 1`）也一样（0/8）。
- 原因是训练数据里**没有恢复样本**。专家从不掉高度、从不被撞开，所以状态的分布很窄：
  `pos.z` 只在 0.25–1.49 m 之间（均值 1.19、标准差 0.25），而闭环里一旦推力偏小，
  飞行器掉到 0.25 m 以下就进入训练分布之外，策略没有可参考的行为，只能继续下沉。
  离线误差小、闭环失败，这不是矛盾，是协变量漂移的标准形态。

这条结论直接决定了后面的判据：**只比成功率不够，还要比"掉下去之后能不能回来"**。
如果 π0.5 也是 0%，那么结论会是"预训练不能凭空造出恢复行为"，而不是"π0.5 不行"；
如果它明显高于 0%，恢复能力就只能来自预训练分布，而不是这份数据。

## 怎么复现

训练在开发机（4090 24 GB）上跑，本机只写代码和看日志：

```bash
cd /opt/robogo/tools/openpi
export HF_LEROBOT_HOME=/root/work/hf_lerobot UAV_DATASET_ROOT=/root/work/hf_lerobot/uav/inspect96
export HF_HUB_OFFLINE=1 UAV_REPO=uav/inspect96 UAV_WORKERS=12
export UAV_STEPS=6000 UAV_BATCH=6 UAV_EXP=a3_uav6000 UAV_TIER=a3
nohup /opt/miniforge3/envs/pi0/bin/python /root/work/repo/projects/vla-transfer/policy/uav_config.py \
    > /root/work/logs/a3_6000.log 2>&1 &
```

评测（服务端 + 客户端分离，客户端就是 `uav-inspect` 的仿真）：

```bash
# 服务端：从 ckpt 起 policy server
UAV_EXP=a3_uav6000 UAV_STEP=6000 UAV_PORT=8001 \
  python projects/vla-transfer/scripts/serve_uav.py &

# 客户端：同一批 test episode，走 run_episode
python projects/vla-transfer/scripts/eval_uav.py --server 127.0.0.1:8001 \
    --data /root/work/data/uav96 --episodes 20 --out runs/a3_eval
```

并表：

```bash
python projects/vla-transfer/scripts/compare_policies.py \
    --runs runs --names scripted_eval bc_eval a3_eval
```

## 文件

```
policy/uav_config.py      TrainConfig（pi05_uav）：action_dim 32 + PadStatesAndActions 补零，A1/A2/A3 三档 freeze_filter
policy/uav_policy.py      UavInputs / UavOutputs：把我们的观测拼成 π0.5 要的 image/state/prompt
policy/pi05_client.py     websocket 客户端 + 分块执行器
scripts/npz_to_lerobot_manual.py   手写 LeRobot v2.1（parquet + 逐帧 PNG）
scripts/compute_norm_uav.py        归一化统计量
scripts/train_bc.py                行为克隆基线（同数据、同预算）
scripts/eval_uav.py                BC / π0.5 / 专家 三路评测，同一协议
scripts/compare_policies.py        并表 + 配对 bootstrap
scripts/serve_uav.py               从我们自己的 ckpt 起服务
```

## 已知限制

- 仿真器是自研的 numpy 刚体后端，接触只在质心、无摩擦锥，所以"撞墙"的判定偏乐观。
- 96×96 输入是显存与速度的折中，目标在画面里只有几个像素；这也是 T1 想验证的一部分
  （视觉带宽不够时，预训练还剩多少）。
- 单卡 24 GB 下 A3 只能 batch 6（18.5 GB），A1/A2 的计划步数是 6000 而不是 12000。
