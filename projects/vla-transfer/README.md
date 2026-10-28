# vla-transfer：把 π0.5 从机械臂搬到四旋翼上

这个目录回答一个问题：**一个在大规模机械臂数据上预训练的视觉-语言-动作模型，换到四旋翼的
动作空间、换到仓储巡检这个任务上，还剩多少能力？** 如果掉了，掉在视觉域、动作表示，还是
指令分布上。

全部数字以 [`RESULTS.md`](RESULTS.md) 为准，进度与队列看 [`STATUS.md`](STATUS.md)，
判据在 [`PLAN.md`](PLAN.md)、[`PLAN-v2`](PLAN-v2-恢复数据与重训.md)、
[`PLAN-v3-分层动作接口.md`](PLAN-v3-分层动作接口.md) 里，跑之前定死。

## 三条线

| 线 | 内容 | 状态 |
|---|---|---|
| T1 | 无人机 π0.5 适配（4 维力矩 / 3 维子目标两种接口） | ✅ 主表定稿：0.3M 的 BC 0.333 打赢 8M 的 π0.5 0.183 |
| T2 | LIBERO 外部锚点：官方 `pi05_libero` 在 `libero_goal` 上 | ✅ 冒烟 10/10；正式 50 trial/任务在跑 |
| T3 | OpenVLA 对照（机械臂侧，条件启动） | ⬜ 未开始 |

## 三条能直接引用的结论

1. **接口比模型重要。** 同一批数据、同一条判据：4 维力矩接口下 π0.5（2B+LoRA）0.150、
   残差 BC（315k）0.217；换成 3 维子目标接口后 π0.5 0.183、子目标 BC（311k）**0.333**。
   换接口带来的提升（+0.18）比换模型（2B → 0.3M 参数差 2600 倍）大得多。
2. **这个任务不需要大模型，也不需要长训练。** π0.5 三档（只训动作头 0.3M / 主干 LoRA /
   全层 LoRA 8M）闭环全是 0.183；训练步数 6000 → 12000 后成功率、覆盖率、`min|u|`
   逐位相同，配对 McNemar 0:0（p=1.0）。容量与时长的曲线都平了。
3. **视觉是必要条件，但只值 18 个点。** 训练与推理都把图像置零，子目标 BC 从 0.333 掉到
   0.183（配对 9:0，p=0.004）；剩下那 18 个点全是"按位姿飞到视点"。更早的两版判据里
   这个差别根本测不出来——那是判据的问题，见 [`docs/02`](docs/02-判据-位置到达不算拍到.md)。

## 怎么跑

数据在本机生成（numpy 后端，不需要 pybullet），训练与评测在开发机上跑。

```bash
# 1) 生成数据（本机，py -3.9）
py -3.9 projects/uav-inspect/scripts/gen_dataset.py --total 4000 --shard 0 --shards 12 \
    --img-size 96 --out /root/work/data/uav96c --backend rigid --perturb --perturb-scale 0.5

# 2) 转 LeRobot v2.1（手写转换器，绕开 lerobot 0.1.0 的 tuple != list 校验）
python projects/vla-transfer/scripts/npz_to_lerobot_manual.py --src /root/work/data/uav96c \
    --dst /root/work/hf_lerobot/uav/inspect96c --flat --repo-id uav/inspect96c \
    --action-mode subgoal --require-coverage 0.5 --splits train

# 3) 归一化统计量（cwd 必须是 openpi，assets_dirs 是相对 cwd 的）
cd /opt/robogo/tools/openpi && UAV_TIER=a3 UAV_REPO=uav/inspect96c UAV_WORKERS=12 \
    python /root/work/repo/projects/vla-transfer/scripts/compute_norm_uav.py

# 4) 训练（三档：a1 只训动作头 / a2 主干 LoRA / a3 全层 LoRA）
UAV_STEPS=6000 UAV_BATCH=4 UAV_EXP=a3_96sg UAV_TIER=a3 \
    python /root/work/repo/projects/vla-transfer/policy/uav_config.py

# 5) 评测（服务端 + 客户端分离，客户端是 uav-inspect 的仿真）
UAV_STEP=5999 UAV_EXP=a3_96sg python projects/vla-transfer/scripts/serve_uav.py &
python projects/vla-transfer/scripts/eval_uav.py --data /root/work/data/uav96c \
    --ckpt runs/sg_big/ckpt.pt --server 127.0.0.1:8000 --episodes 60 --subgoal --out runs/x

# 6) 并表 + 配对检验（不要手抄数字，之前的三个口径 bug 都是手抄引入的）
python projects/vla-transfer/scripts/make_tables.py runs/* --pair runs/A runs/B
```

## 目录

```
policy/       uav_config.py（TrainConfig：三档 freeze_filter、UAV_INIT 续训）、uav_policy.py（观测拼装）
eval/         protocol.py（M6 拍摄判据 + 居中判据 + Wilson）、rollout.py（闭环与分块执行）
bc/           315k 残差 BC 与 311k 子目标 BC 的模型与数据
serving/      推理服务（攒批、过期标记、P99）
scripts/      生成/转换/归一化/训练/评测/并表/诊断/DAgger
tools/        saliency.py（遮挡法）、replay.py（回放）
docs/         十来篇"为什么"，每篇都带"不能证明什么"
```

## 已知限制

- 仿真器是自研 numpy 刚体后端：接触只作用在质心、没有摩擦锥，"撞墙"判定偏乐观。
- 96×96 是显存与速度的折中，目标在画面里只有几十个像素；`min|u|` 中位常常落在 9.000
  （多数回合没进到 0.7 m 内），所以"逐位相同"更可能是"两者在同一处失败"。
- 三档 π0.5 与 12k 步都是 60 集单次评测；三 seed 协议（`scripts/run_protocol.sh`）
  还没全跑完。
- LIBERO 锚点只校准装置，不校准迁移：桌面操作与无人机观测的动作分布差很远。
