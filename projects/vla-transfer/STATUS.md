# 进度板

操作性的：现在在跑什么、日志在哪、下一步按什么顺序。结论以 `RESULTS.md` 为准，
判据以 `docs/02-判据-位置到达不算拍到.md` 为准。

## 校准（2026-10-26 晚，§3.9）

**§3.5–§3.8 的全部成功率作废**：判据里有两个 bug（可行视点按整场算、`min|u|` 门限用了
距目标的距离），加上残差 BC 的归一化用错统计量。补上专家上界后专家 0/60 才暴露出来。
现在唯一可引用的数字在 `RESULTS.md` §3.9。

## 正在跑（2026-10-28）

`/root/work/seq3.sh`：12k 步 a3_96sg（子目标接口，12000 步）→ 同判据 60 集 →
三 seed 子目标协议（40 epoch / patience 6 / 余弦退火）→ 跨仓库微调（`uav96btr` →
`uav96b`）。日志 `/root/work/logs/seq3.out`、每步单独日志同名 `.log`。

`/root/work/auto2.sh`：等 seq3 打出 `SEQ2 DONE` 后自动接**盲训练三 seed**
（训练与推理图像置零），汇总写 `runs/protocol_vision.md`。

四个必须记住的启动条件（每一条都曾让队列整夜空转）：

1. 入口是 `projects/vla-transfer/policy/uav_config.py`，不是 `scripts/train.py`。
2. 训练与服务都要 `cd /opt/robogo/tools/openpi`——`assets_dirs` 相对 cwd。
3. 需要 `UAV_DATASET_ROOT`、`HF_LEROBOT_HOME`、`HF_HUB_OFFLINE=1` 三个环境变量。
4. checkpoint 步号是 **5999 / 11999**，不是 6000；起服务前先 `ls` 一下。

## 已完成（可引用）

| 阶段 | 结果 |
|---|---|
| 专家上界（修判据后） | **1.000** [0.940, 1.000]，60 集 |
| 专家上界（布局 b） | **1.000** [0.940, 1.000]，60 集 |
| 相对位移子目标（看得见） | **0.317** [0.213, 0.442] |
| 同上，盲 | 0.183 [0.106, 0.299] |
| 配对 McNemar | 8:0，**p = 0.0078** |
| 残差 BC（归一化修好） | 0.000 / 0.000（见/盲） |
| 子目标定位误差（held-out） | 8.26 cm（第 40 轮），横向/竖向 5.76 / 5.14 cm |
| π0.5 子目标（A3，6000 步） | 0.183 [0.106, 0.299] |
| π0.5 子目标（只训动作头 a1） | 0.183 [0.106, 0.299] |
| π0.5 子目标（6400 集，12k 步） | 0.183 [0.106, 0.299] |
| 跨仓库零样本（布局 a → b） | 0.183 [0.106, 0.299] |

后四行合起来是一个结论：**在这个任务上，6 B 的 π0.5 无论怎么配（全层 LoRA / 只训动作头 /
加数据 / 换布局）都停在 0.183**，与盲策略同值；0.3 M 的子目标 BC 是 0.317。

## 判定标准（写死不挪）

- 上界不 ≈1.0 时，任何策略数字都不引用。
- 视觉是否必要，看**配对**：同一批 seed，看得见成功而盲失败的计数 vs 反向计数；
  两者为 0 时不得宣称"视觉有用"。
- `min|u|` 只在"到本集指令视点的距离 ≤0.7 m"的步上取；到目标的距离不能当门限
  （专家站位本身 1.15 m）。

## 队列（按代价从低到高）

1. 读 seq3：12k 与 6k 的配对差（同一批 60 集）
2. 三 seed 协议的 mean±std，替代现在所有单 seed 数字
3. 跨仓库微调 vs 零样本（0.183 是零样本的地板）
4. 盲对照三 seed（auto2 自动跑）
5. T2 LIBERO 锚点：环境已打通（osmesa 渲染 OK、96×96 obs、20 步零动作通过），
   官方权重在 `/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero`；
   跑法见 `docs/11-LIBERO锚点.md`（待写）
6. E 栈剩余子系统（见 `../ENGINEERING-STACK.md`）

## 已知环境坑（省下次的排查时间）

- 远端 python 只有 `/opt/miniforge3/envs/pi0/bin/python`；`python` 不在 PATH。
- `pkill -f train_subgoal.py` 会连自己的 ssh shell 一起杀，用 `[t]rain_` 转义。
- 失败的 `serve_uav.py` 会留下僵尸占 18 GB 显存，新训练直接 OOM；起训练前先查
  `nvidia-smi --query-compute-apps`。
- LIBERO 需要 `PYTHONPATH=/root/work/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`；
  EGL 在这台机器上不可用，osmesa 可以。
- PowerShell 里不要用 `$(...)` 拼 ssh 命令、不要用 `Set-Content -Encoding utf8` 写源码
  （会加 BOM / 破坏中文）；一律写 `.sh` → base64 → 远端 `base64 -d`。

## 复现命令

```bash
# 单集判据自检（本机）
py -3.9 projects/uav-inspect/scripts/gen_dataset.py --total 3 --shard 0 --shards 1 \
  --img-size 96 --out .work/smoke --backend rigid --seed-base 7200

# 专家上界（远端，必须先有它，否则判据不可信）
cd /root/work/repo && OMP_NUM_THREADS=1 /opt/miniforge3/envs/pi0/bin/python \
  projects/vla-transfer/scripts/eval_uav.py --scripted --data /root/work/data/uav96c \
  --episodes 60 --img-size 96 --device cpu --out runs/r_oracle

# 整套串行队列（12k → 三 seed → 微调 → 盲对照）
bash /root/work/seq3.sh && bash /root/work/auto2.sh
```
