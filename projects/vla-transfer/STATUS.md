# 进度板

操作性的：现在在跑什么、日志在哪、下一步按什么顺序。结论以 `RESULTS.md` 为准，
判据以 `docs/02-判据-位置到达不算拍到.md` 为准。

## 校准（2026-10-26 晚，§3.9）

**§3.5–§3.8 的全部成功率作废**：判据里有两个 bug（可行视点按整场算、`min|u|` 门限用了
距目标的距离），加上残差 BC 的归一化用错统计量。补上专家上界后专家 0/60 才暴露出来。
现在唯一可引用的数字在 `RESULTS.md` §3.9。

## 正在跑

`/root/work/pi05_go.sh` → π0.5 A3（`a3_96c/5999`）在修好的判据下重评 60 集，
产出 `runs/v_pi05`，日志 `/root/work/logs/v_pi05.log`、`/root/work/logs/pi05_go.out`。
注意：服务必须从 `/opt/robogo/tools/openpi` 启动（`assets_dirs` 是相对 cwd 的），
checkpoint 的步号是 **5999** 不是 6000。

## 已完成（可引用）

| 阶段 | 结果 |
|---|---|
| 专家上界（修判据后） | **1.000** [0.940, 1.000]，60 集 |
| 相对位移子目标（看得见） | **0.317** [0.213, 0.442] |
| 同上，盲 | 0.183 [0.106, 0.299] |
| 配对 McNemar | 8:0，**p = 0.0078** |
| 残差 BC（归一化修好） | 0.000 / 0.000（见/盲） |
| 子目标定位误差（held-out） | 14.41 cm（容差约 9.7 cm） |

## 判定标准（写死不挪）

- 上界不 ≈1.0 时，任何策略数字都不引用。
- 视觉是否必要，看**配对**：同一批 seed，看得见成功而盲失败的计数 vs 反向计数；
  两者为 0 时不得宣称"视觉有用"。
- `min|u|` 只在"到本集指令视点的距离 ≤0.7 m"的步上取；到目标的距离不能当门限
  （专家站位本身 1.15 m）。

## 队列（按代价从低到高）

1. 读 `v_pi05`：π0.5 在修好的判据下是多少（与 BC 同协议）
2. 子目标定位精度：加大 epoch/数据、或改报横向误差；目标把 14.41 cm 压到 10 cm 以下
3. 若能压下去，重跑 60 集，看是否接近专家上界
4. A1/A2 消融（`/root/work/a1a2.sh` 已上传，等 GPU 空）
5. T2 LIBERO 锚点（py3.8 环境已建、`pi05_libero` 权重已在）

## 复现命令

```bash
# 单集判据自检（本机）
py -3.9 projects/uav-inspect/scripts/gen_dataset.py --total 3 --shard 0 --shards 1 \
  --img-size 96 --out .work/smoke --backend rigid --seed-base 7200

# 专家上界（远端，必须先有它，否则判据不可信）
cd /root/work/repo && OMP_NUM_THREADS=1 /opt/miniforge3/envs/pi0/bin/python \
  projects/vla-transfer/scripts/eval_uav.py --scripted --data /root/work/data/uav96c \
  --episodes 60 --img-size 96 --device cpu --out runs/r_oracle

# 子目标（训练 + 三组评测）
bash /root/work/sg_rel.sh
```
