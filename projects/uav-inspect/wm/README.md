# 世界模型（M3）训练说明 —— 在本机只跑推理，训练放开发机

## 为什么分两处

这台开发机上跑 M0/M1/M2 的仿真与数据生成只要几十秒，但 M3 训练要反复调超参，
占用显卡时间。约定：**数据生成在仓库内（确定性、numpy-only），训练在开发机跑**，
产物（`metrics.json` + `ckpt.pt`）拷回来，仓库只留指标与结论。

## 开发机上的一条命令

```bash
# 1) 依赖（一次）
pip install numpy torch            # 有 CUDA 就用 CUDA 版，脚本会自动选

# 2) 生成数据（确定性，8 集约 20 秒；不需要 torch）
python projects/uav-inspect/scripts/m2_inspect.py --episodes 8 --backend rigid

# 3) 训练 + 评测（20 epoch，4060 级显卡约 35 秒）
python projects/uav-inspect/wm/train.py --epochs 20 --device cuda
```

CPU 也能跑，命令末尾换 `--device cpu`。

## 产物与判定

`projects/uav-inspect/runs/m3_wm/metrics.json` 里最关键的三行：

| 键 | 含义 | 通过条件 |
|---|---|---|
| `metrics.val.roll_mse` | 只给动作、开环 10 步（约 1 s）的图像 MSE | 低于 `metrics.val.persist_mse` |
| `metrics.val.vis_auc` | 未来 10 步内目标可见性的 AUC | 高于 `metrics.val.vis_auc_persist` |
| `passed` | 上面两条与 mean-frame 基线的合并判定 | `true` |

`persist_mse` 是"下一帧 = 上一帧"的强基线。它在 0.1 s 间隔上几乎不可战胜（首轮实验：
0.162 对模型 0.163），所以时程放到 10 步（约 1 s）才有意义——这个时程选的不是好看的数，
是"在这个场景里，冻结上一帧什么时候开始失效"。

## 验收后要拷回来的东西

- `runs/m3_wm/metrics.json`（进 git，作为该 commit 的实测数字）
- `runs/m3_wm/ckpt.pt` **不进 git**（`runs/` 已 gitignore），仅用于复现演示
- 终端输出的 epoch 曲线（贴进 `CHANGELOG.md` 或 M3 的 ADR）

## 已知的坑（会浪费你时间的那种）

- **可见性标签是数红色像素得到的**，阈值在 `[0,1]` 尺度上。早期写成 `0.6*255`，
  标签恒为 0，AUC 于是恒 0.500——这种量纲错不会报错，只会让整条指标悄悄失效。
- 数据是 64×64、9.6 Hz（控制 48 Hz 每 5 步拍一帧）。窗口 `ctx=8`、`horizon=10`，
  8 集只有 316 个训练窗口；窗口数不够时先加 episode 数，不要先加模型宽度。
- 解码器以最后一帧为基底做残差（`base=ctx_f[:, -1]`）。去掉这一项，模型基本不可能赢过
  persistence——这不是模型弱，是让从零重建整帧去比较一个免费基线。
