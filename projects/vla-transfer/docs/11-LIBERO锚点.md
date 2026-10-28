# 11 LIBERO 锚点：为什么需要一个外部成绩单

本库所有成功率都来自自己造的判据：飞到目标 0.7 m 内、红点落在画面中心 |u| ≤ 0.12。
这套判据能区分策略好坏（专家 1.000、盲策略 0.183），但**外人无法判断它有多难**——
0.333 到底是强还是弱，没有参照系。这一章补的是参照系。

## 锚点是什么

LIBERO goal 套件是公开基准：10 个任务、每任务 50 次试验，官方 pi05_libero 权重在同一套
评测脚本下有公开成绩。我们用**同一条 openpi 推理路径**（`scripts/serve_policy.py` →
WebSocket → `examples/libero/main.py`）跑一遍，检查三件事：

1. **装置能不能跑通**：环境、渲染、动作接口、episode 循环是否与官方一致。
2. **数字能不能复现**：跑出来的成功率是否落在公开区间的合理位置。差很多说明我们的
   装置有系统偏差，那么本库其它数字都要打折看。
3. **口径差在哪**：LIBERO 是 7 维增量动作 + 10 步等待 + 256² 渲染，我们是 3 维相对
   子目标 + 96² 渲染。两个口径不能互相换算，只能各自报。

## 怎么跑

```bash
# 冒烟：10 任务 × 1 trial（约 10 分钟）
bash projects/vla-transfer/scripts/libero_anchor.sh

# 正式：10 任务 × 50 trial（约 2.5 小时）
NUM_TRIALS=50 bash projects/vla-transfer/scripts/libero_anchor.sh
```

前置条件（这台机器上已验证）：

- `/root/work/LIBERO` 已克隆，`PYTHONPATH` 指过去；
- **必须 `MUJOCO_GL=osmesa`**：这台机器 EGL 不可用，用 EGL 会直接抛
  `GLFWError`/`EGL_NOT_INITIALIZED`；
- 官方权重在 `/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero`
  （`download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")` 拉的）；
- `HF_HUB_OFFLINE=1`，否则 transformers 会去连 huggingface 卡住。

## 结果

_待跑（GPU 队列跑完子目标协议后执行）。落盘位置：`/root/work/logs/libero_eval.log`，
视频在 `/root/work/libero_videos`。_

## 这一章不能替代什么

锚点只证明"我们的推理装置没跑偏"，不证明"我们的任务设计合理"。后者靠的是
`docs/02-判据-位置到达不算拍到.md` 里的区分度实验（专家 1.000 / 盲 0.183）。
两件事要分开看：**装置可信** 与 **任务有区分度** 是两条独立的验收线。
