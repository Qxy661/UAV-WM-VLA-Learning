# PLAN v3：把动作接口从「每 0.1 s 一个推力+力矩」换成「出一个空间子目标」

> 写完即冻结。跑起来只允许改阶段状态，不允许改假设与判据。

## 0 为什么要换接口

三轮实验之后，逐控制步回归（4 维：推力 + 三轴力矩）这条路的天花板已经量出来了：

| 做法 | 闭环成功率（60 集，M6 + 居中判据） |
|---|---|
| 专家数据 3112 集，8 步动作块（bc_c） | 0.217 [0.131, 0.336] |
| 动作低通 α=0.3 / 0.6 / 0.85 | 0.200 / 0.183 / 0.183 |
| 混入 600 集修好标签的 DAgger（bc_g） | 0.000 [0.000, 0.060] |

三件事同时成立：

1. **单步拟合已经很好**（`val_mse` 0.214，目标方差 22，约 1%），所以不是学不会。
2. **低通救不回来**，所以不是单步抖动被放大。
3. **DAgger 反而更差**，因为失败 rollout 的专家标签力矩是示范的 3.9 倍（0.328 vs 0.085），
   按 18% 混进去等于教模型「持续过修正」。

结论：瓶颈在**接口**。让一个 30 万参数的 CNN 每 0.1 s 闭一次姿态环，等于让它同时学
「去哪」和「怎么稳住」。前者是决策，后者是控制——而控制我们已经有一个调好的 P-D 专家
（`uavinspect/sim/rigid.py` 的 `expert()`，串级 P-D + 开环角速率）。

**换接口**：策略只回答「目标大概在哪」，怎么飞交给控制器。这就是层次化 VLA 的标准做法
（SayCan / RT-H / 子目标预测 + 底层技能），也是本库知识层一直讲的那句
「VLA 不能 50 Hz 直连，必须出 chunk 或子目标，由控制桥跟踪」。

## 1 接口定义

策略输出（4 维）：

```
a = [tx, ty, tz, yaw_face]
```

- `(tx, ty, tz)`：**目标标记的世界坐标估计**（米）。
- `yaw_face`：期望朝向，用来让相机对准估计位置。评测时由 `(tx,ty,tz)` 反算，
  这一维只做一致性检查，不单独评分。

底层执行（复用专家控制器，不改）：

```
stand = [0, -1.15, 0.10] * 1.0        # 沿货架法向退 1.15 m，抬高 0.10 m
wp = (tx, ty, tz) + stand             # 站到位后目标应落在画面中心
yaw_ref = atan2(ty - wy, tx - wx)     # 相机对准估计位置
每 K 个控制步重问一次策略（K = 5，即 0.1 s）
```

判据不变：`dist(body, viewpoint) <= 0.7`、目标在视场内、无遮挡、
**且 `|u| <= 0.12`**（`CENTER_TOL`）。

## 2 为什么这个接口让视觉成为必要条件

目标在货架面内抖动 `dx ∈ [-0.40, 0.40]`、`dz ∈ [-0.20, 0.20]`（`sim/scene.py` 哈希取值）。
站位固定在货架中心法向 1.15 m 处，相机 70° 视场：

```
|u| ≈ dx / (1.15 · tan(35°)) ≈ dx / 0.805
```

- `|dx| = 0.40` → `|u| ≈ 0.497`，**被判失败**。
- 盲策略（看不到图）只能回归训练集均值 `dx ≈ 0` → `|u| ≈ 0`，会**蒙对**。

所以判据要配合门限：把 `CENTER_TOL` 从 0.12 收到 **0.08**，
使 `|dx| > 0.064` 的样本（约 84%）无法靠均值蒙过；同时专家用真值算 `|u| ≈ 0`，不受影响。
这个换算写进 `docs/`（见 §6 交付物 3），并在评测脚本里以 `--center-tol` 暴露。

## 3 数据改造（最小改动）

只加一个字段，不重录轨迹：

- `projects/uav-inspect/uavinspect/pipeline/inspect.py`：`run_episode` 把
  `scene.target_pos` 写进返回的 meta；`gen_dataset.py` 落盘时带上。
- `meta.json` 新增 `"target_pos": [x, y, z]`。
- 已有 4320 集不必重跑：`uav96c` 的 `scene.py` 已经是抖动版，**但 meta 里没有 target_pos**，
  所以要重跑一次数据（CPU，12 分片约 45 分钟，用 `gen_r.sh` 的同一套参数）。

## 4 训练（复用 `train_bc.py`，只换头）

- 新脚本 `projects/vla-transfer/scripts/train_subgoal.py`：
  读 `meta["target_pos"]` 做标签，`BCNet` 换 `action_dim=3`（`tx,ty,tz`），
  损失按米算的 L2（不归一化，保留物理量纲，便于直接报"平均定位误差 X cm"）。
- 超参：与 bc_c 同数据同预算（6 epoch、batch 256、every 2、残差不需要）。
- 会报两个数：训练/验证的**平均定位误差（米）**、以及 `dx/dz` 分桶误差表。

## 5 评测（新增一个策略包装）

`projects/vla-transfer/scripts/eval_uav.py` 增加 `--subgoal`：

```python
class SubgoalPolicy:
    def __call__(self, ctx):
        if not self.buf:                    # 每 K 步重问一次
            est = self.model(image, state, text)      # (3,) 米
            wp = est + STAND
            self.yaw_ref = atan2(est[1]-wp[1], est[0]-wp[0])
        return self.ctrl.compute(ctx["st"], wp, yaw_ref=self.yaw_ref)
```

`ctx["ctrl"]` 已经在 `run_episode` 里透传（DAgger 用的就是它），不用改仿真。

三组对照（各 60 集）：

| 组 | 说明 | 预期 |
|---|---|---|
| oracle-subgoal | 用真值 `target_pos` 出子目标 | ≥ 0.90（上界） |
| BC-subgoal（看得见） | 模型回归 | ≥ 0.60 |
| BC-subgoal（盲） | 图像置零 | ≤ 0.20 |

判据：`sighted - blind >= 0.30` 且两个 Wilson 区间不重叠 → 视觉成为必要条件。

## 6 交付物

1. `projects/vla-transfer/scripts/train_subgoal.py` + `runs/subgoal_c/ckpt.pt`
2. `eval_uav.py --subgoal/--center-tol` 与三组对照的 `metrics.json`
3. `docs/03-判据-从逐控制步到空间子目标.md`：为什么换接口、|u| 与 dx 的换算、
   三种接口（逐控制步 / 动作块 / 子目标）各自需要多少数据与控制频率
4. RESULTS §3.9：三组表 + 定位误差 + 「为什么这条路径把视觉变成必要条件」

## 7 止损

- 数据重跑后 `target_pos` 缺失率 > 1% → 停，先修录制。
- oracle-subgoal < 0.80 → 判据或站位有问题，先修判据，不进训练。
- BC-subgoal 看得见 < 0.40 且定位误差 > 15 cm → 怀疑 96×96 分辨率不够定位，
  提高到 128×128 重跑（数据量翻倍，但只有 45 分钟）。
- 盲组 > 0.30 → 说明 `CENTER_TOL` 收得不够或站位泄露答案，回到 §2 重算门限。

## 8 与 T1 主线的关系

这条线是 T1（无人机 π0.5 适配）的**接口修正**，不是新方向：

- π0.5 的 A3 已经在 `inspect96c` 上训完（6000 步，checkpoint 在 5999）。
- 子目标接口验证成功后，π0.5 只需要把输出维度从 4 维力矩换成 3 维坐标（`uav_policy.py`
  里 `UavOutputs(action_dim_out=3)`），复用同一份数据与判据，得到
  「BC-subgoal vs π0.5-subgoal」的迁移对照——这才是 PLAN v1 里 Q1 的干净版本。
