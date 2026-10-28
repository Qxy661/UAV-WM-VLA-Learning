#!/bin/bash
# T2 外部锚点：LIBERO 上跑官方 pi05_libero，验证「我们的评测装置能复现公开数字」。
#
# 为什么需要它：本库里所有成功率都是自己造的判据（居中 + 到达），外人无法判断是任务
# 简单还是策略强。LIBERO goal 套件有公开成绩单，用同一条 openpi 推理路径跑一遍，
# 才能说明「我们的 harness 不是自说自话」。
#
# 用法：
#   bash scripts/libero_anchor.sh              # 冒烟：10 任务 × 1 trial
#   NUM_TRIALS=50 bash scripts/libero_anchor.sh # 正式：10 任务 × 50 trial（约 2.5 h）
#
# 前置：/root/work/LIBERO 已克隆、pi05_libero 权重已在 ~/.cache/openpi；EGL 在这台
# 机器上不可用，渲染必须走 osmesa。
set -u
PY=${PY:-/opt/miniforge3/envs/pi0/bin/python}
OPENPI=${OPENPI:-/opt/robogo/tools/openpi}
LOG_DIR=${LOG_DIR:-/root/work/logs}
NUM_TRIALS=${NUM_TRIALS:-1}
SUITE=${SUITE:-libero_goal}

export PYTHONPATH=/root/work/LIBERO:${PYTHONPATH:-}
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1
mkdir -p "$LOG_DIR"

cd "$OPENPI" || exit 1
$PY scripts/serve_policy.py --env LIBERO > "$LOG_DIR/libero_serve.log" 2>&1 &
SRV=$!
for _ in $(seq 1 90); do
  grep -qi "listening\|server started" "$LOG_DIR/libero_serve.log" 2>/dev/null && break
  sleep 5
done
echo "服务：$(tail -1 "$LOG_DIR/libero_serve.log" | cut -c1-100)"

$PY examples/libero/main.py \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.num-steps-wait 10 \
    --args.video-out-path /root/work/libero_videos \
    > "$LOG_DIR/libero_eval.log" 2>&1
RC=$?
echo "eval rc=$RC"
grep -E "Total success rate|success rate" "$LOG_DIR/libero_eval.log" | tail -3
kill $SRV 2>/dev/null
sleep 2
echo "LIBERO DONE（$SUITE，$NUM_TRIALS trial/任务）"
