#!/bin/bash
# 按 docs/06-训练协议.md 跑一条完整协议：多个种子 ×（训练 → 60 集评测）。
# 每个种子独占一个输出目录，metrics.json 自带逐回合记录，可直接喂配对检验。
#
#   bash projects/vla-transfer/scripts/run_protocol.sh --data /root/work/data/uav96c2 \
#        --ckpt-kind subgoal --seeds "0 1 2" --epochs 40
set -euo pipefail

PY=/opt/miniforge3/envs/pi0/bin/python
DATA=""; KIND="subgoal"; SEEDS="0 1 2"; EPOCHS=40; EPISODES=60; PREFIX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;
    --ckpt-kind) KIND="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --episodes) EPISODES="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    *) echo "未知参数 $1" >&2; exit 2;;
  esac
done
[ -n "$DATA" ] || { echo "必须给 --data" >&2; exit 2; }
[ -n "$PREFIX" ] || PREFIX="p_${KIND}"

cd /root/work/repo
for s in $SEEDS; do
  OUT="runs/${PREFIX}_s${s}"
  EVAL="${OUT}_eval"
  echo "[$(date +%T)] seed=$s 训练开始 → $OUT"
  if [ "$KIND" = "subgoal" ]; then
    $PY projects/vla-transfer/scripts/train_subgoal.py --data "$DATA" \
      --epochs "$EPOCHS" --batch 256 --workers 10 --stride 2 --val-stride 2 \
      --lr 5e-4 --seed "$s" --patience 6 --device cuda --require-coverage 0.5 \
      --out "$OUT" 2>&1 | tee "$OUT.train.log" | tail -3
    CKPT="$OUT/ckpt.pt"; EXTRA="--subgoal"
  else
    $PY projects/vla-transfer/scripts/train_bc.py --data "$DATA" \
      --epochs "$EPOCHS" --batch 256 --workers 10 --every 2 --lr 1e-3 \
      --seed "$s" --device cuda --require-coverage 0.5 --out "$OUT" \
      2>&1 | tee "$OUT.train.log" | tail -3
    CKPT="$OUT/ckpt.pt"; EXTRA=""
  fi
  echo "[$(date +%T)] seed=$s 评测 $EPISODES 集"
  $PY projects/vla-transfer/scripts/eval_uav.py --ckpt "$CKPT" --data "$DATA" \
    --episodes "$EPISODES" --device cuda --out "$EVAL" $EXTRA 2>&1 | tail -3
done
echo "[$(date +%T)] PROTOCOL DONE prefix=$PREFIX"
