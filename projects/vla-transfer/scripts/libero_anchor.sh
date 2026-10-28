#!/bin/bash
# T2 澶栭儴閿氱偣锛氬湪 LIBERO 涓婅窇瀹樻柟 pi05_libero锛岄獙璇併€屾垜浠殑璇勬祴瑁呯疆鑳藉鐜板叕寮€鏁板瓧銆嶃€?#
# 涓轰粈涔堥渶瑕佸畠锛氭湰搴撻噷鎵€鏈夋垚鍔熺巼閮芥槸鑷繁閫犵殑鍒ゆ嵁锛堝眳涓?+ 鍒拌揪锛夛紝澶栦汉鏃犳硶鍒ゆ柇鏄换鍔＄畝鍗曡繕鏄瓥鐣ュ己銆?# LIBERO goal 濂椾欢鏈夊叕寮€鎴愮哗鍗曪紝鐢ㄥ悓涓€鏉?openpi 鎺ㄧ悊璺緞璺戜竴閬嶏紝鎵嶈兘璇存槑銆屾垜浠殑 harness 涓嶆槸鑷鑷瘽銆嶃€?#
# 鐢ㄦ硶锛?#   bash scripts/libero_anchor.sh               # 鍐掔儫锛?0 浠诲姟 脳 1 trial
#   NUM_TRIALS=50 bash scripts/libero_anchor.sh # 姝ｅ紡锛?0 浠诲姟 脳 50 trial锛堢害 2.5 h锛?#
# 鍓嶇疆锛?root/work/LIBERO 宸插厠闅嗐€乸i05_libero 鏉冮噸宸插湪 ~/.cache/openpi锛汦GL 鍦ㄨ繖鍙版満鍣ㄤ笉鍙敤锛?# 娓叉煋蹇呴』璧?osmesa銆?set -u
PY=${PY:-/opt/miniforge3/envs/pi0/bin/python}
OPENPI=${OPENPI:-/opt/robogo/tools/openpi}
LOG_DIR=${LOG_DIR:-/root/work/logs}
NUM_TRIALS=${NUM_TRIALS:-1}
SUITE=${SUITE:-libero_goal}

export PYTHONPATH=/root/work/LIBERO:${PYTHONPATH:-}
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$LOG_DIR"

cd "$OPENPI" || exit 1
$PY scripts/serve_policy.py --env LIBERO > "$LOG_DIR/libero_serve.log" 2>&1 &
SRV=$!
for _ in $(seq 1 90); do
  grep -qi "listening\|server started" "$LOG_DIR/libero_serve.log" 2>/dev/null && break
  sleep 5
done
echo "鏈嶅姟锛?(tail -1 "$LOG_DIR/libero_serve.log" | cut -c1-100)"

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
echo "LIBERO DONE锛?SUITE锛?{NUM_TRIALS} trial/浠诲姟"
