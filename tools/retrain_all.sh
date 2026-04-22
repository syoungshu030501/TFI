#!/usr/bin/env bash
# 重训三架构 seg 集成 (segformer + maxvit + convnext 共 15 fold) + cls 5 fold。
# 共 20 个训练任务, 8 卡 L20 分三阶段并行。
# Phase 1: segformer×5  + maxvit×3              (8 procs)
# Phase 2: maxvit×2     + convnext×5 + cls×1    (8 procs)
# Phase 3: cls×4                                (4 procs)

set -u
cd "$(dirname "$0")/.."
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI

mkdir -p logs checkpoints
export HF_ENDPOINT=https://hf-mirror.com

EPOCHS=${EPOCHS:-30}
PATIENCE=${PATIENCE:-8}
BATCH_SEG=${BATCH_SEG:-4}
BATCH_CLS=${BATCH_CLS:-8}

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "retrain: epochs=$EPOCHS patience=$PATIENCE seg_bs=$BATCH_SEG cls_bs=$BATCH_CLS"

run_seg() {
  local arch=$1 fold=$2 gpu=$3
  python train_seg_ensemble.py --arch $arch --fold $fold --gpu $gpu \
      --data_dir train_resume --save_dir checkpoints \
      --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SEG \
      > logs/seg_${arch}_f${fold}.log 2>&1
}
run_cls() {
  local fold=$1 gpu=$2
  python train_classifier.py --fold $fold --gpu $gpu \
      --data_dir train_resume --save_dir checkpoints \
      --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_CLS \
      > logs/cls_f${fold}.log 2>&1
}

# ====== Phase 1: segformer 0-4 on GPU 0-4, maxvit 0-2 on GPU 5-7 ======
log "Phase 1 start"
PIDS=()
for f in 0 1 2 3 4; do run_seg segformer $f $f & PIDS+=($!); done
for i in 0 1 2; do run_seg maxvit $i $((5+i)) & PIDS+=($!); done
log "Phase 1 PIDs: ${PIDS[*]}"
for p in "${PIDS[@]}"; do wait $p; done
log "Phase 1 done"

# ====== Phase 2: maxvit 3-4 on GPU 0-1, convnext 0-4 on GPU 2-6, cls 0 on GPU 7 ======
log "Phase 2 start"
PIDS=()
for i in 3 4; do run_seg maxvit $i $((i-3)) & PIDS+=($!); done
for f in 0 1 2 3 4; do run_seg convnext $f $((2+f)) & PIDS+=($!); done
run_cls 0 7 & PIDS+=($!)
log "Phase 2 PIDs: ${PIDS[*]}"
for p in "${PIDS[@]}"; do wait $p; done
log "Phase 2 done"

# ====== Phase 3: cls 1-4 on GPU 0-3 ======
log "Phase 3 start"
PIDS=()
for f in 1 2 3 4; do run_cls $f $((f-1)) & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait $p; done
log "Phase 3 done"

log "===== retrain_all complete ====="
log "seg checkpoints:"
for f in checkpoints/seg/*/best_model.pt; do
  size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo "  $f  ${size} bytes"
done
log "cls checkpoints:"
for f in checkpoints/cls/*/best_model.pt; do
  size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo "  $f  ${size} bytes"
done
