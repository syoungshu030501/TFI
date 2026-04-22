#!/usr/bin/env bash
# 智能调度剩余 train job: 监控 GPU 显存, 任意 GPU 空闲 (<5GB) 就启动下一个 job。
# 不动 GPU 0 上的 finevq 进程 (zhangming user)。
#
# 待办队列 (执行顺序):
#   1. maxvit 0,1,2,3,4
#   2. convnext 0,1,2,3,4
#   3. cls 0,1,2,3,4
# 共 15 个 job, 加上已在跑的 segformer 0-4。

set -u
cd "$(dirname "$0")/.."
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
export HF_ENDPOINT=https://hf-mirror.com

EPOCHS=${EPOCHS:-30}
PATIENCE=${PATIENCE:-8}
BATCH_SEG=${BATCH_SEG:-4}
BATCH_CLS=${BATCH_CLS:-8}
GPU_FREE_GB=${GPU_FREE_GB:-30}     # 至少这么多 GiB 空闲才认为可用
POLL_INT=${POLL_INT:-30}           # 每 30s 轮询一次

mkdir -p logs
LOG=logs/schedule.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

# 队列: "arch fold" 格式
QUEUE=(
  "maxvit 0" "maxvit 1" "maxvit 2" "maxvit 3" "maxvit 4"
  "convnext 0" "convnext 1" "convnext 2" "convnext 3" "convnext 4"
  "cls 0" "cls 1" "cls 2" "cls 3" "cls 4"
)

launched_pids=()

run_job() {
  local arch=$1 fold=$2 gpu=$3
  if [ "$arch" = "cls" ]; then
    nohup python train_classifier.py --fold $fold --gpu $gpu \
        --data_dir train_resume --save_dir checkpoints \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_CLS \
        > logs/cls_f${fold}.log 2>&1 &
  else
    nohup python train_seg_ensemble.py --arch $arch --fold $fold --gpu $gpu \
        --data_dir train_resume --save_dir checkpoints \
        --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH_SEG \
        > logs/seg_${arch}_f${fold}.log 2>&1 &
  fi
  echo $!
}

# 找一个空闲 GPU (跳过 GPU 0 因为 finevq 占着)
find_free_gpu() {
  while IFS=, read -r idx free; do
    idx=$(echo $idx | tr -d ' ')
    free=$(echo $free | tr -d ' MiB')
    free_gb=$((free / 1024))
    [ "$idx" = "0" ] && continue   # 跳过 finevq 占用的 GPU 0
    if [ $free_gb -ge $GPU_FREE_GB ]; then
      echo $idx
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader)
  return 1
}

log "Schedule start: ${#QUEUE[@]} jobs queued, GPU_FREE_GB=$GPU_FREE_GB poll=$POLL_INT"

i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  job="${QUEUE[$i]}"
  arch=${job% *}
  fold=${job#* }
  ckpt_dir="checkpoints/seg/${arch}_fold${fold}/best_model.pt"
  [ "$arch" = "cls" ] && ckpt_dir="checkpoints/cls/efficientnet_fold${fold}/best_model.pt"
  if [ -f "$ckpt_dir" ] && [ $(stat -c%s "$ckpt_dir") -gt 1000 ]; then
    log "skip $job (already trained: $ckpt_dir)"
    i=$((i+1)); continue
  fi

  gpu=$(find_free_gpu)
  if [ -z "$gpu" ]; then
    log "  no GPU free, sleeping ${POLL_INT}s ..."
    sleep $POLL_INT
    continue
  fi
  pid=$(run_job $arch $fold $gpu)
  log "  launch $job on GPU $gpu (pid=$pid)"
  launched_pids+=($pid)
  i=$((i+1))
  sleep 15   # 给新进程时间占用显存避免下一轮还误判
done

log "All jobs queued; waiting for completion..."
for p in "${launched_pids[@]}"; do
  wait $p 2>/dev/null
done
log "===== schedule done ====="
ls -lh checkpoints/seg/*/best_model.pt checkpoints/cls/*/best_model.pt 2>/dev/null
