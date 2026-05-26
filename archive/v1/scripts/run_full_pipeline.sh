#!/usr/bin/env bash
# 串行 watchdog:
#   1) 等当前 test/ 推理 pid 结束 (submit.csv 出)
#   2) 启动 val/ 推理 (独立 cache_val) -> submit_val.csv
#   3) 启动 score_official.py -> S_Det/S_Loc/S_Sim/S_Auto/S_Exp/S_Fin
# 用法: bash scripts/run_full_pipeline.sh <test_inference_pid> [val_gpus] [score_gpu]
#   例: bash scripts/run_full_pipeline.sh 1701353 "2,3,6" 7
set -euo pipefail
test_pid=${1:?"need test inference pid (ps -ef|grep 'python inference')"}
val_gpus=${2:-"2,3,6"}
score_gpu=${3:-7}

cd "$(dirname "$0")/.."
mkdir -p logs
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI

WLOG=logs/pipeline_watchdog.log
log() { echo "[$(date +'%F %T')] $*" | tee -a "$WLOG"; }

log "watchdog started, waiting for test pid=$test_pid to finish..."
while kill -0 "$test_pid" 2>/dev/null; do
    sleep 60
done
log "test inference pid=$test_pid finished."

# 检查 submit.csv 是否完整
if [ ! -s submit.csv ] || [ "$(wc -l < submit.csv)" -lt 500 ]; then
    log "WARN: submit.csv 不完整 ($(wc -l < submit.csv 2>/dev/null) lines)"
fi

# Step 2: val inference
log "launching val/ inference on GPUs=$val_gpus ..."
bash scripts/run_val_inference.sh "$val_gpus"
log "val inference done -> submit_val.csv ($(wc -l < submit_val.csv) lines)"

# Step 3: official score
log "launching official scoring on GPU=$score_gpu ..."
export CUDA_VISIBLE_DEVICES="$score_gpu"
unset HF_HUB_OFFLINE
export HF_ENDPOINT="https://hf-mirror.com"
python score_official.py \
    --pred_csv submit_val.csv \
    --val_dir data/raw/val \
    --gpu "$score_gpu" \
    --qwen_model qwen-max \
    --qwen_workers 6 \
    --out_json logs/score_official.json \
    --out_md logs/score_official.md \
    >>"$WLOG" 2>&1
log "scoring done. report -> logs/score_official.md"
echo ""
cat logs/score_official.md
