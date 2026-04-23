#!/usr/bin/env bash
# Stage5 推理: 在 data/raw/test/Image 上跑完整 5 stage 流水线 -> submit.csv
# 用法: bash scripts/run_inference.sh "<gpu_list>"   例: "2,3,6"
set -euo pipefail
gpus=${1:-"2,3,6"}
cd "$(dirname "$0")/.."
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
mkdir -p logs cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log=logs/inference.log
python inference.py \
    --config config.yaml \
    --gpu "$gpus" \
    --output submit.csv \
    >"$log" 2>&1
