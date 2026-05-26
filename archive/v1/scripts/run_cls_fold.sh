#!/usr/bin/env bash
# 单 fold cls 训练
# 用法: bash scripts/run_cls_fold.sh <fold> <gpu> [extra_args...]
set -euo pipefail
fold=$1; gpu=$2; shift 2
cd "$(dirname "$0")/.."
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
mkdir -p logs/cls
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
log=logs/cls/efficientnet_fold${fold}.log
python train_classifier.py \
    --fold "$fold" --gpu "$gpu" \
    --img_size 512 --batch_size 8 \
    --epochs 30 --patience 8 \
    --include_synth --include_real_ext \
    --use_weighted_sampler \
    "$@" \
    >"$log" 2>&1
