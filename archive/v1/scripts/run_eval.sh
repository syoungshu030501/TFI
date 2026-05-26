#!/usr/bin/env bash
# 评测：在 val/ 上跑 segformer 集成 + cls 投票 + calibrator (xgb)，多尺度 TTA
set -euo pipefail
gpu=${1:-3}
cd "$(dirname "$0")/.."
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
mkdir -p logs
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
log=logs/eval.log
# 当前可用 arch：仅 segformer (convnext/maxvit 已删)
# multiscale=True：[640,768,896] 三尺度 TTA
python evaluate.py \
    --val_dir data/raw/val \
    --checkpoint_dir checkpoints \
    --cache_dir cache \
    --seg_arch segformer_only \
    --calibrator xgb \
    --use_cls --use_tta --multiscale \
    --seg_thresh 0.3 \
    --gpu "$gpu" \
    --out_md logs/ablation.md \
    >"$log" 2>&1
