#!/usr/bin/env bash
# Stage3 calibrator: 全 backend 5-fold CV 对比
set -euo pipefail
gpu=${1:-5}
cd "$(dirname "$0")/.."
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
mkdir -p logs
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
log=logs/calibrator.log
python train_calibrator.py \
    --compare_all --cv_folds 5 \
    --val_dir data/raw/val \
    --checkpoint_dir checkpoints \
    --img_size 768 --gpu "$gpu" \
    >"$log" 2>&1
