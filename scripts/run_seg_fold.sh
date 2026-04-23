#!/usr/bin/env bash
# 单 fold seg 训练（被 launch_seg_all.sh 调用）。
# 用法: bash scripts/run_seg_fold.sh <arch> <fold> <gpu> [extra_args...]
set -euo pipefail
arch=$1; fold=$2; gpu=$3; shift 3
cd "$(dirname "$0")/.."
# 忽略 SIGHUP/SIGINT 让 nohup 真正脱离父进程
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
mkdir -p logs/seg
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
log=logs/seg/${arch}_fold${fold}.log
python train_seg_ensemble.py \
    --arch "$arch" --fold "$fold" --gpu "$gpu" \
    --img_size 768 --batch_size 4 \
    --epochs 50 --patience 10 \
    --include_synth --include_real_ext \
    --use_weighted_sampler \
    "$@" \
    >"$log" 2>&1
