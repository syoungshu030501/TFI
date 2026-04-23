#!/usr/bin/env bash
# Stage4 VLM: Qwen3.5-9B LoRA SFT (multi-GPU model parallel via device_map=auto)
# 用法: bash scripts/run_vlm.sh "<gpu_list>"   例: bash scripts/run_vlm.sh "2,3,4,5,6"
set -euo pipefail
gpus=${1:-"2,3,4,5,6"}
cd "$(dirname "$0")/.."
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI
mkdir -p logs checkpoints/qwen35_9b
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_VISIBLE_DEVICES="$gpus"
log=logs/qwen35_9b.log
# 多卡模型并行 (device_map=auto)：base 18GB 切 5 卡 ~3.6GB/卡 + LoRA + 激活
# 最后一卡承担 lm_head + cross_entropy 峰值，多卡可以让它有 30+GB 空闲来吃这个峰
# 限制 image longest_edge=384²=147k px → 单图 ~144 vision tokens
python train_qwen35_9b.py \
    --model_name models/Qwen3.5-9B \
    --data_dir data/raw/train_resume \
    --augmented_dir data/vlm/caption_api_v3 \
    --real_ext_dir data/processed/real_ext \
    --output_dir checkpoints/qwen35_9b \
    --epochs 4 --batch_size 1 --grad_accum 16 \
    --lr 1e-4 --lora_r 64 --lora_alpha 128 \
    --device_map auto \
    --max_length 1536 \
    --max_image_pixels 147456 \
    --chunked_ce_size 256 \
    --inject_evidence --use_caption_clean --include_real_ext \
    --use_weighted_sampler \
    >"$log" 2>&1
