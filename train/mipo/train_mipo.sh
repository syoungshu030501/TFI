#!/usr/bin/env bash
# TFI v2 stage 2: MiPO (Mixed Preference Optimization)
# 参考 Veritas/self_scripts/train/train_mipo.sh
#
# 输入：
#   - SFT 后的模型 ckpt (传入 $1 或 MODEL 环境变量)
#   - mipo 偏好数据 jsonl: 每条 {chosen, rejected} 对
#
# 用 swift rlhf --rlhf_type dpo + rpo_alpha=1.0 + beta=0.0 实现 MiPO 损失
#
# Usage:
#   bash train/mipo/train_mipo.sh <sft_ckpt_path> [run_name] [dataset]
#
# GPU 0 上有历史 ECC 错误，必须跳过；默认 7 卡 (1-7)

set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

SFT_CKPT="${1:?usage: bash train/mipo/train_mipo.sh <sft_ckpt> [run_name] [dataset]}"
RUN_NAME="${2:-v2mipo_$(date +%m%d%H%M)}"
DATASET="${3:-/mnt/nfs/young/TFI/data/v2/mipo.json}"
OUTPUT="${OUTPUT:-/mnt/nfs/young/TFI/runs/mipo/${RUN_NAME}}"
LOG_DIR="$PROJ/logs/v2_train"
mkdir -p "$LOG_DIR" "$OUTPUT"

# **GPU 0 历史 ECC 错误，永远跳过**
GPUS="${GPUS:-1,2,3,4,5,6,7}"
NPROC="${NPROC:-7}"

[[ -d "$SFT_CKPT" ]] || { echo "ERROR: SFT ckpt $SFT_CKPT not found"; exit 1; }
[[ -f "$DATASET" ]] || { echo "ERROR: dataset $DATASET not found (run python data/build/build_v2_mipo.py first)"; exit 1; }

echo "==== TFI v2 MiPO ===="
echo "  sft ckpt: $SFT_CKPT"
echo "  dataset:  $DATASET"
echo "  output:   $OUTPUT"
echo "  GPUs:     $GPUS"
echo

export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=$NPROC
export MASTER_PORT="${MASTER_PORT:-12352}"
export USE_AUG=false

LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

swift rlhf \
    --rlhf_type dpo \
    --model "$SFT_CKPT" \
    --model_type internvl3 \
    --template internvl2_5 \
    --dataset "$DATASET" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --train_type lora \
    --freeze_vit false \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --save_only_model true \
    --logging_steps 10 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir "$OUTPUT" \
    --logging_dir "$OUTPUT/tb" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    --seed 42 \
    --rpo_alpha 1.0 \
    --beta 0.0 \
    2>&1 | tee "$LOG_FILE"

echo
echo "==== MiPO done. Output: $OUTPUT ===="
