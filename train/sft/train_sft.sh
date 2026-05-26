#!/usr/bin/env bash
# TFI v2 Cold-Start SFT
# 参考 Veritas/self_scripts/train/train_sft.sh，针对 TFI 任务调整：
#   - model: Veritas-Cold-Start (InternVL3-8B 已在 HydraFake 上预热)
#   - dataset: 我们自己合并的 v2/sft.json
#   - max_length: 3072 (中文 + 多 bbox 比英文长)
#   - lora_rank: 64 (论文用 128，但样本量小先用 64)
#   - epochs: 3
#
# Usage:
#   bash train/sft/train_sft.sh [run_name] [dataset_json]
#   默认: run_name=v2sft_$(date +%m%d%H%M)，dataset=/mnt/nfs/young/TFI/data/v2/sft.json
#
# GPU 0 上有历史 ECC 错误，必须跳过；默认 7 卡 (1-7)

set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

RUN_NAME="${1:-v2sft_$(date +%m%d%H%M)}"
DATASET="${2:-/mnt/nfs/young/TFI/data/v2/sft.json}"
VAL_DATASET="${VAL_DATASET:-/mnt/nfs/young/TFI/data/v2/sft_val.json}"
MODEL="${MODEL:-/mnt/nfs/young/TFI/models/Veritas-Cold-Start}"
OUTPUT="${OUTPUT:-/mnt/nfs/young/TFI/runs/sft/${RUN_NAME}}"
LOG_DIR="$PROJ/logs/v2_train"
mkdir -p "$LOG_DIR" "$OUTPUT"

# ---- GPU 分配 ----
# **注意：GPU 0 历史 ECC 错误，永远跳过。默认 7 卡 (1-7)**
GPUS="${GPUS:-1,2,3,4,5,6,7}"
NPROC="${NPROC:-7}"

# ---- 校验 ----
[[ -f "$DATASET" ]] || { echo "ERROR: dataset $DATASET not found"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model $MODEL/config.json not found"; exit 1; }

echo "==== TFI v2 SFT ===="
echo "  model:    $MODEL"
echo "  dataset:  $DATASET ($(jq length "$DATASET" 2>/dev/null || python -c "import json;print(len(json.load(open('$DATASET'))))" 2>/dev/null) samples)"
echo "  val:      $VAL_DATASET"
echo "  output:   $OUTPUT"
echo "  GPUs:     $GPUS  ($NPROC processes)"
echo

export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=$NPROC
export MASTER_PORT="${MASTER_PORT:-12355}"
export USE_AUG=false

LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

swift sft \
    --model "$MODEL" \
    --model_type internvl3 \
    --template internvl2_5 \
    --dataset "$DATASET" \
    --val_dataset "$VAL_DATASET" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --max_length 3072 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir "$OUTPUT" \
    --logging_dir "$OUTPUT/tb" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo
echo "==== SFT done. Output: $OUTPUT ===="
echo "logs: $LOG_FILE"
