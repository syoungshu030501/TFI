#!/usr/bin/env bash
# TFI v2 路线 A · Qwen3.5-9B Cold-Start SFT
#
# 与 train_sft.sh (InternVL3-8B / Veritas-Cold-Start) 并列，路线 A 切回
# Qwen3 家族底座以便：
#   1. 与 v1 LoRA 词表/family 对齐（v1 用 Qwen3.5-9B + LoRA r=64）
#   2. GKD 阶段与 teacher Qwen3.6-27B 同 tokenizer（无须词表交集）
#   3. vllm 0.7.3+ rollout 直接支持，FIPO 阶段无 InternVL3 兼容性风险
#
# 环境约束：transformers 4.49 不识别 model_type=qwen3_5；本脚本走 conda env "VLM"
#   - VLM env: torch 2.10+cu128 / transformers 5.5.4 / ms-swift 4.1.3 / peft 0.19.1
#   - GPU 0 ECC 永久禁用，默认 7 卡 (1-7)
#
# Usage:
#   bash train/sft/train_sft_qwen35.sh [run_name] [dataset_json]
#   默认: run_name=qwen35_$(date +%m%d%H%M)，dataset=/mnt/nfs/young/TFI/data/v2/sft_merged.json
set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate VLM

RUN_NAME="${1:-qwen35_$(date +%m%d%H%M)}"
DATASET="${2:-/mnt/nfs/young/TFI/data/v2/sft_merged.json}"
VAL_DATASET="${VAL_DATASET:-/mnt/nfs/young/TFI/data/v2/sft_val.json}"
MODEL="${MODEL:-/mnt/nfs/young/TFI/models/Qwen3.5-9B}"
OUTPUT="${OUTPUT:-/mnt/nfs/young/TFI/runs/sft/${RUN_NAME}}"
LOG_DIR="$PROJ/logs/v2_train"
mkdir -p "$LOG_DIR" "$OUTPUT"

GPUS="${GPUS:-1,2,3,4,5,6,7}"
NPROC="${NPROC:-7}"

[[ -f "$DATASET" ]] || { echo "ERROR: dataset $DATASET not found"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model $MODEL/config.json not found"; exit 1; }

N_TRAIN=$(python -c "import json;print(len(json.load(open('$DATASET'))))")
N_VAL=$(python -c "import json;print(len(json.load(open('$VAL_DATASET'))))" 2>/dev/null || echo 0)

echo "==== TFI v2 路线A · Qwen3.5-9B SFT ===="
echo "  model:    $MODEL"
echo "  dataset:  $DATASET ($N_TRAIN samples)"
echo "  val:      $VAL_DATASET ($N_VAL samples)"
echo "  output:   $OUTPUT"
echo "  GPUs:     $GPUS  ($NPROC processes)"
echo

export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=$NPROC
export MASTER_PORT="${MASTER_PORT:-12356}"
export MAX_PIXELS=${MAX_PIXELS:-1003520}    # Qwen3.5-VL: cap pixel budget per image
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}

LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

# ---- swift 4.1.3 sft  ----
# template=qwen3_5 / model_type=qwen3_5
# bbox 已在数据层归一化到 [0,1000]² (data/build/build_v2_sft.py)，模型空间无关
swift sft \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --dataset "$DATASET" \
    --val_dataset "$VAL_DATASET" \
    --num_train_epochs "${EPOCHS:-3}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
    --tuner_type lora \
    --freeze_vit true \
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
    --max_length "${MAX_LEN:-3072}" \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir "$OUTPUT" \
    --logging_dir "$OUTPUT/tb" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    --seed 42 \
    ${MAX_STEPS:+--max_steps $MAX_STEPS} \
    2>&1 | tee "$LOG_FILE"

echo
echo "==== SFT done. Output: $OUTPUT ===="
echo "logs: $LOG_FILE"
