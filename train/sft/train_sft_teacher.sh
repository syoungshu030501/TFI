#!/usr/bin/env bash
# TFI v2 · Teacher SFT: Qwen3.6-27B on same data as student
#
# Purpose:
#   Fine-tune the 27B teacher on the SAME 1441 SFT data + 6-tag CoT template
#   as the student (Qwen3.5-9B / InternVL3-8B). After SFT, the teacher serves as:
#     1. GKD teacher for student distillation (future stage)
#     2. Verifier/reward-model for FIPO (R_grm hook, replacing rule-only)
#
# Environment: VLM env (torch 2.10+cu128 / transformers 5.5.4 / ms-swift 4.1.3)
# GPU constraint: GPU 0 OFF-LIMITS (ECC errors).
#   Uses FSDP full_shard + CPU offload so 27B parameters are sharded across GPUs.
#   LoRA r=32, per_device_bsz=1, grad_accum=16.
#
# Usage:
#   bash train/sft/train_sft_teacher.sh [run_name] [dataset_json]
set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate VLM

RUN_NAME="${1:-teacher_qwen36_27b_$(date +%m%d%H%M)}"
DATASET="${2:-/mnt/nfs/young/TFI/data/v2/sft_merged.json}"
VAL_DATASET="${VAL_DATASET:-/mnt/nfs/young/TFI/data/v2/sft_val.json}"
MODEL="${MODEL:-/mnt/nfs/young/TFI/models/Qwen3.6-27B}"
OUTPUT="${OUTPUT:-/mnt/nfs/young/TFI/runs/sft/${RUN_NAME}}"
LOG_DIR="$PROJ/logs/v2_train"
mkdir -p "$LOG_DIR" "$OUTPUT"

GPUS="${GPUS:-1,2,3,4,5,6,7}"
NPROC="${NPROC:-7}"

[[ -f "$DATASET" ]] || { echo "ERROR: dataset $DATASET not found"; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model $MODEL/config.json not found"; exit 1; }

N_TRAIN=$(python -c "import json;print(len(json.load(open('$DATASET'))))")
N_VAL=$(python -c "import json;print(len(json.load(open('$VAL_DATASET'))))" 2>/dev/null || echo 0)

echo "==== TFI v2 · Teacher SFT (Qwen3.6-27B) ===="
echo "  model:    $MODEL"
echo "  dataset:  $DATASET ($N_TRAIN samples)"
echo "  val:      $VAL_DATASET ($N_VAL samples)"
echo "  output:   $OUTPUT"
echo "  GPUs:     $GPUS  ($NPROC processes)"
echo

export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=$NPROC
export MASTER_PORT="${MASTER_PORT:-12358}"
export MAX_PIXELS=${MAX_PIXELS:-1003520}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
export TOKENIZERS_PARALLELISM=false

LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

# FSDP full_shard + offload: shards model params across GPUs + offloads to CPU.
# Each GPU only holds 1/N of the 27B params (~8GB/GPU for weights with N=7).
# LoRA r=32 on all-linear — 27B has 3x capacity vs 9B, lower rank sufficient.
# lr=2e-5 — larger model needs lower lr for stability.
swift sft \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --dataset "$DATASET" \
    --val_dataset "$VAL_DATASET" \
    --fsdp fsdp2 \
    --num_train_epochs "${EPOCHS:-3}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-16}" \
    --tuner_type lora \
    --freeze_vit true \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --save_only_model false \
    --logging_steps 5 \
    --max_length "${MAX_LEN:-3072}" \
    --gradient_checkpointing true \
    --output_dir "$OUTPUT" \
    --logging_dir "$OUTPUT/tb" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    --seed 42 \
    ${MAX_STEPS:+--max_steps $MAX_STEPS} \
    2>&1 | tee "$LOG_FILE"

echo
echo "==== Teacher SFT done. Output: $OUTPUT ===="
echo "logs: $LOG_FILE"
echo
echo "Next: merge LoRA"
echo "  conda activate VLM"
echo "  swift export --adapters $OUTPUT/v0-*/checkpoint-XXX --merge_lora true --output_dir /mnt/nfs/young/TFI/models/teacher_qwen36_v2"
