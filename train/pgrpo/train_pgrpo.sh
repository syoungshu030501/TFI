#!/usr/bin/env bash
# TFI v2 stage 3: P-GRPO (Pattern-aware GRPO)
# 参考 Veritas/self_scripts/train/train_pgrpo.sh
#
# 注意：reward funcs 我们用 ms-swift 内置 + 自定义 plugin
#   patternacc       : 答案是否命中标签
#   unifiedprm       : UnifiedReward-qwen-3b 文本质量打分（需先 deploy_reward_model）
#   multi_reason_format : 6 标签格式校验
#
# Usage:
#   bash train/pgrpo/train_pgrpo.sh <mipo_ckpt> [run_name] [dataset]
#
# GPU 0 上有历史 ECC 错误，必须跳过；P-GRPO 用 6 卡 (1-6)，留 7 给 reward server

set -euo pipefail
trap '' HUP
cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

MIPO_CKPT="${1:?usage: bash train/pgrpo/train_pgrpo.sh <mipo_ckpt> [run_name] [dataset]}"
RUN_NAME="${2:-v2pgrpo_$(date +%m%d%H%M)}"
DATASET="${3:-/mnt/nfs/young/TFI/data/v2/pgrpo.json}"
OUTPUT="${OUTPUT:-/mnt/nfs/young/TFI/runs/pgrpo/${RUN_NAME}}"
LOG_DIR="$PROJ/logs/v2_train"
mkdir -p "$LOG_DIR" "$OUTPUT"

# **GPU 0 历史 ECC 错误，永远跳过；6 卡 (1-6) 给训练，GPU 7 留给 reward server**
GPUS="${GPUS:-1,2,3,4,5,6}"
NPROC="${NPROC:-6}"
PLUGIN="${PLUGIN:-/mnt/nfs/young/TFI/code/Veritas/swift/plugin/prm.py}"

[[ -d "$MIPO_CKPT" ]] || { echo "ERROR: MiPO ckpt $MIPO_CKPT not found"; exit 1; }
[[ -f "$DATASET" ]] || { echo "ERROR: dataset $DATASET not found (run python data/build/build_v2_pgrpo.py first)"; exit 1; }

echo "==== TFI v2 P-GRPO ===="
echo "  mipo ckpt: $MIPO_CKPT"
echo "  dataset:   $DATASET"
echo "  plugin:    $PLUGIN"
echo "  output:    $OUTPUT"
echo

export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=$NPROC
export USE_AUG=false

LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

swift rlhf \
    --rlhf_type grpo \
    --external_plugins "$PLUGIN" \
    --reward_funcs patternacc unifiedprm multi_reason_format \
    --reward_weights 1.0 1.0 0.25 \
    --model "$MIPO_CKPT" \
    --model_type internvl3 \
    --template internvl2_5 \
    --dataset "$DATASET" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --use_vllm true \
    --vllm_device auto \
    --vllm_max_model_len 8192 \
    --num_infer_workers 2 \
    --train_type lora \
    --freeze_vit false \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_generations 4 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --eval_strategy "no" \
    --save_steps 100 \
    --save_total_limit 5 \
    --save_only_model true \
    --logging_steps 10 \
    --max_length 4096 \
    --max_completion_length 4096 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir "$OUTPUT" \
    --logging_dir "$OUTPUT/tb" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    --seed 42 \
    --beta 0.0 \
    2>&1 | tee "$LOG_FILE"

echo
echo "==== P-GRPO done. Output: $OUTPUT ===="
