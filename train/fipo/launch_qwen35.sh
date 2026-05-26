#!/usr/bin/env bash
# FIPO 路线 A · Qwen3.5-9B + verl 0.8 + future_kl_loss
#
# 与 launch.sh (InternVL3-8B / Veritas-Cold-Start) 并列；路线 A 切回 Qwen3 家族：
#   - student/policy: Qwen3.5-9B + qwen35_v2 SFT LoRA (merged)
#   - env: VLM (torch 2.10+cu128 / vllm 0.19.1 / verl 0.8.0.dev / transformers 5.5.4)
#   - 训推一致性: bbox 已在数据层归一化到 [0,1000]² (与 train_sft_qwen35.sh 同约定)
#
# 依赖:
#   1. SFT 已完成 + merge_lora 产出 HF dump:
#        /mnt/nfs/young/TFI/models/qwen35_v2_1441
#   2. FIPO data 已 prepare:
#        bash <conda activate VLM> && python -m train.fipo.prepare_fipo_data \
#             --in /mnt/nfs/young/TFI/data/v2/sft.json \
#             --out_dir data/fipo_qwen35
#
# Usage:
#   bash train/fipo/launch_qwen35.sh
set -euo pipefail
trap '' HUP

cleanup() {
    pkill -9 -f VLLM 2>/dev/null || true
    pkill -9 -f "ray::"  2>/dev/null || true
    pkill -9 -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

ENV_NAME="${ENV_NAME:-VLM}"
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export FIPO_PATCH_VERL="${FIPO_PATCH_VERL:-1}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

PROJECT_NAME="${PROJECT_NAME:-TFI-forgery-detection}"
EXP_NAME="${EXP_NAME:-FIPO-qwen35-rule-reward}"
LOGGERS="${LOGGERS:-console}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/young/TFI/models/qwen35_v2_1441}"
TRAIN_FILE="${TRAIN_FILE:-${PWD}/data/fipo_qwen35/train.parquet}"
VAL_FILE="${VAL_FILE:-${PWD}/data/fipo_qwen35/val.parquet}"
CKPTS_DIR="${CKPTS_DIR:-/mnt/nfs/young/TFI/runs/fipo/${EXP_NAME}}"

# Qwen3.5-VL 比 InternVL3-8B 节省视觉 token，把 prompt 上限调小到 4096 节省 KV cache
N_GPUS="${N_GPUS:-7}"
BATCH_SIZE="${BATCH_SIZE:-6}"
N_RESP="${N_RESP:-8}"
MINI_BSZ="${MINI_BSZ:-3}"
MICRO_BSZ_PER_GPU="${MICRO_BSZ_PER_GPU:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"
GEN_TP="${GEN_TP:-1}"

ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
REF_STRATEGY="${REF_STRATEGY:-fsdp2}"

LOSS_MODE="${LOSS_MODE:-future_kl}"
export FIPO_DECAY_RATE="${FIPO_DECAY_RATE:-12.0}"
export FIPO_CHUNK_SIZE="${FIPO_CHUNK_SIZE:-128}"
export FIPO_FKL_CLIP_RATIO="${FIPO_FKL_CLIP_RATIO:-0.2}"
export FIPO_FKL_CLIP_HIGH_ONLY="${FIPO_FKL_CLIP_HIGH_ONLY:-false}"
export FIPO_SAFETY_THRESH="${FIPO_SAFETY_THRESH:-4.0}"

mkdir -p "${CKPTS_DIR}"

python -m train.fipo.main_fipo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.image_key=images \
    data.truncation=left \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.train_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.ref.strategy=${REF_STRATEGY} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEM:-0.55} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.response_length=${MAX_RESP_LEN} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.min_dynamic_patch=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_dynamic_patch=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.dynamic_image_size=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.use_thumbnail=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=TFIAuditRewardManager \
    reward.reward_manager.module.path="${PWD}/train/fipo/verl_patches/reward_manager.py" \
    reward.reward_model.enable=False \
    trainer.logger="$(python -c "import sys,json; print(json.dumps([s.strip() for s in sys.argv[1].split(',') if s.strip()]))" "${LOGGERS}")" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=40 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
