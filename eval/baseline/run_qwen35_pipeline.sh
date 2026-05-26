#!/usr/bin/env bash
# 路线 A · Qwen3.5-9B SFT 全套评测一键脚本
#
#   1. swift export merge-lora → /mnt/nfs/young/TFI/models/qwen35_v2_1441
#   2. val/200 推理 (eval/baseline/sft_v2_inference_qwen35.py)
#   3. score_official (S_Det / S_Loc / S_Sim / S_Exp / S_Fin)
#   4. R1-70B judge (4 维 1-10) — 默认跳过，加 --with-judge 才跑（占 4 卡 TP=4）
#
# Usage:
#   bash eval/baseline/run_qwen35_pipeline.sh
#   bash eval/baseline/run_qwen35_pipeline.sh --with-judge
#
# 依赖：SFT 必须已经完成 (有 checkpoint-N 目录)。
set -euo pipefail
trap '' HUP

cd "$(dirname "$0")/../.."
PROJ="$(pwd)"

source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate VLM

RUN_NAME="${RUN_NAME:-qwen35_v2_1441}"
RUN_DIR=$(ls -d /mnt/nfs/young/TFI/runs/sft/${RUN_NAME}/v0-* 2>/dev/null | tail -1)
[[ -d "$RUN_DIR" ]] || { echo "ERROR: no run dir found for $RUN_NAME"; exit 1; }

LAST_CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
[[ -d "$LAST_CKPT" ]] || { echo "ERROR: no checkpoint in $RUN_DIR"; exit 1; }
echo "[ckpt] $LAST_CKPT"

MERGED_DIR="${MERGED_DIR:-/mnt/nfs/young/TFI/models/${RUN_NAME}}"
INFER_OUT="${INFER_OUT:-$PROJ/eval/baseline/results/sft_v2_qwen35}"
INFER_GPU="${INFER_GPU:-1}"

# ---- 1. merge LoRA → HF dump ----
if [[ ! -f "$MERGED_DIR/config.json" ]]; then
    echo "==== [1/4] swift export merge-lora ===="
    swift export \
        --adapters "$LAST_CKPT" \
        --merge_lora true \
        --output_dir "$MERGED_DIR" \
        --torch_dtype bfloat16
    echo "  → $MERGED_DIR"
else
    echo "[1/4] merged dir already exists at $MERGED_DIR; skipping merge"
fi

# ---- 2. val/200 推理 ----
if [[ ! -f "$INFER_OUT/predictions.csv" ]]; then
    echo "==== [2/4] val/200 推理 (GPU $INFER_GPU) ===="
    python eval/baseline/sft_v2_inference_qwen35.py \
        --model "$MERGED_DIR" \
        --gpu "$INFER_GPU" \
        --out_dir "$INFER_OUT"
else
    echo "[2/4] predictions.csv already exists; skipping inference"
fi

# ---- 3. score_official ----
echo "==== [3/4] score_official ===="
mkdir -p "$INFER_OUT/score"
python eval/score_official.py \
    --pred_csv "$INFER_OUT/predictions.csv" \
    --val_dir  "$PROJ/data/raw/val" \
    --out_json "$INFER_OUT/score/score.json" \
    --out_md   "$INFER_OUT/score/score.md" \
    --qwen_model none \
    --gpu      "$INFER_GPU" \
    2>&1 | tee "$INFER_OUT/score.log"

# ---- 4. R1-70B Judge (可选) ----
if [[ "${1:-}" == "--with-judge" ]]; then
    echo "==== [4/4] R1-70B judge (TP=4, GPU 4-7) ===="
    conda activate TFI_judge
    python eval/baseline/judge_absolute_scoring.py \
        --pred_csvs "qwen35_v2"="$INFER_OUT/predictions.csv" \
        --judge_model /mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b \
        --gpus 4,5,6,7 \
        --out_dir "$INFER_OUT/judge" \
        2>&1 | tee "$INFER_OUT/judge.log"
    conda activate VLM
else
    echo "[4/4] judge SKIPPED (pass --with-judge to run R1-70B; ~30 min on 4 GPUs)"
fi

echo
echo "==== ALL DONE ===="
echo "  predictions: $INFER_OUT/predictions.csv"
echo "  score:       $INFER_OUT/score/"
[[ -d "$INFER_OUT/judge" ]] && echo "  judge:       $INFER_OUT/judge/"
