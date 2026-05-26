#!/usr/bin/env bash
# Run all 3 prompt-only variants then judge. Robust to disconnect / kill.
#
# Usage: bash tools/baseline/run_all.sh
set -euo pipefail
trap '' HUP

cd "$(dirname "$0")/../.."
PROJ_ROOT="$(pwd)"
LOG_DIR="$PROJ_ROOT/tools/baseline/results/_logs"
mkdir -p "$LOG_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# === Stage 1: 3 prompt-only variants on Qwen3.5-9B (single GPU each, sequential)
# We run sequentially on GPU 1 to leave 4-7 free for the judge later (which needs TP=4).
# Estimated time: 3 * 1.5h = 4.5h
GPU_FOR_BASELINE="1"
for MODE in zs fs cot; do
    OUT="$PROJ_ROOT/tools/baseline/results/$MODE/predictions.csv"
    if [[ -f "$OUT" ]]; then
        echo "[$MODE] $OUT exists, skipping (delete to force rerun)"
        continue
    fi
    echo "===== running prompt-only baseline: $MODE on GPU $GPU_FOR_BASELINE ====="
    python tools/baseline/prompt_only_baseline.py \
        --mode "$MODE" \
        --gpu  "$GPU_FOR_BASELINE" \
        2>&1 | tee "$LOG_DIR/baseline_${MODE}.log"
done

# === Stage 2: judge with R1-Distill-Llama-70B on GPU 4-7 (TP=4)
echo "===== running absolute-scoring judge ====="
JUDGE_MODEL="/mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b"
if [[ ! -d "$JUDGE_MODEL" ]] || [[ ! -f "$JUDGE_MODEL/config.json" ]]; then
    echo "ERROR: judge model not found at $JUDGE_MODEL — wait for download to finish."
    echo "  tail -f /mnt/nfs/young/TFI/logs/download_judge.log"
    exit 1
fi

python tools/baseline/judge_absolute_scoring.py \
    --pred_csvs sft="$PROJ_ROOT/submit_val.csv" \
                zs="$PROJ_ROOT/tools/baseline/results/zs/predictions.csv" \
                fs="$PROJ_ROOT/tools/baseline/results/fs/predictions.csv" \
                cot="$PROJ_ROOT/tools/baseline/results/cot/predictions.csv" \
    --judge_model "$JUDGE_MODEL" \
    --gpus 4,5,6,7 \
    --out_dir "$PROJ_ROOT/tools/baseline/results/judge" \
    2>&1 | tee "$LOG_DIR/judge.log"

echo "===== ALL DONE ====="
echo "Summary: $PROJ_ROOT/tools/baseline/results/judge/summary.csv"
echo "Report:  $PROJ_ROOT/tools/baseline/results/judge/report.md"
