#!/usr/bin/env bash
# M(-1) ceiling baseline: Qwen3.6-27B prompt-only (zs + cot)
#
# 目的：评估"换大模型 + prompt"能否逼近 / 超过 SFT (overall=7.991, S_Fin=0.9034)。
# - 跑通 → 决定 v2 OPD 是否仍走 9B 蒸馏路线
# - zs 体现纯实力，cot 体现 reasoning 加成；fs 已被证明无收益（见 results/README.md），跳过
#
# Usage:  bash tools/baseline/run_ceiling.sh
# Resume: 同 v1，每张图 cache 在 raw/<image_name>.json，断电可续

set -euo pipefail
trap '' HUP

cd "$(dirname "$0")/../.."
PROJ_ROOT="$(pwd)"
LOG_DIR="$PROJ_ROOT/tools/baseline/results/_logs"
mkdir -p "$LOG_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- 0. 解析 Qwen3.6-27B 权重路径 ----
# modelscope 默认下到 _cache_qwen36/Qwen/Qwen3.6-27B/
CANDIDATES=(
    "/mnt/nfs/young/TFI/models/Qwen3.6-27B"
    "/mnt/nfs/young/TFI/models/_cache_qwen36/Qwen/Qwen3.6-27B"
)
MODEL=""
for c in "${CANDIDATES[@]}"; do
    if [[ -f "$c/config.json" ]]; then
        MODEL="$c"; break
    fi
done
if [[ -z "$MODEL" ]]; then
    echo "ERROR: Qwen3.6-27B config.json not found in any of:"
    for c in "${CANDIDATES[@]}"; do echo "  - $c"; done
    echo "Hint: tail -f /home/young/TFI/logs/download_qwen36.log"
    exit 1
fi
echo "[ceiling] using model: $MODEL"

# ---- 1. 双卡 device_map=auto ----
# Qwen3.6-27B BF16 ≈ 54GB，单张 L20 (46GB) 装不下
# 用 GPU 1,2（GPU 0 ECC 坏，按 README §五 的算力分配）
GPUS_FOR_CEILING="1,2"
MAX_PER_GPU="42GiB"

# ---- 2. 输出隔离 ----
# 不复用 results/zs/  results/cot/（那是 Qwen3.5-9B 的），改用 _qwen36 后缀
OUT_BASE="$PROJ_ROOT/tools/baseline/results_qwen36"
mkdir -p "$OUT_BASE"

# ---- 3. 跑 zs + cot ----
for MODE in zs cot; do
    OUT="$OUT_BASE/$MODE/predictions.csv"
    if [[ -f "$OUT" ]]; then
        echo "===== [SKIP] $MODE: $OUT exists (delete to force rerun) ====="
        continue
    fi
    echo "===== running ceiling baseline: Qwen3.6-27B $MODE on GPU $GPUS_FOR_CEILING ====="
    python tools/baseline/prompt_only_baseline.py \
        --mode "$MODE" \
        --model "$MODEL" \
        --gpu  "$GPUS_FOR_CEILING" \
        --device_map auto \
        --max_memory_per_gpu "$MAX_PER_GPU" \
        --out_base "$OUT_BASE" \
        2>&1 | tee "$LOG_DIR/ceiling_qwen36_${MODE}.log"
done

echo ""
echo "===== ceiling inference DONE ====="
echo "Outputs:"
echo "  $OUT_BASE/zs/predictions.csv"
echo "  $OUT_BASE/cot/predictions.csv"
echo ""
echo "Next: 加入 judge 对比（GPU 4-7 TP=4，70B judge 已就位）"
echo ""
cat <<EOF
python tools/baseline/judge_absolute_scoring.py \\
    --pred_csvs sft="$PROJ_ROOT/submit_val.csv" \\
                zs="$PROJ_ROOT/tools/baseline/results/zs/predictions.csv" \\
                cot="$PROJ_ROOT/tools/baseline/results/cot/predictions.csv" \\
                qwen36_zs="$OUT_BASE/zs/predictions.csv" \\
                qwen36_cot="$OUT_BASE/cot/predictions.csv" \\
    --judge_model /mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b \\
    --gpus 4,5,6,7 \\
    --out_dir tools/baseline/results/judge \\
    2>&1 | tee "$LOG_DIR/judge_with_ceiling.log"
EOF
