#!/usr/bin/env bash
# 合并 4 个 shard 的 predictions.csv → 一个 csv → 跑 score_official + judge
set -euo pipefail
trap '' HUP

cd "$(dirname "$0")/../.."

source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate VLM

OUT_DIR="${OUT_DIR:-eval/baseline/results/sft_v2_qwen35}"
mkdir -p "$OUT_DIR/score" "$OUT_DIR/raw"

# 1) merge 4 shard CSVs (header + body)
echo "==== merge 4 shard CSVs ===="
HEAD_LINE="image_name,label,location,explanation"
echo "$HEAD_LINE" > "$OUT_DIR/predictions.csv"
for i in 0 1 2 3; do
    SH="eval/baseline/results/sft_v2_qwen35_shard_$i/predictions.csv"
    [[ -f "$SH" ]] || { echo "ERROR: $SH not found"; exit 1; }
    tail -n +2 "$SH" >> "$OUT_DIR/predictions.csv"
    # also merge raw cache
    cp -n eval/baseline/results/sft_v2_qwen35_shard_$i/raw/* "$OUT_DIR/raw/" 2>/dev/null || true
done
N=$(($(wc -l < "$OUT_DIR/predictions.csv") - 1))
echo "  merged $N rows → $OUT_DIR/predictions.csv"

# 2) score_official
echo "==== score_official ===="
python eval/score_official.py \
    --pred_csv "$OUT_DIR/predictions.csv" \
    --val_dir  data/raw/val \
    --out_json "$OUT_DIR/score/score.json" \
    --out_md   "$OUT_DIR/score/score.md" \
    --qwen_model none \
    --gpu 0 \
    2>&1 | tee "$OUT_DIR/score.log" | tail -20

echo
echo "==== DONE ===="
echo "  predictions: $OUT_DIR/predictions.csv"
echo "  score:       $OUT_DIR/score/"
