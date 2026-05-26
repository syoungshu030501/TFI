#!/usr/bin/env bash
# 在 val/ 200 张上跑完整 inference (得到 explanation)，供 score_official.py 评分
# 用法: bash scripts/run_val_inference.sh "<gpu_list>"  例: "2,3,6"
set -euo pipefail
gpus=${1:-"2,3,6"}
cd "$(dirname "$0")/.."
trap '' HUP INT
source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI

mkdir -p logs cache_val data/raw/val_flat/Image
# 把 Black/White 下所有 Image 软链到 val_flat (幂等)
for sub in Black White; do
    for f in data/raw/val/$sub/Image/*; do
        [ -e "$f" ] || continue
        ln -sf "../../val/$sub/Image/$(basename "$f")" "data/raw/val_flat/Image/$(basename "$f")"
    done
done
n=$(ls data/raw/val_flat/Image | wc -l)
echo "[ok] flattened val: $n images"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log=logs/inference_val.log

# cache_dir 用独立的，不污染 test/ 的缓存
python inference.py \
    --config config.yaml \
    --gpu "$gpus" \
    --test_dir data/raw/val_flat/Image \
    --cache_dir cache_val \
    --output submit_val.csv \
    >"$log" 2>&1

echo "[done] submit_val.csv ready, run: python score_official.py --pred_csv submit_val.csv"
