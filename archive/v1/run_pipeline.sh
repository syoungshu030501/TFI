#!/usr/bin/env bash
# 一键运行整套 TFI Pipeline。
# 假定: conda env TFI 已创建; models/Qwen3.5-9B 已下载; checkpoints/seg & cls 已存在。

set -euo pipefail

cd "$(dirname "$0")"

source /home/young/miniconda3/etc/profile.d/conda.sh
conda activate TFI

mkdir -p logs/{seg,vlm,data} cache checkpoints/calibrator checkpoints/qwen35_9b

# ─────────── 0) 数据体检（路径/数量是否齐） ───────────
python scripts/data/guard.py --strict

# ─────────── 1) 训 calibrator (只需一次, 用 data/raw/val) ───────────
if [ ! -f checkpoints/calibrator/calibrator.pkl ]; then
  echo "[1/3] training calibrator on data/raw/val ..."
  python train_calibrator.py --val_dir data/raw/val --backend xgb --gpu 0 \
      2>&1 | tee logs/vlm/train_calibrator.log
else
  echo "[1/3] calibrator exists, skip"
fi

# ─────────── 2) 训 Qwen3.5-9B (LoRA, 单卡) ───────────
if [ ! -d checkpoints/qwen35_9b/checkpoint-1 ] && [ ! -f checkpoints/qwen35_9b/adapter_config.json ]; then
  echo "[2/3] training Qwen3.5-9B (LoRA) ..."
  python train_qwen35_9b.py \
      --model_name models/Qwen3.5-9B \
      --output_dir checkpoints/qwen35_9b \
      --epochs 4 --lr 1e-4 \
      --batch_size 1 --grad_accum 16 \
      --gpu 0 \
      2>&1 | tee logs/vlm/train_qwen35_9b.log
else
  echo "[2/3] Qwen3.5-9B checkpoint exists, skip"
fi

# ─────────── 3) 推理 ───────────
echo "[3/3] running inference ..."
python inference.py --config config.yaml --gpu 0 \
    2>&1 | tee logs/vlm/inference.log

echo "DONE"
