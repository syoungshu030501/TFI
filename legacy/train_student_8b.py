"""
学生模型训练: Qwen3-VL-8B-Thinking 全量微调
使用原始 Caption + 教师增强 Caption 联合训练

用法:
  # 多卡训练 (全量微调)
  accelerate launch --num_processes 4 train_student_8b.py

  # 单卡训练
  python train_student_8b.py --gpu 0
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from qwen_vl_utils import process_vision_info

from dataset import VLMSFTDataset
from train_teacher import VLMDataCollator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Thinking")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "train"))
    parser.add_argument("--augmented_dir", type=str, default=str(PROJECT_ROOT / "augmented_data"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints" / "student_8b"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 加载模型 (全量微调)
    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 梯度检查点
    model.gradient_checkpointing_enable()

    # 数据集: 原始 + 增强
    train_dataset = VLMSFTDataset(args.data_dir)
    print(f"Original training samples: {len(train_dataset)}")

    # 加载增强数据 (如果有)
    aug_dir = Path(args.augmented_dir)
    if aug_dir.exists():
        aug_dataset = VLMSFTDataset(args.data_dir, augmented_captions_dir=str(aug_dir))
        # aug_dataset 已包含原始+增强, 直接使用
        if len(aug_dataset) > len(train_dataset):
            train_dataset = aug_dataset
            print(f"With augmentation: {len(train_dataset)} samples")

    # 数据整理器
    collator = VLMDataCollator(processor, max_length=args.max_length)

    # 训练参数 (全量微调)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("Starting full fine-tuning...")
    trainer.train()

    # 保存完整模型
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Student model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
