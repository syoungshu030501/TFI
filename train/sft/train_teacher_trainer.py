#!/usr/bin/env python
"""Minimal Qwen3.6-27B teacher SFT using transformers Trainer + peft LoRA.

Bypasses ms-swift (which doesn't shard 27B properly with DDP/FSDP+LoRA).
Uses accelerate `device_map=auto` to split model across GPUs, single-process.

Usage:
    conda activate VLM
    python train/sft/train_teacher_trainer.py \
        --model_path /mnt/nfs/young/TFI/models/Qwen3.6-27B \
        --data_path /mnt/nfs/young/TFI/data/v2/sft_merged.json \
        --val_path /mnt/nfs/young/TFI/data/v2/sft_val.json \
        --output_dir /mnt/nfs/young/TFI/runs/sft/teacher_qwen36_trainer \
        --epochs 3 --lr 2e-5 --lora_rank 32
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = (
    "你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。\n\n"
    "首先用 <fast> </fast> 标签给出第一直觉判断；\n"
    "然后用 <reasoning> </reasoning> 标签给出详细取证推理（高难度样本可在其中包含"
    " <planning> 规划与 <reflection> 自校验）；\n"
    "接着用 <conclusion> </conclusion> 标签给出综合结论，对疑似篡改图必须用 "
    "<bbox>x1,y1,x2,y2</bbox> 或 <region>区域文字描述</region> 标注疑似篡改区域，"
    "其中 bbox 坐标已归一化到 [0,1000]×[0,1000]（左上原点，x1<x2，y1<y2）；\n"
    "最后用 <answer>real|fake</answer> 给出最终判断（仅二选一）。"
)
USER_PROMPT = "<image>请判断该图像的真实性，并按规定标签格式输出分析。"


def load_sft_json(path: str, max_length: int = 4096, est_img_tokens: int = 1500) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    records = []
    skipped = 0
    for sample in data:
        messages = sample.get("messages", [])
        images = sample.get("images", [])
        if not messages or not images:
            continue

        total_chars = sum(len(m["content"]) for m in messages)
        est_tokens = int(total_chars * 0.7) + est_img_tokens + 200
        if est_tokens > max_length:
            skipped += 1
            continue

        convo = []
        for msg in messages:
            convo.append({"role": msg["role"], "content": msg["content"]})

        records.append({
            "messages": convo,
            "image_path": images[0],
            "label": sample.get("label", 0),
        })
    print(f"  loaded {len(records)} samples (skipped {skipped} too long, est>{max_length} tok)")
    return records


class SFTCollator:
    def __init__(self, processor, max_length: int = 3072):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        texts = []
        images = []
        for item in batch:
            msgs = item["messages"]
            chat = []
            for msg in msgs:
                role = msg["role"]
                content = msg["content"]
                # Truncate long assistant content to keep total within limits
                if role == "assistant" and len(content) > 4000:
                    answer_match = re.search(r"<answer>.*?</answer>", content, re.DOTALL)
                    if answer_match:
                        content = content[:4000] + "…</reasoning>\n<conclusion>（已截断）</conclusion>\n<answer>" + answer_match.group(0)
                    else:
                        content = content[:4000]
                if role == "user":
                    chat.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": content.replace("<image>", "").strip()},
                        ],
                    })
                else:
                    chat.append({"role": role, "content": content})

            texts.append(chat)

            img = Image.open(item["image_path"]).convert("RGB")
            images.append(img)

        formatted = self.processor.apply_chat_template(
            texts, tokenize=False, add_generation_prompt=False
        )
        processed = self.processor(
            text=formatted if isinstance(formatted, list) else [formatted],
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        labels = processed["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        processed["labels"] = labels
        return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/mnt/nfs/young/TFI/models/Qwen3.6-27B")
    ap.add_argument("--data_path", default="/mnt/nfs/young/TFI/data/v2/sft_merged.json")
    ap.add_argument("--val_path", default="/mnt/nfs/young/TFI/data/v2/sft_val.json")
    ap.add_argument("--output_dir", default="/mnt/nfs/young/TFI/runs/sft/teacher_qwen36_trainer")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=3072)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--log_steps", type=int, default=5)
    ap.add_argument("--max_pixels", type=int, default=5*256*28*28)  # 1003520 → ~1.3k img tokens (matches student SFT)
    args = ap.parse_args()

    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=args.max_pixels,
    )

    # Load with device_map=auto — accelerate splits across available GPUs
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze vision encoder
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Estimate image tokens from max_pixels: (max_pixels/256) merge tokens + overhead
    est_img_tokens = min(args.max_pixels // 256, 1600)

    # Load data (filter long samples to avoid OOM)
    train_data = load_sft_json(args.data_path, max_length=args.max_length, est_img_tokens=est_img_tokens)
    val_data = load_sft_json(args.val_path, max_length=args.max_length, est_img_tokens=est_img_tokens) if args.val_path else []
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    collator = SFTCollator(processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.log_steps,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="tensorboard",
        seed=42,
        max_steps=args.max_steps if args.max_steps else -1,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Training done. Saving to {args.output_dir}")
    trainer.save_model()
    print("Done.")


if __name__ == "__main__":
    main()
