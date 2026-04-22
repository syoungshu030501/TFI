"""Qwen3.5-9B 证据驱动微调 (LoRA / 全量可切换)。

特点:
  - 默认 LoRA r=64, 适配 1-2 卡 RTX 5090 / L20 (46GB)
  - --full_ft 切全量 (需 4+ 卡 + DeepSpeed ZeRO-2/3)
  - 训练数据: 原始 Caption + GT mask 抽取的结构化证据 prompt
  - 通过 dataset.VLMSFTDataset(inject_evidence=True) 保证训练-推理一致

用法:
  # LoRA, 单卡（数据来源默认 data/raw/train_resume + data/vlm/caption_api_v3 + data/processed/real_ext）
  python train_qwen35_9b.py --gpu 0 --batch_size 1 --grad_accum 16

  # LoRA, 多卡 DDP
  accelerate launch --num_processes 2 train_qwen35_9b.py --batch_size 1 --grad_accum 8

  # 全量微调, 4 卡
  accelerate launch --num_processes 4 train_qwen35_9b.py --full_ft \\
      --batch_size 1 --grad_accum 8 --lr 2e-5
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from dataset import VLMSFTDataset
from vlm_collator import VLMDataCollator, find_lora_target_modules


def build_sample_weights(labels):
    counts = Counter(labels)
    if len(counts) <= 1:
        return None
    max_count = max(counts.values())
    return torch.as_tensor(
        [float(max_count) / float(counts[label]) for label in labels],
        dtype=torch.double,
    )


class WeightedSamplerTrainer(Trainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self):
        if self.sample_weights is None or self.args.world_size > 1:
            return super()._get_train_sampler()
        return WeightedRandomSampler(
            self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="models/Qwen3.5-9B")
    p.add_argument("--data_dir", default="data/raw/train_resume")
    p.add_argument("--augmented_dir", default="data/vlm/caption_api_v3",
                   help="API 蒸馏的 evidence-caption 目录（v3 = qwen-vl-max）。"
                        "若想用旧本地 9B 产物，改成 data/processed/caption_local_v2")
    p.add_argument("--real_ext_dir", default="data/processed/real_ext",
                   help="真实图扩充目录")
    p.add_argument("--output_dir", default="checkpoints/qwen35_9b")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--full_ft_lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--full_ft", action="store_true", help="全量微调 (默认 LoRA)")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--inject_evidence", action="store_true", default=True)
    p.add_argument("--no_inject_evidence", action="store_false",
                   dest="inject_evidence")
    p.add_argument("--use_caption_clean", action="store_true", default=True)
    p.add_argument("--no_use_caption_clean", action="store_false",
                   dest="use_caption_clean")
    p.add_argument("--include_real_ext", action="store_true", default=True)
    p.add_argument("--no_include_real_ext", action="store_false",
                   dest="include_real_ext")
    p.add_argument("--use_weighted_sampler", action="store_true")

    p.add_argument("--save_steps", type=int, default=0,
                   help="0 = 按 epoch 保存")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    if args.local_rank == -1 and "LOCAL_RANK" not in os.environ:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[cfg] model={args.model_name}  full_ft={args.full_ft}  "
          f"inject_evidence={args.inject_evidence}")

    # ---- processor & model ----
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if args.full_ft:
        lr = args.full_ft_lr
        print("[mode] full fine-tune")
    else:
        from peft import LoraConfig, get_peft_model, TaskType
        target_modules = find_lora_target_modules(model)
        print(f"[lora] target_modules ({len(target_modules)}): {target_modules}")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        lr = args.lr

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ---- dataset ----
    train_dataset = VLMSFTDataset(
        args.data_dir,
        augmented_captions_dir=args.augmented_dir,
        inject_evidence=args.inject_evidence,
        use_caption_clean=args.use_caption_clean,
        real_ext_dir=args.real_ext_dir if args.include_real_ext else None,
    )
    print(f"[data] training samples: {len(train_dataset)}")
    label_counts = Counter(train_dataset.get_labels())
    print(f"[data] label counts: {dict(label_counts)}")

    collator = VLMDataCollator(processor, max_length=args.max_length)

    # ---- training args ----
    save_kwargs = {}
    if args.save_steps > 0:
        save_kwargs.update(save_strategy="steps", save_steps=args.save_steps)
    else:
        save_kwargs.update(save_strategy="epoch")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        **save_kwargs,
    )

    sample_weights = None
    if args.use_weighted_sampler:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            print("[sampler] distributed mode detected, fallback to default sampler")
        else:
            sample_weights = build_sample_weights(train_dataset.get_labels())
            print("[sampler] weighted sampler enabled" if sample_weights is not None
                  else "[sampler] skipped (single-class labels)")

    trainer = WeightedSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        sample_weights=sample_weights,
    )

    print("=" * 60)
    print("Training start")
    print("=" * 60)
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
n()
