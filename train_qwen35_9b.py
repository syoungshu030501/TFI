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
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from dataset import VLMSFTDataset
from vlm_collator import VLMDataCollator, find_lora_target_modules


def patch_chunked_cross_entropy(chunk_size: int = 512) -> None:
    """Monkey-patch transformers.loss.loss_utils.fixed_cross_entropy 沿 token 维分块计算。

    Qwen3.5 词表 248k，单步 logits ~750MB(bf16)，反向再翻倍 + softmax 中间张量
    会触发 OOM。沿序列维 chunk_size 切分，按 num_items_in_batch 求和后再除一次，
    数学上等价于原 reduction="sum"/mean。
    """
    from transformers.loss import loss_utils

    orig_fce = loss_utils.fixed_cross_entropy

    def chunked_fce(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
        if source.dim() != 2 or source.shape[0] <= chunk_size:
            return orig_fce(source, target, num_items_in_batch, ignore_index, **kwargs)
        total_loss = source.new_zeros((), dtype=torch.float32)
        for i in range(0, source.shape[0], chunk_size):
            sl = slice(i, i + chunk_size)
            chunk_loss = torch.nn.functional.cross_entropy(
                source[sl], target[sl], ignore_index=ignore_index, reduction="sum",
            )
            total_loss = total_loss + chunk_loss.float()
        if num_items_in_batch is not None:
            return total_loss / num_items_in_batch
        valid = (target != ignore_index).sum().clamp(min=1)
        return total_loss / valid

    loss_utils.fixed_cross_entropy = chunked_fce
    print(f"[patch] cross_entropy chunked along seq, chunk_size={chunk_size}")


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

    def _get_train_sampler(self, *args, **kwargs):
        if self.sample_weights is None or self.args.world_size > 1:
            return super()._get_train_sampler(*args, **kwargs)
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
    p.add_argument("--use_4bit", action="store_true",
                   help="QLoRA: 用 NF4 4-bit 量化加载 base，显存约 9GB（与 LoRA 互斥于 full_ft）")
    p.add_argument("--device_map", type=str, default=None,
                   help="多卡模型并行：传 'auto' / 'balanced'，会按 CUDA_VISIBLE_DEVICES 切分。"
                        "与 --gpu 互斥；用此模式时不要走 accelerate/torchrun")
    p.add_argument("--chunked_ce_size", type=int, default=512,
                   help="将 cross_entropy 按 token 维度分块，避免 [seq, vocab=248k] OOM。"
                        "0 = 关闭")
    p.add_argument("--max_image_pixels", type=int, default=512 * 512,
                   help="processor.image_processor.max_pixels，限制视觉 token 数。"
                        "Qwen-VL patch 28×28：512×512≈256 token，768×768≈576 token。0=不改")
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
    if args.local_rank == -1 and "LOCAL_RANK" not in os.environ and args.device_map is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    if args.chunked_ce_size > 0:
        patch_chunked_cross_entropy(args.chunked_ce_size)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[cfg] model={args.model_name}  full_ft={args.full_ft}  "
          f"inject_evidence={args.inject_evidence}")

    # ---- processor & model ----
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 限制图像分辨率，避免 image tokens 失控
    # Qwen3-VL: patch_size=16, merge_size=2，单 vision token ~ 32×32 px
    # 512×512 → 256 tokens；384×384 → 144 tokens
    if args.max_image_pixels > 0:
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            edge2 = args.max_image_pixels
            if hasattr(ip, "size") and ip.size is not None:
                try:
                    ip.size.longest_edge = edge2
                except Exception:
                    ip.size["longest_edge"] = edge2
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = edge2
            ps = getattr(ip, "patch_size", 16)
            ms = getattr(ip, "merge_size", 2)
            tok_per_axis = (int(edge2 ** 0.5)) // (ps * ms)
            print(f"[image] longest_edge<= {edge2} px ({int(edge2**0.5)}×{int(edge2**0.5)})  "
                  f"≈ {tok_per_axis*tok_per_axis} vision tokens")

    model_load_kwargs = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.device_map is not None:
        model_load_kwargs["device_map"] = args.device_map
        n_gpus = torch.cuda.device_count()
        print(f"[mode] model parallel via device_map={args.device_map!r}  "
              f"visible_gpus={n_gpus}")
    if args.use_4bit:
        if args.full_ft:
            raise ValueError("--use_4bit 不能与 --full_ft 同时开")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["quantization_config"] = bnb_cfg
        print("[mode] QLoRA (4-bit NF4 base + bf16 LoRA)")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, **model_load_kwargs
    )

    if args.full_ft:
        lr = args.full_ft_lr
        print("[mode] full fine-tune")
    else:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        if args.use_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
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

    if args.device_map is not None:
        # Trainer 通过这两个 flag 判断已做模型并行，不再对模型 .to(device)
        model.is_parallelizable = True
        model.model_parallel = True

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
