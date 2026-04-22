"""
教师模型训练: Qwen3.5-397B-A17B LoRA r=128 微调
使用 DeepSpeed ZeRO-3 + PEFT LoRA, 8 卡 MI325X

用法:
  # 8B 模型单卡调试
  python train_teacher.py \
      --model_name models/Qwen3-VL-8B-Thinking \
      --epochs 1 --lr 1e-4 --lora_r 16

  # 397B 模型 8 卡训练
  deepspeed --num_gpus 8 train_teacher.py \
      --model_name models/Qwen3.5-397B-A17B \
      --deepspeed ds_config_z3.json \
      --epochs 3 --lr 1e-4 --lora_r 128
"""

import argparse
import json
import os
from pathlib import Path

import torch
import rocm_compat
rocm_compat.patch_grouped_mm()

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

from dataset import VLMSFTDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = str(PROJECT_ROOT / "models" / "Qwen3.5-397B-A17B")


# ============================================================
# 数据预处理
# ============================================================

class VLMDataCollator:
    """
    VLM 数据整理器:
    将 (image_path, conversation) 转为模型可接受的 input_ids + pixel_values。
    """

    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        texts = []
        image_inputs_list = []

        for sample in batch:
            img_path = sample["image_path"]
            conversations = sample["conversations"]

            # 构建 Qwen3-VL 格式的消息
            messages = []
            for conv in conversations:
                if conv["role"] == "user":
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{img_path}"},
                            {"type": "text", "text": conv["content"]},
                        ],
                    })
                elif conv["role"] == "system":
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": conv["content"]}],
                    })
                elif conv["role"] == "assistant":
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": conv["content"]}],
                    })

            # 应用 chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        # 批量编码 (不截断, 避免图像 token 被切断导致 mismatch)
        inputs = self.processor(
            text=texts,
            images=image_inputs_list if any(img is not None for img in image_inputs_list) else None,
            padding=True,
            return_tensors="pt",
        )

        # 设置 labels (与 input_ids 相同, padding 部分设为 -100)
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs


# ============================================================
# 主训练函数
# ============================================================

def find_target_modules(model, exclude_prefixes=("visual", "vision_tower")):
    """
    自动发现模型中所有可应用 LoRA 的线性层名称。
    兼容 Qwen3-VL (标准 Attention) 和 Qwen3.5 (Gated DeltaNet + Gated Attention)。
    """
    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(name.startswith(p) for p in exclude_prefixes):
                continue
            last_name = name.split(".")[-1]
            if last_name not in ("lm_head", "embed_tokens"):
                target_names.add(last_name)
    return sorted(target_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "train"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "checkpoints" / "teacher"))
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    is_large_model = any(tag in args.model_name for tag in ("397B", "397b"))

    # DeepSpeed ZeRO-3 预初始化
    # 对大模型, 必须在 from_pretrained 之前激活 zero.Init,
    # 使参数在加载时直接分片到各 GPU, 避免 CPU 内存爆炸 (OOM)。
    dschf = None
    if args.deepspeed:
        from transformers.integrations import HfDeepSpeedConfig
        dschf = HfDeepSpeedConfig(args.deepspeed)
        print(f"[Fix] DeepSpeed ZeRO-3 pre-init activated — parameters will be "
              f"sharded across GPUs during loading (no full model on CPU)")

    # 加载处理器
    print(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 加载模型
    print(f"Loading model: {args.model_name} (is_large_model={is_large_model})")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("Model loaded successfully")

    # 冻结视觉编码器 (LoRA 只作用于语言模型)
    # 兼容不同架构的属性名: Qwen3-VL 用 "visual", 其他可能用 "vision_tower"
    visual_encoder = getattr(model, "visual", None) or getattr(model, "vision_tower", None)
    if visual_encoder is not None:
        visual_encoder.to(dtype=torch.bfloat16)
        for param in visual_encoder.parameters():
            param.requires_grad = False
        print(f"Visual encoder frozen, dtype={next(visual_encoder.parameters()).dtype}")

    # LoRA 配置
    # 使用显式目标模块列表 (不用自动检测, 因为 ZeRO-3 把权重压成 1D
    # 导致 PEFT 无法区分 Linear/Conv, 会误将 Conv3d 当作目标而崩溃)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # 应用 LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 开启梯度检查点
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 数据集
    train_dataset = VLMSFTDataset(args.data_dir)
    print(f"Training samples: {len(train_dataset)}")

    # 数据整理器
    collator = VLMDataCollator(processor, max_length=args.max_length)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        remove_unused_columns=False,
        report_to="none",
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # 训练
    print("Starting training...")
    trainer.train()

    # 保存 LoRA 权重 (必须用 trainer.save_model, 不能用 model.save_pretrained,
    # 因为 ZeRO-3 下 model.save_pretrained 保存的是分片空权重 shape=[0])
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
