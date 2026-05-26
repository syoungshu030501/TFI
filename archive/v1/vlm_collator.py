"""VLM 数据整理器 (从 legacy/train_teacher.py 抽取重写)。

将 dataset 给出的 (image_path, conversations) 转为 model.forward 需要的
input_ids / pixel_values / labels。

兼容 Qwen3.5-9B / Qwen3-VL 系列, 通过 processor.apply_chat_template 自动适配。
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info


class VLMDataCollator:
    """收集 batch 并交给 processor 编码。"""

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        # 部分 processor 没有 tokenizer 属性
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def _to_messages(self, conversations: List[Dict], img_path: str) -> List[Dict]:
        msgs = []
        attached = False
        for conv in conversations:
            role = conv["role"]
            if role == "user" and not attached:
                msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": conv["content"]},
                    ],
                })
                attached = True
            else:
                msgs.append({
                    "role": role,
                    "content": [{"type": "text", "text": conv["content"]}],
                })
        return msgs

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, image_inputs_list = [], []
        for sample in batch:
            messages = self._to_messages(sample["conversations"], sample["image_path"])
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        any_image = any(im is not None for im in image_inputs_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs_list if any_image else None,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # 超长样本：把多出来的 token 的 label 置 -100，模型仍 forward 但 loss 不计
        # （image token 对齐由 processor 强制要求，不能直接 truncate input_ids）
        if self.max_length and labels.size(1) > self.max_length:
            labels[:, self.max_length:] = -100
        inputs["labels"] = labels
        return inputs


def find_lora_target_modules(model, exclude_prefixes=("visual", "vision_tower",
                                                      "vision_encoder")):
    """自动发现可应用 LoRA 的线性层名 (排除视觉编码器)。

    兼容 Qwen3.5 (Gated DeltaNet + Gated Attention) 与 Qwen3-VL (标准 Attention)。
    """
    target = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(name.startswith(p) or f".{p}." in name for p in exclude_prefixes):
                continue
            last = name.split(".")[-1]
            if last not in ("lm_head", "embed_tokens"):
                target.add(last)
    return sorted(target)
