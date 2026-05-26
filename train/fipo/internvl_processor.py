"""
verl-compatible HF Processor adapter for InternVL3 (model_type=internvl_chat),
which has no canonical multimodal Processor class.

We subclass vLLM's bundled `InternVLProcessor` so that:
  1. `__call__` only tokenizes + computes pixel_values (does NOT pre-substitute
     <image> -> <img>...IMG_CONTEXT*N...</img>). That substitution is left to
     vLLM's own multimodal processor at engine inference time, avoiding the
     "Failed to apply prompt replacement" mismatch caused by independent
     dynamic-tile decisions on either side.
  2. `apply_chat_template` flattens list-content blocks (e.g. {"type":"image"})
     into a "<image>" string before invoking the tokenizer's Qwen2-style
     chat template, since that template can only string-concat scalars.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoTokenizer, BatchFeature


def build_internvl_processor(model_path: str, trust_remote_code: bool = True) -> Any:
    from vllm.transformers_utils.processors.internvl import (
        InternVLImageProcessor,
        InternVLProcessor as _VLLMInternVLProcessor,
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_fast=True
    )

    image_size = (
        getattr(config, "force_image_size", None)
        or config.vision_config.image_size
    )
    patch_size = config.vision_config.patch_size
    downsample_ratio = float(config.downsample_ratio)
    image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

    image_processor = InternVLImageProcessor(
        image_size=image_size,
        min_dynamic_patch=1,
        max_dynamic_patch=1,
        dynamic_image_size=False,
        use_thumbnail=False,
    )
    image_processor.patch_size = patch_size

    def _flatten_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for blk in content:
                if not isinstance(blk, dict):
                    parts.append(str(blk))
                    continue
                t = blk.get("type")
                if t == "image":
                    parts.append("<image>")
                elif t == "video":
                    parts.append("<video>")
                elif t == "text":
                    parts.append(blk.get("text", ""))
                else:
                    parts.append(blk.get("text", ""))
            return "".join(parts)
        return str(content)

    class _InternVLProcessorAdapter(_VLLMInternVLProcessor):
        """Adapter that:
          1. Substitutes <image> with <img>(IMG_CONTEXT*N)</img> when tokenizing
             so HF actor.forward gets correct IMG_CONTEXT positions.
          2. Renames pixel_values_flat -> pixel_values for HF compatibility, and
             also keeps a copy under image_num_patches for vLLM mm processing.
          3. Flattens list-content blocks before chat template rendering.
        """

        def __call__(self, text=None, images=None, videos=None, return_tensors=None, **kwargs):
            out = super().__call__(
                text=text,
                images=images,
                videos=videos,
                return_tensors=return_tensors,
                **kwargs,
            )
            d = dict(out)
            if "pixel_values_flat" in d:
                d["pixel_values"] = d.pop("pixel_values_flat")
            if "image_num_patches" in d:
                import torch
                inp = d.get("input_ids")
                if inp is not None:
                    npatch = d["image_num_patches"]
                    n_total = int(npatch.sum().item()) if hasattr(npatch, "sum") else sum(npatch)
                    d["image_flags"] = torch.ones(n_total, dtype=torch.long)
            return BatchFeature(data=d, tensor_type=return_tensors)

        def apply_chat_template(self, messages, **kwargs):
            flat_msgs = []
            for m in messages:
                mm = dict(m)
                mm["content"] = _flatten_content(mm.get("content", ""))
                flat_msgs.append(mm)
            return self.tokenizer.apply_chat_template(flat_msgs, **kwargs)

    processor = _InternVLProcessorAdapter(
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=None,
        image_seq_length=image_seq_length,
        ctx_video_token=None,
    )

    processor.chat_template = tokenizer.chat_template
    processor.image_token = "<IMG_CONTEXT>"
    processor.image_token_id = processor.ctx_image_token_id
    processor.video_token = None
    processor.video_token_id = None
    processor.config = config

    return processor


def is_internvl_chat(model_path: str, trust_remote_code: bool = True) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return getattr(cfg, "model_type", "") == "internvl_chat"
    except Exception:
        return False
