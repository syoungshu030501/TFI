"""
将 LoRA 权重合并到基座模型并保存完整模型。
用于 vLLM 推理 (vLLM 对多模态 LoRA 动态加载支持不稳定, 合并后更可靠)。

用法:
  python merge_lora.py
"""

from pathlib import Path

import torch
import rocm_compat
rocm_compat.patch_grouped_mm()

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = str(PROJECT_ROOT / "models" / "Qwen3.5-397B-A17B")
LORA_PATH = str(PROJECT_ROOT / "checkpoints" / "teacher")
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "Qwen3.5-397B-A17B-teacher")


def main():
    print(f"Loading base model: {BASE_MODEL}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    print("LoRA merged")

    print(f"Saving merged model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="8GB")
    processor.save_pretrained(OUTPUT_DIR)
    print("Done")


if __name__ == "__main__":
    main()
