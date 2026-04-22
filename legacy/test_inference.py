"""
Qwen3.5-397B-A17B 推理测试
使用 device_map="auto" 将模型分布到 8 张 MI325X 上

用法:
  python test_inference.py
  python test_inference.py --text_only       # 仅测试纯文本
  python test_inference.py --no_thinking     # 关闭思维链
"""

import argparse
import os
import time
from pathlib import Path

import torch

# ── ROCm 兼容性修复 ──
# torch._grouped_mm 使用 CK (Composable Kernel) grouped GEMM, 在 ROCm 上
# 需要预分配 workspace buffer, 当前环境未正确分配导致运行时崩溃。
# 替换为逐专家顺序计算: 功能等价, 略慢但完全兼容。
if hasattr(torch, "_grouped_mm"):
    _orig = torch._grouped_mm

    def _sequential_grouped_mm(input, weight, offs=None, bias=None, **kwargs):
        if offs is None:
            out = input @ weight
            if bias is not None:
                out = out + bias
            return out
        output = torch.empty(
            input.shape[0], weight.shape[-1],
            device=input.device, dtype=input.dtype,
        )
        start = 0
        for i in range(len(offs)):
            end = int(offs[i])
            if end > start:
                output[start:end] = input[start:end] @ weight[i]
                if bias is not None:
                    output[start:end] += bias[i]
            start = end
        return output

    torch._grouped_mm = _sequential_grouped_mm

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "Qwen3.5-397B-A17B")
LORA_PATH = str(PROJECT_ROOT / "checkpoints" / "teacher")
TEST_IMAGE = str(PROJECT_ROOT / "test" / "Image" / "001858037f7846a79c619fda3d915e75.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 权重路径, 不指定则使用 base 模型")
    parser.add_argument("--image", type=str, default=TEST_IMAGE)
    parser.add_argument("--text_only", action="store_true", help="仅测试纯文本推理")
    parser.add_argument("--no_thinking", action="store_true", help="关闭思维链")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    # ── 加载模型 ──
    print(f"Loading model: {args.model_path}")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if args.lora_path and os.path.exists(args.lora_path):
        from peft import PeftModel
        print(f"Loading LoRA from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA merged")

    model.eval()

    load_time = time.time() - t0
    mode = "base + LoRA" if args.lora_path else "base only"
    print(f"Model loaded in {load_time:.1f}s ({mode})")
    print(f"Device map: {set(model.hf_device_map.values())}")

    # ── 测试 1: 纯文本推理 ──
    print("\n" + "=" * 60)
    print("[Test 1] 纯文本推理")
    print("=" * 60)

    text_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "你好，请用一句话介绍你自己。"},
        ]},
    ]

    run_inference(model, processor, text_messages, args.max_new_tokens, args.no_thinking)

    if args.text_only:
        print("\n纯文本测试完成。")
        return

    # ── 测试 2: 图像理解 ──
    print("\n" + "=" * 60)
    print(f"[Test 2] 图像理解 ({args.image})")
    print("=" * 60)

    vision_messages = [
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{args.image}"},
            {"type": "text", "text": "请描述这张图片的内容，并判断它是否存在伪造或篡改痕迹。"},
        ]},
    ]

    run_inference(model, processor, vision_messages, args.max_new_tokens, args.no_thinking)

    # ── 测试 3: 伪造检测专业 prompt ──
    print("\n" + "=" * 60)
    print("[Test 3] 伪造检测专业分析")
    print("=" * 60)

    forensic_messages = [
        {"role": "system", "content": [
            {"type": "text", "text": (
                "你是专业的图像伪造检测分析专家。请仔细检查图片，判断是否存在数字伪造或篡改痕迹。"
                "如存在，请精确指出篡改区域坐标、篡改内容、视觉异常特征和逻辑矛盾。"
                "如不存在，请从视觉一致性、信息准确性等方面论证真实性。"
            )},
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{args.image}"},
            {"type": "text", "text": "请分析这张图片是否存在伪造痕迹，给出详细的中文鉴定分析。"},
        ]},
    ]

    run_inference(model, processor, forensic_messages, args.max_new_tokens, args.no_thinking)

    print("\n所有测试完成。")


def run_inference(model, processor, messages, max_new_tokens, no_thinking):
    """执行单次推理并打印结果"""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=(not no_thinking),
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
        )
    gen_time = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    output_text = processor.tokenizer.decode(generated, skip_special_tokens=True)

    num_tokens = len(generated)
    tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0

    # 分离思维链和最终回答
    thinking = ""
    answer = output_text
    if "<think>" in output_text and "</think>" in output_text:
        think_start = output_text.index("<think>") + len("<think>")
        think_end = output_text.index("</think>")
        thinking = output_text[think_start:think_end].strip()
        answer = output_text[think_end + len("</think>"):].strip()

    if thinking:
        print(f"\n[Thinking] ({len(thinking)} chars)")
        print(thinking[:500] + ("..." if len(thinking) > 500 else ""))

    print(f"\n[Answer]")
    print(answer)
    print(f"\n[Stats] {num_tokens} tokens in {gen_time:.1f}s ({tokens_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()
