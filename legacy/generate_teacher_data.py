"""
教师模型生成增强数据
用合并后的教师模型对训练图片生成多版本 Caption (含 thinking 推理链)

用法:
  python generate_teacher_data.py --data_dir train --output_dir augmented_data/train
"""

import argparse
import json
import os
from pathlib import Path

import torch
import rocm_compat
rocm_compat.patch_grouped_mm()

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


SYSTEM_PROMPT = (
    "你是专业的图像伪造检测分析专家。请仔细检查图片，判断是否存在数字伪造或篡改痕迹。"
    "如存在，请精确指出篡改区域坐标、篡改内容、视觉异常特征和逻辑矛盾。"
    "如不存在，请从视觉一致性、信息准确性等方面论证真实性。"
)

USER_PROMPT = "请分析这张图片是否存在伪造痕迹，给出详细的中文鉴定分析。"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = str(PROJECT_ROOT / "models" / "Qwen3.5-397B-A17B")
LORA_PATH = str(PROJECT_ROOT / "checkpoints" / "teacher")


def collect_images(data_dir):
    """收集所有图片路径和类别"""
    data_dir = Path(data_dir)
    images = []
    for category in ["Black", "White"]:
        img_dir = data_dir / category / "Image"
        if not img_dir.exists():
            continue
        for fname in sorted(os.listdir(img_dir)):
            images.append({
                "image_path": str(img_dir / fname),
                "category": category,
                "stem": os.path.splitext(fname)[0],
            })
    return images


def generate_caption(model, processor, image_path, temperature=0.7, max_new_tokens=2048):
    """用模型生成单张图片的 caption"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=BASE_MODEL)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_versions", type=int, default=3)
    parser.add_argument("--temperatures", type=str, default="0.7,0.9,1.1")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--slice", type=str, default=None,
                        help="数据分片, 格式 'K/N' 表示第K片共N片, 如 '0/2' 取前半")
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]

    from peft import PeftModel

    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if args.lora_path and os.path.exists(args.lora_path):
        print(f"Loading LoRA from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA merged")

    model.eval()
    print("Model ready")

    all_images = collect_images(args.data_dir)
    if args.slice:
        k, n = map(int, args.slice.split("/"))
        chunk_size = len(all_images) // n
        start = k * chunk_size
        end = len(all_images) if k == n - 1 else (k + 1) * chunk_size
        images = all_images[start:end]
        print(f"Slice {k}/{n}: images [{start}:{end}] = {len(images)} / {len(all_images)} total")
    else:
        images = all_images
        print(f"Found {len(images)} images in {args.data_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "augmented_captions.jsonl"

    # 断点续传: 跳过已生成的
    done_keys = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done_keys.add(f"{item['stem']}_v{item['version']}")
        print(f"Resuming: {len(done_keys)} captions already done")

    with open(output_file, "a", encoding="utf-8") as f_out:
        for img_info in tqdm(images, desc="Generating"):
            for ver_idx, temp in enumerate(temperatures[:args.num_versions]):
                key = f"{img_info['stem']}_v{ver_idx}"
                if key in done_keys:
                    continue
                try:
                    caption = generate_caption(
                        model, processor,
                        img_info["image_path"],
                        temperature=temp,
                        max_new_tokens=args.max_new_tokens,
                    )
                    result = {
                        "image_path": img_info["image_path"],
                        "category": img_info["category"],
                        "stem": img_info["stem"],
                        "version": ver_idx,
                        "temperature": temp,
                        "caption": caption,
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    print(f"  Error on {img_info['stem']} v{ver_idx}: {e}")

    total = sum(1 for _ in open(output_file, "r", encoding="utf-8"))
    print(f"\nGenerated {total} captions, saved to {output_file}")


if __name__ == "__main__":
    main()
