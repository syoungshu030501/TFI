#!/usr/bin/env python
"""
v2 路线 A · Qwen3.5-9B SFT inference on val/200.

Loads the merged Qwen3.5-9B + qwen35_v2 LoRA ckpt and runs the same Chinese
6-tag CoT template the model was trained on. Output schema matches v1's
submit_val.csv (image_name, label, location-RLE, explanation), so it plugs
straight into eval/score_official.py and eval/baseline/judge_absolute_scoring.py.

Usage:
  conda activate VLM
  python eval/baseline/sft_v2_inference_qwen35.py --gpu 1
  python eval/baseline/sft_v2_inference_qwen35.py --gpu 1 --limit 5    # smoke
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
sys.path.insert(0, str(PROJ))

VAL_DIR = PROJ / "data/raw/val"
DEFAULT_MODEL = Path("/mnt/nfs/young/TFI/models/qwen35_v2_1441")
DEFAULT_OUT = PROJ / "eval/baseline/results/sft_v2_qwen35"

# 与 build_v2_sft.py / sft_v2_inference.py 完全一致的 6-tag CoT 提示
SYS_CN = (
    "你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。\n\n"
    "首先用 <fast> </fast> 标签给出第一直觉判断；\n"
    "然后用 <reasoning> </reasoning> 标签给出详细取证推理（高难度样本可在其中包含 <planning> 规划与 <reflection> 自校验）；\n"
    "接着用 <conclusion> </conclusion> 标签给出综合结论，对疑似篡改图必须用 <bbox>x1,y1,x2,y2</bbox> 或 <region>区域文字描述</region> 标注疑似篡改区域，"
    "其中 bbox 坐标已归一化到 [0,1000]×[0,1000]（左上原点，x1<x2，y1<y2）；\n"
    "最后用 <answer>real|fake</answer> 给出最终判断（仅二选一）。"
)
USR = "请判断该图像的真实性，并按规定标签格式输出分析。"

ANSWER_RE = re.compile(r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE)
BBOX_RE = re.compile(r"<bbox>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</bbox>")


def collect_val():
    out = []
    for sub, lbl in [("Black", 1), ("White", 0)]:
        d = VAL_DIR / sub / "Image"
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                out.append({"image_name": p.name, "image_path": str(p), "gt_label": lbl})
    return out


def bbox_to_rle_mask(bbox, w, h, normalized: bool = True):
    if bbox is None:
        return {"size": [h, w], "counts": ""}
    try:
        from pycocotools import mask as pmask
    except Exception:
        return {"size": [h, w], "counts": ""}
    x1, y1, x2, y2 = bbox
    if normalized:
        x1 = int(round(x1 / 1000.0 * w))
        y1 = int(round(y1 / 1000.0 * h))
        x2 = int(round(x2 / 1000.0 * w))
        y2 = int(round(y2 / 1000.0 * h))
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    m = np.zeros((h, w), dtype=np.uint8, order="F")
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    rle = pmask.encode(m)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def load_qwen(model_path: Path, gpu_id: int):
    """加载 Qwen3.5-9B (Qwen3_5ForConditionalGeneration) + Qwen3VLProcessor。"""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    device = f"cuda:{gpu_id}"
    print(f"[load] model={model_path} device={device}")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to(device)
    return model, processor, device


def generate_one(model, processor, device, image_path: str, max_new_tokens: int) -> str:
    """单张图片一次 chat。Qwen3.5-VL 走 Qwen3VLProcessor 的 chat-template 流程。"""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYS_CN}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USR}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--start", type=int, default=0, help="shard start index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="shard end index (exclusive)")
    args = ap.parse_args()

    samples = collect_val()
    if args.limit:
        samples = samples[: args.limit]
    if args.end is not None:
        samples = samples[args.start: args.end]
    elif args.start:
        samples = samples[args.start:]
    print(f"[main] {len(samples)} val samples (shard {args.start}:{args.end})")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    model, processor, device = load_qwen(Path(args.model), args.gpu)

    rows = []
    t0 = time.time()
    for i, s in enumerate(tqdm(samples)):
        cache = raw_dir / f"{s['image_name']}.json"
        if cache.exists():
            text = json.loads(cache.read_text(encoding="utf-8")).get("text", "")
        else:
            try:
                text = generate_one(model, processor, device, s["image_path"], args.max_new_tokens)
                cache.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                text = f"[ERROR] {e}"
                cache.write_text(json.dumps({"text": text, "error": str(e)}, ensure_ascii=False), encoding="utf-8")
                print(f"[err] {s['image_name']}: {e}", flush=True)

        m = ANSWER_RE.search(text or "")
        pred_label = 1 if (m and m.group(1).lower() == "fake") else 0
        bm = BBOX_RE.search(text or "")
        bbox = tuple(int(x) for x in bm.groups()) if bm else None

        with Image.open(s["image_path"]) as im:
            W, H = im.size
        loc = bbox_to_rle_mask(bbox, W, H) if pred_label == 1 else {"size": [H, W], "counts": ""}

        explanation = (text or "").strip().replace("\n", " ")[:2000]
        rows.append({
            "image_name": s["image_name"],
            "label": pred_label,
            "location": json.dumps(loc, ensure_ascii=False),
            "explanation": explanation,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] elapsed={elapsed:.0f}s avg={elapsed/(i+1):.1f}s/sample", flush=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "label", "location", "explanation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {csv_path} (n={len(rows)}, total={time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
