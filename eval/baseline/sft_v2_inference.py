#!/usr/bin/env python
"""
v2 SFT-baseline inference on val/200.

Loads the merged InternVL3-8B + v2 SFT-baseline LoRA ckpt-54 and runs the
**same Chinese 6-tag CoT template** that the model was trained on. Output
schema matches v1 submit_val.csv (image_name, label, location-RLE, explanation),
so it plugs directly into eval/score_official.py and eval/baseline/judge_absolute_scoring.py.

Usage:
  python eval/baseline/sft_v2_inference.py --gpu 1
  python eval/baseline/sft_v2_inference.py --gpu 1 --limit 5   # smoke test
"""
from __future__ import annotations
import argparse
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
DEFAULT_MODEL = Path("/mnt/nfs/young/TFI/models/sft_v2_baseline_1009")
DEFAULT_OUT = PROJ / "eval/baseline/results/sft_v2"

# Chinese 6-tag CoT template — verbatim from training data (sft_val.json[0].messages[0]).
SYS_CN = (
    "你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。\n\n"
    "首先用 <fast> </fast> 标签给出第一直觉判断；\n"
    "然后用 <reasoning> </reasoning> 标签给出详细取证推理（高难度样本可在其中包含 <planning> 规划与 <reflection> 自校验）；\n"
    "接着用 <conclusion> </conclusion> 标签给出综合结论，对疑似篡改图必须用 <bbox>x1,y1,x2,y2</bbox> 或 <region>区域文字描述</region> 标注疑似篡改区域；\n"
    "最后用 <answer>real|fake</answer> 给出最终判断（仅二选一）。"
)
USR = "<image>请判断该图像的真实性，并按规定标签格式输出分析。"

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
    """Convert bbox to RLE. If normalized=True, bbox is in [0,1000]×[0,1000] (training-time convention)."""
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


def load_internvl(model_path: Path, gpu_id: int):
    from transformers import AutoModel, AutoTokenizer
    device = f"cuda:{gpu_id}"
    print(f"[load] model={model_path} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to(device)
    # Inject training-time Chinese system prompt; InternVL3.chat() reads model.system_message.
    if hasattr(model, "system_message"):
        model.system_message = SYS_CN
    return model, tokenizer, device


def build_input_pixel_values(image_path: Path, device, dtype=torch.bfloat16, max_num: int = 12):
    """InternVL3 dynamic-tiling preprocessing — must match training (ms-swift InternvlTemplate, max_num=12 + thumbnail)."""
    from swift.llm.template.vision_utils import transform_image

    img = Image.open(image_path).convert("RGB")
    pv = transform_image(img, input_size=448, max_num=max_num)
    pv = pv.to(device).to(dtype)
    return pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    samples = collect_val()
    if args.limit:
        samples = samples[: args.limit]
    print(f"[main] {len(samples)} val samples")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    model, tok, device = load_internvl(Path(args.model), args.gpu)

    rows = []
    t0 = time.time()
    for i, s in enumerate(tqdm(samples)):
        cache = raw_dir / f"{s['image_name']}.json"
        if cache.exists():
            obj = json.loads(cache.read_text(encoding="utf-8"))
            text = obj.get("text", "")
        else:
            try:
                pixel_values = build_input_pixel_values(s["image_path"], device)
                generation_config = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                with torch.inference_mode():
                    text = model.chat(
                        tok,
                        pixel_values,
                        USR,
                        generation_config,
                    )
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

    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "label", "location", "explanation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {csv_path} (n={len(rows)}, total={time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
