#!/usr/bin/env python
"""
M(-1)++ : Veritas-Cold-Start zero-shot on TFI val 200.

目的：测 InternVL3-8B (HydraFake-SFT 后) 在 **未见过 TFI 中文任务** 时的迁移上限。
用 transformers 直接 load (vllm 0.7.3 不一定支持 InternVL3)，单卡 BF16，~30 min 跑完 200。

输出：
  tools/baseline/results_veritas/zero_shot/raw/<image_name>.json
  tools/baseline/results_veritas/zero_shot/predictions.csv (image_name,label,location,explanation)
  其中 location 暂用 bbox-rect mask 简易 RLE（让 score_official.py 不崩；正式实验用 SAM 3.1 refine）

Usage:
  python tools/baseline/veritas_zero_shot.py --gpu 3
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
sys.path.insert(0, str(PROJ))

VAL_DIR = PROJ / "data/raw/val"
DEFAULT_MODEL = Path("/mnt/nfs/young/TFI/models/Veritas-Cold-Start")
DEFAULT_OUT = PROJ / "tools/baseline/results_veritas/zero_shot"

SYS_EN = (
    "You are an image authenticity expert. Your task is to determine the authenticity of the given image.\n\n"
    "Firstly, give an overall judgement to the authenticity of the image, enclosed in <fast> </fast> tags.\n"
    "Then, make a careful and structured thinking before reaching an answer. Based on your thinking, draw a "
    "comprehensive conclusion. Enclose the corresponding part in different tags, e.g., <planning> or <reasoning> "
    "or <reflection> or <conclusion>. For fake images, also annotate the suspected forged region with "
    "<bbox>x1,y1,x2,y2</bbox> inside <conclusion>.\n"
    "Finally, give the final answer with \"real\" or \"fake\", enclosed in <answer> </answer> tags."
)
USR = "<image>\nPlease determine the authenticity of this image."

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


def bbox_to_rle_mask(bbox, w, h):
    """简易 RLE：在 bbox 矩形内填充 1，其它为 0，转 COCO RLE 字符串。"""
    if bbox is None:
        return {"size": [h, w], "counts": ""}
    try:
        from pycocotools import mask as pmask
    except Exception:
        return {"size": [h, w], "counts": ""}
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    m = np.zeros((h, w), dtype=np.uint8, order="F")
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    rle = pmask.encode(m)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def load_internvl(model_path: Path, gpu_id: int):
    import torch
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
    return model, tokenizer, device


def build_input_pixel_values(image_path: Path, device, dtype=torch.bfloat16):
    """InternVL 标准预处理：动态分块到 448x448，最多 6 块。"""
    from torchvision import transforms

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    SIZE = 448

    transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    pv = transform(img).unsqueeze(0).to(device).to(dtype)  # [1, 3, 448, 448]
    return pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 张（dry run 用）")
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
    for s in tqdm(samples):
        cache = raw_dir / f"{s['image_name']}.json"
        if cache.exists():
            obj = json.loads(cache.read_text(encoding="utf-8"))
            text = obj.get("text", "")
        else:
            try:
                pixel_values = build_input_pixel_values(s["image_path"], device)
                question = USR + "\n\n" + SYS_EN
                generation_config = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                with torch.inference_mode():
                    text = model.chat(
                        tok,
                        pixel_values,
                        question,
                        generation_config,
                    )
                cache.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                text = f"[ERROR] {e}"
                cache.write_text(json.dumps({"text": text, "error": str(e)}, ensure_ascii=False), encoding="utf-8")

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

    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "label", "location", "explanation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
