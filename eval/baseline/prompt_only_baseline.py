#!/usr/bin/env python
"""
M(-1) prompt-only baseline.

Run Qwen3.5-9B (no fine-tune) on val/200 with 3 prompt variants and produce
prediction CSVs in the same schema as v1's submit_val.csv (image_name, label,
location-RLE, explanation), so we can plug them straight into score_official.py
*and* into the judge_absolute_scoring.py module.

Usage:
  python tools/baseline/prompt_only_baseline.py --mode zs  --gpu 1
  python tools/baseline/prompt_only_baseline.py --mode fs  --gpu 1
  python tools/baseline/prompt_only_baseline.py --mode cot --gpu 1

Each variant writes to:
  tools/baseline/results/<mode>/predictions.csv
  tools/baseline/results/<mode>/raw/<image_name>.json   (per-sample raw output for caching/debug)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# --- imports from project root ---
THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(THIS.parent))

from prompts import (  # noqa: E402
    build_messages,
    build_fewshot_examples,
    SCHEMA_HINT,
)
from utils import mask_to_rle  # noqa: E402

VAL_DIR = PROJ / "data" / "raw" / "val"
TRAIN_DIR = PROJ / "data" / "raw" / "train"
DEFAULT_MODEL = PROJ / "models" / "Qwen3.5-9B"
DEFAULT_OUT_BASE = PROJ / "tools" / "baseline" / "results"


# ============================================================
# Data: enumerate all 200 val samples
# ============================================================
def collect_val_samples() -> List[Dict[str, Any]]:
    samples = []
    for label_dir, label in [("Black", 1), ("White", 0)]:
        img_dir = VAL_DIR / label_dir / "Image"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            samples.append({
                "image_name": p.name,
                "image_path": str(p),
                "gt_label": label,
            })
    return samples


# ============================================================
# Output parsing: extract JSON, even when wrapped in <think>/```json fences
# ============================================================
JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def fallback_label_from_text(text: str) -> int:
    """Best-effort label guess when model didn't output JSON at all."""
    forged_kw = ["伪造", "篡改", "tamper", "forg", "fake", "PS", "不真实", "可疑", "异常"]
    real_kw = ["真实", "未发现", "无明显", "正常", "未见", "符合"]
    score = sum(text.count(k) for k in forged_kw) - sum(text.count(k) for k in real_kw)
    return 1 if score > 0 else 0


def parse_model_output(raw: str, w: int, h: int) -> Dict[str, Any]:
    cleaned = strip_thinking(raw)
    m = JSON_OBJ_RE.search(cleaned)
    if not m:
        guessed = fallback_label_from_text(cleaned)
        excerpt = cleaned[:1200] if cleaned else "模型未输出可解析内容。"
        return {"label": guessed, "location": [], "explanation": excerpt}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            cand = m.group(0).replace("'", '"')
            cand = re.sub(r",\s*([}\]])", r"\1", cand)
            obj = json.loads(cand)
        except Exception:
            return {"label": 0, "location": [], "explanation": cleaned[:600] or "解析失败：JSON 不合法。"}

    label = int(obj.get("label", 0)) if obj.get("label") in (0, 1, "0", "1") else 0
    loc_in = obj.get("location") or []
    loc_out: List[Dict[str, Any]] = []
    if isinstance(loc_in, list):
        for item in loc_in:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            except Exception:
                continue
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            loc_out.append({"bbox": [x1, y1, x2, y2], "type": str(item.get("type", "篡改"))[:20]})

    if label == 0:
        loc_out = []
    if label == 1 and not loc_out:
        label = 0  # consistency repair: 没有 bbox 就视为真实

    expl = str(obj.get("explanation", "")).strip()
    if not expl:
        expl = "该图像经分析未发现明显异常。" if label == 0 else "该图像经分析存在伪造痕迹。"
    expl = expl[:1500]
    return {"label": label, "location": loc_out, "explanation": expl}


def location_to_rle(loc: List[Dict[str, Any]], w: int, h: int) -> Dict[str, Any]:
    mask = np.zeros((h, w), dtype=np.uint8)
    for item in loc:
        x1, y1, x2, y2 = item["bbox"]
        mask[y1:y2, x1:x2] = 1
    return mask_to_rle(mask)


# ============================================================
# Main inference loop
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["zs", "fs", "cot"])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--gpu", default="1",
                    help="CUDA_VISIBLE_DEVICES override (单卡传 '1'；27B 双卡传 '1,2')")
    ap.add_argument("--device_map", default="single",
                    choices=["single", "auto", "balanced"],
                    help="single: model.to('cuda') (≤9B) / auto: HF accelerate 自动分卡 (≥27B 必选)")
    ap.add_argument("--max_memory_per_gpu", default=None,
                    help="device_map=auto 时每卡上限，例如 '42GiB'；None=不限")
    ap.add_argument("--out_base", default=str(DEFAULT_OUT_BASE))
    ap.add_argument("--n_forged", type=int, default=4)
    ap.add_argument("--n_real", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1500)
    ap.add_argument("--longest_edge_pixels", type=int, default=384 * 384,
                    help="processor.image_processor.size.longest_edge to limit memory")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="sanity-check on first N samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    out_dir = Path(args.out_base) / args.mode
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== prompt-only baseline | mode={args.mode} | model={args.model} ===")

    samples = collect_val_samples()
    if args.limit:
        samples = samples[: args.limit]
    log(f"val samples: {len(samples)} (forged={sum(s['gt_label']==1 for s in samples)}, real={sum(s['gt_label']==0 for s in samples)})")

    fewshot = []
    if args.mode == "fs":
        fewshot = build_fewshot_examples(
            train_dir=str(TRAIN_DIR),
            n_forged=args.n_forged,
            n_real=args.n_real,
            seed=args.seed,
        )
        log(f"fewshot pool: {len(fewshot)} (forged={sum(e['label']==1 for e in fewshot)}, real={sum(e['label']==0 for e in fewshot)})")

    log("loading model + processor ...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ip = getattr(processor, "image_processor", None)
    if ip is not None and hasattr(ip, "size") and ip.size is not None:
        try:
            ip.size.longest_edge = int(args.longest_edge_pixels)
        except Exception:
            pass

    from_kwargs = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.device_map != "single":
        from_kwargs["device_map"] = args.device_map
        if args.max_memory_per_gpu:
            n_gpu = torch.cuda.device_count()
            from_kwargs["max_memory"] = {
                i: args.max_memory_per_gpu for i in range(n_gpu)
            }
            from_kwargs["max_memory"]["cpu"] = "120GiB"
    model = AutoModelForImageTextToText.from_pretrained(args.model, **from_kwargs)
    if args.device_map == "single":
        model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device
    n_gpu = torch.cuda.device_count()
    mem_per = [torch.cuda.memory_allocated(i) / 1e9 for i in range(n_gpu)]
    log(f"model loaded (device_map={args.device_map}, primary={device}); "
        f"per-GPU alloc: {[f'{m:.1f}GB' for m in mem_per]}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, samp in enumerate(tqdm(samples, ncols=80, desc=f"  {args.mode}")):
        name = samp["image_name"]
        cache_path = raw_dir / f"{name}.json"
        if cache_path.exists():
            try:
                rec = json.loads(cache_path.read_text(encoding="utf-8"))
                rows.append(rec["row"])
                continue
            except Exception:
                pass

        try:
            with Image.open(samp["image_path"]) as im:
                w, h = im.size
        except Exception as e:
            log(f"[{name}] PIL open failed: {e}; skipping.")
            continue

        messages = build_messages(args.mode, samp["image_path"], fewshot if args.mode == "fs" else None)

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(device)

            do_sample = args.temperature > 1e-6
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = 0.9

            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_kwargs)
            gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
            raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError as e:
            log(f"[{name}] OOM: {e}")
            torch.cuda.empty_cache()
            raw = ""
        except Exception as e:
            log(f"[{name}] generate failed: {e}\n{traceback.format_exc()[-500:]}")
            raw = ""

        parsed = parse_model_output(raw, w, h)
        rle = location_to_rle(parsed["location"], w, h)

        row = {
            "image_name": name,
            "label": parsed["label"],
            "location": json.dumps(rle, ensure_ascii=False, separators=(",", ":")),
            "explanation": parsed["explanation"],
        }
        rec = {
            "row": row, "raw": raw, "parsed": parsed,
            "gt_label": samp["gt_label"], "image_size": [w, h],
        }
        cache_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            log(f"[{i+1}/{len(samples)}] elapsed={elapsed:.0f}s  avg={elapsed/(i+1):.1f}s/sample")

    csv_path = out_dir / "predictions.csv"
    pd.DataFrame(rows, columns=["image_name", "label", "location", "explanation"]).to_csv(
        csv_path, index=False
    )
    log(f"=== DONE: {csv_path} (n={len(rows)}, total={time.time()-t0:.0f}s) ===")
    log_f.close()


if __name__ == "__main__":
    main()
