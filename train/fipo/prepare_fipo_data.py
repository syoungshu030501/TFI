"""
Convert TFI SFT JSON (data/build/build_v2_sft.py output) -> verl-format FIPO
parquet.

verl multimodal RL schema (matches examples/data_preprocess/geo3k.py):
    data_source:    str
    prompt:         list[{"role": "user"|"system", "content": str}]
    images:         list[{"bytes": <jpeg bytes>}]
    reward_model:   {"style": "rule", "ground_truth": str (json)}
    extra_info:     {"split": str, "index": int, "source": str, ...}

The ground_truth blob is consumed by `train.fipo.verl_patches.reward_manager.
TFIAuditRewardManager` and has shape:
    {"label": 0|1, "bboxes": [[x1,y1,x2,y2], ...], "phrases": [...]}

Bbox extraction order:
  1. Use mask_path → bbox if available (v1 Black/White data with masks).
  2. Else parse <bbox> tags from the SFT assistant response.
  3. Else empty list (real images expectedly have none).

Usage:
    cd /home/young/TFI
    python -m train.fipo.prepare_fipo_data \
        --in_train /mnt/nfs/young/TFI/data/v2/sft.json \
        --in_val   /mnt/nfs/young/TFI/data/v2/sft_val.json \
        --out_dir  data/fipo \
        --max_train 2000 --max_val 200
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from train.fipo.schema import SYSTEM_PROMPT, USER_PROMPT

BBox = Tuple[float, float, float, float]
_BBOX_TAG_RE = re.compile(r"<bbox>\s*([\d.,\s\-+]+)\s*</bbox>")
_BBOX_NUMS_RE = re.compile(
    r"\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,"
    r"\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*"
)


def _mask_to_bbox(mask_path: Path) -> Optional[BBox]:
    """Read mask and return bbox normalized to [0,1000]² to match the model's
    output convention (set in build_v2_sft.py SYS_PROMPT_ZH). Without this
    normalization, mask-derived GT bboxes would live in raw pixel space while
    rollouts emit [0,1000] coords → R_iou_gt would be ≈0 across the board and
    grounding reward signal would vanish."""
    if not mask_path.exists():
        return None
    try:
        m = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            return None
        h, w = m.shape
        x1 = float(xs.min()) / w * 1000.0
        y1 = float(ys.min()) / h * 1000.0
        x2 = float(xs.max()) / w * 1000.0
        y2 = float(ys.max()) / h * 1000.0
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _bboxes_from_text(text: str) -> List[BBox]:
    out: List[BBox] = []
    for m in _BBOX_TAG_RE.finditer(text):
        nm = _BBOX_NUMS_RE.fullmatch(m.group(1))
        if nm:
            out.append(tuple(float(x) for x in nm.groups()))
    return out


def _phrases_from_assistant(text: str) -> List[str]:
    """Extract <region> contents and quoted phrases as GT phrase pool."""
    phrases = re.findall(r"<region>(.*?)</region>", text, flags=re.DOTALL)
    # also grab anything in “双引号” / "double" quotes that often hold tampered text
    phrases += re.findall(r"[“\"]([^“”\"]{1,80})[”\"]", text)
    return [p.strip() for p in phrases if p.strip()]


def _image_to_jpeg_bytes(path: Path, max_side: int = 1024) -> dict:
    """Load image, downscale long side to max_side to keep parquet small."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return {"bytes": buf.getvalue()}


def _row_to_verl(sample: dict, split: str, idx: int) -> Optional[dict]:
    images = sample.get("images") or []
    if not images:
        return None
    img_path = Path(images[0])
    if not img_path.exists():
        return None

    label = int(sample.get("label", 0))
    assistant = ""
    for m in sample.get("messages", []):
        if m.get("role") == "assistant":
            assistant = str(m.get("content", ""))
            break

    # Bboxes: prefer mask, fallback to parsed bbox tags
    bboxes: List[BBox] = []
    mp = sample.get("mask_path")
    if mp:
        b = _mask_to_bbox(Path(mp))
        if b is not None:
            bboxes.append(b)
    if not bboxes:
        bboxes = _bboxes_from_text(assistant)

    phrases = _phrases_from_assistant(assistant)

    gt_blob = json.dumps(
        {"label": label, "bboxes": [list(b) for b in bboxes], "phrases": phrases},
        ensure_ascii=False,
    )

    image_obj = _image_to_jpeg_bytes(img_path)

    return {
        "data_source": "tfi_forgery",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "images": [image_obj],
        "ability": "forgery_detection",
        "reward_model": {"style": "rule", "ground_truth": gt_blob},
        "extra_info": {
            "split": split,
            "index": idx,
            "image_file": str(img_path),
            "label": label,
            "source": str(sample.get("source", "")),
            "type": str(sample.get("type", "")),
        },
    }


def convert(in_json: Path, out_parquet: Path, split: str, max_rows: Optional[int]) -> int:
    with open(in_json) as f:
        data = json.load(f)
    if max_rows is not None and len(data) > max_rows:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(data), size=max_rows, replace=False)
        data = [data[i] for i in idxs]

    rows = []
    n_skipped = 0
    for i, s in enumerate(data):
        r = _row_to_verl(s, split, i)
        if r is None:
            n_skipped += 1
            continue
        rows.append(r)

    out_df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"[prepare_fipo_data] {in_json} -> {out_parquet}  rows={len(out_df)} skipped={n_skipped}")
    return len(out_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train", type=Path, required=True,
                    help="Path to SFT train JSON (e.g. /mnt/nfs/young/TFI/data/v2/sft.json)")
    ap.add_argument("--in_val", type=Path, required=True,
                    help="Path to SFT val JSON (e.g. /mnt/nfs/young/TFI/data/v2/sft_val.json)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/fipo"))
    ap.add_argument("--max_train", type=int, default=2000,
                    help="cap train rows (FIPO needs fewer prompts than SFT). 0=no cap")
    ap.add_argument("--max_val", type=int, default=200)
    args = ap.parse_args()

    cap_t = args.max_train if args.max_train > 0 else None
    cap_v = args.max_val if args.max_val > 0 else None

    convert(args.in_train, args.out_dir / "train.parquet", "train", cap_t)
    convert(args.in_val, args.out_dir / "val.parquet", "val", cap_v)
    print("done.")


if __name__ == "__main__":
    main()
