"""§3.2 Caption bbox 与 GT mask 一致性审计。

对 train_split/Black 和 val/Black 的 (caption, mask) 逐一:
  1. 用正则提取 caption 内的 [x1,y1,x2,y2] 坐标
  2. 用 evidence.extract_regions 从 GT mask 得到 bbox
  3. 计算 max IoU(每个 caption bbox) vs GT bboxes
  4. 输出 CSV 供人工 / 后续处理

输出: logs/caption_bbox_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# make sure we can import evidence.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evidence import extract_regions  # noqa: E402

BBOX_RE = re.compile(r"\[\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*\]")


def extract_caption_bboxes(text: str) -> List[List[int]]:
    out = []
    for m in BBOX_RE.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def max_iou_match(cap_bboxes: List[List[int]], gt_bboxes: List[List[int]]) -> float:
    if not cap_bboxes or not gt_bboxes:
        return 0.0
    best = 0.0
    for c in cap_bboxes:
        for g in gt_bboxes:
            best = max(best, iou(c, g))
    return best


def audit_one(stem: str, img_path: Path, cap_path: Path, mask_path: Path):
    caption = cap_path.read_text(encoding="utf-8")
    has_think = ("</think>" in caption) or ("<think>" in caption)
    cap_bboxes = extract_caption_bboxes(caption)

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    mask = mask.astype(np.uint8)

    # 若 mask 尺寸与图不同, resize 到同尺寸
    if mask.shape != (h, w):
        import cv2
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_regions = extract_regions(mask, min_area_px=64)
    gt_bboxes = [r["bbox"] for r in gt_regions]
    gt_area_ratio = sum(r["area_ratio"] for r in gt_regions)

    m_iou = max_iou_match(cap_bboxes, gt_bboxes)
    return {
        "stem": stem,
        "img_h": h, "img_w": w,
        "n_cap_bbox": len(cap_bboxes),
        "n_gt_bbox": len(gt_bboxes),
        "gt_area_ratio": round(gt_area_ratio, 4),
        "max_iou": round(m_iou, 4),
        "has_think_tag": int(has_think),
        "cap_len": len(caption),
        "cap_bboxes": str(cap_bboxes),
        "gt_bboxes": str(gt_bboxes),
    }


def audit_dir(data_dir: Path) -> List[dict]:
    black = data_dir / "Black"
    img_dir = black / "Image"; cap_dir = black / "Caption"; mask_dir = black / "Mask"
    out = []
    files = sorted(os.listdir(img_dir))
    for i, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        cap_p = cap_dir / f"{stem}.md"
        mask_p = mask_dir / f"{stem}.png"
        if not (cap_p.exists() and mask_p.exists()):
            continue
        try:
            row = audit_one(stem, img_dir / fname, cap_p, mask_p)
            row["split"] = data_dir.name
            out.append(row)
        except Exception as e:
            print(f"  [error] {stem}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  [{data_dir.name}] {i+1}/{len(files)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+",
                   default=["data/raw/train_resume", "data/raw/val"])
    p.add_argument("--out", default="logs/caption_bbox_audit.csv")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in args.dirs:
        path = root / d
        if not path.exists():
            print(f"skip {d}: not found")
            continue
        print(f"auditing {d} ...")
        rows.extend(audit_dir(path))

    fieldnames = ["split", "stem", "img_h", "img_w", "n_cap_bbox", "n_gt_bbox",
                  "gt_area_ratio", "max_iou", "has_think_tag", "cap_len",
                  "cap_bboxes", "gt_bboxes"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    n = len(rows)
    ge_05 = sum(1 for r in rows if r["max_iou"] >= 0.5 and r["n_cap_bbox"] > 0)
    mid = sum(1 for r in rows if 0.2 <= r["max_iou"] < 0.5 and r["n_cap_bbox"] > 0)
    low = sum(1 for r in rows if (r["max_iou"] < 0.2 and r["n_cap_bbox"] > 0) or r["n_cap_bbox"] == 0)
    has_think = sum(1 for r in rows if r["has_think_tag"])
    print("=" * 60)
    print(f"Total Black samples: {n}")
    print(f"  IoU >= 0.5 (keep):       {ge_05} ({ge_05/max(n,1)*100:.1f}%)")
    print(f"  0.2 <= IoU < 0.5 (rewrite): {mid} ({mid/max(n,1)*100:.1f}%)")
    print(f"  IoU < 0.2 or no bbox (regen): {low} ({low/max(n,1)*100:.1f}%)")
    print(f"  with <think> tag: {has_think}")
    print(f"CSV written to: {args.out}")


if __name__ == "__main__":
    main()
