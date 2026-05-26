"""§3.3 Caption 清洗 + 坐标重写。

读 logs/caption_bbox_audit.csv, 根据 max_iou 分桶:
  IoU >= 0.5: 直接复制原文, 仅清理 </think>
  0.2 <= IoU < 0.5: 把 caption 中与 GT 最接近的 bbox 字符串替换为 GT bbox
  IoU < 0.2 or n_cap_bbox==0: 写占位 (留给 §6 VLM 重生); 同时标记到 needs_regen.txt

输出:
  train/Black/Caption_clean/*.md  (与 Caption 同 stem)
  logs/needs_regen.txt           (低质量 stem 清单, 供 §6 调用)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from ast import literal_eval
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

THINK_RE = re.compile(r"</?think>", re.IGNORECASE)
BBOX_RE = re.compile(r"\[\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*\]")


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def rewrite_bboxes(caption: str, cap_bboxes, gt_bboxes) -> str:
    """把每个 caption bbox 替换为 IoU 最高的 GT bbox。"""
    out = caption
    for c in cap_bboxes:
        best, best_iou = None, -1.0
        for g in gt_bboxes:
            if iou(c, g) > best_iou:
                best, best_iou = g, iou(c, g)
        if best is None:
            continue
        old = f"[{c[0]}, {c[1]}, {c[2]}, {c[3]}]"
        new = f"[{best[0]}, {best[1]}, {best[2]}, {best[3]}]"
        if old in out:
            out = out.replace(old, new, 1)
        else:
            # 容错: 全角逗号 / 空格差异
            pattern = re.compile(
                r"\[\s*" + str(c[0]) + r"\s*[,，]\s*" + str(c[1]) +
                r"\s*[,，]\s*" + str(c[2]) + r"\s*[,，]\s*" + str(c[3]) + r"\s*\]"
            )
            out = pattern.sub(new, out, count=1)
    return out


def sanitize(text: str) -> str:
    text = THINK_RE.sub("", text)
    # 压缩多空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit_csv", default="logs/caption_bbox_audit.csv")
    p.add_argument("--root", default=".")
    p.add_argument("--need_regen_list", default="logs/needs_regen.txt")
    args = p.parse_args()

    root = Path(args.root).resolve()
    rows = list(csv.DictReader(open(args.audit_csv, "r", encoding="utf-8")))

    buckets = {"keep": 0, "rewrite": 0, "regen": 0, "err": 0}
    need_regen = []

    for row in rows:
        split = row["split"]; stem = row["stem"]
        cap_path = root / split / "Black" / "Caption" / f"{stem}.md"
        clean_dir = root / split / "Black" / "Caption_clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        dst = clean_dir / f"{stem}.md"

        try:
            text = sanitize(cap_path.read_text(encoding="utf-8"))
            iou_v = float(row["max_iou"]); n_cap = int(row["n_cap_bbox"])
            cap_bboxes = literal_eval(row["cap_bboxes"]) if row["cap_bboxes"] else []
            gt_bboxes = literal_eval(row["gt_bboxes"]) if row["gt_bboxes"] else []

            if iou_v >= 0.5:
                dst.write_text(text, encoding="utf-8")
                buckets["keep"] += 1
            elif iou_v >= 0.2 and gt_bboxes:
                text2 = rewrite_bboxes(text, cap_bboxes, gt_bboxes)
                dst.write_text(text2, encoding="utf-8")
                buckets["rewrite"] += 1
            else:
                # 低质量: 保留原文作占位, 标记到 needs_regen
                dst.write_text(text, encoding="utf-8")
                need_regen.append(f"{split}\t{stem}")
                buckets["regen"] += 1
        except Exception as e:
            print(f"  err {stem}: {e}")
            buckets["err"] += 1

    Path(args.need_regen_list).parent.mkdir(parents=True, exist_ok=True)
    with open(args.need_regen_list, "w", encoding="utf-8") as f:
        for line in need_regen:
            f.write(line + "\n")

    print(f"keep={buckets['keep']}  rewrite={buckets['rewrite']}  "
          f"regen={buckets['regen']}  err={buckets['err']}")
    print(f"needs_regen list -> {args.need_regen_list} ({len(need_regen)} stems)")


if __name__ == "__main__":
    main()
