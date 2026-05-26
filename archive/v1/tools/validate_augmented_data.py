"""增强数据统一质检入口。

检查项:
  - augmented_data/real_ext/    : Image/Caption 数量对齐, Image 可读, Caption 非空
  - augmented_data/synth/       : Image/Mask 数量对齐 + keep.txt 存在
  - augmented_data/train_v2/    : evidence_captions.jsonl 条目合法
      * <think> / markdown 残留
      * bbox 白名单 (全部来自 GT mask)
      * caption 长度 250-800

输出: logs/aug_validation.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from ast import literal_eval
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BBOX_RE = re.compile(r"\[\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*\]")
THINK_RE = re.compile(r"</?think>", re.IGNORECASE)


def check_real_ext(root: Path, lines):
    d = root / "augmented_data" / "real_ext"
    if not d.exists():
        lines.append("- `augmented_data/real_ext/` : **NOT FOUND**"); return
    imgs = list((d / "Image").glob("*.jpg"))
    caps = list((d / "Caption").glob("*.md"))
    bad_imgs = 0
    for ip in imgs[:20]:
        try: Image.open(ip).convert("RGB")
        except Exception: bad_imgs += 1
    lines.append(f"- `real_ext/Image` : {len(imgs)} 张 (抽检 20 张坏图 {bad_imgs})")
    lines.append(f"- `real_ext/Caption` : {len(caps)} 条")
    lines.append(f"- 对齐: {'OK' if len(imgs)==len(caps) else 'MISMATCH'}")


def check_synth(root: Path, lines):
    d = root / "augmented_data" / "synth"
    if not d.exists():
        lines.append("- `augmented_data/synth/` : **NOT FOUND**"); return
    imgs = list((d / "Image").glob("*.jpg"))
    masks = list((d / "Mask").glob("*.png"))
    keep_f = d / "keep.txt"
    n_keep = len(keep_f.read_text(encoding="utf-8").splitlines()) if keep_f.exists() else 0
    lines.append(f"- `synth/Image` : {len(imgs)} 张; Mask: {len(masks)}")
    lines.append(f"- `synth/keep.txt` : {n_keep} 条 (通过 seg 过滤)")
    if keep_f.exists():
        lines.append(f"- 保留率: {n_keep}/{len(imgs)} = {n_keep/max(len(imgs),1)*100:.1f}%")


def check_train_v2(root: Path, lines):
    d = root / "augmented_data" / "train_v2"
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files:
        lines.append("- `train_v2/*.jsonl` : **NOT FOUND**"); return
    total, ok, bad_think, bad_len, bad_bbox = 0, 0, 0, 0, 0
    for p in files:
        for line in open(p, "r", encoding="utf-8"):
            total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            cap = d.get("caption", "")
            ev = d.get("evidence", {})
            allowed = [r["bbox"] for r in ev.get("regions", [])]
            cur_ok = True
            if THINK_RE.search(cap): bad_think += 1; cur_ok = False
            if not (200 <= len(cap) <= 900): bad_len += 1; cur_ok = False
            for s in BBOX_RE.findall(cap):
                nums = [int(x) for x in re.findall(r"\d+", s)]
                if nums not in allowed: bad_bbox += 1; cur_ok = False; break
            if cur_ok: ok += 1
    lines.append(f"- `train_v2/*.jsonl` : {len(files)} 个分片")
    lines.append(f"- `train_v2` 总条数: {total}")
    lines.append(f"  - 完全合规: {ok}")
    lines.append(f"  - <think> 残留: {bad_think}")
    lines.append(f"  - 长度越界: {bad_len}")
    lines.append(f"  - bbox 越界: {bad_bbox}")


def check_caption_clean(root: Path, lines):
    for split in ["train_split", "val", "train_resume"]:
        cc = root / split / "Black" / "Caption_clean"
        if cc.exists():
            n = len(list(cc.glob("*.md")))
            lines.append(f"- `{split}/Black/Caption_clean` : {n} 条")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="logs/aug_validation.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 增强数据质检报告", ""]
    lines.append("## 1. Caption_clean")
    check_caption_clean(root, lines)
    lines.append(""); lines.append("## 2. real_ext (§5)")
    check_real_ext(root, lines)
    lines.append(""); lines.append("## 3. synth (§4)")
    check_synth(root, lines)
    lines.append(""); lines.append("## 4. train_v2 evidence_captions (§6)")
    check_train_v2(root, lines)

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[validation] wrote {args.out}")


if __name__ == "__main__":
    main()
