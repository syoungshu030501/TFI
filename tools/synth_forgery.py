"""§4 像素级自合成伪造样本 (仅供 seg/cls 训练用, 不进 VLM SFT)。

实现四种范式中的三种 (跳过 inpainting, 避免下 lama/SD):
  1. copy-move  : 同图内裁 patch + seamlessClone 到另一位置
  2. splicing   : 跨图拼接 (White 图 A 的 patch -> White 图 B)
  3. text_replace: 基于简单边缘检测+高斯模糊扰动模拟文字篡改 (OCR-free 版本)

每种范式天然知道篡改 mask。输出结构:
  augmented_data/synth/
    Image/*.jpg
    Mask/*.png
    meta.jsonl  (每条样本的范式、源图、patch 坐标等)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pick_random_rect(h: int, w: int, rng: random.Random,
                     min_side: int = 40, max_side_ratio: float = 0.3) -> Tuple[int, int, int, int]:
    max_side = int(min(h, w) * max_side_ratio)
    max_side = max(min_side + 8, max_side)
    sw = rng.randint(min_side, max_side)
    sh = rng.randint(min_side, max_side)
    x = rng.randint(0, w - sw - 1)
    y = rng.randint(0, h - sh - 1)
    return x, y, sw, sh


def copy_move(img_rgb: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    sx, sy, sw, sh = pick_random_rect(h, w, rng)
    # 找一个不与源重叠的目标位置
    for _ in range(20):
        dx, dy, dw, dh = pick_random_rect(h, w, rng)
        if abs(dx - sx) + abs(dy - sy) > max(sw, sh):
            break
    dw, dh = sw, sh  # 目标尺寸同源
    if dx + dw >= w: dx = max(0, w - dw - 1)
    if dy + dh >= h: dy = max(0, h - dh - 1)

    patch = img_rgb[sy:sy + sh, sx:sx + sw].copy()
    # 随机翻转一下让检测器别太容易
    if rng.random() < 0.5:
        patch = cv2.flip(patch, 1)

    dst = img_rgb.copy()
    try:
        mask_patch = np.full((sh, sw), 255, dtype=np.uint8)
        center = (dx + dw // 2, dy + dh // 2)
        bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        blended = cv2.seamlessClone(patch_bgr, bgr, mask_patch, center, cv2.NORMAL_CLONE)
        dst = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    except cv2.error:
        dst[dy:dy + dh, dx:dx + dw] = patch

    mask = np.zeros((h, w), np.uint8)
    mask[dy:dy + dh, dx:dx + dw] = 1
    return dst, mask


def splicing(img_a_rgb: np.ndarray, img_b_rgb: np.ndarray,
             rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """把 A 的一块 patch 拼到 B 上。mask 位于 B 上。"""
    hb, wb = img_b_rgb.shape[:2]
    ha, wa = img_a_rgb.shape[:2]
    # B 上的目标 bbox
    dx, dy, dw, dh = pick_random_rect(hb, wb, rng)
    # 从 A 上等大裁一块
    if wa < dw + 8 or ha < dh + 8:
        img_a_rgb = cv2.resize(img_a_rgb, (max(dw + 20, 64), max(dh + 20, 64)))
        ha, wa = img_a_rgb.shape[:2]
    sx = rng.randint(0, wa - dw - 1)
    sy = rng.randint(0, ha - dh - 1)
    patch = img_a_rgb[sy:sy + dh, sx:sx + dw].copy()

    dst = img_b_rgb.copy()
    try:
        mask_patch = np.full((dh, dw), 255, dtype=np.uint8)
        center = (dx + dw // 2, dy + dh // 2)
        bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        blended = cv2.seamlessClone(patch_bgr, bgr, mask_patch, center, cv2.MIXED_CLONE)
        dst = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    except cv2.error:
        dst[dy:dy + dh, dx:dx + dw] = patch

    mask = np.zeros((hb, wb), np.uint8)
    mask[dy:dy + dh, dx:dx + dw] = 1
    return dst, mask


def text_replace_like(img_rgb: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """在边缘密集区域模拟 "文字被篡改" —— 局部强高斯模糊 + 反差扰动 + 小 patch 搬运。"""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    edges_sum = cv2.boxFilter(edges.astype(np.float32), -1, (32, 32))
    ys, xs = np.where(edges_sum > edges_sum.mean() * 2)
    if len(ys) < 10:
        # 回退: 用 copy_move
        return copy_move(img_rgb, rng)
    idx = rng.randint(0, len(ys) - 1)
    cy, cx = int(ys[idx]), int(xs[idx])
    sw = rng.randint(60, min(180, w // 3))
    sh = rng.randint(24, min(60, h // 4))
    x0 = max(0, cx - sw // 2); y0 = max(0, cy - sh // 2)
    x1 = min(w, x0 + sw); y1 = min(h, y0 + sh)

    dst = img_rgb.copy()
    patch = dst[y0:y1, x0:x1].copy()
    # 模糊 + 提亮/降亮, 模拟二次处理
    patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=rng.uniform(0.8, 2.0))
    factor = rng.uniform(0.7, 1.3)
    patch = np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    # 小位移
    shift = rng.randint(-4, 4)
    if shift != 0:
        patch = np.roll(patch, shift, axis=1)
    dst[y0:y1, x0:x1] = patch

    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 1
    return dst, mask


def load_img(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_dir", default="data/processed/synth")
    ap.add_argument("--pool", default="data/raw/train_resume/White/Image",
                    help="作为 A/B 来源的真实图池 (纯真实图, 避免把伪造贴到伪造)")
    ap.add_argument("--n_copy_move", type=int, default=200)
    ap.add_argument("--n_splicing", type=int, default=300)
    ap.add_argument("--n_text_replace", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    (out_dir / "Image").mkdir(parents=True, exist_ok=True)
    (out_dir / "Mask").mkdir(parents=True, exist_ok=True)

    pool_dir = root / args.pool
    pool = sorted([pool_dir / f for f in os.listdir(pool_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"pool size: {len(pool)}")
    rng = random.Random(args.seed)
    meta_f = open(out_dir / "meta.jsonl", "w", encoding="utf-8")

    # 1. copy-move
    for i in range(args.n_copy_move):
        src = pool[rng.randint(0, len(pool) - 1)]
        img = load_img(src)
        try:
            out_img, mask = copy_move(img, rng)
        except Exception as e:
            print(f"  [copy_move err] {e}"); continue
        stem = f"copymove_{i:04d}"
        Image.fromarray(out_img).save(out_dir / "Image" / f"{stem}.jpg", quality=92)
        Image.fromarray(mask * 255).save(out_dir / "Mask" / f"{stem}.png")
        meta_f.write(json.dumps({"stem": stem, "type": "copy_move",
                                 "source_a": src.name}) + "\n")
        if (i + 1) % 50 == 0: print(f"  copy_move {i+1}/{args.n_copy_move}")

    # 2. splicing
    for i in range(args.n_splicing):
        a = pool[rng.randint(0, len(pool) - 1)]
        b = pool[rng.randint(0, len(pool) - 1)]
        while b == a and len(pool) > 1:
            b = pool[rng.randint(0, len(pool) - 1)]
        try:
            out_img, mask = splicing(load_img(a), load_img(b), rng)
        except Exception as e:
            print(f"  [splicing err] {e}"); continue
        stem = f"splice_{i:04d}"
        Image.fromarray(out_img).save(out_dir / "Image" / f"{stem}.jpg", quality=92)
        Image.fromarray(mask * 255).save(out_dir / "Mask" / f"{stem}.png")
        meta_f.write(json.dumps({"stem": stem, "type": "splicing",
                                 "source_a": a.name, "source_b": b.name}) + "\n")
        if (i + 1) % 50 == 0: print(f"  splicing {i+1}/{args.n_splicing}")

    # 3. text replace-like
    for i in range(args.n_text_replace):
        src = pool[rng.randint(0, len(pool) - 1)]
        try:
            out_img, mask = text_replace_like(load_img(src), rng)
        except Exception as e:
            print(f"  [text_replace err] {e}"); continue
        stem = f"textrep_{i:04d}"
        Image.fromarray(out_img).save(out_dir / "Image" / f"{stem}.jpg", quality=92)
        Image.fromarray(mask * 255).save(out_dir / "Mask" / f"{stem}.png")
        meta_f.write(json.dumps({"stem": stem, "type": "text_replace",
                                 "source_a": src.name}) + "\n")
        if (i + 1) % 20 == 0: print(f"  text_replace {i+1}/{args.n_text_replace}")

    meta_f.close()
    print(f"Done. outputs in {out_dir}")


if __name__ == "__main__":
    main()
