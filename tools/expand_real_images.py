"""§5 真实图 (White) 扩充。

两部分:
  A. §5.2 表最后一行: White 原图强增广  (200 -> 200*N, 默认 N=3)
     - JPEG 重压 q in {65,75,85}
     - 轻度调色 (ColorJitter brightness/contrast/saturation)
     - 轻度 resize + 再resize回原尺寸
  B. §5.2 表第一行: COCO val2017 随机采 500 张, 生成简短中文 caption 模板
     - 假设 cache/coco/val2017.zip 已下载 (由 shell 端启动的 wget)

输出:
  augmented_data/real_ext/
  ├─ Image/*.jpg
  ├─ Caption/*.md
  └─ source.tsv        # 记录来源 (white_aug / coco)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REAL_CAPTION_TMPL = (
    "这是一张真实拍摄的{scene}照片，未发现数字伪造或后期篡改的痕迹。"
    "全图在字体、边缘过渡、纹理连续性、光照方向与强度上完全一致；"
    "JPEG 压缩伪影在整张画面均匀分布，噪点与颗粒感的强度保持一致；"
    "从物理合理性来看，遮挡关系、透视方向、阴影朝向均符合实拍规律，"
    "未见伪造中常出现的二次压缩带、拼接带边缘或局部噪声不匹配。"
    "从上下文逻辑分析，画面中的内容、物体关系、色温与景深彼此自洽，"
    "不存在明显的篡改、合成、重绘或图层叠加痕迹。"
    "综合分析，该图像真实记录了{scene}场景。"
)

DEFAULT_SCENES = [
    "户外自然风景", "城市街景", "室内日常生活", "人物纪实",
    "餐饮食物", "交通工具", "动植物", "体育运动",
    "节日庆典", "商业店铺", "办公场景", "家居环境",
]


def white_strong_aug(img_np: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img_np.shape[:2]

    # 1. JPEG 重压
    q = int(rng.choice([65, 75, 85]))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        img_np = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 2. 调色
    img = Image.fromarray(img_np)
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.85, 1.15)))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.15)))
    img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.85, 1.15)))
    img_np = np.array(img)

    # 3. 小幅 resize 往返
    scale = float(rng.uniform(0.85, 1.15))
    nw, nh = max(64, int(w * scale)), max(64, int(h * scale))
    img_np = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_AREA)
    img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_np


def do_white_aug(root: Path, n_per_image: int, out_dir: Path, source_rows: list):
    src_img = root / "train" / "White" / "Image"
    src_cap = root / "train" / "White" / "Caption"
    out_img = out_dir / "Image"; out_img.mkdir(parents=True, exist_ok=True)
    out_cap = out_dir / "Caption"; out_cap.mkdir(parents=True, exist_ok=True)

    files = sorted(os.listdir(src_img))
    rng = random.Random(42)
    for fname in files:
        stem = os.path.splitext(fname)[0]
        try:
            img = np.array(Image.open(src_img / fname).convert("RGB"))
        except Exception as e:
            print(f"  [skip] {fname}: {e}"); continue
        caption = (src_cap / f"{stem}.md").read_text(encoding="utf-8").strip() \
            if (src_cap / f"{stem}.md").exists() else ""
        for i in range(n_per_image):
            seed = rng.randint(0, 1_000_000)
            aug = white_strong_aug(img, seed=seed)
            out_stem = f"whiteaug_{stem}_{i}"
            Image.fromarray(aug).save(out_img / f"{out_stem}.jpg", "JPEG", quality=90)
            (out_cap / f"{out_stem}.md").write_text(caption, encoding="utf-8")
            source_rows.append({"stem": out_stem, "source": "white_aug", "parent": stem})
    print(f"  [white_aug] wrote {len(files) * n_per_image} samples")


def do_coco(root: Path, n_coco: int, out_dir: Path, source_rows: list):
    zip_path = root / "cache" / "coco" / "val2017.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 500 * 1024 * 1024:
        print(f"  [skip coco] {zip_path} missing or incomplete")
        return

    out_img = out_dir / "Image"; out_img.mkdir(parents=True, exist_ok=True)
    out_cap = out_dir / "Caption"; out_cap.mkdir(parents=True, exist_ok=True)

    print(f"  [coco] opening {zip_path} ...")
    rng = random.Random(123)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist()
                 if n.endswith((".jpg", ".jpeg", ".JPG")) and "val2017/" in n]
        rng.shuffle(names)
        picked = names[:n_coco]
        print(f"  [coco] sampling {len(picked)} / {len(names)}")
        for i, n in enumerate(picked):
            try:
                data = zf.read(n)
                out_stem = f"coco_{os.path.splitext(os.path.basename(n))[0]}"
                (out_img / f"{out_stem}.jpg").write_bytes(data)
                scene = rng.choice(DEFAULT_SCENES)
                cap = REAL_CAPTION_TMPL.format(scene=scene)
                (out_cap / f"{out_stem}.md").write_text(cap, encoding="utf-8")
                source_rows.append({"stem": out_stem, "source": "coco",
                                    "parent": os.path.basename(n)})
            except Exception as e:
                print(f"    err {n}: {e}")
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(picked)}")
    print(f"  [coco] wrote {len(picked)} samples")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out_dir", default="data/processed/real_ext")
    p.add_argument("--white_n", type=int, default=3, help="每张 White 产几张")
    p.add_argument("--coco_n", type=int, default=500)
    p.add_argument("--no_coco", action="store_true")
    p.add_argument("--no_white", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    if not args.no_white:
        print("[A] White 强增广")
        do_white_aug(root, args.white_n, out_dir, source_rows)
    if not args.no_coco:
        print("[B] COCO 采样")
        do_coco(root, args.coco_n, out_dir, source_rows)

    # source.tsv
    tsv = out_dir / "source.tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("stem\tsource\tparent\n")
        for r in source_rows:
            f.write(f"{r['stem']}\t{r['source']}\t{r['parent']}\n")
    print(f"Total {len(source_rows)} samples -> {out_dir}")
    print(f"source.tsv -> {tsv}")


if __name__ == "__main__":
    main()
