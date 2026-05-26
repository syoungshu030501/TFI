"""§4.3 用已训好的 seg 集成反向过滤合成样本。

策略:
  iou < 0.10 -> drop (合成太差, 模型都检不出, 无价值)
  iou > 0.90 -> drop (太容易, 不提供新信息)
  0.10 <= iou <= 0.90 -> keep (hard but solvable, 训练价值最大)

输出:
  augmented_data/synth/keep.txt   # 保留 stem 列表
  augmented_data/synth/dropped.csv # 含原因
  更新 meta.jsonl 中的 keep 字段
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import compute_ela, compute_srm, postprocess_mask, compute_iou  # noqa: E402


def build(arch):
    from train_seg_ensemble import build_segformer, build_smp_model, SegModelWrapper
    if arch == "segformer":
        return SegModelWrapper(build_segformer(in_channels=7, num_classes=1, pretrained=False), "segformer")
    return SegModelWrapper(build_smp_model(arch, in_channels=7, num_classes=1, pretrained=False), "smp")


def load_models(ckpt_dir: Path, archs=("segformer", "maxvit", "convnext")):
    out = []
    seg = ckpt_dir / "seg"
    for d in sorted(seg.iterdir()):
        if not d.is_dir() or not (d / "best_model.pt").exists():
            continue
        for a in archs:
            if a in d.name:
                out.append({"name": d.name, "arch": a, "path": str(d / "best_model.pt")})
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth_dir", default="data/processed/synth")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--img_size", type=int, default=768)
    p.add_argument("--seg_thresh", type=float, default=0.3)
    p.add_argument("--low_iou", type=float, default=0.10)
    p.add_argument("--high_iou", type=float, default=0.90)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="可选: cpu / cuda / cuda:0")
    p.add_argument("--archs", nargs="+", default=["segformer", "maxvit"])
    args = p.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    root = Path(__file__).resolve().parent.parent
    synth = (root / args.synth_dir).resolve()
    img_dir = synth / "Image"; mask_dir = synth / "Mask"

    files = sorted(os.listdir(img_dir))
    print(f"synth samples: {len(files)}")

    models = load_models(root / args.checkpoint_dir, tuple(args.archs))
    print(f"using {len(models)} seg models: {[m['arch'] for m in models]}")
    if not models:
        print("[skip] no segmentation checkpoints found")
        return

    # ========== 对每张合成图, 跑 seg 集成 ==========
    name2ious: Dict[str, List[float]] = {f: [] for f in files}

    for mi in models:
        model = build(mi["arch"])
        state = torch.load(mi["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device).eval()
        for fname in tqdm(files, desc=f"  {mi['name']}", ncols=80):
            try:
                img = np.array(Image.open(img_dir / fname).convert("RGB"))
                stem = os.path.splitext(fname)[0]
                gt = np.array(Image.open(mask_dir / f"{stem}.png").convert("L")) > 127
                gt = gt.astype(np.uint8)

                h, w = img.shape[:2]
                ela = compute_ela(img); srm = compute_srm(img)
                x = np.concatenate([img.astype(np.float32) / 255,
                                    ela.astype(np.float32) / 255,
                                    srm.astype(np.float32)], axis=2)
                x = cv2.resize(x, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
                x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
                amp_ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with torch.no_grad(), amp_ctx:
                    prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
                prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                binary = (prob > args.seg_thresh).astype(np.uint8)
                binary = postprocess_mask(binary, morph_kernel_size=5, min_area=64)
                name2ious[fname].append(compute_iou(binary, gt))
            except Exception as e:
                name2ious[fname].append(0.0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========== 决策 ==========
    keep, dropped = [], []
    for fname, ious in name2ious.items():
        if not ious:
            dropped.append((fname, "no_seg_pred", 0.0)); continue
        mean_iou = float(np.mean(ious))
        if mean_iou < args.low_iou:
            dropped.append((fname, "too_easy_to_miss", mean_iou))
        elif mean_iou > args.high_iou:
            dropped.append((fname, "too_easy", mean_iou))
        else:
            keep.append((fname, mean_iou))

    (synth / "keep.txt").write_text("\n".join(k[0] for k in keep), encoding="utf-8")
    with open(synth / "dropped.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["fname", "reason", "mean_iou"])
        for r in dropped:
            w.writerow(r)

    meta_path = synth / "meta.jsonl"
    if meta_path.exists():
        keep_map = {fname: mean_iou for fname, mean_iou in keep}
        drop_map = {fname: (reason, mean_iou) for fname, reason, mean_iou in dropped}
        rows = []
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            stem = item.get("stem", "")
            fname = f"{stem}.jpg"
            if fname in keep_map:
                item["keep"] = True
                item["filter_iou"] = keep_map[fname]
            elif fname in drop_map:
                item["keep"] = False
                item["filter_iou"] = drop_map[fname][1]
                item["drop_reason"] = drop_map[fname][0]
            rows.append(item)
        with open(meta_path, "w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"keep: {len(keep)}  dropped: {len(dropped)}")
    print(f"  keep mean IoU: {np.mean([k[1] for k in keep]) if keep else 0:.3f}")
    print(f"  keep.txt  -> {synth/'keep.txt'}")
    print(f"  dropped   -> {synth/'dropped.csv'}")


if __name__ == "__main__":
    main()
