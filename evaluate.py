"""消融评估脚本 (Ablation evaluator)。

在 val/ 上对各模块开关跑消融, 输出对照表到 logs/ablation.md。

支持的消融维度:
  --ablate seg_arch={all,segformer_only,maxvit_only,segformer_maxvit,no_convnext}
  --ablate evidence={none,region_text,full_json}
  --ablate calibrator={hard,xgb,logistic,seg_only}
  --ablate cls={on,off}
  --ablate tta={on,off,multiscale}

通用用法:
  # 全消融 (会复用 cache 中的 seg/cls 缓存)
  python evaluate.py --val_dir val --full_ablation

  # 单组实验
  python evaluate.py --val_dir val \
      --seg_arch segformer_maxvit --evidence full_json --calibrator xgb
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

from calibrator import Calibrator, find_best_threshold, fit_calibrator, hard_rule_baseline
from evidence import (
    FEATURE_NAMES_BASE, FEATURE_NAMES_WITH_CLS,
    extract, evidence_to_features,
)
from utils import (
    compute_f1, compute_iou, compute_dice,
    postprocess_mask, mask_to_label, mask_to_rle,
)


SEG_ARCH_PRESETS = {
    "all": ["segformer", "convnext", "maxvit"],
    "segformer_only": ["segformer"],
    "maxvit_only": ["maxvit"],
    "segformer_maxvit": ["segformer", "maxvit"],
    "no_convnext": ["segformer", "maxvit"],
}


def collect_val_samples(val_dir: str) -> List[Dict]:
    val_dir = Path(val_dir)
    samples = []
    for fname in sorted(os.listdir(val_dir / "Black" / "Image")):
        stem = os.path.splitext(fname)[0]
        mask_p = val_dir / "Black" / "Mask" / f"{stem}.png"
        samples.append({
            "name": fname,
            "image_path": str(val_dir / "Black" / "Image" / fname),
            "mask_path": str(mask_p) if mask_p.exists() else None,
            "gt_label": 1,
        })
    for fname in sorted(os.listdir(val_dir / "White" / "Image")):
        samples.append({
            "name": fname,
            "image_path": str(val_dir / "White" / "Image" / fname),
            "mask_path": None,
            "gt_label": 0,
        })
    return samples


def load_seg_models(checkpoint_dir: str, archs: List[str]):
    seg_dir = Path(checkpoint_dir) / "seg"
    out = []
    for d in sorted(seg_dir.iterdir()):
        if not d.is_dir() or not (d / "best_model.pt").exists():
            continue
        for a in archs:
            if a in d.name:
                out.append({"name": d.name, "arch": a, "path": str(d / "best_model.pt")})
                break
    return out


def predict_seg(samples, checkpoint_dir, archs, device, img_sizes, use_tta):
    """返回 {name: avg_prob_at_orig}。"""
    from train_seg_ensemble import build_segformer, build_smp_model, SegModelWrapper
    from utils import compute_ela, compute_srm

    models = load_seg_models(checkpoint_dir, archs)
    print(f"  using {len(models)} seg models, sizes={img_sizes}, tta={use_tta}")
    name2sum, name2cnt = {}, {}

    for sz in img_sizes:
        for mi in models:
            if mi["arch"] == "segformer":
                raw = build_segformer(in_channels=7, num_classes=1, pretrained=False)
                model = SegModelWrapper(raw, "segformer")
            else:
                raw = build_smp_model(mi["arch"], in_channels=7, num_classes=1, pretrained=False)
                model = SegModelWrapper(raw, "smp")
            state = torch.load(mi["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model = model.to(device).eval()
            for s in tqdm(samples, ncols=80, desc=f"    {mi['name']}@{sz}"):
                img = np.array(Image.open(s["image_path"]).convert("RGB"))
                ela = compute_ela(img); srm = compute_srm(img)
                x = np.concatenate([img.astype(np.float32) / 255.0,
                                    ela.astype(np.float32) / 255.0,
                                    srm.astype(np.float32)], axis=2)
                x = cv2.resize(x, (sz, sz), interpolation=cv2.INTER_LINEAR)
                x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)

                with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                    if use_tta:
                        ttas = []
                        for tfn, ifn in [
                            (lambda t: t, lambda t: t),
                            (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
                            (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
                        ]:
                            ttas.append(ifn(torch.sigmoid(model(tfn(x)))))
                        prob = torch.stack(ttas).mean(0)[0, 0].float().cpu().numpy()
                    else:
                        prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()

                h, w = img.shape[:2]
                prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                if s["name"] not in name2sum:
                    name2sum[s["name"]] = prob; name2cnt[s["name"]] = 1
                else:
                    name2sum[s["name"]] += prob; name2cnt[s["name"]] += 1
            del model; torch.cuda.empty_cache()
    return {k: v / name2cnt[k] for k, v in name2sum.items()}


def predict_cls(samples, checkpoint_dir, device):
    from train_classifier import ForgeryClassifier
    from utils import compute_ela
    cls_dir = Path(checkpoint_dir) / "cls"
    model_dirs = sorted([d for d in cls_dir.iterdir()
                         if d.is_dir() and (d / "best_model.pt").exists()])
    print(f"  using {len(model_dirs)} classifiers")
    name2scores = {s["name"]: [] for s in samples}
    for d in model_dirs:
        model = ForgeryClassifier(in_channels=6, num_classes=2)
        state = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device).eval()
        for s in tqdm(samples, ncols=80, desc=f"    {d.name}"):
            img = np.array(Image.open(s["image_path"]).convert("RGB"))
            img_r = cv2.resize(img, (512, 512))
            ela = compute_ela(img_r)
            x = np.concatenate([img_r.astype(np.float32) / 255,
                                ela.astype(np.float32) / 255], axis=2)
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(x), dim=1)[0, 1].item()
            name2scores[s["name"]].append(p)
        del model; torch.cuda.empty_cache()
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in name2scores.items()}


def evaluate_one(samples, probs, cls_scores, gt_masks, *,
                 seg_thresh: float = 0.3,
                 calibrator_mode: str = "seg_only",
                 use_cls: bool = True,
                 cal: Optional[Calibrator] = None,
                 min_area: int = 100):
    """跑一组配置, 返回 {acc, prec, rec, f1, mean_iou, mean_dice}。"""
    pred_labels, true_labels = [], []
    ious, dices = [], []

    for s in samples:
        name = s["name"]
        prob = probs.get(name)
        if prob is None:
            continue
        h, w = prob.shape
        binary = (prob > seg_thresh).astype(np.uint8)
        binary = postprocess_mask(binary, morph_kernel_size=5, min_area=min_area)
        ev = extract(np.array(Image.open(s["image_path"]).convert("RGB")),
                     binary, prob_map=prob, label_threshold=0.001, min_area_px=min_area)
        cls_mean, cls_std = (cls_scores.get(name, (0.5, 0.0)) if use_cls else (None, None))

        if calibrator_mode == "seg_only":
            label = ev["label"]
        elif calibrator_mode == "hard":
            label = hard_rule_baseline(ev["label"], cls_mean if cls_mean is not None else 0.5)
        elif calibrator_mode in ("xgb", "logistic"):
            assert cal is not None, f"need calibrator for mode={calibrator_mode}"
            _, label = cal.predict(ev, cls_mean=cls_mean, cls_std=cls_std)
        else:
            raise ValueError(calibrator_mode)
        pred_labels.append(label); true_labels.append(s["gt_label"])

        # IoU/Dice 仅在 GT mask 存在时计算
        if s["mask_path"] is not None:
            gt_m = gt_masks.get(name)
            if gt_m is None:
                gt_m = (np.array(Image.open(s["mask_path"]).convert("L")) > 127).astype(np.uint8)
                if gt_m.shape != binary.shape:
                    gt_m = cv2.resize(gt_m, (w, h), interpolation=cv2.INTER_NEAREST)
                gt_masks[name] = gt_m
            ious.append(compute_iou(binary, gt_m))
            dices.append(compute_dice(binary, gt_m))

    pred_labels = np.array(pred_labels); true_labels = np.array(true_labels)
    m = compute_f1(pred_labels, true_labels)
    m["mean_iou"] = float(np.mean(ious)) if ious else 0.0
    m["mean_dice"] = float(np.mean(dices)) if dices else 0.0
    return m


def fit_calibrator_on(samples, probs, cls_scores, gt_labels, *, backend, use_cls,
                      seg_thresh=0.3, min_area=100):
    X, y = [], []
    for s in samples:
        prob = probs.get(s["name"])
        if prob is None:
            continue
        h, w = prob.shape
        binary = (prob > seg_thresh).astype(np.uint8)
        binary = postprocess_mask(binary, morph_kernel_size=5, min_area=min_area)
        ev = extract(np.array(Image.open(s["image_path"]).convert("RGB")),
                     binary, prob_map=prob, min_area_px=min_area)
        cls_mean, cls_std = (cls_scores.get(s["name"], (0.5, 0.0)) if use_cls else (None, None))
        feats = evidence_to_features(ev, cls_score=cls_mean, cls_score_std=cls_std)
        X.append(feats); y.append(s["gt_label"])
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int32)
    feature_names = FEATURE_NAMES_WITH_CLS if use_cls else FEATURE_NAMES_BASE
    return fit_calibrator(X, y, backend=backend, feature_names=feature_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir", default="data/raw/val")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--out_md", default="logs/ablation.md")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--full_ablation", action="store_true")
    p.add_argument("--seg_arch", choices=list(SEG_ARCH_PRESETS),
                   default="all")
    p.add_argument("--calibrator", choices=["seg_only", "hard", "xgb", "logistic"],
                   default="xgb")
    p.add_argument("--use_cls", action="store_true", default=True)
    p.add_argument("--no_cls", action="store_false", dest="use_cls")
    p.add_argument("--use_tta", action="store_true", default=True)
    p.add_argument("--no_tta", action="store_false", dest="use_tta")
    p.add_argument("--multiscale", action="store_true",
                   help="启用多尺度 [640,768,896]")
    p.add_argument("--seg_thresh", type=float, default=0.3)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    samples = collect_val_samples(args.val_dir)
    print(f"val samples: {len(samples)}")

    # 缓存全 arch 的 seg probs (一次性贵, 后面消融重用)
    cache_key = lambda arch_key, sizes, tta: f"val_probs_{arch_key}_{'-'.join(map(str,sizes))}_tta{int(tta)}.npz"

    def get_probs(arch_key: str):
        archs = SEG_ARCH_PRESETS[arch_key]
        sizes = [640, 768, 896] if args.multiscale else [768]
        key = cache_dir / cache_key(arch_key, sizes, args.use_tta)
        if key.exists():
            print(f"  cache hit: {key.name}")
            blob = np.load(key)
            return {k: blob[k] for k in blob.files}
        probs = predict_seg(samples, args.checkpoint_dir, archs, device,
                            img_sizes=sizes, use_tta=args.use_tta)
        np.savez_compressed(key, **probs)
        return probs

    # 分类器缓存
    cls_cache = cache_dir / "val_cls_scores.json"
    if cls_cache.exists():
        with open(cls_cache, "r") as f:
            tmp = json.load(f)
        cls_scores = {k: tuple(v) if isinstance(v, list) else v for k, v in tmp.items()}
    else:
        cls_scores = predict_cls(samples, args.checkpoint_dir, device)
        with open(cls_cache, "w") as f:
            json.dump({k: list(v) for k, v in cls_scores.items()}, f)

    gt_masks = {}
    rows: List[Tuple[str, Dict]] = []

    if args.full_ablation:
        archs_grid = ["all", "segformer_only", "maxvit_only", "segformer_maxvit"]
        cal_grid = ["seg_only", "hard", "xgb", "logistic"]
        for arch in archs_grid:
            probs = get_probs(arch)
            for cm in cal_grid:
                cal = None
                if cm in ("xgb", "logistic"):
                    cal = fit_calibrator_on(samples, probs, cls_scores,
                                            None, backend=cm, use_cls=args.use_cls,
                                            seg_thresh=args.seg_thresh)
                m = evaluate_one(samples, probs, cls_scores, gt_masks,
                                 seg_thresh=args.seg_thresh,
                                 calibrator_mode=cm, use_cls=args.use_cls,
                                 cal=cal)
                tag = f"seg={arch} | cal={cm} | cls={'on' if args.use_cls else 'off'}"
                rows.append((tag, m))
                print(f"  {tag}: {m}")
    else:
        probs = get_probs(args.seg_arch)
        cal = None
        if args.calibrator in ("xgb", "logistic"):
            cal = fit_calibrator_on(samples, probs, cls_scores, None,
                                    backend=args.calibrator, use_cls=args.use_cls,
                                    seg_thresh=args.seg_thresh)
        m = evaluate_one(samples, probs, cls_scores, gt_masks,
                         seg_thresh=args.seg_thresh,
                         calibrator_mode=args.calibrator, use_cls=args.use_cls,
                         cal=cal)
        tag = (f"seg={args.seg_arch} | cal={args.calibrator} | cls={'on' if args.use_cls else 'off'}"
               f" | tta={'on' if args.use_tta else 'off'} | multiscale={args.multiscale}")
        rows.append((tag, m))
        print(f"\n=== {tag} ===\n{json.dumps(m, indent=2)}")

    # 写 markdown 报告
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# TFI Ablation Report\n\n")
        f.write(f"_val_dir = `{args.val_dir}`, n = {len(samples)}_\n\n")
        f.write("| Config | Acc | Precision | Recall | F1 | mean IoU | mean Dice |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for tag, m in rows:
            f.write(f"| {tag} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                    f"{m['recall']:.4f} | {m['f1']:.4f} | "
                    f"{m['mean_iou']:.4f} | {m['mean_dice']:.4f} |\n")
    print(f"\nReport written to {args.out_md}")


if __name__ == "__main__":
    main()
