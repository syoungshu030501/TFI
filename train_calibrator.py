"""训练 calibrator: 在 val/ 上跑分割集成 + 分类器集成 + 证据抽取，多 backend + 5-fold CV。

输出:
    checkpoints/calibrator/calibrator.pkl     # 最终全量 refit 的模型
    checkpoints/calibrator/metrics.json       # 选中 backend 的 OOF 指标
    checkpoints/calibrator/compare.md         # --compare_all 时的全 backend 对比表
    cache/val_seg_probs.npz / val_cls_scores.json  # 重特征缓存

用法:
    # 单 backend（5-fold CV，OOF 阈值）
    python train_calibrator.py --backend tabpfn --cv_folds 5

    # 全 backend 对比（推荐先跑一次）
    python train_calibrator.py --compare_all --cv_folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

from calibrator import (
    SUPPORTED_BACKENDS,
    compare_backends,
    find_best_threshold,
    fit_calibrator_cv,
    hard_rule_baseline,
)
from dataset import TestImageDataset
from evidence import (
    FEATURE_NAMES_BASE,
    FEATURE_NAMES_WITH_CLS,
    extract,
    evidence_to_features,
)
from train_seg_ensemble import build_segformer, build_smp_model, SegModelWrapper
from utils import (
    compute_ela, compute_srm,
    postprocess_mask, mask_to_label,
)


# ============================================================
# 1) 在 val/ 上跑分割集成 -> 得到每张图的概率图 + binary mask
# ============================================================

def collect_val_samples(val_dir: str) -> List[Dict]:
    """收集 val 的 (image_path, mask_path|None, gt_label)。"""
    val_dir = Path(val_dir)
    samples = []
    # Black 类有 mask
    for fname in sorted(os.listdir(val_dir / "Black" / "Image")):
        stem = os.path.splitext(fname)[0]
        img_p = val_dir / "Black" / "Image" / fname
        mask_p = val_dir / "Black" / "Mask" / f"{stem}.png"
        samples.append({
            "name": fname, "image_path": str(img_p),
            "mask_path": str(mask_p) if mask_p.exists() else None,
            "gt_label": 1,
        })
    # White 类无 mask
    for fname in sorted(os.listdir(val_dir / "White" / "Image")):
        img_p = val_dir / "White" / "Image" / fname
        samples.append({
            "name": fname, "image_path": str(img_p),
            "mask_path": None, "gt_label": 0,
        })
    return samples


def load_seg_models(checkpoint_dir: str, archs: List[str]):
    seg_dir = Path(checkpoint_dir) / "seg"
    out = []
    for d in sorted(seg_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "best_model.pt").exists():
            continue
        for a in archs:
            if a in d.name:
                out.append({"name": d.name, "arch": a, "path": str(d / "best_model.pt")})
                break
    return out


def predict_seg_ensemble(samples: List[Dict], checkpoint_dir: str, device,
                         img_size: int = 768, archs=("segformer", "convnext", "maxvit")):
    """对 val 跑分割集成, 返回 {name: avg_prob_map_at_orig_size}。"""
    models_info = load_seg_models(checkpoint_dir, list(archs))
    print(f"[seg] using {len(models_info)} models @ {img_size}")
    name2prob_sum = {s["name"]: None for s in samples}

    # 用 TestImageDataset 已实现 7ch 输入预处理
    parent = Path(samples[0]["image_path"]).parent.parent.parent  # val/
    # 直接构 in-memory dataset:从 samples 的 image_path 收集图片
    # TestImageDataset 期望一个目录, 这里用临时方案: 自己读
    for mi in models_info:
        if mi["arch"] == "segformer":
            raw = build_segformer(in_channels=7, num_classes=1, pretrained=False)
            model = SegModelWrapper(raw, "segformer")
        else:
            raw = build_smp_model(mi["arch"], in_channels=7, num_classes=1, pretrained=False)
            model = SegModelWrapper(raw, "smp")
        state = torch.load(mi["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device).eval()

        for s in tqdm(samples, desc=f"  seg {mi['name']}", ncols=80):
            img = np.array(Image.open(s["image_path"]).convert("RGB"))
            ela = compute_ela(img)
            srm = compute_srm(img)
            x = np.concatenate([img.astype(np.float32) / 255.0,
                                ela.astype(np.float32) / 255.0,
                                srm.astype(np.float32)], axis=2)
            x = cv2.resize(x, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)

            tta_probs = []
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                for tfn, ifn in [
                    (lambda t: t, lambda t: t),
                    (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
                    (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
                ]:
                    logits = model(tfn(x))
                    tta_probs.append(ifn(torch.sigmoid(logits)))
            prob = torch.stack(tta_probs).mean(0)[0, 0].float().cpu().numpy()

            if name2prob_sum[s["name"]] is None:
                name2prob_sum[s["name"]] = prob
            else:
                name2prob_sum[s["name"]] = name2prob_sum[s["name"]] + prob

        del model
        torch.cuda.empty_cache()

    # 平均
    n = len(models_info)
    return {k: (v / n) if v is not None else None for k, v in name2prob_sum.items()}


def predict_cls_ensemble(samples: List[Dict], checkpoint_dir: str, device,
                         img_size: int = 512):
    """返回 {name: (mean, std)}。"""
    from train_classifier import ForgeryClassifier
    cls_dir = Path(checkpoint_dir) / "cls"
    model_dirs = sorted([d for d in cls_dir.iterdir()
                         if d.is_dir() and (d / "best_model.pt").exists()])
    print(f"[cls] using {len(model_dirs)} models")

    name2scores = {s["name"]: [] for s in samples}
    for d in model_dirs:
        model = ForgeryClassifier(in_channels=6, num_classes=2)
        state = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device).eval()
        for s in tqdm(samples, desc=f"  cls {d.name}", ncols=80):
            img = np.array(Image.open(s["image_path"]).convert("RGB"))
            img_r = cv2.resize(img, (img_size, img_size))
            ela = compute_ela(img_r)
            x = np.concatenate([img_r.astype(np.float32) / 255,
                                ela.astype(np.float32) / 255], axis=2)
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(x), dim=1)[0, 1].item()
            name2scores[s["name"]].append(p)
        del model
        torch.cuda.empty_cache()

    out = {}
    for k, v in name2scores.items():
        if v:
            out[k] = (float(np.mean(v)), float(np.std(v)))
        else:
            out[k] = (0.5, 0.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="data/raw/val")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--out_dir", default="checkpoints/calibrator")
    parser.add_argument("--backend", choices=list(SUPPORTED_BACKENDS),
                        default="tabpfn",
                        help="单 backend 训练；也可用 --compare_all 跑全部")
    parser.add_argument("--compare_all", action="store_true",
                        help="跑所有 backend 的 5-fold CV 对比，写 compare.md")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="seg 概率二值化阈值")
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_cls", action="store_true", default=True)
    parser.add_argument("--no_cls", action="store_false", dest="use_cls")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1) 收集 val 样本 ==========
    samples = collect_val_samples(args.val_dir)
    print(f"val samples: {len(samples)}")

    # ========== 2) 跑分割 + 分类 ==========
    seg_cache = cache_dir / "val_seg_probs.npz"
    cls_cache = cache_dir / "val_cls_scores.json"

    if not seg_cache.exists():
        t0 = time.time()
        probs = predict_seg_ensemble(samples, args.checkpoint_dir, device, args.img_size)
        np.savez_compressed(seg_cache, **{k: v for k, v in probs.items() if v is not None})
        print(f"[seg] cached in {time.time()-t0:.1f}s -> {seg_cache}")
    probs = dict(np.load(seg_cache))

    if args.use_cls:
        if not cls_cache.exists():
            t0 = time.time()
            cls_scores = predict_cls_ensemble(samples, args.checkpoint_dir, device)
            with open(cls_cache, "w") as f:
                json.dump(cls_scores, f)
            print(f"[cls] cached in {time.time()-t0:.1f}s -> {cls_cache}")
        with open(cls_cache, "r") as f:
            cls_scores = json.load(f)
    else:
        cls_scores = {}

    # ========== 3) 抽证据 + 拼特征 ==========
    print("[evidence] extracting...")
    X, y, evidence_records = [], [], {}
    for s in tqdm(samples, ncols=80):
        prob = probs.get(s["name"])
        if prob is None:
            continue
        img = np.array(Image.open(s["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (prob_full > args.threshold).astype(np.uint8)
        binary = postprocess_mask(binary, morph_kernel_size=5, min_area=args.min_area)

        ev = extract(img, binary, prob_map=prob_full, label_threshold=0.001,
                     min_area_px=args.min_area)
        cls_mean, cls_std = (cls_scores.get(s["name"], [0.5, 0.0])
                             if args.use_cls else (None, None))
        feats = evidence_to_features(ev, cls_score=cls_mean, cls_score_std=cls_std)
        X.append(feats)
        y.append(s["gt_label"])
        evidence_records[s["name"]] = ev

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    print(f"feature matrix: {X.shape}, positive: {y.sum()}/{len(y)}")

    feature_names = FEATURE_NAMES_WITH_CLS if args.use_cls else FEATURE_NAMES_BASE

    # ========== 4) 拟合 calibrator (5-fold CV, OOF threshold) ==========
    if args.compare_all:
        backends_to_run = list(SUPPORTED_BACKENDS)
        reports, md_table = compare_backends(
            X, y, backends=backends_to_run, feature_names=feature_names,
            n_splits=args.cv_folds, seed=args.seed)
        (out_dir / "compare.md").write_text(
            f"# Calibrator backend comparison ({args.cv_folds}-fold CV)\n\n"
            f"n={len(y)}, positive={int(y.sum())}, features={len(feature_names)}\n\n"
            + md_table + "\n", encoding="utf-8")
        print("\n" + md_table + "\n")
        print(f"[compare] table -> {out_dir/'compare.md'}")
        # 选 OOF F1 最大的 backend 作为最终 backend
        best_backend = max(reports.items(), key=lambda kv: kv[1].f1_oof)[0]
        print(f"[compare] picked best backend = {best_backend!r}")
    else:
        best_backend = args.backend

    cal, oof_report = fit_calibrator_cv(
        X, y, backend=best_backend, feature_names=feature_names,
        n_splits=args.cv_folds, seed=args.seed)
    print(f"[final] backend={best_backend}  OOF F1={oof_report.f1_oof:.4f}  "
          f"AUC={oof_report.auc_oof:.4f}  threshold={cal.threshold:.3f}")
    cal.save(str(out_dir))

    # ========== 5) 对照: 旧硬规则 / 仅 seg ==========
    from utils import compute_f1
    seg_labels = np.array([evidence_records[s["name"]]["label"]
                           if s["name"] in evidence_records else 0 for s in samples])
    cls_labels_hard = np.array([
        hard_rule_baseline(int(seg_labels[i]),
                            cls_scores.get(samples[i]["name"], [0.5, 0.0])[0] if args.use_cls else 0.5)
        for i in range(len(samples))
    ])
    m_seg = compute_f1(seg_labels, y)
    m_hard = compute_f1(cls_labels_hard, y)
    # calibrator OOF 预测（不偏估）
    cal_pred_oof = (oof_report.oof_probs > cal.threshold).astype(int)
    m_cal_oof = compute_f1(cal_pred_oof, y)

    metrics = {
        "n_val": int(len(samples)),
        "n_with_pred": int(len(y)),
        "feature_names": feature_names,
        "backend": best_backend,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "threshold_on_oof": float(cal.threshold),
        "seg_only": m_seg,
        "hard_rule": m_hard,
        "calibrator_oof": m_cal_oof,
        "oof_report": oof_report.to_dict(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("=" * 60)
    print(json.dumps({k: v for k, v in metrics.items() if k != "oof_report"},
                     indent=2, ensure_ascii=False))
    print("=" * 60)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
