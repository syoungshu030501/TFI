"""
独立分类器训练脚本
EfficientNet-V2-L, 5-Fold 交叉验证
输入: RGB(3) + ELA(3) = 6 通道
输出: 二分类 (0=真实, 1=伪造)

用法:
  python train_classifier.py --fold all --gpu 3
  python train_classifier.py --fold 0 --gpu 3
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

import timm

from dataset import ForgeryClsDataset, create_kfold_splits
from utils import compute_f1


PROJECT_ROOT = Path(__file__).resolve().parent


class ForgeryClassifier(nn.Module):
    """
    基于 EfficientNet-V2-L 的伪造分类器。
    修改第一层以接受 6 通道 (RGB + ELA)。
    """

    def __init__(self, in_channels=6, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        return self.backbone(x)


def _collect_labels(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [base.samples[i][1] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        labels = []
        for ds in dataset.datasets:
            labels.extend(_collect_labels(ds))
        return labels
    if hasattr(dataset, "get_labels"):
        return dataset.get_labels()
    raise TypeError(f"Unsupported dataset type for labels: {type(dataset)!r}")


def _build_balanced_sampler(labels):
    counts = Counter(labels)
    if len(counts) <= 1:
        return None
    max_count = max(counts.values())
    weights = [float(max_count) / float(counts[label]) for label in labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            logits = model(pixel_values)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"]

        with autocast(dtype=torch.bfloat16):
            logits = model(pixel_values)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    metrics = compute_f1(np.array(all_preds), np.array(all_labels))
    return metrics


def train_single_fold(fold, args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"\n{'='*60}")
    print(f"Classifier Training: fold={fold}, gpu={args.gpu}")
    print(f"{'='*60}")

    full_dataset = ForgeryClsDataset(args.data_dir, img_size=args.img_size, is_train=True)
    val_dataset = ForgeryClsDataset(args.data_dir, img_size=args.img_size, is_train=False)

    folds = create_kfold_splits(args.data_dir, n_folds=5, seed=42)
    train_indices, val_indices = folds[fold]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    extra_dataset = None
    if args.include_synth or args.include_real_ext:
        extra_dataset = ForgeryClsDataset(
            args.data_dir,
            img_size=args.img_size,
            is_train=True,
            include_base=False,
            synth_dir=args.synth_dir if args.include_synth else None,
            real_ext_dir=args.real_ext_dir if args.include_real_ext else None,
            use_keep_list=not args.ignore_synth_keep,
        )

    train_dataset = train_subset
    if extra_dataset is not None and len(extra_dataset) > 0:
        train_dataset = ConcatDataset([train_subset, extra_dataset])

    train_labels = _collect_labels(train_dataset)
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"[data] base_train={len(train_subset)}  extra={len(extra_dataset) if extra_dataset is not None else 0}  "
          f"val={len(val_subset)}  total_train={len(train_dataset)}  pos={n_pos}  neg={n_neg}")

    sampler = _build_balanced_sampler(train_labels) if args.use_weighted_sampler else None
    if args.use_weighted_sampler:
        print("[sampler] weighted sampler enabled" if sampler is not None
              else "[sampler] skipped (single-class labels)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # 类别权重 (Black:White ≈ 4:1)
    class_weights = None if sampler is not None else torch.tensor([1.0, 0.25], device=device)
    model = ForgeryClassifier(in_channels=6, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05,
    )
    scaler = GradScaler()

    save_dir = Path(args.save_dir) / "cls" / f"efficientnet_fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, scheduler
        )
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time": elapsed,
        }
        history.append(log)

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={train_loss:.4f} | "
              f"F1={val_metrics['f1']:.4f} | "
              f"Acc={val_metrics['accuracy']:.4f} | "
              f"{elapsed:.1f}s")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="分类器训练")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data/raw/train_resume")
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--include_synth", action="store_true")
    parser.add_argument("--synth_dir", type=str, default="data/processed/synth")
    parser.add_argument("--ignore_synth_keep", action="store_true")
    parser.add_argument("--include_real_ext", action="store_true")
    parser.add_argument("--real_ext_dir", type=str, default="data/processed/real_ext")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    args = parser.parse_args()

    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    results = {}
    for fold in folds:
        best_f1 = train_single_fold(fold, args)
        results[f"fold{fold}"] = best_f1

    print(f"\n{'='*60}")
    print("Classifier Training Summary:")
    for k, v in results.items():
        print(f"  {k}: F1={v:.4f}")
    print(f"  Average F1: {np.mean(list(results.values())):.4f}")


if __name__ == "__main__":
    main()
