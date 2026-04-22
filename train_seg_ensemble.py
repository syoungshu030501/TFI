"""
分割集成训练脚本
3 种架构 x 5-Fold 交叉验证 = 15 个模型

架构:
  M1: SegFormer-B5 (mit-b5 backbone + MLP head, via HuggingFace)
  M2: ConvNeXt-V2-Large + DeepLabV3+ (via segmentation-models-pytorch)
  M3: MaxViT-Large + FPN (via segmentation-models-pytorch)

输入: 7 通道 (RGB + ELA + SRM)
输出: 二值分割 mask

用法:
  # 训练所有架构所有折 (推荐在多卡上并行)
  python train_seg_ensemble.py --arch all --fold all --gpu 0

  # 训练单个架构单折
  python train_seg_ensemble.py --arch segformer --fold 0 --gpu 0
  python train_seg_ensemble.py --arch convnext --fold 0 --gpu 1
  python train_seg_ensemble.py --arch maxvit --fold 0 --gpu 2

  # 数据来源（默认值，可按需覆盖）：
  #   --data_dir     data/raw/train_resume
  #   --synth_dir    data/processed/synth
  #   --real_ext_dir data/processed/real_ext
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from dataset import ForgerySegDataset, create_kfold_splits
from utils import compute_iou, compute_dice, compute_pixel_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. 损失函数
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """边界损失: 增强对伪造区域边缘的关注"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        # 用 Laplacian 提取边界
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size,
                          device=pred.device) / (self.kernel_size ** 2)
        boundary = torch.abs(
            F.conv2d(target, kernel, padding=self.kernel_size // 2) - target
        )
        boundary = (boundary > 0.1).float()

        # 在边界区域加权 BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted = bce * (1 + 5 * boundary)
        return weighted.mean()


class CombinedLoss(nn.Module):
    """组合损失: Focal + Dice + Boundary"""
    def __init__(self, focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.w_focal = focal_weight
        self.w_dice = dice_weight
        self.w_boundary = boundary_weight

    def forward(self, pred, target):
        return (self.w_focal * self.focal(pred, target)
                + self.w_dice * self.dice(pred, target)
                + self.w_boundary * self.boundary(pred, target))


# ============================================================
# 2. 模型构建
# ============================================================

def build_segformer(in_channels=7, num_classes=1, pretrained=True):
    """
    SegFormer-B5 via HuggingFace transformers.
    修改第一层 patch embedding 以支持 7 通道输入。
    """
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    # 优先使用本地模型
    model_path = str(PROJECT_ROOT / "models" / "segformer-b5")
    if not os.path.exists(model_path):
        model_path = "nvidia/segformer-b5-finetuned-ade-640-640"

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        config = SegformerConfig.from_pretrained(model_path)
        config.num_labels = num_classes
        model = SegformerForSemanticSegmentation(config)

    # 修改第一层 patch embedding: 3ch → 7ch
    old_proj = model.segformer.encoder.patch_embeddings[0].proj
    new_proj = nn.Conv2d(
        in_channels, old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    )
    # 复制原始 3ch 权重, 新增通道初始化为 0
    with torch.no_grad():
        new_proj.weight[:, :3] = old_proj.weight
        new_proj.weight[:, 3:] = 0
        new_proj.bias.copy_(old_proj.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new_proj

    return model


def build_smp_model(arch_name, in_channels=7, num_classes=1, pretrained=True):
    """使用 segmentation-models-pytorch 构建模型。

    Args:
        arch_name: "convnext" 或 "maxvit"
        pretrained: False 时 encoder_weights=None, 避免推理/过滤阶段
                    因为无外网访问而卡住 (timm 从 HF hub 下权重)。
                    只要后续 load_state_dict 加载 checkpoint, 即可覆盖。
    """
    import segmentation_models_pytorch as smp

    encoder_map = {
        "convnext": {
            "encoder": "tu-convnextv2_large.fcmae_ft_in22k_in1k_384",
            "decoder": "DeepLabV3Plus",
        },
        "maxvit": {
            "encoder": "tu-maxvit_large_tf_384.in21k_ft_in1k",
            "decoder": "FPN",
        },
    }

    cfg = encoder_map[arch_name]
    decoder_cls = getattr(smp, cfg["decoder"])

    model = decoder_cls(
        encoder_name=cfg["encoder"],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model


class SegModelWrapper(nn.Module):
    """
    统一接口包装器。
    不管底层是 HF SegFormer 还是 SMP 模型, 都返回 (logits, loss)。
    """

    def __init__(self, model, model_type="smp"):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, pixel_values, mask=None):
        if self.model_type == "segformer":
            # HF SegFormer: 输出 logits shape (B, num_classes, H/4, W/4)
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # (B, 1, H/4, W/4)
            # 上采样到原始分辨率
            logits = F.interpolate(
                logits, size=pixel_values.shape[2:],
                mode="bilinear", align_corners=False
            )
        else:
            # SMP 模型: 输出 logits shape (B, 1, H, W)
            logits = self.model(pixel_values)

        return logits


# ============================================================
# 3. 训练循环
# ============================================================


def _collect_labels(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [base.samples[i][2] for i in dataset.indices]
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


def _parse_resume_log(log_path: str | None, arch: str, fold: int):
    if not log_path:
        return 0, 0.0
    path = Path(log_path)
    if not path.exists():
        return 0, 0.0

    header_re = re.compile(r"Training:\s+arch=(\w+),\s+fold=(\d+),\s+gpu=\d+")
    epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
    best_re = re.compile(r"New best IoU:\s*([0-9.]+)")

    in_target_block = False
    last_epoch = 0
    best_iou = 0.0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        header_match = header_re.search(line)
        if header_match:
            in_target_block = (
                header_match.group(1) == arch and int(header_match.group(2)) == fold
            )
            continue
        if not in_target_block:
            continue

        epoch_match = epoch_re.search(line)
        if epoch_match:
            last_epoch = max(last_epoch, int(epoch_match.group(1)))

        best_match = best_re.search(line)
        if best_match:
            best_iou = float(best_match.group(1))

    return last_epoch, best_iou


def _resolve_resume_state(save_dir: Path, arch: str, fold: int, args):
    history_path = save_dir / "history.json"
    last_ckpt_path = save_dir / "last_checkpoint.pt"
    best_model_path = save_dir / "best_model.pt"

    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not resume_from.exists():
        raise FileNotFoundError(f"resume_from not found: {resume_from}")

    if resume_from is not None:
        if resume_from.name == "last_checkpoint.pt":
            return {"mode": "full", "path": resume_from, "history": history}
        return {
            "mode": "model",
            "path": resume_from,
            "history": history,
            "start_epoch": len(history),
            "best_iou": max((item.get("val_iou", 0.0) for item in history), default=0.0),
        }

    if not args.resume:
        return None

    if last_ckpt_path.exists():
        return {"mode": "full", "path": last_ckpt_path, "history": history}

    if best_model_path.exists():
        start_epoch = len(history)
        best_iou = max((item.get("val_iou", 0.0) for item in history), default=0.0)
        parsed_epoch, parsed_best = _parse_resume_log(args.resume_log, arch, fold)
        start_epoch = max(start_epoch, parsed_epoch)
        best_iou = max(best_iou, parsed_best)
        return {
            "mode": "model",
            "path": best_model_path,
            "history": history,
            "start_epoch": start_epoch,
            "best_iou": best_iou,
        }

    return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            logits = model(pixel_values)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_iou = []
    all_dice = []
    all_pred_labels = []
    all_true_labels = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"]

        with autocast(dtype=torch.bfloat16):
            logits = model(pixel_values)

        pred_prob = torch.sigmoid(logits)
        pred_mask = (pred_prob > threshold).float()

        for i in range(pred_mask.shape[0]):
            pm = pred_mask[i, 0].cpu().numpy().astype(np.uint8)
            tm = mask[i, 0].cpu().numpy().astype(np.uint8)
            metrics = compute_pixel_metrics(pm, tm)
            all_iou.append(metrics["iou"])
            all_dice.append(metrics["dice"])

            # 分类
            pred_label = 1 if pm.sum() / pm.size > 0.001 else 0
            all_pred_labels.append(pred_label)
            all_true_labels.append(labels[i].item())

    cls_correct = sum(p == t for p, t in zip(all_pred_labels, all_true_labels))
    cls_acc = cls_correct / max(len(all_pred_labels), 1)

    return {
        "iou": np.mean(all_iou),
        "dice": np.mean(all_dice),
        "cls_acc": cls_acc,
    }


def train_single_model(
    arch: str,
    fold: int,
    args,
):
    """训练单个模型 (一种架构 + 一折)"""
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"\n{'='*60}")
    print(f"Training: arch={arch}, fold={fold}, gpu={args.gpu}")
    print(f"{'='*60}")

    # 数据集
    full_dataset = ForgerySegDataset(
        args.data_dir, img_size=args.img_size, is_train=True
    )
    val_dataset = ForgerySegDataset(
        args.data_dir, img_size=args.img_size, is_train=False
    )

    # K-Fold 分割
    folds = create_kfold_splits(args.data_dir, n_folds=5, seed=42)
    train_indices, val_indices = folds[fold]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    extra_dataset = None
    if args.include_synth or args.include_real_ext:
        extra_dataset = ForgerySegDataset(
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
        train_dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    save_dir = Path(args.save_dir) / "seg" / f"{arch}_fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    resume_state = _resolve_resume_state(save_dir, arch, fold, args)
    resume_mode = resume_state["mode"] if resume_state is not None else None
    model_resume_start_epoch = (
        resume_state["start_epoch"] if resume_mode == "model" else 0
    )

    # 模型
    use_pretrained = resume_state is None
    if arch == "segformer":
        raw_model = build_segformer(in_channels=7, num_classes=1, pretrained=use_pretrained)
        model = SegModelWrapper(raw_model, model_type="segformer")
    else:
        raw_model = build_smp_model(arch, in_channels=7, num_classes=1, pretrained=use_pretrained)
        model = SegModelWrapper(raw_model, model_type="smp")

    model = model.to(device)

    # 损失、优化器、调度器
    criterion = CombinedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler_epochs = args.epochs
    if resume_mode == "model":
        scheduler_epochs = max(args.epochs - model_resume_start_epoch, 1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=scheduler_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
    )
    scaler = GradScaler()

    start_epoch = 0
    best_iou = 0.0
    patience_counter = 0
    history = []

    if resume_state is not None:
        ckpt = torch.load(resume_state["path"], map_location="cpu", weights_only=False)
        if resume_mode == "full":
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_iou = float(ckpt.get("best_iou", 0.0))
            patience_counter = int(ckpt.get("patience_counter", 0))
            history = ckpt.get("history", resume_state.get("history", []))
            print(f"[resume] full state from {resume_state['path']} | "
                  f"start_epoch={start_epoch} best_iou={best_iou:.4f}")
        else:
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict)
            start_epoch = int(resume_state.get("start_epoch", 0))
            best_iou = float(resume_state.get("best_iou", 0.0))
            history = resume_state.get("history", [])
            print(f"[resume] model weights from {resume_state['path']} | "
                  f"start_epoch={start_epoch} best_iou={best_iou:.4f}")
            print("[resume] optimizer/scheduler reset because only model weights were available")

    if start_epoch >= args.epochs:
        print(f"[resume] {arch}_fold{fold} already reached target epochs ({start_epoch}/{args.epochs}), skip.")
        return best_iou

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, scheduler
        )
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_cls_acc": val_metrics["cls_acc"],
            "time": elapsed,
        }
        history.append(log)

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={train_loss:.4f} | "
              f"IoU={val_metrics['iou']:.4f} | "
              f"Dice={val_metrics['dice']:.4f} | "
              f"Acc={val_metrics['cls_acc']:.4f} | "
              f"{elapsed:.1f}s")

        # 保存最佳模型
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            patience_counter = 0
            print(f"    -> New best IoU: {best_iou:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_iou": best_iou,
                "patience_counter": patience_counter,
                "history": history,
            }, save_dir / "last_checkpoint.pt")
            break

        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_iou": best_iou,
            "patience_counter": patience_counter,
            "history": history,
        }, save_dir / "last_checkpoint.pt")

    # 保存训练历史
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest IoU: {best_iou:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pt'}")
    return best_iou


# ============================================================
# 4. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="分割集成训练")
    parser.add_argument("--arch", type=str, default="all",
                        choices=["segformer", "convnext", "maxvit", "all"])
    parser.add_argument("--fold", type=str, default="all",
                        help="折数 (0-4) 或 'all'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data/raw/train_resume")
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--include_synth", action="store_true")
    parser.add_argument("--synth_dir", type=str, default="data/processed/synth")
    parser.add_argument("--ignore_synth_keep", action="store_true")
    parser.add_argument("--include_real_ext", action="store_true")
    parser.add_argument("--real_ext_dir", type=str, default="data/processed/real_ext")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="优先从 last_checkpoint.pt 恢复；若不存在则从 best_model.pt 继续")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="可选：显式指定恢复的 checkpoint / 权重路径")
    parser.add_argument("--resume_log", type=str, default=None,
                        help="仅有 best_model 时，用日志推断 start_epoch 与 best_iou")
    args = parser.parse_args()

    archs = ["segformer", "convnext", "maxvit"] if args.arch == "all" else [args.arch]
    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]

    results = {}
    for arch in archs:
        for fold in folds:
            key = f"{arch}_fold{fold}"
            best_iou = train_single_model(arch, fold, args)
            results[key] = best_iou

    # 汇总
    print(f"\n{'='*60}")
    print("Training Summary:")
    for key, iou in results.items():
        print(f"  {key}: IoU={iou:.4f}")
    avg_iou = np.mean(list(results.values()))
    print(f"  Average IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()
