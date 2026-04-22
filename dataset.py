"""
数据集模块
- ForgerySegDataset: 多流分割数据集 (RGB + ELA + SRM), 支持增强
- ForgeryClsDataset: 分类数据集 (RGB + ELA)
- VLMSFTDataset: VLM 监督微调对话数据集
"""

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import compute_ela, compute_srm


# ============================================================
# 0. 通用工具
# ============================================================

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_image_files(image_dir: Path):
    if not image_dir.exists():
        return []
    return sorted(
        fname for fname in os.listdir(image_dir)
        if fname.lower().endswith(IMAGE_EXTS)
    )


def _load_keep_stems(synth_dir: Optional[Path], use_keep_list: bool = True):
    if synth_dir is None or not use_keep_list:
        return None
    keep_file = synth_dir / "keep.txt"
    if not keep_file.exists():
        return None
    stems = set()
    for line in keep_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        stems.add(os.path.splitext(os.path.basename(name))[0])
    return stems


def _resolve_caption_dir(base_dir: Path, use_caption_clean: bool = False) -> Path:
    if use_caption_clean:
        clean_dir = base_dir / "Caption_clean"
        if clean_dir.exists():
            return clean_dir
    return base_dir / "Caption"


# ============================================================
# 1. 分割数据集
# ============================================================

def get_seg_train_transforms(img_size: int = 768):
    """分割训练增强 (同时作用于 image 和 mask)"""
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5
        ),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(10.0 / 255, 50.0 / 255)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill="random",
            p=0.2,
        ),
    ])


def get_seg_val_transforms(img_size: int = 768):
    """分割验证/测试变换 (仅 resize)"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
    ])


class ForgerySegDataset(Dataset):
    """
    多流伪造分割数据集。

    输入通道: RGB(3) + ELA(3) + SRM(1) = 7 通道
    输出: 二值 mask (0=真实, 1=伪造)

    目录结构:
        data_dir/
        ├── Black/
        │   ├── Image/  (伪造图片)
        │   ├── Mask/   (伪造 mask, 0/255)
        │   └── Caption/
        └── White/
            ├── Image/  (真实图片)
            └── Caption/
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 768,
        is_train: bool = True,
        ela_quality: int = 90,
        include_base: bool = True,
        synth_dir: Optional[str] = None,
        real_ext_dir: Optional[str] = None,
        use_keep_list: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.ela_quality = ela_quality
        self.include_base = include_base
        self.synth_dir = Path(synth_dir) if synth_dir else None
        self.real_ext_dir = Path(real_ext_dir) if real_ext_dir else None
        self.use_keep_list = use_keep_list

        # 变换 (仅几何和颜色, 不含 normalize/totensor)
        self.transform = (
            get_seg_train_transforms(img_size) if is_train
            else get_seg_val_transforms(img_size)
        )

        # 收集样本: (image_path, mask_path_or_None, label)
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        """收集所有样本"""
        if self.include_base:
            # Black (伪造) 样本
            black_img_dir = self.data_dir / "Black" / "Image"
            black_mask_dir = self.data_dir / "Black" / "Mask"
            for fname in _iter_image_files(black_img_dir):
                stem = os.path.splitext(fname)[0]
                img_path = black_img_dir / fname
                mask_path = black_mask_dir / f"{stem}.png"
                if mask_path.exists():
                    self.samples.append((str(img_path), str(mask_path), 1))

            # White (真实) 样本
            white_img_dir = self.data_dir / "White" / "Image"
            for fname in _iter_image_files(white_img_dir):
                img_path = white_img_dir / fname
                self.samples.append((str(img_path), None, 0))

        if self.is_train and self.real_ext_dir is not None:
            real_img_dir = self.real_ext_dir / "Image"
            for fname in _iter_image_files(real_img_dir):
                img_path = real_img_dir / fname
                self.samples.append((str(img_path), None, 0))

        if self.is_train and self.synth_dir is not None:
            keep_stems = _load_keep_stems(self.synth_dir, use_keep_list=self.use_keep_list)
            synth_img_dir = self.synth_dir / "Image"
            synth_mask_dir = self.synth_dir / "Mask"
            for fname in _iter_image_files(synth_img_dir):
                stem = os.path.splitext(fname)[0]
                if keep_stems is not None and stem not in keep_stems:
                    continue
                mask_path = synth_mask_dir / f"{stem}.png"
                if mask_path.exists():
                    self.samples.append((str(synth_img_dir / fname), str(mask_path), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        # 读取图像
        image = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3) uint8

        # 读取/创建 mask
        if mask_path is not None:
            mask = np.array(Image.open(mask_path).convert("L"))  # (H, W)
            mask = (mask > 127).astype(np.uint8)  # 255 → 1
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 应用几何/颜色增强 (同步作用于 image 和 mask)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]  # (H', W', 3) uint8
        mask = augmented["mask"]    # (H', W')

        # 确保 resize 到目标尺寸 (训练时 RandomResizedCrop 已完成, 以防万一)
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(
                mask, (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # 计算 ELA 特征 (在增强后的图像上计算)
        ela = compute_ela(image, quality=self.ela_quality)  # (H, W, 3) uint8

        # 计算 SRM 特征
        srm = compute_srm(image)  # (H, W, 1) float32

        # 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0     # (H, W, 3)
        ela = ela.astype(np.float32) / 255.0          # (H, W, 3)
        # srm 已经是 [0, 1]

        # 拼接多流输入: RGB(3) + ELA(3) + SRM(1) = 7 通道
        multi_stream = np.concatenate([image, ela, srm], axis=2)  # (H, W, 7)

        # 转为 tensor: (7, H, W)
        multi_stream = torch.from_numpy(multi_stream).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return {
            "pixel_values": multi_stream,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": img_path,
        }

    def get_labels(self):
        return [label for _, _, label in self.samples]


# ============================================================
# 2. 分类数据集
# ============================================================

def get_cls_train_transforms(img_size: int = 512):
    """分类训练增强"""
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(std_range=(10.0 / 255, 50.0 / 255), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


def get_cls_val_transforms(img_size: int = 512):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
    ])


class ForgeryClsDataset(Dataset):
    """
    伪造分类数据集。
    输入: RGB(3) + ELA(3) = 6 通道
    输出: label (0=真实, 1=伪造)
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 512,
        is_train: bool = True,
        ela_quality: int = 90,
        include_base: bool = True,
        synth_dir: Optional[str] = None,
        real_ext_dir: Optional[str] = None,
        use_keep_list: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.ela_quality = ela_quality
        self.include_base = include_base
        self.synth_dir = Path(synth_dir) if synth_dir else None
        self.real_ext_dir = Path(real_ext_dir) if real_ext_dir else None
        self.use_keep_list = use_keep_list
        self.transform = (
            get_cls_train_transforms(img_size) if is_train
            else get_cls_val_transforms(img_size)
        )
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        if self.include_base:
            for category, label in [("Black", 1), ("White", 0)]:
                img_dir = self.data_dir / category / "Image"
                for fname in _iter_image_files(img_dir):
                    self.samples.append((str(img_dir / fname), label))

        if self.is_train and self.real_ext_dir is not None:
            real_img_dir = self.real_ext_dir / "Image"
            for fname in _iter_image_files(real_img_dir):
                self.samples.append((str(real_img_dir / fname), 0))

        if self.is_train and self.synth_dir is not None:
            keep_stems = _load_keep_stems(self.synth_dir, use_keep_list=self.use_keep_list)
            synth_img_dir = self.synth_dir / "Image"
            for fname in _iter_image_files(synth_img_dir):
                stem = os.path.splitext(fname)[0]
                if keep_stems is not None and stem not in keep_stems:
                    continue
                self.samples.append((str(synth_img_dir / fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        augmented = self.transform(image=image)
        image = augmented["image"]

        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size))

        ela = compute_ela(image, quality=self.ela_quality)

        image = image.astype(np.float32) / 255.0
        ela = ela.astype(np.float32) / 255.0

        # RGB(3) + ELA(3) = 6 通道
        multi_stream = np.concatenate([image, ela], axis=2)
        multi_stream = torch.from_numpy(multi_stream).permute(2, 0, 1).float()

        return {
            "pixel_values": multi_stream,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": img_path,
        }

    def get_labels(self):
        return [label for _, label in self.samples]


# ============================================================
# 3. 测试数据集 (推理用, 无标签)
# ============================================================

class TestImageDataset(Dataset):
    """
    测试集数据集, 仅包含图像, 用于推理。
    返回原始图像和多流特征。
    """

    def __init__(
        self,
        image_dir: str,
        img_size: int = 768,
        ela_quality: int = 90,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.ela_quality = ela_quality
        self.transform = get_seg_val_transforms(img_size)

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = str(self.image_dir / fname)

        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # resize
        augmented = self.transform(image=image)
        image_resized = augmented["image"]

        # 计算特征
        ela = compute_ela(image_resized, quality=self.ela_quality)
        srm = compute_srm(image_resized)

        image_norm = image_resized.astype(np.float32) / 255.0
        ela_norm = ela.astype(np.float32) / 255.0

        multi_stream = np.concatenate([image_norm, ela_norm, srm], axis=2)
        multi_stream = torch.from_numpy(multi_stream).permute(2, 0, 1).float()

        return {
            "pixel_values": multi_stream,
            "image_name": fname,
            "orig_size": (orig_h, orig_w),
            "image_path": img_path,
        }


# ============================================================
# 4. K-Fold 分割工具
# ============================================================

def create_kfold_splits(
    data_dir: str,
    n_folds: int = 5,
    seed: int = 42,
):
    """
    创建 K-Fold 分割索引。
    在 Black 和 White 内部分别进行分层分折, 保持类别比例。

    Args:
        data_dir: 数据目录
        n_folds: 折数
        seed: 随机种子

    Returns:
        list of (train_indices, val_indices) for each fold
    """
    data_dir = Path(data_dir)
    rng = random.Random(seed)

    # 收集 Black 和 White 的索引
    black_indices = []
    white_indices = []

    idx = 0
    black_img_dir = data_dir / "Black" / "Image"
    if black_img_dir.exists():
        for fname in sorted(os.listdir(black_img_dir)):
            stem = os.path.splitext(fname)[0]
            mask_path = data_dir / "Black" / "Mask" / f"{stem}.png"
            if mask_path.exists():
                black_indices.append(idx)
                idx += 1

    white_img_dir = data_dir / "White" / "Image"
    if white_img_dir.exists():
        for fname in sorted(os.listdir(white_img_dir)):
            white_indices.append(idx)
            idx += 1

    # 打乱
    rng.shuffle(black_indices)
    rng.shuffle(white_indices)

    # 分折
    folds = []
    for fold_i in range(n_folds):
        black_val = black_indices[
            fold_i * len(black_indices) // n_folds:
            (fold_i + 1) * len(black_indices) // n_folds
        ]
        white_val = white_indices[
            fold_i * len(white_indices) // n_folds:
            (fold_i + 1) * len(white_indices) // n_folds
        ]

        val_set = set(black_val + white_val)
        all_indices = black_indices + white_indices
        train_indices = [i for i in all_indices if i not in val_set]
        val_indices = list(val_set)

        folds.append((train_indices, val_indices))

    return folds


# ============================================================
# 5. VLM SFT 数据集
# ============================================================

class VLMSFTDataset(Dataset):
    """VLM 监督微调对话数据集 (支持结构化证据注入)。

    每个样本: (image, conversation)
    conversation 格式兼容 Qwen3.5/Qwen3-VL 的 chat template。

    inject_evidence=True 时:
        - 训练: 用 GT mask (Black) 或全零 mask (White) 抽取证据
        - 在 user prompt 中嵌入 evidence JSON, 与推理时格式严格一致
        - system prompt 强调 "基于证据论证, 不要凭空生成不在证据中的坐标"
    """

    SYSTEM_PROMPT_PLAIN = (
        "你是专业的图像伪造检测分析专家。请仔细检查图片，判断是否存在数字伪造或篡改痕迹。"
        "如存在，请精确指出篡改区域坐标、篡改内容、视觉异常特征和逻辑矛盾。"
        "如不存在，请从视觉一致性、信息准确性等方面论证真实性。"
    )

    SYSTEM_PROMPT_EVIDENCE = (
        "你是专业的图像伪造鉴定专家。下面会给你一张图片以及一份"
        "由像素级取证模型(分割集成 + ELA + SRM)输出的【结构化证据】。"
        "请严格基于证据中的 bbox、面积占比、异常度比值进行论证，"
        "不要编造证据中未出现的坐标或区域。"
        "输出一段 300-600 字的连续中文鉴定文本，不使用分点、标题或换行。"
    )

    USER_PROMPT_PLAIN = "请分析这张图片是否存在伪造痕迹，给出详细的中文鉴定分析。"

    def __init__(
        self,
        data_dir: str,
        augmented_captions_dir: Optional[str] = None,
        inject_evidence: bool = False,
        use_caption_clean: bool = False,
        real_ext_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.inject_evidence = inject_evidence
        self.use_caption_clean = use_caption_clean
        self.real_ext_dir = Path(real_ext_dir) if real_ext_dir else None
        self.samples = []  # list of (img_path, caption, mask_path|None, gt_label)
        self._collect_samples(augmented_captions_dir)

    def _collect_samples(self, aug_dir: Optional[str] = None):
        for category, has_mask, gt_label in [("Black", True, 1), ("White", False, 0)]:
            img_dir = self.data_dir / category / "Image"
            cap_dir = _resolve_caption_dir(
                self.data_dir / category,
                use_caption_clean=self.use_caption_clean,
            )
            mask_dir = self.data_dir / category / "Mask" if has_mask else None
            if not img_dir.exists() or not cap_dir.exists():
                continue
            for fname in _iter_image_files(img_dir):
                stem = os.path.splitext(fname)[0]
                cap_path = cap_dir / f"{stem}.md"
                if not cap_path.exists():
                    continue
                with open(cap_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                mp = None
                if mask_dir is not None:
                    mp_candidate = mask_dir / f"{stem}.png"
                    if mp_candidate.exists():
                        mp = str(mp_candidate)
                self.samples.append((str(img_dir / fname), caption, mp, gt_label))

        if self.real_ext_dir is not None:
            img_dir = self.real_ext_dir / "Image"
            cap_dir = self.real_ext_dir / "Caption"
            if img_dir.exists() and cap_dir.exists():
                for fname in _iter_image_files(img_dir):
                    stem = os.path.splitext(fname)[0]
                    cap_path = cap_dir / f"{stem}.md"
                    if not cap_path.exists():
                        continue
                    caption = cap_path.read_text(encoding="utf-8").strip()
                    self.samples.append((str(img_dir / fname), caption, None, 0))

        if aug_dir:
            aug_path = Path(aug_dir)
            if aug_path.exists():
                import json as _json
                for fname in sorted(os.listdir(aug_path)):
                    if fname.endswith(".jsonl"):
                        with open(aug_path / fname, "r", encoding="utf-8") as f:
                            for line in f:
                                item = _json.loads(line)
                                self.samples.append((
                                    item["image_path"],
                                    item["caption"],
                                    item.get("mask_path"),
                                    item.get("gt_label", 0),
                                ))

    def __len__(self):
        return len(self.samples)

    def _build_user_prompt(self, img_path: str, mask_path: Optional[str],
                           gt_label: int) -> str:
        if not self.inject_evidence:
            return self.USER_PROMPT_PLAIN
        # 训练: 用 GT 抽取证据 (label 与 GT 一致)
        from evidence import extract_from_gt_mask, evidence_to_prompt_block
        ev = extract_from_gt_mask(img_path, mask_path)
        ev["label"] = gt_label  # 强制使用 GT
        block = evidence_to_prompt_block(ev)
        return (
            "请结合下方【结构化取证证据】对该图像进行伪造鉴定，输出 300-600 字"
            "的连续中文鉴定文本：\n\n"
            f"【证据】\n{block}\n\n"
            "要求：\n"
            "1. 若 label=1，开头使用\"这是一份伪造的[内容简述]\"，并在文中引用证据中的"
            "bbox 坐标 [x1,y1,x2,y2]，分析视觉异常(字体/边缘/纹理/光照/JPEG 伪影)与逻辑矛盾；\n"
            "2. 若 label=0，开头使用\"这是一张真实拍摄的[内容简述]\"，从视觉一致性、"
            "信息准确性、物理合理性论证真实性；\n"
            "3. 严禁输出证据中未提及的坐标；\n"
            "4. 以\"综上所述\"或\"综合分析\"结尾。"
        )

    def __getitem__(self, idx):
        img_path, caption, mask_path, gt_label = self.samples[idx]
        sys_prompt = (self.SYSTEM_PROMPT_EVIDENCE if self.inject_evidence
                      else self.SYSTEM_PROMPT_PLAIN)
        user_prompt = self._build_user_prompt(img_path, mask_path, gt_label)
        return {
            "image_path": img_path,
            "conversations": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": caption},
            ],
        }

    def get_labels(self):
        return [gt_label for _, _, _, gt_label in self.samples]
