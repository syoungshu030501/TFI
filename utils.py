"""
工具函数模块
- ELA / SRM 图像取证特征提取
- COCO RLE 编解码
- 评估指标 (IoU, Dice, F1, Pixel Accuracy)
- Mask 后处理 (形态学, 连通域过滤)
"""

import io
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# ============================================================
# 1. 图像取证特征提取
# ============================================================

def compute_ela(image_np: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    Error Level Analysis (ELA).
    将图像以指定 JPEG 质量重新压缩, 计算差值, 暴露压缩不一致区域。

    Args:
        image_np: RGB 图像, shape (H, W, 3), uint8
        quality: JPEG 重压缩质量 (越低差异越明显)

    Returns:
        ELA 图像, shape (H, W, 3), uint8, 差值被放大以增强可视性
    """
    # 编码为 JPEG 内存缓冲
    pil_img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # 解码重压缩图像
    recompressed = np.array(Image.open(buffer))

    # 如果尺寸因 JPEG 编码有微调, 裁剪到相同大小
    h = min(image_np.shape[0], recompressed.shape[0])
    w = min(image_np.shape[1], recompressed.shape[1])
    image_np = image_np[:h, :w]
    recompressed = recompressed[:h, :w]

    # 计算差值并放大
    ela = np.abs(image_np.astype(np.float32) - recompressed.astype(np.float32))
    # 放大系数: 使差异更明显 (标准 ELA 放大倍率)
    scale = 255.0 / (ela.max() + 1e-8)
    ela = np.clip(ela * scale, 0, 255).astype(np.uint8)

    return ela


# 30 个 SRM 高通滤波核 (Spatial Rich Model 的核心子集)
# 这里使用最有效的 3 类: 1st order, 2nd order, 3rd order edge filters
SRM_KERNELS = []

def _build_srm_kernels():
    """构建 SRM 高通滤波核集合 (取最有效的几种)"""
    global SRM_KERNELS
    if len(SRM_KERNELS) > 0:
        return

    # 1st order: horizontal, vertical, diagonal edges
    k1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
    k3 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

    # 2nd order: Laplacian variants
    k4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    k5 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # 3rd order: high-frequency residual
    k6 = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -3, 3, -1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=np.float32)

    # SQUARE 3x3 and 5x5
    k7 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)

    # Edge-aware (Prewitt-like for noise)
    k8 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0

    SRM_KERNELS.extend([k1, k2, k3, k4, k5, k6, k7, k8])


def compute_srm(image_np: np.ndarray) -> np.ndarray:
    """
    Spatial Rich Model (SRM) 噪声残差特征。
    对图像施加多种高通滤波核, 取各通道最强响应的均值。

    Args:
        image_np: RGB 图像, shape (H, W, 3), uint8

    Returns:
        SRM 特征图, shape (H, W, 1), float32, 范围 [0, 1]
    """
    _build_srm_kernels()

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    responses = []
    for kernel in SRM_KERNELS:
        resp = cv2.filter2D(gray, -1, kernel)
        responses.append(np.abs(resp))

    # 取所有核响应的最大值 (per-pixel), 再归一化
    stacked = np.stack(responses, axis=0)  # (N_kernels, H, W)
    max_response = np.max(stacked, axis=0)  # (H, W)

    # 归一化到 [0, 1]
    vmax = max_response.max()
    if vmax > 1e-8:
        max_response = max_response / vmax

    return max_response[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)


# ============================================================
# 2. COCO RLE 编解码
# ============================================================

def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """
    将二值 mask 编码为 COCO RLE 格式。

    Args:
        binary_mask: shape (H, W), uint8, 值为 0 或 1

    Returns:
        dict: {"size": [H, W], "counts": str}
    """
    # pycocotools 要求 Fortran order (column-major)
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    """
    将 COCO RLE 解码为二值 mask。

    Args:
        rle: {"size": [H, W], "counts": str}

    Returns:
        binary_mask: shape (H, W), uint8
    """
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(np.uint8)


def create_zero_rle(height: int, width: int) -> dict:
    """
    创建全零 mask 的 RLE (用于 label=0 的真实图像)。

    Args:
        height, width: 图像尺寸

    Returns:
        dict: {"size": [H, W], "counts": str}
    """
    zero_mask = np.zeros((height, width), dtype=np.uint8, order="F")
    rle = mask_utils.encode(zero_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ============================================================
# 3. 评估指标
# ============================================================

def compute_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """
    计算 IoU (Intersection over Union)。

    Args:
        pred: 预测 mask, shape (H, W), 0/1
        target: 真实 mask, shape (H, W), 0/1

    Returns:
        IoU 值
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def compute_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """计算 Dice 系数"""
    intersection = (pred * target).sum()
    return float((2 * intersection + eps) / (pred.sum() + target.sum() + eps))


def compute_f1(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    """
    计算分类 F1, Precision, Recall, Accuracy。

    Args:
        pred_labels: 预测标签数组
        true_labels: 真实标签数组

    Returns:
        dict: {"f1", "precision", "recall", "accuracy"}
    """
    pred_labels = np.asarray(pred_labels).astype(int)
    true_labels = np.asarray(true_labels).astype(int)

    tp = ((pred_labels == 1) & (true_labels == 1)).sum()
    fp = ((pred_labels == 1) & (true_labels == 0)).sum()
    fn = ((pred_labels == 0) & (true_labels == 1)).sum()
    tn = ((pred_labels == 0) & (true_labels == 0)).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def compute_pixel_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict:
    """
    计算像素级评估指标汇总。

    Args:
        pred_mask: 预测 mask, (H, W), 0/1
        target_mask: 真实 mask, (H, W), 0/1

    Returns:
        dict: {"iou", "dice", "pixel_acc"}
    """
    correct = (pred_mask == target_mask).sum()
    total = pred_mask.size
    pixel_acc = float(correct / (total + 1e-7))

    return {
        "iou": compute_iou(pred_mask, target_mask),
        "dice": compute_dice(pred_mask, target_mask),
        "pixel_acc": pixel_acc,
    }


# ============================================================
# 4. Mask 后处理
# ============================================================

def postprocess_mask(
    binary_mask: np.ndarray,
    morph_kernel_size: int = 5,
    min_area: int = 100,
) -> np.ndarray:
    """
    对预测 mask 进行后处理:
    1. 形态学开运算 (去小噪点)
    2. 形态学闭运算 (填小孔)
    3. 连通域过滤 (面积 < min_area 的去除)

    Args:
        binary_mask: shape (H, W), uint8, 值 0/1
        morph_kernel_size: 形态学核大小
        min_area: 最小连通域面积阈值

    Returns:
        处理后的 mask, shape (H, W), uint8, 值 0/1
    """
    mask = binary_mask.astype(np.uint8)

    # 形态学开运算 (先腐蚀后膨胀, 去除小噪点)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 形态学闭运算 (先膨胀后腐蚀, 填充小孔)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 连通域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):  # 跳过背景 (label=0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 1

    return filtered


# ============================================================
# 5. 辅助函数
# ============================================================

def get_forged_ratio(mask: np.ndarray) -> float:
    """计算伪造像素占比"""
    return float(mask.sum() / (mask.size + 1e-7))


def mask_to_label(mask: np.ndarray, threshold: float = 0.001) -> int:
    """根据 mask 判定 label: 伪造像素占比超过阈值则为伪造"""
    return 1 if get_forged_ratio(mask) > threshold else 0


def describe_mask_region(mask: np.ndarray) -> str:
    """
    根据 mask 生成区域描述 (用于注入 VLM prompt)。
    返回如 "左上方" "中部偏右" 等位置描述。
    """
    if mask.sum() == 0:
        return ""

    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    cy, cx = ys.mean(), xs.mean()

    # 判断纵向位置
    if cy < h / 3:
        v_pos = "上方"
    elif cy < 2 * h / 3:
        v_pos = "中部"
    else:
        v_pos = "下方"

    # 判断横向位置
    if cx < w / 3:
        h_pos = "左侧"
    elif cx < 2 * w / 3:
        h_pos = "中央"
    else:
        h_pos = "右侧"

    ratio = get_forged_ratio(mask) * 100
    return f"图像{v_pos}{h_pos}区域，覆盖约{ratio:.1f}%的画面"
