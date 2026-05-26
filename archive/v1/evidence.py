"""结构化证据抽取模块。

从分割概率图 + 原图(RGB/ELA/SRM) 中产出结构化证据 JSON，供:
  1) 轻量校准器作为输入特征
  2) Qwen3.5-9B prompt 注入，避免坐标幻觉

证据字段:
  label              : 0/1, 二值化后判定的初始 label
  total_area_ratio   : 篡改像素 / 总像素
  n_regions          : 连通域数
  regions[]          : 每个连通域的 bbox / 面积 / 中心
  seg_confidence     : mask 内分割平均概率
  seg_max_prob       : 全图分割最大概率
  ela_anomaly        : mask 内 ELA 均值 / mask 外 ELA 均值
  srm_anomaly        : 同上 SRM
  edge_sharpness     : mask 边缘梯度均值
  region_text        : 自然语言区域摘要 (中文)

所有 bbox/坐标均按原图分辨率，左上为原点。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from utils import compute_ela, compute_srm, get_forged_ratio


def _safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a) / (float(b) + eps)


def _region_text(regions: List[Dict], total_ratio: float, h: int, w: int) -> str:
    """生成自然语言区域摘要 (中文)。"""
    if not regions:
        return "未检测到伪造区域"

    parts = [f"共检测到 {len(regions)} 个伪造区域"]
    for i, r in enumerate(regions[:3]):  # 最多描述前 3 个
        x1, y1, x2, y2 = r["bbox"]
        cx, cy = r["centroid"]
        v = "上方" if cy < h / 3 else ("中部" if cy < 2 * h / 3 else "下方")
        hpos = "左侧" if cx < w / 3 else ("中央" if cx < 2 * w / 3 else "右侧")
        parts.append(
            f"区域{i+1}位于图像{v}{hpos}, bbox=[{x1},{y1},{x2},{y2}], 占比{r['area_ratio']*100:.2f}%"
        )
    parts.append(f"伪造区域总占比约 {total_ratio*100:.2f}%")
    return "; ".join(parts)


def extract_regions(binary_mask: np.ndarray, min_area_px: int = 64) -> List[Dict]:
    """从二值 mask 中提取连通域 bbox / 面积 / 中心。

    Args:
        binary_mask: (H, W) uint8 / bool, 1 表示伪造像素
        min_area_px: 小于此像素数的连通域忽略

    Returns:
        [{"bbox": [x1,y1,x2,y2], "area": int, "area_ratio": float, "centroid": [cx, cy]}, ...]
    """
    mask = (binary_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return []

    h, w = mask.shape
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = h * w
    regions = []
    for i in range(1, n_labels):  # 0 是背景
        x, y, ww, hh, area = stats[i]
        if area < min_area_px:
            continue
        cx, cy = centroids[i]
        regions.append({
            "bbox": [int(x), int(y), int(x + ww), int(y + hh)],
            "area": int(area),
            "area_ratio": float(area) / total,
            "centroid": [int(cx), int(cy)],
        })
    # 按面积降序
    regions.sort(key=lambda r: -r["area"])
    return regions


def _edge_sharpness(prob_map: np.ndarray, mask: np.ndarray) -> float:
    """概率图在 mask 边界附近的梯度均值, 反映分割边缘清晰度。"""
    if mask.sum() == 0:
        return 0.0
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150) > 0
    if edges.sum() == 0:
        return 0.0
    gx = cv2.Sobel(prob_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob_map, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return float(grad[edges].mean())


def extract(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
    label_threshold: float = 0.001,
    min_area_px: int = 64,
    ela_cache: Optional[np.ndarray] = None,
    srm_cache: Optional[np.ndarray] = None,
) -> Dict:
    """主入口: 抽取结构化证据。

    Args:
        image_rgb: (H, W, 3) uint8 原图
        binary_mask: (H, W) 二值 mask, 与原图同尺寸
        prob_map: (H, W) 分割模型平均概率图 (与原图同尺寸); 可选
        label_threshold: 伪造像素占比阈值, 用于初判 label
        min_area_px: 连通域最小面积
        ela_cache/srm_cache: 预计算的 ELA/SRM 特征, 不传则现场计算

    Returns:
        evidence dict (JSON-serializable)
    """
    h, w = binary_mask.shape[:2]
    binary_mask = (binary_mask > 0).astype(np.uint8)
    inside = binary_mask.astype(bool)
    outside = ~inside

    total_area_ratio = get_forged_ratio(binary_mask)
    label = 1 if total_area_ratio > label_threshold else 0

    regions = extract_regions(binary_mask, min_area_px=min_area_px)

    if prob_map is not None:
        if prob_map.shape != binary_mask.shape:
            prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_map = prob_map.astype(np.float32)
        seg_max_prob = float(prob_map.max())
        seg_confidence = float(prob_map[inside].mean()) if inside.any() else float(prob_map.mean())
        edge_sharpness = _edge_sharpness(prob_map, binary_mask)
    else:
        seg_max_prob = float(label)
        seg_confidence = float(label)
        edge_sharpness = 0.0

    ela = ela_cache if ela_cache is not None else compute_ela(image_rgb)
    srm = srm_cache if srm_cache is not None else compute_srm(image_rgb)
    ela_gray = ela.mean(axis=2) if ela.ndim == 3 else ela
    srm_gray = srm.squeeze() if srm.ndim == 3 else srm

    if inside.any() and outside.any():
        ela_in = float(ela_gray[inside].mean())
        ela_out = float(ela_gray[outside].mean())
        srm_in = float(srm_gray[inside].mean())
        srm_out = float(srm_gray[outside].mean())
        ela_anomaly = _safe_div(ela_in, ela_out)
        srm_anomaly = _safe_div(srm_in, srm_out)
    else:
        ela_anomaly = 1.0
        srm_anomaly = 1.0

    return {
        "image_size": [int(h), int(w)],
        "label": int(label),
        "total_area_ratio": float(total_area_ratio),
        "n_regions": len(regions),
        "regions": regions,
        "seg_confidence": float(seg_confidence),
        "seg_max_prob": float(seg_max_prob),
        "ela_anomaly": float(np.clip(ela_anomaly, 0.0, 10.0)),
        "srm_anomaly": float(np.clip(srm_anomaly, 0.0, 10.0)),
        "edge_sharpness": float(edge_sharpness),
        "region_text": _region_text(regions, total_area_ratio, h, w),
    }


def evidence_to_features(ev: Dict, cls_score: Optional[float] = None,
                         cls_score_std: Optional[float] = None) -> List[float]:
    """把证据 dict 转为 calibrator 的固定长度特征向量。

    特征顺序 (10 维, 加 cls 是 12 维):
      [seg_confidence, seg_max_prob, total_area_ratio, log(n_regions+1),
       largest_area_ratio, ela_anomaly, srm_anomaly, edge_sharpness,
       (cls_score), (cls_score_std)]
    """
    largest = ev["regions"][0]["area_ratio"] if ev["regions"] else 0.0
    feats = [
        ev["seg_confidence"],
        ev["seg_max_prob"],
        ev["total_area_ratio"],
        float(np.log1p(ev["n_regions"])),
        largest,
        ev["ela_anomaly"],
        ev["srm_anomaly"],
        ev["edge_sharpness"],
    ]
    if cls_score is not None:
        feats.append(float(cls_score))
    if cls_score_std is not None:
        feats.append(float(cls_score_std))
    return feats


FEATURE_NAMES_BASE = [
    "seg_confidence", "seg_max_prob", "total_area_ratio", "log_n_regions",
    "largest_area_ratio", "ela_anomaly", "srm_anomaly", "edge_sharpness",
]
FEATURE_NAMES_WITH_CLS = FEATURE_NAMES_BASE + ["cls_score_mean", "cls_score_std"]


def evidence_to_prompt_block(ev: Dict, max_regions: int = 3) -> str:
    """把证据格式化成 prompt 中嵌入的 JSON 代码块 (人类+模型都易读)。"""
    import json as _json
    compact = {
        "label": ev["label"],
        "image_size": ev["image_size"],
        "n_regions": ev["n_regions"],
        "regions": [
            {"bbox": r["bbox"], "area_ratio": round(r["area_ratio"], 4),
             "centroid": r["centroid"]}
            for r in ev["regions"][:max_regions]
        ],
        "total_area_ratio": round(ev["total_area_ratio"], 4),
        "seg_confidence": round(ev["seg_confidence"], 3),
        "ela_anomaly_ratio": round(ev["ela_anomaly"], 3),
        "srm_anomaly_ratio": round(ev["srm_anomaly"], 3),
    }
    return "```json\n" + _json.dumps(compact, ensure_ascii=False, indent=2) + "\n```"


def extract_from_gt_mask(image_path: str, mask_path: Optional[str]) -> Dict:
    """训练时用 GT mask 抽取证据 (无 prob_map, 直接二值)。

    Args:
        image_path: 图像路径
        mask_path : 二值 mask 路径; 若为 None 或文件不存在, 视为真实图

    Returns:
        evidence dict
    """
    from PIL import Image as _Image
    img = np.array(_Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    if mask_path is None:
        binary = np.zeros((h, w), dtype=np.uint8)
    else:
        try:
            m = np.array(_Image.open(mask_path).convert("L"))
            binary = (m > 127).astype(np.uint8)
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            binary = np.zeros((h, w), dtype=np.uint8)
    return extract(img, binary, prob_map=None)
