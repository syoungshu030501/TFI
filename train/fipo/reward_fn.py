"""
Stage-3 FIPO reward function for TFI forgery detection.

Reward composition (README §三, weights sum to 1.0):

    Rule-based (55%) — runnable from text + GT alone, no external services:
      R_format        0.10  4 required tags present + bbox/answer well-formed
      R_consistency   0.10  label↔location coherence (real⇒no bbox, fake⇒bbox or region)
      R_label_gt      0.15  pred answer == ground-truth label
      R_iou_gt        0.15  IoU(pred bboxes, GT bbox) (only when GT bbox available)
      R_phrase_check  0.05  numbers / region tokens mentioned in conclusion appear in GT

    Specialist (30%, optional) — fed in via `external_scores`, computed batched
    by reward_manager from DINOv3 / SigLIP / MaskCLIP servers:
      R_loc           0.10
      R_cls           0.10
      R_forensic      0.10

    Caption rubric (15%, optional) — also via `external_scores`:
      R_grm           0.10
      R_qwen_periodic 0.05  (only sampled every N steps; 0 otherwise — manager handles this)

The function ALWAYS returns the same key set in `breakdown` (zero-padded for
absent components) so verl/_postprocess can build a non_tensor_batch where
every rollout in a batch shares an identical reward_extra_info schema.

Hard early-exits (return immediately, skip remaining components):
    parse failure (no/duplicate required tags) : -0.5 (R_format), 0 elsewhere
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from train.fipo.schema import (
    REQUIRED_TAGS,
    TFIResponse,
    count_required_tags,
    extract_bboxes_anywhere,
    parse_response,
)

BBox = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Default weights — sum to 1.0 (README §三). Override via `weights=` arg.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "format": 0.10,
    "consistency": 0.10,
    "label_gt": 0.15,
    "iou_gt": 0.15,
    "phrase_check": 0.05,
    "loc": 0.10,
    "cls": 0.10,
    "forensic": 0.10,
    "grm": 0.10,
    "qwen_periodic": 0.05,
}

# Keys that always appear in the breakdown dict (verl batch-schema requirement).
BREAKDOWN_SCHEMA: Tuple[str, ...] = (
    "format", "consistency", "label_gt", "iou_gt", "phrase_check",
    "loc", "cls", "forensic", "grm", "qwen_periodic",
    "well_formed", "pred_fake", "gt_fake", "n_bbox_pred", "iou_max", "total",
)

# Components routed through `external_scores` (provided by reward_manager).
EXTERNAL_KEYS: Tuple[str, ...] = ("loc", "cls", "forensic", "grm", "qwen_periodic")

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1, ax2 = sorted((ax1, ax2))
    ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2))
    by1, by2 = sorted((by1, by2))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _max_iou_to_gt(pred: List[BBox], gt: List[BBox]) -> float:
    if not pred or not gt:
        return 0.0
    return max(_bbox_iou(p, g) for p in pred for g in gt)


# ---------------------------------------------------------------------------
# Phrase extraction (very lightweight — just numeric tokens + bbox digits)
# ---------------------------------------------------------------------------
_NUMBER_TOK_RE = re.compile(r"\d+")


def _phrase_overlap(parsed: TFIResponse, gt_phrases: List[str]) -> float:
    """Fraction of NON-bbox digit tokens / region tokens in the response that
    are also present in any GT phrase. Used to discourage hallucinated numbers
    like fake addresses, prices, dates."""
    if not gt_phrases:
        return 0.0
    # Numeric tokens INSIDE conclusion but outside <bbox>...</bbox>
    conclusion = parsed.conclusion
    if not conclusion:
        return 0.0
    # Strip bbox content (those are layout coords, not "phrases")
    stripped = re.sub(r"<bbox>.*?</bbox>", " ", conclusion, flags=re.DOTALL)
    cand_tokens = set(_NUMBER_TOK_RE.findall(stripped))
    # Also include region descriptions
    for r in parsed.regions:
        cand_tokens |= set(_NUMBER_TOK_RE.findall(r))
    if not cand_tokens:
        return 0.0
    gt_blob = " ".join(gt_phrases)
    gt_tokens = set(_NUMBER_TOK_RE.findall(gt_blob))
    if not gt_tokens:
        return 0.0
    overlap = cand_tokens & gt_tokens
    return float(len(overlap) / max(1, len(cand_tokens)))


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    """GT contract for a single rollout. Built by prepare_fipo_data.py."""

    label: int                      # 0 = real, 1 = fake
    bboxes: List[BBox]              # may be empty (real images have none)
    phrases: List[str]              # GT caption / region descriptions for phrase check

    @property
    def is_fake(self) -> bool:
        return bool(self.label)


def compute_reward(
    output_text: str,
    *,
    gt: Optional[GroundTruth] = None,
    external_scores: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    return_breakdown: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    """Compute scalar reward for one rollout response.

    All component subscores are normalised to [0, 1] before weighting; the
    weighted total is therefore in [0, 1] as well. external_scores are also
    expected in [0, 1] (the reward_manager normalises raw RM logits before
    passing them in).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    ext = external_scores or {}
    bd: Dict[str, float] = {k: 0.0 for k in BREAKDOWN_SCHEMA}

    parsed = parse_response(output_text)
    bd["well_formed"] = float(parsed.is_well_formed)
    bd["pred_fake"] = float(parsed.predicts_fake) if parsed.is_well_formed else 0.0
    bd["n_bbox_pred"] = float(len(parsed.bboxes))

    # ----- R_format ---------------------------------------------------------
    # 4 required tags each present exactly once → 0.75 baseline.
    # +0.25 if answer is one of {real, fake} (well-formed).
    n_ok = count_required_tags(output_text)
    fmt_score = 0.75 * (n_ok / 4.0) + (0.25 if parsed.answer in ("real", "fake") else 0.0)
    bd["format"] = fmt_score

    # If completely unparseable, emit zero on everything except format and stop.
    if not parsed.is_well_formed:
        if return_breakdown:
            total = w["format"] * bd["format"]
            bd["total"] = round(total, 4)
            return total, bd
        return w["format"] * bd["format"]

    # ----- R_label_gt -------------------------------------------------------
    if gt is not None:
        bd["gt_fake"] = float(gt.is_fake)
        bd["label_gt"] = 1.0 if (parsed.predicts_fake == gt.is_fake) else 0.0
    else:
        # No GT — leave label_gt at 0 but don't penalise (0 contribution).
        bd["label_gt"] = 0.0

    # ----- R_consistency ----------------------------------------------------
    # real prediction → should have no bbox/region; fake prediction → should
    # localise (bbox OR region). Both rules: 1.0 if satisfied, 0.0 otherwise.
    has_loc = bool(parsed.bboxes) or bool(parsed.regions)
    if parsed.predicts_fake:
        bd["consistency"] = 1.0 if has_loc else 0.0
    else:
        bd["consistency"] = 1.0 if not has_loc else 0.0

    # ----- R_iou_gt ---------------------------------------------------------
    # Only meaningful when both pred & GT have bboxes.
    if gt is not None and gt.bboxes and parsed.bboxes:
        iou = _max_iou_to_gt(parsed.bboxes, gt.bboxes)
        bd["iou_max"] = round(iou, 4)
        bd["iou_gt"] = iou
    elif gt is not None and not gt.bboxes and not parsed.bboxes:
        # real image, no bbox expected, none predicted → full credit
        bd["iou_gt"] = 1.0
    else:
        bd["iou_gt"] = 0.0

    # ----- R_phrase_check ---------------------------------------------------
    if gt is not None:
        bd["phrase_check"] = _phrase_overlap(parsed, gt.phrases)
    else:
        bd["phrase_check"] = 0.0

    # ----- External scores (specialist + caption rubric) -------------------
    for k in EXTERNAL_KEYS:
        v = ext.get(k, 0.0)
        # Clamp to [0, 1] defensively — external models can return weird values.
        bd[k] = float(max(0.0, min(1.0, v)))

    # ----- Weighted total ---------------------------------------------------
    total = (
        w["format"]        * bd["format"]
        + w["consistency"] * bd["consistency"]
        + w["label_gt"]    * bd["label_gt"]
        + w["iou_gt"]      * bd["iou_gt"]
        + w["phrase_check"]* bd["phrase_check"]
        + w["loc"]         * bd["loc"]
        + w["cls"]         * bd["cls"]
        + w["forensic"]    * bd["forensic"]
        + w["grm"]         * bd["grm"]
        + w["qwen_periodic"] * bd["qwen_periodic"]
    )
    bd["total"] = round(total, 4)

    if return_breakdown:
        return total, bd
    return total


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test: a well-formed fake response with one bbox.
    rsp = (
        "<fast>看起来是 fake</fast>\n"
        "<reasoning>边缘锐利，光照不一致</reasoning>\n"
        "<conclusion>综上所述系篡改：<bbox>10,20,100,120</bbox></conclusion>\n"
        "<answer>fake</answer>"
    )
    gt = GroundTruth(label=1, bboxes=[(15.0, 22.0, 95.0, 110.0)], phrases=["伪造区域"])
    score, bd = compute_reward(rsp, gt=gt, return_breakdown=True)
    print(f"score={score:.4f}")
    for k in BREAKDOWN_SCHEMA:
        print(f"  {k}: {bd[k]}")
