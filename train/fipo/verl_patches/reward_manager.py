"""
Custom verl reward manager for TFI forgery-detection FIPO training.

Wires our `train.fipo.reward_fn.compute_reward` into verl's
experimental.reward_loop.RewardManagerBase interface. Loaded via importlib
(NOT register), so this file does not need to be imported by the driver or
pre-registered in any registry. verl reads:

    reward.reward_manager.source=importlib
    reward.reward_manager.name=TFIAuditRewardManager
    reward.reward_manager.module.path=<absolute path to this file>

v1 plumbing
-----------
- Rule-based 5 rewards only (R_format, R_consistency, R_label_gt, R_iou_gt,
  R_phrase_check). Specialist (DINOv3 / SigLIP / MaskCLIP) and caption-rubric
  (GRM / Qwen) hooks are stubbed out — pass them in via `external_scores`
  once those services are deployed.
- GT contract: `reward_model.ground_truth` is a JSON string built by
  prepare_fipo_data with shape:
      {"label": 0|1, "bboxes": [[x1,y1,x2,y2], ...], "phrases": ["...", ...]}
- Heavy work (tokenizer.decode) is dispatched through self.loop.run_in_executor()
  to keep the async event loop responsive.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

# When verl loads this module via importlib.util.spec_from_file_location in a
# Ray worker subprocess, the project root may not be on sys.path, breaking
# `from train.fipo.reward_fn import ...` below. Inject it explicitly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from verl import DataProto  # noqa: E402
from verl.experimental.reward_loop.reward_manager import register  # noqa: E402
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase  # noqa: E402

from train.fipo.reward_fn import (  # noqa: E402
    BREAKDOWN_SCHEMA,
    GroundTruth,
    compute_reward,
)


def _coerce_gt(raw: Any) -> Optional[GroundTruth]:
    """Parse the ground_truth blob written by prepare_fipo_data.

    Accepts dict, JSON string, or None. Falls back to legacy keys when newer
    keys are missing.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    label = int(raw.get("label", 0))
    bboxes = []
    for b in raw.get("bboxes") or []:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                bboxes.append(tuple(float(x) for x in b))
            except (TypeError, ValueError):
                continue
    phrases = [str(p) for p in (raw.get("phrases") or []) if p]
    return GroundTruth(label=label, bboxes=bboxes, phrases=phrases)


@register("tfi_audit_v1")
class TFIAuditRewardManager(RewardManagerBase):
    """Rule-based reward manager for TFI forgery detection (FIPO v1).

    Inherits the verl-latest async interface (`run_single` per rollout).
    External (specialist / GRM) scores are not yet wired — left as zero;
    plug them in by overriding `_external_scores_for(item)` in a subclass.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,            # ignored — we always use reward_fn
        reward_router_address=None,    # accepted for interface compat, unused
        reward_model_tokenizer=None,   # accepted for interface compat, unused
        weight_overrides: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        self.tokenizer = tokenizer
        self.weight_overrides = weight_overrides or {}

    def _external_scores_for(self, item) -> dict:
        """Override in a subclass once specialist / GRM servers are wired.

        Default: empty dict, all external components contribute 0.
        """
        return {}

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "run_single expects exactly one rollout"
        item = data[0]

        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        # Decode in thread pool — tokenizer is sync and can be slow for long seqs.
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        rm_meta = item.non_tensor_batch.get("reward_model", {}) or {}
        gt = _coerce_gt(rm_meta.get("ground_truth"))
        external = self._external_scores_for(item)

        score, breakdown = await self.loop.run_in_executor(
            None,
            lambda: compute_reward(
                response_str,
                gt=gt,
                external_scores=external,
                weights=self.weight_overrides,
                return_breakdown=True,
            ),
        )

        # verl expects every rollout's reward_extra_info to share a fixed key set.
        reward_extra_info = {
            f"reward_v1/{k}": float(breakdown.get(k, 0.0)) for k in BREAKDOWN_SCHEMA
        }
        reward_extra_info["acc"] = float(score)

        return {"reward_score": float(score), "reward_extra_info": reward_extra_info}
