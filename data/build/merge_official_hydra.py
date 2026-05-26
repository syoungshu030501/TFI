#!/usr/bin/env python
"""
Two-stage SFT data prep: merge TFI official corpus with the HydraFake EFG-CN
subset using a fixed mixing ratio.

Stage A (warmup, optional)
    Run SFT for **1 epoch** over `hydra_efg_cn.json` only — teaches the model
    the 6-tag Chinese CoT format on AI-generated faces (broad domain coverage)
    before it sees TFI's smaller, harder cases.

Stage B (main)
    Run SFT for N epochs over the merged set:  TFI ⊕ (ratio * HydraFake-EFG-CN).
    Ratio defaults to 30/70 (HydraFake : TFI) per README §一.7 ADAPT 决策.

Output (matches build_v2_sft.py SFT JSON shape — same loader works):
    /mnt/nfs/young/TFI/data/v2/sft_merged.json   (Stage B input)
    /mnt/nfs/young/TFI/data/v2/sft_merged_meta.json (counts, mix ratio)

Usage:
    python -m data.build.merge_official_hydra \
        --tfi /mnt/nfs/young/TFI/data/v2/sft.json \
        --hydra /mnt/nfs/young/TFI/data/v2/hydra_efg_cn.json \
        --out /mnt/nfs/young/TFI/data/v2/sft_merged.json \
        --hydra_ratio 0.30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import List


def _dedup_by_image(records: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in records:
        imgs = r.get("images") or []
        if not imgs:
            continue
        key = hashlib.md5(imgs[0].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _stratified_sample(records: List[dict], target_n: int, seed: int) -> List[dict]:
    """Sample target_n preserving label balance (label 0 vs 1)."""
    if target_n >= len(records):
        return list(records)
    rng = random.Random(seed)
    by_lbl = {0: [], 1: []}
    for r in records:
        by_lbl[int(r.get("label", 0))].append(r)
    n0, n1 = len(by_lbl[0]), len(by_lbl[1])
    total = n0 + n1
    take_0 = int(round(target_n * n0 / total)) if total else 0
    take_1 = target_n - take_0
    rng.shuffle(by_lbl[0]); rng.shuffle(by_lbl[1])
    return by_lbl[0][:take_0] + by_lbl[1][:take_1]


def merge(tfi_path: Path, hydra_path: Path, hydra_ratio: float, seed: int) -> tuple[List[dict], dict]:
    tfi = json.load(open(tfi_path, "r", encoding="utf-8"))
    hydra = json.load(open(hydra_path, "r", encoding="utf-8")) if hydra_path.exists() else []

    tfi = _dedup_by_image(tfi)
    hydra = _dedup_by_image(hydra)

    # Pick number of HydraFake samples so they make up `hydra_ratio` of the
    # MERGED corpus.  total = T + H,  H/total = r  =>  H = r*T/(1-r).
    if hydra_ratio <= 0.0 or not hydra:
        hydra_kept: List[dict] = []
    elif hydra_ratio >= 1.0:
        hydra_kept = hydra
    else:
        target_h = int(round(hydra_ratio * len(tfi) / (1.0 - hydra_ratio)))
        hydra_kept = _stratified_sample(hydra, target_h, seed)

    merged = tfi + hydra_kept
    random.Random(seed).shuffle(merged)

    meta = {
        "tfi_count": len(tfi),
        "hydra_count_total_input": len(hydra),
        "hydra_count_kept": len(hydra_kept),
        "merged_count": len(merged),
        "hydra_ratio_target": hydra_ratio,
        "hydra_ratio_actual": (
            len(hydra_kept) / len(merged) if merged else 0.0
        ),
        "label_balance": dict(Counter(int(r.get("label", 0)) for r in merged)),
        "source_breakdown": dict(Counter(r.get("source", "?") for r in merged)),
    }
    return merged, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfi", type=Path, required=True,
                    help="TFI SFT JSON (output of build_v2_sft.py without HF mixed-in)")
    ap.add_argument("--hydra", type=Path, required=True,
                    help="HydraFake EFG-CN JSON (output of build_hydra_efg_subset.py)")
    ap.add_argument("--out", type=Path, required=True,
                    help="merged JSON path; meta sidecar written next to it")
    ap.add_argument("--hydra_ratio", type=float, default=0.30,
                    help="target proportion of HydraFake in merged corpus (0–1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    merged, meta = merge(args.tfi, args.hydra, args.hydra_ratio, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=None)
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"merged -> {args.out}  total={len(merged)}")
    print(f"meta   -> {meta_path}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
