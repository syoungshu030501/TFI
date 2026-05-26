#!/usr/bin/env python
"""
v2 P-GRPO rollout 数据准备。

输入：v2 SFT json
输出：pgrpo.json，每条 {prompt, label, type, mask_path}（assistant 留空，等 actor rollout）

P-GRPO 训练时 swift 会按 num_generations=4 让 actor 每个 prompt 出 4 个候选，再用
patternacc + unifiedprm + multi_reason_format reward 对候选打分。
"""
from __future__ import annotations
import argparse
import copy
import json
import random
from pathlib import Path

OUT = Path("/mnt/nfs/young/TFI/data/v2/pgrpo.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/nfs/young/TFI/data/v2/sft.json")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = json.load(open(args.src, "r", encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(src)
    if args.limit:
        src = src[: args.limit]

    out = []
    for r in src:
        msgs = copy.deepcopy(r["messages"])
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        rec = {
            "images": r["images"],
            "label": r["label"],
            "type": r.get("type", ""),
            "source": r.get("source", "") + "_pgrpo",
            "mask_path": r.get("mask_path"),
            "messages": msgs,
        }
        out.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[pgrpo] wrote {len(out)} prompts to {args.out}")


if __name__ == "__main__":
    main()
