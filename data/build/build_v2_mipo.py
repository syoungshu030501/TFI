#!/usr/bin/env python
"""
v2 MiPO 偏好数据合成。

输入：v2 SFT json（已生成）
输出：mipo.json，每条 {chosen_messages, rejected_messages}（ms-swift dpo 格式）

简单策略（v0）：
  - 把 ground-truth 6 标签回复当 chosen
  - rejected = 把 answer 翻转 + 删掉 conclusion 里的 bbox/region（"看似合理但与 grounding 矛盾"）

进阶（待 v2 SFT 跑完再升级）：
  - rejected = v1/v2 SFT 的错误预测（hard negative）
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
from pathlib import Path

OUT = Path("/mnt/nfs/young/TFI/data/v2/mipo.json")


def flip_assistant(asst: str, label: int) -> str:
    """把 answer 翻转，删 bbox/region/篡改 描述，生成"看似合理但答案错了"的 rejected。"""
    new_asst = re.sub(r"<bbox>[^<]+</bbox>", "", asst)
    new_asst = re.sub(r"<region>[^<]+</region>", "", new_asst)
    new_asst = re.sub(r"篡改区域[：:][^。]*。?", "", new_asst)
    flipped = "fake" if label == 0 else "real"
    new_asst = re.sub(r"<answer>\s*\w+\s*</answer>", f"<answer>{flipped}</answer>", new_asst)
    if label == 1:
        new_asst = new_asst.replace("伪造", "真实").replace("篡改", "自然").replace("人工智能生成", "真实拍摄")
    else:
        new_asst = new_asst.replace("真实拍摄", "人工智能生成").replace("未发现", "存在")
    return new_asst.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/nfs/young/TFI/data/v2/sft.json")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ratio", type=float, default=0.5, help="从 SFT 抽多少比例做偏好对")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = json.load(open(args.src, "r", encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(src)
    n = max(100, int(len(src) * args.ratio))
    pool = src[:n]

    out = []
    for r in pool:
        chosen = r["messages"]
        rejected = copy.deepcopy(chosen)
        gt_asst = chosen[-1]["content"]
        rejected[-1]["content"] = flip_assistant(gt_asst, r["label"])
        if rejected[-1]["content"].strip() == gt_asst.strip():
            continue
        rec = {
            "images": r["images"],
            "label": r["label"],
            "type": r.get("type", ""),
            "source": r.get("source", "") + "_mipo_flip",
            "messages": chosen,
            "rejected_response": rejected[-1]["content"],
        }
        out.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[mipo] wrote {len(out)} pairs to {args.out}")


if __name__ == "__main__":
    main()
