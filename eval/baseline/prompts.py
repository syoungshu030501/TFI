"""
Prompt templates for prompt-only baseline (M(-1)).

3 variants:
  - zs:  zero-shot, strict JSON schema
  - fs:  zero-shot + 8-shot in-context (4 forged + 4 real, sampled from train)
  - cot: zero-shot + chain-of-thought (think then JSON)

All variants enforce identical output schema for fair comparison and
easy parsing back to (label, location-RLE, explanation).
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

SCHEMA_HINT = """\
你必须严格按以下 JSON 格式输出，不要输出任何 JSON 以外的内容（包括"```json"标签都不要）：

{
  "label": 0 或 1,                              // 0=真实, 1=伪造
  "location": [                                  // 如果 label=0 必须为空数组 []
    {"bbox": [x1, y1, x2, y2], "type": "..."}    // 篡改区域，bbox 为整数像素坐标
  ],
  "explanation": "300-600 字的中文鉴定结论"     // 一段连续文本，不分段
}

约束:
- label=0 ⇒ location 必须是 []
- label=1 ⇒ location 至少 1 个 bbox
- bbox 中 0 ≤ x1 < x2 ≤ width, 0 ≤ y1 < y2 ≤ height
- explanation 必须 300-600 字、中文、单段、不含 markdown
- explanation 中提到的所有 bbox 坐标必须出现在 location 中
"""

SYSTEM_PROMPT = """\
你是专业的图像伪造鉴定专家。给你一张图片，请判断它是 真实拍摄/扫描 (label=0) 还是 数字伪造 (label=1)，
并对伪造区域定位、给出中文鉴定结论。

伪造类型可能包括：数字篡改、文字替换 (text-replace)、图像拼接 (splicing)、复制粘贴 (copy-move)、
图像修复 (inpainting)、AI 生成 (AIGC) 局部或整图。

注意：
- 真实但低质量图像（热敏小票、扫描件、低光照、压缩失真）不算伪造，请勿误报。
- 仅当确实存在篡改痕迹时才标 label=1。
- 不要编造 bbox 坐标，必须基于图像内能定位的具体区域。
"""

INSTRUCTION_FORGED = """\
请鉴定下图是否伪造。如认为是伪造图，开头使用"这是一份伪造的[内容简述]"，
分析视觉异常（字体差异、边缘不自然、纹理断裂、光照不一致、JPEG 压缩伪影、像素噪声不匹配）
与逻辑矛盾（数学计算、日期、品牌、上下文），并以"综上所述，该图像系[伪造方式]，不具备真实性与可信度"结尾。
如认为是真实图，按真实图模板输出。
"""

INSTRUCTION_TRUE = """\
请鉴定下图是否伪造。如认为是真实图，开头使用"这是一张真实拍摄的[内容简述]，未发现数字伪造或后期篡改的痕迹"，
从视觉一致性（字体统一、边缘过渡自然、纹理连续、光照均匀、噪点分布一致）、
JPEG 压缩伪影分布均匀性、物理合理性、信息准确性进行论证，并以"综合分析，该图像真实记录了[具体场景描述]"结尾。
如认为是伪造图，按伪造图模板输出。
"""

UNIFIED_INSTRUCTION = """\
请仔细观察图片，判断是否伪造，并按上面的 JSON schema 输出鉴定结果。
- explanation 字段中：若 label=1，开头应为"这是一份伪造的[内容简述]"，结尾应为"综上所述，该图像系[伪造方式]，不具备真实性与可信度"。
  若 label=0，开头应为"这是一张真实拍摄的[内容简述]，未发现数字伪造或后期篡改的痕迹"，结尾应为"综合分析，该图像真实记录了[具体场景描述]"。

【极其重要】你的回答必须直接以 `{` 字符开始，最后以 `}` 字符结束。
不要写任何前言、不要写"好的"、"我来分析"，不要 markdown，不要 ```json 围栏，不要解释。
直接吐 JSON。立刻。
"""

COT_INSTRUCTION = """\
请按以下流程思考后再输出最终 JSON：
1. 先在 <think> ... </think> 标签内逐步思考：观察图像是什么类型（收据/截图/产品/证件/...）、
   寻找疑似篡改区域（字体一致性、边缘连续性、噪声纹理、光照、逻辑矛盾），列出每个候选 bbox 及理由。
2. 综合判断 label 与最终 location。
3. 在 </think> 之后立即输出符合 schema 的 JSON，且 JSON 必须以 `{` 开始 `}` 结束。

【严格输出格式】（除以下两段外不要有任何其他文本）:
<think>
[你的逐步思考，不限长度]
</think>
{"label": 0或1, "location": [...], "explanation": "300-600字鉴定结论"}
"""


# ============================================================
# few-shot example pool: 从 train/{Black,White} 抽
# ============================================================
def build_fewshot_examples(
    train_dir: str,
    n_forged: int = 4,
    n_real: int = 4,
    max_caption_chars: int = 300,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample (image_path, target_json) pairs as in-context demonstrations.

    Avoid extremely long captions and keep mixed forgery types if possible.
    """
    rng = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    forged_caption_dir = Path(train_dir) / "Black" / "Caption_clean"
    forged_image_dir = Path(train_dir) / "Black" / "Image"
    forged_mask_dir = Path(train_dir) / "Black" / "Mask"
    real_caption_dir = Path(train_dir) / "White" / "Caption"
    real_image_dir = Path(train_dir) / "White" / "Image"

    if forged_caption_dir.exists():
        forged_ids = sorted([p.stem for p in forged_caption_dir.glob("*.md")])
        rng.shuffle(forged_ids)
        for stem in forged_ids:
            if len([e for e in examples if e["label"] == 1]) >= n_forged:
                break
            img_path = forged_image_dir / f"{stem}.jpg"
            cap_path = forged_caption_dir / f"{stem}.md"
            if not img_path.exists() or not cap_path.exists():
                continue
            cap = cap_path.read_text(encoding="utf-8").strip()
            if len(cap) > 800:
                continue
            bboxes = _parse_bboxes_from_caption(cap)
            if not bboxes:
                continue
            cap_short = cap[:max_caption_chars]
            tgt = {"label": 1,
                   "location": [{"bbox": b, "type": "篡改"} for b in bboxes[:5]],
                   "explanation": cap_short}
            examples.append({"image": str(img_path), "label": 1, "target": tgt})

    if real_caption_dir.exists():
        real_ids = sorted([p.stem for p in real_caption_dir.glob("*.md")])
        rng.shuffle(real_ids)
        for stem in real_ids:
            if len([e for e in examples if e["label"] == 0]) >= n_real:
                break
            img_path = real_image_dir / f"{stem}.jpg"
            cap_path = real_caption_dir / f"{stem}.md"
            if not img_path.exists() or not cap_path.exists():
                continue
            cap = cap_path.read_text(encoding="utf-8").strip()
            if len(cap) > 800:
                continue
            cap_short = cap[:max_caption_chars]
            tgt = {"label": 0, "location": [], "explanation": cap_short}
            examples.append({"image": str(img_path), "label": 0, "target": tgt})

    rng.shuffle(examples)
    return examples


def _parse_bboxes_from_caption(text: str) -> List[List[int]]:
    import re
    pat = re.compile(r"\[(\d{1,5})\s*,\s*(\d{1,5})\s*,\s*(\d{1,5})\s*,\s*(\d{1,5})\]")
    out = []
    for m in pat.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


# ============================================================
# Build chat messages for one sample
# ============================================================
def build_messages(
    mode: str,
    image_path: str,
    fewshot_examples: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Return messages list ready for processor.apply_chat_template."""
    assert mode in ("zs", "fs", "cot"), f"unknown mode: {mode}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT + "\n\n" + SCHEMA_HINT}]}
    ]

    if mode == "fs" and fewshot_examples:
        for ex in fewshot_examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(ex['image'])}"},
                    {"type": "text", "text": UNIFIED_INSTRUCTION},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text",
                     "text": json.dumps(ex["target"], ensure_ascii=False)},
                ],
            })

    final_instruction = COT_INSTRUCTION if mode == "cot" else UNIFIED_INSTRUCTION
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": final_instruction},
        ],
    })
    return messages
