"""
TFI 6-tag CoT response schema + parser.

Output format (matches data/build/build_v2_sft.py SFT system prompt verbatim):

    <fast> first-impression judgement </fast>
    <reasoning> forensic reasoning (may nest <planning> ... <reflection>) </reasoning>
    <conclusion>综合结论, with optional <bbox>x1,y1,x2,y2</bbox> and/or <region>desc</region> </conclusion>
    <answer>real|fake</answer>

Required tags: fast, reasoning, conclusion, answer.
Optional inside reasoning: planning, reflection.
Optional inside conclusion: bbox (multiple OK), region (multiple OK).

Parsers below are tolerant of:
  - extra whitespace / newlines around tag boundaries
  - markdown code fences wrapping the whole response
  - bbox int OR float, with comma+/space separators
But strict on:
  - <answer> must literally be "real" or "fake" (lower-case after .strip().lower())
  - all four required tags must appear exactly once
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# System prompt — must match data/build/build_v2_sft.py SYS_PROMPT_ZH exactly,
# otherwise FIPO rollouts and SFT teacher-forcing diverge in tag expectations.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。\n\n"
    "首先用 <fast> </fast> 标签给出第一直觉判断；\n"
    "然后用 <reasoning> </reasoning> 标签给出详细取证推理（高难度样本可在其中包含"
    " <planning> 规划与 <reflection> 自校验）；\n"
    "接着用 <conclusion> </conclusion> 标签给出综合结论，对疑似篡改图必须用 "
    "<bbox>x1,y1,x2,y2</bbox> 或 <region>区域文字描述</region> 标注疑似篡改区域，"
    "其中 bbox 坐标已归一化到 [0,1000]×[0,1000]（左上原点，x1<x2，y1<y2）；\n"
    "最后用 <answer>real|fake</answer> 给出最终判断（仅二选一）。"
)

USER_PROMPT = "<image>请判断该图像的真实性，并按规定标签格式输出分析。"

REQUIRED_TAGS = ("fast", "reasoning", "conclusion", "answer")
OPTIONAL_TAGS = ("planning", "reflection", "bbox", "region")

# Per-tag regexes (DOTALL — content may span newlines).
_TAG_RE = {
    name: re.compile(rf"<{name}\s*>(.*?)</{name}\s*>", re.DOTALL)
    for name in REQUIRED_TAGS + OPTIONAL_TAGS
}

_BBOX_INNER_RE = re.compile(
    r"\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,"
    r"\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*"
)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    return text


@dataclass
class TFIResponse:
    """Parsed TFI response. Fields default to "" / [] when the tag is absent."""

    fast: str = ""
    reasoning: str = ""
    conclusion: str = ""
    answer: str = ""  # "real" | "fake" | "" (parse failure)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_well_formed(self) -> bool:
        """All four required tags present and answer is real/fake."""
        return bool(
            self.fast and self.reasoning and self.conclusion
            and self.answer in ("real", "fake")
        )

    @property
    def predicts_fake(self) -> bool:
        return self.answer == "fake"


def parse_response(text: str) -> TFIResponse:
    """Tolerant parse. Always returns a TFIResponse — caller checks is_well_formed."""
    text = _strip_code_fences(text)
    out = TFIResponse(raw=text)

    for tag in REQUIRED_TAGS:
        m = _TAG_RE[tag].search(text)
        if m:
            content = m.group(1).strip()
            if tag == "answer":
                # Answer must be exactly real|fake; tolerate punctuation.
                a = content.lower().strip(" .。,，:：;；'\"`")
                out.answer = a if a in ("real", "fake") else ""
            else:
                setattr(out, tag, content)

    # bbox / region only meaningful inside <conclusion>, but we accept anywhere
    # in the response (some rollouts put them outside, still scoreable).
    bbox_search_text = out.conclusion if out.conclusion else text
    for m in _TAG_RE["bbox"].finditer(bbox_search_text):
        inner = m.group(1)
        bm = _BBOX_INNER_RE.fullmatch(inner)
        if bm:
            out.bboxes.append(tuple(float(x) for x in bm.groups()))

    region_search_text = out.conclusion if out.conclusion else text
    for m in _TAG_RE["region"].finditer(region_search_text):
        r = m.group(1).strip()
        if r:
            out.regions.append(r)

    return out


def count_required_tags(text: str) -> int:
    """Count how many of the 4 required tags occur exactly once. Used by R_format."""
    text = _strip_code_fences(text)
    n_ok = 0
    for tag in REQUIRED_TAGS:
        matches = _TAG_RE[tag].findall(text)
        if len(matches) == 1:
            n_ok += 1
    return n_ok


def extract_bboxes_anywhere(text: str) -> List[Tuple[float, float, float, float]]:
    """All <bbox> tags anywhere in the response (used when conclusion parse failed)."""
    text = _strip_code_fences(text)
    out: List[Tuple[float, float, float, float]] = []
    for m in _TAG_RE["bbox"].finditer(text):
        bm = _BBOX_INNER_RE.fullmatch(m.group(1))
        if bm:
            out.append(tuple(float(x) for x in bm.groups()))
    return out
