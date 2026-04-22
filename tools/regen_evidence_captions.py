"""§6 Evidence-aware Caption 重生成 (本地 Qwen3.5-9B base)。

⚠️ DEPRECATED: 本脚本已被 scripts/data/regen_caption_api.py（qwen-vl-max via DashScope）替代。
本地 9B 三分片并行频繁 OOM，且 strict 通过率低（详见 logs/data/aug_validation.md）。
仅作历史参考，不建议再跑。

对每张 Black 图 (训练集) 用 GT mask 抽证据, 让 Qwen3.5-9B 生成 K 个版本的
与 GT bbox 严格对齐的鉴定 caption。生成后做白名单校验。

输出:
  data/processed/caption_local_v2/evidence_captions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence import (  # noqa: E402
    extract_from_gt_mask,
    evidence_to_prompt_block,
)

BBOX_RE = re.compile(r"\[\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*\]")
THINK_RE = re.compile(r"</?think>", re.IGNORECASE)


SYSTEM_PROMPT = (
    "你是专业的图像伪造鉴定专家。下面会给你一张图片以及一份"
    "由像素级取证模型(分割集成 + ELA + SRM)输出的【结构化证据】。"
    "请严格基于证据中的 bbox、面积占比、异常度比值进行论证，"
    "不要编造证据中未出现的坐标或区域。"
    "输出一段 300-600 字的连续中文鉴定文本，不使用分点、标题、换行、markdown。"
    "严禁使用 <think> 思维链标签。"
)


def sanitize(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = re.sub(r"\n+", "", text)
    text = re.sub(r"[#*`]+", "", text)
    return text.strip()


def validate(caption: str, allowed_bboxes: List[List[int]],
             label: int, min_len=250, max_len=800) -> Optional[str]:
    """若不通过返回 None, 通过返回原文。"""
    if THINK_RE.search(caption):
        return None
    if not (min_len <= len(caption) <= max_len):
        return None
    # caption 内出现的所有 bbox 必须是 GT 允许集的子集
    found = BBOX_RE.findall(caption)
    for s in found:
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) != 4:
            return None
        if nums not in allowed_bboxes:
            return None
    # 必须有 "综上所述" 或 "综合分析"
    if not any(k in caption for k in ("综上所述", "综合分析", "综上")):
        return None
    # 开头格式大致正确
    if label == 1 and not caption.startswith(("这是一份", "这是一张伪造",
                                              "这张图", "该图")):
        return None
    if label == 0 and not caption.startswith(("这是一张真实", "这张真实",
                                              "该图", "这是一张")):
        return None
    return caption


def validate_loose(caption: str, allowed_bboxes: List[List[int]],
                   min_len=150, max_len=1200) -> Optional[str]:
    """宽松回退: 只保证无 <think>、长度合理、无越界 bbox 幻觉。"""
    if THINK_RE.search(caption):
        return None
    if not (min_len <= len(caption) <= max_len):
        return None
    found = BBOX_RE.findall(caption)
    for s in found:
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) != 4 or nums not in allowed_bboxes:
            return None
    return caption


def load_model(
    model_path: str,
    dtype=torch.bfloat16,
    device_map: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = device_map or "auto"
    elif load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = device_map or "auto"
    else:
        load_kwargs["dtype"] = dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    return model, processor


def generate_caption(model, processor, img_path: str, ev: dict,
                      temperature: float, max_new_tokens: int = 1024) -> str:
    from qwen_vl_utils import process_vision_info
    block = evidence_to_prompt_block(ev)
    if ev["label"] == 1:
        user = (
            "请基于下方【结构化取证证据】对该图像进行伪造鉴定, 输出 300-600 字"
            "连续中文鉴定文本:\n\n"
            f"{block}\n\n"
            "要求: 开头\"这是一份伪造的[内容简述]\", 文中引用证据中的 bbox 坐标 "
            "[x1,y1,x2,y2], 分析视觉异常(字体/边缘/纹理/光照/JPEG 伪影)与"
            "逻辑矛盾(数学/日期/品牌/上下文)。严禁输出证据中未提及的坐标。"
            "以\"综上所述\"结尾。"
        )
    else:
        user = (
            "请基于下方【结构化取证证据】对该图像进行真实性论证, 输出 300-600 字"
            "连续中文鉴定文本:\n\n"
            f"{block}\n\n"
            "要求: 开头\"这是一张真实拍摄的[内容简述]\", 从视觉一致性、"
            "JPEG 压缩伪影分布均匀性、物理合理性、信息准确性论证。"
            "以\"综合分析\"结尾。"
        )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": user},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=0.9, do_sample=True,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(gen, skip_special_tokens=True)
    if "</think>" in raw:
        raw = raw[raw.index("</think>") + len("</think>"):]
    return sanitize(raw)


def collect_stems(root: Path, split: str):
    """收集某个 split 下所有 Black 的 (stem, img_path, mask_path)。"""
    img_dir = root / split / "Black" / "Image"
    mask_dir = root / split / "Black" / "Mask"
    out = []
    for fname in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(fname)[0]
        mp = mask_dir / f"{stem}.png"
        if mp.exists():
            out.append((stem, str(img_dir / fname), str(mp)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="models/Qwen3.5-9B")
    p.add_argument("--split", default="data/raw/train_resume",
                   help="用哪个 split 的 Black 做 regeneration（相对 TFI 根）")
    p.add_argument("--output",
                   default="data/processed/caption_local_v2/evidence_captions.jsonl")
    p.add_argument("--n_versions", type=int, default=2)
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.8, 1.0])
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device_map", default=None, help="'auto' 用 accelerate 多卡")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--only", default=None, help="可选 stems 白名单文件 (每行 split\\tstem)")
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="仅前 N 张, 0=全部")
    p.add_argument("--num_shards", type=int, default=1, help="总分片数")
    p.add_argument("--shard_index", type=int, default=0, help="当前分片编号 [0, num_shards)")
    args = p.parse_args()

    print("⚠️  DEPRECATED: prefer scripts/data/regen_caption_api.py (qwen-vl-max API).",
          file=sys.stderr)
    if args.device_map != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    root = Path(__file__).resolve().parent.parent
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stems = collect_stems(root, args.split)
    print(f"total Black stems in {args.split}: {len(stems)}")

    if args.only and Path(args.only).exists():
        only_set = set()
        for line in open(args.only, "r", encoding="utf-8"):
            parts = line.strip().split("\t")
            if parts:
                only_set.add(parts[-1])
        stems = [s for s in stems if s[0] in only_set]
        print(f"restricted to --only list: {len(stems)}")
    if args.limit > 0:
        stems = stems[:args.limit]
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError("shard_index 必须满足 0 <= shard_index < num_shards")
        stems = [
            item for idx, item in enumerate(stems)
            if idx % args.num_shards == args.shard_index
        ]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(stems)} stems")

    # resume
    done = set()
    if out_path.exists():
        for line in open(out_path, "r", encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["stem"], d["version"]))
            except Exception:
                pass
        print(f"resume: {len(done)} entries already written")

    dtype = getattr(torch, args.dtype)
    effective_device_map = args.device_map
    if (args.load_in_4bit or args.load_in_8bit) and effective_device_map is None:
        effective_device_map = "auto"
    model, processor = load_model(
        args.model_path,
        dtype=dtype,
        device_map=effective_device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    if effective_device_map is None and not (args.load_in_4bit or args.load_in_8bit):
        model = model.to("cuda:0")
    model = model.eval()

    f_out = open(out_path, "a", encoding="utf-8")
    t_start = time.time()
    total_generated, total_valid = 0, 0

    temps = args.temperatures[:args.n_versions] if len(args.temperatures) >= args.n_versions \
        else (args.temperatures + [1.0] * args.n_versions)[:args.n_versions]

    for i, (stem, ip, mp) in enumerate(tqdm(stems, ncols=80)):
        try:
            ev = extract_from_gt_mask(ip, mp)
            ev["label"] = 1
            allowed = [r["bbox"] for r in ev["regions"]]
        except Exception as e:
            print(f"  [ev err] {stem}: {e}"); continue

        for v, temp in enumerate(temps):
            if (stem, v) in done:
                continue
            cap = None
            validation_mode = None
            raw = None
            for attempt in range(args.max_retries):
                try:
                    raw = generate_caption(model, processor, ip, ev, temp,
                                            max_new_tokens=args.max_new_tokens)
                    total_generated += 1
                    ok = validate(raw, allowed, label=1)
                    if ok:
                        cap = ok
                        validation_mode = "strict"
                        break
                except Exception as e:
                    print(f"  [gen err] {stem} v{v}: {e}")
            if cap is None:
                loose = validate_loose(raw, allowed) if raw else None
                if loose is not None:
                    cap = loose
                    validation_mode = "loose"
            if cap is None:
                continue
            total_valid += 1
            item = {
                "image_path": os.path.relpath(ip, root),
                "mask_path": os.path.relpath(mp, root),
                "stem": stem, "version": v, "temperature": temp,
                "gt_label": 1, "evidence": ev, "caption": cap,
                "validation_mode": validation_mode,
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(stems)}] elapsed={elapsed:.0f}s  valid={total_valid}/{total_generated}")

    f_out.close()
    print(f"Done. valid {total_valid} / generated {total_generated}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
