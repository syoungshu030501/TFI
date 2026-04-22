"""用 qwen-vl-max（DashScope）重生成 evidence-caption，替代旧本地 Qwen3.5-9B 链路。

设计点：
  - 远端推理 → 0 显存占用（旧本地 9B 三分片并行频繁 OOM）
  - 视觉理解更强 → 减少 v2 中 49/902 长度越界、提升 strict 通过率
  - 与 caption_local_v2 字段 100% 兼容，VLMSFTDataset 无需改代码
  - resume-safe：基于 (stem, version) 去重
  - 默认 only-missing 模式：只补 caption_local_v2 中缺失或验证失败的样本

跑法：
    export DASHSCOPE_API_KEY=sk-xxx
    # 1) 只补缺失/低质量（默认；最便宜）
    python scripts/data/regen_caption_api.py --mode missing_only
    # 2) 全量重生 640 stem × 2 版本
    python scripts/data/regen_caption_api.py --mode full --n_versions 2

字段对齐：见 data/vlm/README.md。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evidence import extract_from_gt_mask, evidence_to_prompt_block  # noqa: E402

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


def _user_prompt(ev: dict) -> str:
    block = evidence_to_prompt_block(ev)
    if ev["label"] == 1:
        return (
            "请基于下方【结构化取证证据】对该图像进行伪造鉴定, 输出 300-600 字连续中文鉴定文本:\n\n"
            f"{block}\n\n"
            "要求: 开头\"这是一份伪造的[内容简述]\", 文中引用证据中的 bbox 坐标 [x1,y1,x2,y2], "
            "分析视觉异常(字体/边缘/纹理/光照/JPEG 伪影)与逻辑矛盾(数学/日期/品牌/上下文)。"
            "严禁输出证据中未提及的坐标。以\"综上所述\"结尾。"
        )
    return (
        "请基于下方【结构化取证证据】对该图像进行真实性论证, 输出 300-600 字连续中文鉴定文本:\n\n"
        f"{block}\n\n"
        "要求: 开头\"这是一张真实拍摄的[内容简述]\", 从视觉一致性、JPEG 压缩伪影分布均匀性、"
        "物理合理性、信息准确性论证。以\"综合分析\"结尾。"
    )


def _sanitize(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = re.sub(r"\n+", "", text)
    text = re.sub(r"[#*`]+", "", text)
    return text.strip()


def _validate_strict(caption: str, allowed_bboxes: List[List[int]],
                     label: int, min_len=250, max_len=800) -> Optional[str]:
    if THINK_RE.search(caption) or not (min_len <= len(caption) <= max_len):
        return None
    for s in BBOX_RE.findall(caption):
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) != 4 or nums not in allowed_bboxes:
            return None
    if not any(k in caption for k in ("综上所述", "综合分析", "综上")):
        return None
    if label == 1 and not caption.startswith(("这是一份", "这是一张伪造", "这张图", "该图")):
        return None
    if label == 0 and not caption.startswith(("这是一张真实", "这张真实", "该图", "这是一张")):
        return None
    return caption


def _validate_loose(caption: str, allowed_bboxes: List[List[int]],
                    min_len=150, max_len=1200) -> Optional[str]:
    if THINK_RE.search(caption) or not (min_len <= len(caption) <= max_len):
        return None
    for s in BBOX_RE.findall(caption):
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) != 4 or nums not in allowed_bboxes:
            return None
    return caption


def _img_to_b64_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b}"


def _call_api(client, model: str, img_path: Path, ev: dict,
              temperature: float, max_tokens: int) -> str:
    """走 OpenAI 兼容接口（DashScope 兼容模式）。"""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _img_to_b64_url(img_path)}},
                {"type": "text", "text": _user_prompt(ev)},
            ]},
        ],
    )
    return _sanitize(resp.choices[0].message.content or "")


def _collect_targets(data_dir: Path) -> List[Tuple[str, str, str]]:
    """收集 (stem, image_path, mask_path)，路径相对项目根。"""
    img_dir = data_dir / "Black" / "Image"
    mask_dir = data_dir / "Black" / "Mask"
    out = []
    for fname in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(fname)[0]
        mp = mask_dir / f"{stem}.png"
        if mp.exists():
            out.append((stem,
                        str((img_dir / fname).relative_to(ROOT)),
                        str(mp.relative_to(ROOT))))
    return out


def _load_done(out_path: Path) -> Set[Tuple[str, int]]:
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["stem"], d["version"]))
            except Exception:
                pass
    return done


def _load_existing_v2(v2_dir: Path) -> Tuple[Set[Tuple[str, int]], Set[str]]:
    """返回 (已 strict 通过的 (stem,v) 集合, 任何 v 都失败/缺失的 stem 集合)。"""
    strict_done: Set[Tuple[str, int]] = set()
    seen_stems: Dict[str, bool] = {}  # stem -> 任一 strict 通过
    if not v2_dir.exists():
        return strict_done, set()
    for f in v2_dir.glob("*.jsonl"):
        for line in f.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            stem = d.get("stem")
            v = d.get("version", 0)
            if d.get("validation_mode") == "strict":
                strict_done.add((stem, v))
                seen_stems[stem] = True
            else:
                seen_stems.setdefault(stem, False)
    needs_regen = {s for s, ok in seen_stems.items() if not ok}
    return strict_done, needs_regen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw/train_resume",
                   help="基础训练目录（默认 data/raw/train_resume）")
    p.add_argument("--output", default="data/vlm/caption_api_v3/evidence_captions.jsonl")
    p.add_argument("--mode", choices=["missing_only", "full"], default="missing_only",
                   help="missing_only: 仅补 caption_local_v2 中失败/缺失的；full: 全部重生")
    p.add_argument("--legacy_v2", default="data/processed/caption_local_v2",
                   help="参考的旧 v2 目录（仅 missing_only 模式用）")
    p.add_argument("--n_versions", type=int, default=2)
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.8, 1.0])
    p.add_argument("--model", default="qwen-vl-max",
                   help="DashScope 模型名（qwen-vl-max / qwen-vl-plus / qwen-vl-max-2025-xx-xx）")
    p.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4, help="并发请求数")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="仅前 N 个 stem，0=全部")
    args = p.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"[err] 环境变量 {args.api_key_env} 未设置", file=sys.stderr)
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("[err] 缺少 openai 包：pip install openai", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / args.data_dir
    targets = _collect_targets(data_dir)
    print(f"[targets] {len(targets)} Black stems in {args.data_dir}")
    if args.limit > 0:
        targets = targets[:args.limit]

    done = _load_done(out_path)
    print(f"[resume] {len(done)} entries already in {args.output}")

    needs_regen: Optional[Set[str]] = None
    if args.mode == "missing_only":
        legacy_strict, regen_set = _load_existing_v2(ROOT / args.legacy_v2)
        all_stems = {t[0] for t in targets}
        # 老 v2 里完全没有的 stem 也要补
        v2_stems = {s for s, _ in legacy_strict} | regen_set
        regen_set = regen_set | (all_stems - v2_stems)
        needs_regen = regen_set
        print(f"[mode=missing_only] {len(needs_regen)} stems need regen "
              f"(legacy v2 strict={len(legacy_strict)}, missing={len(all_stems - v2_stems)})")

    temps = (args.temperatures + [1.0] * args.n_versions)[:args.n_versions]

    f_out = out_path.open("a", encoding="utf-8")
    n_valid_strict = n_valid_loose = n_failed = 0
    t_start = time.time()

    def work(item) -> Optional[Tuple[bool, str, dict]]:
        stem, ip_rel, mp_rel = item
        if needs_regen is not None and stem not in needs_regen:
            return None
        ip = ROOT / ip_rel
        mp = ROOT / mp_rel
        try:
            ev = extract_from_gt_mask(str(ip), str(mp))
            ev["label"] = 1
            allowed = [r["bbox"] for r in ev["regions"]]
        except Exception as e:
            return False, f"[ev err] {stem}: {e}", {}

        results = []
        for v, temp in enumerate(temps):
            if (stem, v) in done:
                continue
            cap = mode_used = raw = None
            for attempt in range(args.max_retries):
                try:
                    raw = _call_api(client, args.model, ip, ev, temp, args.max_tokens)
                    if (cap := _validate_strict(raw, allowed, label=1)):
                        mode_used = "strict"
                        break
                except Exception as e:
                    last_err = str(e)[:120]
                    time.sleep(1.5 ** attempt)
            if cap is None and raw and (loose := _validate_loose(raw, allowed)):
                cap, mode_used = loose, "loose"
            if cap is None:
                continue
            results.append({
                "image_path": ip_rel, "mask_path": mp_rel,
                "stem": stem, "version": v, "temperature": temp,
                "gt_label": 1, "evidence": ev, "caption": cap,
                "validation_mode": mode_used, "model": args.model,
            })
        return True, stem, results

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, t) for t in targets]
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res is None:
                continue
            ok, stem, payload = res
            if not ok:
                n_failed += 1
                print(f"  {payload}")
                continue
            for item in payload:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                if item["validation_mode"] == "strict":
                    n_valid_strict += 1
                else:
                    n_valid_loose += 1
            f_out.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t_start
                print(f"  [{i+1}/{len(futures)}] elapsed={el:.0f}s  "
                      f"strict={n_valid_strict}  loose={n_valid_loose}  fail={n_failed}")

    f_out.close()
    print(f"\n[done] strict={n_valid_strict}  loose={n_valid_loose}  fail={n_failed}")
    print(f"[done] saved -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
