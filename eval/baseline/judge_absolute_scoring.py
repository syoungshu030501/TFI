#!/usr/bin/env python
"""
Absolute scoring of multiple prediction sets using DeepSeek-R1-Distill-Llama-70B
as a third-party judge (text-only, no image).

For each (image, prediction) pair, the judge outputs 4 integer scores (1-10):
  - accuracy:     label / bbox / factual alignment with GT
  - evidence:     references concrete bbox + verifiable visual/logical clues
  - completeness: covers all key forgery points (or fully justifies real-image label)
  - language:     fluency / professionalism / 300-600 char length adequacy

Judge runs on vllm with tensor_parallel_size=4 (GPU 4-7), reasoning is stripped
before parsing the final JSON.

Usage:
  python tools/baseline/judge_absolute_scoring.py \\
      --pred_csvs sft=/home/young/TFI/submit_val.csv \\
                  zs=tools/baseline/results/zs/predictions.csv \\
                  fs=tools/baseline/results/fs/predictions.csv \\
                  cot=tools/baseline/results/cot/predictions.csv \\
      --judge_model /mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b \\
      --gpus 4,5,6,7 \\
      --out_dir tools/baseline/results/judge
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

THIS = Path(__file__).resolve()
PROJ = THIS.parents[2]
sys.path.insert(0, str(PROJ))

VAL_DIR = PROJ / "data" / "raw" / "val"


# ============================================================
# Load predictions + GT captions
# ============================================================
def load_gt() -> Dict[str, Dict[str, Any]]:
    """{ image_name: {gt_label, gt_caption} }"""
    gt: Dict[str, Dict[str, Any]] = {}
    for label_dir, label in [("Black", 1), ("White", 0)]:
        img_dir = VAL_DIR / label_dir / "Image"
        cap_dir_clean = VAL_DIR / label_dir / "Caption_clean"
        cap_dir = VAL_DIR / label_dir / "Caption"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            stem = p.stem
            cap_path = (cap_dir_clean / f"{stem}.md") if (cap_dir_clean / f"{stem}.md").exists() \
                       else (cap_dir / f"{stem}.md")
            cap_text = cap_path.read_text(encoding="utf-8").strip() if cap_path.exists() else ""
            gt[p.name] = {"gt_label": label, "gt_caption": cap_text[:1500]}
    return gt


def load_predictions(spec: List[str]) -> Dict[str, pd.DataFrame]:
    """spec is ['name=path', ...]"""
    out: Dict[str, pd.DataFrame] = {}
    for s in spec:
        if "=" not in s:
            raise ValueError(f"--pred_csvs item must be name=path, got: {s}")
        name, path = s.split("=", 1)
        df = pd.read_csv(path)
        for col in ["image_name", "label", "explanation"]:
            assert col in df.columns, f"{path} missing column {col}"
        out[name] = df.set_index("image_name")
    return out


# ============================================================
# Judge prompt
# ============================================================
JUDGE_SYSTEM = """\
你是图像伪造鉴定的专业评分员。
请基于【参考标准答案 GT】，对【待评估输出】按 4 个维度独立打分（每维 1-10 整数）：

1. accuracy 准确性 — label 是否与 GT 一致；bbox 与 GT 描述区域是否吻合；事实陈述无错误。
2. evidence 证据质量 — 是否引用具体 bbox 坐标 [x1,y1,x2,y2]；列出的视觉/逻辑证据是否可核验、不空泛。
3. completeness 完整性 — 是否覆盖 GT 提到的关键篡改点（label=0 时则评估真实性论证是否充分覆盖光照/纹理/字体一致性等多维度）。
4. language 语言 — 中文流畅度、专业术语使用、长度适宜（理想 300-600 字）、无 markdown/分点。

评分标准:
  10 完美等同 GT 水平
  8-9 优秀，仅次要遗漏
  6-7 合格，覆盖主要点但有疏漏
  4-5 部分正确但漏关键证据
  2-3 严重不足
  1   完全错误或无关
"""

JUDGE_USER_TMPL = """\
【参考标准答案 GT】
label: {gt_label}
caption: {gt_caption}

【待评估输出】
label: {pred_label}
location: {pred_loc}
explanation: {pred_explanation}

请严格按以下 JSON 格式输出（不要输出任何 JSON 以外内容，不要加 ```json 围栏）：
{{"accuracy": <1-10 int>, "evidence": <1-10 int>, "completeness": <1-10 int>, "language": <1-10 int>, "comment": "20 字以内简评"}}
"""


def build_judge_prompt(gt: Dict[str, Any], pred_row: pd.Series) -> List[Dict[str, str]]:
    pred_loc_raw = str(pred_row.get("location", ""))[:600]
    pred_label = int(pred_row["label"]) if pd.notna(pred_row["label"]) else 0
    pred_explanation = str(pred_row.get("explanation", ""))[:1500]

    user = JUDGE_USER_TMPL.format(
        gt_label=gt["gt_label"],
        gt_caption=gt["gt_caption"][:1200],
        pred_label=pred_label,
        pred_loc=pred_loc_raw,
        pred_explanation=pred_explanation,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ============================================================
# Judge output parsing (strip <think>)
# ============================================================
JSON_RE = re.compile(r"\{[\s\S]*?\}")


def parse_judge_output(raw: str) -> Dict[str, Any] | None:
    text = raw
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    candidates = JSON_RE.findall(text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            for k in ("accuracy", "evidence", "completeness", "language"):
                if k not in obj:
                    raise KeyError(k)
                v = int(obj[k])
                obj[k] = max(1, min(10, v))
            obj["comment"] = str(obj.get("comment", ""))[:80]
            return obj
        except Exception:
            continue
    return None


# ============================================================
# vllm batch inference
# ============================================================
def run_judge_batch(
    llm,
    sampling_params,
    tokenizer,
    items: List[Dict[str, Any]],
    cache_dir: Path,
    pbar_desc: str = "judge",
) -> List[Dict[str, Any]]:
    """items: [{key, messages}], returns [{key, raw, parsed}]"""
    todo = []
    cached = []
    for it in items:
        cp = cache_dir / f"{it['key']}.json"
        if cp.exists():
            try:
                cached.append(json.loads(cp.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass
        todo.append(it)

    if not todo:
        return cached

    prompts = []
    for it in todo:
        text = tokenizer.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print(f"[judge] {pbar_desc}: cached={len(cached)} todo={len(todo)}", flush=True)
    outputs = llm.generate(prompts, sampling_params)
    results = list(cached)
    for it, out in zip(todo, outputs):
        raw = out.outputs[0].text
        parsed = parse_judge_output(raw)
        rec = {"key": it["key"], "raw": raw, "parsed": parsed}
        (cache_dir / f"{it['key']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(rec)
    return results


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csvs", nargs="+", required=True,
                    help="name=path entries, e.g. sft=submit_val.csv zs=results/zs/predictions.csv")
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--out_dir", default="tools/baseline/results/judge")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    tp = len(args.gpus.split(","))
    out_dir = Path(args.out_dir)
    cache_root = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"=== judge: {args.judge_model} | TP={tp} | GPUs={args.gpus} ===", flush=True)

    gt_map = load_gt()
    print(f"GT samples: {len(gt_map)}", flush=True)
    pred_dfs = load_predictions(args.pred_csvs)
    pred_names = list(pred_dfs.keys())
    print(f"prediction sets: {pred_names}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    all_summary_rows: List[Dict[str, Any]] = []
    per_pred_results: Dict[str, List[Dict[str, Any]]] = {}

    for name, df in pred_dfs.items():
        print(f"\n--- judging set: {name} (n={len(df)}) ---", flush=True)
        cache_dir = cache_root / name
        cache_dir.mkdir(parents=True, exist_ok=True)

        items = []
        for image_name, gt in gt_map.items():
            if image_name not in df.index:
                continue
            row = df.loc[image_name]
            messages = build_judge_prompt(gt, row)
            items.append({
                "key": Path(image_name).stem,
                "messages": messages,
                "image_name": image_name,
            })

        t0 = time.time()
        results = run_judge_batch(llm, sampling_params, tokenizer, items, cache_dir, pbar_desc=name)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        rows = []
        for r in results:
            p = r.get("parsed") or {"accuracy": 1, "evidence": 1, "completeness": 1, "language": 1, "comment": "PARSE_FAIL"}
            rows.append({
                "image_name": r["key"],
                "accuracy": p["accuracy"],
                "evidence": p["evidence"],
                "completeness": p["completeness"],
                "language": p["language"],
                "comment": p.get("comment", ""),
            })
        out_csv = out_dir / f"{name}_judge.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        per_pred_results[name] = rows

        means = {
            k: sum(r[k] for r in rows) / max(1, len(rows))
            for k in ("accuracy", "evidence", "completeness", "language")
        }
        means["overall"] = sum(means.values()) / 4
        all_summary_rows.append({
            "set": name,
            "n": len(rows),
            **{k: round(v, 3) for k, v in means.items()},
        })
        print(f"  means: {means}", flush=True)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n=== summary: {summary_csv} ===\n{summary_df.to_string(index=False)}", flush=True)

    md = ["# Prompt-only Baseline Judge Report",
          f"\nJudge: `{args.judge_model}` (TP={tp})",
          f"\nVal samples: {len(gt_map)}\n",
          "## Mean scores (1-10)\n",
          summary_df.to_markdown(index=False),
          "\n## Per-sample CSVs",
          ""]
    for name in pred_names:
        md.append(f"- `{name}_judge.csv`")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"=== report: {out_dir/'report.md'} ===", flush=True)


if __name__ == "__main__":
    main()
