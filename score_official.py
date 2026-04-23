"""官方评测复现脚本 (Reproduce official S_Fin)。

公式 (赛题官网):
    S_Det  = image-level F1            (二分类: forged vs real)
    S_Loc  = pixel-level F1            (= Dice on forged samples)
    S_Sim  = BERTScore F1 (zh)         (生成解释 vs 参考)
    S_Auto = Qwen3-MAX rubrics score   (0-100 -> /100)
    S_Exp  = 0.5 * S_Auto + 0.5 * S_Sim
    S_Fin  = 0.45 * S_Det + 0.25 * S_Loc + 0.30 * S_Exp

输入:
    --pred_csv      推理输出, 列 [image_name, label, location(rle json), explanation]
    --val_dir       GT 根目录, 期望:
                        Black/Image/*.jpg
                        Black/Mask/*.png
                        Black/Caption_clean/*.txt
                        White/Image/*.jpg
                        White/Caption/*.txt
    --bert_lang     'zh' (默认)
    --qwen_model    'qwen-max' (默认)  设 'none' 跳过 S_Auto
    --gpu           BERTScore 使用的 GPU id

输出:
    JSON 报告 + markdown 表格 + 控制台打印
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# 复用项目内工具 (rle decode + dice)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compute_f1, compute_dice  # noqa: E402


# ============================================================
# 1. 数据装载
# ============================================================

def load_predictions(csv_path: str) -> Dict[str, Dict]:
    """读 submit.csv -> {name: {label, rle(dict), explanation}}"""
    out: Dict[str, Dict] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rle = json.loads(row["location"])
            except Exception:
                rle = None
            out[row["image_name"]] = {
                "label": int(row["label"]),
                "rle": rle,
                "explanation": row.get("explanation", "") or "",
            }
    return out


def rle_to_mask(rle: Dict, h: int, w: int) -> np.ndarray:
    """COCO RLE -> binary mask (H, W) uint8"""
    if rle is None:
        return np.zeros((h, w), dtype=np.uint8)
    try:
        from pycocotools import mask as cocomask
        if isinstance(rle.get("counts"), str):
            r = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
        else:
            r = rle
        m = cocomask.decode(r)
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(np.uint8)
    except Exception as e:
        print(f"[warn] rle_to_mask failed: {e}", file=sys.stderr)
        return np.zeros((h, w), dtype=np.uint8)


def collect_gt(val_dir: str) -> List[Dict]:
    """扫描 val/Black + val/White, 每个样本含 gt_label / gt_mask_path / ref_text"""
    val = Path(val_dir)
    samples: List[Dict] = []

    # forged (Black)
    img_dir = val / "Black" / "Image"
    msk_dir = val / "Black" / "Mask"
    cap_dir = val / "Black" / "Caption_clean"
    cap_dir_fb = val / "Black" / "Caption"
    for fn in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(fn)[0]
        # caption 后缀实际为 .md (非 .txt), 二者都尝试
        cap_path = None
        for d in (cap_dir, cap_dir_fb):
            for ext in (".md", ".txt"):
                p = d / f"{stem}{ext}"
                if p.exists():
                    cap_path = p; break
            if cap_path is not None: break
        ref_text = cap_path.read_text(encoding="utf-8", errors="ignore").strip() if cap_path else ""
        samples.append({
            "name": fn,
            "image_path": str(img_dir / fn),
            "mask_path": str(msk_dir / f"{stem}.png"),
            "ref_text": ref_text,
            "gt_label": 1,
        })

    # real (White)
    img_dir = val / "White" / "Image"
    cap_dir = val / "White" / "Caption"
    for fn in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(fn)[0]
        cap_path = None
        for ext in (".md", ".txt"):
            p = cap_dir / f"{stem}{ext}"
            if p.exists():
                cap_path = p; break
        ref_text = cap_path.read_text(encoding="utf-8", errors="ignore").strip() if cap_path else ""
        samples.append({
            "name": fn,
            "image_path": str(img_dir / fn),
            "mask_path": None,
            "ref_text": ref_text,
            "gt_label": 0,
        })
    return samples


# ============================================================
# 2. S_Det / S_Loc
# ============================================================

def score_detection(samples: List[Dict], preds: Dict[str, Dict]) -> Dict:
    pred_lbls, gt_lbls = [], []
    for s in samples:
        if s["name"] not in preds:
            print(f"[warn] missing pred: {s['name']}")
            pred_lbls.append(0)
        else:
            pred_lbls.append(preds[s["name"]]["label"])
        gt_lbls.append(s["gt_label"])
    return compute_f1(np.asarray(pred_lbls), np.asarray(gt_lbls))


def score_grounding(samples: List[Dict], preds: Dict[str, Dict]) -> Dict:
    """官方: pixel-level F1 = Dice (only on forged samples)."""
    dices: List[float] = []
    for s in samples:
        if s["gt_label"] == 0:
            continue
        if s["name"] not in preds:
            dices.append(0.0)
            continue
        gt = np.array(Image.open(s["mask_path"]).convert("L"))
        gt_bin = (gt > 127).astype(np.uint8)
        h, w = gt_bin.shape
        rle = preds[s["name"]]["rle"]
        if rle is None:
            dices.append(0.0); continue
        # rle 里的 size 应等于 gt 的 (h, w)
        rsz = rle.get("size", [h, w])
        pred_mask = rle_to_mask(rle, rsz[0], rsz[1])
        if pred_mask.shape != gt_bin.shape:
            from PIL import Image as PI
            pred_mask = np.array(PI.fromarray(pred_mask * 255).resize(
                (w, h), PI.NEAREST)) > 127
            pred_mask = pred_mask.astype(np.uint8)
        dices.append(compute_dice(pred_mask, gt_bin))
    return {
        "pixel_f1": float(np.mean(dices)) if dices else 0.0,
        "n_forged": len(dices),
    }


# ============================================================
# 3. S_Sim (BERTScore)
# ============================================================

def score_similarity(samples: List[Dict], preds: Dict[str, Dict],
                     lang: str = "zh", batch_size: int = 32,
                     model_type: Optional[str] = None) -> Dict:
    from bert_score import score as bs_score
    pairs: List[Tuple[str, str, str]] = []  # (name, pred, ref)
    for s in samples:
        ref = s["ref_text"].strip()
        pred = preds.get(s["name"], {}).get("explanation", "").strip()
        if not ref or not pred:
            continue
        pairs.append((s["name"], pred, ref))

    if not pairs:
        return {"bertscore_f1": 0.0, "n_pairs": 0}

    cands = [p[1] for p in pairs]
    refs = [p[2] for p in pairs]
    P, R, F = bs_score(cands, refs, lang=lang, model_type=model_type,
                       batch_size=batch_size, verbose=False, rescale_with_baseline=False)
    f_mean = float(F.mean().item())
    per_sample = {p[0]: float(F[i].item()) for i, p in enumerate(pairs)}
    return {"bertscore_f1": f_mean, "n_pairs": len(pairs),
            "per_sample": per_sample}


# ============================================================
# 4. S_Auto (Qwen3-MAX rubrics 100 分制)
# ============================================================

RUBRICS_PROMPT = """你是图像伪造鉴定领域的资深评估专家。下面给出
- 【参考鉴定文本】(标准答案)
- 【待评估鉴定文本】(待打分的模型输出)

请基于以下 4 个维度按 100 分制对【待评估文本】进行综合打分:
1. 内容准确性 (30 分): 对图像伪造/真实判定结论是否正确, 关键事实(品牌、金额、日期、场景描述)是否与参考一致。
2. 证据具体度 (30 分): 是否引用具体的伪造区域坐标 [x1,y1,x2,y2]、像素特征(字体、边缘、纹理、光照、JPEG伪影、噪声)等可验证证据。
3. 推理逻辑性 (20 分): 论证链条是否完整连贯, 从证据到结论是否自然过渡, 无逻辑跳跃。
4. 表达专业性 (20 分): 用词是否专业准确, 结构是否清晰, 长度是否在 300-600 字范围内。

【参考鉴定文本】
{ref}

【待评估鉴定文本】
{pred}

请严格按以下 JSON 格式输出, 不要任何额外解释:
{{"accuracy": <0-30>, "evidence": <0-30>, "logic": <0-20>, "professional": <0-20>, "total": <0-100>}}
"""


def call_qwen_score(name: str, ref: str, pred: str, model: str,
                    api_key: str, retries: int = 3) -> Tuple[str, Optional[float], Optional[Dict]]:
    """单个样本调 qwen-max, 返回 (name, total/100, raw_dict)。失败返回 (name, None, None)"""
    import dashscope
    from dashscope import Generation
    dashscope.api_key = api_key

    prompt = RUBRICS_PROMPT.format(ref=ref[:2000], pred=pred[:2000])
    last_err = None
    for attempt in range(retries):
        try:
            r = Generation.call(
                model=model, prompt=prompt, result_format="message",
                temperature=0.0, top_p=0.5, max_tokens=200,
            )
            if r.status_code != 200:
                last_err = f"http={r.status_code} {r.message}"
                time.sleep(2 ** attempt)
                continue
            txt = r.output.choices[0].message.content.strip()
            # 提取 JSON
            l, R = txt.find("{"), txt.rfind("}")
            if l < 0 or R < l:
                raise ValueError(f"no json in: {txt[:120]}")
            d = json.loads(txt[l:R+1])
            total = float(d.get("total", -1))
            if total < 0 or total > 100:
                # 尝试用四个分项重算
                total = sum(float(d.get(k, 0)) for k in ["accuracy", "evidence", "logic", "professional"])
            return name, total / 100.0, d
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    print(f"[qwen] {name} failed: {last_err}", file=sys.stderr)
    return name, None, None


def score_auto(samples: List[Dict], preds: Dict[str, Dict],
               api_key: str, model: str = "qwen-max",
               max_workers: int = 4, cache_path: Optional[str] = None) -> Dict:
    cache: Dict[str, Dict] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[s_auto] cache: {len(cache)} pre-scored")

    todo: List[Tuple[str, str, str]] = []
    for s in samples:
        ref = s["ref_text"].strip()
        pred = preds.get(s["name"], {}).get("explanation", "").strip()
        if not ref or not pred:
            continue
        if s["name"] in cache and cache[s["name"]].get("total_norm") is not None:
            continue
        todo.append((s["name"], ref, pred))

    if todo:
        print(f"[s_auto] querying qwen for {len(todo)} samples (workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(call_qwen_score, n, r, p, model, api_key)
                    for n, r, p in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                name, total_norm, raw = fut.result()
                cache[name] = {"total_norm": total_norm, "raw": raw}
                if i % 10 == 0 or i == len(todo):
                    print(f"  qwen progress: {i}/{len(todo)}")
                    if cache_path:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    valid = [v["total_norm"] for v in cache.values() if v.get("total_norm") is not None]
    return {"s_auto": float(np.mean(valid)) if valid else 0.0,
            "n_scored": len(valid),
            "cache_path": cache_path}


# ============================================================
# 5. main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--val_dir", default="data/raw/val")
    ap.add_argument("--out_json", default="logs/score_official.json")
    ap.add_argument("--out_md", default="logs/score_official.md")
    ap.add_argument("--bert_lang", default="zh")
    ap.add_argument("--bert_model", default=None,
                    help="覆盖 bert_score 默认中文模型, 例如 hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--qwen_model", default="qwen-max",
                    help="qwen-max / qwen-max-latest / none(=跳过 S_Auto)")
    ap.add_argument("--qwen_workers", type=int, default=6)
    ap.add_argument("--qwen_cache", default="cache/qwen_rubric_scores.json")
    ap.add_argument("--api_key", default=os.environ.get(
        "DASHSCOPE_API_KEY", "sk-1a444cab439a452cb5cb78d8a208521d"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] preds  : {args.pred_csv}")
    preds = load_predictions(args.pred_csv)
    print(f"        rows  : {len(preds)}")
    print(f"[load] val_dir: {args.val_dir}")
    samples = collect_gt(args.val_dir)
    n_forged = sum(1 for s in samples if s["gt_label"] == 1)
    print(f"        n     : {len(samples)} (forged={n_forged}, real={len(samples)-n_forged})")

    # S_Det
    print("\n=== S_Det (image-level F1) ===")
    det = score_detection(samples, preds)
    print(json.dumps(det, indent=2))
    s_det = det["f1"]

    # S_Loc
    print("\n=== S_Loc (pixel-level F1 = Dice on forged) ===")
    loc = score_grounding(samples, preds)
    print(json.dumps(loc, indent=2))
    s_loc = loc["pixel_f1"]

    # S_Sim
    print("\n=== S_Sim (BERTScore-zh) ===")
    sim = score_similarity(samples, preds, lang=args.bert_lang,
                           model_type=args.bert_model)
    print(f"  bertscore_f1={sim['bertscore_f1']:.4f}  n={sim['n_pairs']}")
    s_sim = sim["bertscore_f1"]

    # S_Auto
    if args.qwen_model.lower() == "none":
        print("\n=== S_Auto (skipped) ===")
        s_auto = 0.0
        auto_info = {"s_auto": 0.0, "skipped": True}
    else:
        print(f"\n=== S_Auto (Qwen rubrics, model={args.qwen_model}) ===")
        Path(args.qwen_cache).parent.mkdir(parents=True, exist_ok=True)
        auto_info = score_auto(samples, preds, args.api_key,
                               model=args.qwen_model,
                               max_workers=args.qwen_workers,
                               cache_path=args.qwen_cache)
        s_auto = auto_info["s_auto"]
        print(f"  s_auto={s_auto:.4f}  n={auto_info['n_scored']}")

    # 汇总
    s_exp = 0.5 * s_auto + 0.5 * s_sim
    s_fin = 0.45 * s_det + 0.25 * s_loc + 0.30 * s_exp

    summary = {
        "S_Det": s_det,
        "S_Loc": s_loc,
        "S_Sim": s_sim,
        "S_Auto": s_auto,
        "S_Exp": s_exp,
        "S_Fin": s_fin,
        "details": {
            "detection": det, "grounding": loc,
            "similarity": {"bertscore_f1": s_sim, "n_pairs": sim["n_pairs"]},
            "automatic": auto_info,
        },
    }
    print("\n" + "=" * 60)
    print(f"  S_Det = {s_det:.4f}")
    print(f"  S_Loc = {s_loc:.4f}")
    print(f"  S_Sim = {s_sim:.4f}")
    print(f"  S_Auto= {s_auto:.4f}")
    print(f"  S_Exp = {s_exp:.4f}  ( = 0.5*S_Auto + 0.5*S_Sim )")
    print(f"  S_Fin = {s_fin:.4f}  ( = 0.45*S_Det + 0.25*S_Loc + 0.30*S_Exp )")
    print("=" * 60)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# TFI Official Score (val)\n\n")
        f.write(f"_pred = `{args.pred_csv}`, val_dir = `{args.val_dir}`_\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| S_Det (img-F1)  | **{s_det:.4f}** |\n")
        f.write(f"| S_Loc (pix-F1)  | **{s_loc:.4f}** |\n")
        f.write(f"| S_Sim (BERTSc.) | {s_sim:.4f} |\n")
        f.write(f"| S_Auto (Qwen)   | {s_auto:.4f} |\n")
        f.write(f"| S_Exp           | **{s_exp:.4f}** |\n")
        f.write(f"| **S_Fin**       | **{s_fin:.4f}** |\n")
    print(f"\n[done] -> {args.out_json}\n        -> {args.out_md}")


if __name__ == "__main__":
    main()
