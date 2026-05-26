#!/usr/bin/env python
"""
v2 SFT data builder.

把 4 个数据源合并成 ms-swift 标准 vlm SFT 格式：
  1. v1 train/Black   (800 fake，有 mask + 中文 caption)
  2. v1 train/White   (200 real，无 mask + 中文 caption)
  3. processed/synth  (合成 fake，由 keep.txt 过滤)
  4. processed/real_ext (1100 extra real，无 mask)
  5. HydraFake sft_36k.json (筛 type='entire face generation' 的 EFG fake + 20% real)

输出：
  /mnt/nfs/young/TFI/data/v2/sft.json    （主训练集）
  /mnt/nfs/young/TFI/data/v2/sft_meta.json  （统计信息）
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ============================================================
# 路径
# ============================================================
PROJ = Path("/home/young/TFI")
NFS = Path("/mnt/nfs/young/TFI")
V1_TRAIN = Path("/mnt/nfs/young/my_dt/True-or-Fake-Image-main/train_resume")
SYNTH = PROJ / "data/processed/synth"
REAL_EXT = PROJ / "data/processed/real_ext"
HYDRA_SFT = NFS / "data/HydraFake/jsons/train/sft_36k.json"
HYDRA_IMG_ROOT = NFS / "data/HydraFake"

OUT_DIR = NFS / "data/v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 中文 system prompt + user prompt（6 标签 template，TFI 化）
# ============================================================
SYS_PROMPT_ZH = (
    "你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。\n\n"
    "首先用 <fast> </fast> 标签给出第一直觉判断；\n"
    "然后用 <reasoning> </reasoning> 标签给出详细取证推理（高难度样本可在其中包含"
    " <planning> 规划与 <reflection> 自校验）；\n"
    "接着用 <conclusion> </conclusion> 标签给出综合结论，对疑似篡改图必须用 "
    "<bbox>x1,y1,x2,y2</bbox> 或 <region>区域文字描述</region> 标注疑似篡改区域，"
    "其中 bbox 坐标已归一化到 [0,1000]×[0,1000]（左上原点，x1<x2，y1<y2）；\n"
    "最后用 <answer>real|fake</answer> 给出最终判断（仅二选一）。"
)
USR_PROMPT_ZH = "<image>请判断该图像的真实性，并按规定标签格式输出分析。"


# ============================================================
# v1 caption → 6 标签 template
# ============================================================
BBOX_RE = re.compile(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]")
CONCLUSION_HINTS = ("综上", "综合", "因此", "总而言之", "总的来说", "故")
FAKE_INTRO = ("人工智能", "数字伪造", "伪造", "AI 生成", "AI生成", "篡改")
REAL_INTRO = ("真实拍摄", "真实", "未发现", "未经", "原始")


def split_sentences(text: str) -> list[str]:
    text = text.strip().replace("\n", " ")
    parts = re.split(r"(?<=[。！？])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def mask_to_bbox(mask_path: Path) -> Optional[tuple[int, int, int, int]]:
    if not mask_path.exists():
        return None
    try:
        m = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    except Exception:
        return None


def mask_to_multi_bboxes(mask_path: Path, max_n: int = 4, min_area_ratio: float = 0.0005) -> list[tuple[int, int, int, int]]:
    """提取所有连通分量 bbox，按面积排序取 top max_n。"""
    if not mask_path.exists():
        return []
    try:
        from scipy import ndimage  # type: ignore
    except Exception:
        b = mask_to_bbox(mask_path)
        return [b] if b else []
    try:
        m = np.array(Image.open(mask_path).convert("L")) > 0
        if not m.any():
            return []
        H, W = m.shape
        min_area = max(1, int(H * W * min_area_ratio))
        labeled, n = ndimage.label(m)
        bboxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < min_area:
                continue
            bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), len(ys)))
        bboxes.sort(key=lambda b: -b[4])
        return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in bboxes[:max_n]]
    except Exception:
        b = mask_to_bbox(mask_path)
        return [b] if b else []


def bbox_to_region_phrase(bbox: tuple[int, int, int, int], img_size: Optional[tuple[int, int]] = None) -> str:
    """根据 bbox 在图像中的相对位置生成中文区域描述（用于 SAM phrase grounding）。"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if img_size is None:
        return f"图像中心区域 ({x1},{y1})-({x2},{y2})"
    W, H = img_size
    h_zone = "左侧" if cx < W / 3 else ("右侧" if cx > 2 * W / 3 else "中部")
    v_zone = "上方" if cy < H / 3 else ("下方" if cy > 2 * H / 3 else "中部")
    if h_zone == v_zone == "中部":
        return "图像中央区域"
    return f"图像{v_zone}{h_zone}区域"


def img_size_of(p: Path) -> Optional[tuple[int, int]]:
    try:
        with Image.open(p) as im:
            return im.size  # (W, H)
    except Exception:
        return None


def caption_to_template(
    caption: str,
    label: int,
    bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    img_size: Optional[tuple[int, int]] = None,
) -> str:
    """v1 中文 caption 拆成 4 标签（fast/reasoning/conclusion/answer），并把多 bbox + region 嵌入 conclusion。"""
    # Normalize any raw [x1,y1,x2,y2] pixel coords scattered throughout the
    # caption prose to [0,1000]. Otherwise the reasoning text reinforces the
    # model to emit raw-pixel coordinates, undoing the <bbox> normalization.
    if img_size and label == 1:
        W, H = img_size
        if W > 0 and H > 0:
            def _norm_bracket(m):
                x1, y1, x2, y2 = (int(m.group(i)) for i in (1, 2, 3, 4))
                nx1 = max(0, min(1000, round(x1 / W * 1000)))
                ny1 = max(0, min(1000, round(y1 / H * 1000)))
                nx2 = max(0, min(1000, round(x2 / W * 1000)))
                ny2 = max(0, min(1000, round(y2 / H * 1000)))
                return f"[{nx1}, {ny1}, {nx2}, {ny2}]"
            caption = re.sub(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", _norm_bracket, caption)

    sentences = split_sentences(caption)
    if not sentences:
        ans = "fake" if label == 1 else "real"
        return f"<fast> 图像存疑。 </fast>\n<reasoning> {caption} </reasoning>\n<conclusion> 综合判断为 {ans}。 </conclusion>\n<answer>{ans}</answer>"

    fast = sentences[0]
    body = sentences[1:] if len(sentences) > 1 else []
    concl_idx = next(
        (i for i, s in enumerate(body) if any(h in s for h in CONCLUSION_HINTS)), -1
    )
    if concl_idx >= 0:
        reasoning_sents = body[:concl_idx]
        conclusion_sents = body[concl_idx:]
    else:
        cut = max(1, int(len(body) * 0.7))
        reasoning_sents = body[:cut]
        conclusion_sents = body[cut:] if cut < len(body) else [sentences[-1]]
    reasoning = " ".join(reasoning_sents) if reasoning_sents else " ".join(body)
    conclusion = " ".join(conclusion_sents) if conclusion_sents else "综合上述分析得出结论。"

    bbox_tags: list[str] = []
    if label == 1:
        # Bbox normalization to [0,1000]×[0,1000]: tile-based vision tokens lose
        # raw-pixel scale; training with normalized coords matches what the
        # model can actually learn from the input embedding.
        def _norm(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
            if not img_size:
                return x1, y1, x2, y2
            W, H = img_size
            if W <= 0 or H <= 0:
                return x1, y1, x2, y2
            return (
                max(0, min(1000, round(x1 / W * 1000))),
                max(0, min(1000, round(y1 / H * 1000))),
                max(0, min(1000, round(x2 / W * 1000))),
                max(0, min(1000, round(y2 / H * 1000))),
            )

        cap_bboxes = BBOX_RE.findall(caption)
        if cap_bboxes:
            # caption was already normalized at the top of this function (raw
            # [x,y,x,y] brackets in prose were rewritten to [0,1000] coords).
            # So cap_bboxes are ALREADY in normalized space — emit verbatim.
            for x1, y1, x2, y2 in cap_bboxes[:4]:
                bbox_tags.append(f"<bbox>{int(x1)},{int(y1)},{int(x2)},{int(y2)}</bbox>")
        elif bboxes:
            # mask-derived bboxes are still in raw pixel space; normalize here.
            for bb in bboxes:
                x1, y1, x2, y2 = bb
                phrase = bbox_to_region_phrase(bb, img_size)
                nx1, ny1, nx2, ny2 = _norm(x1, y1, x2, y2)
                bbox_tags.append(f"<bbox>{nx1},{ny1},{nx2},{ny2}</bbox><region>{phrase}</region>")

    if bbox_tags and "<bbox>" not in conclusion:
        conclusion = conclusion.rstrip("。 ").rstrip() + "。篡改区域：" + " ".join(bbox_tags) + "。"

    ans = "fake" if label == 1 else "real"
    return (
        f"<fast> {fast} </fast>\n"
        f"<reasoning> {reasoning} </reasoning>\n"
        f"<conclusion> {conclusion} </conclusion>\n"
        f"<answer>{ans}</answer>"
    )


def make_record(image_path: Path, assistant: str, label: int, type_tag: str, source: str) -> dict:
    return {
        "images": [str(image_path)],
        "type": type_tag,
        "label": label,
        "source": source,
        "messages": [
            {"role": "system", "content": SYS_PROMPT_ZH},
            {"role": "user", "content": USR_PROMPT_ZH},
            {"role": "assistant", "content": assistant},
        ],
    }


# ============================================================
# 各数据源 builders
# ============================================================
def build_v1_fake(limit: Optional[int] = None) -> list[dict]:
    img_dir = V1_TRAIN / "Black/Image"
    cap_dir = V1_TRAIN / "Black/Caption_clean"
    msk_dir = V1_TRAIN / "Black/Mask"
    out = []
    paths = sorted(img_dir.glob("*"))
    if limit:
        paths = paths[:limit]
    for p in paths:
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cap_p = cap_dir / f"{p.stem}.md"
        if not cap_p.exists():
            continue
        caption = cap_p.read_text(encoding="utf-8").strip()
        msk_p = msk_dir / f"{p.stem}.png"
        bboxes = mask_to_multi_bboxes(msk_p)
        size = img_size_of(p)
        assistant = caption_to_template(caption, label=1, bboxes=bboxes, img_size=size)
        rec = make_record(p, assistant, label=1, type_tag="v1_fake", source="v1_train_black")
        rec["mask_path"] = str(msk_p) if msk_p.exists() else None
        out.append(rec)
    return out


def build_v1_real(limit: Optional[int] = None) -> list[dict]:
    img_dir = V1_TRAIN / "White/Image"
    cap_dir = V1_TRAIN / "White/Caption"
    out = []
    paths = sorted(img_dir.glob("*"))
    if limit:
        paths = paths[:limit]
    for p in paths:
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cap_p = cap_dir / f"{p.stem}.md"
        if not cap_p.exists():
            continue
        caption = cap_p.read_text(encoding="utf-8").strip()
        assistant = caption_to_template(caption, label=0)
        out.append(make_record(p, assistant, label=0, type_tag="v1_real", source="v1_train_white"))
    return out


def build_synth() -> list[dict]:
    keep_set = set()
    keep_p = SYNTH / "keep.txt"
    if keep_p.exists():
        for line in keep_p.read_text().splitlines():
            line = line.strip()
            if line:
                keep_set.add(line)

    meta_p = SYNTH / "meta.jsonl"
    if not meta_p.exists():
        return []
    out = []
    for line in meta_p.read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        stem = m["stem"]
        img_name_jpg = f"{stem}.jpg"
        img_name_png = f"{stem}.png"
        in_keep = (img_name_jpg in keep_set) or (img_name_png in keep_set) or m.get("keep", False)
        if not in_keep:
            continue
        img_p = SYNTH / "Image" / img_name_jpg
        if not img_p.exists():
            img_p = SYNTH / "Image" / img_name_png
        if not img_p.exists():
            continue
        msk_p = SYNTH / "Mask" / f"{stem}.png"
        bboxes = mask_to_multi_bboxes(msk_p)
        if not bboxes:
            continue
        size = img_size_of(img_p)
        type_en = m.get("type", "splice")
        type_zh = {
            "copy_move": "复制粘贴 (copy-move)",
            "splice": "拼接 (splice)",
            "text_replace": "文字编辑 (text-edit)",
            "text": "文字编辑 (text-edit)",
        }.get(type_en, "局部篡改")
        bbox_phrase_strs = []
        W, H = (size if size else (None, None))
        for bb in bboxes:
            x1, y1, x2, y2 = bb
            phr = bbox_to_region_phrase(bb, size)
            if W and H and W > 0 and H > 0:
                nx1 = max(0, min(1000, round(x1 / W * 1000)))
                ny1 = max(0, min(1000, round(y1 / H * 1000)))
                nx2 = max(0, min(1000, round(x2 / W * 1000)))
                ny2 = max(0, min(1000, round(y2 / H * 1000)))
            else:
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            bbox_phrase_strs.append(f"<bbox>{nx1},{ny1},{nx2},{ny2}</bbox><region>{phr}</region>")
        bbox_inline = bbox_phrase_strs[0]
        bbox_full = " ".join(bbox_phrase_strs)
        assistant = (
            f"<fast> 图像疑似经过{type_zh}操作，存在数字伪造痕迹。 </fast>\n"
            f"<reasoning> 仔细观察图像，目标区域的边缘过渡、纹理一致性、光照与阴影方向均与周围环境存在差异。"
            f"{type_zh}操作通常会在篡改边缘留下细微的不连续，包括色彩、噪点、JPEG 压缩痕迹的不自然过渡。"
            f"本图在 {bbox_inline} 区域内呈现上述特征。 </reasoning>\n"
            f"<conclusion> 综合边缘伪影、纹理矛盾与上下文不一致性，可判定该图为伪造图像，主要篡改类型为 "
            f"{type_zh}。篡改区域：{bbox_full}。 </conclusion>\n"
            f"<answer>fake</answer>"
        )
        rec = make_record(img_p, assistant, label=1, type_tag=f"synth_{type_en}", source="processed_synth")
        rec["mask_path"] = str(msk_p) if msk_p.exists() else None
        out.append(rec)
    return out


def build_hydrafake(efg_fake_limit: Optional[int] = None, real_ratio: float = 0.20, seed: int = 42) -> list[dict]:
    """筛 HydraFake：EFG fake 全量 + real 抽 20%（按 README §3.1）。"""
    data = json.load(open(HYDRA_SFT, "r", encoding="utf-8"))
    rng = random.Random(seed)
    out = []
    fake_efg = []
    real_pool = []
    for r in data:
        t = r.get("type", "")
        if t == "entire face generation":
            fake_efg.append(r)
        elif t == "real":
            real_pool.append(r)
        # FS / FR 完全跳过

    if efg_fake_limit:
        rng.shuffle(fake_efg)
        fake_efg = fake_efg[:efg_fake_limit]
    real_n = int(len(real_pool) * real_ratio)
    rng.shuffle(real_pool)
    real_kept = real_pool[:real_n]

    for r in fake_efg + real_kept:
        rel = r["images"][0]
        abs_p = HYDRA_IMG_ROOT / rel
        if not abs_p.exists():
            continue
        rec = {
            "images": [str(abs_p)],
            "type": f"hydra_{r['type'].replace(' ', '_')}",
            "label": r["label"],
            "source": "hydrafake_sft36k",
            "messages": r["messages"],
        }
        out.append(rec)
    return out


# ============================================================
# main
# ============================================================
def dedup_by_image(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        key = hashlib.md5(r["images"][0].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "sft.json"))
    ap.add_argument("--out_meta", default=str(OUT_DIR / "sft_meta.json"))
    ap.add_argument("--val_split", type=float, default=0.05, help="切多少给 val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hydra_efg_limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="只跑前 20 张做 dry run")
    args = ap.parse_args()

    if args.smoke:
        print("[smoke] limit=20 each source")
        v1_f = build_v1_fake(limit=20)
        v1_r = build_v1_real(limit=20)
        synth = build_synth()[:20]
        hydra = []
    else:
        v1_f = build_v1_fake()
        v1_r = build_v1_real()
        synth = build_synth()
        # real_ext: 用户判定质量太低，弃用
        # HydraFake: 图片下载需要时间（5.94G+），暂时只拿 jsons 用作 template 参考
        try:
            hydra = build_hydrafake(efg_fake_limit=args.hydra_efg_limit) if args.hydra_efg_limit else []
        except Exception as e:
            print(f"[warn] hydrafake skipped: {e}")
            hydra = []

    src_counts = {
        "v1_fake": len(v1_f),
        "v1_real": len(v1_r),
        "synth": len(synth),
        "hydrafake": len(hydra),
    }
    print(f"[build] src counts: {src_counts}")

    all_recs = v1_f + v1_r + synth + hydra
    all_recs = dedup_by_image(all_recs)
    print(f"[build] total after dedup: {len(all_recs)}")

    rng = random.Random(args.seed)
    rng.shuffle(all_recs)
    n_val = max(1, int(len(all_recs) * args.val_split))
    val_recs = all_recs[:n_val]
    train_recs = all_recs[n_val:]

    out_main = Path(args.out)
    out_val = out_main.with_name(out_main.stem + "_val.json")
    out_main.parent.mkdir(parents=True, exist_ok=True)
    with open(out_main, "w", encoding="utf-8") as f:
        json.dump(train_recs, f, ensure_ascii=False, indent=1)
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(val_recs, f, ensure_ascii=False, indent=1)

    label_counts = {0: 0, 1: 0}
    for r in train_recs:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    src_dist = {}
    for r in train_recs:
        src_dist[r["source"]] = src_dist.get(r["source"], 0) + 1

    meta = {
        "src_counts_raw": src_counts,
        "total_after_dedup": len(all_recs),
        "train_n": len(train_recs),
        "val_n": len(val_recs),
        "train_label_dist": label_counts,
        "train_source_dist": src_dist,
        "out_train": str(out_main),
        "out_val": str(out_val),
    }
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
