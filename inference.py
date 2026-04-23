"""新版推理流水线 (Qwen3.5-9B + 证据驱动 + 校准器)。

阶段:
  Stage 1   : 分割集成 + 多尺度 TTA -> 平均概率图 -> 二值 mask
  Stage 1.5 : 5x EfficientNet 分类器投票
  Stage 2   : 结构化证据抽取 (evidence.py)
  Stage 2.5 : 轻量校准器融合 -> 最终 label
  Stage 3   : Qwen3.5-9B with evidence prompt -> explanation
  Stage 4   : 写 submit.csv

所有中间结果缓存到 cache/, 中断后重启自动跳过已完成阶段。

用法:
  python inference.py --config config.yaml
  python inference.py  # 使用默认配置
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

# ============================================================
# 配置 (可被 --config yaml 覆盖)
# ============================================================

DEFAULT_CFG = {
    "test_dir": "data/raw/test/Image",
    "output": "submit.csv",
    "cache_dir": "cache",
    "log_file": "logs/vlm/inference.log",
    "checkpoint_dir": "checkpoints",
    "vlm_model": "checkpoints/qwen35_9b",
    "vlm_base": "models/Qwen3.5-9B",
    "calibrator_dir": "checkpoints/calibrator",
    "use_calibrator": True,
    "use_classifier": True,
    "use_convnext": True,    # 简历主集成只用 segformer+maxvit, 但默认全用
    # 分割
    "img_sizes": [768, 896],   # 多尺度 TTA
    "img_sizes_no_maxvit": [640],
    "threshold": 0.3,
    "use_tta": True,
    "min_area": 100,
    "morph_kernel": 5,
    "label_threshold": 0.001,
    # 旧硬规则 (calibrator 不可用时回退)
    "cls_override_low": 0.2,
    "cls_override_high": 0.9,
    # VLM
    "max_new_tokens": 1024,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
    "gpu": 0,
}


def load_config(path: Optional[str]) -> Dict:
    cfg = dict(DEFAULT_CFG)
    if path and os.path.exists(path):
        import yaml
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("inference")


# ============================================================
# Stage 1: 分割集成
# ============================================================

def load_seg_models(checkpoint_dir: str, archs: List[str], blacklist=None):
    blacklist = set(blacklist or [])
    seg_dir = Path(checkpoint_dir) / "seg"
    out = []
    if not seg_dir.exists():
        return out
    for d in sorted(seg_dir.iterdir()):
        if not d.is_dir() or d.name in blacklist:
            continue
        if not (d / "best_model.pt").exists():
            continue
        for a in archs:
            if a in d.name:
                out.append({"name": d.name, "arch": a, "path": str(d / "best_model.pt")})
                break
    return out


def _build_seg_model(arch: str):
    from train_seg_ensemble import build_segformer, build_smp_model, SegModelWrapper
    if arch == "segformer":
        raw = build_segformer(in_channels=7, num_classes=1, pretrained=False)
        return SegModelWrapper(raw, "segformer")
    raw = build_smp_model(arch, in_channels=7, num_classes=1, pretrained=False)
    return SegModelWrapper(raw, "smp")


def stage1_segmentation(cfg: Dict, device, logger) -> Dict[str, Dict]:
    """跑分割集成 + 多尺度 TTA, 返回 {name: {prob_map, orig_size}}。"""
    from dataset import TestImageDataset
    archs = ["segformer", "maxvit"]
    if cfg["use_convnext"]:
        archs.append("convnext")
    models = load_seg_models(cfg["checkpoint_dir"], archs)
    if not models:
        logger.warning("[stage1] no segmentation models found, returning zeros")
        ds = TestImageDataset(cfg["test_dir"], img_size=max(cfg["img_sizes"]))
        results = {}
        for i in range(len(ds)):
            s = ds[i]
            h, w = s["orig_size"]
            results[s["image_name"]] = {"prob": np.zeros((h, w), np.float32),
                                        "orig_size": (h, w)}
        return results

    logger.info(f"[stage1] {len(models)} models, scales={cfg['img_sizes']}+{cfg['img_sizes_no_maxvit']}")

    # name -> list of (prob_at_some_size, weight)
    accum: Dict[str, List[np.ndarray]] = {}
    orig_sizes: Dict[str, tuple] = {}

    all_scales = sorted(set(cfg["img_sizes"] + cfg["img_sizes_no_maxvit"]))

    for sz in all_scales:
        ds = TestImageDataset(cfg["test_dir"], img_size=sz)
        for mi in models:
            if mi["arch"] == "maxvit" and sz not in cfg["img_sizes"]:
                continue
            t0 = time.time()
            logger.info(f"  [{mi['name']} @ {sz}]")
            model = _build_seg_model(mi["arch"])
            state = torch.load(mi["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model = model.to(device).eval()

            with torch.no_grad():
                for idx in tqdm(range(len(ds)), ncols=80, desc=f"    {mi['name']}@{sz}"):
                    s = ds[idx]
                    name = s["image_name"]
                    orig_sizes[name] = s["orig_size"]
                    x = s["pixel_values"].unsqueeze(0).to(device)
                    if cfg["use_tta"]:
                        ttas = []
                        for tfn, ifn in [
                            (lambda t: t, lambda t: t),
                            (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
                            (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),
                        ]:
                            with autocast("cuda", dtype=torch.bfloat16):
                                logits = model(tfn(x))
                            ttas.append(ifn(torch.sigmoid(logits)))
                        prob = torch.stack(ttas).mean(0)[0, 0].float().cpu().numpy()
                    else:
                        with autocast("cuda", dtype=torch.bfloat16):
                            logits = model(x)
                        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

                    h, w = s["orig_size"]
                    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                    accum.setdefault(name, []).append(prob)

            del model
            torch.cuda.empty_cache()
            logger.info(f"    done {time.time()-t0:.1f}s")

    out = {}
    for name, probs in accum.items():
        out[name] = {
            "prob": np.mean(probs, axis=0).astype(np.float32),
            "orig_size": orig_sizes[name],
        }
    return out


# ============================================================
# Stage 1.5: 分类器投票
# ============================================================

def stage1_5_classifier(cfg: Dict, device, logger) -> Dict[str, Dict]:
    from train_classifier import ForgeryClassifier
    from utils import compute_ela
    cls_dir = Path(cfg["checkpoint_dir"]) / "cls"
    if not cls_dir.exists():
        return {}
    test_dir = cfg["test_dir"]
    test_imgs = sorted([f for f in os.listdir(test_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    name2scores = {n: [] for n in test_imgs}

    model_dirs = sorted([d for d in cls_dir.iterdir()
                         if d.is_dir() and (d / "best_model.pt").exists()])
    logger.info(f"[stage1.5] {len(model_dirs)} classifiers")

    for d in model_dirs:
        t0 = time.time()
        model = ForgeryClassifier(in_channels=6, num_classes=2)
        state = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device).eval()
        for n in tqdm(test_imgs, desc=f"    {d.name}", ncols=80):
            img = np.array(Image.open(os.path.join(test_dir, n)).convert("RGB"))
            img_r = cv2.resize(img, (512, 512))
            ela = compute_ela(img_r)
            x = np.concatenate([img_r.astype(np.float32) / 255,
                                ela.astype(np.float32) / 255], axis=2)
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
            with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(x), dim=1)[0, 1].item()
            name2scores[n].append(p)
        del model
        torch.cuda.empty_cache()
        logger.info(f"    done {time.time()-t0:.1f}s")

    return {n: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for n, v in name2scores.items() if v}


# ============================================================
# Stage 2: 证据抽取 + 校准
# ============================================================

def stage2_evidence_and_calibrate(seg_outputs, cls_scores, cfg, logger):
    from evidence import extract
    from utils import postprocess_mask, mask_to_rle, create_zero_rle
    from calibrator import Calibrator, hard_rule_baseline

    cal = None
    if cfg["use_calibrator"]:
        cal_dir = Path(cfg["calibrator_dir"])
        if (cal_dir / "calibrator.pkl").exists():
            cal = Calibrator.load(str(cal_dir))
            logger.info(f"[stage2] calibrator loaded, threshold={cal.threshold:.3f}")
        else:
            logger.warning(f"[stage2] no calibrator at {cal_dir}, fallback to hard rule")

    results = {}
    for name in tqdm(sorted(seg_outputs.keys()), desc="  evidence", ncols=80):
        seg = seg_outputs[name]
        prob = seg["prob"]
        h, w = seg["orig_size"]
        binary = (prob > cfg["threshold"]).astype(np.uint8)
        binary = postprocess_mask(binary, morph_kernel_size=cfg["morph_kernel"],
                                  min_area=cfg["min_area"])
        img = np.array(Image.open(os.path.join(cfg["test_dir"], name)).convert("RGB"))
        ev = extract(img, binary, prob_map=prob,
                     label_threshold=cfg["label_threshold"],
                     min_area_px=cfg["min_area"])

        if cls_scores:
            raw = cls_scores.get(name, None)
            if isinstance(raw, dict):
                cls_mean = raw.get("mean", 0.5)
                cls_std = raw.get("std", 0.0)
            elif isinstance(raw, (list, tuple)) and len(raw) >= 1:
                cls_mean = float(raw[0])
                cls_std = float(raw[1]) if len(raw) > 1 else 0.0
            elif isinstance(raw, (int, float)):
                cls_mean = float(raw); cls_std = 0.0
            else:
                cls_mean = 0.5; cls_std = 0.0
        else:
            cls_mean = None; cls_std = None

        if cal is not None:
            p_forged, label = cal.predict(ev, cls_mean=cls_mean, cls_std=cls_std)
        else:
            p_forged = float(ev["seg_max_prob"])
            label = hard_rule_baseline(ev["label"], cls_mean if cls_mean is not None else 0.5,
                                       low=cfg["cls_override_low"], high=cfg["cls_override_high"])

        if label == 0:
            rle = create_zero_rle(h, w)
        else:
            if binary.sum() == 0:  # calibrator 强行翻转, mask 为空
                # 用 prob>较低阈值 的区域兜底
                fallback = (prob > 0.2).astype(np.uint8)
                fallback = postprocess_mask(fallback, morph_kernel_size=cfg["morph_kernel"],
                                            min_area=cfg["min_area"])
                rle = mask_to_rle(fallback) if fallback.sum() > 0 else create_zero_rle(h, w)
            else:
                rle = mask_to_rle(binary)

        results[name] = {
            "label": int(label),
            "rle": rle,
            "p_forged": float(p_forged),
            "evidence": ev,
        }
    return results


# ============================================================
# Stage 3: Qwen3.5-9B 解释生成
# ============================================================

def stage3_vlm_generate(stage2_results, cfg, device, logger, cache_file):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from evidence import evidence_to_prompt_block

    done = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            done = json.load(f)
        logger.info(f"[stage3] resumed {len(done)}/{len(stage2_results)}")

    remaining = [n for n in sorted(stage2_results.keys()) if n not in done]
    if not remaining:
        logger.info("[stage3] all done, skipping")
        return done

    vlm_dir = cfg["vlm_model"]
    base_dir = cfg["vlm_base"]
    has_adapter = os.path.exists(os.path.join(vlm_dir, "adapter_config.json"))
    has_full = os.path.exists(os.path.join(vlm_dir, "config.json"))

    if has_adapter and not has_full:
        # LoRA adapter 模式：基座 + adapter 合并
        from peft import PeftModel
        logger.info(f"[stage3] base={base_dir}  +  LoRA={vlm_dir}")
        processor = AutoProcessor.from_pretrained(vlm_dir, trust_remote_code=True)
        # 限制图像分辨率与训练时一致 (384²) 避免 OOM
        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "size") and ip.size is not None:
            try:
                ip.size.longest_edge = 384 * 384
            except Exception:
                pass
        # 多卡 device_map=auto, 5 卡训得起单卡也跑得起 (推理无激活+无优化器)
        n_gpu = torch.cuda.device_count()
        load_kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True,
                           attn_implementation="sdpa")
        if n_gpu > 1:
            load_kwargs["device_map"] = "auto"
        base = AutoModelForImageTextToText.from_pretrained(base_dir, **load_kwargs)
        model = PeftModel.from_pretrained(base, vlm_dir)
        model = model.merge_and_unload()
        if n_gpu == 1:
            model = model.to(device)
        model.eval()
    else:
        model_path = vlm_dir if has_full else base_dir
        logger.info(f"[stage3] loading VLM (full): {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "size") and ip.size is not None:
            try:
                ip.size.longest_edge = 384 * 384
            except Exception:
                pass
        n_gpu = torch.cuda.device_count()
        load_kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True,
                           attn_implementation="sdpa")
        if n_gpu > 1:
            load_kwargs["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        if n_gpu == 1:
            model = model.to(device)
        model.eval()

    system_prompt = (
        "你是专业的图像伪造鉴定专家。下面会给你一张图片以及一份"
        "由像素级取证模型(分割集成 + ELA + SRM)输出的【结构化证据】。"
        "请严格基于证据中的 bbox、面积占比、异常度比值进行论证，"
        "不要编造证据中未出现的坐标或区域。"
        "输出一段 300-600 字的连续中文鉴定文本，不使用分点、标题或换行。"
    )

    for name in tqdm(remaining, desc="  VLM", ncols=80):
        rec = stage2_results[name]
        ev = rec["evidence"]
        ev_for_prompt = dict(ev)
        ev_for_prompt["label"] = rec["label"]  # 注入校准后的 label
        block = evidence_to_prompt_block(ev_for_prompt)

        if rec["label"] == 1:
            user_prompt = (
                "请结合下方【结构化取证证据】对该图像进行伪造鉴定，输出 300-600 字的连续中文鉴定文本：\n\n"
                f"【证据】\n{block}\n\n"
                "要求：开头使用\"这是一份伪造的[内容简述]\"，文中引用证据中的 bbox 坐标 [x1,y1,x2,y2]，"
                "分析视觉异常(字体差异、边缘不自然、纹理断裂、光照不一致、JPEG 压缩伪影、像素噪声不匹配)与"
                "逻辑矛盾(数学计算、日期、品牌、上下文)。严禁输出证据中未提及的坐标。"
                "以\"综上所述，该图像系[伪造方式]，不具备[真实性/可信度]\"结尾。"
            )
        else:
            user_prompt = (
                "请结合下方【结构化取证证据】对该图像进行真实性论证，输出 300-600 字的连续中文鉴定文本：\n\n"
                f"【证据】\n{block}\n\n"
                "要求：开头使用\"这是一张真实拍摄的[内容简述]，未发现数字伪造或后期篡改的痕迹\"，"
                "从视觉一致性(字体统一、边缘过渡自然、纹理连续、光照均匀、噪点分布一致)、"
                "JPEG 压缩伪影分布均匀性、物理合理性(遮挡/透视/阴影)、信息准确性(品牌/数值/上下文)进行论证。"
                "以\"综合分析，该图像真实记录了[具体场景描述]\"结尾。"
            )

        img_path = os.path.join(cfg["test_dir"], name)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": user_prompt},
            ]},
        ]

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                    do_sample=cfg["do_sample"],
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            explanation = processor.tokenizer.decode(generated, skip_special_tokens=True)
            if "</think>" in explanation:
                explanation = explanation[explanation.index("</think>") + len("</think>"):].strip()
            if not explanation.strip():
                explanation = ("该图像经分析未发现明显异常。" if rec["label"] == 0
                               else "该图像经分析存在伪造痕迹。")
        except Exception as e:
            logger.error(f"[VLM] {name}: {e}")
            explanation = ("该图像经分析未发现明显异常。" if rec["label"] == 0
                           else "该图像经分析存在伪造痕迹。")

        done[name] = explanation
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(done, f, ensure_ascii=False, indent=2)

    del model
    torch.cuda.empty_cache()
    return done


# ============================================================
# Stage 4: 写 CSV
# ============================================================

def write_csv(stage2_results, explanations, output_path, logger):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "label", "location", "explanation"])
        for name in sorted(stage2_results.keys()):
            w.writerow([
                name,
                stage2_results[name]["label"],
                json.dumps(stage2_results[name]["rle"], ensure_ascii=False),
                explanations.get(name, ""),
            ])
    logger.info(f"[csv] wrote {len(stage2_results)} rows -> {output_path}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gpu", type=str, default=None,
                        help="单卡传 '3'；多卡推理传 '4,5,6' 启用 device_map=auto (VLM 大模型用)")
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache_dir", default=None,
                        help="覆盖 config.cache_dir (val 推理用独立缓存避免冲突)")
    parser.add_argument("--no_calibrator", action="store_true")
    parser.add_argument("--no_classifier", action="store_true")
    parser.add_argument("--no_convnext", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.gpu is not None: cfg["gpu"] = args.gpu
    if args.test_dir: cfg["test_dir"] = args.test_dir
    if args.output: cfg["output"] = args.output
    if args.cache_dir: cfg["cache_dir"] = args.cache_dir
    if args.no_calibrator: cfg["use_calibrator"] = False
    if args.no_classifier: cfg["use_classifier"] = False
    if args.no_convnext: cfg["use_convnext"] = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    # 干净 default device，多卡时第一个 visible GPU 即 cuda:0
    cache_dir = Path(cfg["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg["log_file"])
    logger.info("=" * 60)
    logger.info("Inference Pipeline (Qwen3.5-9B + Evidence)")
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    device = torch.device("cuda")

    # ---- Stage 1: seg ----
    seg_cache = cache_dir / "seg_outputs.npz"
    if seg_cache.exists():
        logger.info("[stage1] cache hit")
        blob = np.load(seg_cache, allow_pickle=True)
        seg_outputs = {k: {"prob": blob[k].item()["prob"],
                            "orig_size": blob[k].item()["orig_size"]}
                       for k in blob.files}
    else:
        t0 = time.time()
        seg_outputs = stage1_segmentation(cfg, device, logger)
        np.savez_compressed(seg_cache, **{k: np.array(v, dtype=object)
                                          for k, v in seg_outputs.items()})
        logger.info(f"[stage1] done {time.time()-t0:.1f}s")

    # ---- Stage 1.5: cls ----
    cls_cache = cache_dir / "cls_scores.json"
    if cfg["use_classifier"]:
        if cls_cache.exists():
            logger.info("[stage1.5] cache hit")
            with open(cls_cache, "r") as f:
                cls_scores = json.load(f)
        else:
            t0 = time.time()
            cls_scores = stage1_5_classifier(cfg, device, logger)
            with open(cls_cache, "w") as f:
                json.dump(cls_scores, f)
            logger.info(f"[stage1.5] done {time.time()-t0:.1f}s")
    else:
        cls_scores = {}

    # ---- Stage 2: evidence + calibrator ----
    s2_cache = cache_dir / "stage2_results.json"
    if s2_cache.exists():
        logger.info("[stage2] cache hit")
        with open(s2_cache, "r") as f:
            stage2_results = json.load(f)
    else:
        t0 = time.time()
        stage2_results = stage2_evidence_and_calibrate(seg_outputs, cls_scores, cfg, logger)
        with open(s2_cache, "w") as f:
            json.dump(stage2_results, f, ensure_ascii=False)
        logger.info(f"[stage2] done {time.time()-t0:.1f}s")

    labels = [r["label"] for r in stage2_results.values()]
    logger.info(f"  total={len(labels)}  real={labels.count(0)}  forged={labels.count(1)}")

    # ---- Stage 3: VLM ----
    expl_cache = cache_dir / "explanations.json"
    t0 = time.time()
    explanations = stage3_vlm_generate(stage2_results, cfg, device, logger, str(expl_cache))
    logger.info(f"[stage3] done {time.time()-t0:.1f}s ({len(explanations)} explanations)")

    # ---- Stage 4: CSV ----
    write_csv(stage2_results, explanations, cfg["output"], logger)
    logger.info(f"DONE -> {cfg['output']}")


if __name__ == "__main__":
    main()
