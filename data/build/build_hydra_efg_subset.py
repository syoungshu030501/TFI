#!/usr/bin/env python
"""
Build the HydraFake EFG-only subset with **synthetic Chinese 6-tag CoT
responses** for TFI Stage-A warmup.

Why this exists
---------------
data/build/build_v2_sft.py已经把 HydraFake 的 EFG fake / real 抽了进来，但是直接
保留了 HF 自带的 *English* assistant content (`r["messages"]`)，这会让
中文 SFT 训出来的模型在 EFG 数据上语言切换。本脚本输出 *TFI 中文* 模板：

  - system prompt: TFI 6-tag CoT (与 build_v2_sft 一致)
  - user prompt:   `<image>请判断该图像的真实性，并按规定标签格式输出分析。`
  - assistant:     根据 label / sub-generator hint 生成的中文 CoT 回答

Sub-generator 信息从 image path 路径解析 (e.g. `EFG/Dall-E1/img_644.png` →
"Dall-E"). 我们在 reasoning 里**显式提到 sub-generator 名**，让模型学到
"AI 全脸生成 = 数字伪造" 而不是死记某个 generator 的指纹.

Output shape (per sample) matches build_v2_sft.make_record() 约定：
  {"images": [abs path], "type": "hydra_efg_<gen>", "label": 1|0,
   "source": "hydrafake_efg_cn", "messages": [system, user, assistant]}

Usage:
    python -m data.build.build_hydra_efg_subset \
        --out /mnt/nfs/young/TFI/data/v2/hydra_efg_cn.json \
        --efg_limit 4000 --real_limit 4000

Defaults are sized for Stage-A warmup (1 epoch over ~8k samples ≈ 1 hr on 7×L20).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Optional

NFS = Path("/mnt/nfs/young/TFI")
HYDRA_SFT = NFS / "data/HydraFake/jsons/train/sft_36k.json"
HYDRA_IMG_ROOT = NFS / "data/HydraFake"

# System / user prompts must match data/build/build_v2_sft.py SYS_PROMPT_ZH /
# USR_PROMPT_ZH verbatim. Imported here so a future schema bump propagates.
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

# ---------------------------------------------------------------------------
# Sub-generator name extraction.  EFG paths look like:
#   hydrafake/train/fake/EFG/Dall-E1/img_644.png
#   hydrafake/train/fake/EFG/StyleGAN3/img_5.jpg
# Real paths:
#   hydrafake/train/real/celebahq/img_3.png
# ---------------------------------------------------------------------------
_SUB_GEN_RE = re.compile(r"/(?:fake|real)/(?:[^/]+/)?([^/]+)/[^/]+\.(?:jpg|png|jpeg)$", re.IGNORECASE)


def parse_sub_generator(path: str) -> str:
    m = _SUB_GEN_RE.search(path)
    return m.group(1) if m else "未知来源"


# ---------------------------------------------------------------------------
# Chinese CoT templates — keep variety high so RL rollouts don't memorise.
# Each template is a function: (sub_gen) -> (fast, reasoning, conclusion).
# Conclusion is REGION-only (no bbox) since EFG is whole-face — there is no
# meaningful localisation bbox for "the entire generated face".
# ---------------------------------------------------------------------------
FAKE_TEMPLATES = [
    lambda gen: (
        f"这张人像的整体风格高度一致，但缺乏自然人脸应有的微表情与皮肤瑕疵，疑似 AI 全脸生成。",
        f"细看皮肤纹理过度平滑，毛孔、细纹等高频细节大面积缺失，这是 {gen} 等扩散/GAN 模型的典型特征。"
        f"再观察五官——眼神焦点对称得过于规整，瞳孔反光位置完全镜像，符合生成模型在对称约束下的统计偏置。"
        f"耳部与发际线衔接处出现轻度模糊伪影，背景虚化的边界被人脸轮廓"
        f"过渡得不自然。综上，这张图像在纹理统计与几何对称两个维度均偏离真实人脸分布。",
        f"综合上述纹理伪影与对称性异常，可判定为 AI 整脸生成图像。"
        f"<region>整张人脸区域</region>",
    ),
    lambda gen: (
        f"画面构图自然，但人物面部存在数字生成的几个共性破绽。",
        f"主要疑点集中在三处：其一，皮肤反光呈整体一致的柔光，缺少真实拍摄环境下"
        f"由皮肤油脂与汗水带来的局部高光；其二，发丝与背景的交界出现细碎噪声，"
        f"是 {gen} 一类模型在高频细节上常见的"
        f"重建误差；其三，牙齿边缘或眼角细节存在亚像素错位。"
        f"逐项排查后，这些迹象共同指向 AI 整脸合成。",
        f"基于上述多重视觉证据，该图为伪造的 AI 生成人脸。"
        f"<region>整张面部及发际线区域</region>",
    ),
    lambda gen: (
        f"这是一张视觉上颇为流畅的人像，但仔细看后判断为 AI 生成。",
        f"<planning> 我会从皮肤纹理、对称性、光照一致性、背景衔接四个角度依次核查。 </planning>"
        f"皮肤纹理过滑，缺乏真实的皮下血色变化；左右脸特征几乎严格对称，违背真实人脸的微小不对称；"
        f"光照方向在面部与脖颈/背景之间存在轻微矛盾；"
        f"发际、耳廓与背景的过渡带含有 {gen} 模型典型的色彩泄漏。"
        f"<reflection> 上述四点同时成立，单一指标可解释为成像差异，但叠加则指向生成。 </reflection>",
        f"综合考虑，这张人脸属于 AI 整脸生成内容。"
        f"<region>面部主体区域</region>",
    ),
]

REAL_TEMPLATES = [
    lambda gen: (
        f"该图像呈现真实拍摄的人像特征，未发现数字伪造痕迹。",
        f"皮肤具有自然的纹理变化，毛孔、痘印、细纹分布随机；面部不完全对称，"
        f"鼻梁、唇形、眉毛长度均存在生理上的轻微差异；光照方向与背景一致，"
        f"环境反光在眼球高光、皮肤油脂高光上呈现合理梯度。整体证据链指向真实拍摄图像。",
        f"综合纹理细节、对称性偏差与光照一致性，可判定为真实图像。",
    ),
    lambda gen: (
        f"图像看起来是一张未经处理的真实人像。",
        f"逐区域检查：发丝细节清晰可分，单根发丝与肤色背景过渡自然，"
        f"未见 AI 生成常见的高频细碎噪声；瞳孔反光位置左右不完全对称，"
        f"符合真实拍摄中相机与光源的几何关系；服装与皮肤结合部位的"
        f"阴影方向一致；图像 EXIF 风格的高斯噪声分布在前景与背景一致，"
        f"未发现 AI 生成的纹理"
        f"统计偏移。",
        f"图像各项视觉特征均符合真实拍摄人像，无数字伪造证据。",
    ),
    lambda gen: (
        f"凭直观感受这是真实图像，没有 AI 生成的"
        f"塑料感。",
        f"细看，皮肤反映出自然的次表面散射效果，光线在面颊与鼻尖产生柔和的红色透光；"
        f"眉毛与睫毛的密度、长度均存在自然差异；眼角与嘴角的细小阴影"
        f"层次分明，与 GAN/Diffusion 模型常见的过度平滑形成对比；"
        f"背景物体与人脸的景深关系合理，无 AI 合成常见的几何错位。",
        f"基于上述特征链，判定为真实拍摄图像。",
    ),
]


def _pick_template(label: int, rng: random.Random):
    return rng.choice(FAKE_TEMPLATES if label == 1 else REAL_TEMPLATES)


def synthesize_assistant(label: int, sub_gen: str, rng: random.Random) -> str:
    fast, reasoning, conclusion = _pick_template(label, rng)(sub_gen)
    answer = "fake" if label == 1 else "real"
    return (
        f"<fast> {fast} </fast>\n"
        f"<reasoning> {reasoning} </reasoning>\n"
        f"<conclusion> {conclusion} </conclusion>\n"
        f"<answer>{answer}</answer>"
    )


def build(efg_limit: int, real_limit: int, seed: int = 42) -> List[dict]:
    data = json.load(open(HYDRA_SFT, "r", encoding="utf-8"))
    rng = random.Random(seed)

    fake_efg = [r for r in data if r.get("type") == "entire face generation"]
    real_pool = [r for r in data if r.get("type") == "real"]
    rng.shuffle(fake_efg)
    rng.shuffle(real_pool)
    if efg_limit > 0:
        fake_efg = fake_efg[:efg_limit]
    if real_limit > 0:
        real_pool = real_pool[:real_limit]

    out: List[dict] = []
    n_skipped = 0
    for r in fake_efg + real_pool:
        rel = r["images"][0]
        abs_p = HYDRA_IMG_ROOT / rel
        if not abs_p.exists():
            n_skipped += 1
            continue
        sub_gen = parse_sub_generator(rel)
        label = int(r["label"])
        assistant = synthesize_assistant(label, sub_gen, rng)
        out.append({
            "images": [str(abs_p)],
            "type": f"hydra_efg_{sub_gen}",
            "label": label,
            "source": "hydrafake_efg_cn",
            "messages": [
                {"role": "system", "content": SYS_PROMPT_ZH},
                {"role": "user", "content": USR_PROMPT_ZH},
                {"role": "assistant", "content": assistant},
            ],
        })
    rng.shuffle(out)
    print(f"[hydra_efg_cn] kept={len(out)} skipped(missing_image)={n_skipped} "
          f"(efg_limit={efg_limit}, real_limit={real_limit})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=NFS / "data/v2/hydra_efg_cn.json")
    ap.add_argument("--efg_limit", type=int, default=4000,
                    help="cap on EFG fake samples (0 = no cap)")
    ap.add_argument("--real_limit", type=int, default=4000,
                    help="cap on HydraFake real samples (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = build(args.efg_limit, args.real_limit, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=None)
    print(f"wrote {args.out}  total={len(recs)}")


if __name__ == "__main__":
    main()
