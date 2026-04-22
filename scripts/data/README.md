# 数据 pipeline 脚本

按 `S0 → S1 → S2 → S3 → guard` 的顺序排，每个阶段产物落到 `data/` 对应位置。

| 阶段 | 入口 | 输入 | 产物 | 备注 |
|---|---|---|---|---|
| S0 切分 | [../../split_train_val.py](../../split_train_val.py) | `data/raw/train` | `train_split/` + `val/` | 一次性，已完成 |
| S1 caption 清洗 | [../../tools/clean_captions.py](../../tools/clean_captions.py) | Black/Caption + Mask | `Black/Caption_clean/` | 已完成（800/800） |
| S1 caption 审计 | [../../tools/check_caption_bbox.py](../../tools/check_caption_bbox.py) | 同上 | `logs/data/caption_bbox_audit*.csv` | 已完成 |
| S2 合成伪造 | [../../tools/synth_forgery.py](../../tools/synth_forgery.py) | `White/Image` | `data/processed/synth/{Image,Mask,meta.jsonl}` | 已完成（750 张） |
| S2 合成过滤 | [../../tools/filter_synth_by_seg.py](../../tools/filter_synth_by_seg.py) | synth + seg ckpt | `data/processed/synth/keep.txt` | 已完成（62 keep） |
| S3 真实图扩充 | [../../tools/expand_real_images.py](../../tools/expand_real_images.py) | `train/White` + COCO val2017 | `data/processed/real_ext/{Image,Caption}` | 已完成（1100 张） |
| S4 caption 重生（旧） | [../../tools/regen_evidence_captions.py](../../tools/regen_evidence_captions.py) | 本地 Qwen3.5-9B | `data/processed/caption_local_v2/*.jsonl` | **已废弃**，OOM 严重 |
| **S4 caption 重生（新）** | [regen_caption_api.py](regen_caption_api.py) | qwen-vl-max API | `data/vlm/caption_api_v3/*.jsonl` | **当前推荐** |
| guard | [guard.py](guard.py) | data/ 全量 | `data/meta/data_health.md` | 任何阶段后跑一次 |

## 一键体检

```bash
python scripts/data/guard.py            # 写报告
python scripts/data/guard.py --strict   # CI 用，硬契约失败 exit 1
```

## 一键重生 caption（API）

```bash
export DASHSCOPE_API_KEY=sk-xxx          # 阿里云 DashScope key
# 默认：仅补 caption_local_v2 中失败/缺失的 stem
python scripts/data/regen_caption_api.py
# 全量重生 640 stem × 2 版本（成本高）
python scripts/data/regen_caption_api.py --mode full --workers 6
```

依赖：`pip install openai`（DashScope OpenAI 兼容模式）。
