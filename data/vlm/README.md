# VLM SFT 数据入口

`train_qwen35_9b.py` 默认从这三个目录拼训练集：

| flag | 默认 | 内容 |
|---|---|---|
| `--data_dir` | `data/raw/train_resume` | 主样本（用 Caption_clean） |
| `--augmented_dir` | `data/vlm/caption_api_v3` | 强 API 蒸馏的 evidence-caption（jsonl） |
| `--real_ext_dir` | `data/processed/real_ext` | 真实图扩充（带模板 caption） |

## caption_api_v3 字段

由 [scripts/data/regen_caption_api.py](../../scripts/data/regen_caption_api.py)
（qwen-vl-max via DashScope）生成，每行：

```json
{
  "image_path": "data/raw/train_resume/Black/Image/xxx.png",
  "mask_path":  "data/raw/train_resume/Black/Mask/xxx.png",
  "stem": "xxx", "version": 0, "temperature": 0.8,
  "gt_label": 1,
  "evidence": {...},
  "caption": "...",
  "validation_mode": "strict",
  "model": "qwen-vl-max-2025-xx-xx"
}
```

格式与 `caption_local_v2`（旧本地 Qwen3.5-9B）100% 兼容，因此 `VLMSFTDataset` 切换数据源不需要改代码。

## 为什么换 API？

- 本地 Qwen3.5-9B：3 分片并行各占一份完整 BF16 权重，频繁 OOM；详见 logs/data/aug_validation.md
- qwen-vl-max：远端推理，0 显存占用；视觉理解显著更强，能减少长度越界（v2 中 49/902 越界）
- 成本可控：~640 张 × 2 版本 ≈ 1280 次调用，预估 ¥30-40
