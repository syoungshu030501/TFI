# TFI — 证据驱动的图像伪造鉴别

电商竞赛三任务：**伪造判别 / 伪造定位 / 可解释分析**。在 1000 张训练图（Black 800 = 伪造，
White 200 = 真实）+ 测试集上跑：分割集成 → 结构化证据抽取 → 轻量校准器 → Qwen3.5-9B 证据驱动微调。

输出 `submit.csv`：`image_name, label, location, explanation`。

本 README 按以下顺序展开：

- **Part 1 — 数据工程**：原料、处理、消费者三层入口（重点）
- **Part 2 — 训练阶段总览**：每阶段的输入/损失/产物
- **Part 3 — 评估与指标**
- **Part 4 — 项目结构与配置**
- **Part 5 — 快速开始**
- **Part 6 — 已知限制 + 下一步实验计划**

> 旧方案（397B 教师 → 8B 学生 CoT 蒸馏）已归档至 [legacy/](legacy/README.md)，仅作历史回溯。

---

## Part 1 · 数据工程

数据来源、清洗、合成、扩充全部走 `data/` 这一个入口，**按消费者**而非来源切。
所有比特仍存在 `/mnt/nfs/young/my_dt/True-or-Fake-Image-main/`，`data/` 用 symlink 组织。

### 1.1 最终数据现状

| 层 | 路径 | 数量 | 用途 | 来源 |
|---|---|---|---|---|
| raw | [data/raw/train_resume](data/raw/train_resume) | Black 640 + White 160 | seg/cls/VLM 主训练 | `train_split` ∪ `val` 的稳定枚举视图 |
| raw | [data/raw/val](data/raw/val) | Black 160 + White 40 | calibrator 拟合 + ablation | 8:2 切分的 val |
| raw | [data/raw/test](data/raw/test) | 500 张 (Image only) | 推理输入 | 竞赛测试集 |
| processed | [data/processed/synth](data/processed/synth) | 750 生成 / 62 keep | seg/cls 像素级合成正例 | copy_move / splicing / text_replace_like |
| processed | [data/processed/real_ext](data/processed/real_ext) | 1100 张 + 1100 caption | seg/cls 负例补齐 + VLM 真实样本 | White 强增广 + COCO val2017 抽样 |
| processed | [data/processed/caption_local_v2](data/processed/caption_local_v2) | 902 行 jsonl | **已废弃**（OOM、49 越界） | 本地 Qwen3.5-9B 三分片并行生成 |
| **vlm** | [data/vlm/caption_api_v3](data/vlm/caption_api_v3) | 0 / 1280 行（待生成） | VLM SFT 主增强 caption | qwen-vl-max via DashScope |

**核心硬契约**（[scripts/data/guard.py](scripts/data/guard.py)，输出 [data/meta/data_health.md](data/meta/data_health.md)）：

- raw 各子集行数达准入线（train_resume Black ≥ 600, White ≥ 100；val Black ≥ 100, White ≥ 30；test ≥ 100）
- processed/synth Image == Mask 数量 + `keep.txt` 存在
- processed/real_ext Image == Caption 数量
- 所有 symlink 解析后真实存在（`is_link_alive` 检查）
- vlm/caption_api_v3 ≥ 90% target（API 蒸馏完成的准入线）

### 1.2 三层数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase A · 一次性原料采集 + 切分                                      │
│    split_train_val.py → train_split/ + val/                          │
│    train_resume = train_split ∪ val 的稳定视图（绕开 NFS 枚举卡死）   │
│  → data/raw/{train_resume, val, test}                                │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase B · caption 清洗（仅 Black）                                   │
│    tools/check_caption_bbox.py → logs/data/caption_bbox_audit*.csv   │
│    tools/clean_captions.py     → Black/Caption_clean/                │
│  - IoU≥0.5: 复制原文，仅清 </think>                                  │
│  - 0.2≤IoU<0.5: 用 GT bbox 重写 caption 中的最近 bbox                │
│  - IoU<0.2 或无 bbox: 占位 + 写入 needs_regen.txt 给 Phase D         │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase C · 像素增强（仅 seg/cls 用）                                  │
│    tools/synth_forgery.py        → data/processed/synth/             │
│    tools/filter_synth_by_seg.py  → keep.txt（用 fold0 ckpt 反向过滤）│
│    tools/expand_real_images.py   → data/processed/real_ext/          │
│  - synth 必经 0.10 < mean_iou < 0.90 过滤后才 keep（750 → 62）       │
│  - real_ext = White JPEG 重压缩/调色 + COCO val2017 抽样             │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase D · evidence-caption 重生（VLM 用）                            │
│    旧: tools/regen_evidence_captions.py（本地 9B，3 分片 OOM）        │
│         → data/processed/caption_local_v2/*.jsonl                    │
│    新: scripts/data/regen_caption_api.py（qwen-vl-max API）          │
│         → data/vlm/caption_api_v3/*.jsonl                            │
│  - extract_from_gt_mask 抽证据 → evidence_to_prompt_block 拼提示     │
│  - 每 stem 出 N 个温度版本（默认 [0.8, 1.0]）                        │
│  - strict 校验: bbox 必须 ∈ GT 集合 + 长度 + 开头模板 + 结尾"综上"   │
│  - loose 回退: 仅校验长度 + bbox 不越界                              │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase E · 全量体检                                                   │
│    scripts/data/guard.py → data/meta/data_health.md                  │
│    任一硬契约失败 → guard --strict 退出码 1                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 caption_local_v2 的问题（为什么换 API）

旧的 `tools/regen_evidence_captions.py` 走本地 Qwen3.5-9B：

| 问题 | 现象 | 根因 |
|---|---|---|
| 显存爆炸 | 3 个 shard 并行 → 每卡一份完整 BF16 9B 权重，OOM 频繁（见旧 `train_v2_shard*.log`） | `from_pretrained` × 3 进程，无权重共享 |
| 长度越界 | 902 条中 49 条 > 800 字（`logs/data/aug_validation.md`） | base 模型遵从指令能力弱 |
| 视觉理解弱 | strict 通过率不达标，需要 loose 回退 | 9B 远小于 max 版 |

新的 `scripts/data/regen_caption_api.py` 走 qwen-vl-max：

| 优势 | 说明 |
|---|---|
| 0 显存 | 远端推理，本地只跑 IO 和 evidence 抽取 |
| 视觉强 | qwen-vl-max 是 Qwen 视觉系列最强档 |
| 字段兼容 | 输出 schema 与 v2 一致，VLMSFTDataset 切换零代码改动 |
| 成本可控 | 640 stem × 2 版本 ≈ 1280 次调用，约 ¥30-40 |
| Resume-safe | 按 (stem, version) 去重，支持多次断点续跑 |
| 默认 missing-only | 仅补 v2 中失败/缺失的 stem，避免重复花钱 |

---

## Part 2 · 训练阶段总览

| 阶段 | 入口 | 输入 | 损失 / 目标 | 产物 |
|---|---|---|---|---|
| 0 数据 | [scripts/data/](scripts/data/) | NFS 原料 | — | `data/{raw,processed,vlm}` |
| 1 分割 | [train_seg_ensemble.py](train_seg_ensemble.py) | `data/raw/train_resume` + `data/processed/{synth,real_ext}` | Dice + Focal + Boundary（7 通道 RGB+ELA+SRM） | `checkpoints/seg/{arch}_fold{0-4}/` |
| 2 分类 | [train_classifier.py](train_classifier.py) | 同上 | CE（6 通道 RGB+ELA） | `checkpoints/cls/efficientnet_fold{0-4}/` |
| 3 校准 | [train_calibrator.py](train_calibrator.py) | `data/raw/val` + seg/cls 推理 | 5-fold CV 多 backend：`logistic / xgb / lgbm_mono / tabpfn / ebm`，OOF 选阈值 | `checkpoints/calibrator/{calibrator.pkl,metrics.json,compare.md}` |
| 4 VLM SFT | [train_qwen35_9b.py](train_qwen35_9b.py) | `data/raw/train_resume` + `data/vlm/caption_api_v3` + `data/processed/real_ext` | LoRA r=64 + evidence prompt 注入 | `checkpoints/qwen35_9b/` |
| 5 推理 | [inference.py](inference.py) | `data/raw/test/Image` | seg ensemble + calibrator + VLM | `submit.csv` |

**Stage 1 细节**
- 三架构：SegFormer-B5 / ConvNeXt-V2-Large / MaxViT-Large
- 默认 5-fold（`dataset.create_kfold_splits`，按 Black/White 分层）
- **当前仅 fold0 有合格 ckpt（IoU 0.8370，[logs/seg/segformer_fold0.log](logs/seg/segformer_fold0.log)）**
- fold1/2 之前的训练崩了或 IoU 偏低（已删 ckpt 与日志），等数据 pipeline 收敛后用同一份增强数据重训

**Stage 4 细节**
- LoRA r=64, alpha=128, target_modules 自动发现（排除 vision tower）
- `inject_evidence=True`：训练时用 GT mask 抽证据拼到 user prompt，与推理时的"用预测 mask 抽证据"格式严格对齐
- 数据 = `Caption_clean`（800 主样本）+ `caption_api_v3`（≈1280 行增强）+ `real_ext`（1100 行真实）
- 单卡 RTX 5090 / L20 (46GB) batch=1 + grad_accum=16 可跑

---

## Part 3 · 评估与指标

入口：[evaluate.py](evaluate.py)，输入 `--val_dir data/raw/val`。

**指标集合**：
- `seg_iou / seg_dice`：分割任务对 Black/Mask
- `label_f1 / precision / recall`：判别任务（label∈{0,1}）
- `vlm_bbox_hallucination_rate`：VLM 输出中 bbox 与 GT 不匹配的比例
- `caption_format_acc`：开头/结尾模板符合率

**消融维度**（`evaluate.py --full_ablation` 写到 `logs/data/ablation.md`）：
- `seg_arch`: `all / segformer_only / maxvit_only / segformer_maxvit`
- `calibrator`: `seg_only / hard / logistic / xgb / lgbm_mono / tabpfn / ebm`
- `cls`: `on / off`
- `tta`: `on / off`
- `multiscale`: `on / off`

---

## Part 4 · 项目结构与配置

```
TFI/
├── README.md                            ← 本文档
├── PROJECT.md                           # 详细设计（旧版，部分信息可能与 README 不同步）
├── DATA_AUGMENTATION.md                 # 增强方法论（旧版深入）
│
├── data/                                ← ★ 统一数据入口
│   ├── README.md                        # 数据契约总览
│   ├── raw/{train_resume, train, val, test}/      # symlink → NFS
│   ├── processed/{synth, real_ext, caption_local_v2}/
│   ├── seg/README.md                    # 消费者口径
│   ├── cls/README.md
│   ├── vlm/{README.md, caption_api_v3/} # API 蒸馏目标
│   └── meta/{data_health.md, README.md}
│
├── scripts/
│   └── data/                            # 数据 pipeline 脚本（新风格）
│       ├── README.md
│       ├── guard.py                     # 全量硬契约体检
│       └── regen_caption_api.py         # qwen-vl-max evidence-caption 重生
│
├── tools/                               # 历史数据脚本（仍在用）
│   ├── check_caption_bbox.py            # caption bbox 审计
│   ├── clean_captions.py                # caption 清洗
│   ├── synth_forgery.py                 # 合成伪造
│   ├── filter_synth_by_seg.py           # 用 seg ckpt 反向过滤合成
│   ├── expand_real_images.py            # 真实图扩充
│   ├── regen_evidence_captions.py       # 旧本地 9B 重生（已被 scripts/data 替代）
│   └── validate_augmented_data.py       # 老版增强验证
│
├── logs/                                # 按消费者拆分
│   ├── seg/segformer_fold0.log          # IoU 0.8370，留作 baseline
│   ├── vlm/                             # qwen35_9b 训练日志（待生成）
│   └── data/                            # 数据 pipeline 历史报表
│       ├── aug_validation.md
│       ├── caption_bbox_audit{,_split}.csv
│       ├── evidence_caption_preflight_bf16.jsonl
│       └── filter_synth_cpu.log
│
├── dataset.py                           # ForgerySegDataset / ClsDataset / VLMSFTDataset
├── evidence.py                          # 结构化证据抽取（10 维特征）
├── calibrator.py                        # 多 backend 校准器（5-fold CV + isotonic + monotonic）
├── vlm_collator.py                      # Qwen3.5/Qwen3-VL collator + LoRA target
├── utils.py                             # ELA / SRM / RLE / 指标
│
├── train_seg_ensemble.py                # Stage 1：3 arch × 5 fold
├── train_classifier.py                  # Stage 2：EfficientNet 5 fold
├── train_calibrator.py                  # Stage 3：多 backend + 5-fold CV + 自动 backend 选优
├── train_qwen35_9b.py                   # Stage 4：Qwen3.5-9B LoRA
├── inference.py                         # Stage 5：证据驱动推理流水线
├── evaluate.py                          # 消融评估
├── split_train_val.py                   # 一次性切分
│
├── config.yaml                          # 推理/评估统一配置
├── run_pipeline.sh                      # 一键 pipeline
├── requirements.txt
│
├── checkpoints/                         # symlink → NFS
│   └── seg/segformer_fold0/             # 当前唯一合格 seg ckpt
├── models/Qwen3.5-9B/                   # symlink → NFS（ModelScope 下载）
├── cache/                               # symlink → NFS
└── legacy/                              # 397B 蒸馏链路归档（read-only）
```

**config.yaml 关键路径**（已统一到 `data/raw/`）：
```yaml
test_dir: data/raw/test/Image
val_dir:  data/raw/val
train_dir: data/raw/train_resume
log_file: logs/vlm/inference.log
```

---

## Part 5 · 快速开始

### 5.1 环境

```bash
conda create -n TFI python=3.11 -y
conda activate TFI
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install openai      # scripts/data/regen_caption_api.py 需要
```

硬件假设：1-2 卡 NVIDIA L20 (46GB) / RTX 5090，CUDA 12.4。

### 5.2 数据：补齐 / 体检

```bash
# 检查 data/ 入口、计数、symlink 是否齐
python scripts/data/guard.py
# CI 模式（任一硬契约失败 exit 1）
python scripts/data/guard.py --strict

# 用 qwen-vl-max API 重生 evidence-caption（默认仅补 v2 缺失/失败的）
export DASHSCOPE_API_KEY=sk-xxx
python scripts/data/regen_caption_api.py
# 全量重生（成本更高但起点干净）
python scripts/data/regen_caption_api.py --mode full --workers 6
```

### 5.3 训练

```bash
# 全 pipeline（guard → calibrator → VLM → inference）
bash run_pipeline.sh

# 或分步：
# Stage 1：SegFormer fold0 重训（其它 fold 同理）
python train_seg_ensemble.py --arch segformer --fold 0 --gpu 0 \
    2>&1 | tee logs/seg/segformer_fold0.log

# Stage 2：分类 5 fold
python train_classifier.py --fold all --gpu 1 \
    2>&1 | tee logs/seg/classifier.log

# Stage 3：校准器（默认 data/raw/val，5-fold CV）
# 默认 backend=tabpfn（2026 SOTA，需 pip install 'tabpfn>=2.5'）
python train_calibrator.py --backend tabpfn --cv_folds 5 --gpu 0
# 或一次性跑全 backend 对比（写 checkpoints/calibrator/compare.md，自动选 OOF F1 最高的）
python train_calibrator.py --compare_all --cv_folds 5 --gpu 0

# Stage 4：Qwen3.5-9B LoRA
python train_qwen35_9b.py --gpu 0 --epochs 4 --batch_size 1 --grad_accum 16 \
    2>&1 | tee logs/vlm/train_qwen35_9b.log

# Stage 5：推理 + submit.csv
python inference.py --config config.yaml --gpu 0
```

### 5.4 消融

```bash
python evaluate.py --full_ablation --gpu 0       # 写到 logs/data/ablation.md
python evaluate.py --seg_arch segformer_only --calibrator xgb --multiscale --gpu 0
```

---

## Part 6 · 已知限制 + 下一步实验计划

### 6.1 当前现状

| 阶段 | 状态 | 备注 |
|---|---|---|
| Stage 1 seg fold0 | ✅ IoU 0.8370 | 唯一合格 ckpt，作为后续 fold 的初始化参考 |
| Stage 1 seg fold1-4 | ❌ ckpt 已删 | 之前 OOM/IoU 0.44-0.59 不达标，待数据 pipeline 收敛后重训 |
| Stage 1 ConvNeXt / MaxViT | ❌ 未训练 | 双架构集成是简历主线，需补 |
| Stage 2 cls 全部 fold | ❌ 未训练 | `checkpoints/cls/` 不存在 |
| Stage 3 calibrator | ⚠️ 代码就绪，未拟合 | 已重构为 5-fold CV + 多 backend（logistic/xgb/lgbm_mono/tabpfn/ebm），等 stage 1+2 ckpt 齐 |
| Stage 4 caption_local_v2 | ⚠️ 902 行（49 越界） | **已弃用**，等 caption_api_v3 替换 |
| Stage 4 caption_api_v3 | ⏳ 0/1280 行 | 等用户跑 `scripts/data/regen_caption_api.py` |
| Stage 4 qwen35_9b ckpt | ❌ 未训练 | 等 caption_api_v3 |

### 6.2 下一步实验计划（按优先级）

#### P0 · 用强 API 把 caption 数据补到准入线（数据先行）

```bash
export DASHSCOPE_API_KEY=sk-xxx
# 默认 missing_only：只补 caption_local_v2 中失败/缺失的 stem
python scripts/data/regen_caption_api.py --workers 6 \
    2>&1 | tee logs/data/regen_caption_api.log
python scripts/data/guard.py --strict
```

成功标准：`data/vlm/caption_api_v3/*.jsonl` 累计 ≥ 1152 行（90% 准入线），strict 通过率 ≥ 70%（v2 baseline ≈ 9%）。

#### P1 · 用新数据重训 seg 5 fold（fold0 已是 baseline）

```bash
# 多卡并行，一卡一 fold
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$f python train_seg_ensemble.py \
      --arch segformer --fold $f --gpu 0 \
      2>&1 | tee logs/seg/segformer_fold${f}.log &
done
wait
```

对照实验：fold0 用 `--include_synth + --include_real_ext` vs 不用，对比 IoU 提升幅度（之前 fold0 是裸 800 张，IoU 0.8370）。

#### P2 · 训 ConvNeXt + MaxViT 双架构，做 3-arch ensemble

简历主线是"SegFormer-B5 / MaxViT-Large 双架构 5 折集成"，目前只跑了 SegFormer。补：

```bash
python train_seg_ensemble.py --arch convnext --fold all --gpu 0
python train_seg_ensemble.py --arch maxvit   --fold all --gpu 1
```

#### P3 · 训 cls 5 fold + calibrator + 推理一遍

```bash
python train_classifier.py --fold all --gpu 0
# calibrator：先全 backend 对比，挑 OOF F1 最高的写 calibrator.pkl
python train_calibrator.py --compare_all --cv_folds 5 --gpu 0
python inference.py --config config.yaml --gpu 0
```

#### P3.5 · Calibrator backend 消融（n≈200, d=9 的小表）

旧 baseline 是裸 XGBoost，问题是 (a) 200 行样本对 200 棵树严重过参；(b) 阈值在训练集上偏估；(c) 输出概率未校准。重构后：

| backend | 何时用 | 依赖 |
|---|---|---|
| `logistic` | n<300 默认强 baseline，概率最干净 | sklearn |
| `xgb` | 旧基线对照 | xgboost |
| `lgbm_mono` | 加单调约束 (`seg_conf↑/area↑/anomaly↑ → P↑`) + isotonic 校准 | lightgbm |
| `tabpfn` | **2026 SOTA**：n≤10K 表数据对 XGBoost 100% 胜率（非商业许可，权重 ~140MB） | `tabpfn>=2.5` |
| `ebm` | 全局 + 交互项可解释，写 ablation 表用 | interpret |

跑法（一条命令出对比表）：
```bash
python train_calibrator.py --compare_all --cv_folds 5 \
    2>&1 | tee logs/data/calibrator_ablation.log
# -> checkpoints/calibrator/compare.md   全 backend OOF F1/AUC/Brier/LogLoss
# -> checkpoints/calibrator/metrics.json 当选 backend 的 per-fold 报告
```

成功标准（vs 旧 XGBoost baseline）：
- OOF F1 ≥ baseline + 1.0pp
- Brier 分数下降 ≥ 5%（说明概率校准更好）
- 阈值 std across 5 folds ≤ 0.05（稳定，没在 noise 上 overfit）

#### P4 · Qwen3.5-9B LoRA SFT（在 caption_api_v3 上）

```bash
python train_qwen35_9b.py --gpu 0 --epochs 4 \
    --augmented_dir data/vlm/caption_api_v3 \
    2>&1 | tee logs/vlm/train_qwen35_9b.log
```

对照：用 `--augmented_dir data/processed/caption_local_v2`（旧本地 9B 数据）跑一次，测 caption 质量提升对下游 hallucination_rate / format_acc 的影响。

#### P5 · 端到端消融

```bash
python evaluate.py --full_ablation --gpu 0
```

成功标准：`logs/data/ablation.md` 上每加一个模块（cls/tta/multiscale/calibrator/vlm-evidence），label_f1 单增 ≥ 0.5pp。

### 6.3 风险与已知坑

| 风险 | 缓解 |
|---|---|
| `data/raw/train` 的 NFS 目录枚举偶发卡死 | 训练默认走 `data/raw/train_resume`，禁用 train/ |
| `caption_api_v3` 调用频次受 DashScope 限流 | `--workers ≤ 6`，retries=3 + 指数退避，已内置 |
| `qwen-vl-max` 输出中 bbox 仍可能出 GT 集合 | strict→loose 双层校验，loose 也校 bbox 白名单 |
| seg fold0 是 fold0/5 切分，重训其它 fold 时 val 子集不一致 | 5 个 fold 各自的 val 集互斥，用同一份 `data/raw/train_resume` |

### 6.4 删除/归档清单（本次重构）

| 项 | 原因 |
|---|---|
| `gen_v4.py / gen_v5.py / gen_v5_hybrid.py` | 旧 VLM 生成对照脚本，被 `inference.py` 替代 |
| `logs/seg_segformer_fold1_gpu{0,6}_bs2.log`, `fold2_gpu4_bs2.log` | OOM / IoU 0.44-0.59，不达标 |
| `logs/train_v2_shard{0,1,2}.log` | 旧本地 9B 生成日志，链路已废弃 |
| `checkpoints/seg/segformer_fold{1,2}` | 同上，重训 |
| `augmented_data/train_legacy/` | 旧 397B 蒸馏 caption，未启用 |

---

## 相关文档

- 详细设计：[PROJECT.md](PROJECT.md)
- 增强方法论历史版：[DATA_AUGMENTATION.md](DATA_AUGMENTATION.md)
- 数据入口契约：[data/README.md](data/README.md)
- 数据 pipeline 脚本：[scripts/data/README.md](scripts/data/README.md)
- 数据健康报告：[data/meta/data_health.md](data/meta/data_health.md)
- 旧 397B 链路：[legacy/README.md](legacy/README.md)
