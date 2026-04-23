# TFI · 证据驱动的图像伪造鉴定

电商竞赛三任务一体化方案：**伪造判别 (Detection) / 伪造定位 (Grounding) / 可解释分析 (Explanation)**。
1000 张训练图（Black 800 = 伪造，White 200 = 真实）+ 200 张验证 + 500 张测试。

> **官方评测复现 · S_Fin = 0.9034 / 1.0** （val 200 张，按官方公式）

| 子项 | 值 | 权重 | 贡献 |
|---|---:|---:|---:|
| **S_Det** (image-level F1) | **0.9845** | 0.45 | 0.4430 |
| **S_Loc** (pixel-level F1 / Dice) | **0.8735** | 0.25 | 0.2184 |
| ┃ S_Sim (BERTScore-zh) | 0.7552 | — | — |
| ┃ S_Auto (Qwen3-MAX rubrics) | 0.8582 | — | — |
| **S_Exp** = 0.5·Sim + 0.5·Auto | **0.8067** | 0.30 | 0.2420 |
| **S_Fin** | — | — | **🏆 0.9034** |

完整报告见 [logs/score_official.md](logs/score_official.md)、[logs/score_official.json](logs/score_official.json)。

---

## 目录

- [一、整体架构](#一整体架构)
- [二、数据工程](#二数据工程)
- [三、训练阶段](#三训练阶段)
- [四、推理流水线](#四推理流水线)
- [五、官方评测复现](#五官方评测复现)
- [六、快速开始 / 复现](#六快速开始--复现)
- [七、项目结构](#七项目结构)
- [八、关键设计与踩坑记录](#八关键设计与踩坑记录)

---

## 一、整体架构

```
                 ┌────────────────────────────────────────┐
                 │  Stage1  SegFormer-B5 × 5-fold ensemble │
   image (RGB) ──┤        (输入 7 通道: RGB + ELA + SRM)   ├──► prob map
                 │        多尺度 TTA [640,768,896] + flip   │
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage1.5  EfficientNet-V2-L × 5-fold   │
   image (RGB) ──┤             (RGB + ELA, 6 通道)        ├──► p_forged
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage2  evidence.py 抽 10 维结构化证据  │
                 │  (bbox / 面积比 / 异常度 / cls 共识 …)   │
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage2.5  XGBoost calibrator (5-fold)  │
                 │            OOF F1 = 0.9937              ├──► label, mask
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage3  Qwen3.5-9B + LoRA (r=64)       │
                 │  evidence prompt 注入, bbox 防幻觉       ├──► explanation
                 └────────────────────────────────────────┘
                                  │
                                  ▼
                  submit.csv (image_name, label, location, explanation)
```

核心设计：

1. **证据驱动** — 用 segmentation 输出的结构化证据（bbox 坐标、面积、异常度比值）注入 VLM prompt，强制 VLM 引用真实坐标，规避坐标幻觉。
2. **K-Fold 集成** — 5-fold SegFormer + 5-fold EfficientNet 平均，多尺度 TTA 进一步提升像素 F1（单 fold 平均 IoU 0.62 → 集成后 Dice 0.87）。
3. **校准器分离判别与定位** — XGBoost 在 (seg + cls + evidence) 10 维特征上拟合，把判定 F1 从 seg-only 的 0.95 提到 0.9937，同时给 VLM 一个干净的 label 上下文。

---

## 二、数据工程

数据全部走 `data/` 入口，按 **消费者** 而非来源分层。底层比特位于 NFS，本地用 symlink 组织。

### 2.1 数据现状

| 层 | 路径 | 数量 | 用途 |
|---|---|---:|---|
| raw | [data/raw/train_resume](data/raw/train_resume) | Black 640 + White 160 | seg / cls / VLM 主训练（5-fold 切分） |
| raw | [data/raw/val](data/raw/val) | Black 160 + White 40 | calibrator 拟合 + 官方评测 |
| raw | [data/raw/test](data/raw/test) | 500 | 推理输入（提交） |
| processed | [data/processed/synth](data/processed/synth) | 750 生成 / 62 keep | seg / cls 像素级合成正例（copy-move / splicing） |
| processed | [data/processed/real_ext](data/processed/real_ext) | 1100 张 + 1100 caption | seg / cls 真实负例补齐 + VLM 真实样本 |
| **vlm** | [data/vlm/caption_api_v3](data/vlm/caption_api_v3) | **1600 行 (100% strict)** | VLM SFT 主增强 caption（qwen-vl-max API） |

硬契约由 [scripts/data/guard.py](scripts/data/guard.py) 强制（产物：[data/meta/data_health.md](data/meta/data_health.md)）：

- raw 各子集行数达准入线（train_resume Black ≥ 600 / White ≥ 100；val Black ≥ 100 / White ≥ 30；test ≥ 100）
- processed/synth：Image == Mask 数量 + `keep.txt` 存在
- processed/real_ext：Image == Caption 数量
- 所有 symlink 解析后真实存在（`is_link_alive` 检查）
- vlm/caption_api_v3 ≥ 90% target

### 2.2 五个数据 Phase

```
A · 切分      split_train_val.py → train_split / val
              train_resume = train_split ∪ val 的稳定枚举视图（绕开 NFS 卡死）

B · caption 清洗 (Black)
   tools/check_caption_bbox.py → caption_bbox_audit.csv
   tools/clean_captions.py     → Black/Caption_clean/
   规则：IoU≥0.5 复制；0.2≤IoU<0.5 用 GT bbox 重写；其余进 needs_regen

C · 像素增强 (seg/cls 用)
   tools/synth_forgery.py        → data/processed/synth/
   tools/filter_synth_by_seg.py  → keep.txt（fold0 ckpt 反向过滤 750 → 62）
   tools/expand_real_images.py   → data/processed/real_ext/

D · evidence-caption 重生 (VLM 用)
   scripts/data/regen_caption_api.py → data/vlm/caption_api_v3/
   核心：extract_from_gt_mask 抽证据 → evidence_to_prompt_block 拼提示
   strict 校验：bbox ∈ GT 集合 + 长度 + "这是一份伪造的…" 开头 + "综上所述…" 结尾
   loose 回退：仅长度 + bbox 不越界
   每 stem 出 N 个温度版本（默认 [0.8, 1.0]）

E · 全量体检
   scripts/data/guard.py [--strict] → data/meta/data_health.md
```

### 2.3 caption_api_v3 vs caption_local_v2

旧本地方案 `tools/regen_evidence_captions.py` 走本地 Qwen3.5-9B，3 shard 并行 OOM 频繁、902 条中 49 条 > 800 字越界、strict 通过率 ≈9%。
新方案 `scripts/data/regen_caption_api.py` 走 `qwen-vl-max`：

| 维度 | local v2 | api v3 |
|---|---|---|
| 显存 | OOM × 3 进程 | 0（远端） |
| 行数 | 902 (49 越界) | **1600 (100% strict)** |
| 视觉理解 | 弱（9B base） | qwen-vl-max（视觉旗舰） |
| 成本 | — | ¥30-40 / 1280 调用 |

---

## 三、训练阶段

| 阶段 | 入口 | 输入 | 损失 / 目标 | 产物 | 关键指标 |
|---|---|---|---|---|---|
| 1 分割 | [train_seg_ensemble.py](train_seg_ensemble.py) | train_resume + synth + real_ext | Dice + Focal + Boundary（7 通道） | `checkpoints/seg/segformer_fold{0..4}/` | 单 fold IoU 0.5989-0.6514，**ensemble + TTA 后 val Dice = 0.8735** |
| 2 分类 | [train_classifier.py](train_classifier.py) | 同上 | CE（6 通道 RGB+ELA） | `checkpoints/cls/efficientnet_fold{0..4}/` | F1: 0.857 / 0.804 / 0.901 / 0.936 / 0.939 → **mean 0.887** |
| 3 校准 | [train_calibrator.py](train_calibrator.py) | val 集 evidence + cls + seg | 5-fold CV 多 backend，OOF 选阈值 | `checkpoints/calibrator/{calibrator.pkl, compare.md}` | **xgb OOF F1 = 0.9937, AUC = 0.9868**（vs logistic 0.9906） |
| 4 VLM SFT | [train_qwen35_9b.py](train_qwen35_9b.py) | train_resume + caption_api_v3 + real_ext | LoRA r=64 + evidence prompt 注入 | `checkpoints/qwen35_9b/` (LoRA adapter, 692 MB) | 4 epoch 8h25min（5 卡 device_map=auto），final loss = 0.15 |

### 3.1 Stage 1 · SegFormer 5-fold

- 架构：`nvidia/segformer-b5-finetuned-ade-640-640`，输入扩展为 7 通道（RGB + ELA + SRM）
- 5-fold 切分：[dataset.create_kfold_splits](dataset.py)，按 Black/White 分层
- 训练数据：`train_resume`（800） + `synth keep`（62） + `real_ext`（1100，全 0 mask 作真实负例）
- 优化：BF16 AMP + AdamW + OneCycleLR，patience=15
- **每个 fold 单尺度 768 上的 best IoU 0.60-0.65；推理时多尺度 TTA + ensemble 把 val Dice 推到 0.8735**

### 3.2 Stage 2 · EfficientNet-V2-L 5-fold

- 输入 6 通道（RGB + ELA），CE loss + WeightedRandomSampler
- 与 seg 同分层

### 3.3 Stage 3 · 校准器（XGBoost）

10 维特征：

```
seg_confidence, seg_max_prob, seg_mean_prob, seg_area_ratio,
ela_anomaly, srm_anomaly, ela_srm_ratio, num_evidence_regions,
cls_score_mean, cls_score_std
```

5 backend 对比（[checkpoints/calibrator/compare.md](checkpoints/calibrator/compare.md)）：

| backend | OOF F1 | AUC | Brier↓ | LogLoss↓ |
|---|---:|---:|---:|---:|
| **xgb** ⭐ | **0.9937** | 0.9868 | 0.0106 | 0.1155 |
| logistic | 0.9906 | 0.9978 | 0.0273 | 0.0809 |
| lgbm_mono / tabpfn / ebm | （见 compare.md） | | | |

挑选标准：OOF F1 优先，平局取 AUC。最终阈值 `t = 0.350`。

### 3.4 Stage 4 · Qwen3.5-9B LoRA SFT

- 模型：`models/Qwen3.5-9B`（ModelScope 下载到 NFS）
- LoRA：r=64, alpha=128, dropout=0.05，target 自动发现（排除 vision tower）
- 数据：`Caption_clean`（GT）+ `caption_api_v3`（增强）+ `real_ext`（真实样本）
- **inject_evidence=True**：训练时用 GT mask 抽证据拼到 user prompt，与推理时"用预测 mask 抽证据"格式严格对齐 → 防止训练-推理分布漂移
- **多卡模型并行**：5 卡 L20 (46GB) `device_map="auto"`，base 18 GB 切 5 卡 + LoRA 激活 + 末尾 lm_head 峰值
- **关键优化**：
  - chunked cross-entropy 沿 token 维分块（patch 在 [train_qwen35_9b.py](train_qwen35_9b.py)），避开 248k 词表 logits 12 GB 显存峰
  - 视觉 token 限制 384×384 ≈ 144 token，靠 `image_processor.size.longest_edge` 实现（`max_pixels` 参数对 Qwen2VLImageProcessor 无效）
  - manual label masking 替代 truncation，保留 image token 对齐
- 4 epoch / 928 step，train_loss 0.32（avg），最后 step 0.15

---

## 四、推理流水线

入口：[inference.py](inference.py)，每阶段都带 `cache/`，断点续跑零代价。

```
Stage 1   分割集成 + 多尺度 TTA → 平均概率图 → 二值 mask
Stage 1.5 5x EfficientNet 投票 → p_classifier
Stage 2   evidence.py 抽证据 (10 维)
Stage 2.5 XGBoost calibrator → label, p_forged
Stage 3   Qwen3.5-9B (base + LoRA merged, device_map=auto)
          - 系统 prompt = 鉴定专家
          - 用户 prompt 注入证据 + 模板（"这是一份伪造的…" / "这是一张真实拍摄的…"）
          - 生成 300-600 字鉴定文本
Stage 4   写 CSV
```

测试集 500 张全流水线，单 L20 上 stage1+1.5+2+2.5 ≈ 30 min；stage3 单卡 35 s/张 ≈ 5 h，多卡 device_map=auto 加速比有限（夜里挂着跑）。

---

## 五、官方评测复现

官方公式：

```
S_Fin = 0.45 × S_Det + 0.25 × S_Loc + 0.30 × S_Exp
S_Exp = 0.5  × S_Auto + 0.5  × S_Sim
```

| 指标 | 实现 | 备注 |
|---|---|---|
| **S_Det** | image-level F1 | `compute_f1(pred_labels, gt_labels)` |
| **S_Loc** | pixel-level F1 | 数学等价 Dice = 2TP/(2TP+FP+FN)，仅在 forged 样本上算 |
| **S_Sim** | BERTScore F1 (zh) | `bert-base-chinese`，cands vs val/{Black/Caption_clean,White/Caption}（`.md` 后缀） |
| **S_Auto** | Qwen3-MAX rubrics 100 制 | DashScope API，4 维度（accuracy 30 / evidence 30 / logic 20 / professional 20） |

复现命令：

```bash
# 1) 在 val/ 上跑完整 inference 拿 explanation
bash scripts/run_val_inference.sh "2,3,6"
# -> submit_val.csv (200 行)

# 2) 跑官方评分 (BERTScore + Qwen-MAX)
export DASHSCOPE_API_KEY=sk-xxx
python score_official.py \
    --pred_csv submit_val.csv \
    --val_dir data/raw/val \
    --gpu 7 --qwen_model qwen-max --qwen_workers 8
# -> logs/score_official.md / logs/score_official.json
# -> cache/qwen_rubric_scores.json (200 个样本的 4 维度详细打分)
```

或者一键 watchdog（先 test/ 推理 → 自动接 val 推理 → 自动接评分）：

```bash
# 启动 test/ 推理
bash scripts/run_inference.sh "6,7"
# 拿到 pid 后启动 watchdog
test_pid=$(pgrep -f "python inference.py" | head -1)
bash scripts/run_full_pipeline.sh "$test_pid" "6,7" 7
```

---

## 六、快速开始 / 复现

### 6.1 环境

```bash
conda create -n TFI python=3.11 -y
conda activate TFI
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install bert-score dashscope     # score_official.py 需要
pip install openai                   # scripts/data/regen_caption_api.py 需要
```

硬件：1-7 卡 NVIDIA L20 (46 GB)，CUDA 12.4。VLM 训练/推理建议 ≥ 5 卡（device_map=auto）。

### 6.2 数据：体检 / 重生

```bash
# 体检
python scripts/data/guard.py --strict

# 用 qwen-vl-max API 重生 evidence-caption（默认 missing-only）
export DASHSCOPE_API_KEY=sk-xxx
python scripts/data/regen_caption_api.py --workers 6
```

### 6.3 训练（5-fold 并行示例）

```bash
# Stage 1 · SegFormer 5 fold
for f in 0 1 2 3 4; do
    bash scripts/run_seg_fold.sh segformer $f $((f % 7)) &
done
wait

# Stage 2 · 分类 5 fold
for f in 0 1 2 3 4; do
    bash scripts/run_cls_fold.sh $f $((f % 7)) &
done
wait

# Stage 3 · 校准器（全 backend 对比 + 自动选优）
bash scripts/run_calibrator.sh 5

# Stage 4 · Qwen3.5-9B LoRA（5 卡 device_map=auto）
bash scripts/run_vlm.sh "2,3,4,5,6"
```

### 6.4 推理 + 评分

```bash
# test/ 推理 → submit.csv
bash scripts/run_inference.sh "6,7"

# 在 val/ 上推理 + 评分
bash scripts/run_val_inference.sh "6,7"
python score_official.py --pred_csv submit_val.csv --val_dir data/raw/val --gpu 7
```

### 6.5 消融

```bash
# 单组（segformer + xgb + cls + tta + multiscale）
bash scripts/run_eval.sh 7

# 全消融（多 arch / 多 calibrator / cls on-off / tta on-off）
python evaluate.py --full_ablation --gpu 7
# → logs/ablation.md
```

---

## 七、项目结构

```
TFI/
├── README.md                           ← 本文档
├── DATA_AUGMENTATION.md                # 数据增强方法论（历史文档）
│
├── data/                               ← 统一数据入口（NFS symlink）
│   ├── README.md                       # 数据契约
│   ├── raw/{train_resume, val, test}/
│   ├── processed/{synth, real_ext}/
│   ├── vlm/caption_api_v3/             # API 蒸馏 caption 主增强
│   └── meta/data_health.md             # guard.py 输出
│
├── scripts/
│   ├── data/                           # 数据 pipeline
│   │   ├── guard.py                    # 全量硬契约体检
│   │   └── regen_caption_api.py        # qwen-vl-max evidence-caption
│   ├── run_seg_fold.sh                 # 单 fold seg 训练
│   ├── run_cls_fold.sh                 # 单 fold cls 训练
│   ├── run_calibrator.sh               # 校准器训练
│   ├── run_vlm.sh                      # Qwen3.5-9B LoRA SFT
│   ├── run_inference.sh                # test/ 推理
│   ├── run_val_inference.sh            # val/ 推理（用于评分）
│   ├── run_eval.sh                     # 单组评估
│   └── run_full_pipeline.sh            # watchdog: test → val → score
│
├── tools/                              # 数据 pipeline 历史脚本（仍在用）
│   ├── check_caption_bbox.py
│   ├── clean_captions.py
│   ├── synth_forgery.py / filter_synth_by_seg.py / expand_real_images.py
│   └── ...
│
├── dataset.py                          # 三个 Dataset (Seg / Cls / VLM)
├── evidence.py                         # 10 维结构化证据
├── calibrator.py                       # 5 backend 校准器
├── vlm_collator.py                     # Qwen3.5-VL collator + LoRA target
├── utils.py                            # ELA / SRM / RLE / 指标
│
├── train_seg_ensemble.py               # Stage 1
├── train_classifier.py                 # Stage 2
├── train_calibrator.py                 # Stage 3
├── train_qwen35_9b.py                  # Stage 4
├── inference.py                        # Stage 5 推理流水线
├── evaluate.py                         # 消融评估
├── score_official.py                   # 官方公式评分（S_Det/S_Loc/S_Sim/S_Auto/S_Fin）
├── split_train_val.py                  # 一次性切分
│
├── config.yaml                         # 推理/评估配置
├── run_pipeline.sh                     # 一键 pipeline
├── requirements.txt
│
├── checkpoints/  →  NFS symlink
│   ├── seg/segformer_fold{0..4}/       # SegFormer 5 fold
│   ├── cls/efficientnet_fold{0..4}/    # EfficientNet 5 fold
│   ├── calibrator/{calibrator.pkl, compare.md, metrics.json}
│   └── qwen35_9b/                      # LoRA adapter (692 MB)
│
├── models/Qwen3.5-9B/  →  NFS symlink
├── cache/              →  NFS symlink (test/ 推理缓存)
├── cache_val/                          # val/ 推理缓存（独立）
├── submit.csv                          # 最终提交（test/ 500 张）
└── submit_val.csv                      # val/ 200 张（用于评分）
```

---

## 八、关键设计与踩坑记录

### 8.1 Segmentation：单 fold IoU 0.62 → ensemble Dice 0.87

单 fold IoU 看着低（0.5989-0.6514），但 5-fold ensemble + 多尺度 TTA + 后处理把 val Dice 推到 0.8735。三个杠杆：

1. **5-fold 平均**：互相纠错，单 fold noise 抹掉
2. **多尺度 TTA [640, 768, 896] + 翻转**：每张图过 6 次前向
3. **后处理 morphology + min_area=100 + calibrator 反推 mask**：清噪点

ablation 验证（[logs/ablation.md](logs/ablation.md)）：

```
seg=segformer_only | cal=xgb | cls=on | tta=on | multiscale=True
  → F1=0.9846  Precision=0.9697  Recall=1.0000  Acc=0.9750
  → mean IoU=0.8160  mean Dice=0.8724
```

> 我们也试过 ConvNeXt-V2-Large + DeepLabV3+ 和 MaxViT-Large + FPN，但在 800 训练样本上不收敛（base IoU 0.4 左右）。SegFormer 的 ImageNet-22k pretrained + 全 transformer encoder 在小数据上更稳。最终方案放弃 3-arch ensemble，专注把 SegFormer 5-fold 推到极致。

### 8.2 VLM 训练 OOM 三连击与解法

- **症状**：Qwen3.5-9B + LoRA r=64 在 L20 (46 GB) 上 OOM，backward 时 cross_entropy 一次吃 12 GB
- **根因**：Qwen 词表 248k，loss 计算时 logits = `[seq_len, 248000]` × bf16 = 全量 materialize
- **解法 1 · 多卡 model parallel**：`device_map="auto"` 把 base 切到 3-5 卡，末尾 lm_head 峰值落在最后一卡的 30+ GB 空闲上
- **解法 2 · chunked CE**：monkey-patch `transformers.loss.loss_utils.fixed_cross_entropy`，沿 token 维分块计算（chunk_size=256），避开单次大分配
- **解法 3 · 视觉 token 限制**：`image_processor.size.longest_edge = 384*384` ≈ 144 token（注意 `max_pixels` 参数对 `Qwen2VLImageProcessor` 不生效）
- **解法 4 · label masking 替代 truncation**：collator 里 `labels[:, max_length:] = -100`，保留 image token 对齐（Qwen-VL processor 强校验图像 token 数）

### 8.3 SegFormer v1 → v2 重训失败的教训

为提升单 fold IoU，曾尝试 `OneCycleLR` 第二轮重训（resume 自 v1 best_model）：

- **结果**：epoch 1 学习率从 1e-5 突跳到 5e-5，瞬间冲掉 v1 学到的特征，所有 fold early-stop 在 epoch 1
- **教训**：resume training 必须同时 resume optimizer + scheduler state，单纯 resume model 配新 scheduler 等于 fine-tune from a destroyed init
- **决策**：放弃 v2，接受 v1 的 5-fold（mean IoU 0.62），用 ensemble + TTA 推到 0.87

### 8.4 caption_api_v3：strict→loose 双层校验

`qwen-vl-max` 偶发 bbox 越界（< 1%）：

- **strict**：bbox 必须 ∈ GT mask 的连通域 bbox 集合
- **loose**：bbox 在图像范围内即可（`0 ≤ x1 < x2 ≤ W`）
- 默认 strict，生成 ≥ 90% 后切 loose 补齐

### 8.5 校准器：为什么 XGBoost 在 n=200 上不过拟合

普遍担心 200 行数据训 XGBoost 会过参，实测 OOF F1 0.9937（per-fold 1.000 / 0.984 / 1.000 / 0.984 / 1.000）：

- 特征只有 10 维（人工设计 + cls 共识），不是高维稀疏
- 树深 max_depth=4，n_estimators=200，正则 reg_alpha=0.1
- val 集 forged:real = 4:1，不平衡但样本充足

logistic 回归 OOF F1 0.9906 是更"稳"的对照（AUC 反而更高），但阈值更敏感。**生产用 xgb（F1 高），ablation 报 logistic 作 baseline**。

### 8.6 GPU 0 ECC 错误

GPU 0 持续报 `CUDA error: uncorrectable ECC error encountered`，已从所有训练/推理脚本中排除。可用 GPU = {1, 2, 3, 4, 5, 6, 7}。

---

## 致谢与依赖

- **基座模型**：SegFormer-B5 (NVIDIA) / EfficientNet-V2 / Qwen3.5-9B (Alibaba) / qwen-vl-max
- **校准器**：XGBoost / scikit-learn / LightGBM / TabPFN / interpret
- **评分**：bert-score / dashscope (Qwen3-MAX)
- **数据增强参考**：COCO val2017（真实负例补齐）

---

> 最后更新：2026-04-24，对应 commit 见 `git log`。
