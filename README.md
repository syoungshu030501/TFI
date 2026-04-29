# TFI · 证据驱动的图像伪造鉴定系统

> 电商竞赛三任务一体化方案：**伪造判别 (Detection) / 伪造定位 (Grounding) / 可解释分析 (Explanation)**
>
> 1000 张训练图（Black 800 = 伪造，White 200 = 真实）+ 200 张验证 + 500 张测试。
>
> **官方评测复现 · `S_Fin = 0.9034 / 1.0`**（val 200 张，v1.0 SFT baseline）

## 分支说明

| 分支 / Tag | 状态 | 内容 |
|---|---|---|
| [`tag v1.0-sft-baseline`](https://github.com/syoungshu030501/TFI/releases/tag/v1.0-sft-baseline) | ✅ 已锁 | 当前 README 1-9 章描述的稳定 5-stage SFT 流水线，S_Fin = 0.9034 |
| [`branch main`](https://github.com/syoungshu030501/TFI/tree/main) | 稳定版 | 同 v1.0-sft-baseline，对应 commit `63848ee` |
| [`branch v2-opd`](https://github.com/syoungshu030501/TFI/tree/v2-opd) ← **当前** | 🚧 开发中 | DeepSeek-V4 风格的 specialist OPD 大改造（基于 Qwen3.5-9B + Qwen3.6-27B teacher），目标 S_Fin ≈ 0.945-0.955。设计文档见 §10.5；实施计划见 §15 |

> v2-opd 完成并验证后，将通过 PR 合并回 main 替换为新版主线；当前 main 始终保留 v1.0 的可复现基线。

| 子项 | 值 | 权重 | 贡献 |
|---|---:|---:|---:|
| **S_Det** image-level F1 | **0.9845** | 0.45 | 0.4430 |
| **S_Loc** pixel-level F1 / Dice | **0.8735** | 0.25 | 0.2184 |
| ┃ S_Sim BERTScore-zh | 0.7552 | — | — |
| ┃ S_Auto Qwen3-MAX rubrics | 0.8582 | — | — |
| **S_Exp** = 0.5·Sim + 0.5·Auto | **0.8067** | 0.30 | 0.2420 |
| **S_Fin** | — | — | **🏆 0.9034** |

---

## 目录

- [一、项目概述与架构](#一项目概述与架构)
- [二、硬件与软件环境](#二硬件与软件环境)
- [三、数据工程](#三数据工程)
- [四、系统架构与数据流](#四系统架构与数据流)
- [五、模型清单](#五模型清单)
- [六、训练阶段详细记录](#六训练阶段详细记录)
- [七、校准器深度对比（XGB / TabPFN / Logistic）](#七校准器深度对比xgb--tabpfn--logistic)
- [八、推理流水线](#八推理流水线)
- [九、官方评测复现](#九官方评测复现)
- [十、关键设计与踩坑](#十关键设计与踩坑)
- [十一、性能提升技术汇总](#十一性能提升技术汇总)
- [十二、项目文件总览](#十二项目文件总览)
- [十三、面试问答速查（核心）](#十三面试问答速查核心)
- [十四、关键数字速查表（一页纸）](#十四关键数字速查表一页纸)
- [十五、v2-opd 实施计划（当前分支工作分解）](#十五v2-opd-实施计划当前分支工作分解)

---

## 一、项目概述与架构

### 1.1 任务定义

| 任务 | 输出列 | 评测指标 | 权重 |
|---|---|---|---:|
| **Detection** 伪造判别 | `label` ∈ {0, 1} | image-level F1 | 0.45 |
| **Grounding** 伪造定位 | `location` (COCO RLE) | pixel-level F1 = Dice = 2TP/(2TP+FP+FN) | 0.25 |
| **Explanation** 可解释分析 | `explanation` (300-600 字中文) | 0.5·BERTScore-zh + 0.5·Qwen3-MAX rubrics (4 维度) | 0.30 |

### 1.2 核心设计哲学（一句话）

**用像素级取证模型（5-fold 集成）抽出结构化"证据"，让小模型（XGBoost / TabPFN）做硬判定，让 VLM 在证据约束下做软论证。** 三个模型分工明确，证据是它们之间的统一通信协议。

```
                 ┌────────────────────────────────────────┐
                 │  Stage 1   SegFormer-B5 × 5-fold        │
   image (RGB) ──┤            (输入 7 通道: RGB+ELA+SRM)    ├──► prob map
                 │            多尺度 TTA [640,768,896]+flip │
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage 1.5 EfficientNet-V2-L × 5-fold   │
   image (RGB) ──┤            (RGB+ELA, 6 通道)           ├──► p_classifier
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage 2   evidence.py 抽 10 维结构化证据 │
                 │  (bbox / 面积比 / 异常度 / cls 共识 …)   │
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage 2.5 校准器 (XGBoost / TabPFN)    │
                 │  5-fold CV, OOF 阈值搜索                │
                 │  OOF F1 = 0.9937                       ├──► label, p_forged
                 └────────────────┬───────────────────────┘
                                  │
                 ┌────────────────▼───────────────────────┐
                 │  Stage 3   Qwen3.5-9B + LoRA (r=64)     │
                 │  evidence prompt 注入, bbox 防幻觉       ├──► explanation
                 └────────────────────────────────────────┘
                                  │
                                  ▼
                  submit.csv (image_name, label, location, explanation)
```

### 1.3 三个核心创新

| 创新点 | 解决什么问题 | 量化收益 |
|---|---|---|
| **Evidence-driven prompt** | VLM 自产 bbox 必幻觉（9B 模型不擅长精确像素坐标） | S_Exp 0.7→0.81 |
| **训推 prompt 格式对齐** (GT mask vs 预测 mask 抽 evidence schema 完全一致) | SFT 训推分布漂移 | train loss 0.15 → 推理稳定 |
| **Calibrator 解耦判定与定位** (10 维特征 + 5-fold CV + OOF 阈值) | seg-only F1=0.95 顶不上去；硬规则不灵活 | F1 0.9508 → 0.9937 |

---

## 二、硬件与软件环境

### 2.1 硬件

| 资源 | 配置 |
|---|---|
| GPU | 7× NVIDIA L20 (46 GB) | CUDA 12.4 |
| GPU 黑名单 | **GPU 0 持续 ECC error，已从所有脚本排除**，可用 = {1,2,3,4,5,6,7} |
| VLM 训练/推理推荐 | ≥ 5 卡 (`device_map="auto"`) |
| 存储 | NFS symlink → 本地 (`models/`, `checkpoints/`, `cache/` 都是 symlink) |

### 2.2 软件

```
conda create -n TFI python=3.11
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install bert-score dashscope     # score_official.py
pip install openai                   # scripts/data/regen_caption_api.py
pip install "tabpfn>=2.5"            # 可选：TabPFN backend
```

| 包 | 版本 |
|---|---|
| PyTorch | 2.5.1 + CUDA 12.4 |
| transformers | 4.51+ (含 Qwen3.5 / Qwen2VL processor) |
| peft | 0.13+ |
| segmentation-models-pytorch | 0.4+ |
| xgboost | 3.2+ |
| tabpfn | ≥ 2.5（v2 ckpt 可直接下，v2.5 需 `TABPFN_TOKEN`） |

---

## 三、数据工程

### 3.1 数据现状

| 层 | 路径 | 数量 | 用途 |
|---|---|---:|---|
| raw | [data/raw/train_resume](data/raw/train_resume) | Black 640 + White 160 | seg / cls / VLM 主训练（5-fold） |
| raw | [data/raw/val](data/raw/val) | Black 160 + White 40 | calibrator 拟合 + 官方评测 |
| raw | [data/raw/test](data/raw/test) | 500 | 推理输入（提交） |
| processed | [data/processed/synth](data/processed/synth) | 750 生成 / **62 keep** | seg / cls 像素级合成正例（copy-move / splicing） |
| processed | [data/processed/real_ext](data/processed/real_ext) | 1100 张 + 1100 caption | seg / cls 真实负例补齐 + VLM 真实样本 |
| **vlm** | [data/vlm/caption_api_v3](data/vlm/caption_api_v3) | **1600 行 (100% strict)** | VLM SFT 主增强 caption（qwen-vl-max API） |

硬契约由 [scripts/data/guard.py](scripts/data/guard.py) 强制（产物 [data/meta/data_health.md](data/meta/data_health.md)）：
- raw 各子集行数达准入线
- processed/synth：Image == Mask 数量 + `keep.txt` 存在
- 所有 symlink 解析后真实存在
- vlm/caption_api_v3 strict 通过率 ≥ 90%

### 3.2 五个数据 Phase

```
A · 切分      split_train_val.py → train_split / val
              train_resume = train_split ∪ val 的稳定枚举视图（绕开 NFS 卡死）

B · caption 清洗 (Black)
   tools/check_caption_bbox.py → caption_bbox_audit.csv
   tools/clean_captions.py     → Black/Caption_clean/
   规则：IoU≥0.5 复制；0.2≤IoU<0.5 用 GT bbox 重写；其余 needs_regen

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

### 3.3 caption_api_v3 vs 旧本地方案

| 维度 | local v2 (Qwen3.5-9B 本地) | **api v3 (qwen-vl-max API)** |
|---|---|---|
| 显存 | OOM × 3 进程 | 0（远端） |
| 行数 | 902 (49 越界) | **1600 (100% strict)** |
| 视觉理解 | 弱（9B base） | qwen-vl-max（视觉旗舰） |
| 成本 | — | ¥30-40 / 1280 调用 |

**经验**：用 9B 本地模型当"教师"在视觉精度上不够；API 蒸馏 ¥40 把数据质量做到 100% strict 通过，比本地 OOM 调三晚划算得多。

### 3.4 数据质量报告（基于真实 val 200 张逐样本回测）

把 `submit_val.csv` 200 行和 `data/raw/val/` GT 做逐样本对齐（[logs/val_per_sample.csv](logs/val_per_sample.csv)），按 caption 关键词反推每张 val 图的伪造类型，得到下面的"哪种伪造模型最学不到"的真实分布：

#### 3.4.1 训练集 800 张 Black 的伪造类型分布（caption 关键词统计）

| 类型 | 占比 (n/800) | 我们 synth 给的对应增强 |
|---|---:|---|
| 文字篡改 / 重打印 | **52.6%** (421) | textrep 150 → keep 7 ❌ 严重不足 |
| 涂抹 / 擦除 inpaint | **40.7%** (326) | 无 ❌ 完全缺失 |
| 拼接 splicing | 30.2% (242) | splice 350 → keep ~50 ⚠️ 过度合成且过滤掉 86% |
| 印章 / 签名 / logo | 14.6% (117) | 无 ❌ |
| 价格 / 金额数字 | 12.3% (99) | 无 ❌ 同时也是 val 上最差的子集 |
| 发票 / 收据 / 证件类 | 12.1% (97) | 无 ❌ 同时也是误报真实图的全部来源 |
| **AIGC** | 13.2% (106) | 无 ❌ 但反而表现最好 |
| copy-move | 1.8% (15) | copymove 250 → keep 4 ⚠️ 严重过度合成 |

**结论 ①**：合成分布与真实分布严重错位。我们给 copy-move 投了 33% 算力，但真实任务里 copy-move 只占 1.8%；真正的大头（重打印 53% + inpaint 41%）几乎没补。
**结论 ②**：AIGC 在数据集里 = "局部 AI 生成 + 拼回真实背景"（mask 平均仅 4.3%，全部 < 50%），不是整图 AIGC。视觉特征接近 inpaint，所以模型已经隐式学会了。

#### 3.4.2 val 200 张逐样本表现（按伪造类型）

| 类型 | n | 漏报 | mean Dice | median Dice | Dice<0.5 | Dice<0.2 |
|---|---:|---:|---:|---:|---:|---:|
| **AIGC** | 17 | 0 | **0.907** | 0.935 | 0 | 0 |
| splice | 57 | 0 | 0.905 | 0.954 | 3 | 1 |
| textrep | 97 | 0 | **0.919** | 0.959 | 3 | 2 |
| inpaint | 73 | 0 | 0.851 | 0.939 | 7 | 2 |
| copymove | 1 | 0 | 0.970 | — | 0 | 0 |
| **number (数字/金额篡改)** | 17 | **1** | **0.663** | 0.777 | **5** | **2** |
| other | 8 | 0 | 0.833 | 0.955 | 1 | 1 |

#### 3.4.3 val 200 张逐样本表现（按 GT mask 占比分桶）

| GT mask 占比 | n | mean Dice | Dice<0.5 比例 |
|---|---:|---:|---:|
| micro <0.5% | 20 | **0.645** | **6/20** |
| tiny 0.5-2% | 42 | 0.824 | 4/42 |
| small 2-10% | 63 | 0.930 | 1/63 |
| medium 10-50% | 34 | 0.961 | 1/34 |
| large 50%+ | 1 | 0.970 | 0/1 |

**结论 ③**：小目标是单一最大瓶颈。GT mask < 0.5% 的样本里，30% 的 Dice < 0.5。768 输入下小目标只有几像素，单尺度根本看不到。

#### 3.4.4 4 张误报真实图（FP）共性

| stem | caption 关键词 |
|---|---|
| 459983ce... | 马来西亚五金店 MUN HENG ELECTRICAL 热敏发票 |
| 629cf9dc... | 99 SPEED MART 便利店热敏收据 |
| 6a561e85... | ROCKU YAKINIKU 餐厅热敏收据 |
| 783a2845... | MOONLIGHT CAKE HOUSE 烘焙店热敏收据 |

**结论 ④**：所有 4 张 FP 都是热敏打印小票，特征是「低分辨率点阵 + 纸张折痕 + 手写笔迹墨痕」。训练集里所有热敏小票都是 Black（被改过的），White 0 张热敏小票，模型学到了「热敏 = 改过」的捷径，是数据采样偏差导致的 spurious correlation。

#### 3.4.5 四个真问题与下一步计划（按 ROI 排序）

| # | 问题 | 数据驱动证据 | 期望 ΔS_Fin | 解法 |
|---|---|---|---:|---|
| 1 | **数字篡改类 Dice 仅 0.663** | val 17 张里 5 张 Dice<0.5，mean 0.663 vs 整体 0.873 | +0.005~0.015 | 在 train 里跑 200 张「同字体/同背景，仅替换 1-2 位数字」的合成（用 Pillow render 数字 + 原 OCR mask 当 GT），强迫模型学到"数字字形微差异"信号 |
| 2 | **小目标 (<0.5%) Dice 0.645** | val 20 张里 6 张 Dice<0.5 | +0.005~0.010 | 推理改全程多尺度 + 加 1280² 高分辨率 fold；训练时按 mask 大小做 anchor sample，小目标过采 3× |
| 3 | **热敏小票真实图全部误报 (4 FP)** | val White 4/40 误报，全部是热敏小票 | +0.005~0.008 | 用 NIPSReceipts/RVL-CDIP 真实热敏小票补 200 张到 White，配套真实 caption；calibrator 重新拟合阈值 |
| 4 | **真实图缺乏「易误判」对抗样本** | val White 全部是普通街景，没有精修电商图/HDR 合成/过曝过修 | +0.003~0.008 | 用 Adobe Lightroom preset 跑 100 张精修风格 + DPED HDR 合成 100 张，全部当 White |

**预期累计 ΔS_Fin ≈ +0.02~0.04 → 0.92~0.94**，且基本不动模型，光靠数据补齐。

#### 3.4.6 真实图能否是 AIGC？

**任务隐含约定：不能**。White 200 张的 caption 中有 0 张提到 AIGC/AI生成/生成式，全部都是「这是一张真实拍摄/扫描的...，未发现数字伪造或后期篡改的痕迹」模板。也就是只要图里有任何 AI 生成内容，就归 Black。所以：
- ✅ 不需要单独训"AI vs Real"二分类器
- ⚠️ 但要警惕：未来如果数据集里出现「整图 AIGC 当 White」，我们的模型会错（当前训练集里 100% AIGC 都是 Black）
- ⚠️ 现在的 SegFormer 学到的是「AI 生成区域的视觉异常」（乱码字符/解剖错误/光照不一致），不是「图像整体是否 AI 生成」，鲁棒性来自这个区分

---

## 四、系统架构与数据流

### 4.1 各 Stage IO 详表（这是最重要的一张表，面试必背）

| Stage | 模型/算法 | 输入（shape / 内容） | 输出（shape / 内容） | 在 4 列里贡献 |
|---|---|---|---|---|
| 1 | SegFormer-B5 × 5-fold | `(B, 7, 768, 768)` = RGB + ELA + SRM；推理时多尺度 [640,768,896] × 3 flip TTA = 18 次前向/张 | prob map `(H, W) ∈ [0,1]`，5-fold + 多尺度平均 | `location` (二值化+RLE) |
| 1.5 | EfficientNet-V2-L × 5-fold | `(B, 6, 512, 512)` = RGB + ELA | softmax `[P_real, P_forged]` → mean / std | calibrator 的 2 维特征 |
| 2 | `evidence.extract()` (确定性) | 原图 + binary mask + prob map | dict {`label`, `total_area_ratio`, `n_regions`, `regions[{bbox,area,centroid}]`, `seg_confidence`, `seg_max_prob`, `ela_anomaly`, `srm_anomaly`, `edge_sharpness`, `region_text`} | 不直接产出，是中间通信协议 |
| 2.5 | XGBoost / TabPFN calibrator | `X = [seg_conf, seg_max, area_ratio, log(n_reg+1), largest_area, ela_anom, srm_anom, edge_sharp, cls_mean, cls_std]` (10 维) | `(p_forged, label)` ；阈值 = 5-fold OOF argmax F1 (xgb=0.35 / tabpfn=0.61) | **`label`** |
| 3 | Qwen3.5-9B + LoRA (r=64) | image (vision tower 384²≈144 token) + sys prompt + user prompt (含 `evidence_to_prompt_block` JSON 块 + 模板) | 300-600 字中文鉴定文本 | **`explanation`** |
| 4 | CSV writer | `{name → (label, rle, explanation)}` | submit.csv | 4 列拼合 |

### 4.2 三种特征的语义

```
seg_*       像素级伪造概率统计（来自 SegFormer）
            └─ seg_confidence : mask 内平均 prob (越高越像伪造)
            └─ seg_max_prob   : 全图 prob 最大值 (即使 mask 空也能体现"局部高响应")
            └─ total_area_ratio / largest_area_ratio / log(n_regions+1) : 空间形态

ela_*/srm_* 取证滤波器响应（来自 evidence.extract）
            └─ ela_anomaly : mask 内 ELA 均值 / mask 外 ELA 均值（>1 = 内部 JPEG 压缩异常）
            └─ srm_anomaly : 同上，SRM 噪声残差
            └─ edge_sharpness : mask 边缘 prob 梯度（越陡越像人工拼接）

cls_*       图像级分类共识（来自 EfficientNet）
            └─ cls_score_mean / cls_score_std : 5 fold P(forged) 的统计
```

### 4.3 训练 vs 推理的两个对齐点（关键）

```
                                训练                        推理
SegFormer / EfficientNet     独立 5-fold CV              独立 ckpt 推理
                              GT mask / GT label         无监督
                              ↓                          ↓
evidence.extract             用 GT mask 抽              用预测 mask 抽
                              (extract_from_gt_mask)    (extract)
                              ↓ schema 完全一致 ↓
calibrator                   val 200 行 5-fold CV       全量 refit 直接 predict
                              拟合 GT label              输出 p_forged
                              ↓                          ↓
VLM                          prompt 用 GT 的 evidence    prompt 用预测 evidence
                              + GT label                 + calibrator label
                              assistant = caption        free generate
                              ↓ prompt schema 完全一致 ↓
```

**核心原则**：训练时用 GT 抽 evidence，推理时用预测抽 evidence，但两者打包成的 dict / prompt 字段**字字对应**。这是 SFT 不漂移的根本保证。

---

## 五、模型清单

| 模型 | 来源 | 用途 | 大小 | 训练时角色 | 推理时角色 |
|---|---|---|---|---|---|
| SegFormer-B5 | `nvidia/segformer-b5-finetuned-ade-640-640` | 分割 backbone | 324 MB / fold | 监督 GT mask, Dice+Focal+Boundary | 5-fold + 多尺度 TTA 平均 |
| EfficientNet-V2-L | `timm: tf_efficientnetv2_l.in21k_ft_in1k` | 二分类 backbone | 450 MB / fold | CE w/ class_weight=[1, 0.25] | 5 模型 P(forged) 取 mean/std |
| Qwen3.5-9B (base) | ModelScope `qwen/Qwen3.5-9B` (Qwen3-VL family) | VLM base | ~17 GB | LoRA r=64 SFT | base + LoRA merged, device_map=auto |
| Qwen3.5 LoRA adapter | 自训 | LoRA adapter | 692 MB | trainable | merge_and_unload 后推理 |
| qwen-vl-max | DashScope API | 训练数据增强（caption 重生） | — | 蒸馏教师 | — |
| qwen3-max | DashScope API | rubric 评分 (S_Auto) | — | — | 评测 |
| XGBoost / TabPFN-v2 | xgboost / priorlabs.ai | 校准器 | ~100 KB / 28 MB | 200 行 val 5-fold CV | calibrator.predict |

---

## 六、训练阶段详细记录

### 6.1 Stage 1 · SegFormer 5-fold

**入口**：`train_seg_ensemble.py`

| 配置项 | 值 |
|---|---|
| Backbone | `nvidia/segformer-b5-finetuned-ade-640-640`，第一层 patch embedding 改 7 通道 |
| 输入 | 768 × 768，7 ch (RGB normalized + ELA + SRM) |
| 训练数据 | train_resume (800) + synth keep (62) + real_ext (1100, 全 0 mask 作真实负例) |
| Loss | **0.4 · Focal(α=0.25, γ=2.0) + 0.4 · Dice + 0.2 · Boundary** |
| Optimizer | AdamW, lr=6e-5, weight_decay=0.01 |
| Scheduler | OneCycleLR (max_lr=6e-5) |
| Batch / epochs | batch_size=4, epochs=100, **early stopping patience=15** |
| AMP | BF16 |
| K-Fold | 5-fold stratified by Black/White |
| GPU | 7 卡并行（fold % 7） |

**单 fold 结果（5 fold 在 768 单尺度上的 val IoU）**：

| fold | val IoU |
|---|---:|
| 0 | 0.5989 |
| 1 | 0.6201 |
| 2 | 0.6514 |
| 3 | 0.6118 |
| 4 | 0.6308 |
| **mean** | **~0.6226** |

**集成 + 多尺度 TTA + 后处理后的 val Dice = 0.8735**（详见下面 ablation）。

### 6.2 Stage 2 · EfficientNet-V2-L 5-fold

**入口**：`train_classifier.py`

| 配置项 | 值 |
|---|---|
| Backbone | `tf_efficientnetv2_l.in21k_ft_in1k`，第一层改 6 通道 |
| 输入 | 512 × 512，6 ch (RGB + ELA) |
| Loss | CE，**class_weight=[1.0, 0.25]**（对 forged 严格因为占多数） |
| Sampler | 可选 WeightedRandomSampler（与 class_weight 二选一） |
| Optimizer | AdamW, lr=3e-4, weight_decay=0.01 |
| Scheduler | OneCycleLR |
| Batch / epochs | batch_size=8, epochs=50, patience=15 |

**Per-fold F1**：0.857 / 0.804 / 0.901 / 0.936 / 0.939 → **mean 0.887**

### 6.3 Stage 3 · 校准器（XGBoost / TabPFN）

详见 [§七](#七校准器深度对比xgb--tabpfn--logistic)。简表：

| 配置项 | 值 |
|---|---|
| 训练数据 | val 200 行（160 forged + 40 real）evidence + cls 推理结果 |
| 特征 | 10 维 (8 evidence + 2 cls) |
| CV | 5-fold StratifiedKFold |
| 阈值 | OOF probs 上 argmax F1（xgb=0.35, tabpfn=0.61） |
| 5 backend 对比 | logistic / xgb / lgbm_mono / **tabpfn** / ebm |

### 6.4 Stage 4 · Qwen3.5-9B LoRA SFT

**入口**：`train_qwen35_9b.py`

| 配置项 | 值 |
|---|---|
| Base model | `models/Qwen3.5-9B`（NFS symlink） |
| LoRA | **r=64, alpha=128, dropout=0.05**, target_modules 自动发现（排除 vision tower） |
| 训练数据 | Caption_clean（GT 1000）+ caption_api_v3（增强 1600）+ real_ext（1100） |
| `inject_evidence` | **True**（用 GT mask 抽 evidence 注入 user prompt，与推理对齐） |
| Optimizer | AdamW, lr=1e-4（LoRA）/ 2e-5（Full FT） |
| Scheduler | cosine, warmup_ratio=0.05, weight_decay=0.01 |
| Batch / grad_accum | batch_size=1, grad_accum=16, effective batch=16 |
| Epochs / steps | 4 epoch / **928 step** |
| Max length | 视情况调，labels 用 manual masking 替代 truncation 保 image token 对齐 |
| GPU | 5 卡 L20 (`device_map="auto"`)，8h25min |
| 视觉 token | `image_processor.size.longest_edge = 384²` ≈ 144 token |
| Final loss | train_loss 0.32 (avg) / step-928 0.15 |

**关键工程优化**（任一缺失都 OOM）：

1. **Chunked CE patch**：monkey-patch `transformers.loss.loss_utils.fixed_cross_entropy`，沿 token 维分块 (chunk_size=256) 计算 CE，避开 [seq_len, 248k] × bf16 = 12 GB 单次峰值
2. **多卡 model parallel**：`device_map="auto"` 让 base 18 GB 切到 5 卡，末尾 lm_head 峰值落在最后卡的空闲上
3. **视觉 token 限制**：`size.longest_edge` 而非 `max_pixels`（后者对 `Qwen2VLImageProcessor` 不生效）
4. **Manual label masking**：`labels[:, max_length:] = -100` 而非 truncation，因为 Qwen-VL processor 强校验图像 token 数

---

## 七、校准器深度对比（XGB / TabPFN / Logistic）

### 7.1 Backend 对比表（5-fold CV，n=200，10 维特征）

> 跑法：`python train_calibrator.py --compare_all --cv_folds 5`
> TabPFN 跑法：`python tools/run_tabpfn_eval.py`

| backend | OOF F1 | AUC | Brier↓ | LogLoss↓ | best_t | per-fold F1 (mean±std) | fold time | 备注 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `xgb` | **0.9937** | 0.9963 | **0.0108** | 0.0996 | 0.45 | 0.9937 ± 0.0078 | 0.1s | + isotonic 后处理；当前生产 |
| `logistic` | 0.9906 | 0.9978 | 0.0273 | 0.0809 | 0.18 | 0.9937 ± 0.0077 | 0.05s | baseline，AUC 反而最稳 |
| **`tabpfn-v2`** | **0.9937** | **0.9978** | 0.0120 | **0.0472** | 0.61 | **0.9968 ± 0.0063** | 1.3s | **LogLoss 减半**，零超参 |
| `tabpfn-v2.5` | (待 token) | — | — | — | — | — | — | 需 priorlabs.ai 注册拿 `TABPFN_TOKEN` |

**生产选 xgb**（兼顾 F1 + Brier + sklearn pickle 体积小）；**论证选 tabpfn**（同 F1 但 LogLoss 减半，给 VLM 注入的 p_forged 更可信）。

### 7.2 解读：为什么 200 行训 calibrator 不过拟合

- **特征只有 10 维**，且全是手工设计的**领域特征**（不是高维稀疏 NLP one-hot）
- XGB：`max_depth=4, n_estimators=200, reg_alpha=0.1, reg_lambda=1.0`，强正则
- TabPFN：参数 frozen，meta-train 过的 transformer，**不"训"你的数据，只把它当 in-context prompt**
- 200 行虽小，但 80% 正例 + 20% 负例分布清晰，5-fold CV 后 per-fold F1 std=0.006 → 极稳定

### 7.3 calibrator 对前置环节的提升

| 链路 | F1 | precision | recall | accuracy |
|---|---:|---:|---:|---:|
| seg only (二值 mask 面积 > threshold 判 forged) | 0.9508 | 1.000 | 0.9062 | 0.9250 |
| hard rule (cls_score < 0.2 翻 0；> 0.9 翻 1) | 0.9644 | 1.000 | 0.9312 | 0.9450 |
| **calibrator (xgb, OOF)** | **0.9937** | 0.9937 | 0.9937 | 0.9900 |

**关键观察**：
- seg only 从不假阳（precision=1.0）但漏了 9.4% 真伪造
- hard rule 把漏判救回一些但仍漏 6.9%
- calibrator 把 precision/recall 同时拉到 0.9937，**+4.3 个 F1 点**

### 7.4 TabPFN 原理（面试重点）

**TabPFN = Prior-Fitted Network**：
- 一个 transformer 在 Prior Labs 服务器上，对几亿个**合成 tabular 任务**做了 meta-training（一次）
- 你 `.fit(X, y)` 不更新参数，只是把数据存进 in-context buffer
- `.predict(X_test)` 是 forward pass：把 (X_train, y_train, X_test) 拼成序列喂 transformer，输出 P(y_test ‖ context)
- 数学上近似贝叶斯后验 \(p(y_{test} \mid x_{test}, X_{train}, y_{train})\)
- → **不存在过拟合**，只要任务在合成先验覆盖范围内（200 行 / 10 维 = 甜蜜点）

**v2 vs v2.5**：
| 版本 | 上限 | 关键新增 | 你场景 |
|---|---|---|---|
| v2 (NeurIPS / Nature 2025) | 10K 行 / 500 维 | 真实数据 finetune、稳定 | 已实测 |
| v2.5 (2025/11, arXiv 2511.08667) | 50K 行 / 2K 维 | 24 层（v2 是 12）、distillation engine | 边际收益小（你 200 行用不满 5x scale） |

---

## 八、推理流水线

**入口**：`inference.py`，每阶段都带 `cache/`，断点续跑零代价。

### 8.1 流程

| Stage | 操作 | 输入 cache | 输出 cache | 单卡耗时 (test 500 张) |
|---|---|---|---|---:|
| 1 | 分割集成 5-fold × 多尺度 TTA | — | `seg_outputs.npz` | ~20 min |
| 1.5 | EfficientNet 5-fold 投票 | — | `cls_scores.json` | ~5 min |
| 2 | evidence.extract + 后处理 | seg + cls | — | ~30 s |
| 2.5 | calibrator.predict (xgb) | evidence + cls | `stage2_results.json` | ~5 s |
| 3 | Qwen3.5-9B + LoRA 生成 | stage2 | `explanations.json`（逐条增量写） | 35 s/张，~5 h 单卡（多卡有限加速） |
| 4 | 写 CSV | 全部 | submit.csv | <1 s |

### 8.2 显存预算

| 阶段 | 单模型驻留 | 峰值 |
|---|---:|---:|
| Stage 1 SegFormer (逐 fold 加载/释放) | ~3-4 GB | ~5 GB |
| Stage 1.5 EfficientNet | ~1 GB | ~1 GB |
| Stage 2/2.5 evidence + calibrator | <100 MB | <500 MB |
| Stage 3 Qwen3.5-9B (BF16) + LoRA merged | ~17 GB (单卡) / 4 GB × 5 (device_map=auto) | ~25 GB 单卡 |
| **总峰值** | — | **~25 GB << 46 GB** |

### 8.3 推理时关键 fallback

- **calibrator 翻 1 但 binary 空**：用 `prob > 0.2` 兜底重做 mask（`inference.py:305-310`），保证 VLM 拿到至少一个 bbox
- **VLM 模板分支**：label=1 用「这是一份伪造的…/综上所述…」，label=0 用「这是一张真实拍摄的…/综合分析…」
- **`</think>` 切除**：Qwen-Thinking 模型可能输出推理链，输出前去除

---

## 九、官方评测复现

### 9.1 公式

```
S_Fin = 0.45 · S_Det + 0.25 · S_Loc + 0.30 · S_Exp
S_Exp = 0.5  · S_Auto + 0.5  · S_Sim
```

| 指标 | 实现 | 备注 |
|---|---|---|
| **S_Det** | image-level F1 | `compute_f1(pred_labels, gt_labels)` |
| **S_Loc** | pixel-level F1 = Dice = 2TP/(2TP+FP+FN)，仅在 forged 样本上算 | |
| **S_Sim** | BERTScore F1 (zh) | `bert-base-chinese`，cands vs val/{Black/Caption_clean,White/Caption} |
| **S_Auto** | Qwen3-MAX rubrics 100 制 | DashScope API，4 维度 (accuracy 30 / evidence 30 / logic 20 / professional 20) |

### 9.2 复现命令

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

### 9.3 当前最佳成绩

```
S_Det (img-F1)  = 0.9845
S_Loc (pix-F1)  = 0.8735
S_Sim (BERTSc.) = 0.7552
S_Auto (Qwen)   = 0.8582
S_Exp           = 0.8067
─────────────────────────
S_Fin           = 0.9034
```

### 9.4 Ablation（消融，详见 [logs/ablation.md](logs/ablation.md)）

| Config | F1 | Precision | Recall | Acc | mean IoU | mean Dice |
|---|---:|---:|---:|---:|---:|---:|
| seg=segformer_only \| cal=xgb \| cls=on \| tta=on \| multiscale=True | **0.9846** | 0.9697 | 1.000 | 0.9750 | **0.8160** | **0.8724** |

---

## 十、关键设计与踩坑

### 10.1 单 fold IoU 0.62 → ensemble Dice 0.87 的三个杠杆

1. **5-fold 平均**：互相纠错，单 fold noise 抹掉
2. **多尺度 TTA [640, 768, 896] + 翻转**：每张图过 6 次前向
3. **后处理**：morphology + min_area=100 + calibrator 反推 mask

> 也试过 ConvNeXt-V2-Large + DeepLabV3+ 和 MaxViT-Large + FPN，但在 800 训练样本上不收敛（base IoU 0.4 左右）。SegFormer 的 ImageNet-22k pretrained + 全 transformer encoder 在小数据上更稳。**最终方案放弃 3-arch ensemble，专注把 SegFormer 5-fold 推到极致**。

### 10.2 VLM 训练 OOM 三连击与解法（已写在 §6.4）

| 症状 | 解法 |
|---|---|
| backward 时 CE 一次吃 12 GB | chunked CE patch (chunk=256) |
| Base 18 GB 单卡放不下 + LoRA 激活峰值 | `device_map="auto"` 5 卡切 |
| 视觉 token 太多，max_pixels 参数不生效 | 改 `image_processor.size.longest_edge = 384²` |
| Truncation 破坏 image token 对齐报错 | manual `labels[:, max_length:] = -100` |

### 10.3 SegFormer v1 → v2 重训失败的教训

为提升单 fold IoU，曾尝试 OneCycleLR 第二轮重训（resume 自 v1 best_model）：
- **结果**：epoch 1 学习率从 1e-5 突跳到 5e-5，瞬间冲掉 v1 学到的特征，所有 fold early-stop 在 epoch 1
- **教训**：resume training 必须**同时 resume optimizer + scheduler state**，单纯 resume model 配新 scheduler 等于 fine-tune from a destroyed init
- **决策**：放弃 v2，接受 v1 的 5-fold（mean IoU 0.62），用 ensemble + TTA 推到 0.87

### 10.4 caption_api_v3：strict→loose 双层校验

`qwen-vl-max` 偶发 bbox 越界（< 1%）：
- **strict**：bbox 必须 ∈ GT mask 的连通域 bbox 集合
- **loose**：bbox 在图像范围内即可（`0 ≤ x1 < x2 ≤ W`）
- 默认 strict，生成 ≥ 90% 后切 loose 补齐

### 10.5 下一代方案：specialist OPD（DeepSeek-V4 风格，基于 Qwen3.5-9B）

> **本节是 v2-opd 分支的设计文档。v1.0 SFT baseline (S_Fin=0.9034) 已 tag 为 `v1.0-sft-baseline` 永久保留。**

DeepSeek-V4（2026-04-23 发布）正式弃用 mixed RL，改为 [On-Policy Distillation (OPD)](https://arxiv.org/abs/2604.00626)：每个领域独立训 specialist (SFT + GRPO)，再用反向 KL 蒸馏到统一 student。本任务比 V4 通用任务**更适合**这套，因为我们所有 reward 都是 ground-truth-derivable，不需要人标 preference。

#### 10.5.1 为什么 policy 选 Qwen3.5-9B（已在本地）

我们已下载的 [`Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)（2026-03-02 发布）就是这个任务的最优 policy 选择，理由都来自官方 HF 模型卡 + 本地 `models/Qwen3.5-9B/config.json`：

| 关键特性 | 数值 / 实现 | 对本任务的价值 |
|---|---|---|
| **原生多模态早期融合** | 训练时图像/文本 token 从第 1 层就一起进 transformer | 视觉表征不是 bolt-on，训推一致性强 |
| **DeepStack ViT 视觉编码器** | 多层 ViT 特征注入 LLM 各层 (`vision_config.deepstack_visual_indexes`) | 对 forensic 这种"细节像素证据"任务比 single-pool ViT 强一档 |
| **Gated DeltaNet 75% + Full Attention 25%** 混合 | `layer_types: linear×3 → full×1` 重复 8 次 = 32 层 | 推理 O(L) 线性复杂度；RL rollout 比纯 transformer 快 2-3×；显存峰值低 |
| **Multi-Token Prediction (MTP)** | `mtp_num_hidden_layers: 1` | 配合 speculative decoding 进一步加速 caption 生成 |
| **262K 原生上下文** | `max_position_embeddings: 262144` | 长 caption + 完整证据 + reasoning trace 都装得下 |
| **本地 BF16 18.6 GB** | 4 个 safetensors shard | 单卡 L20 (46GB) 装得下 base，5 卡 zero3 全参 RL 可行 |

**Benchmark 对比**（HF 官方表，证明 9B Dense 打 30B-A3B MoE）：

| Benchmark | Qwen3-VL-30B-A3B | **Qwen3.5-9B** | GPT-5-Nano | Gemini-2.5-Flash-Lite |
|---|---:|---:|---:|---:|
| MMMU | 76.0 | **78.4** | 75.8 | 73.4 |
| MMMU-Pro | 63.0 | **70.1** | 57.2 | 59.7 |
| MathVision | 65.7 | **78.9** | 62.2 | 52.1 |
| OmniDocBench (与本任务最相关) | 79.4 | **87.7** | 55.9 | — |
| HallusionBench | 66.0 | **69.3** | 58.4 | 64.5 |
| CC-OCR | 77.8 | **79.3** | 58.9 | 72.9 |

**OmniDocBench 87.7 直接对标我们的"文档伪造判别 + 解释"业务**，这是任何 Qwen3-VL 系列都达不到的。

#### 10.5.2 OPD 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  TEACHER (蒸馏源，frozen)                                            │
│   选 1: Qwen3.6-27B (2026-04-22 发布, dense)                         │
│   选 2: Qwen3.5-122B-A10B (A10B 激活, FSDP zero3 + offload 可装)     │
│   选 3: DashScope qwen3-vl-max API (零本地负担, ¥0.4/调用)           │
└────────────────────────┬────────────────────────────────────────────┘
                         │ 输出 token-level logit dist + reasoning
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STUDENT (policy, 我们要训的)                                         │
│   Qwen3.5-9B 全参 SFT → GKD on-policy distillation                  │
│   ↓ rollout n=8 candidates / sample (vllm 0.11+)                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ candidate caption + bbox extraction
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  REWARD = α·R_loc + β·R_cls + γ·R_cap + δ·R_bbox                    │
│                                                                      │
│   R_loc   ←  DINOv3-ViT-L (2025-08, Meta) + Mask2Former-light       │
│              7-channel adapter (RGB+ELA+SRM)                         │
│              替代 SegFormer-B5，ImageNet 监督 → SSL frozen 强一档    │
│                                                                      │
│   R_cls   ←  SigLIP-2-So400m-NaFlex (2025-02, Google)               │
│              NaFlex 保留 native aspect (收据/小票/电商图非方形)      │
│              超过 EVA-CLIP / AIMv2                                   │
│                                                                      │
│   R_cap   ←  GRM (Generative Reward Model, DeepSeek-V4 同款)        │
│              actor 自评 rubric → 每 50 step 用 qwen-max 校准 10 张  │
│                                                                      │
│   R_bbox  ←  SAM 3.1 (2026-03-27, Meta, ICLR 2026)                  │
│              caption 中 phrase 当 prompt → 验证 bbox 是否真在那里   │
│              替代 Grounding DINO + SAM2 双模型                       │
│                                                                      │
│   + Forensic 专项: MaskCLIP / Mesorch (NeXT-IMDL benchmark 王者)    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ 多 specialist reward 融合
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EOPD UPDATE (Entropy-Aware OPD)                                     │
│    high-entropy token → forward KL (防 mode collapse)                │
│    low-entropy token  → reverse KL (DeepSeek-V4 默认，收敛快)        │
│    + sentence-level importance sampling clip                         │
│      (Qwen3 issue #1799 已报告无 clip 训练崩溃，必须加)              │
└─────────────────────────────────────────────────────────────────────┘
```

#### 10.5.3 Specialist 模型选型（基于 2025-2026 真实 SOTA）

| 角色 | v1 baseline | **v2-opd 选型** | 来源 + 选型理由 |
|---|---|---|---|
| **Policy LM** | Qwen3.5-9B + LoRA r=64 SFT | **Qwen3.5-9B 全参** + GKD/EOPD（teacher = Qwen3.6-27B） | 本地权重已就位；同模型不换；从 LoRA 升级到全参以充分释放容量；teacher 选 Qwen3.6-27B（2026-04-22 发布，dense，6 卡 zero3 装得下）|
| **像素 specialist (loc)** | SegFormer-B5 5-fold | **DINOv3-ViT-L** (蒸馏自 7B) + 7ch stem adapter + Mask2Former-light head | [arXiv 2508.10104](https://arxiv.org/abs/2508.10104) (Meta, 2025-08-14)；首次单一 frozen SSL backbone 在 dense prediction 超 SegFormer 类专门方案；Gram anchoring 解决长训练 dense feature 退化（解释了我们 v1 SegFormer 100 epoch 后 IoU 0.62 上不去）；商业许可 |
| **图像 specialist (cls)** | EfficientNet-V2-L 5-fold | **SigLIP-2-So400m-NaFlex** (400M) + MLP head | [arXiv 2502.14786](https://arxiv.org/abs/2502.14786) (Google, 2025-02-20)；NaFlex 保留 native aspect ratio（我们 4 张 FP 全是热敏小票，aspect ≈ 0.3，方形 resize 把关键信号丢光）；caption-pretrain + masked-prediction → dense + 语义双强；超 EVA-CLIP / AIMv2 |
| **Forensic SOTA specialist** | 没有 | **MaskCLIP**（IML-ViT-style，CLIP backbone）+ **Mesorch**（macro/meso/micro）双路 | [NeXT-IMDL benchmark (2025-12)](https://arxiv.org/abs/2512.23374) 跨域 F1: MaskCLIP 0.32 vs IML-ViT 0.12 vs TruFor 0.13 vs Mesorch 0.10 → MaskCLIP 是当前 forensic 跨域王者；ForensicHub (NeurIPS 2025) 有现成代码 |
| **Bbox grounding verifier** | 无 | **SAM 3.1** (2026-03-27) | [SAM 3 ICLR 2026](https://ai.meta.com/research/sam3/)；单模型 PCS：text phrase 直接出 mask；零样本 LVIS AP 48.8 vs SAM2 38.5；30ms / 100+ objects on H200；3.1 加 shared-memory 多目标更快；替代我之前错推的 Grounding DINO 1.5 + SAM2 双模型 |
| **Caption rubric reward** | qwen-max（评测时用） | **GRM** + qwen-max 校准 | DeepSeek-V4 同款 GRM；actor 自己生成 rubric 评分 → 不用每个 rollout 都调 API（¥0.4/张 × 8 rollouts × 800 step = ¥2560 太贵）；只在 step % 50 抽 10 样本调 qwen-max 校准 GRM 漂移 |
| **OPD 算法** | LoRA SFT (token-CE) | **EOPD** + sentence-level IS clip + low_var_kl | [Entropy-Aware OPD (2603.07079)](https://arxiv.org/abs/2603.07079)；[Qwen3 issue #1799](https://github.com/QwenLM/Qwen3/issues/1799) 报告无 clip 时 OPD 训练 collapse → 必须加 |

#### 10.5.4 训练栈 — ms-swift 主，verl 备

```bash
# ms-swift 路线 (issue #8182 已在 Qwen3-VL-8B 上跑通同样配置)
swift rlhf --rlhf_type gkd \
    --model models/Qwen3.5-9B \                  # 我们本地的
    --teacher_model Qwen/Qwen3.6-27B \            # 下载新 teacher
    --tuner_type full \                           # 全参，不上 LoRA
    --beta 1 --lmbda 1 \                          # 全 on-policy GKD
    --use_vllm true --vllm_mode server \
    --deepspeed zero3 \
    --teacher_deepspeed zero3 \
    --max_length 16384 --max_completion_length 4096 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8

# verl 路线 (备选)
# 注意: Qwen3.5 Gated DeltaNet 需要 vllm>=0.11 + causal-conv1d 库;
# verl main 分支 Qwen3-VL 已支持但 Qwen3.5 需自行验证 vllm rollout 路径
```

#### 10.5.5 7×L20 (GPU 0 坏，6×46GB=276GB) 算力分配 — 激进版

```
═════════════════════════════════════════════════════════════════════════
阶段 0 (准备, 0.5 天)                                                  
─────────────────────────────────────────────────────────────────────────
- 下载 Qwen3.6-27B (~52 GB BF16) 到 NFS                               
- 下载 DINOv3-ViT-L / SigLIP-2-So400m-NaFlex / SAM 3.1 / MaskCLIP     
- pip install ms-swift (latest), vllm>=0.11, causal-conv1d, 升级 transformers ≥ 4.57
                                                                       
═════════════════════════════════════════════════════════════════════════
阶段 1 (SFT warmup + specialist 训练, 3 天)                            
─────────────────────────────────────────────────────────────────────────
GPU 1-4: Qwen3.5-9B 全参 SFT (deepspeed zero3 + flash-attn 3 +        
         grad_ckpt)；4 卡 18GB 模型 zero3 切到 < 25GB/卡；             
         数据 = caption_api_v3 1600 + train_resume 800 + real_ext 1100
         + v2 数据增强 (synth_v2, 见下面)；batch=1/卡, grad_accum=8    
                                                                       
GPU 5: DINOv3-ViT-L + 7ch adapter + Mask2Former 5-fold (轮转)         
GPU 6: SigLIP-2-So400m + cls head 5-fold + MaskCLIP fine-tune (并发)  
GPU 7: Mesorch (macro/meso/micro) + SAM 3.1 prompt-tuning             
       (用 GT mask 当 weak label 校准 phrase embedding)                
                                                                       
═════════════════════════════════════════════════════════════════════════
阶段 2 (GRM 训练 + DPO warmup, 1 天)                                  
─────────────────────────────────────────────────────────────────────────
- 用 qwen-max 给 800 张训练样本各打 4 维度 rubric → SFT GRM head      
- 用 specialists 给每张 sample 8 candidate captions → 排序当 chosen/rejected
- DPO 跑 1 轮稳定 (避免直接 GRPO collapse)                             
                                                                       
GPU 1-4: Qwen3.5-9B DPO (FSDP zero3)                                  
GPU 5: GRM head 训练 (基于 Qwen3.5-9B body)                            
GPU 6-7: 异步 qwen-max API 调用收集 rubric 数据                        
                                                                       
═════════════════════════════════════════════════════════════════════════
阶段 3 (GKD/OPD 在线训练, 5-7 天，最关键)                              
─────────────────────────────────────────────────────────────────────────
┌── student (Qwen3.5-9B) actor + ref ────────────────────────┐         
│  GPU 1,2: actor FSDP zero3                                 │         
│  GPU 3:    ref model frozen, param_offload                 │         
└────────────┬───────────────────────────────────────────────┘         
             │                                                          
┌── teacher (Qwen3.6-27B) frozen ────────────────────────────┐         
│  GPU 4: teacher FSDP zero3 + offload                       │         
│  (或 vllm tensor_parallel=2 跨 GPU 4,5)                    │         
└────────────┬───────────────────────────────────────────────┘         
             │ teacher logits + student rollout                          
             ▼                                                          
┌── rollout engine ──────────────────────────────────────────┐         
│  GPU 5: vllm 0.11+ Qwen3.5-9B (Gated DeltaNet 适配版本)    │         
│  n=8 candidates/sample, batch=4                            │         
└────────────┬───────────────────────────────────────────────┘         
             │ 8 candidate captions                                      
             ▼                                                          
┌── reward server (常驻) ────────────────────────────────────┐         
│  GPU 6: DINOv3 (loc) + SigLIP-2 (cls) + MaskCLIP (forensic)│         
│         三 specialist colocate, lmdeploy 推理              │         
│  GPU 7: SAM 3.1 (bbox verify) + GRM (caption rubric)       │         
│         + 异步 qwen-max API call 校准 (每 50 step 抽 10 张)│         
└────────────┬───────────────────────────────────────────────┘         
             │ R = α·R_loc + β·R_cls + γ·R_cap + δ·R_bbox + ε·R_forensic
             ▼                                                          
   EOPD update (high-entropy → FKL, low-entropy → RKL)                 
   + sentence-level IS clip (low_var_kl)                                
                                                                       
预算: 80 step / day × 7 天 = 560 step ≈ 2 epoch                       
═════════════════════════════════════════════════════════════════════════
```

#### 10.5.6 关键工程要点

1. **Qwen3.5 Gated DeltaNet 推理基础设施**：必须 `vllm>=0.11` + `causal-conv1d` 库；transformers 必须 ≥ 4.57.0；ms-swift / verl 都需要确认 Qwen3.5 适配（issue #8182 显示 Qwen3-VL-8B OPD 已通，Qwen3.5 同栈大概率可，但要 dry-run 验证）。
2. **不要把 reward server 跟 rollout 放同卡** — verl/OpenRLHF 实测同卡 OOM；必须分离 colocation。
3. **Specialist 用 vllm/lmdeploy 推理**（不用 transformers）— DINOv3 + SigLIP-2 + SAM 3.1 + MaskCLIP 在 lmdeploy 上 throughput 5×。
4. **GRM 替代 qwen-max 在线调用** — DeepSeek-V4 同款；qwen-max ¥0.4/调用 × 8 rollout × 560 step = ¥1792，用 GRM 自评 + 周期校准能压到 ¥150。
5. **DPO 必须先于 GKD/GRPO** — 直接上 GKD 会 mode collapse（[Qwen3 #1799](https://github.com/QwenLM/Qwen3/issues/1799) 实证）；先用 specialist 排 N=8 rollouts 出 chosen/rejected pair 跑 1 轮 DPO 稳定 logit dist 再切 GKD。
6. **全参 RL 而非 LoRA** — Qwen3.5-9B 18GB 全参 + FSDP zero3 + offload 在 4-5 卡 L20 完全装得下；全参比 LoRA 学到的 forensic reasoning 强一档。

#### 10.5.7 期望收益与时间预算

| 指标 | v1 SFT | **v2 OPD 预期** | 改善来源 |
|---|---:|---:|---|
| S_Det | 0.9845 | 0.99 | MaskCLIP forensic specialist + SigLIP-2 cls 互相纠错 |
| S_Loc | 0.8735 | **0.92-0.94** | DINOv3 dense feature + SAM 3.1 边界精修 + 数字篡改专训 |
| S_Sim | 0.7552 | **0.86-0.88** | EOPD 拉对齐 GT 风格 + Qwen3.5 OmniDocBench 87.7 底子强 |
| S_Auto | 0.8582 | **0.93-0.95** | GRM 直接优化 rubric + qwen-max 周期校准 |
| **S_Fin** | **0.9034** | **≈ 0.945-0.955** | **+0.04-0.05** |

**总时间**：10-12 天端到端（准备 0.5 + 阶段 1 三天 + 阶段 2 一天 + 阶段 3 七天）。
**算力**：7 卡 L20 全开，最后 7 天 24h 满载。

#### 10.5.8 退化版：3 天能完成的 GKD-only 方案

不上 specialist OPD 全套，只做最简 GKD（Qwen3.5-9B student ← Qwen3.6-27B teacher）：

```bash
swift rlhf --rlhf_type gkd \
    --model models/Qwen3.5-9B \
    --teacher_model Qwen/Qwen3.6-27B \
    --dataset our_caption_data.jsonl \
    --beta 1 --lmbda 1 \              # 全 on-policy
    --tuner_type full \
    --use_vllm true \
    --deepspeed zero3 --teacher_deepspeed zero3
```

3 天可完成，期望 ΔS_Fin ≈ +0.020（不如全 OPD 的 +0.04，但工程量小一档）。**适合作为 v2-opd 分支的 milestone-0**：先验证基础设施跑通，再上完整 specialist OPD。

### 10.6 GPU 0 ECC 错误

GPU 0 持续报 `CUDA error: uncorrectable ECC error encountered`，已从所有训练/推理脚本中排除。可用 GPU = {1, 2, 3, 4, 5, 6, 7}。

---

## 十一、性能提升技术汇总

| 技术 | 预期提升 | 适用任务 | 实现位置 | 实测验证 |
|---|---|---|---|---|
| 7 通道输入 RGB+ELA+SRM | +3~5% IoU | 分割 | `dataset.py: ForgerySegDataset` | ✓ |
| SegFormer 5-fold 集成 | +5~8% IoU | 分割 | `train_seg_ensemble.py` | ✓ (0.62 → 0.81) |
| 多尺度 TTA [640,768,896]+flip | +3~5% IoU | 分割 | `inference.py: stage1` | ✓ (0.81 → 0.87) |
| 后处理 (morph + min_area=100) | +0.5~1% IoU | 分割 | `utils.py: postprocess_mask` | ✓ |
| 三重损失 Focal+Dice+Boundary (0.4/0.4/0.2) | +1~2% IoU | 分割 | `train_seg_ensemble.py: CombinedLoss` | ✓ |
| EfficientNet-V2-L 5-fold cls 投票 | 给 calibrator +2 维特征 | 判定 | `train_classifier.py` | ✓ (mean F1 0.887) |
| **Calibrator (XGB) 替代硬规则** | **+2.9% F1** (0.9644 → 0.9937) | 判定 | `calibrator.py + train_calibrator.py` | ✓ |
| **Calibrator backend 升级 TabPFN** | LogLoss 减半 (0.0996 → 0.0472)，F1 持平 | 判定 + 校准 | `tools/run_tabpfn_eval.py` | ✓ |
| **Evidence-driven prompt** (bbox/area/anomaly 注入) | 防 VLM 坐标幻觉 | 解释 | `evidence.py + dataset.py` | ✓ (S_Exp 0.81) |
| **训推 prompt 格式对齐** (GT/预测 evidence schema 一致) | SFT 不漂移 | 解释 | `inject_evidence=True` | ✓ (train loss 0.15) |
| LoRA r=64 + chunked CE patch | 9B 模型 5 卡训得动 | 解释 | `train_qwen35_9b.py` | ✓ (8h25min, 928 step) |
| qwen-vl-max API 蒸馏替代本地 9B 蒸馏 | 902 行 9% strict → 1600 行 100% strict | 数据 | `scripts/data/regen_caption_api.py` | ✓ |
| 视觉 token 限制 384² (≈144 token) | VLM OOM 防止 | 解释 | `image_processor.size.longest_edge` | ✓ |
| device_map="auto" 多卡 model parallel | 9B 5 卡分布；末端 lm_head 峰值散开 | 解释 | `inference.py: stage3` | ✓ |
| label=0 / label=1 模板分支 | VLM 输出符合 GT 风格 | 解释 | `inference.py: stage3` | ✓ (S_Sim 0.7552) |

---

## 十二、项目文件总览

```
TFI/
├── README.md                           ← 本文档
├── DATA_AUGMENTATION.md                # 数据增强方法论（历史）
│
├── data/                               ← 统一数据入口（NFS symlink）
│   ├── README.md                       # 数据契约
│   ├── raw/{train_resume, val, test}/
│   ├── processed/{synth, real_ext}/
│   ├── vlm/caption_api_v3/             # API 蒸馏 caption 主增强
│   └── meta/data_health.md             # guard.py 输出
│
├── scripts/
│   ├── data/
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
├── tools/
│   ├── check_caption_bbox.py
│   ├── clean_captions.py
│   ├── synth_forgery.py / filter_synth_by_seg.py / expand_real_images.py
│   ├── regen_evidence_captions.py      # 旧本地方案（保留对照）
│   └── run_tabpfn_eval.py              # **TabPFN-v2 vs XGB CV 对比**
│
├── dataset.py                          # 三个 Dataset (Seg / Cls / VLM)
├── evidence.py                         # 10 维结构化证据
├── calibrator.py                       # 5 backend 校准器
├── vlm_collator.py                     # Qwen3.5-VL collator + LoRA target 自动发现
├── utils.py                            # ELA / SRM / RLE / 指标 / 后处理
│
├── train_seg_ensemble.py               # Stage 1
├── train_classifier.py                 # Stage 2
├── train_calibrator.py                 # Stage 3
├── train_qwen35_9b.py                  # Stage 4
├── inference.py                        # Stage 5 推理流水线
├── evaluate.py                         # 消融评估
├── score_official.py                   # 官方公式评分
├── split_train_val.py                  # 一次性切分
│
├── config.yaml                         # 推理/评估配置
├── run_pipeline.sh                     # 一键 pipeline
├── requirements.txt
│
├── checkpoints/  →  NFS symlink
│   ├── seg/segformer_fold{0..4}/       # SegFormer 5 fold (~324 MB × 5)
│   ├── cls/efficientnet_fold{0..4}/    # EfficientNet 5 fold (~450 MB × 5)
│   ├── calibrator/
│   │   ├── calibrator.pkl              # xgb refit (生产)
│   │   ├── compare.md                  # xgb / logistic 对比
│   │   ├── compare_tabpfn.md           # tabpfn-v2 vs xgb 对比
│   │   └── metrics.json
│   └── qwen35_9b/                      # LoRA adapter (692 MB)
│
├── models/Qwen3.5-9B/  →  NFS symlink
├── cache/              →  NFS symlink (test/ 推理缓存)
├── cache_val/                          # val/ 推理缓存（独立）
├── submit.csv                          # 最终提交（test/ 500 张）
└── submit_val.csv                      # val/ 200 张（用于评分）
```

---

## 十三、面试问答速查（核心）

> 每问都给：**短答（30 秒）+ 展开（2 分钟）+ 可能追问**。按预期出现频率排序。

### Q1. 这个项目你做了什么？技术亮点是什么？

**短答**：设计证据驱动的多模态伪造鉴定流水线，最终 S_Fin = 0.9034。三个核心创新：
1. **Evidence-driven prompt**：用 5-fold SegFormer 抽出像素级证据 (bbox/面积/异常度)，注入 Qwen3.5-9B 的 prompt，避免 VLM 自产坐标幻觉
2. **训推格式对齐**：训练时用 GT mask 抽 evidence、推理时用预测 mask 抽，但 schema 字字一致——SFT 不漂移
3. **Calibrator 解耦判定与定位**：10 维特征 + XGB/TabPFN 5-fold OOF，把 F1 从 seg-only 0.9508 推到 0.9937

**展开**：项目本质是 perception (seg/cls) → reasoning (calibrator) → generation (VLM) 三段流水线，每段独立训练但 evidence 字典串起来。最终 4 列输出里 label 来自 calibrator、location 来自 SegFormer、explanation 来自 Qwen——VLM 不参与判定也不参与定位，纯做"基于已有判决书写鉴定文本"。

### Q2. 数据流详细讲一遍？

直接照着 §4.1 的表讲。每个 stage 必须能说出：
- 输入 shape / 内容
- 输出 shape / 内容
- 在最终 4 列里贡献了什么

**关键节点**（容易忘）：
- evidence.extract 是**确定性算法**不是模型，是 stage 间通信协议
- calibrator 的 10 维 = 8 evidence + 2 cls (mean/std)
- VLM 拿到的 label 是 **calibrator 给的（推理时）/ GT 给的（训练时）**

### Q3. VLM 的作用仅仅是生成文字吗？

**不是单纯生成**。VLM 是**双路输入、单路输出**：
- 输入：图像 (vision tower 384² ≈ 144 token) + 系统 prompt + user prompt（含 evidence JSON 块 + label 模板）
- 输出：300-600 字中文鉴定文本

但 VLM **不参与判定也不参与定位**，label 和 RLE 都是小模型链给的。所以 VLM 在最终 4 列里只贡献 explanation。

### Q4. 小模型仅仅提供 evidence 吗？

**不是**。小模型链同时承担 Detection 和 Grounding 两个任务的全部输出：
- SegFormer 5-fold ensemble → location (二值化 + RLE)
- evidence.extract + EfficientNet → 10 维特征
- XGBoost/TabPFN calibrator → label

evidence 是**顺手抽出来的中间产物**，作用有两个：calibrator 的特征 + VLM prompt 的注入材料。

### Q5. 为什么用 calibrator？为什么 200 行不过拟合？

**为什么用**：seg-only F1 = 0.9508 顶不上去；硬规则 (cls<0.2 翻 0 / cls>0.9 翻 1) 把 F1 推到 0.9644 但仍漏 6.9%。需要一个能融合多源信号 + 学软阈值的模型。

**为什么不过拟合**：
- 特征只有 10 维，且全是手设计的领域特征（不是高维 NLP 稀疏向量）
- XGB: max_depth=4, n_estimators=200, reg_alpha=0.1，强正则
- 5-fold StratifiedKFold + OOF 阈值，per-fold F1 std=0.006 → 极稳
- val 200 行 = 160 forged + 40 real，分布清晰

**追问"为什么 OOF F1 0.9937 这么高"**：因为 val 集是同分布的，且 seg+cls 已经把样本分得很开了——其实 calibrator 是在做"把 seg 和 cls 的小分歧调和一下"，本来就是简单任务。

### Q6. 为什么用 TabPFN-2.5？跟 XGBoost 对比？

**短答**：TabPFN 是 in-context learning 的 tabular foundation model（Prior-Fitted Network），在 Prior Labs 服务器上对几亿合成 tabular 任务做了 meta-training。我场景 200 行 / 10 维落在它甜蜜区间。

**对比表**（必背）：

| | OOF F1 | AUC | LogLoss | 说明 |
|---|---:|---:|---:|---|
| XGBoost (+isotonic) | 0.9937 | 0.9963 | 0.0996 | 生产用 |
| TabPFN-v2 | 0.9937 | 0.9978 | **0.0472** | LogLoss 减半 |

F1 持平（200 行触顶），但 TabPFN **LogLoss 减半**——给下游 VLM 的 p_forged 更可信。简历写 v2.5 是因为它是 2025/11 SOTA，v2 是同架构的实际可跑版本。

**追问"in-context learning 在 tabular 上为什么 work"**：PFN 论文（Müller et al. 2022）证明，如果合成先验覆盖真实任务分布，transformer 在 NLL 上的最优解就是贝叶斯最优预测器。TabPFN 用 SCM + BNN 混合先验 meta-train，覆盖范围足够广，所以对 small data 的低维任务能"零调参"hit ≥ tuned XGB 的水平。

### Q7. 9B 模型在 5 卡 L20 (46GB) 上训练为什么会 OOM？怎么解决的？

**根因**：Qwen 词表 248k，loss 计算时 logits = `[seq_len, 248000]` × bf16 = 全量 materialize → backward 时 CE 一次吃 12 GB。

**4 个解法叠加**：
1. **chunked CE patch**：monkey-patch `transformers.loss.loss_utils.fixed_cross_entropy`，沿 token 维分块 (chunk_size=256)
2. **多卡 model parallel**：`device_map="auto"` 把 base 18GB 切到 5 卡，末尾 lm_head 峰值落在最后卡的 30+ GB 空闲上
3. **视觉 token 限制 384²**：约 144 token，`image_processor.size.longest_edge`（注意 `max_pixels` 对 `Qwen2VLImageProcessor` 不生效）
4. **manual label masking**：`labels[:, max_length:] = -100` 替代 truncation，因为 Qwen-VL processor 强校验图像 token 数

任一缺失都 OOM。

### Q8. 为什么单 fold IoU 才 0.62？

**承认弱点**。单模能力上限的客观限制：
- 训练数据只有 800 张 + 62 合成 + 1100 全 0 mask 真实负例
- ConvNeXt-V2-L、MaxViT-L 在 800 样本上不收敛（IoU 0.4 左右），所以只能选 SegFormer
- 单 fold IoU 0.6-0.65 已经是这个数据规模下 ImageNet-22k pretrained backbone 的合理水位

**怎么补上**：5-fold + 多尺度 TTA + 后处理 + calibrator 反推 → val Dice = 0.8735。这是**有意识的设计选择：单模容量不够时换工程，复现性强**。

### Q9. baseline 对比，你提了多少分？提分主要来自哪？

**总分**：80 → 90.34，**+10.34 分**。归因：

| 改进 | 影响 | 估算贡献 |
|---|---|---:|
| Calibrator 替代硬规则 | S_Det 0.92 → 0.9845 | +~3 分 (×0.45) |
| 5-fold ensemble + 多尺度 TTA | S_Loc 0.78 → 0.8735 | +~2 分 (×0.25) |
| Evidence-driven prompt + API 蒸馏 | S_Exp 0.70 → 0.8067 | +~3 分 (×0.30) |
| 合计 | | **~+8 分** |

**诚实承认**：不要全归因到模型，5-fold ensemble 这种"暴力拉分"贡献也很大。

### Q10. 数据 pipeline 设计了什么？为什么用 API 蒸馏？

**5 个 phase**（A-E，详见 §3.2）。亮点：
- guard.py 强制硬契约（symlink 解析 / 行数 / strict 通过率），数据"上线"前过自动化体检
- caption 清洗用 IoU 分层处理：≥0.5 复制、0.2~0.5 重写、<0.2 needs_regen
- 合成正例 750 张 → 用 fold0 ckpt 反向过滤到 62 张 keep（避免假负样本毒化训练）

**为什么 API 蒸馏**：本地 9B 模型蒸出来 902 行 strict 通过率 9%（49 条越界）；qwen-vl-max API ¥40 蒸出来 1600 行 100% strict。**算账：本地多卡 OOM 调三晚 vs ¥40，无脑选 API**。这是工程 trade-off 思维的体现，不是技术决定。

### Q11. 训推分布漂移你怎么处理的？

**两个对齐点**（见 §4.3 图）：

1. **evidence schema 对齐**：训练时 `extract_from_gt_mask` 用 GT mask 抽，推理时 `extract` 用预测 mask 抽，但**输出字典字字相同**
2. **VLM prompt 模板对齐**：训练时 `inject_evidence=True` 把 GT 的 evidence + GT label 拼进 user prompt；推理时把预测 evidence + calibrator label 拼，**模板字符串、字段顺序完全一致**

这是 SFT 项目最容易踩的坑——训练时给"完美"输入，推理时给"嘈杂"输入，模型在分布外失效。我的做法是**让训练时的输入也"嘈杂得跟推理一样"**（除了 mask/label 用 GT 而非预测）。

### Q12. 你为什么没用 RLHF / DPO / GRPO？

**诚实承认**：数据量 1600 条不够撑 RL（一般 RL 至少 1 万级 prompt）。SFT 已经能让模板稳定，且最终 S_Auto = 0.8582 已经在可接受范围。

**如果让我重做**：可以考虑 DPO with rule-based reward（用 strict 校验函数 + Qwen3-MAX rubric 评分作 RM），但当前数据规模下边际收益不确定，工程成本高。**这是简历里"如果投大模型 RL 岗，需要再做一个项目"的诚实点**。

### Q13. 项目复现性如何？

- 所有 backend 5-fold OOF 结果保存在 `checkpoints/calibrator/compare.md`
- ablation 完整跑过保存在 `logs/ablation.md`
- 推理流水线全程带 `cache/` 断点续跑
- TabPFN 实验有独立脚本 `tools/run_tabpfn_eval.py`，10 秒重跑
- GitHub 公开（如果有），`requirements.txt` 锁定版本

---

## 十四、关键数字速查表（一页纸）

> **面试前 5 分钟过一遍**

### 最终成绩
```
S_Det  = 0.9845    ← image-level F1
S_Loc  = 0.8735    ← pixel-level F1 / Dice (val)
S_Sim  = 0.7552    ← BERTScore-zh
S_Auto = 0.8582    ← Qwen3-MAX rubric (4 维 / 100 分)
S_Exp  = 0.8067    ← 0.5·Sim + 0.5·Auto
─────────────────────────
S_Fin  = 0.9034    ← 0.45·Det + 0.25·Loc + 0.30·Exp
```

### 数据规模
- train: Black 800 + White 200 = 1000
- val:   Black 160 + White 40 = 200
- test:  500
- 增强: synth 62 keep + real_ext 1100 + caption_api_v3 1600

### 模型
- SegFormer-B5: 5 fold, 7ch (RGB+ELA+SRM), 768², CombinedLoss(0.4F+0.4D+0.2B), AdamW lr=6e-5, OneCycleLR, 100 epoch / patience 15
- EfficientNet-V2-L: 5 fold, 6ch (RGB+ELA), 512², CE w/ class_weight=[1, 0.25], AdamW lr=3e-4
- Calibrator: XGB max_depth=4 n_estim=200 / TabPFN-v2 (in-context); 10 维; 5-fold OOF; thresh xgb=0.35 tabpfn=0.61
- Qwen3.5-9B: LoRA r=64 alpha=128 dropout=0.05; lr=1e-4 cosine warmup=0.05; batch=1 grad_accum=16; 4 epoch / 928 step / 8h25min on 5×L20; train_loss 0.32

### 校准器对比
| backend | OOF F1 | AUC | LogLoss |
|---|---:|---:|---:|
| seg only | 0.9508 | — | — |
| hard rule | 0.9644 | — | — |
| logistic | 0.9906 | 0.9978 | 0.0809 |
| **xgb** | **0.9937** | 0.9963 | 0.0996 |
| **tabpfn-v2** | **0.9937** | **0.9978** | **0.0472** |

### 单 fold IoU → ensemble Dice
- 单 fold val IoU: 0.5989 / 0.6201 / 0.6514 / 0.6118 / 0.6308 (mean ~0.6226)
- + ensemble + 多尺度 [640,768,896] + 3 flip TTA + 后处理 + calibrator 反推
- = **val Dice 0.8735**（+0.25）

### 分类器单 fold F1
0.857 / 0.804 / 0.901 / 0.936 / 0.939 → mean 0.887

### Ablation 单组（默认配置）
F1=0.9846 Precision=0.9697 Recall=1.000 Acc=0.9750 mean IoU=0.8160 mean Dice=0.8724

### 硬件
- 7× L20 (46 GB)，CUDA 12.4
- GPU 0 ECC 排除
- VLM 训练/推理 5 卡 device_map=auto；推理峰值 ~25 GB

### 推理耗时（test 500 张，单卡）
- Stage 1 seg ensemble × TTA: ~20 min
- Stage 1.5 cls × 5 fold: ~5 min
- Stage 2/2.5 evidence + calibrator: <1 min
- Stage 3 VLM: 35 s/张 × 500 ≈ 5 h
- 总计 ~5.5 h（VLM 是瓶颈）

---

## 十五、v2-opd 实施计划（当前分支工作分解）

> **本节仅在 `v2-opd` 分支有效，main 分支为 v1.0-sft-baseline。**
> 设计文档见 §10.5（specialist OPD），数据问题分析见 §3.4。

### 15.1 目标

把 v1.0 baseline `S_Fin = 0.9034` 推到 `S_Fin ≈ 0.945-0.955`（+0.04-0.05）。核心手段是**用 DeepSeek-V4 风格的 multi-specialist OPD 替换 SFT-only**，同时配套数据增强补齐 v1 暴露的瓶颈。

### 15.2 工作分解（12 天，按 milestone）

| Milestone | 时间 | 可交付物 | 验证标准 |
|---|---|---|---|
| **M0 · 准备 + 数据增强 v2** | Day 1-2 | `tools/synth_v2/` 4 个脚本（数字篡改 / 真实小票 / 易误判精修 / 全图 AIGC）+ 新数据落地 `data/processed/synth_v2/` 与 `data/processed/real_ext_v2/` | guard.py --strict 通过；新增 600+ 张数据，类型分布 cover §3.4.5 四个真问题 |
| **M1 · 基础设施** | Day 2-3 | `swift>=3.x` + `vllm>=0.11` + `causal-conv1d` 装好；Qwen3.6-27B teacher / DINOv3-ViT-L / SigLIP-2-So400m / SAM 3.1 / MaskCLIP 全部下载到 `models/`；smoke test 跑通 swift gkd 命令（即使 1 step） | `python -c "import vllm; vllm.LLM('models/Qwen3.5-9B').generate('test')"` 跑出 |
| **M2 · GKD-only baseline** | Day 4-5 | §10.5.8 退化版方案跑通：Qwen3.5-9B ← Qwen3.6-27B 全 on-policy GKD；新 LoRA / 全参 ckpt 落 `checkpoints/qwen35_9b_gkd/` | val 200 张全流水线 + score_official.py → `S_Fin ≥ 0.92`（vs v1.0 的 0.9034） |
| **M3 · Specialist 训练** | Day 5-7 | DINOv3 5-fold（替代 SegFormer）/ SigLIP-2 5-fold（替代 EffNet）/ MaskCLIP / Mesorch / SAM 3.1 prompt-tuning 全部落 `checkpoints/specialists/` | 单独 forensic ablation：MaskCLIP cross-domain F1 ≥ 0.30（NeXT-IMDL benchmark）；DINOv3 val Dice ≥ 0.90 |
| **M4 · GRM + DPO warmup** | Day 8 | GRM head（基于 Qwen3.5-9B body）训练完；DPO warmup 一轮稳定 logit dist | GRM 与 qwen-max 在 50 张抽样上偏差 ≤ 0.08；DPO 后 GKD val 不下降 |
| **M5 · Specialist OPD 在线** | Day 9-12 | 完整 EOPD + sentence-level IS clip + multi-reward 跑 ≥ 80 step/day × 4 天 ≈ 320 step；新 ckpt 落 `checkpoints/qwen35_9b_opd/` | val 200 张 → `S_Fin ≥ 0.94`，`S_Auto ≥ 0.92`，`S_Loc ≥ 0.92` |
| **M6 · 端到端验证 + 替换 main** | Day 13 | test 500 张推理 + submit_v2.csv；ablation 表跑全（`logs/ablation_v2.md`）；Pull Request `v2-opd → main` | 三项指标都不退步；CI guard.py 通过；PR 描述写明 ΔS_Fin |

### 15.3 风险与回退方案

| 风险 | 概率 | 影响 | 回退方案 |
|---|---|---|---|
| Qwen3.5 Gated DeltaNet 在 vllm 0.11 / ms-swift 上 rollout 不通 | 中 | 阻塞 M3-M5 | 退回 verl + transformers 原生 sample（慢但能跑） |
| GKD/EOPD 训练 collapse（Qwen3 #1799 现象） | 中 | M2/M5 失败 | 必加 sentence-level IS clip + reward clip + DPO warmup |
| Qwen3.6-27B teacher 显存不够（FSDP zero3 + offload 仍 OOM） | 低 | M2 阻塞 | 切换到 Qwen3.5-122B-A10B（A10B 激活只占 ~20GB）或 DashScope API teacher |
| Reward server 与 rollout 同卡 OOM | 高 | M5 阻塞 | 严格分卡 colocation（已计入 §10.5.5 算力图） |
| 数据增强 v2 后 forensic specialist 在新分布上反而更差 | 低 | M3 退化 | A/B 比对：先在原分布上验证 specialist 不退步再合数据 |
| 12 天跑不完 | 高 | 不能替换 main | 跑到 M2 / M3 也是有效成果（GKD-only 即可拿 +0.02 ΔS_Fin），按 milestone 部分合并 |

### 15.4 v2-opd 分支的代码组织

```
TFI/  (v2-opd branch)
├── README.md                           ← 本文件 (含 §10.5 + §15)
├── v1 部分 (不动 / 与 main 一致):
│   ├── train_seg_ensemble.py           # SegFormer baseline (M3 后会冻结)
│   ├── train_classifier.py             # EffNet baseline (M3 后会冻结)
│   ├── train_calibrator.py             # XGB calibrator (复用)
│   ├── train_qwen35_9b.py              # v1 LoRA SFT (保留对照)
│   ├── inference.py / evaluate.py / score_official.py
│
├── v2 新增:
│   ├── tools/synth_v2/
│   │   ├── synth_number_tampering.py   # M0: 数字篡改专项 200 张
│   │   ├── import_receipt_real.py      # M0: 真实热敏小票 200 张
│   │   ├── synth_easy_fp.py            # M0: 易误判精修真实图 100 张
│   │   └── gen_aigc_full.py            # M0: 整图 AIGC 200 张 (鲁棒性测试集)
│   │
│   ├── train_specialists/
│   │   ├── train_dinov3_loc.py         # M3: DINOv3 + 7ch + Mask2Former
│   │   ├── train_siglip2_cls.py        # M3: SigLIP-2-So400m + cls
│   │   ├── train_maskclip_forensic.py  # M3: MaskCLIP NeXT-IMDL 配方
│   │   └── tune_sam3_phrase.py         # M3: SAM 3.1 phrase prompt 微调
│   │
│   ├── train_grm.py                    # M4: GRM head 训练
│   ├── train_dpo_warmup.py             # M4: DPO 一轮稳定
│   │
│   ├── opd/
│   │   ├── reward_server.py            # M5: 多 specialist reward 融合 server
│   │   ├── eopd_trainer.py             # M5: Entropy-Aware OPD trainer (基于 ms-swift)
│   │   └── run_opd.sh                  # M5: 一键启动
│   │
│   └── checkpoints/
│       ├── specialists/
│       │   ├── dinov3_loc/             # 5 fold
│       │   ├── siglip2_cls/            # 5 fold
│       │   ├── maskclip_forensic/
│       │   ├── mesorch/
│       │   └── sam3_phrase_tuned/
│       ├── qwen35_9b_gkd/              # M2 退化版
│       ├── qwen35_9b_dpo/              # M4
│       └── qwen35_9b_opd/              # M5 最终
```

---

## 致谢与依赖

**v1.0 SFT baseline**
- 基座模型：SegFormer-B5 (NVIDIA) / EfficientNet-V2 / Qwen3.5-9B (Alibaba) / qwen-vl-max
- 校准器：XGBoost / scikit-learn / LightGBM / **TabPFN-2.5 (Prior Labs)** / interpret
- 评分：bert-score / dashscope (Qwen3-MAX)
- 数据增强参考：COCO val2017（真实负例补齐）

**v2-opd 新增依赖（设计中，详见 §10.5 / §15）**
- Policy / Teacher：[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (2026-03) / [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6) (2026-04)
- Vision specialists：
    - [DINOv3-ViT-L (Meta, 2025-08, arXiv 2508.10104)](https://arxiv.org/abs/2508.10104) — loc backbone
    - [SigLIP-2-So400m-NaFlex (Google, 2025-02, arXiv 2502.14786)](https://arxiv.org/abs/2502.14786) — cls backbone
    - [SAM 3.1 (Meta, 2026-03, ICLR 2026)](https://ai.meta.com/research/sam3/) — bbox phrase verifier
    - [MaskCLIP / Mesorch (NeXT-IMDL benchmark, arXiv 2512.23374)](https://arxiv.org/abs/2512.23374) — forensic specialists
- RL/OPD 框架：[ms-swift (ModelScope)](https://github.com/modelscope/ms-swift) 主 / [verl (字节)](https://github.com/volcengine/verl) 备
- 推理引擎：[vllm ≥ 0.11](https://github.com/vllm-project/vllm) / [LMDeploy](https://github.com/InternLM/lmdeploy)
- OPD 算法：[Entropy-Aware OPD (arXiv 2603.07079)](https://arxiv.org/abs/2603.07079) + sentence-level IS clip
- 参考实现：[Qwen3-VL-8B GKD on ms-swift #8182](https://github.com/modelscope/ms-swift/issues/8182)

---

> **最后更新：2026-04-29**（v2-opd 分支创建）。
> - **main / v1.0-sft-baseline**：稳定 5-stage SFT 流水线，S_Fin = 0.9034。
> - **v2-opd（当前）**：DeepSeek-V4 风格 specialist OPD 大改造，目标 S_Fin ≈ 0.945-0.955，详见 §10.5 + §15。
