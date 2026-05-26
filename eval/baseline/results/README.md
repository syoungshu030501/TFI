# M(-1) Prompt-Only Baseline 评测结果

> 截至 **2026-05-01**，4 组 baseline 已全部跑完并完成 judge 评分。
> 本文档说明各项指标含义、4 组 baseline 设置与结果，以及对后续 v2 OPD/GKD 路线的指示。

---

## 1. 评测体系（两套指标，必须区分）

本仓库实际在用两套指标体系，**不可混淆**：

### 1.1 官方 S_Fin（4 维加权 0–1，决定竞赛排名）

复现脚本：`score_official.py`，用于生成提交后的真实排名预估（v1 SFT 在此体系拿到 **S_Fin = 0.9034**）。

| 子项 | 计算方式 | 取值 | 权重 | 现有 v1 SFT 得分 |
|---|---|---|---|---|
| `S_Det` | image-level F1（二分类：forged vs real） | 0–1 | 0.45 | 0.9845 |
| `S_Loc` | pixel-level F1 = **Dice** on forged samples | 0–1 | 0.25 | 0.8735 |
| `S_Sim` | **BERTScore-zh F1**（生成 explanation vs GT caption） | 0–1 | — | 0.7552 |
| `S_Auto` | **Qwen3-MAX rubrics**（100 分 → /100） | 0–1 | — | 0.8582 |
| `S_Exp` | `0.5·S_Sim + 0.5·S_Auto` | 0–1 | 0.30 | 0.8067 |
| **`S_Fin`** | `0.45·S_Det + 0.25·S_Loc + 0.30·S_Exp` | 0–1 | — | **0.9034** |

`S_Auto` 内部 4 维度（Qwen3-MAX rubrics 100 分制）：
- 内容准确性（30 分）：label/事实是否对，关键事实（品牌、金额、日期、场景）是否一致
- 证据具体度（30 分）：是否引用具体 bbox + 像素特征（字体/边缘/纹理/光照/JPEG 伪影/噪声）
- 推理逻辑性（20 分）：论证链是否完整连贯
- 表达专业性（20 分）：用词专业度 + 结构清晰度 + 长度 300–600 字

> **代价**：S_Auto 要花 qwen-max API（按 token 计费），所以平时迭代用下面的 Judge 体系替代。

### 1.2 R1-Distill-70B Judge（本地 4 维 1–10，用于 baseline 对比）

替代脚本：`tools/baseline/judge_absolute_scoring.py`，用 **DeepSeek-R1-Distill-Llama-70B**（本地 vllm TP=4，零 API 费用）做 4 维度 1–10 整数打分。

| 维度 | 含义 | 与官方 S_Auto 对应关系 |
|---|---|---|
| `accuracy` | label / bbox / 事实陈述与 GT 一致 | ≈ 内容准确性 |
| `evidence` | 是否引用具体 bbox + 可核验视觉/逻辑证据 | ≈ 证据具体度 |
| `completeness` | 覆盖 GT 提到的关键篡改点（label=0 时评估真实性论证多维度） | ≈ 推理逻辑性的扩展 |
| `language` | 中文流畅度 + 专业术语 + 长度 300–600 字 | ≈ 表达专业性 |
| `overall` | 4 维度算术平均 | — |

> **本文档所有数字都是 1.2 体系（1–10 整数）**，不是官方 S_Fin（0–1）。
> 两套体系不直接可比，只在 baseline 间相对比较时用 1.2，最终提交才上 1.1。

---

## 2. 4 组 Baseline 设置

| 名称 | 模型 | 训练 | Prompt | 推理框架 | GPU | 用途 |
|---|---|---|---|---|---|---|
| **sft** | Qwen3.5-9B | v1 LoRA r=64 SFT（5 stage） | evidence-injected | transformers | 1 | **v2 必须超过的 baseline**（v1 final，S_Fin=0.9034） |
| **zs** | Qwen3.5-9B | 无（原始权重） | zero-shot 严格 JSON schema | transformers | 1 | **天花板下限**：什么都不调能拿多少 |
| **fs** | Qwen3.5-9B | 无 | zero-shot + 8-shot（4 forged + 4 real，从 train 采样） | transformers | 1 | 测 in-context learning 能不能逼近 SFT |
| **cot** | Qwen3.5-9B | 无 | zero-shot + chain-of-thought（先思考再 JSON） | transformers | 1 | 测显式推理能不能逼近 SFT |

样本：val 全集 200 张（forged=160, real=40，与 v1 评估一致）。

Judge：`/mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b`（132 GB），vllm TP=4 跑 GPU 4–7。

---

## 3. 结果（Judge 1–10 体系）

| 设置 | n | accuracy | evidence | completeness | language | **overall** | Δ vs sft |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **sft** | 200 | **8.085** | **7.930** | **7.380** | 8.570 | **7.991** | — |
| cot | 200 | 4.025 | 4.895 | 4.825 | 7.055 | 5.200 | **−2.79** |
| fs  | 200 | 4.275 | 4.645 | 4.805 | 6.910 | 5.159 | **−2.83** |
| zs  | 200 | 3.655 | 4.215 | 4.280 | 7.395 | 4.886 | **−3.11** |

### 3.1 关键观察

1. **SFT 比 prompt-only 高 ≈ 2.8 分**（5.2 → 8.0），微调收益巨大，v2 OPD 投入有数据支撑。
2. **CoT 只比 zs 高 0.31**（4.89 → 5.20），prompt-tuning 触顶很快，再优化 prompt 边际收益已极小。
3. **fs 在 language 维度反而下降**（zs 7.40 → fs 6.91）：few-shot 让模型模仿例子格式而非自由组织语言，证明 fs **不是 prompt-only 的真天花板**。
4. **三组 prompt-only 在 accuracy/evidence/completeness 三维都集中在 4–5 分**：原始模型缺的不是表达能力（language 7+），而是**专业判别能力 + 证据抽取能力**——这正是 SFT/OPD 该补的洞。
5. **language 是 prompt-only 唯一不太弱的维度**（7+），说明大模型本身中文表达没问题，只要专业知识跟上，整体得分会快速上去。

### 3.2 对 v2 OPD 路线的指示

- prompt 类优化（CoT/fs）已经无空间，**必须走训练（SFT/GKD/RLVR）**
- 微调能补的核心是 accuracy + evidence 两维，所以 v2 的 reward 设计里：
  - `R_label_gt`（accuracy）+ `R_phrase_check`（evidence）+ specialist verifier（accuracy/evidence）应占 reward 权重大头 ✓ 已在 README §三 reward 设计中体现
  - language 维度不需要专门 reward
- **缺一组关键对照**：Qwen3.6-27B prompt-only ceiling（同 zs/cot 协议），决定"换大模型"是否能甩 SFT，影响 v2 是否有必要继续 9B 蒸馏路线。**该实验已下载 Qwen3.6-27B 权重并准备就绪，待跑。**

---

## 4. 复现命令

### 4.1 重跑 prompt-only 推理（已有 cache，删 raw/ 才会重跑）

```bash
cd /home/young/TFI
bash tools/baseline/run_all.sh
# 等价于：
# python tools/baseline/prompt_only_baseline.py --mode zs  --gpu 1
# python tools/baseline/prompt_only_baseline.py --mode fs  --gpu 1
# python tools/baseline/prompt_only_baseline.py --mode cot --gpu 1
# 然后 judge_absolute_scoring.py 用 GPU 4-7 TP=4 跑 70B judge
```

### 4.2 单独重跑 judge（不重做推理）

```bash
python tools/baseline/judge_absolute_scoring.py \
    --pred_csvs sft=submit_val.csv \
                zs=tools/baseline/results/zs/predictions.csv \
                fs=tools/baseline/results/fs/predictions.csv \
                cot=tools/baseline/results/cot/predictions.csv \
    --judge_model /mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b \
    --gpus 4,5,6,7 \
    --out_dir tools/baseline/results/judge
```

### 4.3 用官方 S_Fin 体系评（提交前用）

```bash
python score_official.py --pred_csv submit_val.csv --val_dir data/raw/val --gpu 7
# 需要 DASHSCOPE_API_KEY，或 --qwen_model none 跳过 S_Auto
```

---

## 5. 文件清单

```
tools/baseline/results/
├── README.md            ← 本文件
├── judge/
│   ├── summary.csv      ← 4 组 baseline overall 对比
│   ├── report.md        ← Judge 自动生成的 markdown 报告
│   ├── sft_judge.csv    ← 200 张 per-sample 4 维度评分
│   ├── zs_judge.csv
│   ├── fs_judge.csv
│   ├── cot_judge.csv
│   └── cache/           ← 每张图的 raw judge 输出，断点续跑用
├── zs/  fs/  cot/
│   ├── predictions.csv  ← 推理结果（image_name,label,location-RLE,explanation）
│   ├── raw/*.json       ← 每张图的原始 model output，断点续跑用
│   └── run.log
└── _logs/               ← 各阶段 stdout/stderr
```
