# TFI · 证据驱动的图像伪造鉴定系统 — v2-opd

> 电商竞赛三任务：**伪造判别 (Detection) / 伪造定位 (Grounding) / 可解释分析 (Explanation)**
>
> 本分支 (`v2-opd`) 是基于 v1 SFT baseline 的下一代设计，目标 `S_Fin ≈ 0.945-0.955`（v1 = 0.9034）。
>
> v1 的完整方案、复现命令、所有训练记录见 [`reference/README.v1.md`](reference/README.v1.md)。

## 分支与版本

| 引用 | Commit | 内容 |
|---|---|---|
| `tag v1.0-sft-baseline` | `63848ee` | v1 稳定 5-stage SFT 流水线，**S_Fin = 0.9034**（val 200） |
| `branch main` | `63848ee` | 同 v1.0-sft-baseline，永远是可复现稳定版 |
| `branch v2-opd` ← **当前** | head | DeepSeek-V4 风格 specialist-verified RLVR + GKD 大改造 |

> v2-opd 完成 §6 验收标准后通过 PR 合并回 main。

---

## 一、v1 → v2 的核心改动

| 维度 | v1 baseline | **v2-opd** | 改动理由 |
|---|---|---|---|
| **输入通道** | 7ch (RGB+ELA+SRM) | **3ch RGB only** | ELA/SRM 在热敏小票/发票上结构性误报（v1 的 4 张 FP 全是热敏小票，§3.4）；现代 SSL backbone (DINOv3/SigLIP-2) 自学的 dense feature 已覆盖 ELA 信号且具域感知；强行改 stem 还破坏 pretrained feature |
| **Loc backbone** | SegFormer-B5 5-fold | **DINOv3-ViT-L** (Meta 2025-08, frozen pretrained) + Mask2Former-light head | DINOv3 (arXiv 2508.10104) Gram anchoring 解决 v1 SegFormer 100 epoch 后 IoU 0.62 不动；零改 stem 保留 RGB pretrained 几何 |
| **Cls backbone** | EfficientNet-V2-L 5-fold | **SigLIP-2-So400m-NaFlex** (Google 2025-02) + MLP head | NaFlex 保留 native aspect ratio（小票 aspect ≈ 0.3，方形 resize 把关键信号 crop 掉）；caption-pretrain 对语义级伪造敏感 |
| **Forensic 信号** | 拼进 stem | **MaskCLIP + Mesorch 独立 specialist** (NeXT-IMDL benchmark 王者) | 域感知强 + 可独立替换/淘汰，不污染 backbone |
| **Bbox grounding** | 无 | **SAM 3.1** (Meta 2026-03, ICLR 2026) | text phrase → mask 单模型，30ms / 100+ objects，验证 caption 提到的 region 是不是真在那里 |
| **Policy LM** | Qwen3.5-9B + LoRA r=64 SFT | **Qwen3.5-9B 全参** + GKD (Qwen3.6-27B teacher) → RLVR | 同模型升级到全参 + 蒸馏；本地权重已就位；OmniDocBench 87.7 强基础 |
| **Teacher** | 无 | **Qwen3.6-27B Dense** (主) + Qwen3.5-122B-A10B (rejection-sampling pool 多样性) | 27B vs 9B 容量 gap 充分；A10B MoE 在 OmniDoc 上跟 9B 几乎平 → 不当主 teacher |
| **后训练** | LoRA SFT only | **rule-based RLVR + GKD + EOPD** | rule-based 占 55% 权重提供不可 hack 的 reward floor；EOPD + sentence-level IS clip 防 collapse |

> **术语澄清**：v1 README §10.5 写"specialist OPD"不严谨。严格说：Qwen3.6→3.5 那段是**真正的 OPD/GKD**（大→小 logit KL），而 specialists 给的是**标量 reward 不是 logit**，属 RLVR (Reinforcement Learning with Verifiable Rewards) 范式，verifier 比 policy 小是正常（DeepSeek-R1 / OpenAI o1 / AlphaGo 都这样）。

---

## 二、v1 暴露的真实瓶颈（v2 必须解决）

详细数据来自 v1 `logs/val_per_sample.csv` 的 200 张 per-sample 拆解：

1. **数字篡改 Dice 仅 0.663**（最差子类，但占伪造样本 12.3%）—— 数据增强里数字篡改样本严重不足
2. **小掩码 (<0.5% GT 占比) Dice 0.645** —— SegFormer 768² 输入分辨率不够 + 7ch stem 丢失高频细节
3. **4 张 FP 全是热敏小票** —— ELA 在低对比/单色调介质上结构性误报
4. **AIGC 小目标 mean 4.3% mask coverage** —— 大多是"局部 AI 生成 + 真实背景"，但合成数据是整图 AI 生成 → 训练分布与测试分布严重错位

---

## 三、v2 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  TEACHER (frozen)                                                    │
│   主: Qwen3.6-27B Dense (vllm TP=2, gap 充足且显存友好)              │
│   辅: Qwen3.5-122B-A10B (rejection-sampling pool, 多样性来源)        │
└────────────────────────┬────────────────────────────────────────────┘
                         │ token-level logit KL (大→小, 真 OPD/GKD)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STUDENT / POLICY                                                    │
│   Qwen3.5-9B (本地已下载) 全参 FSDP zero3                           │
│   ↓ rollout n=8 candidates / sample (vllm 0.11+, GDN 适配)          │
└────────────────────────┬────────────────────────────────────────────┘
                         │ candidate (label, location, explanation)
                         ▼
┌──── REWARD = w·R_xxx, sum 1.0, rule-based ≥ 55% ────────────────────┐
│                                                                      │
│  Rule-based (不可 hack, 55%)                                         │
│   ├─ R_format       0.10  schema/JSON/长度/关键词/bbox 合法性       │
│   ├─ R_consistency  0.10  label-location 一致 + bbox⊆location        │
│   ├─ R_label_gt     0.15  pred_label == gt (训练时)                  │
│   ├─ R_iou_gt       0.15  mask IoU(pred, gt)                         │
│   └─ R_phrase_check 0.05  caption 提到的数字/region 是否在 GT 中     │
│                                                                      │
│  Specialist verifiers (30%, 与 policy 异源)                          │
│   ├─ R_loc          0.10  DINOv3 (RGB only)                          │
│   ├─ R_cls          0.10  SigLIP-2-So400m-NaFlex                     │
│   └─ R_forensic     0.10  MaskCLIP + Mesorch (NeXT-IMDL SOTA)        │
│                                                                      │
│  Caption rubric (15%, model-based, 易 hack 故权重低)                 │
│   ├─ R_grm          0.10  GRM 自评 (DeepSeek-V4 同款)                │
│   └─ R_qwen_periodic 0.05  qwen-max 每 50 step 抽 10 张校准           │
└────────────────────────┬────────────────────────────────────────────┘
                         │ 多 reward 融合 (trust-region: 单一 ≤ 25%)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EOPD UPDATE (Entropy-Aware OPD, arXiv 2603.07079)                  │
│    high-entropy token → forward KL  (防 mode collapse)               │
│    low-entropy token  → reverse KL  (DeepSeek-V4 默认, 收敛快)       │
│    + sentence-level IS clip + low_var_kl (防 Qwen3 issue #1799 崩溃) │
│    + SFT KL 拉项 (防 policy 跑飞)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、Specialist 选型（基于 2025-2026 真实 SOTA）

| 角色 | 模型 | 来源 / 选型理由 |
|---|---|---|
| **Loc backbone** | [DINOv3-ViT-L](https://arxiv.org/abs/2508.10104) (蒸馏自 7B) + Mask2Former-light | Meta 2025-08；首次单一 frozen SSL backbone 在 dense prediction 超 SegFormer 类专门方案；商业许可 |
| **Cls backbone** | [SigLIP-2-So400m-NaFlex](https://arxiv.org/abs/2502.14786) (400M) + MLP head | Google 2025-02；NaFlex 保留 aspect ratio + caption-pretrain + masked-prediction；超 EVA-CLIP / AIMv2 |
| **Forensic SOTA** | **MaskCLIP** + **Mesorch** 双路 | [NeXT-IMDL benchmark (arXiv 2512.23374)](https://arxiv.org/abs/2512.23374) 跨域 F1：MaskCLIP 0.32 vs IML-ViT 0.12 vs TruFor 0.13；ForensicHub (NeurIPS 2025) 现成代码 |
| **Bbox phrase verifier** | [SAM 3.1](https://ai.meta.com/research/sam3/) | Meta 2026-03 ICLR 2026；text phrase 直接出 mask；零样本 LVIS AP 48.8 vs SAM2 38.5 |
| **Caption rubric reward** | GRM (DeepSeek-V4 同款) + qwen-max 周期校准 | actor 自评 rubric 不用每步调 API；只在 step % 50 抽 10 样本调 qwen-max 校准漂移 |
| **OPD 算法** | EOPD + sentence-level IS clip + low_var_kl | [arXiv 2603.07079](https://arxiv.org/abs/2603.07079)；[Qwen3 issue #1799](https://github.com/QwenLM/Qwen3/issues/1799) 报告无 clip 时 collapse |
| **训练栈** | ms-swift 主 / verl 备 | [ms-swift #8182](https://github.com/modelscope/ms-swift/issues/8182) 已在 Qwen3-VL-8B 跑通同款配置 |

---

## 五、7×L20 (GPU 0 ECC 坏, 6 卡 × 46GB = 276GB) 算力分配

```
═════════════════════════════════════════════════════════════════════════
M(-1) · prompt-only baseline (1-2 天)                  ← 当前正在做
─────────────────────────────────────────────────────────────────────────
- Qwen3.5-9B + 3 个 prompt 变体 (zero-shot / 8-shot / CoT) × 200 val
- DeepSeek-R1-Distill-Llama-70B 当 judge 做 absolute 4 维度评分
- 输出: tools/baseline/results.md
- 决定: prompt 上限 vs SFT 真实增益, 给 v2 OPD ceiling 预估
- GPU: 1 (prompt-only inference) + 4-7 (judge vllm TP=4)
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M0 · 数据增强 v2 + 基础设施 (2-3 天)
─────────────────────────────────────────────────────────────────────────
- tools/synth_v2/synth_number_tampering.py (数字篡改 200 张, 解决 §2.1)
- tools/synth_v2/import_receipt_real.py    (热敏小票真实样本 200 张, 解决 §2.3)
- tools/synth_v2/synth_easy_fp.py          (易误判精修 100 张)
- tools/synth_v2/gen_aigc_local.py         (局部 AIGC 200 张, 解决 §2.4)
- 下载: Qwen3.6-27B / DINOv3-ViT-L / SigLIP-2-So400m / SAM 3.1 / MaskCLIP
- 装栈: ms-swift>=3.x + vllm>=0.11 + causal-conv1d (Qwen3.5 GDN 适配)
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M1 · GKD-only baseline (2 天) — 退化版, 验证基础设施跑通
─────────────────────────────────────────────────────────────────────────
GPU 1-4: Qwen3.5-9B 全参 ← Qwen3.6-27B teacher GKD (FSDP zero3 + grad_ckpt)
         beta=1 lmbda=1 全 on-policy, ms-swift swift rlhf --rlhf_type gkd
GPU 5-6: 同时跑 specialist 训练 (DINOv3 5-fold 轮转)
GPU 7:   reward server / vllm rollout 备
验收: val 200 → S_Fin ≥ 0.92 (vs v1 的 0.9034)
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M2 · Specialist 训练 (2 天)
─────────────────────────────────────────────────────────────────────────
GPU 1: DINOv3-ViT-L + Mask2Former-light, 5-fold 轮转
GPU 2: SigLIP-2-So400m + cls head, 5-fold 并发
GPU 3: MaskCLIP fine-tune (NeXT-IMDL 配方)
GPU 4: Mesorch (macro/meso/micro 三路)
GPU 5: SAM 3.1 phrase prompt-tuning (用 GT mask 校准 phrase embedding)
GPU 6-7: GKD baseline 持续训练 (overlap)
验收: MaskCLIP 跨域 F1 ≥ 0.30 (NeXT-IMDL); DINOv3 val Dice ≥ 0.90
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M3 · GRM + DPO warmup (1 天)
─────────────────────────────────────────────────────────────────────────
- 用 qwen-max 给 800 张训练样本各打 4 维度 rubric → SFT GRM head (Qwen3.5-9B body)
- specialists 给每张 sample 8 candidates → 排序当 chosen/rejected
- DPO 跑 1 轮稳定 logit dist
GPU 1-4: Qwen3.5-9B DPO (FSDP zero3)
GPU 5: GRM head 训练
GPU 6-7: 异步 qwen-max API 收集 rubric 数据
验收: GRM 与 qwen-max 在抽样 50 张上偏差 ≤ 0.08; DPO 后 val 不退步
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M4 · Specialist-verified RLVR + EOPD 在线 (5-7 天, 最关键)
─────────────────────────────────────────────────────────────────────────
┌── student (Qwen3.5-9B) actor + ref ────────────────────────┐
│  GPU 1,2: actor FSDP zero3                                 │
│  GPU 3:    ref model frozen (param_offload)                │
└────────────┬───────────────────────────────────────────────┘
┌── teacher (Qwen3.6-27B) frozen ────────────────────────────┐
│  GPU 4: teacher FSDP zero3 + offload (or vllm TP=2)        │
└────────────┬───────────────────────────────────────────────┘
┌── rollout engine ──────────────────────────────────────────┐
│  GPU 5: vllm 0.11+ Qwen3.5-9B (Gated DeltaNet 适配)        │
│  n=8 candidates/sample, batch=4                            │
└────────────┬───────────────────────────────────────────────┘
┌── reward server (常驻) ────────────────────────────────────┐
│  GPU 6: rule-based eval + DINOv3 + SigLIP-2 + MaskCLIP     │
│         (lmdeploy colocate)                                │
│  GPU 7: SAM 3.1 (bbox verify) + GRM + 异步 qwen-max 校准   │
└────────────┬───────────────────────────────────────────────┘
             │ R = Σ w_i·R_i (rule 55% + specialist 30% + caption 15%)
             ▼
   EOPD update + sentence-level IS clip + SFT KL 拉项
预算: 80 step / day × 7 天 ≈ 560 step ≈ 2 epoch
验收: val S_Fin ≥ 0.94, S_Auto ≥ 0.92, S_Loc ≥ 0.92
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M5 · 端到端 + PR 合并 main (1 天)
─────────────────────────────────────────────────────────────────════════
- test 500 张推理 → submit_v2.csv
- ablation_v2.md (各组件单独剔除)
- gh pr create --base main --head v2-opd
═════════════════════════════════════════════════════════════════════════
```

---

## 六、当前 milestone 进度

| Milestone | 状态 | 关键产出 |
|---|---|---|
| **M(-1) prompt-only baseline** | 🚧 进行中 | judge 模型 (DS-R1-Distill-Llama-70B) 下载中 (PID 1816528, /mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b/)；3 个 prompt 变体脚本就绪 |
| M0 数据增强 v2 + 基础设施 | ⏳ pending | — |
| M1 GKD-only | ⏳ pending | — |
| M2 Specialists | ⏳ pending | — |
| M3 GRM + DPO | ⏳ pending | — |
| M4 RLVR + EOPD | ⏳ pending | — |
| M5 PR 合并 | ⏳ pending | — |

---

## 七、风险与回退

| 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|
| Qwen3.5 GDN 在 vllm 0.11 / ms-swift 不通 | 中 | 阻塞 M1+ | 回退 verl + transformers 原生 sample（慢） |
| EOPD/GKD collapse (Qwen3 #1799) | 中 | M1/M4 失败 | 必加 sentence-level IS clip + reward clip + DPO warmup |
| Qwen3.6-27B teacher OOM | 低 | M1 阻塞 | 切 Qwen3.5-122B-A10B (A10B 激活只占 20GB) 或 vllm offload |
| Reward server 与 rollout 同卡 OOM | 高 | M4 阻塞 | 严格分卡 (已计入 §五) |
| 数据增强 v2 后 specialist 反而变差 | 低 | M2 退化 | A/B 比对：先在原分布上验证 specialist 不退步再合数据 |
| 12 天跑不完 | 高 | 不能替换 main | 跑到 M1 / M2 也算成果 (GKD-only +0.02 ΔS_Fin)，按 milestone 部分合并 |

---

## 八、复现 v1 baseline

v1 的所有训练脚本、checkpoint、超参、评测、踩坑记录见 [`reference/README.v1.md`](reference/README.v1.md)（共 1222 行）。最佳成绩 `S_Fin = 0.9034`：

```
S_Det  = 0.9845   image-F1
S_Loc  = 0.8735   pixel-F1 / Dice
S_Sim  = 0.7552   BERTScore-zh
S_Auto = 0.8582   Qwen3-MAX rubrics (4 维 / 100 分)
S_Exp  = 0.8067   = 0.5·Sim + 0.5·Auto
─────────────────
S_Fin  = 0.9034   = 0.45·Det + 0.25·Loc + 0.30·Exp
```

复现：`git checkout v1.0-sft-baseline` 后按 `reference/README.v1.md` §九。

---

## 九、目录结构

```
TFI/  (v2-opd branch)
├── README.md                        本文件
├── reference/
│   └── README.v1.md                 v1 完整文档 (1222 行)
│
├── v1 部分 (与 main 一致, M5 前不动):
│   ├── train_seg_ensemble.py        SegFormer baseline
│   ├── train_classifier.py          EffNet baseline
│   ├── train_calibrator.py          XGB calibrator
│   ├── train_qwen35_9b.py           v1 LoRA SFT
│   ├── inference.py / evaluate.py / score_official.py
│   └── checkpoints/{seg,cls,calibrator,qwen35_9b}/
│
├── v2 新增 (本分支):
│   ├── tools/baseline/              ← M(-1) 当前正在写
│   │   ├── prompt_only_baseline.py  3 个变体: zero-shot / 8-shot / CoT
│   │   ├── judge_absolute_scoring.py DS-R1-Distill-Llama-70B 评分
│   │   └── prompts/                 prompt 模板 + few-shot 例子
│   │
│   ├── tools/synth_v2/              ← M0
│   │   ├── synth_number_tampering.py
│   │   ├── import_receipt_real.py
│   │   ├── synth_easy_fp.py
│   │   └── gen_aigc_local.py
│   │
│   ├── train_specialists/           ← M2
│   │   ├── train_dinov3_loc.py
│   │   ├── train_siglip2_cls.py
│   │   ├── train_maskclip_forensic.py
│   │   └── tune_sam3_phrase.py
│   │
│   ├── train_grm.py                 ← M3
│   ├── train_dpo_warmup.py          ← M3
│   │
│   ├── opd/                         ← M4
│   │   ├── reward_server.py
│   │   ├── eopd_trainer.py
│   │   └── run_opd.sh
│   │
│   └── checkpoints/
│       ├── specialists/{dinov3_loc,siglip2_cls,maskclip,mesorch,sam3_phrase}/
│       ├── qwen35_9b_gkd/           M1
│       ├── qwen35_9b_dpo/           M3
│       └── qwen35_9b_opd/           M4 final
│
└── /mnt/nfs/young/TFI/              (NFS, 大文件)
    ├── judge_model/r1-distill-llama-70b/
    └── logs/
```

---

## 致谢与依赖

**v1 (reference/README.v1.md)**: SegFormer-B5 / EfficientNet-V2 / Qwen3.5-9B / qwen-vl-max / XGBoost / TabPFN-2.5 / bert-score / dashscope

**v2 新增**:
- Policy / Teacher：[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) / [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6) (2026-04)
- Vision specialists (RGB-only)：
    - [DINOv3-ViT-L (Meta, arXiv 2508.10104)](https://arxiv.org/abs/2508.10104)
    - [SigLIP-2-So400m-NaFlex (Google, arXiv 2502.14786)](https://arxiv.org/abs/2502.14786)
    - [SAM 3.1 (Meta, ICLR 2026)](https://ai.meta.com/research/sam3/)
    - [MaskCLIP / Mesorch (NeXT-IMDL benchmark, arXiv 2512.23374)](https://arxiv.org/abs/2512.23374)
- Judge for prompt-only baseline：[DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- RL/OPD 框架：[ms-swift](https://github.com/modelscope/ms-swift) 主 / [verl](https://github.com/volcengine/verl) 备
- 推理引擎：[vllm ≥ 0.11](https://github.com/vllm-project/vllm) / [LMDeploy](https://github.com/InternLM/lmdeploy)
- OPD 算法：[Entropy-Aware OPD (arXiv 2603.07079)](https://arxiv.org/abs/2603.07079) + sentence-level IS clip
- 参考实现：[ms-swift #8182](https://github.com/modelscope/ms-swift/issues/8182)

---

> **更新**：2026-04-29，v2-opd 分支主 README 重写。v1 完整文档已迁移至 `reference/README.v1.md`。
