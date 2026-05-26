# TFI · 证据驱动的图像伪造分析系统 — v2-opd

> **图像伪造分析比赛**三任务：**伪造判别 (Detection) / 伪造定位 (Grounding) / 可解释分析 (Explanation)**
>
> 本分支 (`v2-opd`) 是基于 v1 SFT baseline 的下一代设计，目标 `S_Fin ≈ 0.945-0.955`（v1 = 0.9034）。

本 README 聚焦**技术架构 / 环境 / 参数配置 / 代码文件讲解**四个部分；
日常实验日志、状态更新、踩坑记录全部移到 [`journal.md`](journal.md)；
接手须知（含一键启动）见 [`HANDOVER.md`](HANDOVER.md)。

## 目录

- [一、分支与版本](#一分支与版本)
- [二、技术架构](#二技术架构)
  - [2.1 v1 → v2 的核心改动](#21-v1--v2-的核心改动)
  - [2.2 v2 整体架构图](#22-v2-整体架构图)
  - [2.3 Specialist 选型](#23-specialist-选型)
  - [2.4 Veritas / HydraFake 整合](#24-veritashydrafake-整合)
- [2.5 RL 算法选型：FIPO](#25-rl-算法选型fipo)
- [2.6 Bbox 坐标约定（[0,1000]² 归一化）](#26-bbox-坐标约定01000-归一化)
- [2.7 路线 A：底座切回 Qwen3.5-9B（与 v1 / Qwen3.6-27B teacher 对齐）](#27-路线-a底座切回-qwen35-9b与-v1--qwen36-27b-teacher-对齐)
- [三、环境配置](#三环境配置)
- [四、算力分配（7×L20）](#四算力分配7l20)
- [五、参数配置](#五参数配置)
  - [5.1 SFT 超参（ms-swift）](#51-sft-超参ms-swift)
  - [5.2 FIPO 超参（verl）](#52-fipo-超参verl)
  - [5.3 9-Reward 权重](#53-9-reward-权重)
- [六、目录结构](#六目录结构)
- [七、代码文件讲解](#七代码文件讲解)
- [八、致谢与依赖](#八致谢与依赖)

---

## 一、分支与版本

| 引用 | Commit | 内容 |
|---|---|---|
| `tag v1.0-sft-baseline` | `63848ee` | v1 稳定 5-stage SFT 流水线，**S_Fin = 0.9034**（val 200，仅 git 记录，ckpt 已删） |
| `branch main` | `63848ee` | 历史快照，**v1 ckpt 已删除，inference.py 不可直接跑**（需重训 SegFormer/EffNet 才能复现） |
| `branch v2-opd` ← **当前** | head | DeepSeek-V4 风格 specialist-verified RLVR + GKD + Veritas + FIPO 三阶段 |

> v2-opd 完成验收后通过 PR 合并回 main。**v1 → v2 的迁移成本（重训 5-fold seg/cls）已被接受**，全面转 v2。

---

## 二、技术架构

### 2.1 v1 → v2 的核心改动

| 维度 | v1 baseline | **v2-opd** | 改动理由 |
|---|---|---|---|
| **输入通道** | 7ch (RGB+ELA+SRM) | **3ch RGB only** | ELA/SRM 在热敏小票/发票上结构性误报（v1 的 4 张 FP 全是热敏小票）；现代 SSL backbone (DINOv3/SigLIP-2) 自学的 dense feature 已覆盖 ELA 信号且具域感知 |
| **Loc backbone** | SegFormer-B5 5-fold | **DINOv3-ViT-L** + Mask2Former-light head | DINOv3 (arXiv 2508.10104) Gram anchoring 解决 v1 SegFormer 100 epoch 后 IoU 0.62 不动；零改 stem 保留 RGB pretrained 几何 |
| **Cls backbone** | EfficientNet-V2-L 5-fold | **SigLIP-2-So400m-NaFlex** + MLP head | NaFlex 保留 native aspect ratio（小票 aspect ≈ 0.3，方形 resize 把关键信号 crop 掉）；caption-pretrain 对语义级伪造敏感 |
| **Forensic 信号** | 拼进 stem | **MaskCLIP + Mesorch 独立 specialist** | 域感知强 + 可独立替换/淘汰，不污染 backbone（NeXT-IMDL benchmark 王者） |
| **Bbox grounding** | 无 | **SAM 3.1** | text phrase → mask 单模型，30ms / 100+ objects，验证 caption 提到的 region 是不是真在那里 |
| **Policy LM 起点** | Qwen3.5-9B + LoRA r=64 SFT | **Veritas-Cold-Start (InternVL3-8B)** + LoRA r=64 → FIPO | Veritas-Cold-Start 已在 HydraFake 上做过 SFT，作为 v2 起点强 prior |
| **后训练** | LoRA SFT only | **SFT → FIPO (9-reward RLVR)** | rule-based 占 55% 权重提供不可 hack 的 reward floor；future-KL 加权 token-level credit |

> **术语澄清**：v1 README §10.5 的"specialist OPD"严格说是 RLVR (Reinforcement Learning with Verifiable Rewards) 范式，verifier 比 policy 小是正常（DeepSeek-R1 / OpenAI o1 / AlphaGo 都这样）。

### 2.2 v2 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│  STUDENT / POLICY  (LoRA r=64, target_modules=all-linear)            │
│   InternVL3-8B (Veritas-Cold-Start) - SFT - FIPO                     │
│   ↓ rollout n=8 candidates / sample (verl + vllm 0.7.3)              │
└────────────────────────┬────────────────────────────────────────────┘
                         │ candidate (label, location, explanation)
                         │  6-tag CoT: <fast><reasoning><conclusion><answer>
                         │  inside <conclusion>: <bbox>x1,y1,x2,y2</bbox>, <region>desc</region>
                         ▼
┌──── REWARD = w·R_xxx, sum 1.0, rule-based ≥ 55% ────────────────────┐
│                                                                      │
│  Rule-based (不可 hack, 55%)                                         │
│   ├─ R_format       0.10  schema/JSON/长度/关键词/bbox 合法性         │
│   ├─ R_consistency  0.10  label-location 一致 + bbox⊆location         │
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
│   └─ R_qwen_periodic 0.05  qwen-max 每 50 step 抽 10 张校准          │
└────────────────────────┬────────────────────────────────────────────┘
                         │ 多 reward 融合 (trust-region: 单一 ≤ 25%)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FIPO UPDATE (Future-KL weighted PPO, arXiv 2603.19835)             │
│    future-KL token weighting → 关键 token (answer/bbox/region)       │
│                                  自动放大梯度                         │
│    + clip_ratio 0.2/0.28 (low/high) + grad_clip 1.0                  │
│    + KL coef 0 (use_kl_in_reward=False) — 由 future_kl 内部正则       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Specialist 选型

| 角色 | 模型 | 来源 / 选型理由 |
|---|---|---|
| **Loc backbone** | [DINOv3-ViT-L](https://arxiv.org/abs/2508.10104) (蒸馏自 7B) + Mask2Former-light | Meta 2025-08；首次单一 frozen SSL backbone 在 dense prediction 超 SegFormer 类专门方案；商业许可 |
| **Cls backbone** | [SigLIP-2-So400m-NaFlex](https://arxiv.org/abs/2502.14786) (400M) + MLP head | Google 2025-02；NaFlex 保留 aspect ratio + caption-pretrain + masked-prediction；超 EVA-CLIP / AIMv2 |
| **Forensic SOTA** | **MaskCLIP** + **Mesorch** 双路 | [NeXT-IMDL benchmark](https://arxiv.org/abs/2512.23374) 跨域 F1：MaskCLIP 0.32 vs IML-ViT 0.12 vs TruFor 0.13；ForensicHub (NeurIPS 2025) 现成代码 |
| **Bbox phrase verifier** | [SAM 3.1](https://ai.meta.com/research/sam3/) | Meta 2026-03 ICLR 2026；text phrase 直接出 mask；零样本 LVIS AP 48.8 vs SAM2 38.5 |
| **Caption rubric reward** | GRM (DeepSeek-V4 同款) + qwen-max 周期校准 | actor 自评 rubric 不用每步调 API；只在 step % 50 抽 10 样本调 qwen-max 校准漂移 |
| **Judge for prompt-only baseline** | [DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | 4 维 rubric（accuracy / evidence / consistency / faithfulness）1-10 打分 |

### 2.4 Veritas / HydraFake 整合

> 论文 [arXiv 2508.21048 ICLR 2026 Oral](https://arxiv.org/abs/2508.21048) ·
> 完整工作笔记：[`reference/Veritas_method.md`](reference/Veritas_method.md)

- **5 句话核心**：
  1. Veritas = InternVL3-8B + 6 标签 reasoning template + SFT→MiPO→P-GRPO 三阶段；HydraFake 4 级 OOD 平均 **92.1%** 准确率。
  2. 其训练数据 jsons 的 **schema 直接借用**，但 face-only 图像**选择性混入**（仅 EFG 整图 AIGC 部分加入 TFI 训练，face swap/reenactment 全弃）。
  3. **Veritas-Cold-Start (15G InternVL3-8B)** 作 v2 SFT 起点（论文作者明确推荐自定义训练用此）。
  4. 跳过 MiPO 阶段（v0 翻转策略偏脆弱）；P-GRPO 替换为 **FIPO**（详见 §2.5）。
  5. **HydraFake 没有 grounding 标签**，TFI v2 的 Loc 监督**完全靠自有数据**：v1 遗留 mask + augmented_data/synth + SAM 3.1 zero-shot 标注。

- **HydraFake 训练集（48,320 张）三档**：
  - `sft_36k.json` (36,750) — Cold-Start SFT，real 18,712 / fake 18,038（≈ 1:1）
  - `mipo_3k.json` (3,480) — MiPO 偏好对齐
  - `pgrpo_8k.json` (8,033) — P-GRPO 在线 RL（assistant 留空）
- **Reasoning template 6 标签**（命中率）：`<fast>`/`<reasoning>`/`<conclusion>`/`<answer>` 各 100%；`<planning>` 24.6%、`<reflection>` 20.4%（仅困难样本）。回复中位长度 1,269 字符（~320 token）。
- **Test 4 级 OOD**：id 13,819 / cm 11,249 / cf 12,736 / cd 15,468，严格 50/50 平衡。
- **三类 fake 子集**：face swapping (FS, 7,402) / face reenactment (FR, 5,063) / **entire face generation (EFG, 5,573)** — TFI v2 仅取 EFG 子集 + real 池。

数据合并策略详 [`data/analysis/distribution_report.md`](data/analysis/distribution_report.md)。

### 2.5 RL 算法选型：FIPO

> FIPO = Future-KL Inference Preference Optimization ([arXiv 2603.19835](https://arxiv.org/abs/2603.19835))。
> **训练栈：verl 0.7.x + future_kl_loss patch（forward-port 自 verl 0.5.x）**

#### 三算法对比

| 算法 | credit assignment | TFI 适配度 | 训练栈 |
|---|---|---|---|
| MiPO | 序列级 DPO（loss 平均到所有 token） | **中**：v0 "翻转标签 + 删 bbox" 构造的 rejected 太机械，无法教模型分辨"分类对但定位差"等中间错误 | ms-swift |
| P-GRPO | uniform per-token | **中**：TFI 多 reward 叠加 → 信号被稀释，关键 token credit 不足 | ms-swift |
| **FIPO** | **future-KL 加权 token loss** | **高**：TFI 关键决策集中在少数 token（`<answer>` + `<bbox>` 4 个数字 + `<region>` phrase），future-KL 权重恰好放大这些位置 | verl + 1 行 patch |

#### 为什么不要 MiPO

1. TFI 的 rejected 没法机械造（v0 翻转 `<answer>` 但 explanation 还在描述伪造痕迹，模型学不到细粒度偏好）。
2. 真正想区分的偏好对 — "分类对但 bbox 漂移 30px" vs "分类对且 bbox 准确" — 需要在线 rollout + reward model 打分，正好就是 FIPO 做的事。
3. 504 对 MiPO 训完只贡献 ~0.5–1.5pt（Veritas 论文 ablation），不如把卡时砸进 FIPO。

#### 为什么是 FIPO 而不是 P-GRPO

| 维度 | P-GRPO | FIPO |
|---|---|---|
| token-level 区分 | ❌ 全 token uniform | ✅ future-KL 自动加权 |
| 短 CoT 任务（TFI ~150 tok output） | 一般（reward 稀疏被均摊） | **强**（VLM-posttraining 已验证：hallucination 30.27% → 23.04%） |
| 多 reward 融合稳定性 | reward variance 高 → mode collapse 风险 | future-KL 起到隐式正则 |
| 实现复杂度 | ms-swift 自带 | verl + 50 行 patch |

### 2.7 路线 A：底座切回 Qwen3.5-9B（与 v1 / Qwen3.6-27B teacher 对齐）

> 在进入 FIPO 前对 student 底座做选型复盘，结论 **Qwen3.5-9B 综合优于 InternVL3-8B (Veritas-Cold-Start)**，2026-05-06 启用为 v2-opd 主底座。
> Veritas-Cold-Start 路线作为 ablation 备份保留（`train/sft/train_sft.sh` + `train/fipo/launch.sh` 不动）。

#### 选型对比

| 维度 | InternVL3-8B (Veritas-Cold-Start) | **Qwen3.5-9B (路线 A)** | 谁占优 |
|---|---|---|---|
| Deepfake 域 cold-start prior | HydraFake 36k 已 SFT | 通用 VLM | InternVL3 ✅ |
| v1 LoRA 续训 | 底座不同要从 0 训 | 同家族可续训 | Qwen ✅ |
| GKD 词表对齐（teacher Qwen3.6-27B） | 词表交集 ~130k，需 top-k 截断 | **同家族同 tokenizer**，全词表 KL 直接算 | Qwen ✅✅ |
| vllm rollout（FIPO 关键） | InternVL3 在 vllm 0.7.3 不稳；vllm 0.19 才支持 | dense 老架构，vllm 0.7.3 / 0.19 都即开即用 | Qwen ✅ |
| ms-swift 模板 | 必须 Veritas fork（pip 版无 internvl3） | **ms-swift 4.1.3 原生 `qwen3_5` 模板** | Qwen ✅ |
| dense vision grounding | dynamic-tiling 12 tile + thumbnail | fixed-resolution，小目标弱 | InternVL3 ✅ |
| 6 标签 CoT 数据匹配 | Veritas 配套训过 | 需要 1 epoch warmup 引导 schema | InternVL3 ✅ |

→ 5 票 vs 2 票，Qwen3.5-9B 综合更稳；vision grounding 代价由 reward 端 specialist verifier (DINOv3 / SAM 3.1) 等价补回。

#### 环境冲突与解决（关键工程坑）

`Qwen3.5-9B` 实际 `model_type=qwen3_5` (混合 linear+full attention，head_dim=256)，transformers 4.49 / 4.55 都不识别。结论：**路线 A 必须切 conda env `VLM`**。

| env | torch | transformers | vllm | swift | Qwen3.5-9B 加载 | 用途 |
|---|---|---|---|---|---|---|
| TFI | 2.5.1+cu124 | 4.49.0 | 0.7.3 | ms-swift 3.4 (Veritas fork) | ❌ | 路线 B 训练 / InternVL3 路线 |
| TFI_judge | 2.5.1+cu124 | 4.55.x | 0.7.3 | — | ❌ | judge：R1-Distill-70B 推理 |
| **VLM** | **2.10.0+cu128** | **5.5.4** | **0.19.1** | **ms-swift 4.1.3** + verl 0.8.0.dev | ✅ | **路线 A：SFT + FIPO + 推理全流程** |

VLM env 装 ms-swift 4.1.3 后会自动拉 deepspeed 0.18.9，触发 `MissingCUDAException`（机器无 nvcc），`pip uninstall -y deepspeed` 即可，swift 走原生 DDP 训 LoRA 完全不需要 deepspeed。

ms-swift 4.x 与 3.x 的 API 漂移：`--train_type lora` → `--tuner_type lora`；`warmup_ratio` / `logging_dir` / `torch_dtype` 仍可用但有 deprecation warning。

#### 路线 A 训练栈

| 阶段 | 入口 | env | 说明 |
|---|---|---|---|
| **SFT** | `train/sft/train_sft_qwen35.sh` | VLM | swift 4.1.3 / `--model_type qwen3_5 --template qwen3_5` / LoRA r=64 / freeze_vit=true / 7×L20 |
| **推理** | `eval/baseline/sft_v2_inference_qwen35.py` | VLM | `AutoModelForImageTextToText` + `AutoProcessor` (Qwen3VLProcessor)；与训练一致的中文 6 标签 CoT 提示 |
| **FIPO** | `train/fipo/launch_qwen35.sh` | VLM | verl 0.8 + future_kl_loss patch / vllm 0.19 / MAX_PROMPT_LEN=4096 / VLLM_GPU_MEM=0.55 |

> **路线 B (Veritas-Cold-Start)** 入口保持原样：`train/sft/train_sft.sh` + `train/fipo/launch.sh`，跑在 TFI env。两路并行不互相干扰，公用同一份 9-reward 奖励、同一套 build_v2_sft 数据（`<image>` 通用占位符 + bbox [0,1000]² 归一化）。

#### 路线 A SFT 落地数字（val/200, 2026-05-06）

| 模型 | S_Det | S_Loc | S_Sim | S_Fin | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| v2 InternVL3-8B (1441, bbox-norm) | 0.888 | 0.412 | 0.718 | 0.610 | 0.806 | 0.988 |
| **v2 Qwen3.5-9B (1441, bbox-norm) — 路线A** | 0.714 | 0.265 | 0.708 | 0.494 | **0.958** | 0.569 |

**镜像偏置（Mirror Bias）**：Qwen3.5-9B 与 InternVL3-Veritas 在同份数据上呈现完全相反的类别偏置 — InternVL3 高 recall 低 precision（HydraFake 36k 强 fake 先验），Qwen3.5 高 precision 低 recall（通用 web 强 real 先验）。这反而是 FIPO 9-reward 的理想起点：
1. 高 precision 起点意味着 rollout "fake" 时几乎对应真 reward，不会被 specialist 反向修正；
2. FN 越多 R_label_gt + R_iou_gt 的可学梯度越大，训练稳定且收益空间大；
3. 配 v1 同家族 Qwen3.5 + Qwen3.6-27B GKD teacher，词表对齐、GKD 全词表 KL 直接算；
→ **路线 A 既有工程一致性，又因镜像偏置形成 FIPO 友好基线**。

#### FIPO 工程栈联调（24 次 smoke 串行 / 路线 A 主算栈 + 路线 B 兜底）

> 因驱动锁死 CUDA 12.4（vllm 0.20+ 需 CUDA 13），路线 A 在 vllm 0.19.1 × Qwen3.5-9B hybrid-attn 处遇到 KV 预算 bug；
> 同时用 InternVL3-8B SFT 产物 (`sft_merged_1441_v2`) 走路线 B 兜底联调，证明 FIPO 算法栈工程闭环可行。

| 阶段 | 状态 |
|---|---|
| trl 0.29 移除 `AutoModelForCausalLMWithValueHead` | ✅ patch `verl/models/transformers/monkey_patch.py` try/except |
| verl 0.8 `POLICY_LOSS_REGISTRY` 接入 `future_kl` | ✅ |
| FIPO parquet 准备（1009 train / 53 val） | ✅ `train/fipo/prepare_fipo_data.py` |
| 9-reward + custom reward_manager | ✅ `train/fipo/verl_patches/reward_manager.py` |
| InternVL3 × transformers 5.x 适配（路线 B） | ✅ custom processor + 10 处 patch（详 journal.md 2026-05-06 晚条目）：FA2/sdpa→eager、`all_tied_weights_keys`、`text_config` alias、`_no_split_modules` 清理、`img_context_token_id`、RoPE fallback、dual prompt ids |
| FSDP wrap + vLLM HttpServer + EngineCore | ✅ 4×L20 全部 load InternVL3 7.94B 并 launch vLLM engine |
| AgentLoopWorker × 8 启动 | ✅ |
| **路线 B smoke24**：InternVL3-8B FIPO 12-step online RL | ✅ `critic/score/mean` step1 0.2627 → step6 0.3259；val reward@1 step8 0.3167 → step12 0.3252（短跑 +2.7%） |
| **Blocker 1（路线 A）**：vllm 0.19.1 × Qwen3.5-9B hybrid-attn KV 预算估错 | P1 工程项（升 vllm 0.20+ 需 CUDA 13；或显式 `--mamba-page-size-padded`；或退 hf rollout） |
| **Blocker 2（路线 B）**：checkpoint 保存时 custom `InternVLImageProcessor` 缺 `_auto_class` | 非训练阻塞；smoke 用 `trainer.save_freq=-1` 绕过，正式训练前需补 `save_pretrained` 兼容 |

→ FIPO 算法层（future-KL loss + 9-reward + reward_manager + parquet 数据）已完整组装并联调到 AgentLoop / vLLM EngineCore；路线 B 已用 InternVL3-8B 跑通 12-step FIPO smoke 并拿到 validation 指标，可作为“方案可行性”证据，但正式百分比仍需更长训练 + R1 judge 对比补齐。

### 2.6 Bbox 坐标约定（[0,1000]² 归一化）

> **train / infer 一致性的硬约束**。InternVL3 用 dynamic-tiling（12 tile + thumbnail）做视觉编码，
> 视觉 token 已经丢失"原图绝对像素"信息。bbox 必须用与 tile 无关的归一化坐标，否则模型无解。

**约定（Qwen2.5-VL / InternVL3 标准）**：

- 所有 `<bbox>x1,y1,x2,y2</bbox>` 中坐标归一化到 **[0,1000]×[0,1000]**，左上原点，整数
- `<region>` 文字描述无坐标
- prose 内 `[x,y,x,y]` 引用 bbox 的也走同一归一化
- `SYS_PROMPT_ZH` 显式声明该约定（让模型从 system 即知道空间）

**实现位置**：

| 文件 | 角色 | 关键改动 |
|---|---|---|
| [`data/build/build_v2_sft.py`](data/build/build_v2_sft.py) | 构建主 SFT JSON | `caption_to_template()` prose-sub 归一化；`build_synth()` mask-bbox 归一化；`SYS_PROMPT_ZH` 声明 |
| [`data/build/build_hydra_efg_subset.py`](data/build/build_hydra_efg_subset.py) | 构建 HF EFG 中文子集 | `SYS_PROMPT_ZH` 同步声明（schema 一致性） |
| [`eval/baseline/sft_v2_inference.py`](eval/baseline/sft_v2_inference.py) | val 推理 | `build_input_pixel_values()` 用 `transform_image(max_num=12)` + thumb；`bbox_to_rle_mask(normalized=True)` 反归一化 |

**实证收益**（同 ckpt, 同 val/200, 详见 [`journal.md`](journal.md) 2026-05-04 → 05 条目）：

| 实验变体 | S_Loc | S_Fin | Δ |
|---|---:|---:|---|
| raw-pixel bbox + 单 tile 推理 | 0.025 | 0.514 | baseline |
| raw-pixel bbox + 12-tile 推理（仅训推一致 tile） | 0.025 | 0.527 | +0.013 |
| **[0,1000] 归一化 bbox + 12-tile 推理** ✅ | **0.411** | **0.610** | **+0.096** |

---

## 三、环境配置

### 3.1 conda envs

| env | torch | vllm | transformers | swift | 用途 |
|---|---|---|---|---|---|
| `TFI` | 2.5.1+cu124 | 0.7.3 | **4.49.0** | **ms-swift 3.4 (Veritas fork, editable)** | v2 训练（SFT/FIPO）+ prompt-only + 模型加载 |
| `TFI_judge` | 2.5.1+cu124 | 0.7.3 | 5.5.4 | — | judge：R1-Distill-70B 推理（GPU 4-7 TP=4） |

**关键发现**：transformers 5.x 已删 `EvaluationStrategy`，与 swift 3.4 不兼容 → TFI env 必须停在 transformers 4.49。如要用 transformers 5.x（仅 judge），用独立 env `TFI_judge`。

### 3.2 关键依赖锁定

```
torch                2.5.1+cu124   # cuda 12.4 driver 550 兼容上限
vllm                 0.7.3         # 不要升级，新版要求 driver ≥ 560
transformers         4.49.0        # swift 3.4 依赖 EvaluationStrategy
tokenizers           0.21.4
huggingface_hub      0.36.2
peft                 0.19.1
ms-swift             3.4.0.dev0    # Veritas fork (editable, /mnt/nfs/young/TFI/code/Veritas/)
verl                 0.7.x         # /home/young/TFI/code/verl/
deepspeed / accelerate / json_repair / datasets / multiprocess / pyarrow / tensorboard / addict / dacite / jieba / rouge
```

### 3.3 NFS 资源布局（193 GB models + 277 MB data + 230 MB code）

```
/mnt/nfs/young/TFI/
├── models/   (193 GB)
│   ├── Qwen3.5-122B-A10B-AWQ/  77 GB   ← rejection-sampling 辅 teacher (MoE, AWQ INT4)
│   ├── Qwen3.6-27B/            52 GB   ← 主 teacher (GKD)
│   ├── Qwen3.5-9B/             19 GB   ← v1 base
│   ├── Veritas-Cold-Start/     15 GB   ← InternVL3-8B base, SFT 强 prior (论文作者推荐)
│   ├── Veritas/                15 GB   ← Veritas 完整模型 (cold-start 后 P-GRPO 完成)
│   ├── UnifiedReward-qwen-3b/  7.1 GB  ← P-GRPO/FIPO reward model
│   ├── SigLIP2-So400m-NaFlex/  4.3 GB  ← v2 cls backbone
│   ├── SAM-3.1/                3.3 GB  ← v2 grounding
│   └── DINOv3-ViT-L/           1.2 GB  ← v2 loc backbone
├── data/   (277 MB jsons)
│   ├── HydraFake/jsons/        119 MB  ← Veritas 全部训练 + 测试 jsons
│   ├── HydraFake/{train,val,test}/  57 GB  ← 全图（已解压）
│   └── v2/                     ← TFI v2 SFT/MiPO/P-GRPO/FIPO 输入数据
├── code/   (230 MB)
│   ├── Veritas/         54 MB ← ms-swift Veritas fork (editable)
│   ├── sam3/           131 MB ← SAM 3.1 inference 必需
│   ├── dinov3/          32 MB ← DINOv3 PyTorch Hub 加载
│   └── ForensicHub/     11 MB ← Mesorch / 其他 forensic specialist 实现
├── runs/sft/v2sft_baseline_1009/  ← v2 SFT 输出
└── judge_model/r1-distill-llama-70b/  132 GB
```

**所有训练/评测脚本默认引用 `/mnt/nfs/young/TFI/models/...`**。本地 `/home/young/TFI/` 不再保留任何模型文件或符号链接。

### 3.4 镜像策略

- HF 直连**不通**（连接超时），`hf-mirror.com` **不支持 gated**（403）
- DINOv3 / SAM 3.1 走 **modelscope 镜像**（`facebook/dinov3-vitl16-pretrain-lvd1689m` / `facebook/sam3.1` 在 ms 都有）
- GitHub 走 **kkgithub.com 镜像**（ghproxy.net / 99988866.xyz 都失效）

---

## 四、算力分配（7×L20）

> **GPU 0 历史 ECC 错误，永久禁用**。可用 7 张（GPU 1-7）= 7 × 46GB = 322GB。
> 所有训练脚本 `CUDA_VISIBLE_DEVICES` 默认 `1,2,3,4,5,6,7`。

```
═════════════════════════════════════════════════════════════════════════
M(-1) · prompt-only baseline                          ← ✅ 完成
─────────────────────────────────────────────────────────────────────────
- Qwen3.5-9B + 3 prompt 变体 (zs/fs/cot) × 200 val
- DeepSeek-R1-Distill-Llama-70B 当 judge (GPU 4-7 TP=4)
- Qwen3.6-27B 同协议 ceiling (zs/cot, GPU 1+2 device_map=auto)
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M1 · SFT (Veritas-Cold-Start LoRA, 已完成 baseline)
─────────────────────────────────────────────────────────────────────────
GPU 1-7: torchrun NPROC=7
         per_device_train_batch_size=1  grad_accum=8  → effective bsz 56
         lora_rank=64  lora_alpha=128  target_modules=all-linear
         max_length=3072  bf16  freeze_vit=false
         3 epoch / 1009 train + 53 val ≈ 40 min
═════════════════════════════════════════════════════════════════════════

═════════════════════════════════════════════════════════════════════════
M4 · FIPO 主训练 (5-7 天, 最关键)
─────────────────────────────────────────────────────────────────────────
┌── student (InternVL3-8B) actor + ref ─────────────────────┐
│  GPU 1,2: actor FSDP2 zero3 (param_offload=True)          │
│  GPU 3:    ref model frozen (param_offload=True)          │
└────────────┬──────────────────────────────────────────────┘
┌── rollout engine ─────────────────────────────────────────┐
│  GPU 4: vllm 0.7.3 InternVL3-8B                            │
│  n=8 candidates/prompt, batch=6                            │
│  gpu_memory_utilization=0.62                               │
│  max_prompt_len=8192, max_resp_len=1024                    │
└────────────┬──────────────────────────────────────────────┘
┌── reward server (常驻) ───────────────────────────────────┐
│  GPU 5: rule-based eval + DINOv3 + SigLIP-2 + MaskCLIP    │
│  GPU 6: SAM 3.1 (bbox verify)                              │
│  GPU 7: GRM + 异步 qwen-max 校准                            │
└────────────┬──────────────────────────────────────────────┘
             │ R = Σ w_i·R_i (rule 55% + specialist 30% + caption 15%)
             ▼
   FIPO update (future-KL weighted PPO)
预算: 80 step / day × 7 天 ≈ 560 step ≈ 2 epoch
验收: val S_Fin ≥ 0.94, S_Auto ≥ 0.92, S_Loc ≥ 0.92
═════════════════════════════════════════════════════════════════════════
```

---

## 五、参数配置

### 5.1 SFT 超参（ms-swift）

`train/sft/train_sft.sh`：

```bash
swift sft \
    --model        /mnt/nfs/young/TFI/models/Veritas-Cold-Start \
    --model_type   internvl3 \
    --template     internvl2_5 \
    --dataset      /mnt/nfs/young/TFI/data/v2/sft.json \
    --val_dataset  /mnt/nfs/young/TFI/data/v2/sft_val.json \
    --num_train_epochs                 3 \
    --per_device_train_batch_size      1 \
    --gradient_accumulation_steps      8        # → effective bsz = 7 GPU × 8 = 56
    --train_type                       lora \
    --freeze_vit                       false \
    --lora_rank                        64 \
    --lora_alpha                       128 \
    --target_modules                   all-linear \
    --torch_dtype                      bfloat16 \
    --learning_rate                    5e-5 \
    --weight_decay                     0.01 \
    --warmup_ratio                     0.05 \
    --lr_scheduler_type                cosine \
    --max_length                       3072 \
    --gradient_checkpointing           true
```

### 5.2 FIPO 超参（verl）

`train/fipo/launch.sh`：

```bash
N_GPUS=7
BATCH_SIZE=6                  # train_prompt_bsz; (BATCH_SIZE * N_RESP) % (N_GPUS * MICRO_BSZ_PER_GPU) == 0
N_RESP=8                      # rollouts per prompt
MINI_BSZ=3                    # PPO mini-batch (prompts)
MICRO_BSZ_PER_GPU=1           # PPO micro-batch (per-GPU)
MAX_PROMPT_LEN=8192           # InternVL3 image tokens 不可截断
MAX_RESP_LEN=1024
GEN_TP=1

ACTOR_STRATEGY=fsdp2
REF_STRATEGY=fsdp2
LOSS_MODE=future_kl           # set "vanilla" 退化为 GRPO

# FIPO 超参（POLICY_LOSS dataclass 严格，通过 env 注入）
FIPO_DECAY_RATE=12.0
FIPO_CHUNK_SIZE=128
FIPO_FKL_CLIP_RATIO=0.2
FIPO_FKL_CLIP_HIGH_ONLY=false
FIPO_SAFETY_THRESH=4.0

# PPO clip & KL
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
use_kl_loss=False
use_kl_in_reward=False
kl_ctrl.kl_coef=0.0           # FIPO 内部 future_kl 已正则

# Optimizer
lr=1e-6
lr_warmup_steps=10
grad_clip=1.0
entropy_coeff=0
total_epochs=2

# vLLM rollout
gpu_memory_utilization=0.62
temperature=1.0
top_p=0.95
enable_chunked_prefill=True

# Training book-keeping
val_before_train=True
test_freq=10
save_freq=40
max_actor_ckpt_to_keep=2
RAY_memory_usage_threshold=0.97
```

### 5.3 9-Reward 权重

`train/fipo/reward_fn.py:DEFAULT_WEIGHTS`：

| Reward | 权重 | 类型 | 说明 |
|---|---:|---|---|
| `R_format` | 0.10 | rule | 4 个必需 tag (`<fast>`/`<reasoning>`/`<conclusion>`/`<answer>`) 各出现 1 次 + answer 是 real/fake |
| `R_consistency` | 0.10 | rule | fake⇒有 bbox/region；real⇒无 |
| `R_label_gt` | 0.15 | rule | `<answer>` == GT label |
| `R_iou_gt` | 0.15 | rule | bbox max IoU vs GT bbox（real⇒空匹配=1） |
| `R_phrase_check` | 0.05 | rule | conclusion 中的数字/region 出现在 GT phrase 池 |
| `R_loc` | 0.10 | hook | DINOv3 specialist 打分（待部署） |
| `R_cls` | 0.10 | hook | SigLIP-2 specialist 打分（待部署） |
| `R_forensic` | 0.10 | hook | MaskCLIP+Mesorch specialist 打分（待部署） |
| `R_grm` | 0.10 | hook | GRM caption rubric 自评（待部署） |
| `R_qwen_periodic` | 0.05 | hook | qwen-max 每 50 step 抽 10 张校准（待部署） |

**FIPO v1 起步**：仅 5 项 rule-based 跑（合计 0.55 权重，max 总分 0.55）。
specialist + GRM 部署后通过子类化 `TFIAuditRewardManager._external_scores_for(item)` 接入，**无需改 reward_fn 主路**。

GT blob schema（`reward_model.ground_truth` per sample）：
```json
{"label": 0 | 1, "bboxes": [[x1, y1, x2, y2], ...], "phrases": ["<region>...</region>", "“引号文字”", ...]}
```
bbox 来源优先级：`mask_path → mask_to_bbox` > 文本 `<bbox>` 解析 > 空列表。

---

## 六、目录结构

```
TFI/  (v2-opd branch)
├── README.md                  本文（架构 / 环境 / 参数 / 代码讲解）
├── journal.md                 实验日志、状态更新、踩坑记录
├── HANDOVER.md                Agent 接手须知（含一键启动）
├── sitecustomize.py           Ray worker 自动 import FIPO patch + GPU remap
├── utils.py                   ELA / SRM / RLE / IoU / Dice 等工具
├── requirements.txt
│
├── archive/v1/                v1 baseline 历史代码（只读归档）
│   ├── train_seg_ensemble.py / train_classifier.py / train_calibrator.py
│   ├── train_qwen35_9b.py / inference.py / evaluate.py
│   ├── config.yaml            ← v1 stale 配置（SegFormer/MaxVit/calibrator + gpu:0）
│   └── DATA_AUGMENTATION.md
│
├── data/
│   ├── build/                 数据构建脚本
│   │   ├── build_v2_sft.py             SFT 主集（1009 条）
│   │   ├── build_v2_mipo.py            MiPO 偏好对（兜底）
│   │   ├── build_v2_pgrpo.py           P-GRPO prompts（兜底）
│   │   ├── build_hydra_efg_subset.py   ★ HF-EFG-CN 子集（4k EFG + 4k real，中文 6-tag CoT）
│   │   ├── merge_official_hydra.py     ★ TFI ⊕ HF 30/70 stratified merge
│   │   └── data_guard.py               数据契约体检（写 data/meta/data_health.md）
│   ├── analysis/
│   │   └── distribution_report.md      ★ TFI / HydraFake 分布与合并策略
│   ├── meta/                  data_health.md 等数据元信息
│   ├── processed/             v1 留下的 synth + caption_local_v2
│   └── raw/                   train_resume / val / test（软链 NFS）
│
├── train/
│   ├── sft/train_sft.sh                ms-swift SFT (GPU 1-7, NPROC=7)
│   ├── mipo/train_mipo.sh              兜底
│   ├── pgrpo/train_pgrpo.sh            兜底
│   └── fipo/                 ★ 主路线（移植自 VLM-posttraining + TFI 改写）
│       ├── main_fipo.py                verl entry，注册 future_kl + reward_manager
│       ├── launch.sh                   启动脚本（GPU 1-7 默认）
│       ├── schema.py                   TFI 6-tag CoT 解析器 + system prompt
│       ├── reward_fn.py                9-reward 计算（5 rule + 4 hook）
│       ├── prepare_fipo_data.py        SFT JSON → verl parquet（含 GT bboxes/phrases）
│       ├── config/train.yaml           FIPO 超参
│       └── verl_patches/
│           ├── future_kl_loss.py       注册 POLICY_LOSS_REGISTRY["future_kl"]
│           └── reward_manager.py       TFIAuditRewardManager
│
├── eval/
│   ├── score_official.py               官方公式复现 (S_Det/S_Loc/S_Sim/S_Auto/S_Exp/S_Fin)
│   ├── baseline/                       M(-1) prompt-only baseline
│   │   ├── prompt_only_baseline.py     Qwen3.5-9B 3 变体 (zs/fs/cot)
│   │   ├── prompts.py                  prompt 模板 + few-shot 例子
│   │   ├── judge_absolute_scoring.py   R1-Distill-70B 4 维 1-10 打分
│   │   ├── veritas_zero_shot.py        Veritas-Cold-Start zero-shot
│   │   ├── run_ceiling.sh              Qwen3.6-27B ceiling
│   │   ├── run_all.sh                  一键跑全 baseline
│   │   ├── results/{zs,cot,fs,judge}/  4 组 baseline 结果
│   │   └── results_qwen36/{zs,cot}/    Qwen3.6-27B ceiling 结果
│   └── grounding/                      ⏳ 待写：grounding 指标（IoU, Dice）
│
├── code/verl/                 verl-latest（从 VLM-posttraining 复制 18MB）
├── reference/                 论文 + Veritas_method.md + FIPO.pdf + README.v1.md
└── logs/                      运行日志
```

NFS 不变：见 [§3.3](#33-nfs-资源布局193-gb-models--277-mb-data--230-mb-code)。

---

## 七、代码文件讲解

### 7.1 数据构建（`data/build/`）

| 文件 | 输入 | 输出 | 关键逻辑 |
|---|---|---|---|
| **`build_v2_sft.py`** | v1 train/Black（800 fake）+ White（200 real）+ processed/synth + 可选 HF EFG | `/mnt/nfs/young/TFI/data/v2/sft.json` (1009 条) + `sft_val.json` (53 条) | 4 数据源合并；中文 6-tag CoT template；mask → bbox + region phrase 注入 `<conclusion>`；ms-swift VLM SFT 标准格式 |
| **`build_v2_mipo.py`** | v2 SFT json | `mipo.json` (504 偏好对) | v0 简单翻转策略：chosen=原 SFT；rejected=`<answer>` 翻转 + bbox 删除（**已知偏脆弱，等 hard negative 升级**） |
| **`build_v2_pgrpo.py`** | v2 SFT json | `pgrpo.json` (1009 prompts) | assistant 留空，仅保留 `{prompt, label, type, mask_path}`；rollout 阶段由 actor 生成 candidates |
| **`build_hydra_efg_subset.py`** ★ | `/mnt/nfs/young/TFI/data/HydraFake/jsons/train/sft_36k.json` (HF 全集) | `hydra_efg_cn.json` (4k EFG + 4k real, ~8k 条) | 仅取 EFG fake + real 池；丢弃 HF 自带英文 assistant；用 3+3 中文 6-tag CoT template 重写 reasoning（fake/real 各 3 模板）；sub-generator 名（Dall-E1 / StyleGAN3 / ...）从 path 解析后注入模板 |
| **`merge_official_hydra.py`** ★ | TFI sft.json + hydra_efg_cn.json | `sft_merged.json` + `sft_merged_meta.json` | 按 image hash 去重；分层抽样保 label 0/1 平衡；公式 `H = r·T/(1-r)` 算 HF 抽取量；默认 `hydra_ratio=0.30` (HF 30% / TFI 70%) |
| **`data_guard.py`** | `data/` 各层 | `data/meta/data_health.md` | 数据契约体检：检测层级 / mask 完整性 / label 平衡 / 域分布；`--strict` 失败 exit 1（CI 用） |

### 7.2 训练入口（`train/`）

#### SFT（ms-swift）

| 文件 | 说明 |
|---|---|
| **`train/sft/train_sft.sh`** | 主 SFT 入口；环境变量 `RUN_NAME` / `DATASET` / `MODEL` 可覆盖；默认 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`、`NPROC=7`；产物落 `/mnt/nfs/young/TFI/runs/sft/$RUN_NAME/v0-*/checkpoint-N` |
| **`train/mipo/train_mipo.sh`** | 兜底：MiPO 偏好对齐（主路线已跳过） |
| **`train/pgrpo/train_pgrpo.sh`** | 兜底：P-GRPO 在线 RL（主路线已换 FIPO） |

#### FIPO（verl，主路线）

| 文件 | 说明 |
|---|---|
| **`train/fipo/main_fipo.py`** | verl entry：先注册 `future_kl_loss` 到 `POLICY_LOSS_REGISTRY` 与 `TFIAuditRewardManager` 到 `REWARD_MANAGER_REGISTRY`，然后 wrap `verl.trainer.main_ppo`；不要直接 invoke，必须通过 `launch.sh`（Hydra config path resolution 依赖 cwd） |
| **`train/fipo/launch.sh`** | 主启动脚本：conda activate `TFI` env / `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` / `FIPO_PATCH_VERL=1` / `RAY_memory_usage_threshold=0.97` / 默认 `CUDA_VISIBLE_DEVICES=1-7`；trap cleanup 杀残留 `VLLM` / `ray::` 进程；MODEL_PATH 默认 `/mnt/nfs/young/TFI/models/sft_merged` |
| **`train/fipo/schema.py`** | TFI 6-tag CoT 解析器 + `SYSTEM_PROMPT`（与 `data/build/build_v2_sft.py` 一致）；`parse_response(text) -> TFIResponse` dataclass；提取 `<fast>` / `<reasoning>` / `<conclusion>` / `<answer>` 与嵌入的 `<bbox>` / `<region>` |
| **`train/fipo/reward_fn.py`** | 9-reward 主计算；`compute_reward(response_text, GroundTruth) -> float \| (float, breakdown)`；smoke test：`python -m train.fipo.reward_fn` 输出 `score=0.4673`（well-formed fake on fake-with-bbox GT） |
| **`train/fipo/prepare_fipo_data.py`** | SFT JSON → verl multimodal parquet（schema 同 `examples/data_preprocess/geo3k.py`）；写入 `reward_model.ground_truth` 包含 `{label, bboxes, phrases}`；bbox 来源优先级 mask_path → 文本 `<bbox>` 解析 → 空列表 |
| **`train/fipo/config/train.yaml`** | TFI FIPO Hydra 默认值（顶层超参也可在 launch.sh 用 dot-notation 覆盖） |
| **`train/fipo/verl_patches/future_kl_loss.py`** | FIPO future-KL policy loss，forward-port 自 verl 0.5.x 到 verl-latest (>=0.7.x) `(pg_loss, metrics_dict)` 接口；通过 env vars (`FIPO_DECAY_RATE` 等) 读 dataclass 不接受的字段 |
| **`train/fipo/verl_patches/reward_manager.py`** | `TFIAuditRewardManager`（`importlib` 加载，无须 register）；调用 `train.fipo.reward_fn.compute_reward`；预留 `_external_scores_for(item)` hook 给 specialist + GRM server 接入 |

### 7.3 评测（`eval/`）

| 文件 | 说明 |
|---|---|
| **`eval/score_official.py`** | 官方 S_Fin 公式复现：`S_Det = image-F1 / S_Loc = pixel-F1 (Dice) / S_Sim = BERTScore-zh / S_Auto = Qwen3-MAX rubrics(/100) / S_Exp = 0.5·Sim+0.5·Auto / S_Fin = 0.45·Det+0.25·Loc+0.30·Exp` |
| **`eval/baseline/prompt_only_baseline.py`** | M(-1) prompt-only baseline：Qwen3.5-9B + 3 变体 (zs / 8-shot fs / cot) on val/200；输出与 v1 `submit_val.csv` 同 schema |
| **`eval/baseline/prompts.py`** | 3 变体 prompt 模板 + few-shot 例子 |
| **`eval/baseline/judge_absolute_scoring.py`** | R1-Distill-70B 当 judge：4 维 1-10 (`accuracy` / `evidence` / `consistency` / `faithfulness`) → overall 平均；text-only 不输入图 |
| **`eval/baseline/veritas_zero_shot.py`** | M(-1)++ Veritas-Cold-Start zero-shot on val/200；transformers 直接 load（vllm 0.7.3 不一定支持 InternVL3）；单卡 BF16 |
| **`eval/baseline/run_ceiling.sh`** | Qwen3.6-27B prompt-only ceiling (zs+cot)；GPU 1+2 device_map=auto；输出 `eval/baseline/results_qwen36/{zs,cot}/predictions.csv` |
| **`eval/baseline/run_all.sh`** | 一键跑全 baseline |

### 7.4 顶层工具

| 文件 | 说明 |
|---|---|
| **`utils.py`** | ELA / SRM 取证特征提取 / COCO RLE 编解码 / IoU·Dice·F1·PixelAcc 评估 / mask 形态学后处理（v1 遗留，v2 部分仍在用） |
| **`sitecustomize.py`** | 自动 import — 通过 `site.py` 在 Ray worker 子进程启动时把 `train.fipo.verl_patches.future_kl_loss` 注入 verl 的 `POLICY_LOSS_REGISTRY`，并按 `FIPO_PATCH_VERL=1` 启用 `torch.cuda.set_device` remap（让非连续 `CUDA_VISIBLE_DEVICES` 与 verl `worker.set_device(physical_id)` 兼容） |

### 7.5 数据流总览

```
v1 raw + processed/synth                ┐
HydraFake jsons (NFS, 4 级)             ├──► build/build_v2_sft.py        ──► sft.json (1009)
                                        │                                        ↓
                                        ├──► build/build_hydra_efg_subset.py ──► hydra_efg_cn.json (~8k)
                                        │                                        ↓
                                        └──► build/merge_official_hydra.py    ──► sft_merged.json (~1441, 30/70)
                                                                                  │
sft_merged.json ──► train/sft/train_sft.sh ──► /mnt/nfs/young/TFI/runs/sft/.../checkpoint-N (LoRA)
                                                                                  │
                                                                        merge_lora → HF dump
                                                                                  │
sft.json + sft_val.json ──► train/fipo/prepare_fipo_data.py ──► fipo/{train,val}.parquet
                                                                                  │
                                                                                  ▼
                                                              train/fipo/launch.sh
                                                              (verl + future_kl + 9-reward)
                                                                                  │
                                                                                  ▼
                                                              /mnt/nfs/young/TFI/runs/fipo/.../checkpoint-N
                                                                                  │
val/200 + test/500 ──► eval/baseline/*.py ──► judge_absolute_scoring.py ──► 综合对比表
                                                                                  │
                                       ──► eval/score_official.py ──► S_Det/S_Loc/S_Exp/S_Fin
```

---

## 八、致谢与依赖

### 论文 / 模型
- **Veritas / HydraFake** (ICLR 2026 Oral)：[arXiv 2508.21048](https://arxiv.org/abs/2508.21048) · [HydraFake on modelscope](https://www.modelscope.cn/datasets/EricTanh/HydraFake) · [Veritas-Cold-Start](https://www.modelscope.cn/models/EricTanh/Veritas-Cold-Start)
- **FIPO**：[arXiv 2603.19835](https://arxiv.org/abs/2603.19835) (future-KL inference preference optimization)
- **DINOv3**：[arXiv 2508.10104](https://arxiv.org/abs/2508.10104) (Meta 2025-08, frozen SSL backbone)
- **SigLIP-2-NaFlex**：[arXiv 2502.14786](https://arxiv.org/abs/2502.14786) (Google 2025-02, native aspect ratio)
- **SAM 3.1**：[Meta ICLR 2026](https://ai.meta.com/research/sam3/) (text phrase → mask)
- **MaskCLIP / Mesorch**：[NeXT-IMDL benchmark, arXiv 2512.23374](https://arxiv.org/abs/2512.23374)
- **EOPD**：[arXiv 2603.07079](https://arxiv.org/abs/2603.07079) (Entropy-Aware OPD + sentence-level IS clip)
- **Reward Model**：[CodeGoat24/UnifiedReward-qwen-3b](https://huggingface.co/CodeGoat24/UnifiedReward-qwen-3b)
- **Judge**：[DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- **Policy / Teacher 候选**：[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) / [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)（实际 model_type=`qwen3_5`，含 vision_config 是 MLLM）/ [Qwen3.5-122B-A10B-AWQ](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)（rejection-sampling 辅 teacher）

### 训练框架
- **ms-swift** (Veritas fork, editable)：[modelscope/ms-swift](https://github.com/modelscope/ms-swift) — SFT / MiPO / P-GRPO
- **verl** (≥0.7.x)：[volcengine/verl](https://github.com/volcengine/verl) — FIPO 训练栈
- **DeepSpeed / accelerate / Ray**

### 推理引擎
- **student rollout**：vllm 0.7.3 (cu124 兼容, dense 老架构早支持)
- **teacher (Qwen3.6-27B)**：transformers 5.5.4 + FSDP zero3（**不走 vllm**，driver 12.4 装不了新版）
- **122B-A10B rejection sampling**：[LMDeploy v0.12.2](https://github.com/InternLM/lmdeploy/releases/tag/v0.12.2) turbomind + AWQ ([PR #4389](https://github.com/InternLM/lmdeploy/pull/4389))
- **judge (R1-Distill-70B)**：vllm 0.7.3 TP=4 (TFI_judge env)

### v1 历史依赖
SegFormer-B5 / EfficientNet-V2 / Qwen3.5-9B / qwen-vl-max / XGBoost / TabPFN-2.5 / bert-score / dashscope（详见 `reference/README.v1.md`）。
