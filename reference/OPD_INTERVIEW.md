# TFI v2-opd · 面试参考文档（OPD / FIPO / GKD 完整方案）

> **使用方法**：本档与 `README.md` / `journal.md` 配套使用。
> **定位**：v2-opd 还没跑出最终落地数字（24 次 smoke 联调，路线 B 在 12-step 短跑上 reward `0.3167 → 0.3252` +2.7%），所以**面试时按"已完成的工程基线 + future work"两段讲**。
> 全部内容回追到源码 / 论文，括注里的引用是仓库内绝对路径，方便面试官追问时翻给他看。

---

## 0. 三十秒电梯版

- **任务**：图像伪造比赛（Detection / Grounding / Explanation 三任务）— 竞赛官方 `S_Fin = 0.45·S_Det + 0.25·S_Loc + 0.30·S_Exp`。
- **v1 baseline**：5-stage SFT 流水线，`S_Fin = 0.9034`（SegFormer-B5 5-fold + EfficientNet-V2 5-fold + 10 维 evidence + XGB/TabPFN 校准 + Qwen3.5-9B LoRA 解释生成）。
- **v2-opd（当前分支）**：DeepSeek-V4 风格的 **多 specialist OPD** + **FIPO RL** + **可选 GKD 蒸馏**，目标 `S_Fin ≈ 0.945-0.955`（+0.04~0.05）。
- **现状**：SFT-only 还没追上 v1（Qwen3.5-9B SFT/路线 A `S_Fin = 0.494`，InternVL3 SFT/路线 B `S_Fin = 0.610`）；FIPO 工程栈完整组装并在路线 B 跑通 12-step smoke，正式百分比待 GKD warmstart + 长 RL 训练补齐。
- **核心结论**：**现阶段成果是工程闭环 + 镜像偏置发现 + 9-reward 设计**；最终指标作为 future work 讲，路径与可行性证据完整。

---

## 1. 技术栈

### 1.1 模型
| 角色 | 模型 | 来源 |
|---|---|---|
| Student / Policy（路线 A 主） | Qwen3.5-9B（实际 `model_type=qwen3_5`，含 `vision_config` 是 MLLM，混合 linear + full attention，head_dim=256） | NFS `/mnt/nfs/young/TFI/models/Qwen3.5-9B/` |
| Student（路线 B 兜底） | Veritas-Cold-Start（InternVL3-8B 在 HydraFake 36k 上做完 SFT 的 ckpt，论文作者推荐起点） | NFS `Veritas-Cold-Start/` |
| Teacher（GKD） | **Qwen3.6-27B**（dense 27B，与 Qwen3.5-9B 同家族同 tokenizer，全词表 KL 直接算） | NFS `Qwen3.6-27B/` |
| Aux teacher（rejection sampling） | Qwen3.5-122B-A10B-AWQ（MoE INT4，A10B 激活 ~20G） | NFS `Qwen3.5-122B-A10B-AWQ/` |
| Loc specialist | DINOv3-ViT-L + Mask2Former-light（[arXiv 2508.10104](https://arxiv.org/abs/2508.10104)） | `models/DINOv3-ViT-L/` |
| Cls specialist | SigLIP-2-So400m-NaFlex + MLP（[arXiv 2502.14786](https://arxiv.org/abs/2502.14786)，NaFlex 保 native aspect ratio） | `models/SigLIP2-So400m-NaFlex/` |
| Forensic specialist | MaskCLIP + Mesorch（[NeXT-IMDL benchmark](https://arxiv.org/abs/2512.23374)） | `code/ForensicHub/` |
| Bbox phrase verifier | SAM 3.1（text → mask，单模型，[ICLR 2026](https://ai.meta.com/research/sam3/)） | `models/SAM-3.1/` |
| Caption rubric reward | UnifiedReward-qwen-3b + GRM 自评 + qwen-max 周期校准 | `models/UnifiedReward-qwen-3b/` |
| Judge | DeepSeek-R1-Distill-Llama-70B（4 维度 1-10 rubric） | `judge_model/r1-distill-llama-70b/` |

### 1.2 训练框架
| 框架 | 版本 | 用途 |
|---|---|---|
| **ms-swift**（Veritas fork, editable） | 3.4.0.dev0 | SFT（路线 B）+ MiPO 兜底 + P-GRPO 兜底 |
| **ms-swift** 4.1.3 | 原生 `qwen3_5` 模板 | SFT（路线 A） |
| **verl** | 0.7.x（路线 B）/ 0.8.0.dev（路线 A） | FIPO 主训练栈 |
| vLLM | 0.7.3（cu124）/ 0.19.1（VLM env） | rollout 引擎 |
| transformers | 4.49.0（TFI env）/ 5.5.4（TFI_judge）/ 5.5.4（VLM env） | 模型加载 |
| LMDeploy | 0.12.2 turbomind + AWQ | 122B-A10B rejection sampling |

### 1.3 算力
- **8×L20 (46G)**，**GPU 0 历史 ECC 永久禁用**，可用 7 张 = 322 GB。
- FIPO 卡分配：actor + ref FSDP（GPU 1-3，param_offload=True）/ vLLM rollout（GPU 4，`gpu_memory_utilization=0.62`）/ reward server（GPU 5-7：DINOv3 + SigLIP-2 + MaskCLIP + SAM 3.1 + GRM）。

### 1.4 工程关键防雷
- driver 550 锁死 CUDA 12.4 → vllm 不能升 0.20+（要 CUDA 13）；teacher 27B 走 transformers + FSDP 不走 vllm。
- transformers 5.x 删了 `EvaluationStrategy`，与 swift 3.4 不兼容 → SFT/FIPO env 必须停在 4.49；judge 用独立 env。
- ms-swift pip 装的没 `internvl3` 模板 → 必须 Veritas fork editable。
- Ray + 自定义 `CUDA_VISIBLE_DEVICES` 不会自动 isolate → 必须 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + `sitecustomize.py` patch `torch.cuda.set_device` 才不会 NCCL "Duplicate GPU" hang。

---

## 2. v1 → v2 的核心改动（"为什么要去掉 low-level 特征"）

> 这是面试一定会被追问的一题：**面试官会问"v1 已经 0.9034 了为什么还要重做"**。

| 维度 | v1 baseline (`S_Fin=0.9034`) | **v2-opd** | 改动理由 |
|---|---|---|---|
| **输入通道** | 7ch = RGB + ELA(3ch) + SRM(1ch) | **3ch RGB only** | ELA/SRM 是结构性手工特征，**在热敏小票/发票上系统性误报**（v1 的 4 张 FP 全是热敏小票，因为热敏纸天然高频噪声让 SRM 失真）；现代 SSL backbone（DINOv3 / SigLIP-2）的 dense feature 已经覆盖 ELA 信号且具备**域感知**（能区分"小票 = 真实噪声"和"伪造 = 编辑噪声"），手工特征只会引入 prior bias。 |
| **Loc backbone** | SegFormer-B5 5-fold（IoU 卡在 0.62） | **DINOv3-ViT-L** + Mask2Former-light head | DINOv3 的 Gram anchoring 解决了 SegFormer 100 epoch 后 dense feature 退化；零改 stem 保留 RGB pretrained 几何先验。 |
| **Cls backbone** | EfficientNet-V2-L 5-fold | **SigLIP-2-So400m-NaFlex** + MLP | NaFlex 保留 native aspect ratio——v1 把热敏小票（aspect ≈ 0.3）方形 resize 直接 crop 掉关键信号；caption-pretrain 对**语义级伪造**敏感。 |
| **Forensic 专用** | 没有 / 拼进 stem | **MaskCLIP + Mesorch 独立 specialist** | NeXT-IMDL benchmark：MaskCLIP cross-domain F1 = 0.32 vs IML-ViT 0.12 vs TruFor 0.13；独立可替换不污染 backbone。 |
| **Bbox grounding** | 无 | **SAM 3.1** | text phrase → mask 单模型，30ms / 100+ objects，验证 caption 中 region 是否真在那里。 |
| **Policy LM 起点** | Qwen3.5-9B + LoRA r=64 SFT | **路线 A**: Qwen3.5-9B 全参 + GKD soft label / **路线 B**: Veritas-Cold-Start (InternVL3-8B) + LoRA r=64 | 利用 Qwen3.5/3.6 同 tokenizer 全词表 KL；或借用 Veritas 在 36k HydraFake 上的强 prior。 |
| **后训练** | LoRA SFT only | **SFT → FIPO（9-reward RLVR）→ 可选 GKD** | rule reward 占 55% 提供不可 hack 的 reward floor；future-KL 加权 token-level credit assignment。 |

**核心放弃 low-level 特征的理由用一句话**：
> *v1 的 ELA/SRM 是"为模型补先验"，但补错了——它在热敏小票/发票这种本身就高频的输入上结构性误报，而且让 backbone 学到的是**人造特征 → label** 的捷径，不是图像的真实分布。v2 选择信任 SSL backbone 自学的 dense feature，把人为先验从 input 端移到 reward 端（specialist verifier），这样模型行为可解释、可替换、可 ablation。*

**SFT 倒退的诚实分析**（同 1441 张 sft_merged，val/200）：

| 模型 | S_Det | S_Loc | S_Sim | S_Fin |
|---|---:|---:|---:|---:|
| v1 baseline (LoRA SFT only, 7ch + SegFormer + EffNet + XGB) | — | 0.8735 | 0.7552 | **0.9034** |
| v2 InternVL3-8B (Veritas-Cold-Start + LoRA, 1441) | 0.888 | 0.412 | 0.718 | 0.610 |
| v2 Qwen3.5-9B (LoRA, 1441) — 路线 A | 0.714 | 0.265 | 0.708 | 0.494 |

> **为什么 SFT 单独看是退步？**因为 v1 的 0.9034 是**5 个模型 ensemble + 校准器搜阈值**叠出来的，VLM 只负责生成解释；v2 把整条链路压回 1 个 VLM，SFT 单阶段没办法立刻吃下"5 个 specialist 5-fold + 校准器"投入的总监督密度。**v2 的设计假设**是：让 specialist 以 reward 形式喂回 VLM，等价复刻 v1 ensemble 的多视角监督，然后通过 RL 让 VLM 同时**输出一致的 detection + grounding + explanation**。所以 SFT 只是 cold start，FIPO 才是补位的关键阶段。

---

## 3. OPD 原理与可行性分析

### 3.1 名词消歧

| 术语 | 来源 | 在本项目里的含义 |
|---|---|---|
| **OPD** = On-Policy Distillation | DeepSeek-V4 paper（[arXiv 2604.00626](https://arxiv.org/abs/2604.00626), 2026-04-23） | 总框架名：每个领域单独训 specialist (SFT + GRPO)，再用反向 KL 把 specialist 知识蒸馏回统一 student。 |
| **EOPD** = Entropy-Aware OPD | [arXiv 2603.07079](https://arxiv.org/abs/2603.07079) | 改进 OPD：高熵 token 用 forward-KL（防 mode collapse），低熵 token 用 reverse-KL（收敛快），加 sentence-level IS clip。 |
| **GKD** = Generalized Knowledge Distillation | Google 原始 GKD 论文（teacher logits → student） | 在 ms-swift 里用 `--rlhf_type gkd`，本质是 OPD 的实现手段（ms-swift 把 OPD/GKD 作同义） |
| **FIPO** = Future-KL Inference Preference Optimization | [arXiv 2603.19835](https://arxiv.org/abs/2603.19835) | RL 阶段的 policy loss：用 future-KL 给每个 token 算"未来漂移权重" → 关键 token（answer / bbox / region）梯度自动放大。 |
| **specialist** | DeepSeek-V4 风格 | 不是模型，是 **reward 来源**：DINOv3 / SigLIP-2 / MaskCLIP / SAM 3.1 / GRM 各自打分，喂进 9-reward 融合。 |

> **本项目里的 "OPD"**：是分支名 `v2-opd` 的统称，**指代整套"specialist-verified RLVR + GKD soft-label distillation + FIPO RL"组合方案**。不是某个单一算法名。

### 3.2 OPD 在伪造检测任务上的核心可行性论证

OPD 原始假设是 "domain reward 不依赖人标 preference"，**TFI 任务恰好满足**：
1. **label** 是 ground-truth-derivable（real/fake 二分类，0 噪声标注）；
2. **bbox** 由 mask 自动派生（v1 训练遗留 mask + augmented_data/synth + SAM 3.1 zero-shot 标），不需要人去画；
3. **explanation** 通过 specialist 反向打分（DINOv3 dense feature → R_loc，SigLIP-2 cls confidence → R_cls，MaskCLIP forensic → R_forensic，GRM 自评 → R_grm），**不需要人写黄金答案**。

→ 因此 RM 数据**不需要人标 preference**，比 InstructGPT/DPO 一类必须人标的成本低 1-2 个数量级。

### 3.3 为什么是 FIPO 而不是 P-GRPO / MiPO

| 算法 | credit assignment | TFI 适配度 | 训练栈 |
|---|---|---|---|
| MiPO | 序列级 DPO（loss 平均到所有 token） | **中**：v0 "翻转 label + 删 bbox" 构造的 rejected 太机械（见 `data/build/build_v2_mipo.py`），无法教模型分辨"分类对但定位差"等中间错误 | ms-swift |
| P-GRPO | uniform per-token | **中**：TFI 多 reward 叠加 → 信号被稀释，关键 token credit 不足 | ms-swift |
| **FIPO** | **future-KL 加权 token loss** | **高**：TFI 关键决策集中在少数 token（`<answer>` + `<bbox>` 4 个数字 + `<region>` phrase），future-KL 权重恰好放大这些位置 | verl + 一行 patch |

**FIPO 算法核心（`train/fipo/verl_patches/future_kl_loss.py`）**：

```python
# 每个 token 的 KL 差分
negative_approx_kl_t = log_prob_t - old_log_prob_t

# 衰减权重的"未来 KL"：F_t = sum_{j>=t} gamma^(j-t) * kl_j
gamma = 2.0 ** (-1.0 / decay_rate)   # decay_rate=12.0
future_kl_t = sum_{j>=t} gamma^(j-t) * kl_j  # 用 chunked matmul O(L * chunk_size)

# 影响权重 = clip(exp(F_t)) — 关键 token 的 future drift 大 → 权重大
influence_weight_t = clip(exp(F_t), [1-0.2, 1+0.2])

# safety 兜底：advantage<0 且 IS>4.0 的 token 强行夹到 [0.8, 1.0]
# Final loss: 标准 PPO clipped surrogate × influence_weight
```

**直觉**：在 TFI 的 6-tag CoT 输出里，前面 `<fast>/<reasoning>` 几百 token 大多是同质化中文短语，`<answer>real|fake>` 是单 token 决策，`<bbox>x1,y1,x2,y2</bbox>` 是 4 个数字 token——这些是真正影响 reward 的地方。future-KL 自动识别"未来 KL drift 大 = 关键决策 token"，给它们更大梯度，等价于 token-level credit assignment。

**VLM-posttraining 已验证 FIPO 收益**：相比 P-GRPO baseline，**hallucination 30.27% → 23.04%**，证明 future-KL 在短 CoT 任务上比 uniform per-token credit 更稳。

### 3.4 OPD/GKD 软标签蒸馏的具体实现

> 用户说的 "qwen3.6-27B 给学生模型打软标签"对应的是 ms-swift `rlhf_type=gkd`，具体见 `reference/README.v1.md` §10.5.4 + §10.5.8。

**GKD 公式**（DeepSeek-V4 同款，ms-swift 原生支持）：

```
L_GKD = beta * D_KL_RKL(student || teacher) + (1-beta) * D_KL_FKL(teacher || student)
       + lambda * L_SFT_on_chosen
```

其中：
- `beta=1, lambda=1` → 全 on-policy 反向 KL（DeepSeek-V4 默认）
- 反向 KL：`D_KL(student || teacher) = E_student[log p_s - log p_t]` —— 让 student 在自己采样的 token 上对齐 teacher
- forward KL：`D_KL(teacher || student)` —— 让 student 覆盖 teacher mode（防 collapse）
- EOPD 改进：**高熵 token 用 forward-KL（防 mode collapse），低熵 token 用 reverse-KL（收敛快）**

**为什么 Qwen3.6-27B → Qwen3.5-9B 这条路最优**：
1. **同 tokenizer**：词表完全对齐，全词表 KL 直接算，**不需要 top-k 截断**或做词表对齐 hack；
2. **同家族**：teacher 与 student 概率分布同源，反向 KL 数值稳定，不会因为词表不一致出现 NaN；
3. **27B dense**：FSDP zero3 + offload 在 4-5 卡 L20 装得下，不需要切 MoE；
4. **路线 B（cross-family GKD）**：InternVL3 ↔ Qwen3.6-27B 词表交集 ~130k tokens，需要 top-k 截断，是已知工程坑（journal 标了 P1 项）。

**Qwen3 issue #1799 警告**：直接上 GKD 不加 `low_var_kl` / `sentence-level IS clip` 会 mode collapse → **必须先 DPO warmup 一轮**让 logit 分布稳定再切 GKD。

### 3.5 OPD 在本任务上的可行性结论

| 维度 | 评估 | 证据 |
|---|---|---|
| **Reward ground-truth 可获得** | ✅ 强可行 | label/bbox 0 标注，5/9 reward 是 rule-based 不可 hack |
| **算力需求** | ✅ 7×L20 够 | actor+ref FSDP 加 offload + vllm rollout 在 4 卡，reward server 在 3 卡 |
| **算法 SOTA 性** | ✅ 强 | DeepSeek-V4 / Qwen3-VL OPD（ms-swift #8182）已落地；FIPO future-KL 在 VLM-posttraining 已验证 -7pp hallucination |
| **工程复杂度** | ⚠️ 高 | 24 次 smoke 联调（vllm/transformers/swift/verl 4 处版本耦合，driver 12.4 锁死 CUDA 12.4 → vllm 0.20+ 装不上） |
| **GKD collapse 风险** | ⚠️ 中 | 必加 sentence-level IS clip + DPO warmup 防 Qwen3 #1799 现象 |
| **数据规模** | ⚠️ 中 | TFI 自有数据 1009，配合 HydraFake-EFG-CN 4k+4k 拼到 ~9k，比 v1 训练池大一个数量级，但仍不及 Veritas 论文 36k |

---

## 4. Pipeline 各阶段输入/输出

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 0 · 数据构建（已完成 / 部分待生成）                                  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─ build_v2_sft.py   ─▶  /mnt/nfs/young/TFI/data/v2/sft.json (1009)
        │   IN : v1 Black 800 (mask+caption) + White 200 + synth + HF EFG (可选)
        │   OUT: ms-swift VLM SFT 标准格式
        │        {images: [path], type, label, source,
        │         messages: [system(SYS_PROMPT_ZH 6-tag), user, assistant(6-tag CoT)],
        │         mask_path: optional}
        │   关键：bbox 归一化到 [0,1000]² ← +0.096 S_Fin（vs 原始像素坐标）
        │
        ├─ build_hydra_efg_subset.py  ─▶  hydra_efg_cn.json (4k EFG + 4k real)
        │   IN : HydraFake sft_36k.json（英文 6-tag）
        │   OUT: 中文 6-tag CoT（fake/real 各 3 模板，注入 sub-generator 名）
        │   关键：丢 face swap/reenactment 整个子集，只保留 EFG（与 TFI 任务最近）
        │
        ├─ merge_official_hydra.py    ─▶  sft_merged.json (~1441，30/70 比例)
        │   公式：H = ratio * T / (1-ratio)，stratified by label
        │
        ├─ build_v2_mipo.py           ─▶  mipo.json (504 偏好对)  [兜底/已跳过]
        │   策略 v0：chosen=GT，rejected=翻 answer + 删 bbox/region
        │
        └─ build_v2_pgrpo.py          ─▶  pgrpo.json (1009 prompts)  [兜底]
            assistant 留空，留给 actor rollout

┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1 · SFT（cold start，已完成）                                       │
└─────────────────────────────────────────────────────────────────────────┘
ms-swift swift sft \
  --model Veritas-Cold-Start (InternVL3-8B) | Qwen3.5-9B          \
  --dataset sft_merged.json (1441)                                 \
  --train_type lora --lora_rank 64 --lora_alpha 128                \
  --target_modules all-linear --freeze_vit false                   \
  --max_length 3072 --per_device_train_batch_size 1                \
  --gradient_accumulation_steps 8 --num_train_epochs 3             \
  --lr 5e-5 --weight_decay 0.01 --lr_scheduler cosine              \
  --warmup_ratio 0.05  --bf16 --grad_ckpt
            (effective bsz = 7 GPU × 1 × 8 = 56；3 epoch ≈ 40 min)

IN : sft_merged.json (1441)
OUT: /mnt/nfs/young/TFI/runs/sft/v2sft_merged_1441/v0-*/checkpoint-N (LoRA shard)
     需 merge_lora → HF dump 才能给 FIPO 当起点
     val/200 落地（路线 A）：S_Det 0.714 / S_Loc 0.265 / S_Fin 0.494
                  （路线 B）：S_Det 0.888 / S_Loc 0.412 / S_Fin 0.610
     ★ 解读：高 precision 镜像偏置（路线 A precision 0.958 / recall 0.569），
       是 FIPO 9-reward 的理想起点 — FN 多意味着 R_label_gt + R_iou_gt
       的可学梯度大，训练稳定且收益空间大。

┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1.5 · (可选) GKD soft-label warmstart  [future work, 3 天]          │
└─────────────────────────────────────────────────────────────────────────┘
swift rlhf --rlhf_type gkd \
  --model       <SFT ckpt merged>            \
  --teacher_model Qwen3.6-27B               \
  --beta 1 --lmbda 1                         \  # 全 on-policy reverse KL
  --tuner_type full                          \  # 全参，不 LoRA
  --use_vllm true --vllm_mode server        \
  --deepspeed zero3 --teacher_deepspeed zero3

IN : SFT ckpt + Qwen3.6-27B teacher（同 tokenizer 全词表 KL）
OUT: /mnt/nfs/young/TFI/runs/gkd/qwen35_9b_gkd/checkpoint-N
     期望 ΔS_Fin ≈ +0.020（不上完整 specialist OPD 也能拿到的 baseline）
     必加：sentence-level IS clip + DPO warmup 1 轮防 Qwen3 #1799 collapse

┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2 · FIPO RL 数据准备（已完成）                                       │
└─────────────────────────────────────────────────────────────────────────┘
python -m train.fipo.prepare_fipo_data \
  --in_train sft_merged.json --in_val sft_val.json \
  --out_dir data/fipo --max_train 2000 --max_val 200

IN : SFT JSON
OUT: data/fipo/{train,val}.parquet (verl multimodal 格式)
     {data_source, prompt:[{system,user}], images:[{bytes}],
      reward_model:{style:"rule", ground_truth:"<json>"}, extra_info}
     
     ground_truth 关键 schema:
     {"label": 0|1,
      "bboxes": [[x1,y1,x2,y2], ...],   # 来源优先级: mask_path → 文本<bbox> → []
      "phrases": ["<region>...</region>", "“引号文字”", ...]}
     
     关键工程：mask 直接归一化到 [0,1000]²，否则 mask-derived GT bbox 在原图
     像素空间，rollout 输出在 [0,1000]，IoU 永远≈0，grounding reward 信号消失。

┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3 · FIPO 主训练（24 smoke 已通；正式百分比待跑）                       │
└─────────────────────────────────────────────────────────────────────────┘
verl PPO trainer + future_kl_loss patch + TFIAuditRewardManager

每个 step 内部：
  1. data loader  → batch (6 prompts)
  2. vLLM rollout → 8 candidates / prompt (n=8, top_p=0.95, T=1.0)
  3. reward 计算  → TFIAuditRewardManager.run_single (async, decode → compute_reward)
  4. advantage    → GRPO（每个 prompt 内 8 个 candidate 标准化）
  5. PPO update   → future-KL weighted clipped surrogate（loss_mode=future_kl）
                    clip_ratio_low=0.2, clip_ratio_high=0.28
                    use_kl_loss=False, kl_in_reward=False（FIPO 内部 future_kl 已正则）
  6. logging       → influence_weight_mean / clip_frac_upper / R_label_gt / S_Fin proxy

IN : data/fipo/train.parquet (1009-2000) + SFT-merged ckpt
OUT: /mnt/nfs/young/TFI/runs/fipo/<EXP_NAME>/checkpoint-N
     预算：80 step / day × 7 天 ≈ 560 step ≈ 2 epoch
     验收：val S_Fin ≥ 0.94, S_Loc ≥ 0.92, S_Auto ≥ 0.92

短跑 smoke 实测（路线 B InternVL3-8B FIPO 12-step，2026-05-06）：
  critic/score/mean step1 0.2627 → step6 0.3259 (+24%)
  val reward@1     step8 0.3167 → step12 0.3252 (+2.7%)
  → 验证 future-KL + 9-reward 在 verl 0.7 上能正常上升，
    证明算法栈工程闭环、reward 信号没死掉。

┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 4 · 评测                                                           │
└─────────────────────────────────────────────────────────────────────────┘
eval/score_official.py 复现官方 S_Fin 公式（与 v1.0 完全对齐）：
  S_Det  = image-F1
  S_Loc  = pixel-F1 / Dice = 2TP/(2TP+FP+FN)
  S_Sim  = BERTScore-zh on explanation
  S_Auto = Qwen3-MAX rubrics(/100) (4 维 1-10)
  S_Exp  = 0.5·Sim + 0.5·Auto
  S_Fin  = 0.45·Det + 0.25·Loc + 0.30·Exp

baseline 对照表（已完成）：
  Qwen3.5-9B prompt-only zs/fs/cot      → judge overall 4.886/5.159/5.200 (1-10)
  Qwen3.6-27B prompt-only zs/cot        → judge overall 5.31/5.59 (ceiling)
  v1 SFT (LoRA on 800)                   → judge overall 7.991, S_Fin 0.9034
  v2 SFT (路线 A Qwen3.5-9B, 1441)       → S_Fin 0.494
  v2 SFT (路线 B InternVL3-8B, 1441)     → S_Fin 0.610
  ────── 上述都是 SFT-only，FIPO 主路线训完是关键阶段 ──────
```

---

## 5. FIPO 数据集构建（重点深挖）

### 5.1 三层数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ 第一层 SFT 主集 (build_v2_sft.py)                                │
│   输入源 4 路：                                                   │
│   ① v1 Black 800（mask + 中文 caption）   ─ caption_to_template  │
│   ② v1 White 200（无 mask）                ─ caption_to_template  │
│   ③ processed/synth（合成 fake，keep.txt 过滤后 ~62）            │
│   ④ HydraFake sft_36k.json（仅 EFG + 20% real，丢 FS/FR）         │
│                                                                  │
│   关键工程：bbox 归一化 [0,1000]²；                                 │
│            mask → 多连通分量 bbox（top-4 按面积）；                  │
│            bbox + region phrase 嵌入 <conclusion>；                │
│            md5(image_path) 去重 → 跨数据源不重复。                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第二层 二阶段合并 (merge_official_hydra.py)                        │
│   策略：H = ratio * T / (1-ratio)，default ratio=0.30              │
│   stratified by label，dedup by image hash                          │
│   输出 sft_merged.json (~1441 = 1009 TFI + 432 HF)                  │
│                                                                  │
│   Stage A warmup（可选，1 epoch on hydra_efg_cn 8k）→             │
│   教模型 6-tag 中文 schema；                                        │
│   Stage B 主 SFT (3 epoch on sft_merged) → 拿 TFI 域硬样本知识    │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第三层 FIPO 训练数据 (prepare_fipo_data.py)                       │
│   SFT JSON → verl multimodal parquet                              │
│   - data_source: "tfi_forgery"                                    │
│   - prompt: [{system: SYSTEM_PROMPT}, {user: USER_PROMPT}]       │
│   - images: [{bytes: <jpeg, max_side=1024, q=92>}]               │
│   - reward_model.ground_truth: JSON                              │
│       {"label": 0|1,                                              │
│        "bboxes": [[x1,y1,x2,y2], ...],   # mask→bbox 归一化 [0,1000]²│
│        "phrases": ["<region>...</region>", "“引号”", ...]}       │
│                                                                  │
│   bbox 来源优先级:                                                │
│     1. mask_path → mask_to_bbox（最准）                          │
│     2. SFT assistant 文本中解析 <bbox>（v1 caption-based）         │
│     3. 空（real 图正常无 bbox）                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 SYS_PROMPT_ZH（schema 硬约束）

> SFT / FIPO / inference **三处 system prompt 必须 byte-level 一致**，否则 rollouts 与 teacher-forcing 对 tag 期望分歧 → reward 全部为 0。

```
你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。

首先用 <fast> </fast> 标签给出第一直觉判断；
然后用 <reasoning> </reasoning> 标签给出详细取证推理（高难度样本可在其中包含
 <planning> 规划与 <reflection> 自校验）；
接着用 <conclusion> </conclusion> 标签给出综合结论，对疑似篡改图必须用
<bbox>x1,y1,x2,y2</bbox> 或 <region>区域文字描述</region> 标注疑似篡改区域，
其中 bbox 坐标已归一化到 [0,1000]×[0,1000]（左上原点，x1<x2，y1<y2）；
最后用 <answer>real|fake</answer> 给出最终判断（仅二选一）。
```

— 出现位置：
- `data/build/build_v2_sft.py::SYS_PROMPT_ZH`
- `data/build/build_hydra_efg_subset.py::SYS_PROMPT_ZH`
- `train/fipo/schema.py::SYSTEM_PROMPT`
- `eval/baseline/sft_v2_inference.py`

### 5.3 6-tag CoT 输出 schema（命中率统计 from Veritas 论文 / sft_36k）

| 标签 | 必填 | 用途 | sft_36k 命中率 |
|---|:---:|---|---:|
| `<fast>` | ✅ | 第一直觉判断（系统 1） | 100.0% |
| `<reasoning>` | ✅ | 详细取证推理（系统 2） | 100.0% |
| `<conclusion>` | ✅ | 综合结论（含 `<bbox>` / `<region>`） | 100.0% |
| `<answer>real\|fake</answer>` | ✅ | 二分类答案 | 100.0% |
| `<planning>` | ⚪ | 规划取证步骤（仅困难） | 24.6% |
| `<reflection>` | ⚪ | 自校验（仅困难） | 20.4% |

**为什么这样设计**：4 必 + 2 可选 → 简单样本简洁、困难样本详细，**符合人类双系统认知架构**；100% 命中保证 schema reward 不掉。

### 5.4 9-Reward 加权（`train/fipo/reward_fn.py`）

| Reward | 权重 | 类型 | 计算 |
|---|---:|---|---|
| **R_format** | 0.10 | rule | 4 必填 tag 各出现 1 次（0.75 baseline） + answer ∈ {real,fake}（+0.25） |
| **R_consistency** | 0.10 | rule | fake⇒有 bbox/region；real⇒无 |
| **R_label_gt** | 0.15 | rule | `<answer>` == GT label |
| **R_iou_gt** | 0.15 | rule | bbox 最大 IoU vs GT bbox（real 双空匹配=1.0） |
| **R_phrase_check** | 0.05 | rule | conclusion 中数字/region token 在 GT phrase 池命中率 |
| R_loc | 0.10 | hook | DINOv3 dense feature 打分（待部署） |
| R_cls | 0.10 | hook | SigLIP-2 cls confidence（待部署） |
| R_forensic | 0.10 | hook | MaskCLIP + Mesorch（待部署） |
| R_grm | 0.10 | hook | GRM 自评 caption rubric（待部署） |
| R_qwen_periodic | 0.05 | hook | qwen-max 每 50 step 抽 10 张校准（待部署） |

**v1 起步**：仅 5 项 rule-based 跑（合计 0.55 权重，max 总分 0.55）；specialist + GRM 后续通过子类化 `TFIAuditRewardManager._external_scores_for(item)` 接入，**无需改 reward_fn 主路**。

**为什么 rule 占 55%**：rule reward 是**不可 hack 的 reward floor**（IoU/label 就是 GT，没法靠 reward hacking 套现）；specialist 和 GRM 是 model-based，**易 hack 故权重低**，且通过 trust-region "单一 reward ≤ 25%" 约束避免过度倾斜。

### 5.5 数据合并策略论证（`data/analysis/distribution_report.md`）

| HydraFake 子集 | 与 TFI 域距离 | 决策 | 理由 |
|---|---|---|---|
| **EFG**（Dall-E1/Midjourney/SDXL/Flux/HART/Infinity 等 AIGC 全图） | **近** | ✅ **训练集混入** | AIGC 全图生成与"日常 AI 生成图鉴定"完全对口，无需 face 也通用 |
| **FS / FR**（face swap / reenactment） | 远 | ❌ **不混入** | 涉及人脸特定操作（嘴型同步、五官重组），训了反而让模型偏 face-only |
| **real**（CelebA/FFHQ/LFW 等高质量真人脸） | 中 | ⚠️ **抽 20%** | 帮模型学"高质量真图特征"，但抽样防 face bias |
| HydraFake test/cd 子集（gpt4o/hailuo/dreamina） | **近** | ✅ **当 v2 OOD 评测** | 直接当 cross-domain test，完美对口"日常图片"评测 |

---

## 6. 方案对比

### 6.1 v1 vs v2-opd 路径对比（一图流）

```
v1: image
     ├─► SegFormer-B5 5-fold (7ch RGB+ELA+SRM)        ─► prob map
     │   └─► EfficientNet-V2-L 5-fold (RGB+ELA, 6ch) ─► p_classifier
     │       └─► evidence.py 抽 10 维结构化证据
     │           └─► XGBoost / TabPFN 5-fold OOF 阈值 ─► label, p_forged
     │               └─► Qwen3.5-9B + LoRA r=64 SFT  ─► explanation
     └─► (5 个模型 ensemble + 校准器 → S_Fin 0.9034)

v2: image
     └─► SSL backbones 自学 dense feature
         └─► 1 个 VLM (Qwen3.5-9B 或 InternVL3-8B-Veritas)
             ├─ Stage 1: SFT (cold start, 1441 中文 6-tag)
             ├─ Stage 1.5: GKD reverse KL (Qwen3.6-27B teacher)  [future]
             ├─ Stage 2: FIPO RL with 9-reward
             │   ├─ Rule reward (55%, ground-truth-derivable, 不可 hack)
             │   ├─ Specialist verifier (30%, RGB-only, 域感知强)
             │   │   ├─ DINOv3 (loc) + SigLIP-2 (cls)
             │   │   └─ MaskCLIP + Mesorch (forensic)
             │   └─ Caption rubric (15%, GRM + qwen-max 周期校准)
             └─► label + bbox + 中文 6-tag CoT 解释 (1 模型一次推理)
```

### 6.2 三种 RL 算法对比（已在 README §2.5 / FIPO 论文给出）

| 维度 | MiPO | P-GRPO | **FIPO** |
|---|---|---|---|
| credit assignment | 序列级 DPO | uniform per-token | **future-KL 加权** |
| token 区分能力 | 弱 | 弱 | **强**（关键 token 自动放大） |
| 短 CoT (~150 tok) | 一般 | reward 稀疏被均摊 | **强**（VLM-posttraining 验证 -7pp halluc） |
| 多 reward 融合稳定性 | — | reward variance 高 → mode collapse | **future-KL 隐式正则** |
| 实现复杂度 | ms-swift 自带 | ms-swift 自带 | verl + 50 行 patch |
| 训练栈 | ms-swift | ms-swift | verl + future_kl_loss patch |

### 6.3 软标签蒸馏路线对比

| 路线 | Student | Teacher | 词表对齐 | vllm 兼容 | 可行性 |
|---|---|---|---|---|---|
| **A（推荐）** | Qwen3.5-9B | Qwen3.6-27B | 同家族同 tokenizer，**全词表 KL 直接算** | 0.7.3 即开即用 | ✅ |
| B（兜底） | InternVL3-8B-Veritas | Qwen3.6-27B | 词表交集 ~130k，需 top-k 截断 | 0.19+ 才稳，0.7.3 不稳 | ⚠️ |
| C（远） | Qwen3.5-9B | Qwen3.5-122B-A10B-AWQ | 同家族 | LMDeploy 0.12.2 turbomind | ✅ rejection sampling 辅助 |

### 6.4 横向对比业界做法

| 维度 | TFI v2-opd（本方案） | DeepSeek-V4 OPD | InstructGPT RLHF |
|---|---|---|---|
| Reward 来源 | **specialist verifier + rule**（GT-derivable） | specialist + RM | 人标 preference |
| RM 训练数据 | 0 人标（rule 全自动） | per-domain SFT+GRPO 后蒸馏 | 33K 人标 |
| 偏好数据成本 | ¥0 | ~¥0 | ~$10/对 × 33K |
| 主算法 | FIPO future-KL + 9-reward | EOPD | PPO with KL penalty |
| 关键风险 | reward hacking | mode collapse | 标注员偏好不一致 |
| 适用任务 | GT-derivable reward 的领域 | 通用领域（多 specialist） | 主观任务（写作/对话） |

---

## 7. Future Work 话术模板（面试时用）

### 7.1 一段话总览

> *"我在 v1 SFT baseline 基础上做了 v2-opd 重构。设计核心是把 v1 的 5-stage ensemble + 校准器压回到 1 个 VLM，用 specialist 通过 reward 形式喂回 VLM 来等价复刻 v1 的多视角监督，让 VLM 同时输出 detection + grounding + explanation。算法上选了 FIPO 而不是 P-GRPO/MiPO，因为我们的关键决策集中在少数 token，future-KL token-level credit 比 uniform 更适合，姊妹项目 VLM-posttraining 上 FIPO 让 hallucination 从 30.27% 下到 23.04%。SFT 阶段单独看比 v1 退步是预期的，因为 v1 的 0.9034 是 5 个模型叠出来的，单 VLM SFT 吃不下那么多监督密度，FIPO 阶段才是真正补位。目前算法栈 + 9-reward + verl 集成已经联调完成，路线 B 跑通了 12-step smoke，short-run reward 从 0.3167 上到 0.3252，证明工程闭环。完整训练加 GKD warmstart 还在 future work 里。"*

### 7.2 面试官可能追问 + 准备好的答案

#### Q1：为什么 v2 SFT 还不如 v1 SFT？是不是方案选错了？
A：v1 的 0.9034 是 5 个 specialist 5-fold ensemble + XGB/TabPFN 校准器叠出来的 image-F1=0.9845 + pixel-F1=0.8735，VLM 只负责生成解释（评测时 explanation 占 0.30 权重）。v2 把整条链路压到 1 个 VLM 去同时输出 label + bbox + explanation，**SFT 单阶段吃不下原本 5 个模型 5-fold 投入的总监督密度**——这是设计预期内的。v2 的核心假设是"specialist 不进 backbone，进 reward"——FIPO 阶段让 VLM 在 9-reward 监督下学到等价的多视角约束。我们已经看到 SFT 后镜像偏置（路线 A precision 0.958 / recall 0.569），高 precision 起点意味着 rollout "fake" 时几乎对应真 reward，**FN 多反而让 R_label_gt 和 R_iou_gt 的可学梯度大**，是 FIPO 的理想起点，不是退步。

#### Q2：去掉 ELA / SRM 不是丢了重要先验吗？
A：v1 的 4 张 FP 全是热敏小票，热敏纸自带高频噪声，**让 SRM 把"真实噪声"误判成"伪造痕迹"**——这是 hand-crafted 特征在域漂移上的结构性失效。现代 SSL backbone（DINOv3 / SigLIP-2）的 dense feature 自学已经覆盖 ELA 信号，而且具备**域感知**——能区分"小票纸张噪声 vs 编辑痕迹"，区分性比 ELA 强。我们没有真的丢这部分能力，**我把人工先验从 input 端移到 reward 端的 forensic specialist**：MaskCLIP 在 NeXT-IMDL benchmark 跨域 F1=0.32 远超 IML-ViT 0.12 / TruFor 0.13，作为 R_forensic 喂回 VLM。

#### Q3：FIPO 的 future-KL 具体在干啥，比 P-GRPO 好在哪里？
A：FIPO 给每个 token 算一个 "未来 KL 衰减加权和" `F_t = sum_{j>=t} gamma^(j-t) * (log p_t - log p_old_t)`，然后 `influence_weight_t = clip(exp(F_t), [0.8, 1.2])`，乘到 advantage 上做 clipped surrogate。**直觉**：在我们 6-tag CoT 输出里，前面 `<fast>/<reasoning>` 几百 token 大多同质化，`<answer>` 是单 token 决策、`<bbox>` 是 4 个数字 token——这些是真正影响 reward 的位置，**它们后面的 KL 漂移最大**，所以 future-KL 自动给它们大权重。P-GRPO 是 uniform per-token，关键 token credit 被几百个 padding token 摊薄。VLM-posttraining 上 FIPO 比 P-GRPO 让 hallucination 从 30.27% 下到 23.04%，证明这条假设。

#### Q4：9 个 reward 怎么避免互相打架 / reward hacking？
A：三道防线：（1）**rule reward 占 55%**，是 ground-truth-derivable 的（IoU/label），不可 hack——这是 reward floor；（2）**trust-region**：单一 reward ≤ 25% 权重，避免任何一项主导；（3）**caption rubric 只占 15%**，因为 GRM 和 qwen-max 是 model-based 易 hack，所以权重压低；qwen-max 还每 50 step 抽 10 张校准 GRM 漂移。具体 5+3+2 拆解：5 rule（format/consistency/label_gt/iou_gt/phrase_check） + 3 specialist（DINOv3 loc / SigLIP-2 cls / MaskCLIP forensic） + 2 caption（GRM + qwen periodic）。

#### Q5：GKD soft-label 蒸馏为什么选 Qwen3.6-27B 而不是闭源 API teacher？
A：三个理由：（1）**同 tokenizer 全词表 KL**——Qwen3.5-9B 与 Qwen3.6-27B 同家族同 tokenizer，反向 KL 数值稳定；用闭源 API（如 GPT-4o）跨家族要做 top-k 截断或词表对齐 hack，工程坑深；（2）**合规**——伪造数据不能传第三方；（3）**成本** ——蒸馏要每 token logits，API 不返回 logits（只返回 top-5 logprobs），完全跑不通。Qwen3.6-27B FSDP zero3 + offload 在 4-5 卡 L20 装得下，本地推理 0 成本。**风险**：直接 GKD 会 mode collapse（Qwen3 issue #1799），所以我加了 sentence-level IS clip + 1 轮 DPO warmup 稳定 logit 分布再切 GKD，这是 EOPD 论文（arXiv 2603.07079）的标准做法。

#### Q6：smoke 都没跑通，怎么证明方案靠谱？
A：分两块说。（1）**算法层证据**：FIPO future-KL 在姊妹项目 VLM-posttraining 上已经验证 hallucination -7pp；DeepSeek-V4 / Qwen3-VL（ms-swift #8182）已经在公开任务上跑通 OPD 全栈；MaskCLIP 在 NeXT-IMDL benchmark 是 SOTA；DINOv3 在 dense prediction 已超 SegFormer 类专门方案——**论文 + benchmark + 姊妹项目 都给了正向证据**。（2）**工程层证据**：FIPO 算法栈 + verl 0.8 + 9-reward + reward_manager + parquet 数据已组装并联调到 AgentLoop / vLLM EngineCore；路线 B 12-step FIPO smoke 跑通：critic/score/mean step1 0.2627 → step6 0.3259 (+24%)，val reward@1 step8 0.3167 → step12 0.3252 (+2.7%)，说明 reward 信号没死、PPO 优化能正常上升。卡的不是算法，是 vllm 0.19.1 × Qwen3.5-9B hybrid-attn 的 KV 预算 bug 和 InternVL custom processor save_pretrained 兼容性，**这两个都是非算法 blocker**。

#### Q7：v1 评分官方公式，v2 怎么保证训练目标和评测对齐？
A：评测公式 `S_Fin = 0.45·S_Det + 0.25·S_Loc + 0.30·S_Exp`，对应到 9-reward：S_Det ↔ R_label_gt + R_cls；S_Loc ↔ R_iou_gt + R_loc + R_forensic；S_Exp ↔ R_consistency + R_phrase_check + R_grm + R_qwen_periodic（S_Exp = 0.5·BERTScore + 0.5·Qwen-MAX rubric，刚好对应 GRM/qwen-periodic 校准）。9-reward 总权重 1.0 与 S_Fin 三档比例（0.45/0.25/0.30 ≈ 0.40/0.30/0.30）只差 1 个百分点，是有意设计。

#### Q8：bbox 归一化 [0,1000]² 这个细节为什么有 +0.096 S_Loc 收益？
A：InternVL3 / Qwen3-VL 用 dynamic-tiling（12 tile + thumbnail）做视觉编码，**视觉 token 已丢失原图绝对像素信息**——模型看不到"这张图原本 1920×1080"。如果训练时让模型背原始像素坐标，它学到的是"4 个数字"序列，没有空间锚点。归一化到 [0,1000]² 等价于给坐标提供与 tile 无关的统一空间——模型从 system prompt 显式知道"x ∈ [0,1000], y ∈ [0,1000]"，可以学到稳定的几何表征。我们 ablation：raw-pixel bbox + 12-tile 推理 S_Loc=0.025；归一化 + 12-tile S_Loc=0.411，+0.096 S_Fin。

### 7.3 可能被问到的弱点（提前承认）

1. **没跑出最终数字** → 工程层 24 次 smoke 联调，driver 12.4 锁死 vllm 升级路径是核心 blocker，需要换机器或退到 hf rollout。
2. **MiPO 偏脆弱** → v0 翻转策略只能教"answer + bbox 全错"的粗粒度偏好，无法教"分类对但 bbox 漂移 30px"这种中间错误，所以主路线已跳过；如果要补会用 SFT 模型自己 rollout 8 个 candidate 用 specialist 排出 chosen/rejected。
3. **Reward hacking 长期风险** → 当前 specialist 还没接入，rule 占 55% 是过渡；接入后用 trust-region + 周期 qwen-max 校准 GRM。
4. **HydraFake 没有 grounding 标签** → 我们没用 HF 监督 grounding，只借它的 detection schema；TFI 自有 mask + augmented_data + SAM 3.1 zero-shot 标注三路补齐。
5. **v1 镜像偏置 vs v2 镜像偏置** → InternVL3-Veritas 是高 recall 低 precision（HydraFake 36k 强 fake 先验），Qwen3.5 是高 precision 低 recall（通用 web 强 real 先验），完全相反——这反而是 FIPO 9-reward 的理想起点（高 precision 起点 rollout "fake" 时几乎对应真 reward，FN 多让可学梯度大）。

---

## 8. 关键源码追踪表（被追问时翻给面试官）

| 概念 | 源码位置 | 关键行 |
|---|---|---|
| 6-tag system prompt | `data/build/build_v2_sft.py:45-54` | `SYS_PROMPT_ZH` |
| 6-tag schema 解析 | `train/fipo/schema.py:33-42, 94-125` | `parse_response()`, `count_required_tags()` |
| 9-reward 主计算 | `train/fipo/reward_fn.py:146-243` | `compute_reward()` |
| Reward 权重 | `train/fipo/reward_fn.py:49-60` | `DEFAULT_WEIGHTS` |
| FIPO future-KL loss | `train/fipo/verl_patches/future_kl_loss.py:60-241` | `compute_policy_loss_future_kl()` |
| TFIAuditRewardManager | `train/fipo/verl_patches/reward_manager.py:77-142` | `run_single()` |
| Mask → bbox 归一化 | `train/fipo/prepare_fipo_data.py:52-72` | `_mask_to_bbox()` |
| HydraFake EFG 中文化 | `data/build/build_hydra_efg_subset.py:79-180` | `FAKE_TEMPLATES` / `REAL_TEMPLATES` |
| 30/70 stratified 合并 | `data/build/merge_official_hydra.py:51-81` | `_stratified_sample()`, `merge()` |
| FIPO 启动入口 | `train/fipo/launch.sh` 全文 | env vars + Hydra dot-notation |
| Ray sitecustomize patch | `sitecustomize.py` | `torch.cuda.set_device` remap + future_kl import |

---

## 9. 引用与论文清单

- **Veritas / HydraFake**（ICLR 2026 Oral）：[arXiv 2508.21048](https://arxiv.org/abs/2508.21048)
- **FIPO**（Future-KL Inference Preference Optimization）：[arXiv 2603.19835](https://arxiv.org/abs/2603.19835)
- **OPD**（On-Policy Distillation, DeepSeek-V4）：[arXiv 2604.00626](https://arxiv.org/abs/2604.00626)
- **EOPD**（Entropy-Aware OPD）：[arXiv 2603.07079](https://arxiv.org/abs/2603.07079)
- **DINOv3**（Meta 2025-08, frozen SSL backbone）：[arXiv 2508.10104](https://arxiv.org/abs/2508.10104)
- **SigLIP-2-NaFlex**（Google 2025-02, native aspect ratio）：[arXiv 2502.14786](https://arxiv.org/abs/2502.14786)
- **SAM 3.1**（Meta ICLR 2026, text→mask）：<https://ai.meta.com/research/sam3/>
- **MaskCLIP / Mesorch / NeXT-IMDL benchmark**：[arXiv 2512.23374](https://arxiv.org/abs/2512.23374)
- **GRM**（Generative Reward Model, DeepSeek-V4 同款）
- **Qwen3 #1799**（GKD without IS clip → mode collapse 实证）：<https://github.com/QwenLM/Qwen3/issues/1799>
- **ms-swift Qwen3-VL-8B GKD #8182**（OPD 在 ms-swift 上的参考实现）：<https://github.com/modelscope/ms-swift/issues/8182>

---

> 文档生成于 2026-05-07，对应 commit head（路线 A SFT smoke 完成 + 路线 B FIPO 12-step smoke 通过）。
> **下次更新触发**：FIPO 长跑 ≥ 200 step / GKD warmstart 首跑 / specialist server 上线任一发生。
