# Veritas 方法 + HydraFake 数据集报告（TFI v2 参考）

> 论文：[arXiv 2508.21048 (ICLR 2026 Oral)](https://arxiv.org/abs/2508.21048) · 代码：[EricTan7/Veritas](https://github.com/EricTan7/Veritas) · 数据：[modelscope EricTanh/HydraFake](https://www.modelscope.cn/datasets/EricTanh/HydraFake)
>
> 本文件是 TFI v2-opd 整合 Veritas 工作的**完整工作笔记**，主 README 只保留摘要并链接到此。

---

## 0. 一句话总结

Veritas 用 **pattern-aware reasoning** 输出模板（6 标签结构）+ **三阶段训练**（SFT → MiPO → P-GRPO），让 InternVL3-8B 在 face deepfake 4 级 OOD 评测上达到平均 **92.1%** 准确率，相对 vanilla CoT 提升 ~10 个百分点。

---

## 1. Veritas 方法详解

### 1.1 模型架构
- **Base**: InternVL3-8B（Qwen2.5-7B LLM + InternViT-300M-448 vision encoder，自定义 `InternVLChatModel`，需 `trust_remote_code=True`）
- **Framework**: [ms-swift](https://github.com/modelscope/ms-swift)（与 TFI README §四 一致）
- **输出格式**：6 个 XML 标签的 reasoning template

| 标签 | 必填？ | 用途 | 在 sft_36k 命中率 |
|---|:---:|---|---:|
| `<fast>...</fast>` | ✅ 必填 | 第一直觉判断（系统 1） | 100.0% |
| `<reasoning>...</reasoning>` | ✅ 必填 | 详细取证推理（系统 2） | 100.0% |
| `<conclusion>...</conclusion>` | ✅ 必填 | 综合结论 | 100.0% |
| `<answer>real|fake</answer>` | ✅ 必填 | 最终二分类答案 | 100.0% |
| `<planning>...</planning>` | ⚪ 可选 | 规划取证步骤（仅困难样本） | 24.6% |
| `<reflection>...</reflection>` | ⚪ 可选 | 自校验（仅困难样本） | 20.4% |

设计意图：4 必 + 2 可选 = 简单样本简洁、困难样本详细，**符合双系统认知架构**。

### 1.2 三阶段训练 pipeline

#### Stage 1: Cold-Start SFT
- 数据：`sft_36k.json`（36,750 样本，含 6 标签完整回复）
- 脚本：`Veritas/self_scripts/train/train_sft.sh`
- 关键超参：
  ```
  --model /path/to/InternVL3-8B
  --train_type lora
  --lora_rank 128 --lora_alpha 256
  --freeze_vit false        # vision encoder 也训
  --num_train_epochs 3
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 8
  --learning_rate 5e-5 --weight_decay 0.01
  --max_length 2048
  --torch_dtype bfloat16
  ```
- 8 卡 effective batch size = 1 × 8 × 8 = 64

#### Stage 2: MiPO (Mixed Preference Optimization)
- 数据：`mipo_3k.json`（3,480 偏好对，单字段 `rejected_response`）
- 与纯 DPO 区别：保留 SFT loss 项，只在错答上加 preference loss → **更稳，不易 collapse**
- chosen = sft_36k 风格的标准回复；rejected = LLM 生成的"看似正确但取证逻辑错"的答案

#### Stage 3: P-GRPO (Pattern-aware GRPO)
- 数据：`pgrpo_8k.json`（8,033 prompt + label，assistant 留空）
- Reward model：`CodeGoat24/UnifiedReward-qwen-3b`（论文用，可换 7B 更强）
- Reward 设计：
  - **format reward**：6 标签结构是否正确
  - **answer reward**：二分类是否正确
  - **pattern reward**：planning/reflection 等高级 pattern 是否合适使用
  - **reflection quality**：UnifiedReward 打分

### 1.3 评测协议（HydraFake 4 级 OOD）
共 53,272 图、严格 50/50 平衡，分 4 个泛化层级：

| 层级 | 含义 | 样本数 | 论文 Veritas 准确率 |
|---|---|---:|---:|
| **id** (in-domain) | 训练见过的伪造方法 | 13,819 | 97.3% |
| **cm** (cross-model) | 同伪造类型，新模型 | 11,249 | 98.6% |
| **cf** (cross-forgery) | 新伪造类型 | 12,736 | 90.3% |
| **cd** (cross-domain) | 完全跨域 | 15,468 | 82.2% |
| **平均** | — | 53,272 | **92.1%** |

消融关键：去掉 MiPO，平均掉到 ~90.7%（cf/cd 上掉得更多）；说明**MiPO 对 OOD 泛化贡献最大**。

---

## 2. HydraFake 数据集深度报告

### 2.1 训练集（48,320 张图，3 个训练 json）

| json | 样本数 | 大小 | 用途 | 字段 |
|---|---:|---:|---|---|
| `sft_36k.json` | **36,750** | 95 MB | Cold-Start SFT | `images / type / video_id / label / messages` |
| `mipo_3k.json` | **3,480** | 15 MB | MiPO 偏好对齐 | sft 字段 + `rejected_response`（一段错误推理文本） |
| `pgrpo_8k.json` | **8,033** | 9 MB | P-GRPO 在线 RL | 仅 prompt + label（assistant 留空，待 rollout） |
| `all.json` | 48,320 | 55 MB | 全量参考 | — |

**SFT 36k 标签分布**：real 18,712 / fake 18,038（≈ 1:1，无 imbalance）

**SFT 36k 伪造类型分布（3 大类，9+6+6 子类）**：

| 大类 | 数量 | 子类 |
|---|---:|---|
| **real** | 18,712 | CelebA / CelebAHQ / FFHQ / FF++ / LFW |
| **face swapping (FS)** | 7,402 | FaceForensics++ / blendface / facedancer / fsgan / mobileswap / simswap |
| **entire face generation (EFG)** | 5,573 | Dall-E1 / Midjourney / SD-Cascade / SD-AandE / SDXL / StyleGAN / StyleGAN2 / VQGAN / seeprettyface |
| **face reenactment (FR)** | 5,063 | AniPortrait / EmoPortrait / Hallo / Hallo2 / LivePortrait / facevid2vid |

**回复长度（字符）**：median = 1,269 / mean = 1,467 / min = 120 / max = 4,041
（≈ 320–370 token，适合 max_new_tokens=1024 训练设定）

### 2.2 测试集（4 级 OOD，53,272 张）

| 层级 | 子集 | 样本数 |
|---|---|---:|
| **id** | FF++ (8959) / Hallo2 (1660) / Midjourney (600) / StyleGAN (600) / facevid2vid (2000) | 13,819 |
| **cm** | AdobeFirefly (600) / Flux1.1Pro (600) / HART (4201) / Infinity (4200) / MAGI (1048) / StarryAI (600) | 11,249 |
| **cf** | codeformer (1750) / faceadapter (300) / iclight (2082) / infiniteyou (3244) / pulid (3360) / starganv2 (2000) | 12,736 |
| **cd** | FFIW (6832) / deepfacelab (3094) / dreamina (952) / **gpt4o** (630) / hailuo (1000) / infiniteyou (2960) | 15,468 |

**val 集**：4,000 张（real 2,000 + fake 2,000），无层级，纯 binary 验证。

### 2.3 数据质量评分

| 维度 | 评分 | 说明 |
|---|:---:|---|
| 规模 | ⭐⭐⭐⭐⭐ | 训练 48k + 测试 53k，远超 v1 的 800+200 |
| 标签质量 | ⭐⭐⭐⭐⭐ | 100% 二分类标签 + 类型标签 + 子类标签 |
| Reasoning 模板 | ⭐⭐⭐⭐⭐ | 6 标签结构、必填 100%、长度合理 |
| 类型覆盖 | ⭐⭐⭐ | 仅人脸 3 大类，缺 OCR 篡改 / 商品图 / 文字替换 |
| OOD 评测 | ⭐⭐⭐⭐⭐ | 4 级层次设计是当前 deepfake benchmark 最严格的之一 |
| Grounding 标签 | ⭐ | **完全没有 bbox/mask 标签** |
| 中文 | ⭐ | 全英文 reasoning |

---

## 3. HydraFake 怎么用到 TFI 任务（**实操方案**）

### 3.1 图像数据：选择性混入（不全用，不全弃）

| HydraFake 子集 | 与 TFI 域距离 | 决策 | 理由 |
|---|---|---|---|
| **EFG**（Dall-E1/Midjourney/SDXL/Flux/HART/Infinity 等 AIGC 生成图） | **近** | ✅ **训练集混入 100%** | AIGC 全图生成与"日常 AI 生成图鉴定"完全对口，无需 face 也通用 |
| **FS / FR**（face swap / reenactment） | 远 | ❌ **不混入** | 涉及人脸特定操作（嘴型同步、五官重组），训了反而让模型偏 face-only |
| **real**（CelebA/FFHQ/LFW 等高质量真人脸） | 中 | ⚠️ **抽 20% 当 hard-real** | 帮模型学"高质量真图特征"，但抽样防 face bias |
| **HydraFake test/cd 子集**（gpt4o / hailuo / dreamina） | **近** | ✅ **当 v2 OOD 评测扩展** | 直接当 cross-domain test，完美对口"日常图片"评测 |

**预估实际混入量**：EFG 5,573 + real 抽样 ~3,700 ≈ 9,300 张 HydraFake 图加入 TFI v2 训练集（v1 800 张 + augmented_data + 这 9,300 = ~20K 训练规模）。

### 3.2 训练 jsons：100% 借用模板，逐条改写图片路径

- `sft_36k.json` 整体借鉴 messages 格式 → **从中筛选 EFG + real 部分（约 24k 样本）**直接用，face swap 部分丢弃
- `mipo_3k.json` rejected_response 模板 → 学习其错误推理写法，**自合成** TFI 域的偏好对
- `pgrpo_8k.json` → 直接当 P-GRPO rollout 数据池（reward model 在线打分）

### 3.3 Reasoning Template：直接抄结构，中文化 + 加 grounding

把 system prompt 改成：
```
你是图像伪造鉴定专家。任务是对给定图像判断真伪、定位伪造区域并给出可解释分析。

首先用 <fast>...</fast> 给出第一直觉判断；
然后用 <reasoning>...</reasoning> 详细取证推理（疑难样本可包含 <planning> 规划与 <reflection> 自校验）；
接着用 <conclusion>...</conclusion> 综合结论 + 用 <region>...</region> 描述疑似篡改区域；
最后用 <answer>real|fake</answer> 给最终判断。

如果是 fake，conclusion 必须包含至少一个 <region>x1,y1,x2,y2|文字描述</region> 标注。
```

---

## 4. 关于 Grounding 标签的明确决策

### 4.1 HydraFake 提供 grounding 监督吗？
**不提供**。HydraFake 是纯 detection benchmark，所有图标签只有 `label∈{0,1}` + `type` 字符串，**没有 bbox / 没有 mask**。原因：face deepfake 通常是整脸生成或整脸替换，不需要 region 定位。

### 4.2 TFI v2 还要 grounding 吗？
**必须要**。原因：
- 比赛官方指标 `S_Loc = 25%`（pixel-F1 / Dice），grounding 是核心
- v1 已经达到 S_Loc = 0.8735（pixel-F1）
- 没 grounding 就退化成纯 detection

### 4.3 grounding 标签的 3 个来源（**与 HydraFake 互补**）

| 来源 | 类型 | 数量 | 质量 |
|---|---|---:|---|
| **v1 训练遗留**（`/mnt/nfs/.../train_resume/Black/Mask/`） | 人工标注 mask | ~640 | 高 ✓ |
| **v2 augmented_data/synth**（`/mnt/nfs/.../augmented_data/synth`） | 合成时自动生成的 GT mask | ~62 keep + 更多备选 | 高（合成时已知 ground truth） |
| **SAM 3.1 phrase grounding 自动标注** | text → mask（zero-shot） | 任意（按需） | 中（用于 unannotated 数据） |

**结论**：grounding 标签**不依赖** HydraFake，已有数据足够 + SAM 3.1 可补。HydraFake 只贡献 detection + reasoning 监督。

### 4.4 训练时如何让 LLM 输出 grounding？
- **TFI 自有数据**：messages 的 assistant 字段在 `<conclusion>` 内嵌入 `<region>x1,y1,x2,y2</region>` 监督
- **HydraFake EFG 借用样本**：因为是整图 AIGC，conclusion 里写 `<region>整图</region>` 或 `<region>global</region>` 即可
- **inference 时**：模型输出的 `<region>` 标签 → 解析成 bbox → SAM 3.1 phrase prompt 精化为 pixel mask

---

## 5. v2 训练脚本如何改造 Veritas 的脚本

### 5.1 字段映射（Veritas json → TFI json）

| Veritas 字段 | TFI v2 字段 | 改造点 |
|---|---|---|
| `images: ['hydrafake/train/...']` | `images: ['/mnt/nfs/.../train_resume/Black/Image/0001.png']` | 路径绝对化 |
| `type: 'face swapping' / 'real'` | `type: 'splice' / 'copy-move' / 'aigc-global' / 'real' / ...` | 7+ 类替代 3 大类 |
| `label: 0/1` | `label: 0/1` | 不变 |
| `messages[2].content`（assistant） | 同结构，**中文 + 加 `<region>` 标签** | template 改造 |

### 5.2 训练脚本参数调整（基于 `Veritas/self_scripts/train/train_sft.sh`）

```bash
# 改动点
--model /mnt/nfs/young/TFI/models/Veritas-Cold-Start  # 用 Cold-Start 起点（Veritas 推荐）
--dataset /mnt/nfs/young/TFI/data/v2_sft.json         # TFI 自合成 + HydraFake EFG 子集 = ~24K
--lora_rank 64                                        # 降到 64（v1 也是 64，省卡）
--num_train_epochs 2                                  # v1 跑 3 epoch 收敛，2 epoch 足够
--max_length 3072                                     # 中文回复 token 多，加大
--output_dir /mnt/nfs/young/TFI/checkpoints/v2_sft    # 写 NFS
```

8 张 L20 (46G/卡) → effective batch 64，预估单 epoch ≈ 2-3 小时（24K 样本）。

---

## 6. 与 v2 原计划（README §五）的差异总览

| 阶段 | v2 原计划 | 整合 Veritas 后 | 增量收益 |
|---|---|---|---|
| **Cold-Start SFT** | Qwen3.5-9B + LoRA on TFI 800 张 | Veritas-Cold-Start (InternVL3-8B) + LoRA on **TFI 800 + HydraFake EFG 24K** | 训练规模 ×30，收敛更稳 |
| **偏好对齐** | DPO | **MiPO**（保留 SFT loss + preference loss 混合） | 更不易 collapse |
| **RL** | RLVR + EOPD | **P-GRPO**（pattern-aware reward + UnifiedReward） | 反思质量更高 |
| **GKD** | Qwen3.6-27B → 9B | **保留**（与 P-GRPO 路并行做对比） | 蒸馏路线作为 backup |
| **评测** | val 200 算总分 S_Fin | **拆 4 级 OOD**（id/cm/cf/cd 各报） | 看真实泛化能力 |

---

## 7. 参考链接
- 论文：<https://arxiv.org/abs/2508.21048>
- 代码（已 clone）：`/mnt/nfs/young/TFI/code/Veritas/`
- 训练数据：`/mnt/nfs/young/TFI/data/HydraFake/jsons/train/`
- 测试数据：`/mnt/nfs/young/TFI/data/HydraFake/jsons/test/`
- Cold-Start 模型：`/mnt/nfs/young/TFI/models/Veritas-Cold-Start/` (InternVL3-8B, 15 GB)
- Final 模型：`/mnt/nfs/young/TFI/models/Veritas/` (15 GB)
- Reward 模型：`/mnt/nfs/young/TFI/models/UnifiedReward-qwen-3b/` (7.1 GB)
