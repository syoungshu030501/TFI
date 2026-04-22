# TFI - 图像伪造检测项目 (Text & Forgery Investigation)

> **2026-04 更新**: 已切换到 Qwen3.5-9B 证据驱动方案。本节以下记录的是 **旧 397B 教师蒸馏方案**(已归档至 `legacy/`), 新方案见 [README.md](README.md) 第八节"关键设计"以及下面"零、新方案速览"。

## 零、新方案速览 (2026-04 重构)

| 维度 | 旧方案 (legacy/) | 新方案 (主干) |
|---|---|---|
| VLM 基座 | Qwen3-VL-8B-Thinking | **Qwen3.5-9B** (2026-03 阿里发布, 原生多模态) |
| 教师 | Qwen3.5-397B-A17B (LoRA) | **无** (直接 LoRA 微调小模型) |
| 训练数据 | 1000 + 3000 教师增强 | 1000 原始 + 证据 prompt |
| 关键创新 | CoT 蒸馏 | **结构化证据抽取** + **XGBoost 校准器** |
| 算力门槛 | 8x MI325X 256GB | **1-2 卡 RTX 5090 / L20 (46GB)** |
| 数据生成时间 | ~47 小时 | 0 |
| 重构原因 | 算力高、坐标幻觉、提升有限 | 用证据 grounding 解决幻觉, 用 calibrator 替代手调阈值 |

新增模块:
- `evidence.py` — 从 mask + ELA + SRM 抽取 10 维结构化证据 (bbox/面积/异常度比值/边缘清晰度)
- `calibrator.py` + `train_calibrator.py` — XGBoost 校准器, 替代旧 `cls_override_low/high` 硬规则
- `vlm_collator.py` — 抽取自 legacy/train_teacher.py, 兼容 Qwen3.5/Qwen3-VL 的 collator 与 LoRA target 自动发现
- `train_qwen35_9b.py` — Qwen3.5-9B LoRA / 全量微调脚本
- `config.yaml` + `run_pipeline.sh` — 统一配置 + 一键脚本

新增 `dataset.py: VLMSFTDataset(inject_evidence=True)` — 训练时把 GT mask 抽取的 evidence JSON 注入 prompt, 与推理 prompt 严格一致 → 消除训练-推理 gap。

旧脚本归档目录: [legacy/README.md](legacy/README.md)

---

## 一、项目概述

本项目实现一个完整的图像伪造检测系统，完成三个子任务：

1. **伪造判别 (Task 1)**: 判断图片是否经过伪造，输出 label=0(真实) / label=1(伪造)
2. **伪造定位 (Task 2)**: 像素级定位伪造区域，输出 COCO RLE 格式的二值 mask
3. **可解释分析 (Task 3)**: 生成详细的中文鉴定分析文本，说明判断依据

最终输出为 `submit.csv`，格式：`image_name, label, location, explanation`

---

## 二、硬件与软件环境

### 2.1 硬件

- **GPU**: 8x AMD Instinct MI325X (每卡 256GB HBM3, 共 2TB)
- **CPU RAM**: 3TB
- **GPU 架构**: ROCm 7.1
- **推理限制**: 单卡 48GB 显存

### 2.2 软件版本

| 包 | 版本 | 说明 |
|---|------|------|
| PyTorch | 2.11.0.dev20260216+rocm7.1 | ROCm nightly (从 2.10.0 升级, 原版已不在索引) |
| transformers | 5.2.0 | 升级以支持 `qwen3_5_moe` 架构 |
| peft | 0.18.0 | LoRA 适配器 |
| deepspeed | 0.18.6 | ZeRO-3 分布式训练 |
| accelerate | 1.9.0 | 分布式启动 |
| fla | (随 transformers 安装) | Flash Linear Attention, Qwen3.5 DeltaNet 依赖 |
| causal-conv1d | 1.5.0.post8 | DeltaNet 因果卷积 (重新编译匹配新 PyTorch) |
| qwen-vl-utils | 最新 | Qwen VL 图像/视频处理 |

### 2.3 ROCm 兼容性问题与修复

**问题**: `torch._grouped_mm` 使用 CK (Composable Kernel) grouped GEMM 后端, 在 MI325X 上 workspace buffer 未正确分配, 导致 MoE 前向传播时崩溃:
```
RuntimeError: The gemm workspace buffer is not allocated!
```

**修复**: `rocm_compat.py` — 将 `torch._grouped_mm` 替换为逐专家顺序矩阵乘法 (含 `bias` 参数支持), 功能等价, 所有加载 Qwen3.5 模型的脚本在导入 transformers 之前调用 `rocm_compat.patch_grouped_mm()`。

**影响**: 推理速度略慢 (~2.9 tok/s 图像模式, ~10 tok/s 纯文本模式), 训练 ~350s/step。功能和精度不受影响。

**PyTorch 版本升级注意**: 安装 vLLM 时意外替换了 ROCm PyTorch → CUDA PyTorch, 后手动恢复为 2.11.0 nightly。由此导致 `causal_conv1d`, `torchao`, `fbgemm_gpu` 等预编译扩展的 ABI 不兼容, 需重新编译 `causal_conv1d` (已完成), 卸载 `torchao` (SGLang 依赖, 已弃用)。

---

## 三、数据分析

### 3.1 原始数据结构

```
/wekafs/datongxu/tfi/
├── train/
│   ├── Black/          # 伪造图片 (800 张)
│   │   ├── Image/      # .jpg/.png 图片 (尺寸不一, 512x512 ~ 4961x7016)
│   │   ├── Mask/       # .png 二值 mask (0/255, 与图片同尺寸)
│   │   └── Caption/    # .md 中文分析文本 (260~1320 字)
│   └── White/          # 真实图片 (200 张)
│       ├── Image/      # .jpg 图片
│       └── Caption/    # .md 中文分析文本 (290~919 字)
├── test/
│   └── Image/          # 500 张测试图片 (无标签)
└── submit_example.csv  # 提交格式示例
```

### 3.2 数据集划分 (split_train_val.py)

使用固定随机种子 (seed=42) 按 8:2 比例分割：

| 类别 | 原始 | train_split (80%) | val (20%) |
|------|------|-------------------|-----------|
| Black (伪造) | 800 | 640 | 160 |
| White (真实) | 200 | 160 | 40 |
| 合计 | 1000 | 800 | 200 |

使用符号链接，不占用额外磁盘空间。Image/Caption/Mask 对应关系已验证无误。

### 3.3 关键数据特征

- **图像尺寸差异大**: 从 512x512 到 4961x7016，训练时统一 resize
- **Mask 格式**: 灰度图, 0=真实区域, 255=伪造区域; 与图片完全同尺寸
- **Caption 内容**: Black 类包含坐标 `[x1, y1, x2, y2]` 和详细视觉/逻辑分析; White 类描述真实性论证
- **RLE 格式**: COCO 标准 RLE, `{"size": [H, W], "counts": "..."}`, 真实图 counts 很短(全零)
- **submit_example.csv**: 500 条全部对应 test 目录, label 全为 0 (占位), 与训练集无交集

---

## 四、系统架构

### 4.1 总体流水线

```
训练阶段 (8x MI325X 全部可用):
  ┌─────────────────────────┐
  │ Step 1: 分割集成训练      │ ← 3架构 x 5折 = 15 个模型
  │ Step 2: 分类器训练        │ ← EfficientNet-V2-L x 5折
  │ Step 3: 397B 教师模型微调  │ ← LoRA r=128, 8 卡 DeepSpeed ZeRO-3
  │ Step 4: 教师生成增强数据   │ ← 多温度采样, 3 版本/图
  │ Step 5: 8B 学生模型微调    │ ← 全量微调, 原始+增强数据
  └─────────────────────────┘

推理阶段 (单卡 ≤ 48GB):
  测试图片
    → 阶段1: 分割集成 + TTA → mask → label + RLE        (~3GB 峰值)
    → 阶段1.5: 分类器投票 → 修正 label                    (~0.5GB 峰值)
    → 阶段2: Qwen3-VL-8B-Thinking → explanation          (~25GB 峰值)
    → 输出: submit.csv
```

### 4.2 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 分割输入 | RGB + ELA + SRM (7通道) | 利用图像取证领域经典特征, 暴露压缩/噪声不一致 |
| 分割集成 | 3架构 x 5折 = 15模型 | 架构多样性 + 数据多样性, 最大化集成增益 |
| 教师模型 | Qwen3.5-397B-A17B | 最新开源 VLM, 原生视觉早期融合, DeltaNet+MoE 混合架构 |
| 教师训练 | LoRA r=128 (非全量) | 全量需 800GB/卡放不下; LoRA r=128 差距 <1% |
| 学生模型 | Qwen3-VL-8B-Thinking 全量微调 | BF16 ~16GB, 远低于 48GB 限制; 全量微调最大吸收 |
| 损失函数 | Focal + Dice + Boundary | 三重损失: 不平衡 + 区域 + 边缘, 各司其职 |

---

## 五、模型清单

所有模型保存在 `/wekafs/datongxu/tfi/models/` 目录：

| 模型 | 来源 | 用途 | 本地路径 | 大小 |
|------|------|------|---------|------|
| SegFormer-B5 | nvidia/segformer-b5-finetuned-ade-640-640 | 分割 backbone | `models/segformer-b5/` | ~340 MB |
| ConvNeXt-V2-Large | timm (fcmae_ft_in22k_in1k_384) | 分割 backbone | `models/convnextv2_large.pth` | 749 MB |
| MaxViT-Large | timm (in21k_ft_in1k) | 分割 backbone | `models/maxvit_large.pth` | 806 MB |
| EfficientNet-V2-L | timm (in21k_ft_in1k) | 分类 backbone | `models/efficientnetv2_l.pth` | 450 MB |
| Qwen3-VL-8B-Thinking | Qwen/Qwen3-VL-8B-Thinking | 学生 VLM | `models/Qwen3-VL-8B-Thinking/` | ~16 GB |
| Qwen3.5-397B-A17B | Qwen/Qwen3.5-397B-A17B | 教师 VLM | `models/Qwen3.5-397B-A17B/` | ~800 GB |

### 5.1 Qwen3.5-397B-A17B 详细架构

| 属性 | 值 |
|------|---|
| 架构类型 | `qwen3_5_moe` (Gated DeltaNet + Gated Attention + MoE) |
| 总参数 | 397B (403B 含 embeddings) |
| 活跃参数/token | 17B |
| 层数 | 60 (15 blocks × 4 layers) |
| 层布局 | 每 block: 3× DeltaNet→MoE + 1× Attention→MoE |
| 专家数 | 512 个路由专家 + 1 个共享专家 |
| 每 token 激活专家 | 10 个路由 + 1 个共享 |
| 专家中间维度 | 1024 |
| 隐藏维度 | 4096 |
| 注意力头数 | 32 (Q), 2 (KV) — GQA |
| DeltaNet 头数 | 16 (QK), 64 (V) |
| 上下文长度 | 262,144 (原生) |
| 视觉编码器 | ViT-27层, patch_size=16, temporal_patch_size=2 |
| 视觉融合 | 早期融合 (early fusion on multimodal tokens) |

---

## 六、代码文件详解

### 6.1 `rocm_compat.py` — ROCm 兼容性修复 (新增)

**功能**: 修复 `torch._grouped_mm` 在 ROCm MI325X 上 CK grouped GEMM workspace 未分配的崩溃问题。
- `patch_grouped_mm()`: 将 `torch._grouped_mm` 替换为逐专家顺序矩阵乘法
- 必须在 transformers 模型代码导入之前调用
- 所有加载 Qwen3.5-397B 的脚本共享此修复

### 6.2 `utils.py` — 工具函数库 (345 行)

**图像取证特征提取:**
- `compute_ela(image, quality=90)`: Error Level Analysis. 将图片以指定质量 JPEG 重压缩, 计算原图与重压缩图的差值并放大。伪造区域因二次压缩而产生不同的误差级别, ELA 可暴露这种不一致。输出 (H,W,3) uint8。
- `compute_srm(image)`: Spatial Rich Model 噪声残差. 使用 8 个高通滤波核(1阶/2阶/3阶边缘检测器)提取图像噪声残差, 取各核最大响应并归一化。伪造区域的噪声统计特征与原始区域不一致。输出 (H,W,1) float32 [0,1]。

**COCO RLE 编解码:**
- `mask_to_rle(binary_mask)`: 将 0/1 二值 mask 编码为 COCO RLE 格式 `{"size":[H,W], "counts":"..."}`
- `rle_to_mask(rle)`: 解码 RLE 为 numpy 数组
- `create_zero_rle(H, W)`: 创建全零 mask 的 RLE (真实图片用)

**评估指标:**
- `compute_iou(pred, target)`: Intersection over Union
- `compute_dice(pred, target)`: Dice 系数
- `compute_f1(pred_labels, true_labels)`: 分类 F1/Precision/Recall/Accuracy
- `compute_pixel_metrics(pred, target)`: IoU + Dice + PixelAcc 汇总

**Mask 后处理:**
- `postprocess_mask(mask, morph_kernel_size=5, min_area=100)`: 形态学开运算(去噪) → 闭运算(填孔) → 连通域面积过滤

**辅助函数:**
- `mask_to_label(mask, threshold=0.001)`: 根据伪造像素占比判定 label
- `describe_mask_region(mask)`: 生成区域描述文本 ("图像上方左侧区域, 覆盖约X%的画面")

### 6.3 `dataset.py` — 数据集模块 (483 行)

**ForgerySegDataset** — 多流分割数据集:
- 输入: 7 通道 = RGB(3) + ELA(3) + SRM(1)
- 输出: 二值 mask (1,H,W) + label
- 训练增强: RandomResizedCrop, Flip, Rotate90, ShiftScaleRotate, ColorJitter, GaussNoise, CoarseDropout
- 验证: 仅 Resize 到目标尺寸

**ForgeryClsDataset** — 分类数据集:
- 输入: 6 通道 = RGB(3) + ELA(3)
- 输出: label (0/1)
- 增强: RandomResizedCrop, Flip, Rotate, ColorJitter, GaussNoise

**TestImageDataset** — 推理数据集:
- 输入: 7 通道, 无标签
- 返回: 多流特征 + 图片名 + 原始尺寸

**create_kfold_splits(data_dir, n_folds=5, seed=42)** — K-Fold 分割:
- 在 Black/White 内部分别分层分折, 保持类别比例
- 5 折时每折: train=640, val=160

**VLMSFTDataset** — VLM 微调数据集:
- 输入: (image_path, caption) 对
- 输出: 对话格式 (system + user + assistant)
- 支持加载教师增强的 JSONL 数据

### 6.4 `train_seg_ensemble.py` — 分割集成训练 (446 行)

**损失函数 (三重组合):**
- `FocalLoss(alpha=0.25, gamma=2.0)`: 解决前景/背景严重不平衡, 降低易分样本权重
- `DiceLoss(smooth=1.0)`: 直接优化区域重叠度, 对小目标友好
- `BoundaryLoss(kernel_size=3)`: 用 Laplacian 提取 mask 边界, 在边界区域加 5 倍 BCE 权重
- `CombinedLoss`: 0.4*Focal + 0.4*Dice + 0.2*Boundary

**三种分割架构:**
- **SegFormer-B5**: HuggingFace `nvidia/segformer-b5-finetuned-ade-640-640`, ~85M 参数. 层级 Transformer 编码器 + MLP 解码头. 修改第一层 patch embedding 从 3ch→7ch (新增通道权重初始化为 0, 不破坏预训练)
- **ConvNeXt-V2-Large + DeepLabV3+**: via SMP. ~235M 参数. 强局部纹理建模 + 多尺度空洞卷积
- **MaxViT-Large + FPN**: via SMP. ~212M 参数. Multi-axis attention + 特征金字塔, 兼顾全局和局部

**训练超参:**
- 输入: 768x768, batch_size=4, AdamW lr=6e-5, OneCycleLR (warmup 5%)
- 100 epochs, early stopping patience=15
- BF16 混合精度, 梯度裁剪 max_norm=1.0

**多卡并行训练命令:**
```bash
python train_seg_ensemble.py --arch segformer --fold all --gpu 0 &
python train_seg_ensemble.py --arch convnext --fold all --gpu 1 &
python train_seg_ensemble.py --arch maxvit --fold all --gpu 2 &
```

### 6.5 `train_classifier.py` — 分类器训练 (172 行)

**模型:** EfficientNet-V2-L (via timm, `tf_efficientnetv2_l.in21k_ft_in1k`)
- 修改 in_chans=6 (RGB+ELA)
- ImageNet-21K 预训练 → IN1K 微调
- 512x512 输入, batch_size=8

**训练策略:**
- 类别权重 CrossEntropyLoss: [1.0, 0.25] (补偿 Black:White=4:1 不平衡)
- AdamW lr=3e-4, OneCycleLR
- 50 epochs, early stopping patience=10
- 5-fold 交叉验证

### 6.6 `train_teacher.py` — 教师模型 LoRA 微调

**模型:** Qwen3.5-397B-A17B (详见 5.1 节)

**训练配置:**
- LoRA r=128, alpha=256, dropout=0.05
- target_modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 可训练参数: 198,574,080 (占总参数 0.05%)
- DeepSpeed ZeRO-3, 8 卡分布式
- `HfDeepSpeedConfig` 预初始化 — 模型参数加载时直接分片到 GPU, 避免 CPU OOM
- `gradient_checkpointing_kwargs={"use_reentrant": True}` — 避免 ZeRO-3 重计算元数据不匹配
- lr=1e-4, 3 epochs, cosine warmup 5%
- batch=1/卡, gradient_accumulation=4 → effective batch 32
- Gradient Checkpointing 开启
- ROCm `_grouped_mm` 兼容性修复 (via `rocm_compat`)

**实际显存占用 (每卡):**
- 冻结模型 BF16 分片: ~92GB
- LoRA + 优化器状态: ~1GB
- 激活值 (梯度检查点): ~15-20GB
- 总计: ~110GB / 256GB (利用率 43%)

**训练命令:**
```bash
deepspeed --num_gpus 8 train_teacher.py \
    --model_name models/Qwen3.5-397B-A17B \
    --deepspeed ds_config_z3.json \
    --epochs 3 --lr 1e-4 --lora_r 128
```

**VLM 数据整理器 (VLMDataCollator):**
- 将 (image_path, conversation) 转为模型可接受的 input_ids + pixel_values
- 应用 chat template, 处理视觉信息 (via `qwen_vl_utils`)
- 设置 labels (padding 部分设为 -100)

### 6.7 `merge_lora.py` — LoRA 权重合并 (新增)

将 LoRA 适配器合并到基座模型, 生成独立的完整模型, 用于推理/生成 (避免 peft/accelerate 版本兼容性问题)。
- 输入: base model + LoRA checkpoint
- 输出: 合并后完整模型 `models/Qwen3.5-397B-A17B-teacher/` (121 shards, 740GB)
- 耗时: ~15 分钟 (加载 5 min + 保存 10 min)

### 6.8 `generate_teacher_data.py` — 教师增强数据生成

**流程:**
1. 加载合并后的教师模型 (`device_map="auto"` 8 卡推理, ~134GB/卡)
2. 遍历 train 目录全部 1000 张图片
3. 每张图用 3 个不同 temperature (0.7, 0.9, 1.1) 生成 3 个版本的 caption
4. 保留 thinking 推理链 (不剥离 `<think>...</think>`, 用于 CoT 蒸馏)
5. 逐条写入 JSONL, 支持断点续传
6. 支持 `--slice K/N` 参数进行数据分片 (多实例并行)
7. ROCm `_grouped_mm` 兼容性修复 (via `rocm_compat`)

**生成命令:**
```bash
python generate_teacher_data.py \
    --model_path models/Qwen3.5-397B-A17B-teacher \
    --data_dir train \
    --output_dir augmented_data/train
```

### 6.9 `train_student_8b.py` — 学生模型训练 (113 行)

**模型:** Qwen3-VL-8B-Thinking **全量微调** (非 LoRA)

**训练数据:** 原始 Caption + 教师增强 Caption

**训练配置:**
- lr=2e-5, 5 epochs, cosine warmup
- weight_decay=0.01
- batch=1/卡, gradient_accumulation=8
- Gradient Checkpointing 开启
- BF16 混合精度

**多卡训练:**
```bash
accelerate launch --num_processes 4 train_student_8b.py
```

### 6.10 `inference.py` — 完整推理流水线 (380 行)

**阶段 1: 分割集成推理**
- 加载 15 个分割模型 (逐个加载, 推理后释放)
- 对每张图做 4 种 TTA: 原图 + 水平翻转 + 垂直翻转 + 旋转180
- 所有模型概率图取平均 → 二值化 (阈值 0.5)
- Resize 回原始尺寸 → 形态学后处理 → 连通域过滤
- Label 判定 + RLE 编码

**阶段 1.5: 分类器投票**
- 加载 5 个 EfficientNet 分类器 (逐个加载)
- 每张图计算 P(伪造) 的平均值
- 与分割 label 加权投票: `final = 0.6*seg_label + 0.4*cls_score > 0.5`

**阶段 2: VLM 解释生成**
- 加载 Qwen3-VL-8B-Thinking 学生模型
- 将分割结果 (label + 区域描述) 注入 prompt
- 伪造图: "经过像素级分析, 该图像被检测为伪造图像。伪造区域位于[位置描述]。请结合以上信息..."
- 真实图: "经过像素级分析, 该图像未检测到明显伪造痕迹。请结合以上信息..."
- temperature=0.3, max_new_tokens=1024
- 去除 thinking 标签

**推理命令:**
```bash
python inference.py --gpu 0 --use_tta --test_dir test/Image
```

### 6.11 `test_inference.py` — Qwen3.5 推理测试脚本

**功能**: 验证 Qwen3.5-397B-A17B 模型加载和推理是否正常。

**3 个测试:**
1. 纯文本推理 — 验证模型加载和生成
2. 图像理解 — 测试视觉能力
3. 伪造检测专业分析 — 使用项目 system prompt

**关键实现:**
- `device_map="auto"` 分布到 8 卡
- ROCm `_grouped_mm` 顺序回退
- 思维链 (thinking) 分离显示

**推理性能 (实测):**
| 模式 | 速度 | 说明 |
|------|------|------|
| 纯文本 | ~10 tok/s | 512 tokens / 51s |
| 图像理解 | ~2.9 tok/s | 512 tokens / 175s (含图像编码) |

### 6.12 配置文件

**`ds_config_z3.json`** — DeepSpeed ZeRO-3 配置:
- BF16 启用
- ZeRO Stage 3 (参数/梯度/优化器全分片)
- `train_micro_batch_size_per_gpu`: 1
- `gradient_accumulation_steps`: 4
- overlap_comm, contiguous_gradients 开启
- 16bit 权重收集保存
- 不使用 `"auto"` 值 (DeepSpeed 0.18.6 在 `zero.Init` 阶段无法解析 "auto")

**`requirements.txt`** — 依赖清单

### 6.13 `split_train_val.py` — 数据集划分 (251 行)

- 固定种子 42, 8:2 分割
- 符号链接方式, 不复制文件
- 自动验证: 无重叠、无遗漏、文件对应关系正确

---

## 七、性能提升技术汇总

| 技术 | 预期提升 | 适用任务 | 实现位置 |
|------|---------|---------|---------|
| 多流输入 RGB+ELA+SRM (7ch) | +3~5% IoU | 分割 | dataset.py |
| 三架构集成 (SegFormer+ConvNeXt+MaxViT) | +2~4% IoU | 分割 | train_seg_ensemble.py |
| 5-Fold 交叉验证 (15模型) | +1~2% IoU | 分割+分类 | train_seg_ensemble.py |
| TTA 4x (原图+翻转+旋转) | +1~2% IoU | 分割 | inference.py |
| 高分辨率 768x768 | +2~3% IoU | 分割 | train_seg_ensemble.py |
| 三重损失 Focal+Dice+Boundary | +1~2% IoU | 分割 | train_seg_ensemble.py |
| 后处理 (形态学+连通域) | +0.5~1% IoU | 分割 | inference.py + utils.py |
| 397B 教师蒸馏 → 8B 学生 | +8~15% 生成质量 | 解释 | train_teacher.py |
| 学生全量微调 (非 LoRA) | +2~5% | 解释 | train_student_8b.py |
| Thinking CoT 推理 | +3~5% 分析深度 | 解释 | 模型原生能力 |
| 信息融合 (seg 结果→VLM prompt) | +2~3% 一致性 | 解释 | inference.py |
| 独立分类器投票 | +1~2% Acc | 分类 | train_classifier.py |

---

## 八、推理显存预算

顺序加载策略 (不同阶段不同时驻留):

| 阶段 | 峰值显存 |
|------|---------|
| 分割集成 (逐模型加载) | ~1-2 GB |
| 分类器 (逐模型加载) | ~0.5 GB |
| VLM 8B BF16 + KV Cache + 生成 | ~25 GB |
| **总峰值** | **~25 GB << 48 GB** |

---

## 九、执行步骤与命令

### Step 1: 分割集成训练 (3 架构并行, 各占 1 卡)
```bash
cd /wekafs/datongxu/tfi

# 三种架构分别在 GPU 0/1/2 上训练 5 折
python train_seg_ensemble.py --arch segformer --fold all --gpu 0 &
python train_seg_ensemble.py --arch convnext --fold all --gpu 1 &
python train_seg_ensemble.py --arch maxvit --fold all --gpu 2 &

# 等待所有完成
wait
```

### Step 2: 分类器训练 (与 Step 1 并行)
```bash
python train_classifier.py --fold all --gpu 3 &
```

### Step 3: 教师模型 LoRA 微调 (8 卡)
```bash
deepspeed --num_gpus 8 train_teacher.py \
    --model_name models/Qwen3.5-397B-A17B \
    --deepspeed ds_config_z3.json \
    --epochs 3 --lr 1e-4 --lora_r 128
```

### Step 4: 教师生成增强数据
```bash
python generate_teacher_data.py \
    --base_model models/Qwen3.5-397B-A17B \
    --lora_path checkpoints/teacher \
    --data_dir train_split \
    --output_dir augmented_data/train

python generate_teacher_data.py \
    --base_model models/Qwen3.5-397B-A17B \
    --lora_path checkpoints/teacher \
    --data_dir val \
    --output_dir augmented_data/val
```

### Step 5: 学生模型全量微调
```bash
accelerate launch --num_processes 4 train_student_8b.py \
    --model_name models/Qwen3-VL-8B-Thinking \
    --augmented_dir augmented_data \
    --epochs 5 --lr 2e-5
```

### Step 6: 推理生成提交文件
```bash
python inference.py \
    --gpu 0 \
    --test_dir test/Image \
    --vlm_model checkpoints/student_8b \
    --output submit.csv \
    --use_tta
```

---

## 十、已完成步骤 (详细记录)

### [已完成] 10.1 数据集划分 — `split_train_val.py`

**时间**: 2026-02-13

**操作**: 将 `/wekafs/datongxu/tfi/train/` 按 8:2 分割为 `train_split/` 和 `val/`

**具体结果**:
- Black 类: 800 → train_split 640 + val 160
- White 类: 200 → train_split 160 + val 40
- 使用符号链接, 零额外磁盘占用
- 随机种子: 42 (可复现)

**验证通过**:
- train 和 val 之间无重叠、无遗漏
- Image/Caption/Mask 文件对应关系完全正确
- submit_example.csv 中 500 张图全属于 test 目录, 与训练集无交集

### [已完成] 10.2 依赖安装与升级

**初始安装** (2026-02-13):
- pycocotools, albumentations, opencv-python-headless, timm, segmentation-models-pytorch, qwen-vl-utils

**Qwen3.5 适配升级** (2026-02-17):
- transformers: 4.55.0 → **5.2.0** (支持 `qwen3_5_moe` 架构)
- 新安装: deepspeed 0.18.6
- 重新安装 (升级 transformers 时丢失): qwen-vl-utils, opencv-python-headless, albumentations, pycocotools

**当前关键包版本**:
- torch 2.10.0.dev20251112+rocm7.1
- transformers 5.2.0
- peft 0.18.0
- deepspeed 0.18.6
- accelerate 1.9.0

### [已完成] 10.3 `utils.py` 编写与测试

**功能验证结果** (实际数据测试):
```
Image shape: (512, 512, 3)
ELA shape: (512, 512, 3), dtype: uint8, range: [0, 255]
SRM shape: (512, 512, 1), dtype: float32, range: [0.0000, 1.0000]
Mask shape: (512, 512), sum: 992
RLE size: [512, 512], counts_len: 121
Decoded match: True  ← RLE 编解码完全一致
Zero RLE: size=[768, 1024], counts=PPPh0
Postprocessed mask sum: 983 (original: 992)  ← 后处理去除 9 个噪点像素
```

### [已完成] 10.4 `dataset.py` 编写与测试

**功能验证结果**:
```
Seg Dataset: 800 samples
  pixel_values: torch.Size([7, 768, 768])  ← 7通道输入 (RGB+ELA+SRM)
  mask: torch.Size([1, 768, 768])
  label: 1

Cls Dataset: 800 samples
  pixel_values: torch.Size([6, 512, 512])  ← 6通道输入 (RGB+ELA)

K-Fold splits:
  Fold 0: train=640, val=160
  ...
  Fold 4: train=640, val=160

Test Dataset: 500 samples
  pixel_values: torch.Size([7, 768, 768])
  image_name: 001858037f7846a79c619fda3d915e75.jpg
  orig_size: (7016, 4961)
```

### [已完成] 10.5 `train_seg_ensemble.py` 编写与模型验证

**三种分割架构构建验证**:
```
SegFormer-B5:
  Input: (1, 7, 768, 768) → Output: (1, 1, 768, 768) ← 通过

ConvNeXt-V2-L + DeepLabV3+:
  eval batch=1: (1, 1, 768, 768) ← 通过
  train batch=4: (4, 1, 768, 768) ← 通过

MaxViT-Large + FPN:
  eval batch=2: (2, 1, 768, 768) ← 通过
```

**注意**: ConvNeXt+DeepLabV3+ 的 ASPP 模块含 BatchNorm, train 模式下 batch_size 必须 > 1。eval 模式无此限制。

### [已完成] 10.6 `train_classifier.py` 编写

- EfficientNet-V2-L via timm, 语法验证通过
- 支持 5-fold 交叉验证

### [已完成] 10.7 VLM 训练脚本编写

- `train_teacher.py`: 教师模型 LoRA 训练, 语法验证通过
- `generate_teacher_data.py`: 增强数据生成, 语法验证通过
- `train_student_8b.py`: 学生模型全量微调, 语法验证通过
- `rocm_compat.py`: ROCm 兼容性修复模块

### [已完成] 10.8 `inference.py` 编写

- 完整推理流水线, 语法验证通过
- 支持 TTA + 分类器投票 + VLM 生成
- 输出 submit.csv

### [已完成] 10.9 配置文件

- `ds_config_z3.json`: DeepSpeed ZeRO-3 配置 (已适配 DeepSpeed 0.18.6, 移除 "auto" 值)
- `requirements.txt`: 依赖清单

### [已完成] 10.10 模型下载

**已下载到本地 `models/` 目录**:
- `models/segformer-b5/` — SegFormer-B5 HuggingFace 模型 (完整)
- `models/convnextv2_large.pth` — ConvNeXt-V2-Large 权重 (749.4 MB)
- `models/maxvit_large.pth` — MaxViT-Large 权重 (806.1 MB)
- `models/efficientnetv2_l.pth` — EfficientNet-V2-L 权重 (449.7 MB)
- `models/Qwen3-VL-8B-Thinking/` — 已完成 (4 个 safetensors 分片)
- `models/Qwen3.5-397B-A17B/` — 已完成 (94 个 safetensors 分片, ~800GB)
- `models/Qwen3.5-397B-A17B-teacher/` — LoRA 合并后完整模型 (121 个 safetensors 分片, 740GB)

### [已完成] 10.11 分割集成训练 — 2026-02-13 启动

**启动命令** (3 卡并行):
```bash
python train_seg_ensemble.py --arch segformer --fold all --gpu 0 > logs/seg_segformer.log 2>&1 &
python train_seg_ensemble.py --arch convnext --fold all --gpu 1 > logs/seg_convnext.log 2>&1 &
python train_seg_ensemble.py --arch maxvit --fold all --gpu 2 > logs/seg_maxvit.log 2>&1 &
```

**最后已知进度** (截至 2026-02-13 14:51 UTC, fold0 训练中):

| 模型 | GPU | Epoch | 最佳 IoU | 最佳 Dice | 分类 Acc | 每 epoch 耗时 |
|------|-----|-------|---------|----------|---------|-------------|
| SegFormer-B5 | 0 | 13/100 | 0.5648 | 0.6312 | 0.8125 | ~41s |
| ConvNeXt-V2-L | 1 | 7/100 | 0.4220 | 0.5007 | 0.7688 | ~73s |
| MaxViT-L | 2 | 11/100 | 0.4798 | 0.5547 | 0.7750 | ~41s |

### [已完成] 10.12 分类器训练 — 2026-02-13 启动

**最后已知进度** (截至 2026-02-13 14:51 UTC):
- fold0, fold1 已完成; fold2 训练中 (epoch 23/50, F1=0.9160)

### [已完成] 10.13 Qwen3.5-397B-A17B 推理验证 — 2026-02-17

**测试脚本**: `test_inference.py`

**测试结果**:
```
Model loaded in 314.1s
Device map: {0, 1, 2, 3, 4, 5, 6, 7}

[Test 1] 纯文本: "我是通义千问..." — 10.0 tok/s ✓
[Test 2] 图像理解: 正确识别收据内容, 发现数学不一致 — 2.9 tok/s ✓
[Test 3] 伪造检测: 专业分析收据的数字异常 — 9.7 tok/s ✓
```

**遇到并解决的问题**:
1. `torch._grouped_mm` CK workspace 崩溃 → `rocm_compat.py` 顺序回退
2. `torch_dtype` 弃用 → 改为 `dtype`

### [已完成] 10.14 教师模型 LoRA 微调 — 共训练两轮

**第一轮** (2026-02-17, train_split 800 样本):
```bash
deepspeed --num_gpus 8 train_teacher.py \
    --model_name models/Qwen3.5-397B-A17B \
    --deepspeed ds_config_z3.json \
    --epochs 3 --lr 1e-4 --lora_r 128
```

**第二轮** (2026-02-18, 全量 train 1000 样本):
```bash
deepspeed --num_gpus 8 train_teacher.py \
    --model_name models/Qwen3.5-397B-A17B \
    --data_dir train \
    --deepspeed ds_config_z3.json \
    --epochs 3 --lr 1e-4 --lora_r 128
```

**训练结果**:
```
trainable params: 198,574,080 || all params: 397,000,934,896 || trainable%: 0.0500
```
- LoRA 权重: `checkpoints/teacher/adapter_model.safetensors` (379 MB)
- Epoch checkpoints: `checkpoint-64/`, `checkpoint-96/`
- 训练时长: ~5-6 小时/轮
- 每卡显存: ~110GB / 256GB (43%)

**训练过程中解决的问题**:
1. DeepSpeed 0.18.6 不支持 `"auto"` batch size → 显式设置 micro_batch=1, grad_accum=4
2. PEFT 在 ZeRO-3 下误将 Conv3d 当作 LoRA 目标 → 改用显式 target_modules 列表
3. 梯度检查点重计算元数据不匹配 → `use_reentrant=True`
4. `model.save_pretrained()` 在 ZeRO-3 下保存空权重 → 改用 `trainer.save_model()`

### [已完成] 10.15 LoRA 合并 — 2026-02-18

使用 `merge_lora.py` 将 LoRA 权重合并到基座模型:
```bash
python merge_lora.py
```
- 输出: `models/Qwen3.5-397B-A17B-teacher/` (121 shards, 740GB)
- 耗时: ~15 分钟
- 合并后可直接用 `device_map="auto"` 加载, 无需 peft 依赖

**合并原因**: peft + accelerate 版本兼容性问题导致动态 LoRA 加载报错 (`unhashable type: 'set'`), 合并后完全绕过 peft。

### [生成中] 10.16 增强数据生成 — 2026-02-18 启动

**命令** (使用合并后模型, 8 卡 device_map=auto):
```bash
python generate_teacher_data.py \
    --model_path models/Qwen3.5-397B-A17B-teacher \
    --data_dir train \
    --output_dir augmented_data/train
```

**生成参数**:
| 参数 | 值 |
|------|---|
| 输入图片 | 1000 张 (800 Black + 200 White) |
| 每张图生成 | 3 个版本 (temperature 0.7, 0.9, 1.1) |
| 总输出 | 3000 条 caption (含 thinking 推理链) |
| max_new_tokens | 2048 |
| 输出格式 | JSONL, 逐条写入, 支持断点续传 |

**资源占用**:
- GPU 显存: 每卡 ~134GB / 256GB (51%)
- CPU RAM: ~85GB / 3TB (3%)
- GPU 利用率: ~15% (pipeline 并行固有限制)

**速度**: ~170s/图 (pipeline 并行 8 卡), 预计总时间 ~47 小时

**输出文件**: `augmented_data/train/augmented_captions.jsonl`

### [待执行] 10.17 学生模型微调

**前置条件**: 10.15 增强数据生成完成

**命令**:
```bash
accelerate launch --num_processes 4 train_student_8b.py \
    --model_name models/Qwen3-VL-8B-Thinking \
    --augmented_dir augmented_data \
    --epochs 5 --lr 2e-5
```

### [待执行] 10.18 推理与提交

**前置条件**: 10.11 + 10.12 + 10.17 全部完成

**命令**:
```bash
python inference.py --gpu 0 --use_tta --test_dir test/Image --output submit.csv
```

---

## 十一、Bug 修复历史

### 2026-02-17: 教师模型训练 Bug 修复 (从 Qwen3 235B → Qwen3.5 397B)

| # | Bug | 原因 | 修复 |
|---|-----|------|------|
| 1 | CPU 内存不断增长直到死机 | `from_pretrained` 用了 `dtype=` (参数名错误, 被忽略) → FP32 加载 940GB | 改为 `torch_dtype=` → 后又改为 `dtype=` (transformers 5.2.0) |
| 2 | CPU OOM (8 进程各自加载完整模型) | 缺少 `HfDeepSpeedConfig` 预初始化, ZeRO-3 未在加载时分片 | 在 `from_pretrained` 前调用 `HfDeepSpeedConfig(ds_config)` |
| 3 | `torch._grouped_mm` 崩溃 | ROCm CK grouped GEMM workspace 未分配 | `rocm_compat.py` — 逐专家顺序矩阵乘法回退 |
| 4 | `fla` 导入链断裂 | 删除 `torch._grouped_mm` 导致 `torch._inductor` 导入失败 | 改为替换 (非删除) `_grouped_mm` 函数 |
| 5 | `torch_dtype` 弃用警告 | transformers 5.2.0 改用 `dtype` 参数 | 所有脚本统一改为 `dtype=torch.bfloat16` |
| 6 | DeepSpeed batch size 验证失败 | `"auto"` 字符串无法与 int 比较 (DeepSpeed 0.18.6) | `ds_config_z3.json` 中使用显式数值, 移除 `train_batch_size` |
| 7 | PEFT Conv3d 崩溃 | ZeRO-3 压扁权重为 1D, PEFT 误判模块类型 | 改用显式 target_modules 列表 (不用自动检测) |
| 8 | 梯度检查点重计算 metadata 不匹配 | ZeRO-3 反向重计算时参数未 gather, shape 变为 [0] | `gradient_checkpointing_kwargs={"use_reentrant": True}` |
| 9 | LoRA 保存为空权重 (shape=[0]) | ZeRO-3 下 `model.save_pretrained()` 保存分片空参数 | 改用 `trainer.save_model()`, 从 `checkpoint-96/` 恢复正确权重 |
| 10 | `_grouped_mm` 缺少 `bias` 参数 | PyTorch 2.11.0 新增 `bias` 关键字参数 | `rocm_compat.py` 回退函数添加 `bias=None, **kwargs` |
| 11 | `causal_conv1d` ABI 不兼容 | PyTorch 从 2.10.0 升级到 2.11.0, 预编译 .so 的 C10 HIP 符号变化 | 终端手动 `pip install causal-conv1d --no-build-isolation --force-reinstall` |
| 12 | PEFT `from_pretrained` 崩溃 | peft/accelerate 版本与新 transformers 不兼容 (`unhashable type: 'set'`) | 使用 `merge_lora.py` 预合并, 绕过 peft 动态加载 |

---

## 十二、快速恢复指南 (如果聊天窗口关闭)

### 检查训练状态
```bash
cd /wekafs/datongxu/tfi

# 查看分割训练进度
tail -5 logs/seg_segformer.log
tail -5 logs/seg_convnext.log
tail -5 logs/seg_maxvit.log

# 查看分类器训练进度
tail -5 logs/cls_efficientnet.log

# 查看已保存的 checkpoint
ls checkpoints/seg/*/best_model.pt 2>/dev/null
ls checkpoints/cls/*/best_model.pt 2>/dev/null
ls checkpoints/teacher/ 2>/dev/null

# 查看 GPU 使用情况
amd-smi
```

### 判断当前应该执行什么
1. **如果增强数据生成仍在进行** (`augmented_data/train/augmented_captions.jsonl` 行数 < 3000): 等待, 用 `wc -l augmented_data/train/augmented_captions.jsonl` 查看进度
2. **如果增强数据已生成 (3000 条)**: 执行 Step 10.17 学生模型微调
3. **如果学生模型已完成 (`checkpoints/student_8b/` 存在)**: 执行 Step 10.18 推理生成 submit.csv
4. **如果 `submit.csv` 已生成**: 项目完成

### 恢复增强数据生成 (如果中断)
```bash
cd /wekafs/datongxu/tfi
python generate_teacher_data.py \
    --model_path models/Qwen3.5-397B-A17B-teacher \
    --data_dir train \
    --output_dir augmented_data/train
# 断点续传: 自动跳过已生成的条目
```

### 关键文件
- 本文档: `PROJECT.md` — 完整项目说明和进度记录
- 所有训练脚本: `train_*.py` — 可直接运行
- ROCm 修复: `rocm_compat.py` — Qwen3.5 模型必须导入
- 推理测试: `test_inference.py` — 验证模型加载
- 推理脚本: `inference.py` — 最终输出 submit.csv
- 日志目录: `logs/` — 训练日志

---

## 十三、项目文件总览

```
/wekafs/datongxu/tfi/
├── PROJECT.md                   # 本文档
├── requirements.txt             # 依赖清单
├── ds_config_z3.json            # DeepSpeed ZeRO-3 配置
├── rocm_compat.py               # ROCm 兼容性修复 (grouped_mm)
│
├── split_train_val.py           # 数据集划分脚本
├── utils.py                     # 工具函数 (ELA/SRM/RLE/指标/后处理)
├── dataset.py                   # 数据集 (分割/分类/VLM SFT)
├── train_seg_ensemble.py        # 分割集成训练
├── train_classifier.py          # 分类器训练
├── train_teacher.py             # 教师模型 LoRA 微调 (Qwen3.5-397B)
├── merge_lora.py                # LoRA 权重合并到基座模型
├── generate_teacher_data.py     # 教师增强数据生成 (含 thinking)
├── train_student_8b.py          # 8B 学生全量微调
├── inference.py                 # 完整推理流水线
├── test_inference.py            # Qwen3.5 推理验证脚本 (支持 LoRA)
│
├── models/                      # 预训练模型权重
│   ├── segformer-b5/
│   ├── convnextv2_large.pth
│   ├── maxvit_large.pth
│   ├── efficientnetv2_l.pth
│   ├── Qwen3-VL-8B-Thinking/
│   ├── Qwen3.5-397B-A17B/      # 基座模型 (94 shards, ~800GB)
│   └── Qwen3.5-397B-A17B-teacher/  # LoRA 合并后 (121 shards, 740GB)
│
├── train_split/                 # 训练集 (符号链接)
│   ├── Black/ (640)
│   └── White/ (160)
├── val/                         # 验证集 (符号链接)
│   ├── Black/ (160)
│   └── White/ (40)
├── test/                        # 测试集
│   └── Image/ (500)
│
├── checkpoints/                 # 训练产出
│   ├── seg/                     # 15 个分割模型
│   │   ├── segformer_fold{0-4}/
│   │   ├── convnext_fold{0-4}/
│   │   └── maxvit_fold{0-4}/
│   ├── cls/                     # 5 个分类器
│   │   └── efficientnet_fold{0-4}/
│   ├── teacher/                 # 教师 LoRA 权重 (~380MB, 已完成)
│   │   ├── adapter_model.safetensors
│   │   ├── checkpoint-64/
│   │   └── checkpoint-96/
│   └── student_8b/              # 学生完整模型 (待生成)
│
├── logs/                        # 训练日志
│   ├── seg_segformer.log
│   ├── seg_convnext.log
│   ├── seg_maxvit.log
│   └── cls_efficientnet.log
│
├── augmented_data/              # 教师增强数据
│   └── train/
│       └── augmented_captions.jsonl  # 生成中, 目标 3000 条
│
├── submit_example.csv           # 提交格式示例
└── submit.csv                   # 最终提交文件 (待生成)
```
