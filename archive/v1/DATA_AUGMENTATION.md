# 数据增强与训练数据视图（v2 实施版）

> 本文档描述当前仓库里已经实现并正在使用的数据增强流程、质量控制逻辑、训练接入方式，以及 `train -> train_resume` 的路径切换约定。

---

## 1. 原始数据与 `train_resume`

### 1.1 原始数据规模

- `train/Black/Image` + `Mask` + `Caption`: 800 条伪造样本
- `train/White/Image` + `Caption`: 200 条真实样本
- 原始基础集总规模: 1000 条，类别比约为 `Black:White = 4:1`

### 1.2 为什么不再直接用 `train`

当前存储环境下，`train/Black/Image` 目录的**枚举操作**会偶发卡死或超时，典型表现是：

- `os.listdir("train/Black/Image")` 可能长时间不返回
- 单个文件路径仍然可读，说明不是图片内容损坏
- `train_split`、`val` 以及它们指向的原始文件访问正常

这更像是 **NFS / 目录元数据层面的问题**，不是训练代码逻辑错误，也不是样本本身坏掉。

### 1.3 `train_resume` 的设计

为绕开目录枚举问题，训练统一切换到 `train_resume/`。这个目录不是重新复制一份数据，而是一个**稳定可枚举的训练视图**：

```text
train_resume/
├── Black/
│   ├── Image/          # 由 train_split + val 拼成的 800 条 Black 视图
│   ├── Mask/           # 对应 800 条 GT mask
│   ├── Caption -> train/Black/Caption
│   └── Caption_clean -> train/Black/Caption_clean
└── White/
    ├── Image/          # 由 train_split + val 拼成的 200 条 White 视图
    └── Caption -> train/White/Caption
```

关键点：

- `train_resume/Black/Image` 和 `train_resume/White/Image` 里的条目仍然指向原始 `train` 文件，没有重复存图
- `seg` / `cls` / `VLM` 三类训练脚本现在都可以统一读取 `train_resume`
- `train_v2` 的 caption 重生仍然默认读 `train_split`，因为它只需要训练子集里的 Black 样本，并且这样更利于和验证集隔离

---

## 2. 增强总览

当前增强不是“单一路径”，而是四条并行的数据流：

| 数据流 | 作用对象 | 主要产物 | 进入哪些模型 |
|---|---|---|---|
| A. `Caption_clean` | 清洗原始 GT caption | `Caption_clean/*.md` | VLM |
| B. `synth` | 像素级合成伪造样本 | `augmented_data/synth/` | seg / cls |
| C. `real_ext` | 真实负例扩充 | `augmented_data/real_ext/` | seg / cls / VLM |
| D. `train_v2` | 基于 GT evidence 的 caption 重生 | `augmented_data/train_v2/*.jsonl` | VLM |

设计原则：

- **合成伪造样本只喂给像素模型**，不直接喂给 VLM
- **caption 重生只增强文本监督信号**，不改变像素标签
- **真实图扩充同时服务负例补齐和类平衡**
- **所有增强都要经过自动质检**，不允许未校验数据直接进入训练

---

## 3. A. Caption 清洗与 bbox 对齐

### 3.1 目标

原始 Black caption 存在两个核心问题：

- 文本里的 `[x1,y1,x2,y2]` 坐标与 GT mask 不完全一致
- 部分样本残留 `<think>` / `</think>` 等不该进入 SFT 的标记

如果直接把这些 caption 喂给 Qwen3.5-9B，会让模型学到：

- 错误的区域坐标
- 与推理时 evidence prompt 冲突的证据引用方式
- 非目标格式的输出模式

### 3.2 一致性审计：`tools/check_caption_bbox.py`

输入目录默认是：

- `train_split`
- `val`
- `train_resume`

核心逻辑：

1. 用正则 `BBOX_RE` 从 caption 中抽取所有 `[x1,y1,x2,y2]`
2. 读取 GT mask，调用 `evidence.extract_regions(mask, min_area_px=64)` 抽 GT 区域
3. 计算 caption bbox 与 GT bbox 的最大 IoU
4. 记录以下字段到 `logs/caption_bbox_audit.csv`

主要输出字段：

- `split`
- `stem`
- `n_cap_bbox`
- `n_gt_bbox`
- `gt_area_ratio`
- `max_iou`
- `has_think_tag`
- `cap_len`
- `cap_bboxes`
- `gt_bboxes`

分桶逻辑：

- `max_iou >= 0.5`: 可直接保留
- `0.2 <= max_iou < 0.5`: 进入 bbox 重写
- `max_iou < 0.2` 或 caption 内没有 bbox: 标记为后续重生对象

### 3.3 清洗与重写：`tools/clean_captions.py`

该脚本读取 `logs/caption_bbox_audit.csv`，按上面的分桶规则处理原始 caption：

1. `sanitize()`
   - 去掉 `<think>` / `</think>`
   - 压缩过多空行
   - 保留原始中文内容主体
2. `rewrite_bboxes()`
   - 对 caption 中每个 bbox，寻找 IoU 最高的 GT bbox
   - 使用 GT bbox 替换原字符串
   - 同时兼容半角逗号、全角逗号、空格差异
3. 对低质量样本：
   - 先写入清洗后的原文作为占位
   - 同时把 `split	stem` 写入 `logs/needs_regen.txt`
   - 留给阶段 D 做 evidence-aware 重生

产物：

- `train/Black/Caption_clean/*.md`
- `train_split/Black/Caption_clean/*.md`
- `val/Black/Caption_clean/*.md`
- `logs/needs_regen.txt`

当前已验证数量：

- `train_split/Black/Caption_clean`: 640
- `val/Black/Caption_clean`: 160
- 原始全量 `train/Black/Caption_clean`: 800

### 3.4 为什么这一步必须在 `train_v2` 前完成

因为阶段 D 的 caption 重生不是从零开始“瞎写”，而是围绕：

- GT mask 导出的证据
- 已清洗的 caption 习惯用语
- 统一的开头/结尾模板

也就是说，`Caption_clean` 是后续 VLM 数据增强的**基础层**，不是可选项。

---

## 4. B. 像素级合成伪造样本 `synth`

### 4.1 总原则

`synth` 只服务于 `train_seg_ensemble.py` 和 `train_classifier.py`。

原因很直接：

- 像素模型需要 `(image, mask, label)`，天然适合吃合成样本
- VLM 需要高质量中文鉴定文本，而合成图没有可信 caption
- 把低质量 caption 强行喂给 VLM，收益通常小于风险

### 4.2 样本来源

脚本：`tools/synth_forgery.py`

默认真实图池：`train/White/Image`

这里故意只从 White 图池采样，避免“把伪造贴到伪造上”，破坏标签语义。

### 4.3 三种实际实现的合成范式

#### 4.3.1 Copy-Move

函数：`copy_move(img_rgb, rng)`

实现细节：

- 从同一张图随机采一个矩形 patch
- patch 采样范围由 `pick_random_rect()` 控制：
  - `min_side=40`
  - `max_side_ratio=0.3`
- 在同图中寻找一个尽量不与原 patch 重叠的目标区域
- 随机做一次水平翻转，增加扰动
- 优先用 `cv2.seamlessClone(..., cv2.NORMAL_CLONE)` 做融合
- 如果 OpenCV clone 失败，则回退到直接覆盖
- GT mask 直接由目标框区域生成

#### 4.3.2 Splicing

函数：`splicing(img_a_rgb, img_b_rgb, rng)`

实现细节：

- 在目标图 `B` 上随机采一个目标框
- 从来源图 `A` 上裁一块等尺寸 patch
- 若 `A` 太小，则先 resize 到足够大
- 优先用 `cv2.seamlessClone(..., cv2.MIXED_CLONE)` 融到 `B`
- 回退逻辑仍是直接覆盖
- GT mask 仍由目标框直接生成

#### 4.3.3 Text-Replace-like

函数：`text_replace_like(img_rgb, rng)`

实现思路：

- 不依赖 OCR，而是先用 `cv2.Canny(60, 180)` 找边缘
- 再用 `cv2.boxFilter(..., (32, 32))` 找“边缘密集区域”
- 在这些区域里抽一个候选框，模拟“文字被改动”的位置
- 对局部 patch 施加：
  - `GaussianBlur(sigma=0.8~2.0)`
  - 亮度倍率扰动 `0.7~1.3`
  - 水平小位移 `[-4, 4]`
- 如果找不到足够稳定的边缘密集区，则回退到 copy-move

这一步的目的不是精确恢复某类 OCR 篡改，而是制造局部纹理、边缘、压缩痕迹不一致的 hard negative / hard positive 模式。

### 4.4 产物格式

目录结构：

```text
augmented_data/synth/
├── Image/*.jpg
├── Mask/*.png
└── meta.jsonl
```

`meta.jsonl` 记录：

- `stem`
- `type` (`copy_move` / `splicing` / `text_replace`)
- `source_a`
- `source_b`（仅 splicing）

### 4.5 反向过滤：`tools/filter_synth_by_seg.py`

合成并不等于可用。脚本会用**已有分割 checkpoint**对 `synth` 做“反向可解性筛选”。

关键流程：

1. 读取 `augmented_data/synth/Image` 与 `Mask`
2. 为每张图构造分割模型输入：
   - RGB
   - ELA (`compute_ela`)
   - SRM (`compute_srm`)
   - 合并成 7 通道输入
3. resize 到 `img_size=768`
4. 用已有 `segformer / maxvit / convnext` checkpoint 推理
5. `sigmoid -> threshold(0.3) -> postprocess_mask(kernel=5, min_area=64)`
6. 与合成 GT mask 计算 IoU，并对多模型取均值

筛选阈值：

- `mean_iou < 0.10`: drop，说明样本差到连现有模型都无法稳定感知
- `mean_iou > 0.90`: drop，说明样本过于容易，对泛化帮助有限
- `0.10 <= mean_iou <= 0.90`: keep，属于“hard but solvable”

输出文件：

- `augmented_data/synth/keep.txt`
- `augmented_data/synth/dropped.csv`
- `augmented_data/synth/meta.jsonl`（补充 `keep` / `filter_iou` / `drop_reason`）

当前质检结果：

- `synth/Image`: 750
- `synth/Mask`: 750
- `keep.txt`: 62
- 当前保留率约 `8.3%`

这个保留率不高，但它是有意为之：我们只保留真正对现有模型有区分价值的合成样本。

---

## 5. C. 真实图扩充 `real_ext`

### 5.1 目标

White 基础样本只有 200 条，远小于 Black 的 800 条。这个失衡会同时影响：

- seg 的前景/背景判定
- cls 的真假分类先验
- VLM 对“真实图”论证模板的学习密度

因此 `real_ext` 的目标是补齐真实负例，并提供更丰富的视觉分布。

### 5.2 脚本与目录

脚本：`tools/expand_real_images.py`

输出：

```text
augmented_data/real_ext/
├── Image/*.jpg
├── Caption/*.md
└── source.tsv
```

`source.tsv` 记录每个样本的来源：

- `white_aug`
- `coco`
- 以及父样本 stem / 文件名

### 5.3 White 强增广

函数：`white_strong_aug(img_np, seed)`

每张 White 原图默认产生 `n_per_image=3` 个扩增样本。增强链为：

1. JPEG 重压缩
   - 质量从 `{65, 75, 85}` 中随机选一个
2. 轻度调色
   - `Brightness`: `0.85~1.15`
   - `Contrast`: `0.85~1.15`
   - `Color`: `0.85~1.15`
3. 小幅 resize 往返
   - 缩放比 `0.85~1.15`
   - resize 到新尺寸后再 resize 回原尺寸

注意：这里的增强刻意保持“仍然真实”，不会引入明显篡改区域，因此：

- 没有 mask
- caption 直接复用原 White caption
- 这些样本可安全作为负例补齐

### 5.4 COCO 真实图采样

函数：`do_coco(root, n_coco, out_dir, source_rows)`

实现逻辑：

- 读取 `cache/coco/val2017.zip`
- 只要 zip 文件大于约 `500MB` 就认为可用
- 在 `val2017/` 下随机抽取 `n_coco=500` 张 JPG
- 不做复杂 caption 生成，而是用中文真实性模板：
  - 从 `DEFAULT_SCENES` 中随机取场景名
  - 填入 `REAL_CAPTION_TMPL`
  - 形成完整中文真实图鉴定文本

这样做的目的不是追求描述细节，而是给 VLM 提供**稳定的“真实图论证模板”**，并给 seg/cls 提供更多负例视觉分布。

### 5.5 当前状态

最新质检结果：

- `real_ext/Image`: 1100
- `real_ext/Caption`: 1100
- 抽检 20 张坏图: 0
- 图文数量对齐: OK

---

## 6. D. Evidence-aware caption 重生成 `train_v2`

### 6.1 目标

`train_v2` 不是普通的数据扩写，而是围绕 GT mask 生成**证据对齐**的额外 caption，用来增强 Qwen3.5-9B 的监督信号。

目标约束：

- 文本中引用的 bbox 必须来自 GT evidence
- 不能残留 `<think>`
- 不能引入证据外的坐标幻觉
- 尽量保持与推理时 prompt 结构一致

### 6.2 脚本入口

脚本：`tools/regen_evidence_captions.py`

默认关键参数：

- `--split train_split`
- `--n_versions 2`
- `--temperatures 0.8 1.0`
- `--dtype bfloat16`
- `--num_shards / --shard_index` 用于并行分片

当前生产环境使用 **BF16 非量化** 路径，不走 4bit / 8bit。

### 6.3 核心生成逻辑

1. `collect_stems(root, split)`
   - 从 `split/Black/Image` 和 `Mask` 收集 `(stem, img_path, mask_path)`
2. `extract_from_gt_mask(img_path, mask_path)`
   - 从 GT mask 抽结构化证据
3. `evidence_to_prompt_block(ev)`
   - 把证据序列化成 prompt block
4. 构造 user prompt
   - 明确要求输出连续中文鉴定文本
   - 强制引用证据中的 bbox
   - 禁止使用证据外坐标
5. 调用 `AutoModelForImageTextToText` 生成
   - `attn_implementation="sdpa"`
   - `dtype=torch.bfloat16`
6. 对输出做 `sanitize()`
   - 删除 `<think>`
   - 删除换行与 markdown 符号
7. 执行严格校验 / 宽松回退

### 6.4 严格校验与宽松回退

严格校验 `validate()` 要求：

- 无 `<think>`
- 长度在 `250~800`
- caption 内所有 bbox 都属于 `allowed_bboxes`
- 包含 `综上所述` / `综合分析` 等收束短语
- 开头与 fake / real 模板一致

宽松回退 `validate_loose()` 要求：

- 无 `<think>`
- 长度在 `150~1200`
- 不允许出现证据外 bbox

这样做的原因是：

- 严格校验负责保证高质量格式
- 宽松回退负责降低“空输出”概率
- 两者叠加后，能兼顾质量与吞吐

### 6.5 输出格式

当前 `train_v2` 采用**分片 jsonl**，而不是单个大文件：

```text
augmented_data/train_v2/
├── evidence_captions.shard0.jsonl
├── evidence_captions.shard1.jsonl
└── evidence_captions.shard2.jsonl
```

单条记录字段：

```json
{
  "image_path": "...",
  "mask_path": "...",
  "stem": "...",
  "version": 0,
  "temperature": 0.8,
  "gt_label": 1,
  "evidence": {"regions": [...]},
  "caption": "..."
}
```

### 6.6 Resume 与并行机制

该脚本支持两个重要能力：

1. **分片并行**
   - 通过 `--num_shards` 和 `--shard_index` 把 stem 列表切成多个互斥子集
   - 适合多卡并发
2. **断点续跑**
   - 如果输出 shard 已存在，会统计“已写入条目数”
   - 然后直接跳过前面已完成的样本

这意味着 `train_v2` 在被外部中断后，可以低成本继续跑，不需要删文件重来。

---

## 7. 训练接入逻辑

### 7.1 分割：`ForgerySegDataset` + `train_seg_ensemble.py`

基础逻辑：

- `data_dir` 现在默认是 `train_resume`
- 训练集来自 K-fold 对 `train_resume` 的划分
- 验证集仍然只来自 `train_resume` 基础数据，不混入增强集

增强接入方式：

- `--include_synth` 时，从 `augmented_data/synth` 读 `(image, mask, label=1)`
- `--include_real_ext` 时，从 `augmented_data/real_ext` 读 `(image, label=0)`
- `--use_weighted_sampler` 时，对基础训练集 + 增强集的合并标签做重采样

分割侧平衡权重：

```python
weights = [max_count / counts[label] for label in labels]
```

### 7.2 分类：`ForgeryClsDataset` + `train_classifier.py`

分类接入方式与分割一致：

- 基础根目录默认 `train_resume`
- 支持拼接 `synth` 和 `real_ext`
- 支持 `WeightedRandomSampler`

额外逻辑：

- 如果启用了 sampler，则不再额外给 `CrossEntropyLoss` 传类别权重
- 如果没启用 sampler，则用固定 `class_weights=[1.0, 0.25]` 缓和 4:1 失衡

### 7.3 VLM：`VLMSFTDataset` + `train_qwen35_9b.py`

VLM 训练现在也统一默认读 `train_resume`，因为其中已经接上了：

- `Black/Caption`
- `Black/Caption_clean`
- `White/Caption`

VLM 数据来源由三部分拼接：

1. `train_resume`
   - 原始 1000 条基础图文
   - Black 默认优先读 `Caption_clean`
2. `augmented_data/real_ext`
   - 真实负例图文扩充
3. `augmented_data/train_v2/*.jsonl`
   - evidence-aware 重生 caption

当 `inject_evidence=True` 时：

- 用户 prompt 不再是简单问句
- 而是由 GT mask 抽出的结构化证据块
- 训练时就强制模型学会“基于证据论证”，从而与推理保持一致

VLM 侧 sampler 逻辑与 seg/cls 一致，也是：

```python
weights = [max_count / counts[label] for label in labels]
```

---

## 8. 自动质检与验收标准

统一质检脚本：`tools/validate_augmented_data.py`

### 8.1 `Caption_clean`

检查项：

- `train_split` / `val` / `train_resume` 中 `Caption_clean` 目录是否存在
- 文件数是否符合预期

### 8.2 `real_ext`

检查项：

- `Image/*.jpg` 与 `Caption/*.md` 数量是否一致
- 抽检图像是否可正常打开
- caption 是否非空

### 8.3 `synth`

检查项：

- `Image` 与 `Mask` 数量是否一致
- `keep.txt` 是否存在
- 统计保留率

### 8.4 `train_v2`

检查项：

- 支持读取 `train_v2/*.jsonl` 的**多分片**格式
- `<think>` 残留
- 长度越界
- bbox 越界（caption 中引用了 evidence 之外的坐标）

---

## 9. 当前状态快照

截至当前实现，增强资产状态如下：

- `Caption_clean`
  - `train_split`: 640
  - `val`: 160
  - 基础全量: 800
- `real_ext`
  - 1100 张图
  - 1100 条 caption
- `synth`
  - 750 对 image/mask
  - 62 条通过 seg 过滤
- `train_v2`
  - 采用 `shard*.jsonl` 分片续跑
  - 当前仍在生成中，不固定为单文件

因此，当前可直接用于训练的输入组合为：

- seg / cls:
  - `train_resume`
  - 可选加 `augmented_data/synth`
  - 可选加 `augmented_data/real_ext`
- VLM:
  - `train_resume`
  - `augmented_data/real_ext`
  - `augmented_data/train_v2/*.jsonl`

---

## 10. 建议的训练命令

### 10.1 分割

```bash
python train_seg_ensemble.py   --arch segformer --fold 0 --gpu 0   --data_dir train_resume --save_dir checkpoints   --resume --resume_log logs/seg_segformer_gpu0_serial.log
```

若要接入增强：

```bash
python train_seg_ensemble.py   --arch segformer --fold 0 --gpu 0   --data_dir train_resume --save_dir checkpoints   --include_synth --include_real_ext --use_weighted_sampler
```

### 10.2 分类

```bash
python train_classifier.py   --fold all --gpu 0   --data_dir train_resume --save_dir checkpoints   --include_synth --include_real_ext --use_weighted_sampler
```

### 10.3 VLM

```bash
python train_qwen35_9b.py   --gpu 0   --data_dir train_resume   --augmented_dir augmented_data/train_v2   --real_ext_dir augmented_data/real_ext   --use_weighted_sampler
```

---

## 11. 关键约束与风险

- `train_resume` 是当前训练默认入口；在底层目录枚举问题修复前，不建议再把训练脚本切回 `train`
- `synth` 不应直接进入 VLM SFT
- `train_v2` 生成必须保留 bbox 白名单校验，不能只看“有没有生成文本”
- `real_ext` 的 COCO caption 是真实性模板，不是细粒度描述语料；它的主要用途是补充负例风格和稳定“真实图论证”模板
- `validate_augmented_data.py` 现在已经支持读取 `train_v2/*.jsonl` 多分片；旧的单文件假设不再成立

---

## 12. 一句话总结

当前 v2 增强的真实结构是：

- 用 `Caption_clean` 修基础文本监督
- 用 `synth` 扩像素级伪造分布
- 用 `real_ext` 补真实负例
- 用 `train_v2` 追加 evidence-aware caption
- 用 `train_resume` 统一承接所有训练脚本，绕开底层 `train/Black/Image` 枚举故障
