# 数据分布与合并策略报告

> 写于 2026-05-02。给出 (1) TFI 官方 GT-清洗数据分布、(2) HydraFake 训练子集分布、(3) 合并可行性评估、(4) 推荐合并策略。后续 SFT/FIPO 数据准备依据本文档。

## 1. 数据源概览

| 数据源 | 样本数 | label=fake | label=real | 备注 |
|---|---|---|---|---|
| TFI v1 train/Black | 800 | 800 | 0 | 中文 caption + mask |
| TFI v1 train/White | 200 | 0 | 200 | 中文 caption，无 mask |
| TFI processed/synth | ~50–60 | 全部 fake | 0 | 合成的 copy-move/splice/text-edit，含 mask |
| TFI processed/real_ext | 1100 | 0 | 1100 | 真实图，无 mask |
| **TFI 合计 (build_v2_sft 当前输出)** | **1009** | 824 | 195 | 见 `/mnt/nfs/young/TFI/data/v2/sft.json` |
| HydraFake `sft_36k.json` | 36750 | 18038 | 18712 | 4 子类，全英文 CoT |
| ↳ entire face generation (EFG) | 5573 | 5573 | 0 | **唯一与 TFI 任务相邻的 HydraFake 子类** |
| ↳ face swapping (FS) | 7402 | 7402 | 0 | 跳过：与 TFI 任务无关 |
| ↳ face reenactment (FR) | 5063 | 5063 | 0 | 跳过：视频帧任务 |
| ↳ real | 18712 | 0 | 18712 | EFG 配对 real 池 |
| HydraFake test (4 级 OOD) | 53272 | ~26500 | ~26500 | 仅作 OOD 评测，**不进训练** |

## 2. TFI 当前 SFT 分布问题

```
build_v2_sft.py 当前输出 1009 条:
  label=fake : 824 (81.7%)
  label=real : 185 (18.3%)
```

**问题**: 严重正负失衡。v1 baseline 学到的就是过度激进倾向 fake — `S_Fin = 0.9034` 中 Det 的 false-positive 偏高。要把 baseline 推到 0.945+ 必须把负样本拉到至少 35%。

**可用补救**:
- TFI 自己的 `processed/real_ext` 1100 张未全用上（build_v2_sft 只取了 200）→ 把全量 1100 拉进来 = label_real ≈ 1295；fake 824 → 真:伪 ≈ 61:39
- 加 HydraFake-EFG-CN 真实人脸子集 4000 张 → 大幅扩充 real 多样性

## 3. HydraFake 合并可行性

### 3.1 GO 项
| 维度 | TFI | HydraFake-EFG | 评估 |
|---|---|---|---|
| 任务定义 | 真伪二分类 + bbox/region 解释 | 真伪二分类 + 整脸语义解释 | ✅ 完全相邻，bbox→region 转换显然 |
| 标注格式 | `<fast>/<reasoning>/<conclusion>/<answer>` 6-tag | 同一套 4-tag 子集 + 英文 | ✅ tag schema 兼容 |
| 文件格式 | local PIL JPG/PNG | local PNG | ✅ 直接 PIL.open() 即可 |
| 图像质量 | 街拍/广告图为主 | celebahq + Dall-E/StyleGAN3 等 sub-gen | ✅ 多样性互补 |

### 3.2 STOP 项（必须显式处理）
| 风险 | 影响 | 缓解 |
|---|---|---|
| HF 自带 assistant 全英文 | 中文 SFT 受污染语言切换 | **重写 assistant 为中文**（见 `data/build/build_hydra_efg_subset.py`） |
| HF EFG 无 bbox（整脸生成） | bbox-based reward (R_iou_gt) 无监督 | EFG 样本 bbox 字段留空，reward_fn 检测到 GT.bboxes=[] 时仍能给出 R_consistency 全分（real⇒no bbox 等价）|
| FS / FR 子类与 TFI 任务无关 | 训混会教模型"找替换的那张脸" | 显式过滤，仅保留 EFG + real |
| HF real 与 TFI real 域差距 | celebahq 是高分 GAN-prone 图，TFI real 是日常照片 | 分两阶段；阶段 A 仅 HF (1 epoch)，阶段 B 30/70 mixed |

### 3.3 ADAPT 项
- HF 子生成器名 (`Dall-E1` / `StyleGAN3` / ...) 通过 path 解析后注入到 reasoning 模板，让模型学到 "AI 整脸生成 = 数字伪造" 而不是某个生成器指纹。
- HF EFG real 池只取 4000（全 18k 用完会让 real 子集比例失衡）。

## 4. 推荐合并策略（两阶段）

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage A · 全 HydraFake-EFG-CN warmup (1 epoch, ~1 hr on 7×L20)  │
│   data: data/build/build_hydra_efg_subset.py 输出               │
│         (efg_limit=4000, real_limit=4000, 共 ~8k)               │
│   目的: 让 base 模型先适配 6-tag 中文 CoT 在 AI 全脸场景        │
│   验证: 在 HF test 4-level OOD 上跑评测，落在 baseline 论文     │
│         ±5% 即可继续                                              │
└────────────────────────┬─────────────────────────────────────────┘
                         │ 取 Stage A ckpt (last 不要 best)
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage B · TFI ⊕ HF-EFG (70/30, ~2-3 epoch)                      │
│   data: data/build/merge_official_hydra.py 输出                 │
│         hydra_ratio=0.30 → 1009 TFI + 432 HF (按 30%)           │
│   目的: 主任务训练，重点压测 TFI 的局部篡改 (text/copy-move)    │
│   验证: TFI val (53 张) S_Fin → 目标 ≥ 0.94                     │
└──────────────────────────────────────────────────────────────────┘

可选 Stage A 跳过路线: 直接 Stage B 单阶段 (节省 1 epoch)，仅在数据
ablation 时比较两阶段是否真的赢；先按论文 default 直接跑两阶段。
```

## 5. FIPO 数据策略

FIPO 不需要 HydraFake — RL 阶段使用 SFT 已学到的能力做更细颗粒优化，HF EFG
样本对 R_iou_gt 没贡献，混进来反而拉低 reward 信号。FIPO 用：

```
data/fipo/train.parquet ← prepare_fipo_data 从 sft_merged.json 抽 2000
data/fipo/val.parquet   ← prepare_fipo_data 从 sft_val.json 抽 200
```

GT blob (`reward_model.ground_truth`) 通过 `_row_to_verl()` 写入：
```json
{"label": 0|1,
 "bboxes": [[x1,y1,x2,y2], ...],
 "phrases": ["<region>...", "“引号文字”", ...]}
```

bbox 来源优先级: **mask_path → mask_to_bbox** > **<bbox> tag 解析** > **空列表**.

## 6. 当前数据状态 (2026-05-02 23:30)

| 件 | 路径 | 状态 |
|---|---|---|
| TFI SFT 主集 | `/mnt/nfs/young/TFI/data/v2/sft.json` | ✅ 1009 条已存在（含部分 HF EFG）|
| TFI SFT val | `/mnt/nfs/young/TFI/data/v2/sft_val.json` | ✅ 53 条 |
| HF-EFG-CN 子集 | `/mnt/nfs/young/TFI/data/v2/hydra_efg_cn.json` | ⏳ 待生成: `python -m data.build.build_hydra_efg_subset` |
| 合并集 (Stage B 输入) | `/mnt/nfs/young/TFI/data/v2/sft_merged.json` | ⏳ 待生成 (依赖上一行) |
| FIPO parquet | `data/fipo/train.parquet`, `data/fipo/val.parquet` | ⏳ 待 SFT 完成后跑 prepare_fipo_data |

## 7. 后续动作 checklist

- [ ] 生成 `hydra_efg_cn.json` (4k EFG + 4k real)，保存到 NFS
- [ ] 生成 `sft_merged.json`（hydra_ratio=0.30）
- [ ] 起 Stage A SFT（仅 hydra_efg_cn.json，1 epoch）
- [ ] 取 Stage A 末轮 ckpt 作为 Stage B 起点
- [ ] 起 Stage B SFT（sft_merged.json，2 epoch）
- [ ] SFT 完成 → 跑 `prepare_fipo_data`
- [ ] 起 FIPO（`train/fipo/launch.sh`）
