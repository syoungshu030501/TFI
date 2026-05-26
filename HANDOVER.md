# TFI · Agent 交接文档（2026-05-02 深夜更新）

> **给下一任 agent**：读完本文 + [`README.md`](README.md)（架构 / 环境 / 参数 / 代码讲解） +
> [`journal.md`](journal.md)（实验日志）+ [`data/analysis/distribution_report.md`](data/analysis/distribution_report.md) 即可上手。
> 本文只讲"做什么、怎么做、坑在哪"。

---

## 0. 30 秒 primer

- **项目**：图像伪造分析比赛（Detection / Grounding / Explanation 三任务），目标 `S_Fin 0.945-0.955`（v1 = 0.9034）。
- **当前阶段**：项目重构完成（archive/v1, data/build, train/{sft,mipo,pgrpo,fipo}, eval/baseline 全部到位），FIPO 模块已从 VLM-posttraining 移植并改写为 TFI 任务专用，**v2 SFT-baseline 已完成**（54/54 step，ckpt-54 落 NFS）。
- **下一任要做的事**：跑 (3.2) HF-EFG-CN 子集 → (3.3) 合并集 → (3.5) Stage B 主 SFT → (3.6) prepare_fipo_data → (3.7) FIPO 主训练。
- **环境**：conda env `TFI`（torch 2.5.1+cu124 / transformers 4.49.0 / **ms-swift 3.4 Veritas fork (editable)** / vllm 0.7.3）和 `TFI_judge`（R1-Distill-70B 推理）。
- **关键资源**：所有大文件在 NFS `/mnt/nfs/young/TFI/{models,data,code,judge_model,runs}`，本地 `/home/young/TFI/` 仅项目代码（**无任何模型文件或软链接**，所有脚本默认值已统一指向 NFS 绝对路径）。
- **GPU**：8×L20×46G，**GPU 0 历史 ECC 错误，全程禁用**。可用 7 张 (GPU 1-7) = 322G。

---

## 1. 重构后的目录（已就位）

```
/home/young/TFI/
├── archive/v1/                  # v1 baseline 历史代码（只读归档，含 stale config.yaml）
├── data/
│   ├── build/                   # 数据构建脚本
│   │   ├── build_v2_sft.py            # SFT 主集构建 (1009 条已生成)
│   │   ├── build_v2_mipo.py           # MiPO 偏好对（兜底，主路线不用）
│   │   ├── build_v2_pgrpo.py          # P-GRPO 数据
│   │   ├── build_hydra_efg_subset.py  # ★ HF-EFG-CN 子集 (4k EFG + 4k real)
│   │   ├── merge_official_hydra.py    # ★ TFI ⊕ HF 30/70 合并
│   │   └── data_guard.py
│   ├── analysis/                # ★ 数据分布报告
│   │   └── distribution_report.md
│   ├── processed/               # v1 留下的 synth + caption_local_v2
│   ├── meta/  raw/              # data 元信息 / 原始数据软链
│   └── (大数据在 NFS: /mnt/nfs/young/TFI/data/{v2,HydraFake})
├── train/
│   ├── __init__.py              # ★ Python 包标记
│   ├── sft/train_sft.sh         # ms-swift SFT (GPU 1-7, NPROC=7)
│   ├── mipo/train_mipo.sh       # 兜底
│   ├── pgrpo/train_pgrpo.sh     # 兜底
│   └── fipo/                    # ★ 主路线：从 VLM-posttraining 移植 + TFI 改写
│       ├── __init__.py
│       ├── main_fipo.py         # verl entry，注册 future_kl + reward_manager
│       ├── launch.sh            # 启动脚本（CUDA_VISIBLE_DEVICES 默认 1-7）
│       ├── schema.py            # ★ 新：TFI 6-tag CoT 解析器 + system prompt
│       ├── reward_fn.py         # ★ 新：9-reward 计算（5 rule + 4 external hook）
│       ├── prepare_fipo_data.py # ★ 新：SFT JSON → verl parquet (含 GT bboxes)
│       ├── *_legacy.py          # VLM-posttraining 留下的待重写参考
│       ├── config/train.yaml    # ★ 新：TFI FIPO 超参
│       └── verl_patches/
│           ├── future_kl_loss.py      # 保留：注册 POLICY_LOSS["future_kl"]
│           ├── reward_manager.py      # ★ 新：TFIAuditRewardManager
│           └── reward_manager_legacy.py
├── eval/baseline/               # M(-1) prompt-only baseline 代码 + 结果
├── code/verl/                   # ★ 新：从 VLM-posttraining 复制 (18 MB)
├── sitecustomize.py             # ★ 新：Ray worker 自动 import future_kl + GPU remap
├── reference/                   # 论文/方法学 PDF
├── README.md
├── HANDOVER.md                  # 本文
└── logs/                        # 运行日志
```

NFS 布局不变：
```
/mnt/nfs/young/TFI/
├── models/        193 GB
├── data/v2/       sft.json (1009) / sft_val.json (53) / mipo.json / pgrpo.json
├── data/HydraFake/jsons/ + hydrafake/{train,val,test}/  (全图已解压)
├── code/Veritas/  ms-swift Veritas fork (editable)
├── judge_model/   R1-Distill-Llama-70B
└── runs/sft/v2sft_baseline_1009/  ← 当前 SFT-baseline 输出在这
```

---

## 2. 已完成（不必重做）

| 类别 | 项 | 状态 |
|---|---|---|
| **重构** | 目录调整、v1 归档、训练脚本搬到 train/{sft,mipo,pgrpo,fipo}/ | ✅ |
| **重构** | 移除所有 src.stage3_fipo.* 旧路径，全部改为 train.fipo.* | ✅ |
| **数据** | TFI v2 SFT 1009 + 53 val | ✅ |
| **数据** | HydraFake jsons + 全图（解压完成） | ✅ |
| **数据** | 分布分析报告 → `data/analysis/distribution_report.md` | ✅ |
| **代码** | FIPO 模块 5 件套（schema / reward_fn / reward_manager / prepare_fipo_data / config） | ✅ |
| **代码** | sitecustomize.py（Ray worker auto-import） | ✅ 改为 train.fipo.* |
| **代码** | code/verl/（从 VLM-posttraining 复制 18MB） | ✅ |
| **训练** | v2 SFT-baseline 1009 完成（54/54 step，train_loss 1.486，ckpt-54） | ✅ |
| **路径** | 本地 `models/` 软链清除；所有脚本统一 NFS `/mnt/nfs/young/TFI/models/...` | ✅ |
| **路径** | v1 stale `config.yaml` 归档到 `archive/v1/` | ✅ |
| **文档** | README 拆分：`README.md`（架构/环境/参数/代码）+ `journal.md`（实验日志）| ✅ |
| **GPU** | GPU 0 ECC 风险已记入 memory + 所有脚本默认跳过 | ✅ |

---

## 3. 待办（按优先级）

### P0（决定 v2 主路线）

#### 3.1 SFT-baseline 已完成 ✅
ckpt 在 `/mnt/nfs/young/TFI/runs/sft/v2sft_baseline_1009/v0-20260502-224638/checkpoint-54`
（54/54 step，train_loss 1.486，39 min runtime）。

#### 3.2 生成 HydraFake-EFG-CN 子集（~3 min）
```bash
conda activate TFI && cd /home/young/TFI
python -m data.build.build_hydra_efg_subset \
    --out /mnt/nfs/young/TFI/data/v2/hydra_efg_cn.json \
    --efg_limit 4000 --real_limit 4000
# 输出：8000 条带中文 6-tag CoT 的 EFG fake + real
```

#### 3.3 生成合并集（~30 sec）
```bash
python -m data.build.merge_official_hydra \
    --tfi /mnt/nfs/young/TFI/data/v2/sft.json \
    --hydra /mnt/nfs/young/TFI/data/v2/hydra_efg_cn.json \
    --out /mnt/nfs/young/TFI/data/v2/sft_merged.json \
    --hydra_ratio 0.30
# 输出：~1441 条 (1009 TFI + 432 HF)，meta 见 sft_merged_meta.json
```

#### 3.4 (可选) Stage A warmup — 仅 HF-EFG-CN 1 epoch
若直接 Stage B 单阶段先跑通也行；想压性能就两阶段。
```bash
# 改 train/sft/train_sft.sh 的 DATASET_PATH 指向 hydra_efg_cn.json，epoch=1
# 取末轮 ckpt 作为 Stage B 起点
```

#### 3.5 主 SFT (Stage B)
```bash
# 改 train/sft/train_sft.sh 的 DATASET_PATH 指向 sft_merged.json
bash train/sft/train_sft.sh v2sft_merged_1441
```

#### 3.6 FIPO 数据准备
```bash
python -m train.fipo.prepare_fipo_data \
    --in_train /mnt/nfs/young/TFI/data/v2/sft_merged.json \
    --in_val   /mnt/nfs/young/TFI/data/v2/sft_val.json \
    --out_dir  data/fipo \
    --max_train 2000 --max_val 200
```

#### 3.7 FIPO 主训练
```bash
# 把上一步 SFT 末轮 ckpt 设为 MODEL_PATH（必须先 merge LoRA → HF dump）
MODEL_PATH=/mnt/nfs/young/TFI/runs/sft/v2sft_merged_1441/v0-*/checkpoint-XXX_merged \
    bash train/fipo/launch.sh
# 默认：CUDA_VISIBLE_DEVICES=1-7，N_GPUS=7，verl + future_kl
```

### P1（评测/对比）
- v2 SFT ckpt 跑 val 53 + 200 official → 算 S_Fin
- judge 综合评分（R1-70B 跑 baseline + SFT + FIPO 对比）
- HydraFake 4 级 OOD 评测

### P2（可选）
- 把 specialist verifiers (DINOv3/SigLIP/MaskCLIP) 部署后接入 reward_manager 的 `_external_scores_for()`
- GRM caption rubric server 接入

---

## 4. FIPO 9-reward 现状

> 完整选型论证: [`README.md` §2.5](README.md#25-rl-算法选型fipo)；
> 完整 reward 权重表 + GT schema: [`README.md` §5.3](README.md#53-9-reward-权重)；
> 9-reward 计算逻辑: `train/fipo/reward_fn.py` docstring。

| Reward | 权重 | 已实现? | 说明 |
|---|---|---|---|
| R_format | 0.10 | ✅ rule | 4 个必需 tag 各出现 1 次 + answer 是 real/fake |
| R_consistency | 0.10 | ✅ rule | fake⇒有 bbox/region；real⇒无 |
| R_label_gt | 0.15 | ✅ rule | `<answer>` == GT label |
| R_iou_gt | 0.15 | ✅ rule | bbox max IoU vs GT bbox（real⇒空匹配=1） |
| R_phrase_check | 0.05 | ✅ rule | conclusion 中数字/region 在 GT phrase 池中 |
| R_loc | 0.10 | 🔌 hook | 等 DINOv3 server 部署 |
| R_cls | 0.10 | 🔌 hook | 等 SigLIP-2 server |
| R_forensic | 0.10 | 🔌 hook | 等 MaskCLIP+Mesorch |
| R_grm | 0.10 | 🔌 hook | 等 GRM caption rubric server |
| R_qwen_periodic | 0.05 | 🔌 hook | qwen-max API，每 50 step 抽 10 张 |

**FIPO v1 起步**：仅 rule-based 5 项跑（合计 0.55 权重，max 总分 0.55）。
specialist + GRM 部署后通过子类化 `TFIAuditRewardManager._external_scores_for(item)` 接入，无需改 reward_fn 主路。

---

## 5. 关键路径速查

```
代码主目录:           /home/young/TFI/
├── data/build/                    数据构建脚本
├── data/analysis/distribution_report.md   ★ 数据合并策略
├── train/fipo/                    FIPO v1 全套
├── train/{sft,mipo,pgrpo}/        SFT/MiPO/P-GRPO 训练脚本
├── eval/baseline/                 prompt-only / ceiling / judge / Veritas zero-shot
├── code/verl/                     verl-latest（从 VLM-posttraining 复制）
├── sitecustomize.py               Ray worker 自动 patch
├── reference/                     论文 + Veritas_method.md + FIPO.pdf
├── README.md                      主文档（架构 / 环境 / 参数 / 代码讲解）
├── journal.md                     实验日志、状态更新、踩坑记录
└── HANDOVER.md                    本文

NFS:                  /mnt/nfs/young/TFI/
├── models/           Qwen3.5-9B / Veritas-Cold-Start / specialists
├── data/v2/          sft / sft_val / mipo / pgrpo + (待生成) hydra_efg_cn / sft_merged
├── data/HydraFake/   jsons + 全图
├── code/Veritas/     ms-swift Veritas fork (editable)
├── judge_model/      R1-Distill-Llama-70B
└── runs/sft/v2sft_baseline_1009/  当前 SFT-baseline 输出

姊妹项目:             /home/young/VLM-posttraining/   ← FIPO 参考实现
```

---

## 6. 第一次接手 60 秒自检

```bash
# 1) 环境
conda activate TFI && python -c "import torch, transformers, swift, vllm; \
  print(torch.__version__, transformers.__version__, swift.__version__, vllm.__version__)"
# 期望：2.5.1+cu124 4.49.0 3.4.0.dev0 0.7.3

# 2) FIPO 模块导入
cd /home/young/TFI && python -c "
from train.fipo.schema import SYSTEM_PROMPT, parse_response
from train.fipo.reward_fn import compute_reward, GroundTruth, BREAKDOWN_SCHEMA
from train.fipo.prepare_fipo_data import _row_to_verl
print('FIPO modules OK; reward keys:', BREAKDOWN_SCHEMA[:5], '...')"

# 3) 数据
ls -lh /mnt/nfs/young/TFI/data/v2/
# 期望: sft.json (~3MB) sft_val.json mipo.json pgrpo.json
# (新): hydra_efg_cn.json sft_merged.json (待生成)

# 4) GPU + SFT 状态
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ps -p 1314727 -o pid,etime,stat 2>/dev/null
tail -3 logs/v2_train/v2sft_baseline_1009.log

# 5) FIPO reward 自检
python -m train.fipo.reward_fn
# 期望: 一行 score=0.4673（well-formed fake on fake-with-bbox GT）
```

---

## 7. 全局禁忌

- 🚫 **GPU 0 永远不要用**（历史 ECC 错误；已写入 ~/.claude memory 与所有训练脚本默认值）
- 🚫 **不要 `pip install -U vllm` / `transformers`**（driver 550 装不了 vllm 0.17+）
- 🚫 **不要重装 ms-swift**（pip 装的没 `internvl3` 模板，必须用 Veritas fork editable）
- 🚫 **不要把 NFS 当 scratch 写**（高频 ckpt 慢，可考虑 `runs_local/` 中转）
- 🚫 **不要直连 huggingface.co / github.com**（用 ModelScope / kkgithub.com 镜像）

---

## 8. 完成定义（DoD）

- [x] 重构: archive/v1, data/build, train/{sft,mipo,pgrpo,fipo}, eval/baseline 落地
- [x] FIPO 模块从 VLM-posttraining 移植 + TFI 改写
- [x] sitecustomize.py 路径修正（src.stage3_fipo → train.fipo）
- [x] data/analysis/distribution_report.md 写完
- [x] v2 SFT-baseline 跑完（pure TFI 1009 条 sanity check，ckpt-54）
- [x] 模型路径迁移：本地 `models/` 删除，脚本统一 NFS 绝对路径
- [x] 文档拆分：README（架构/环境/参数/代码）+ journal（实验日志）
- [ ] HF-EFG-CN 子集 + 合并集生成
- [ ] Stage B SFT 跑完（merged 数据）
- [ ] FIPO dry-run 1 step
- [ ] FIPO 主训练 ≥ 200 step，val_freq 50 见到 R_label 上升
- [ ] judge 评分 baseline + SFT + FIPO 完整对比表
- [ ] journal 加入新条目

---

更新：2026-05-02 深夜（v2 SFT-baseline 完成 + 模型路径迁移 NFS + README/journal 拆分）
