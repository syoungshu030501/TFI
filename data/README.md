# TFI 数据入口

整个项目的所有数据都从 `data/` 进，按**消费者**而非来源切分。所有子目录里的实际比特仍存在
`/mnt/nfs/young/my_dt/True-or-Fake-Image-main/` 下面，`data/` 用 symlink 组织，避免拷贝。

## 目录契约

```
data/
├── raw/                  # ① 原料层（read-only，不要在这里加工生成）
│   ├── train_resume/     # 实际训练用的稳定 800/200 视图（NFS 枚举安全）
│   ├── train/            # 历史原始 train（目录枚举偶发卡死，禁用）
│   ├── val/              # 200 张验证集（Black=160, White=40）
│   └── test/             # 测试集，仅 Image
│
├── processed/            # ② 处理层（pipeline 中间产物）
│   ├── synth/            # 合成伪造（copy_move/splicing/text_replace_like）
│   │                     #   只喂 seg/cls，不进 VLM SFT；keep.txt 控制最终入选
│   ├── real_ext/         # 真实图扩充（White 强增广 + COCO val2017 抽样）
│   │                     #   服务 seg/cls 的负例平衡与 VLM 真实样本
│   └── caption_local_v2/ # 旧本地 Qwen3.5-9B 生成的 evidence_captions（待 v3 替换）
│
├── seg/                  # ③ 分割训练入口（仅 README，直接消费 raw + processed）
│   └── README.md
├── cls/                  # ③ 分类训练入口
│   └── README.md
├── vlm/                  # ③ VLM SFT 入口
│   ├── README.md
│   └── caption_api_v3/   # 由 scripts/data/regen_caption_api.py 写入（强 API 重生）
│
└── meta/                 # ④ 共享元数据
    ├── data_health.md    # guard.py 输出
    └── README.md
```

## 为什么不直接读 `train_resume/`、`augmented_data/`？

历史上脚本里到处写 `--data_dir train_resume`、`--synth_dir augmented_data/synth`、
`--augmented_dir augmented_data/train_v2`，路径分散且名字含义模糊（`train_v2` 既不是
"v2 的训练集"也不是"训练任务"，而是一个旧本地 VLM 蒸馏产物）。统一入口后：

- **来源透明**：raw（一手）/ processed（脚本生成）/ {seg,cls,vlm}（消费者口径）
- **命名达意**：`caption_local_v2`（旧本地 9B 生成）vs `caption_api_v3`（新 qwen-vl-max API 生成）
- **路径稳定**：训练脚本只认 `data/...`，底层 NFS 物理路径换地方不影响上层

## 数据 → 消费者矩阵

| 来源 | seg 训练 | cls 训练 | VLM SFT | calibrator/eval |
|---|:-:|:-:|:-:|:-:|
| `raw/train_resume/Black` (640 张) | ✓ | ✓ | ✓ | — |
| `raw/train_resume/White` (160 张) | ✓ | ✓ | ✓ | — |
| `raw/val` (200 张) | — | — | — | ✓ |
| `raw/test` (Image only) | — | — | — | ✓ (推理输入) |
| `processed/synth` (62 张 keep) | ✓ | ✓ | ✗ | — |
| `processed/real_ext` (1100 张) | ✓ | ✓ | ✓ | — |
| `processed/caption_local_v2` (902 行) | — | — | ✗ (废弃) | — |
| `vlm/caption_api_v3` (~1280 行 目标) | — | — | ✓ | — |

## 健康检查

```bash
# 一键体检：路径/数量/symlink 是否齐全
python scripts/data/guard.py
```

输出到 [meta/data_health.md](meta/data_health.md)。
