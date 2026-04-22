# data 健康报告

`data/` 当前快照（由 scripts/data/guard.py 生成）

## ① raw 原料层

| 项 | 值 | 判定 |
|---|---|:-:|
| `raw/train_resume/Black/Image` | 800 | ✓ |
| `raw/train_resume/White/Image` | 200 | ✓ |
| `raw/val/Black/Image` | 160 | ✓ |
| `raw/val/White/Image` | 40 | ✓ |
| `raw/test/Image` | 500 | ✓ |

## ② processed 处理层

| 项 | 值 | 判定 |
|---|---|:-:|
| `processed/synth/Image` | 750 | ✓ |
| `processed/synth/Mask` | 750 | ✓ |
| `processed/real_ext/Image` | 1100 | ✓ |
| `processed/real_ext/Caption` | 1100 | ✓ |
| `processed/synth/keep.txt` | 62 行 | ✓ |
| `processed/caption_local_v2/*.jsonl` | 902 行（旧本地 9B） | — |

## ③ vlm 层（API 蒸馏）

| 项 | 值 | 判定 |
|---|---|:-:|
| `vlm/caption_api_v3/*.jsonl` | 0/1280 行 | ⏳ |

## ④ symlink 健康

| 项 | 值 | 判定 |
|---|---|:-:|
| `data/raw/train_resume` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/train_resume | ✓ |
| `data/raw/val` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/val | ✓ |
| `data/raw/test` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/test | ✓ |
| `data/processed/synth` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/augmented_data/synth | ✓ |
| `data/processed/real_ext` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/augmented_data/real_ext | ✓ |
| `data/processed/caption_local_v2` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/augmented_data/train_v2 | ✓ |
| `data/vlm/caption_api_v3` | /mnt/nfs/young/my_dt/True-or-Fake-Image-main/augmented_data/caption_api_v3 | ✓ |

## ⑤ checkpoint 现状（参考，不卡阻）

| 项 | 值 | 判定 |
|---|---|:-:|
| `checkpoints/seg/segformer_fold0` | 存在 | ✓ |
| `checkpoints/seg/segformer_fold1` | 缺失 | · |
| `checkpoints/seg/segformer_fold2` | 缺失 | · |
| `checkpoints/seg/segformer_fold3` | 缺失 | · |
| `checkpoints/seg/segformer_fold4` | 缺失 | · |
| `checkpoints/cls/efficientnet_fold0` | 缺失 | · |
| `checkpoints/calibrator` | 缺失 | · |
| `checkpoints/qwen35_9b` | 缺失 | · |

## 失败的硬契约

- vlm/caption_api_v3 0 < 1152 (90% 准入线)
