# 分割训练数据入口

`train_seg_ensemble.py` 直接消费这三个目录：

| flag | 默认 | 内容 |
|---|---|---|
| `--data_dir` | `data/raw/train_resume` | 800 张基础（Black 640 + White 160） |
| `--synth_dir` | `data/processed/synth` | 62 张通过 keep.txt 过滤的合成伪造 |
| `--real_ext_dir` | `data/processed/real_ext` | 1100 张真实图扩充 |

5-fold 在内存中按 `dataset.create_kfold_splits` 实时切，未持久化到文件。
**只有 fold0 有合格 ckpt**（IoU 0.8370），fold1-4 在数据 pipeline 收敛后重训。
