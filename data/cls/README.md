# 分类训练数据入口

`train_classifier.py` 与 seg 同源，多了 ELA 流而少了 mask：

| flag | 默认 |
|---|---|
| `--data_dir` | `data/raw/train_resume` |
| `--synth_dir` | `data/processed/synth` |
| `--real_ext_dir` | `data/processed/real_ext` |

当前 `checkpoints/cls/` 不存在，整个分类支线尚未训练。
