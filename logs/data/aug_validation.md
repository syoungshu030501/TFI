# 增强数据质检报告

## 1. Caption_clean
- `train_split/Black/Caption_clean` : 640 条
- `val/Black/Caption_clean` : 160 条
- `train_resume/Black/Caption_clean` : 800 条

## 2. real_ext (§5)
- `real_ext/Image` : 1100 张 (抽检 20 张坏图 0)
- `real_ext/Caption` : 1100 条
- 对齐: OK

## 3. synth (§4)
- `synth/Image` : 750 张; Mask: 750
- `synth/keep.txt` : 62 条 (通过 seg 过滤)
- 保留率: 62/750 = 8.3%

## 4. train_v2 evidence_captions (§6)
- `train_v2/*.jsonl` : 3 个分片
- `train_v2` 总条数: 128
  - 完全合规: 79
  - <think> 残留: 0
  - 长度越界: 49
  - bbox 越界: 0