# Prompt-only Baseline Judge Report

Judge: `/mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b` (TP=4)

Val samples: 200

## Mean scores (1-10)

| set        |   n |   accuracy |   evidence |   completeness |   language |   overall |
|:-----------|----:|-----------:|-----------:|---------------:|-----------:|----------:|
| qwen36_zs  | 200 |      4.305 |       4.91 |          5.015 |      7.025 |     5.314 |
| qwen36_cot | 200 |      4.76  |       5.15 |          5.195 |      7.24  |     5.586 |

## Per-sample CSVs

- `qwen36_zs_judge.csv`
- `qwen36_cot_judge.csv`