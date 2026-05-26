# Prompt-only Baseline Judge Report

Judge: `/mnt/nfs/young/TFI/judge_model/r1-distill-llama-70b` (TP=4)

Val samples: 200

## Mean scores (1-10)

| set   |   n |   accuracy |   evidence |   completeness |   language |   overall |
|:------|----:|-----------:|-----------:|---------------:|-----------:|----------:|
| sft   | 200 |      8.085 |      7.93  |          7.38  |      8.57  |     7.991 |
| zs    | 200 |      3.655 |      4.215 |          4.28  |      7.395 |     4.886 |
| fs    | 200 |      4.275 |      4.645 |          4.805 |      6.91  |     5.159 |
| cot   | 200 |      4.025 |      4.895 |          4.825 |      7.055 |     5.2   |

## Per-sample CSVs

- `sft_judge.csv`
- `zs_judge.csv`
- `fs_judge.csv`
- `cot_judge.csv`