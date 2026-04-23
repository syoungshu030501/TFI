# TFI Ablation Report

_val_dir = `data/raw/val`, n = 200_

| Config | Acc | Precision | Recall | F1 | mean IoU | mean Dice |
|---|---|---|---|---|---|---|
| seg=segformer_only | cal=xgb | cls=on | tta=on | multiscale=True | 0.9750 | 0.9697 | 1.0000 | 0.9846 | 0.8160 | 0.8724 |
