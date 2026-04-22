# Legacy 归档目录

本目录保留 **旧方案**（Qwen3.5-397B 教师 LoRA → 数据生成 → Qwen3-VL-8B-Thinking 学生 CoT 蒸馏）的全部脚本，仅作历史回溯与对照实验。

## 归档原因

1. **算力门槛过高**：397B 教师模型训练 + 推理需要 8x MI325X (256GB) 集群，
   生成 1000×3 条增强数据耗时约 47 小时；
2. **效果有限**：教师生成的 CoT caption 中坐标幻觉严重，蒸馏出的小模型在结构化
   伪造定位任务上未见显著提升；
3. **依赖复杂**：依赖 ROCm 7.1、transformers 5.2 nightly、`fla`、`causal-conv1d`
   等专用库，且 `torch._grouped_mm` 在 ROCm 上需 `rocm_compat.py` 打补丁。

## 新方案

参见仓库根目录的 [README.md](../README.md)：使用 Qwen3.5-9B（2026.3 发布的原生
多模态模型）+ 结构化证据注入（`evidence.py`）+ 轻量校准器（`calibrator.py`），
在 2 卡 RTX 5090 / 1 卡 L20 上即可复现。

## 文件清单

| 文件 | 说明 |
|---|---|
| `train_teacher.py` | 397B Qwen3.5 教师 LoRA 微调（DeepSpeed ZeRO-3, 8 卡） |
| `merge_lora.py` | LoRA 合并到 397B 基座 |
| `generate_teacher_data.py` | 教师生成增强 caption |
| `train_student_8b.py` | Qwen3-VL-8B-Thinking 学生全量微调（旧 baseline） |
| `test_inference.py` | 397B 推理验证 |
| `rocm_compat.py` | ROCm 上 `torch._grouped_mm` 的顺序回退补丁 |
| `ds_config_z3.json` | DeepSpeed ZeRO-3 配置 |

## 如何恢复使用

```bash
cd legacy/
cp *.py *.json ../
# 重新创建 ROCm 环境（参见旧版 PROJECT.md 第 2.2 节）
```
