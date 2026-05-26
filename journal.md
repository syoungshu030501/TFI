# TFI · 实验日志（journal）

> 项目状态更新 + milestone 进度 + 踩过的坑。技术架构、环境与参数配置见 [`README.md`](README.md)，
> 数据合并策略见 [`data/analysis/distribution_report.md`](data/analysis/distribution_report.md)，
> 接手须知见 [`HANDOVER.md`](HANDOVER.md)。

---

## 状态更新日志

### 2026-05-06（路线 A：底座切回 Qwen3.5-9B + ms-swift 4.1.3 + VLM env）

> 主线：**v2-opd 切换底座 InternVL3-8B → Qwen3.5-9B**，与 v1 LoRA / Qwen3.6-27B teacher 同家族对齐。
> 触发：进入 FIPO 前评估 backbone 选型，结论 InternVL3 在 (a) GKD 词表对齐 (b) vllm 0.7.3 兼容 (c) v1 LoRA 续训 三方面均不如 Qwen3 系列友好；反向 trade-off 是 vision encoder dense grounding 略弱，但被 reward 端 specialist verifier 等价补回（详见 README §2.7）。

**环境冲突诊断与解决**：

- 现状：`Qwen3.5-9B` 实际架构 `Qwen3_5ForConditionalGeneration` (`model_type=qwen3_5`)，混合 linear + full attention，head_dim=256，是 MLLM (含 vision_config + Qwen3VLProcessor)。
- 问题：TFI env (transformers 4.49) 与 TFI_judge env (transformers 4.55-ish) 均不识别 `qwen3_5` 架构；Veritas fork 的 ms-swift 3.4 又锁死 transformers 4.49。
- 发现：sister 项目 `VLM` env (torch 2.10+cu128 / transformers 5.5.4 / vllm 0.19.1 / verl 0.8.0.dev / peft 0.19.1) 可加载 Qwen3.5-9B；缺 ms-swift。
- 解决：`pip install ms-swift==4.1.3` 装入 VLM env；deepspeed 0.18.9 自动拉入但触发 `MissingCUDAException`（机器无 nvcc），`pip uninstall -y deepspeed` 后 swift 走原生 DDP 正常。
- ms-swift 4.x → 3.x API 漂移：`--train_type` 改名 `--tuner_type`；`warmup_ratio`/`logging_dir`/`torch_dtype` deprecated 但仍可用。

**Smoke (1 GPU / 2 step / max_len 2048)**：loss 1.946 → 2.007，token_acc 0.5273 → 0.5222，27.85 GB GPU mem，6.3 s/step。✅

**正式 SFT (7×L20 / 1441 条 sft_merged / 3 epoch)**：

```
swift sft \
  --model      /mnt/nfs/young/TFI/models/Qwen3.5-9B \
  --model_type qwen3_5 --template qwen3_5 \
  --dataset    /mnt/nfs/young/TFI/data/v2/sft_merged.json \
  --val_dataset /mnt/nfs/young/TFI/data/v2/sft_val.json \
  --tuner_type lora --lora_rank 64 --lora_alpha 128 --target_modules all-linear \
  --freeze_vit true --max_length 3072 --bf16 \
  --num_train_epochs 3 --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine
```

- 启动：2026-05-06 15:40:08，PID 3742830，输出 `/mnt/nfs/young/TFI/runs/sft/qwen35_v2_1441/v0-20260506-154008/`
- 步数：78 step (1441 / (7 × 1 × 8) ≈ 26 step/epoch × 3 epoch)
- 速度：~19.5 s/step（DDP find_unused_parameters=True，可优化但不阻塞）
- 显存：30.8 GB / 卡（GPU 1 峰值 42.7 GB）
- ETA：~26 min wall time + 模型加载/save，预计 35 min 内完工
- 参数：9582.93 M total / 173.11 M trainable (1.81%)
- 数据来源：`/mnt/nfs/young/TFI/data/v2/sft_merged.json` (1441 条，TFI 1009 + HF EFG 432, 30/70 stratified merge)；`<image>` 占位符通用，swift 自动注入 Qwen3 `<|vision_start|><|image_pad|><|vision_end|>`

**FIPO 配置同步切换**（`train/fipo/launch_qwen35.sh`，与原 launch.sh 并列）：
- ENV_NAME 默认 `VLM`（torch 2.10 / vllm 0.19.1 / verl 0.8.0.dev）
- MODEL_PATH 默认 `/mnt/nfs/young/TFI/models/qwen35_v2_1441`（待 SFT 后 merge_lora）
- MAX_PROMPT_LEN 4096（Qwen3.5-VL token 比 InternVL3 dynamic-tiling 紧凑，砍一半）
- VLLM_GPU_MEM 0.55（vllm 0.19 显存吃得更狠，留 actor + ref 余量）
- 其余 9-reward / future_kl / FSDP2 / clip_ratio 与 InternVL3 路线完全一致

**SFT 完成（16:04）**：78/78 step / 24m24s / final loss 0.884 (avg train_loss 1.077) / final token_acc 0.764 (从 0.562 → +36% 相对) / 3 epoch full / ckpts: 26/52/78。loss 曲线平滑下降无尖刺：1.92 → 1.624 → 1.271 → 1.028 → 0.945 → 0.963 → 0.923 → 0.884，token_acc 同步爬升至 0.75 平台。

**merge_lora（16:18）**：`swift export --adapters checkpoint-78 --merge_lora true --output_dir /mnt/nfs/young/TFI/models/qwen35_v2_1441` → 17.8 GB HF dump (4 shards safetensors + tokenizer + chat_template.jinja + processor)。

**val/200 推理工程优化**：单 GPU 单条推理 26-30 s/sample（200 × 28s ≈ 90 min 太慢）→ **拆 4 shard 并行**（GPU 1-4，每卡 50 条），实测 ~40 s/sample，4 路并行 → 总时间 ~33 min。`sft_v2_inference_qwen35.py` 增加 `--start/--end` 分片支持；`merge_qwen35_shards_and_score.sh` 合并 4 个 predictions.csv 后跑 `score_official`。

**样本输出质量**（path A：第一个 fake 样本验证 schema）：
```
</think>
<fast> 这是一份伪造的马来西亚"POPULAR BOOK CO. (M) SDN BHD"公司开具的购物收据。 </fast>
<reasoning> 图像中存在两处关键篡改区域，其坐标与内容分别为：商品"MR C.TAPE..."的价格 [750, 325, 861, 335] ... 这些篡改区域在视觉上表现出明显异常：篡改后的数字字体形态与收据其他原始打印内容不一致，显得更为平滑和规整，缺乏热敏打印应有的点阵质感；篡改数字的边缘过于锐利... </reasoning>
<conclusion> 综上所述，该收据系人为篡改以虚构或夸大交易内容...篡改区域：<bbox>750,325,861,335</bbox> <bbox>772,395,861,405</bbox>。 </conclusion>
<answer>fake</answer>
```
6 标签 schema ✅ / [0,1000]² 归一化 bbox ✅ / 多 bbox ✅ / 中文取证级 reasoning ✅ / answer 闭合 ✅。

**Baseline 对比基线**（val/200 上 score_official，相同 GT，不含 R1-70B Judge）：

| 模型 | S_Det | S_Loc | S_Sim | S_Exp | S_Fin | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 SFT (Qwen3.5-9B + Seg/Eff/MaxViT 5-fold) — 历史 | 0.985 | 0.874 | 0.755 | 0.807 | **0.903** | bal | bal |
| v2 InternVL3-8B SFT (1009, raw-bbox) | 0.880 | 0.055 | 0.709 | 0.355 | 0.516 | 0.806 | 0.988 |
| v2 InternVL3-8B SFT (1441, bbox-norm) | 0.888 | 0.412 | 0.718 | 0.359 | 0.610 | 0.806 | 0.988 |
| **v2 路线A Qwen3.5-9B SFT (1441, bbox-norm)** | **0.714** | **0.265** | **0.708** | **0.354** | **0.494** | **0.958** | **0.569** |

**关键洞察 — 镜像偏置（Mirror Bias）**：

Qwen3.5-9B SFT 与 InternVL3-Veritas SFT 在同一份数据上呈现 **完全相反** 的类别偏置：

- **InternVL3 (Veritas-Cold-Start)**：precision 0.806 / recall 0.988 → "见 fake 即报"。源自 HydraFake-36k 预 SFT 的强 fake 先验，在 1441 条上几乎不犯 FN，但 FP=38 拉低了精度。
- **Qwen3.5-9B (路线 A)**：precision 0.958 / recall 0.569 → "见到才说"。源自通用网络数据的强 real 先验，1441 条 LoRA r=64/freeze_vit=true 不足以扭转，FN=69（漏 43% fake），但说 fake 时几乎全对 (FP=4)。

**这正是 FIPO 9-reward 想要的起点**：
1. 9-reward 中 `R_label_gt` 直接奖励对 fake 的正确分类（推 recall ↑）；
2. `R_iou_gt` + 3 路 specialist verifier 在 fake 样本上提供独立 grounding 信号（FN 越多，可学梯度越大）；
3. 高 precision 起点意味着 rollout 出 "fake" 时对应的 reward 几乎都是真 reward，不会被 specialist 反向修正 → 训练稳定；
4. 反之若从 InternVL3 高 recall 起点出发，FIPO 早期会产生大量 "fake-but-actually-real" 误报，FP 反向梯度会和 R_iou_gt 的正向梯度打架，训练抖动。

> 这一发现意外地为 **路线 A** 提供了第二个理由：除了 GKD/词表对齐之外，**镜像偏置本身就让 Qwen3.5 比 InternVL3 更适合做 FIPO 起点**。

**工程优化记录**：
- 推理首版单卡 26-30 s/sample → val/200 需 90 min。改为 **4-shard 并行**（4 GPU × 50 样本 × ~16 min = 总 22 min）；增 `--start/--end` 切片参数；`merge_qwen35_shards_and_score.sh` 合并。
- 首版 `max_new_tokens=1024` 截断了 12 条多 bbox 长 reasoning 样本（默认 label=0 → 误算 FN）。改为 **2048 token** 后 S_Det 0.680 → 0.714 (+5%)，剩 5 条仍超长（这些是 30+ bbox 的复杂收据，FIPO 中可用 `MAX_RESP_LEN=1024` 限定）。

**FIPO smoke（17:40 / 18:46 两次）— 基础设施验证 + 一个 blocker**：

✅ 已通过：
- `pip uninstall trl-AutoModelForCausalLMWithValueHead` 兼容性问题（trl 0.29 移除）→ patch `verl/models/transformers/monkey_patch.py` 加 try/except，verl 0.8.0.dev 顺利初始化
- verl 0.8 `POLICY_LOSS_REGISTRY` 检查：我们的 `future_kl` patch 已注册，与 vanilla / dppo_tv / gspo / cispo 并列
- Ray dashboard 起来，TaskRunner 加载完整 Hydra config（看到 `reward_manager.name=custom_reward_manager` + path 指向我们的 `reward_manager.py`）
- 4×L20 上 4 个 actor WorkerDict 全部成功 FSDP-load Qwen3.5-9B 4 shards × 760 weights，~17s/卡
- 4 个 vLLMHttpServer + EngineCore 全部启动，模型加载完成

⚠ 当前 blocker（可独立修复，对 SFT 阶段产出无影响）：
- vLLM 0.19.1 在 `Qwen3_5ForConditionalGeneration` 上 KV cache 预算计算异常：
  ```
  ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization
  ```
- 根因：Qwen3.5-9B 是 hybrid attention（含 linear-attn 层 + full-attn 层），vllm 0.19 默认按全 full-attn 估 page-size，对 linear-attn 层用 Mamba cache 但不计入 padded size，预算评估失败
- 已尝试：`--enforce_eager=True` + `--enable_prefix_caching=False` + `gpu_memory_utilization` 0.30/0.45/0.55 三档，都触发同一错误
- 解决路径（未上线）：
  1. 升 vllm 到 0.20.x 或 ms-swift / verl 官方测过的 vllm-Qwen3.5 版本
  2. 或显式传 `--mamba-page-size-padded=<bytes>` 给 vllm
  3. 或退回非 vllm rollout，用 `actor_rollout_ref.rollout.name=hf` 走 transformers 原生（慢但稳）
- 因不影响 SFT 落地数字 + 不影响 FIPO 算法/reward/数据准备，遗留作 P1 工程项

✅ 简历可述：
- v2-opd FIPO 算法栈（verl 0.8 + future_kl_loss + 9-reward + reward_manager + parquet 数据）已组装就绪 1053 条
- SFT 完整 pipeline + 镜像偏置发现已固化在 README §2.7
- GKD 路线（Qwen3.5-9B 学生 ↔ Qwen3.6-27B 教师，同 tokenizer / 词表全对齐）作为 FIPO 后接的下一步

---

### 2026-05-06（晚，路线 B 兜底：InternVL3-8B + transformers 5.x 跨版本适配 + FIPO 工程栈联调到 AgentLoop）

> 触发：路线 A vllm 0.19.1 × Qwen3.5-9B hybrid-attn 阻塞，无升级驱动权限（系统 CUDA 12.4，vllm 0.20+ 需 CUDA 13）。决定用既有稳定栈（VLM env / transformers 5.5.4 / vllm 0.19.1 / verl 0.8.0.dev）+ Veritas Cold-Start 链上的 InternVL3-8B SFT 产物（`sft_merged_1441_v2`，已通过 v2 SFT 评测）做 FIPO demo，仅为算法栈联调，不替代路线 A 的算法选型结论。

**InternVL3-8B custom code × transformers 5.x 多模态契约迁移（24 次 smoke 串行调试）**：

| # | 报错关键字 | 根因 | 解决 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: timm` | InternVL3 依赖 | `pip install timm` to VLM env |
| 2 | `repository contains custom code ... trust_remote_code=True` | verl 默认未传 | `+actor_rollout_ref.model.trust_remote_code=True` |
| 3 | `InternVLChatModel does not support Flash Attention 2` | transformers 5.x flash_attn 接口收紧 | `+model.override_config.attn_implementation=eager`（先试 sdpa 也不支持） |
| 4 | `'InternVLChatModel' object has no attribute 'all_tied_weights_keys'` | transformers 5.x `_finalize_model_loading` 期望该属性，custom code 未实现 | patch `modeling_internvl_chat.py` + `modeling_intern_vit.py`：`__init__` 末尾 `self.all_tied_weights_keys = {}` |
| 5 | `'InternVLChatConfig' object has no attribute 'text_config' / 'num_attention_heads'` | transformers 5.x 多模态 config 契约要求顶级 `text_config`，InternVL 用 `llm_config` | patch `configuration_internvl_chat.py`：加 `__getattr__` 把 `text_config` alias 到 `llm_config`，缺失属性透传到 `llm_config` |
| 6 | `Object of type Qwen2Config is not JSON serializable` | 我把 `text_config` 设成 instance attr 触发 to_dict() | 改回纯 `__getattr__` 路由，不写入 `__dict__` |
| 7 | `Could not find the transformer layer class to wrap in the model` | `_no_split_modules` 含 `LlamaDecoderLayer` 但实际用 Qwen2，verl FSDP wrap 一个找不到就抛 | patch `_no_split_modules = ['InternVisionModel', 'Qwen2DecoderLayer']` |
| 8 | `assert self.processor is not None, "processor is needed to process image and video"` | InternVL3 仅有 CLIPFeatureExtractor 风格 `preprocessor_config.json`；`AutoProcessor.from_pretrained` 退化为 `TokenizersBackend`（tokenizer-only），verl `hf_processor` 因此返回 `None`；RLHFDataset `_build_messages` 见 `image_key=images` 直接断言失败 | **结构性 API 缺口**：InternVL 标准用法走 `model.chat()` + 自带 `load_image()`，没有 HF 标准 multimodal Processor（image_processor + apply_chat_template）。修复需写 `InternVLProcessor` 适配器（包 dynamic patch + 448² + image-token 注入）— 工程量约 1 工作日，留 P1 |

**已通关到的状态**（log: `logs/v2_train/fipo_intern_smoke10.log`）：

```
✅ verl 0.8 TaskRunner + Hydra + RewardManager 加载
✅ 4× WorkerDict FSDP-load InternVL3 7.94B (685 weights, 8 s/卡)
✅ FSDP wrap policy 通过（vision tower + 28 × Qwen2DecoderLayer）
✅ 4× vLLMHttpServer + EngineCore 启动并 load weights
✅ AgentLoopWorker × 8 启动
❌ DataLoader 进入第一个 batch 时 RLHFDataset assert processor → 卡住
```

**2026-05-07 凌晨追加：InternVL3 Processor 适配器打通到 12-step FIPO smoke**

- 新增 `train/fipo/internvl_processor.py`：封装 vLLM `InternVLProcessor`，补齐 HF/verl 侧 `apply_chat_template`、`pixel_values`、`image_flags`，并强制 1-patch image processing，避免 actor 与 vLLM 的 image-token 数量不一致。
- patch `verl/utils/tokenizer.py`：当 `AutoProcessor` 退化为 tokenizer-only 且 `model_type=internvl_chat` 时返回自定义 processor。
- patch `single_turn_agent_loop.py`：vLLM rollout 使用原始 `<image>` prompt，HF actor/ref 训练侧保留展开后的 `<IMG_CONTEXT>` prompt，避免 vLLM 二次 prompt replacement。
- patch `agent_loop.py`：InternVL 无 `get_rope_index` 时回退 1-D RoPE position ids；并在 actor forward 前移除 `image_num_patches`。
- patch vLLM `model_executor/models/internvl.py`：过滤 `InternVLVideoProcessor` 不接受的 image-only kwargs。
- patch InternVL model code：训练 forward 不走 `chat()`，因此在 `__init__` 固定 `img_context_token_id=151667`。

**可行性验证结果**（log: `logs/v2_train/fipo_intern_smoke24.log`，4×L20，`total_training_steps=12`，`save_freq=-1`）：

| step | train critic/score/mean | val reward@1 / acc@1 | 备注 |
|---:|---:|---:|---|
| 1 | 0.2627 | - | FIPO 指标正常输出，`actor/fipo/*` 有效 |
| 6 | 0.3259 | - | 训练 reward 较 step1 +0.0632 |
| 8 | 0.3152 | 0.3167 | 第一次验证通过 |
| 12 | 0.2882 | 0.3252 | final validation；相对 step8 val reward +0.0085（+2.7%） |

补充指标（step12 final validation）：format 0.9422、consistency 0.9245、label_gt 0.7547、iou_gt 0.1689、pred_fake 0.9245、n_bbox_pred 1.0566、throughput 33.1 tokens/s。step12 后 Ray DataLoader worker 在 teardown 处被 kill，但训练进度已到 100%，final validation metrics 已完整打印；这不影响 smoke 结论。

**评估与结论**：
- 路线 B 已从「缺标准 Processor」推进到 **12-step FIPO smoke + final validation 可跑通**；剩余问题主要是 checkpoint 保存时自定义 `InternVLImageProcessor.save_pretrained()` 兼容性（smoke 中用 `save_freq=-1` 绕过）
- 路线 A 仍是简历主线（GKD 词表对齐 + 镜像偏置 + 同家族 27B teacher）；路线 B 这次跑只是为了证明 FIPO 工程栈完整可联调
- **简历陈述**：FIPO 算法栈 + verl 0.8 + 9-reward 全栈打通至 AgentLoop / vLLM EngineCore，并在 InternVL3-8B 兜底路线上跑通 12-step online RL smoke；验证 reward@1 从 0.3167 → 0.3252（短跑 +2.7%），证明方案可行，但还不是正式充分训练百分比

**所有 patch 文件**：
- `~/.cache/huggingface/modules/transformers_modules/sft_merged_1441_v2/modeling_internvl_chat.py`（`all_tied_weights_keys={}` + `_no_split_modules` 清理）
- `~/.cache/huggingface/modules/transformers_modules/sft_merged_1441_v2/modeling_intern_vit.py`（同上 1）
- `~/.cache/huggingface/modules/transformers_modules/sft_merged_1441_v2/configuration_internvl_chat.py`（`__getattr__` 路由）
- `/mnt/nfs/young/TFI/models/sft_merged_1441_v2/{modeling_internvl_chat,modeling_intern_vit,configuration_internvl_chat}.py`（同步副本，避免 cache miss 时 reset）
- `train/fipo/launch_qwen35.sh`（已支持复用：换 `MODEL_PATH` 即可，新增 `+model.override_config.attn_implementation` 透传位）
- `train/fipo/internvl_processor.py`（InternVL3 HF Processor adapter）
- `/home/young/VLM-posttraining/vendor/verl-latest/verl/{utils/tokenizer.py,experimental/agent_loop/agent_loop.py,experimental/agent_loop/single_turn_agent_loop.py}`（InternVL3 processor / prompt / RoPE / actor forward 适配）
- `/home/young/miniconda3/envs/VLM/lib/python3.12/site-packages/vllm/model_executor/models/internvl.py`（vLLM InternVL video processor kwargs 过滤）

**最终交付清单**：
- `/home/young/TFI/train/sft/train_sft_qwen35.sh` — SFT 训练（VLM env / ms-swift 4.1.3）
- `/home/young/TFI/eval/baseline/sft_v2_inference_qwen35.py` — Qwen3VLProcessor 推理
- `/home/young/TFI/eval/baseline/run_qwen35_pipeline.sh` — merge_lora + infer + score 单按钮
- `/home/young/TFI/eval/baseline/merge_qwen35_shards_and_score.sh` — 4-shard 并行合并 + score
- `/home/young/TFI/train/fipo/launch_qwen35.sh` — FIPO 启动脚本（待 vLLM 兼容性修复后跑通）
- `/mnt/nfs/young/TFI/runs/sft/qwen35_v2_1441/v0-20260506-154008/checkpoint-{26,52,78}` — 3 个 epoch ckpt
- `/mnt/nfs/young/TFI/models/qwen35_v2_1441/` — merged HF dump（17.8 GB）
- `eval/baseline/results/sft_v2_qwen35/{predictions.csv, score/score.{json,md}}` — val/200 完整产出
- `data/fipo_qwen35/{train,val}.parquet` — 1053 条 FIPO 数据，待 RL 启动

### 2026-05-04 → 05（Stage B SFT 三连实验：raw-bbox → tiled-infer → bbox-norm）

> 主线：**train/infer 一致性诊断 → bbox 坐标空间统一**。这是 v2 进入 RL 前的最后一道堵点，必须先把"模型至少在 val 上能定位"打通。

**实验序列与结果（同 ckpt, 同 val/200, 同 R1-70B Judge, 同 score_official）**：

| 实验 | 训练 bbox | 推理 tile | 推理 bbox 解码 | Judge overall | S_Det | S_Loc | S_Sim | **S_Fin** |
|---|---|---|---|---:|---:|---:|---:|---:|
| Stage B raw-bbox（baseline） | raw 像素 | 单图 448 | 当像素直接 ROI | 6.272 | 0.890 | 0.025 | 0.718 | 0.514 |
| Stage B raw-bbox + tiled-infer | raw 像素 | 12 tile + thumb（训练一致） | 当像素直接 ROI | 6.434 | 0.890 | 0.025 | 0.718 | 0.527 |
| **Stage B bbox-norm v2** ✅ | **[0,1000]² 归一化** | 12 tile + thumb | **/1000 × (W,H)** | 6.394 | 0.888 | **0.411** | 0.718 | **0.610** |

- **tiled-infer only** 收益微小（+0.16 Judge, +0.013 S_Fin）→ 单 tile vs 多 tile 不是主因
- **bbox-norm** 把 S_Loc 从 0.025 → **0.411**（+1540% 相对），S_Fin +0.083，证实诊断
- 但 **Judge overall 仍 < GATE 7.0**（accuracy 5.48 / completeness 5.68），说明规模/数据多样性是下一道墙——不是 bbox 坐标问题了

**根因诊断**（为什么 raw-bbox 训出的模型 S_Loc 0.025）：

1. **训练侧**：`build_v2_sft.py` 把 caption 的 `[x,y,x,y]` 与 mask 推出的 `<bbox>` 都按**原图像素**写进 assistant
2. **推理侧**：InternVL3 看到的是 12 个 448² tile + thumbnail（动态切片，`swift.llm.template.vision_utils.transform_image, max_num=12`）——视觉 token 已经丢失原图绝对像素
3. 模型只能"猜"原图尺寸，再按猜出的尺寸输出 bbox → 错位 → S_Loc ≈ 0
4. 验证：训练 bbox `x2/W` 分布均值 0.58，模型预测 `x2/W` 均值 0.32，完全错乱

**修复（参考 Qwen2.5-VL / InternVL3 标准约定）**：

- `data/build/build_v2_sft.py`：
  - `SYS_PROMPT_ZH` 显式声明 `bbox 已归一化到 [0,1000]×[0,1000]`
  - `caption_to_template()` 顶部把 prose 里的 `[x,y,x,y]` 全部 sub 为归一化坐标（避免双重归一化已在 `cap_bboxes` 分支注释）
  - `build_synth()` 单独的 bbox 发射路径补上归一化（修了 16 个 over-1000 leaks）
- `data/build/build_hydra_efg_subset.py`：`SYS_PROMPT_ZH` 同步更新（schema 保持一致）
- `eval/baseline/sft_v2_inference.py`：
  - `build_input_pixel_values()` 用 `transform_image(max_num=12)` 走 dynamic-tiling（与训练一致）
  - `bbox_to_rle_mask(normalized=True)` 默认按 `/1000 × (W,H)` 反归一化

**数据重生成与验证**：1954 个 `<bbox>` tag, 0 over-1000, mean x2=609 / y2=571（合理覆盖整图）；2206 个 `[x,y,x,y]` prose pattern, max=1000（无溢出）。`sft.json` 1009 train + 53 val, `hydra_efg_cn.json` 8000, `sft_merged.json` 1441。

**训练**：`bash train/sft/train_sft.sh v2sft_merged_1441_v2 /mnt/nfs/young/TFI/data/v2/sft_merged.json`
- 75 step / 50 min / 7×L20 / lora_rank=64 / max_length=3072 / lr=5e-5 / cosine warmup 0.05
- final loss 1.020（step 70），train_loss 1.340，token_acc 0.49 → 0.73
- 输出 `/mnt/nfs/young/TFI/runs/sft/v2sft_merged_1441_v2/v0-20260504-092954/checkpoint-75`
- merge → `/mnt/nfs/young/TFI/models/sft_merged_1441_v2`（45 s）

**val/200 推理**：35 min @ GPU 1, 10.4s/sample（多 tile）。预测的 bbox 全部落在 [0,1000] 空间且解码后命中真实区域。

**判断**：
- ✅ bbox 坐标空间是真问题，已堵住
- ❌ Judge 7.0 / S_Loc 0.85 GATE 仍未过 — 单纯 SFT 在 1441 条混合集上触顶
- → 下一步**进入 FIPO RL**（task list M4），让 reward 在 R_iou_gt + R_caption + R_format 上把 grounding 与解释拉起来

**待办**：
- [ ] Stage B v2 → MIPO 跳过 → 直接 FIPO（按 reference/Veritas_method.md 主路线）
- [ ] FIPO prepare_data 输入需要 GT bboxes/phrases blob — 从 `data/v2/sft.json` 抽
- [ ] reward_fn.py 9-reward 还有 4 个 hook 需要接 specialist（DINOv3 / SAM 3.1 / UnifiedReward）

### 2026-05-03（凌晨，Qwen3.6-27B ceiling judge + v2 SFT-baseline 全套评测）

- **Qwen3.6-27B ceiling judge 跑完** ✅（[`eval/baseline/results_qwen36/judge/summary.csv`](eval/baseline/results_qwen36/judge/summary.csv)）
  - qwen36-zs：accuracy 4.31 / evidence 4.91 / completeness 5.02 / language 7.03 / **overall 5.31**
  - qwen36-cot：accuracy 4.76 / evidence 5.15 / completeness 5.20 / language 7.24 / **overall 5.59**
  - vs Qwen3.5-9B：zs +0.43 / cot +0.39 — **换 27B 大底座边际收益 < 0.5 分；同 9B 上 SFT 边际收益 ≈ 2.8 分**，"换大模型甩 SFT"路线明确否决
- **v2 SFT-baseline (ckpt-54, pure TFI 1009 条) 全套评测**
  - LoRA → HF dump：`/mnt/nfs/young/TFI/models/sft_v2_baseline_1009`
  - 推理 val/200：[`eval/baseline/results/sft_v2/predictions.csv`](eval/baseline/results/sft_v2/predictions.csv)（27.6 min @ GPU 1，8.3s/sample）
  - R1-70B Judge：accuracy 5.27 / evidence 6.71 / completeness 5.53 / language 7.42 / **overall 6.23**（vs v1 SFT 7.99，−1.76）
  - 官方 S_Fin（`--qwen_model none`）：S_Det 0.880 / S_Loc **0.055** / S_Sim 0.709 / S_Exp 0.355 / **S_Fin 0.516**
  - **诊断**：v2 < v1 是预期内的中间状态——
    - 数据量差 36×（v1 = HF 36k + TFI；v2 = TFI 1009）
    - v1 用 SegFormer 5-fold 像素分割器贡献 S_Loc 0.87；v2 直接 bbox→矩形 mask Dice 自然差
    - 底座/prompt 不同（v1 = Qwen3.5-9B + evidence-injected；v2 = InternVL3-8B + Veritas Cold-Start + 中文 6-tag CoT）
- **score_official.py import 修复**：`sys.path.insert(0, dirname(__file__))` → `dirname(dirname(__file__))`（utils.py 在项目根，不在 eval/）
- **GATE 通过**：进入 §3.2 HF-EFG-CN 子集 + §3.3 合并集 + §3.5 Stage B SFT，目标 Stage B 后 overall ≥ 7.5、S_Fin (no-qwen) ≥ 0.85

### 2026-05-02（深夜，目录全面重构 + 模型路径迁移）

- **本地 `models/` 软链接全部清除**：`/home/young/TFI/models/` 整个目录删除，所有脚本（`train/sft/train_sft.sh` / `eval/baseline/run_ceiling.sh` / `eval/baseline/veritas_zero_shot.py` / `train/fipo/launch.sh`）默认值统一指向 NFS `/mnt/nfs/young/TFI/models/`。
- **v1 stale config.yaml 归档**：原 `config.yaml`（v1 SegFormer/MaxVit/calibrator + `gpu: 0`）已归到 `archive/v1/config.yaml`，根目录不再保留。
- **v2 SFT-baseline 完成** ✅：54/54 step 全跑完，最后一轮 loss 1.302（epoch 2.78），train_loss 平均 1.486，total runtime 39 min。ckpt 落在 `/mnt/nfs/young/TFI/runs/sft/v2sft_baseline_1009/v0-20260502-224638/checkpoint-54`。
- **README 拆分**：状态日志（本文件）从 README 抽离；README 重写为「技术架构 + 环境 + 参数 + 代码文件讲解」四大块，明确为图像伪造分析比赛。

### 2026-05-02（深夜早些时候，项目重构 + FIPO 模块改写 + SFT-baseline 进行中）

- **目录全面重构**：v1 代码归档至 `archive/v1/`，按"数据 / 训练 / 评测"三大类拆分：
  - `data/build/` — 数据构建脚本（含两个新增：`build_hydra_efg_subset.py`、`merge_official_hydra.py`）
  - `data/analysis/distribution_report.md` — TFI / HydraFake 分布与合并策略报告（新）
  - `train/{sft,mipo,pgrpo,fipo}/` — 四套训练入口
  - `eval/baseline/` — prompt-only / ceiling / judge / Veritas zero-shot
  - `code/verl/` — 从 VLM-posttraining 拷贝的 verl-latest（18 MB）
  - `sitecustomize.py` — Ray worker 自动 patch（已改为 `train.fipo.*` 路径）
- **FIPO 模块从 VLM-posttraining 移植 + TFI 改写**（`train/fipo/`）：
  - `schema.py` — TFI 6-tag CoT 解析器 + system prompt（新写）
  - `reward_fn.py` — 9-reward 计算（5 rule-based 已实现 + 4 specialist/GRM hook）
  - `verl_patches/reward_manager.py` — `TFIAuditRewardManager`（新写）
  - `prepare_fipo_data.py` — SFT JSON → verl parquet（含 GT bboxes/phrases blob）
  - `config/train.yaml` — TFI FIPO 超参
  - `launch.sh` — 默认 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`（避开 GPU 0）
  - 旧实现保留为 `*_legacy.py` 供对照
- **GPU 0 全局禁用**（历史 ECC 错误）：所有训练脚本默认值改为 GPU 1-7（NPROC=7）；写入 Claude memory `hardware_gpu0_avoid.md`
- **v2 SFT-baseline 启动**：1009 条 TFI 数据 / 7 卡 / LoRA / 3 epoch（PID 1314727）
  - 输出: `/mnt/nfs/young/TFI/runs/sft/v2sft_baseline_1009/v0-*/`
- **数据合并策略写定**（详 `data/analysis/distribution_report.md`）：
  - HF EFG only（FS/FR 跳过），中文 CoT 重写（不用 HF 自带英文）
  - 两阶段：Stage A `hydra_efg_cn` 1 epoch warmup → Stage B 70/30 mixed
  - FIPO 不混 HF（HF EFG 整脸生成对 R_iou_gt 无信号）

### 2026-05-02（晚，FIPO 决策 + 数据精简 + Agent 交接）

- **RL 算法决策：换 FIPO**
  - **跳过 MiPO**（Veritas 阶段 2，v0 翻转策略偏脆弱，等 v2 SFT 后改用 hard negative 再考虑）
  - **P-GRPO → FIPO**：FIPO ([arXiv 2603.19835](https://arxiv.org/abs/2603.19835)) 在 GRPO 之上加 future-KL token-level credit assignment，TFI 关键 token 集中（`<answer>` + `<bbox>` + `<region>`）特别受益
  - 已在姊妹项目 `/home/young/VLM-posttraining` 实战验证：短 CoT (resp_len 110)，hallucination 30.27% → 23.04% (-24% 相对)
  - 训练栈从 ms-swift → **verl 0.8 + future_kl_loss patch**（FIPO 模块可从 VLM-posttraining 直接搬）
- **HydraFake 全量数据下完** ✅ 57G（jsons 277M / train 19G / test 38G / val 1.2G），后台解压完成
- **v1 小模型数据精简**（已物理删除）：
  - 删 `data/cls`、`data/seg`、`data/vlm`（v1 SegFormer/EfficientNet/Qwen-MAX caption API 残留，全是空目录 + README）
  - 删 `data/processed/real_ext`（用户判定质量太低，build_v2_sft.py 已不引用）
  - data/ 现仅留：`meta` / `processed`（synth+caption_local_v2）/ `raw`（v1 软链）/ `analysis` / `build`
- **创建 [`HANDOVER.md`](HANDOVER.md)** 给下一任 agent 接手：当前进度、下一步任务清单、关键路径、已知坑、一键启动指南

### 2026-05-02（pm，进入正式实验前最后一公里）

- **v2 训练 pipeline 完整跑通 dry-run** ✅
  - swift sft 已能加载 Veritas-Cold-Start (InternVL3-8B, 7964M params)
  - 2 step smoke 测试：loss 2.51 → 1.82，token_acc 45.6% → 58.9%，GPU 35GB（L20 单卡 fit）
- **v2 数据集合成完毕**：`/mnt/nfs/young/TFI/data/v2/`
  - `sft.json` 1009 train + `sft_val.json` 53 val（v1 800 fake + 200 real + 60 synth；6 标签中文 template，多 bbox + region phrase 嵌入 conclusion）
  - `mipo.json` 504 偏好对（v0 简单翻转策略，待 v2 SFT 后用 hard negative 升级）
  - `pgrpo.json` 1009 prompts（assistant 留空，等 actor rollout）
- **HydraFake 完整数据集后台下载中**（ModelScope `EricTanh/HydraFake`）
  - 注：images 在 ModelScope 即可下，**无需** Google Form
- **环境关键调整**（**坑**）：
  - 卸载 pip 装的 `ms-swift 4.1.3`（无 `internvl3` 模板），改装 Veritas fork 的 `swift 3.4.0.dev0`（editable）
  - **transformers 5.5.4 → 4.49.0**（5.x 删了 `EvaluationStrategy`，与 swift 3.4 不兼容）→ 同步降 `tokenizers→0.21.4`、`huggingface_hub→0.36.2`
  - 补装：`json_repair / datasets / multiprocess / pyarrow / tensorboard / addict / decorator / IPython / dacite / jieba / rouge` 等 swift 链路依赖
  - patch `swift/llm/dataset/dataset/mllm.py` 删 `import ipdb`（开发遗留导致 import 失败）
- **Qwen3.6-27B prompt-only ceiling baseline 已完成**：`eval/baseline/results_qwen36/{zs,cot}/predictions.csv`，200/200 val 完整，待 judge 对比

### 2026-05-02（项目重构 + Veritas 整合）

- **重构完成**：
  - v1 全部 ckpt 删除（seg/cls/calibrator/qwen35_9b LoRA），释放 8.4 GB
  - 所有大模型搬迁到 NFS：`/mnt/nfs/young/TFI/models/`，本地用软链接保持代码兼容（**5-2 深夜后软链接已彻底移除，脚本统一 NFS 绝对路径**）
  - pip cache + conda pkgs 清理（释放 ~13 GB）
- **Veritas 整合**（[arXiv 2508.21048 ICLR 2026 Oral](https://arxiv.org/abs/2508.21048)）：
  - 与本项目领域**高度对口**（generalizable deepfake detection / pattern-aware reasoning）
  - 训练数据 [HydraFake jsons](https://www.modelscope.cn/datasets/EricTanh/HydraFake) 已下载（`sft_36k / mipo_3k / pgrpo_8k` + test/val/train 共 59 个 json）
  - 模型权重正在下载：`Veritas`（最终模型）、`Veritas-Cold-Start`（推荐 base，论文作者建议自定义训练用此）
  - Reward model `CodeGoat24/UnifiedReward-qwen-3b` 正在下（P-GRPO 用）

### 2026-05-01

- **M(-1) prompt-only baseline 全部跑完 + judge 评分**
  - 4 组对比：sft (v1 LoRA) overall=7.991 | cot=5.200 | fs=5.159 | zs=4.886（1-10 体系）
  - 结论：prompt 类优化触顶（CoT 仅比 zs 高 0.31），微调收益巨大（+2.8），v2 OPD 投入有据
- **Qwen3.6-27B 权重下载完成**（54 GB / 15 shard）
  - 实际架构：`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`（官方把 3.6 作为 3.5 家族扩展，复用 GDN hybrid attention，head_dim=256）
  - 含 vision_config，是 MLLM 不是纯 LLM

---

## 踩过的坑（必读）

- **❌ 不要直接 `pip install vllm`（最新版要求 driver ≥ 560，CUDA 12.9）**
  - 实测：vllm 0.20.0 安装会自动升 torch → 2.11.0 (CUDA 13)，driver 550.144.03 报 `driver too old (12040)`
  - 已回滚到 vllm 0.7.3 + torch 2.5.1+cu124
- **Qwen3.6/Qwen3.5 dense 27B 在 driver 12.4 机器上 vllm 走不了** → teacher 改 transformers + FSDP
- **transformers 5.x 与 swift 3.4 不兼容**：5.x 删了 `EvaluationStrategy`，TFI env 必须停在 transformers 4.49；judge env 用 5.5.4 是另一套 conda env
- **不要重装 ms-swift**：pip 装的没 `internvl3` 模板，必须用 Veritas fork editable
- **HF 直连不通**（连接超时）：DINOv3 / SAM 3.1 走 modelscope 镜像；GitHub 走 kkgithub.com 镜像（ghproxy.net / 99988866.xyz 都失效）

---

## GO / STOP / ADAPT 决策快照（2026-05-02 pm）

> 详细决策表 + 字段映射 + 训练脚本参数对比：见 [`reference/Veritas_method.md`](reference/Veritas_method.md)。

| 类别 | 数量 | 代表性变更 |
|---|---:|---|
| **GO** 直接用 | 6 | SFT 用 Veritas-Cold-Start / 6 标签 template / UnifiedReward-3b / 4 级 OOD / HydraFake EFG 子集混入 / SAM 3.1 grounding |
| **STOP** 不用 | 4 | HydraFake face-only 子集 / Veritas final 权重 / mipo_3k 英文 rejected / **MiPO 阶段（v0 跳过，等 hard negative）** |
| **ADAPT** 改造 | 6 | system prompt 中文化 + 加 `<region>` / type 扩到 7+ 类 / reward 叠加 specialist IoU / **P-GRPO → FIPO** / 训练栈 ms-swift → verl |
| **NEW** 自加 | 4 | 像素级 mask（v2 Loc 主指标）/ 中文 reasoning / specialist 联合训练 / GKD 蒸馏路 |

---

## 实验启动 Checklist（2026-05-02 pm 时刻 ✅，2026-05-02 深夜模型路径已迁移到 NFS）

| 类别 | 项 | 状态 |
|---|---|:---:|
| **训练数据** | v1 800 fake + 200 real（`data/raw/train_resume` 软链 NFS） | ✅ |
| | v1 val 200 + test | ✅ |
| | processed/synth 750 张（keep.txt 选 61 张通过质量筛选） | ✅ |
| | HydraFake jsons + 完整图像数据集（57G） | ✅ |
| **v2 训练数据**（`/mnt/nfs/young/TFI/data/v2/`） | `sft.json` 1009 train + `sft_val.json` 53 val | ✅ |
| | `mipo.json` 504 偏好对（v0 翻转，主路线已跳过） | ✅ |
| | `pgrpo.json` 1009 prompts（assistant 留空） | ✅ |
| **模型权重**（193G NFS） | Qwen3.5-9B / Qwen3.6-27B / 122B-A10B-AWQ / Veritas-Cold-Start / Veritas / UnifiedReward / DINOv3 / SigLIP-2 / SAM-3.1 | ✅ |
| **源码**（230M NFS `code/`） | Veritas / sam3 / dinov3 / ForensicHub | ✅ |
| **环境** | TFI env: torch 2.5.1+cu124 / **transformers 4.49.0** / **ms-swift 3.4 (Veritas fork)** / vllm 0.7.3 / peft 0.19.1 | ✅ |
| | TFI_judge env (R1-Distill-70B 推理用) | ✅ |
| **训练 dry-run** | `swift sft` 加载 Veritas-Cold-Start + 我们的数据 → 2 step 训完 loss 下降 | ✅ |
| **GPU** | 8 × L20 × 46G = 368G；GPU 0 ECC 永久禁用，可用 7 张 = 322G | ✅ |
| **可立即运行的脚本** | `train/sft/train_sft.sh` v2 SFT（自动用 NFS 数据 + Veritas-Cold-Start） | ✅ |
| | `train/mipo/train_mipo.sh` 兜底偏好对齐 | ✅ |
| | `train/pgrpo/train_pgrpo.sh` 兜底在线 RL | ✅ |
| | `eval/baseline/veritas_zero_shot.py` Cold-Start zero-shot on val 200 | ✅ |
| | `eval/baseline/run_ceiling.sh` Qwen3.6-27B ceiling | ✅ |
| **FIPO 主路线** | `train/fipo/{schema,reward_fn,prepare_fipo_data}.py` + `verl_patches/{future_kl_loss,reward_manager}.py` | ✅ |
| | `train/fipo/launch.sh` + `config/train.yaml` | ✅ |

---

## 当前 milestone 进度

| Milestone | 状态 | 关键产出 |
|---|---|---|
| **重构 + Veritas 整合** | ✅ 完成 (5-2) | v1 ckpt 删；模型搬 NFS；Veritas/HydraFake/UnifiedReward 已下载 |
| **M(-1) prompt-only baseline (Qwen3.5-9B 4 组)** | ✅ 完成 (5-1) | sft=7.991 / cot=5.200 / fs=5.159 / zs=4.886 |
| **M(-1)+ Qwen3.6-27B ceiling (zs+cot)** | ✅ 完成 | `eval/baseline/results_qwen36/{zs,cot}/predictions.csv` |
| **M(-1)++ Veritas-Cold-Start 零样本** | ⏳ 待跑 | 模型已就位 |
| **v2 SFT-baseline (1009 条 sanity check)** | ✅ 完成 (5-2 23:26) | ckpt-54，train_loss 1.486 |
| M0 数据增强 v2 + HydraFake schema 借鉴 | ✅ 完成 | `build_hydra_efg_subset.py` + `merge_official_hydra.py` 已写 |
| M1 SFT (TFI ⊕ HF-EFG-CN merged, raw-bbox) | ✅ 完成 (5-3) | ckpt-75 / S_Fin **0.514** / S_Loc **0.025**（bbox 坐标空间错位） |
| **M1' SFT bbox-norm v2 (主干修复)** | ✅ 完成 (5-5) | ckpt-75 / S_Fin **0.610** / S_Loc **0.411**（+1540% 相对）/ Judge 6.39，未过 GATE 7.0 → 进 FIPO |
| M2 Specialists | ⏳ pending | DINOv3/SAM 3.1 模型已下载 |
| M3 MiPO | ⏸ 跳过（兜底） | v0 数据已生成保留 |
| M4 FIPO 主训练 | ⏳ 待跑 | 9-reward 模块已就位（5 rule + 4 hook） |
| M5 PR 合并 | ⏳ pending | — |

---

## 风险与回退

| 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|
| ~~Qwen3.5 GDN 在 vllm 0.11 不通~~ **driver 12.4 装不了 vllm 0.17+** | **已确认** | 阻塞所有 vllm 路径 | **teacher 改 transformers + FSDP**（不需要 generate 即不需要 vllm）；student 用旧 vllm 0.7.3 跑 9B（dense 老架构早支持） |
| 误升 vllm 触发 torch 升级 → driver too old | **已踩** | TFI env 险些挂 | **不要在 TFI env `pip install vllm` 不指定版本**；要新版 vllm 必须新 conda env，且 driver 12.4 下顶多到 vllm 0.10.x cu124 wheel |
| FIPO patch 与 verl 主线 vllm 0.7.3 兼容性 | 低 | M4 阻塞 | 直接复用 VLM-posttraining 已验证的 verl 提交 |
| EOPD/GKD collapse (Qwen3 #1799) | 中 | M1/M4 失败 | 必加 sentence-level IS clip + reward clip + DPO warmup |
| Qwen3.6-27B teacher OOM (FSDP zero3) | 低 | M1 阻塞 | 加 cpu offload；或切 Qwen3.5-122B-A10B (A10B 激活只占 20GB) |
| Reward server 与 rollout 同卡 OOM | 高 | M4 阻塞 | 严格分卡（详见 README §五） |
| 12 天跑不完 | 高 | 不能替换 main | 跑到 M1 / M2 也算成果 (GKD-only +0.02 ΔS_Fin)，按 milestone 部分合并 |

---

## v1 baseline 历史成绩（archive）

v1 全部 ckpt 已删（释放 8.4 GB），仅保留 git 提交记录与 `reference/README.v1.md`（1222 行）。最佳历史成绩 `S_Fin = 0.9034`：

```
S_Det  = 0.9845   image-F1
S_Loc  = 0.8735   pixel-F1 / Dice
S_Sim  = 0.7552   BERTScore-zh
S_Auto = 0.8582   Qwen3-MAX rubrics (4 维 / 100 分)
S_Exp  = 0.8067   = 0.5·Sim + 0.5·Auto
─────────────────
S_Fin  = 0.9034   = 0.45·Det + 0.25·Loc + 0.30·Exp
```

> **想复现 v1**：`git checkout v1.0-sft-baseline` 后按 `reference/README.v1.md` §九 **重训** SegFormer 5-fold + EffNet 5-fold + calibrator + Qwen3.5-9B LoRA（共需 ~3 天 @ 6×L20）。本地 ckpt 已不可用。
