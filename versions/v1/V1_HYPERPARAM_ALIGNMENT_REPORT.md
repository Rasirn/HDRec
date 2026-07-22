# v1 超参数对齐报告

## 参数优先级

`parameters.py` 默认值 < `versions/v1/run.sh` profile 显式参数 < 用户显式命令行。
v1 不再通过 `versions/run_common.sh` 注入数据集专用覆盖。

## src_original 与旧配置

| 参数 | src 原始值 / src_original | legacy_tuned Video | 修复后值 |
|---|---:|---:|---:|
| epochs | 40 | 12 | 40 |
| learning_rate | 5e-5 | 1.5e-4 | 5e-5 |
| batch_size | 4 | 10 | 4 |
| gradient_accumulation_steps | 4 | 4 | 4 |
| effective_batch_size | 16 | 40 | 16 |
| weight_decay | 0 | 1e-2 | 0 |
| warmup_steps | 2000 | 3969 | 2000 |
| mixed_precision | no | bf16 | no |
| score_dropout | 0.5 | 0.4 | 0.5 |
| hidden_dropout | 0 | 0.05 | 0 |
| adapter_dropout | 0.3 | 0.5 | 0.3 |
| lora_r / lora_alpha | 8 / 32 | 8 / 32 | 8 / 32 |
| lora_frequency / hd_frequency | 1 / 1 | 1 / 8 | 1 / 1 |
| kl_loss_weight | 1.0 | 0.3 | 1.0 |
| alternating_learning | 2 | 2 | 2 |
| fusion_alpha / temperature | 0.5 / 0.5 | 0.5 / 1.2 | 0.5 / 0.5 |
| fusion_type | text | text | text |
| max_item_num / max_token_num | 30 / 1024 | 10 / 1024 | 30 / 1024 |
| skip_valid / patient | 15 / 10 | 0 / 1 | 15 / 10 |
| seed / num_workers | 42 / 1 | 42 / 1 | 42 / 1 |

`src_original` 以外层 `src/parameters.py` 为基线。旧 Video 的性能下降风险主要来自
学习率、batch/effective batch、精度、正则化与 dropout、训练/验证/早停时序、融合温度
以及 `max_item_num` 的联合漂移，不能归因于模型代码差异。

## 运行身份

每次 v1 运行会在输出目录写入 `run_manifest.json`，记录 profile、完整解析参数、命令、
数据/SASRec/checkpoint SHA256、Git 状态、运行环境、最佳验证 epoch 与最终 test 指标。

## Video src_original 启动记录

- 启动时间：2026-07-22，screen=`hdrec_v1_video_src`，物理 GPU 7。
- 参数已按 `src_original` 完整解析并写入日志；模型与数据集初始化成功。
- 首个 forward 因 GPU 7 上已有外部进程占用约 35 GB 显存而 OOM（仅余约 87 MB），未产生训练 step、checkpoint 或可用结果。
- 未改动 `src_original` 的 `mixed_precision=no`、batch size 或其他基线参数以规避 OOM；待 GPU 7 有足够空闲显存后使用同一命令重启。
