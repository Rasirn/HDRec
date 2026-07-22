# v4 Candidate Gate 流水线审计

## 唯一正式定义

v4 = 冻结的 v1 checkpoint + Candidate-Level Reliability Gate。LLM、两个 LoRA、
SASRec 与投影器不参与 v4 Gate 训练。

## 旧入口问题

旧 `versions/v4/run.sh -> versions/run_common.sh -> model/main.py` 复用了固定
`confidence_fusion`，因此名称带 v4 的普通训练日志不是 Candidate Gate。`model/main.py`
现在默认直接失败；仅显式 `--allow_legacy_fixed_fusion` 可用于历史排查。

## 正式调用链

`v1 checkpoint -> valid cache -> user-disjoint calibration split -> Candidate Gate -> final Gate retrain -> test cache -> official Candidate Gate test`。

`versions/v4/run.sh --dataset DATASET --v1_checkpoint PATH --gpu GPU` 是唯一正式入口。
cache、Gate、test cache 与正式结果均绑定同一个 v1 checkpoint 的绝对路径、SHA256、数据集、
profile、feature schema 与融合温度；任一不一致会失败。

## 正式结果身份

正式 JSON 的固定字段为：

```json
{"method": "v4_candidate_gate", "uses_candidate_gate": true}
```

正式主结果只写 Candidate Gate 指标。Text-only、ID-only、v1 fixed 与 Oracle-Alpha
仅写入 `diagnostics/`，不属于正式 v4 指标。
