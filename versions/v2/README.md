# v2：第一版 FlyLoRA 整合

`v2` 是在原始 HDRec 上快速接入 FlyLoRA 的版本，主要用于验证 FlyLoRA 的稀疏 rank 路由是否能缓解 `v1` 中两路信号的梯度冲突。

## 要解决的 v1 问题

`v1` 的 text 分支和 ID/协同分支虽然有 adapter 区分和交替训练，但两路信号仍可能在共享表示空间里互相干扰。`v2` 尝试用 FlyLoRA 替代普通 LoRA，让参数更新进入稀疏 rank 子空间，从而减少直接冲突。

## 模型设计

`v2` 使用单个 FlyLoRA 线性层包装目标 Linear 模块：

- `flylora_A`：冻结稀疏随机投影。
- `flylora_B`：可训练升维矩阵。
- `flylora_d`：任务相关 routing bias。
- top-k rank 激活：每次只激活部分 rank。

text/cf 通过 task 状态切换共享 FlyLoRA 层：

```text
set_flylora_task(model, "text")
set_flylora_task(model, "cf")
```

## 局限

`v2` 是快速验证版。它虽然把 LoRA 替换成 FlyLoRA，但整体融合思路仍比较接近原始 HDRec，主要问题是：

- 融合仍偏末端 score/logits 侧。
- text 和 ID/协同没有形成真正独立的 FlyLoRA 分支。
- 对“LoRA 层输出端直接融合”的 FlyLoRA 思想利用还不充分。

这些问题正是 `v3` 要解决的重点。

## 代码位置

```text
versions/v2/model/
```

关键文件：

```text
main.py
parameters.py
trainer.py
model_utils.py
flylora.py
```

## 启动方式

```bash
bash versions/v2/run.sh Industrial_and_Scientific
```

## 输出位置

```text
versions/v2/outputs/
```

当前保持为空。之前的早期结果不再作为正式 v2 输出保留。

## 主要超参

- `FLYLORA_R`
- `FLYLORA_K`
- `FLYLORA_ALPHA`
- `FLYLORA_SPARSITY_RATIO`
- `FLYLORA_BIAS_LR`
