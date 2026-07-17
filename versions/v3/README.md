# v3：双路 FlyLoRA 层输出端融合

`v3` 是当前主线版本。它在 `v2` 的基础上进一步改造 FlyLoRA 结构，把 text 和 ID/协同信号拆成两条 FlyLoRA 分支，并在每个注入层的 LoRA 输出端直接融合。

## 要解决的 v2 问题

`v2` 已经引入 FlyLoRA，但仍更像是在原始 HDRec 框架里替换 LoRA 模块。它没有充分解决一个关键问题：HDRec 的两路信号主要在最终 score/logits 端融合，而 FlyLoRA 更适合解决 LoRA 层输出端的融合与冲突控制。

因此 `v3` 的核心目标是：

- 将融合位置从末端 score/logits 前移到 LoRA 层输出端。
- 给 text 和 ID/协同信号各自独立的 FlyLoRA 子空间。
- 用双向 stop-gradient KL 保持两路预测一致性。
- 在缓解梯度冲突的同时提高推荐指标。

## 模型设计

每个被替换的 Linear 模块变为 `DualFlyLoRALinear`：

- `flylora_A_text`：text 分支冻结稀疏投影。
- `flylora_A_id`：ID/协同分支冻结稀疏投影。
- `flylora_B_text`：text 分支可训练矩阵。
- `flylora_B_id`：ID/协同分支可训练矩阵。
- `flylora_d_text` / `flylora_d_id`：分支内 routing bias。

在 fused 模式下：

```text
delta_text = FlyLoRA_text(x)
delta_id   = FlyLoRA_id(x)
delta      = output_mix * delta_text + (1 - output_mix) * delta_id
output     = base_output + delta
```

这个融合发生在每个 LoRA 注入层，而不是最后的 logits 端。

## 训练策略

训练损失：

```text
L_total = L_text + cf_loss_weight * L_id + kl_loss_weight * KL_bi
```

其中 `KL_bi` 是双向 stop-gradient KL：

- `KL(stop_grad(id_logits) || text_logits)`
- `KL(stop_grad(text_logits) || id_logits)`

当前实现中，text loss 和 id loss 都走 fused 模式。后续可以做一个 `v4` 消融：训练时 text loss 只走 text 分支，id loss 只走 id 分支，评估时再走 fused 分支。

## 代码位置

```text
versions/v3/model/
```

关键文件：

```text
main.py
parameters.py
trainer.py
model_utils.py
flylora_dual.py
```

## 启动方式

```bash
bash versions/v3/run.sh Industrial_and_Scientific
```

示例：

```bash
FLYLORA_R=32 FLYLORA_K=8 FLYLORA_OUTPUT_MIX=0.5 bash versions/v3/run.sh Video_Games deepseek-ai/DeepSeek-R1-Distill-Llama-8B v3_r32_k8_mix0.5
```

## 输出位置

```text
versions/v3/outputs/<dataset>/<model>_<suffix>/
```

历史 v3 实验结果已经搬到这里。

## 主要超参

- `FLYLORA_R`
- `FLYLORA_K`
- `FLYLORA_ALPHA`
- `FLYLORA_SPARSITY_RATIO`
- `FLYLORA_BIAS_LR`
- `FLYLORA_OUTPUT_MIX`
