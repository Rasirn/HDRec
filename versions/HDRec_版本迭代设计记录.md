# HDRec 版本迭代设计记录

本文档是当前项目唯一的版本迭代总说明。后续所有模型版本都按 `v1`、`v2`、`v3`、`v4` 的方式递增，不再使用临时实现名称作为版本名。

当前核心目标：

1. 解决或缓解 HDRec 两路信号联合训练时的梯度冲突。
2. 提高最终推荐性能，主要看 NDCG@10、Recall@10、MRR。

## 目录约定

```text
versions/
  v1/
    model/
    run.sh
    outputs/
    README.md
  v2/
    model/
    run.sh
    outputs/
    README.md
  v3/
    model/
    run.sh
    outputs/
    README.md
  HDRec_版本迭代设计记录.md
```

每个版本自己的模型、启动脚本、输出结果都放在同一个版本目录下。后续新增版本也遵循这个结构。

## 输出归档

历史输出已经完成迁移：

| 原目录 | 新目录 | 说明 |
|---|---|---|
| 根目录原始输出 | `versions/v1/outputs/` | 原始输出整体归入 v1 |
| 根目录 v3 历史输出 | `versions/v3/outputs/` | 层输出端融合版本输出整体归入 v3 |
| v2 历史输出 | `versions/v2/outputs/` | 当前为空 |

根目录不再保留新的实验输出目录。

---

## v1：原始 HDRec Baseline

### 设计理念

`v1` 是原始 HDRec 版本，也是后续所有实验的性能基线。它的核心思想是利用 LLM 的文本理解能力，同时引入协同过滤/ID 序列信号，构成 hybrid dual-semantics recommendation。

原始 HDRec 认为推荐任务中存在两类互补信息：

- 文本语义：物品标题、品牌、类别等属性能够帮助 LLM 理解物品内容。
- 协同语义：用户历史交互序列和传统推荐模型嵌入能够表达行为偏好。

`v1` 的目标是把这两类信号都接入 LLM 推荐模型，并通过 LoRA 适配器和 score 融合完成推荐。

### 模型结构

`v1` 使用 Llama 风格主干模型，并在目标线性层注入 PEFT LoRA：

- text adapter：`lora_text`
- cf adapter：`lora_cf`
- 注入模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`up_proj`、`down_proj`

两路分支：

- text 分支输入用户历史物品文本序列。
- ID/协同分支输入 item 序列、交互信息或 small model 表示。

融合方式主要发生在最终 score/logits 侧：

```text
scores = confidence_fusion(text_scores, cf_scores)
```

### 训练方式

`v1` 采用原始 HDRec 的分阶段/交替训练策略：

1. 前向 text 分支。
2. 前向 cf/id 分支。
3. 按 `alternating_learning` 控制两路更新节奏。
4. 使用 KL 约束保持两路输出一致性。
5. 根据验证集指标保存最优模型。

### v1 暴露的问题

`v1` 的主要问题是两路信号存在潜在梯度冲突：

- text 分支和 cf/id 分支可能对同一组共享参数给出相反更新方向。
- 虽然交替训练能减少直接冲突，但不能从结构上隔离两路更新。
- 融合主要在最终 score/logits 侧，中间 LoRA 层里的表示干扰仍然存在。
- 分阶段训练逻辑复杂，后续迭代和消融成本较高。

因此，`v2` 开始尝试用 FlyLoRA 的稀疏 rank 路由缓解这种冲突。

### 代码与启动

```text
versions/v1/model/
versions/v1/run.sh
versions/v1/outputs/
```

```bash
bash versions/v1/run.sh Industrial_and_Scientific
```

---

## v2：第一版 FlyLoRA 整合

### 要解决的 v1 问题

`v1` 的问题在于两路信号容易在共享低秩更新空间中互相干扰。`v2` 的出发点是：如果 LoRA 的更新可以通过稀疏 rank 路由分散到不同子空间，梯度冲突可能会减弱。

### 设计理念

`v2` 将原始 LoRA 替换为 FlyLoRA：

- 使用冻结稀疏随机投影 `A`。
- 使用可训练矩阵 `B` 学习低秩增量。
- 每次只激活 top-k rank。
- 使用 routing bias 调节 rank 使用均衡。

text/cf 两路信号通过 task 状态切换共享 FlyLoRA 层。

### 模型结构

单个 FlyLoRA 层包含：

```text
flylora_A      # 冻结稀疏随机投影
flylora_B      # 可训练矩阵
flylora_d      # task-aware routing bias
active_task_id # text/cf 任务切换
```

前向过程：

```text
y = A x
选择 top-k rank
delta = B (y * mask)
output = base_output + delta
```

### v2 的局限

`v2` 是快速验证版，证明 FlyLoRA 可以接入 HDRec，但仍存在不足：

- text 和 cf/id 仍共享一套 `A/B`，只是通过 task bias 区分。
- 融合位置仍偏最终 score/logits 侧。
- 没有充分利用 FlyLoRA 在 LoRA 层输出端直接融合的能力。
- 对“text 与 ID/协同信号应有独立子空间”的结构约束不够强。

因此，`v3` 改为双路 FlyLoRA，并把融合前移到每个 LoRA 注入层输出端。

### 代码与启动

```text
versions/v2/model/
versions/v2/run.sh
versions/v2/outputs/
```

```bash
bash versions/v2/run.sh Industrial_and_Scientific
```

---

## v3：双路 FlyLoRA 层输出端融合

### 要解决的 v2 问题

`v2` 虽然使用了 FlyLoRA，但仍没有从结构上彻底区分 text 与 ID/协同信号。`v3` 要解决两个关键问题：

1. 让两路信号拥有独立 FlyLoRA 子空间。
2. 将融合位置从最终 score/logits 端前移到 LoRA 层输出端。

### 设计理念

`v3` 使用 `DualFlyLoRALinear` 替换目标 Linear 层。每个注入层同时维护 text 和 ID 两套 FlyLoRA 分支：

```text
A_text, B_text, d_text
A_id,   B_id,   d_id
```

这样做的动机是：

- text 和 ID/协同信号不再强行共享同一个 FlyLoRA 子空间。
- 两路信号可以在每个注入层形成独立增量。
- 融合不再主要依赖最后一层 score，而是在模型中间层持续发生。

### 层内融合

在 fused 模式下：

```text
delta_text = FlyLoRA_text(x)
delta_id   = FlyLoRA_id(x)
delta      = output_mix * delta_text + (1 - output_mix) * delta_id
output     = base_output + delta
```

这一步发生在每个被注入的 LoRA 层输出端，是 `v3` 相比 `v2` 的核心变化。

### 损失函数

`v3` 使用联合损失：

```text
L_total = L_text + cf_loss_weight * L_id + kl_loss_weight * KL_bi
```

其中 `KL_bi` 是双向 stop-gradient KL：

```text
KL(stop_grad(id_logits) || text_logits)
KL(stop_grad(text_logits) || id_logits)
```

这样可以让两路预测互相对齐，同时避免一条分支直接拖拽另一条分支的梯度。

### 当前注意点

当前 `v3` 的 text loss 和 id loss 都走 fused 模式。这符合“层内融合主路径”的设计，但也值得继续消融：

- 训练时 text loss 只走 text 分支。
- 训练时 id loss 只走 id 分支。
- 推理时走 fused 分支。

这个方向可以作为 `v4`。

### 代码与启动

```text
versions/v3/model/
versions/v3/run.sh
versions/v3/outputs/
```

```bash
bash versions/v3/run.sh Industrial_and_Scientific
```

---

## 关键差异对照

| 维度 | v1 | v2 | v3 |
|---|---|---|---|
| 定位 | 原始 baseline | FlyLoRA 快速接入 baseline | 当前主线 |
| LoRA 结构 | 双 PEFT LoRA adapter | 单 FlyLoRA + task 切换 | 双路 FlyLoRA |
| 子空间隔离 | 弱 | 中等 | 强 |
| 融合位置 | 最终 score/logits 侧 | 仍偏末端 | LoRA 层输出端 |
| 梯度冲突缓解 | 交替训练 | 稀疏 rank 路由 | 双分支空间隔离 + 层内融合 |
| 主要风险 | 两路梯度冲突 | 融合位置不充分 | 需要更多消融验证 |

---

## 推荐性能记录

Version	Arts_Crafts_and_Sewing	Industrial_and_Scientific	Musical_Instruments	Prime_Pantry	Video_Games	改动点
paper	0.1412 / 0.1730 / 0.1368	0.1190 / 0.1568 / 0.1140	0.1023 / 0.1330 / 0.0989	0.0718 / 0.1034 / 0.0680	0.0940 / 0.1416 / 0.0882	
v1	0.1377 / 0.1682 / 0.1335	0.1061 / 0.1411 / 0.1016	0.0933 / 0.1222 / 0.0902	0.0626 / 0.0888 / 0.0600	0.0945 / 0.1421 / 0.0888	原始 HDRec baseline
v2	0.1343 / 0.1610 / 0.1313	"0.1150 / 0.1516 / 0.1106
"	0.0985 / 0.1270 / 0.0955	0.0678 / 0.0960 / 0.0649	0.0910 / 0.1346 / 0.0862	调整LoRA矩阵，去除交替训练策略
"v3
"	"0.1348 / 0.1620 / 0.1315
"	"0.1171 / 0.1505 / 0.1129
"	0.0986 / 0.1267 / 0.0957	"0.0682 / 0.0957 / 0.0652
"	"0.09023/0.1344/0.08513
"	在v2基础上，将后融合改成前融合，直接在LoRA模块后做参数融合

## 维护规则

- 新版本必须放在 `versions/vN/`。
- 每个版本必须有 `model/`、`run.sh`、`outputs/`、`README.md`。
- 每个 README 使用中文说明。
- 版本名只用 `vN`，不要把临时方法名写进目录名。
- 设计理念和实验结论统一更新到本文档。
