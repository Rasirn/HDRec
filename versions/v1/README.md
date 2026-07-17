# v1：原始 HDRec Baseline

`v1` 是原始 HDRec 版本，也是后续所有版本对比的基线。这个版本的目标是复现 HDRec 原始双语义建模流程，并保留原始模型中可能产生梯度冲突的训练方式，作为 FlyLoRA 改进的参照。

## 要解决的问题背景

HDRec 的核心思想是同时利用两类信号：

- 文本语义信号：来自用户历史物品的标题、品牌等文本属性。
- 协同/ID 信号：来自用户交互序列和传统推荐模型产生的偏好表示。

这两路信号都有助于推荐，但在联合训练时可能对共享参数产生方向不一致的梯度。也就是说，文本分支希望模型朝一个方向更新，ID/协同分支可能希望模型朝另一个方向更新，最终导致优化互相拉扯。

`v1` 保留这个原始问题，用来回答两个问题：

1. 原始 HDRec 在各数据集上的推荐性能是多少。
2. 后续 FlyLoRA 版本是否真正缓解了梯度冲突并提升推荐指标。

## 模型设计

### 主干模型

- 基座模型：`DeepSeek-R1-Distill-Llama-8B`
- 推荐模型封装：`model/models/llama.py` 中的 `llama_rec`
- 任务形式：Next-Item Recommendation
- 输出：候选物品分类分数

### 双语义分支

`v1` 同时建模 text 分支和 ID/协同分支：

- text 分支使用用户历史序列的文本 token。
- ID/协同分支使用 item 序列、交互信息或小模型偏好表示。
- 推理或训练中可通过 `confidence_fusion` 对两路 score 做融合。

### LoRA 结构

`v1` 使用 PEFT LoRA：

- text adapter：`lora_text`
- cf adapter：`lora_cf`
- 目标模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`up_proj`、`down_proj`

LoRA 注入逻辑在：

```text
versions/v1/model/models/model_utils.py
```

## 训练策略

原始 HDRec 使用分阶段/交替训练来缓解两路信号直接冲突：

1. 前向 text 分支。
2. 前向 cf/id 分支。
3. 根据 `alternating_learning` 控制两路更新节奏。
4. 使用 KL 约束让两路输出保持一定一致性。
5. 最终根据验证集指标选择最优模型。

这种策略的好处是实现了初步解耦；问题是训练逻辑复杂，而且冲突本质上仍然发生在共享参数和最终融合路径上。

## 代码位置

```text
versions/v1/model/
```

关键文件：

```text
main.py
parameters.py
trainer.py
models/model_utils.py
models/llama.py
models/modules.py
```

## 启动方式

```bash
bash versions/v1/run.sh Industrial_and_Scientific
```

可选参数：

```bash
bash versions/v1/run.sh <dataset> <model_name_or_path> <suffix>
```

## 输出位置

```text
versions/v1/outputs/<dataset>/<model>_<suffix>/
```

历史 `output/` 目录下的结果已经搬到这里。

## 后续版本要解决的 v1 问题

- 两路信号仍可能存在梯度方向冲突。
- 末端融合无法控制中间 LoRA 子空间中的干扰。
- 交替训练增加训练复杂度。
- 后续版本需要在保持或提升推荐指标的同时，让 text 和 ID/协同信号的更新更稳定。
