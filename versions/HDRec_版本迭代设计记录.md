# HDRec 版本迭代设计记录

> 仓库：`Rasirn/HDRec`  
> 当前目标版本：`v4`  
> 主线：可靠性感知、文本锚定的动态融合  
> 后续预留：`v5` 选择性语义对齐

---

## 1. 项目背景

HDRec 使用文本语义分支与协同语义分支完成序列推荐。原方法的关键并不是简单地同时使用文本和 ID，而是形成了以下非对称协作关系：

1. 文本语义作为稳定主信号；
2. 协同语义作为补充信号；
3. 两个分支在训练阶段保持相对独立；
4. 在共同的候选物品分数空间中进行推理期融合；
5. 当协同信号不可靠时，应能够退回文本分支。

现有版本如下：

| 版本 | 核心改动 | 实验现象 | 结论 |
|---|---|---|---|
| v1 | HDRec 原始方法复现 | 固定为比较基线 | 保留 |
| v2 | FlyLoRA 解耦与稀疏路由 | 小数据集提升，大数据集下降 | 未真正解决任务冲突，且存在路由与容量问题 |
| v3 | 将融合前移至 LoRA 模块后 | 小数据集提升，大数据集下降 | 改变融合空间、公式和训练耦合，破坏原设计原则 |
| v4 | 可靠性感知动态融合 | 待实现 | 当前主线 |
| v5 | 选择性语义对齐 | 暂不实现 | 等 v4 结果后再决定 |

---

## 2. 前序版本结论

### 2.1 v2

v2 的参数解耦没有严格成立：

- 文本任务和协同任务仍可能更新共享 LoRA 参数；
- 路由依赖冻结随机投影与手工负载均衡，而不是推荐目标；
- routing bias 随前向次数累积，对训练步数和数据规模敏感；
- 固定低秩和 top-k 激活在小数据集上可能是正则化，在大数据集上可能限制容量；
- 隐藏层路由与最终输出端融合之间存在较弱的信用分配。

因此，“继续做更强的解耦”暂不作为下一版本主线。

### 2.2 v3

v3 不只是改变融合位置，还同时改变了：

- 融合空间：物品分数空间变成隐藏表示空间；
- 融合公式：文本主干加协同残差变成固定比例凸组合；
- 训练方式：两种任务共同依赖融合后的两套 LoRA；
- 路由统计：文本和协同输入交叉更新路由状态；
- 推理路径：删除原始候选级 `confidence_fusion`。

因此，v3 不能用于证明“提前融合更好或更差”，它实际已经改变了 HDRec 的基本归纳偏置。

---

## 3. v4 核心研究问题

v1 的融合使用 ID 分数相对均值构造置信权重，并对全部用户和候选使用全局固定 `fusion_alpha`。它实际衡量的是“ID 分数是否相对较高”，而不是：

> 当前用户、当前序列和当前候选上的 ID 修正是否真的可靠，是否会改善文本预测。

固定 alpha 无法刻画：

- 用户历史长度；
- 用户协同证据丰富程度；
- 文本与 ID 分支各自的不确定性；
- 两路预测分歧；
- 头部与长尾物品差异；
- 不同数据规模和局部复杂度。

v4 的核心假设是：

> HDRec 的主要局限不是两个语义分支融合得不够充分，而是当前方法无法准确判断“何时应该融合”以及“应该融合多少”。

---

## 4. v4 方法定义

### 4.1 暂定名称

**RATF-HDRec: Reliability-Aware Text-Anchored Fusion**

中文：**可靠性感知的文本锚定融合 HDRec**

### 4.2 必须保留的原则

1. 文本 logits 完整保留；
2. ID 分支只作为残差修正；
3. v1 主模型训练方式不变；
4. 动态融合器独立训练；
5. 融合器梯度不得进入 LLM、LoRA、SASRec 或投影器；
6. 当动态权重为 0 时严格退化为 text-only；
7. 不使用测试标签或测试统计训练融合器；
8. 第一版优先实现序列级门控，再考虑候选级门控；
9. 不在 v4 中加入表示对齐、动态 rank、FlyLoRA 或层内融合。

### 4.3 数学形式

设两分支 logits 为：

\[
z_{u,i}^{text},\quad z_{u,i}^{id}
\]

v1 可抽象为：

\[
z_{u,i}^{final}=z_{u,i}^{text}+lpha r_{u,i}^{id}
\]

v4 改为：

\[
z_{u,i}^{final}=z_{u,i}^{text}+lpha_{u,i}r_{u,i}^{id}
\]

第一阶段只使用序列级权重：

\[
z_{u,i}^{final}=z_{u,i}^{text}+lpha_u r_{u,i}^{id}
\]

推荐使用安全残差参数化：

\[
lpha_u=
clip(lpha_0+
ho\Deltalpha_u,0,lpha_{max})
\]

其中：

- `alpha0`：v1 在当前数据集上的固定最优值；
- `Delta alpha`：轻量 MLP 根据可靠性特征输出；
- `rho`：限制动态调整幅度；
- `alpha_max`：最大协同修正强度。

这样当融合器学习不足时，可以回退到 v1 固定融合附近，而不是产生完全随机的权重。

---

## 5. 可靠性特征

第一版序列级特征：

| 特征 | 含义 |
|---|---|
| `text_entropy` | 文本预测分布熵 |
| `id_entropy` | ID 预测分布熵 |
| `text_margin` | 文本 Top-1 与 Top-2 分数差 |
| `id_margin` | ID Top-1 与 Top-2 分数差 |
| `branch_jsd` | 两路预测分布的 Jensen-Shannon divergence |
| `topk_overlap` | 两路 Top-K 候选集合重叠率 |
| `history_length` | 用户行为序列长度 |
| `history_pop_mean` | 历史物品平均流行度 |
| `history_pop_std` | 历史物品流行度波动 |
| `text_id_score_corr` | 两路候选分数相关系数 |

后续候选级版本可加入：

- `text_score`、`id_score`；
- `text_rank`、`id_rank`；
- `score_gap`；
- `candidate_popularity`；
- `candidate_cf_support`；
- 候选级分支分歧。

规范：

- 流行度等统计只由训练数据计算；
- 特征标准化只在融合器训练 split 上拟合；
- 处理 NaN、Inf 和零方差；
- 大候选空间下避免逐物品 Python 循环；
- 特征 schema 和 normalization state 必须随 checkpoint 保存。

---

## 6. 边际效用监督

v4 学习的不是“ID 分数是否较高”，而是“加入 ID 修正是否有益”。

定义：

\[
U=\ell(z^{text},y)-\ell(z^{text}+r^{id},y)
\]

- \(U>0\)：ID 修正有益；
- \(U<0\)：ID 修正有害；
- \(|U|\)：收益或伤害强度。

第一版构造二分类标签：

\[
y^{utility}=\mathbb{I}(U>0)
\]

融合器同时优化最终排名和 utility 判断：

\[
L_{fusion}
=
L_{rank}
+\lambda_uL_{utility}
+\lambda_rL_{shrink}
\]

其中：

\[
L_{shrink}=\|lpha-lpha_0\|_2^2
\]

`L_rank` 应尽量复用 v1 的 next-item loss，`L_utility` 使用 BCE。

---

## 7. 融合器结构

第一版 Context Gate：

```text
features
→ LayerNorm
→ Linear
→ GELU
→ Dropout
→ Linear
→ Tanh
→ Delta alpha
```

默认建议：

```yaml
hidden_dim: 32
dropout: 0.1
alpha_max: 1.0
rho: 0.5
use_alpha_residual: true
```

第一版不使用：

- Transformer 融合器；
- LoRA 层内融合；
- token 级路由；
- 动态 rank；
- 图神经网络；
- 端到端更新主模型；
- 表示对齐损失。

---

## 8. 数据与训练流程

### 阶段 0：固定 v1

1. 确认 v1 checkpoint 能加载；
2. 复现 text-only、id-only 和 v1 fusion；
3. 记录每个数据集的最优固定 alpha；
4. 不修改或覆盖 v1。

### 阶段 1：生成融合数据

使用冻结 v1 缓存：

```text
sample_id
target_item
text_logits 或可重建指标的候选摘要
id_logits
history_length
item popularity statistics
text_loss
base_fused_loss
utility_label
```

数据划分必须避免泄漏：

- 主训练集用于 v1；
- calibration-train 用于融合器；
- validation 用于调参和早停；
- test 只做最终评估。

若不方便直接增加 calibration split，可使用 out-of-fold 预测生成融合器训练数据。

### 阶段 2：先做诊断

在训练复杂门控前，必须完成：

1. Oracle-Select；
2. Oracle-Alpha；
3. utility 标签分布；
4. 简单 Logistic Regression 或线性层的 utility AUC。

只有 Oracle 上界明显高于 v1，且 utility 可预测性高于随机水平，才继续 Context Gate。

### 阶段 3：训练 Context Gate

1. 冻结 v1 全部参数；
2. 只训练融合器；
3. 以 validation NDCG@10 或仓库主指标早停；
4. 保存最优 fuser checkpoint；
5. 每轮记录 alpha 均值、标准差、最小值和最大值。

### 阶段 4：Candidate Gate

只有 Context Gate 在至少一个大数据集上稳定超过 v1 后才实现。

---

## 9. Oracle 诊断

### 9.1 Oracle-Select

逐样本在 text-only 和 v1 fixed fusion 中选择目标排名更好的结果。

用途：判断“是否融合”的潜在上界。

### 9.2 Oracle-Alpha

对离散 alpha：

```text
0.0, 0.1, 0.2, ..., alpha_max
```

逐样本选择最优 alpha，输出：

- Oracle Recall/NDCG；
- 最优 alpha 分布；
- alpha=0 的样本比例；
- 不同历史长度、流行度和分支分歧分桶下的最优 alpha；
- 相对 v1 的理论提升空间。

若 Oracle 与 v1 差距很小，应暂停 v4，避免继续堆叠复杂门控。

---

## 10. 实验设计

### 10.1 对比方法

| 方法 | 说明 |
|---|---|
| v1 text-only | 只用文本分支 |
| v1 id-only | 只用协同分支 |
| v1 fixed fusion | 原始固定融合 |
| v2 | FlyLoRA |
| v3 | LoRA 后融合 |
| v4-global | 重新调优的固定 alpha |
| v4-context | 序列级动态融合 |
| v4-candidate | 候选级动态融合 |
| Oracle-Select | text/fused 样本级选择上界 |
| Oracle-Alpha | 样本级 alpha 上界 |

### 10.2 主指标

沿用仓库现有指标，至少包含：

- Recall@5、Recall@10；
- NDCG@5、NDCG@10；
- 其他已有指标保持一致。

### 10.3 新增分析指标

- `harm_rate`：融合比 text-only 更差的样本比例；
- `benefit_rate`：融合比 text-only 更好的样本比例；
- `average_target_rank_gain`；
- `utility_auc`；
- Brier Score 或 ECE；
- alpha 分布；
- 额外参数量；
- 推理耗时和显存开销。

### 10.4 分组实验

按以下维度分桶：

1. 用户历史长度；
2. 目标物品流行度；
3. text entropy；
4. ID entropy；
5. branch JSD；
6. Top-K overlap；
7. utility 正负。

重点分析 Arts 和 Video。

### 10.5 必做消融

1. 固定 alpha vs context alpha；
2. 只用熵特征；
3. 只用分支分歧特征；
4. 只用数据统计特征；
5. 去掉 utility loss；
6. 去掉 shrink loss；
7. 不同 `alpha_max`；
8. 不同 `rho`；
9. 不同融合器容量；
10. 不同 calibration split 大小；
11. 原始 ID residual vs 简单 ID residual；
12. Candidate Gate 的增益与开销。

---

## 11. 大小数据集差异的预期解释

大数据集通常具有更强异质性，全局固定 alpha 更容易出现：

- 对协同可靠样本修正不足；
- 对协同噪声样本过度修正；
- 对长尾和短历史用户使用错误的信任强度。

v4 预期学习到：

- 高 ID 不确定性、低协同支持、高分支分歧时降低 alpha；
- 长历史、高协同支持、ID 高置信时提高 alpha；
- 在文本已经非常确定时减少不必要的 ID 干预；
- 在文本不确定且 ID 可靠时增加协同修正。

---

## 12. 代码组织建议

以 v1 为基础创建：

```text
versions/v4/
├── README.md
├── model/
│   ├── reliability_fusion.py
│   ├── reliability_features.py
│   ├── utility_label.py
│   └── ...
├── scripts/
│   ├── cache_fusion_data.py
│   ├── analyze_oracle.py
│   ├── train_fuser.py
│   ├── evaluate_v4.py
│   └── analyze_reliability.py
├── results/
├── logs/
└── 开发与实验记录.md
```

实际目录服从仓库原有风格。

兼容要求：

- v1 命令继续可运行；
- 不启用动态融合时可退化到 v1 fixed fusion；
- 主模型 checkpoint 和 fuser checkpoint 分开保存；
- 日志记录数据集、随机种子、主模型 checkpoint 和全部融合参数；
- 新参数必须提供默认值。

---

## 13. 分阶段版本

### v4.0：诊断与上界

交付：

- 双分支 logits 缓存；
- Oracle-Select；
- Oracle-Alpha；
- 可靠性特征统计；
- utility 可预测性。

通过条件：

- Arts 或 Video 上 Oracle 明显优于 v1；
- utility 标签不是极端单一；
- 简单分类器 AUC 高于随机。

### v4.1：Context Gate

交付：

- 序列级动态 alpha；
- 独立 fuser 训练；
- text-only 安全回退；
- 至少一个大数据集稳定超过 v1。

### v4.2：Candidate Gate

仅在 v4.1 成立后实现，目标是进一步提升长尾、高分歧样本。

### v4.3：可靠性校准

可选：

- temperature scaling；
- utility calibration；
- fallback threshold；
- 风险受控融合。

### v5：选择性语义对齐

v4 完成后再考虑：

- 置信加权 KL；
- Top-K 选择性 KL；
- 共享—私有表示；
- 对比对齐。

---

## 14. 成功判据

v4 应同时满足：

1. Arts 和 Video 至少有一个版本稳定超过 v1；
2. 两个大数据集平均优于 v1；
3. 四个小数据集整体不显著退化；
4. 至少 3 个随机种子方向一致；
5. 动态门控优于重新调优后的全局 alpha；
6. Oracle 分析能说明动态融合上限；
7. 分组实验能解释门控行为；
8. 额外参数和推理开销可控；
9. 不修改 v1 主分支即可获得收益；
10. 能形成“发现问题—提出方法—验证机制”的完整叙事。

---

## 15. 风险与失败判定

### 融合器过拟合

应对：

- 小型 MLP；
- dropout、weight decay；
- shrink 到 alpha0；
- out-of-fold 特征；
- 多随机种子。

### 计算与存储过高

应对：

- 先做 Context Gate；
- logits 分块缓存；
- 只缓存必要候选；
- 全物品评估分块计算；
- 避免逐候选 Python 循环。

### 门控只学到流行度

应对：

- popularity-only 对照；
- 长尾分组；
- 特征消融；
- permutation importance。

### 动态融合不优于全局 alpha

若 Oracle 差距小或动态门控无稳定收益，应停止继续堆叠复杂门控，转向 v5 或重新检查 v1 复现。

---

## 16. 论文叙事草案

### 核心发现

HDRec 通过训练期隔离和推理期融合结合文本与协同语义，但其置信融合仅利用 ID 分数的相对幅值，并采用全局固定融合强度，无法识别协同修正在不同用户、候选和数据区域上的真实可靠性。

### 根本原因

协同信号的有效性依赖用户行为丰富度、物品流行度、两个分支的不确定性和分歧程度。因此，同一个固定 alpha 会在部分样本上提供有益补充，在另一些样本上造成有害修正，且这一问题在大规模异质数据上更加明显。

### 方法

提出可靠性感知、文本锚定的动态融合框架：

- 保留并冻结 HDRec 双分支；
- 显式估计协同残差的条件边际效用；
- 根据用户上下文动态生成融合强度；
- 只在协同证据可靠时修正文本预测；
- 保留 text-only 安全回退。

### 一句话 Takeaway

**HDRec 的关键局限不是双语义融合不足，而是缺少对协同修正条件可靠性的建模；v4 将固定融合升级为边际效用驱动的可靠性受控融合。**

---

## 17. 实验记录模板

```markdown
## 第 N 次迭代

### 目标
### Git commit
### 数据集与随机种子
### 主模型 checkpoint
### 融合器配置
### 修改文件
### 运行命令
### 测试与实验结果
### 与 v1 差异
### 分组结果
### 已知问题
### 下一步
```

失败实验也必须保留。

---

## 18. 下一步执行顺序

1. 从 v1 复制创建 v4；
2. 审计 v1 logits、标签、评估和融合调用路径；
3. 实现 text/ID logits 缓存；
4. 实现 Oracle-Select 与 Oracle-Alpha；
5. 实现基础可靠性特征；
6. 运行 utility 可预测性诊断；
7. 诊断成立后实现 Context Gate；
8. 先在小数据集调通，再重点运行 Arts 和 Video；
9. v4.1 稳定后再决定 Candidate Gate；
10. v4 完成后再讨论 v5。

---

## 19. 当前 v4 代码交付状态

当前已完成代码实现，但尚未运行真实 GPU 全量实验。

已交付：

- `versions/v4/` 独立目录；
- 从 v1 复制的主模型与运行入口；
- 双分支 logits 缓存脚本；
- 可靠性特征提取；
- utility label 构造；
- Oracle-Select；
- Oracle-Alpha；
- 线性 utility 可预测性诊断；
- Context Gate；
- 独立 fuser 训练脚本；
- v4 评估脚本；
- CPU 合成 smoke test。

未完成：

- 真实数据集的完整缓存；
- Arts、Video 和全部数据集的 Oracle 诊断；
- 真实 fuser 训练和多随机种子实验；
- Candidate Gate。

### 当前代码入口

```bash
# 1. 缓存冻结 v1 的双分支预测
conda run -n hdrec python versions/v4/scripts/cache_fusion_data.py \
  --dataset Industrial_and_Scientific \
  --split valid \
  --checkpoint_path versions/v1/outputs/Industrial_and_Scientific/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_v1/pytorch_model.bin \
  --cache_path versions/v4/results/cache/Industrial_and_Scientific/valid.pt \
  --overwrite

# 2. 运行 Oracle 诊断
conda run -n hdrec python versions/v4/scripts/analyze_oracle.py \
  --cache_path versions/v4/results/cache/Industrial_and_Scientific/valid.pt \
  --output_json versions/v4/results/cache/Industrial_and_Scientific/valid.oracle.json

# 3. 运行 utility 可预测性诊断
conda run -n hdrec python versions/v4/scripts/analyze_reliability.py \
  --cache_path versions/v4/results/cache/Industrial_and_Scientific/valid.pt \
  --output_json versions/v4/results/cache/Industrial_and_Scientific/valid.utility.json

# 4. 训练 Context Gate
conda run -n hdrec python versions/v4/scripts/train_fuser.py \
  --train_cache versions/v4/results/cache/Industrial_and_Scientific/valid.pt \
  --valid_cache versions/v4/results/cache/Industrial_and_Scientific/valid.pt \
  --output_path versions/v4/results/fuser/Industrial_and_Scientific/fuser.pt \
  --device cuda

# 5. 评估 v4
conda run -n hdrec python versions/v4/scripts/evaluate_v4.py \
  --cache_path versions/v4/results/cache/Industrial_and_Scientific/test.pt \
  --fuser_path versions/v4/results/fuser/Industrial_and_Scientific/fuser.pt \
  --output_json versions/v4/results/fuser/Industrial_and_Scientific/test.eval.json \
  --device cuda
```

注意：上面的 `train_cache` 当前示例使用 valid 作为 calibration；正式实验不能用 test 训练或拟合标准化。若后续构造独立 calibration split，应优先使用独立 split。
