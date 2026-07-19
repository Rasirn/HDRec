# v4：可靠性感知、文本锚定的动态融合

`v4` 从 `v1` 复制而来，保留原始 HDRec 的文本分支、ID/协同分支、交替训练、checkpoint 与评估逻辑。与 `v2/v3` 不同，`v4` 不使用 FlyLoRA，不做 LoRA 层内融合，也不做动态 rank。

## 核心目标

`v1` 使用全局固定 `fusion_alpha`。`v4` 将其升级为样本级动态可靠性权重：

```text
z_final = z_text + alpha_u * r_id
```

第一阶段只实现序列级动态权重：

```text
alpha_u = clamp(alpha0 + rho * DeltaAlpha_u, 0, alpha_max)
```

其中 `DeltaAlpha_u` 由轻量 Context Gate 根据可靠性特征预测。

## 安全原则

- 文本 logits 完整保留。
- `alpha=0` 时严格退化为 text-only。
- 融合器训练只更新 fuser，冻结 v1 主模型。
- 不使用测试集训练融合器或拟合特征标准化。
- 先做 Oracle 和 utility 可预测性诊断，再决定是否继续复杂门控。

## 主要入口

```bash
# 1. 缓存冻结 v1 双分支预测
python versions/v4/scripts/cache_fusion_data.py --help

# 2. Oracle 诊断
python versions/v4/scripts/analyze_oracle.py --help

# 3. Utility 可预测性诊断
python versions/v4/scripts/analyze_reliability.py --help

# 4. 训练 Context Gate
python versions/v4/scripts/train_fuser.py --help

# 5. 评估 v4
python versions/v4/scripts/evaluate_v4.py --help
```

## 开发记录

所有实现细节、运行命令、测试结果和未完成事项维护在：

```text
versions/v4/开发与实验记录.md
```
