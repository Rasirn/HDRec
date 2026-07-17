# HDRec 项目工作区

本仓库基于 HDRec（Hybrid Dual-Semantics Modeling for Enhancing LLM-Based Recommendation）进行后续实验迭代。当前开发重点不是继续在根目录堆临时版本，而是把所有新模型版本统一收敛到 `versions/` 目录中。

## 当前迭代入口

正式版本工作区：

```text
versions/
  v1/    # 原始 HDRec baseline
  v2/    # 第一版 FlyLoRA 整合 baseline
  v3/    # 当前主线：双路 FlyLoRA 层输出端融合
```

每个版本内部都包含：

```text
model/      # 该版本模型代码
run.sh      # 该版本启动脚本
outputs/   # 该版本输出结果
README.md  # 该版本中文说明
```

统一版本说明见：

```text
versions/HDRec_版本迭代设计记录.md
```

## 运行方式

```bash
bash versions/v1/run.sh Industrial_and_Scientific
bash versions/v2/run.sh Industrial_and_Scientific
bash versions/v3/run.sh Industrial_and_Scientific
```

示例：

```bash
FLYLORA_R=32 FLYLORA_K=8 FLYLORA_OUTPUT_MIX=0.5 bash versions/v3/run.sh Video_Games deepseek-ai/DeepSeek-R1-Distill-Llama-8B v3_r32_k8_mix0.5
```

## 输出位置

新旧实验结果已经按版本收拢：

```text
versions/v1/outputs/   # 原 output/ 下的历史结果
versions/v2/outputs/   # 当前为空
versions/v3/outputs/   # 原 v3 历史结果
```

后续新实验也只写入对应版本的 `outputs/`。

## 环境准备

```bash
conda env create -f hdrec.yml
conda activate hdrec
```

## 数据准备

1. 从 Amazon Product Data 下载原始数据。
2. 运行数据处理脚本：

```bash
python ./src/data/process_data_18.py
```

3. 将处理后的数据放入 `./data/`。

协同过滤嵌入（如 SASRec）仍放在：

```text
temp/SASRec/
```

## 开发约定

- 新版本按 `v4`、`v5` 递增。
- 新版本目录必须放在 `versions/` 下。
- 不再新增根目录临时模型目录。
- 每个版本自己的启动脚本和输出结果必须跟版本放在一起。

## Citation

```bibtex
@inproceedings{liu2026hybrid,
  title={Hybrid Dual-Semantics Modeling for Enhancing Large Language Model Based Recommendation},
  author={Liu, Canyi and Li, Tianyi and Li, Wei and Zhang, Youchen and Li, Xiaodong and Li, Hui},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining (WSDM)},
  pages={396--405},
  year={2026},
  publisher={ACM}
}
```
