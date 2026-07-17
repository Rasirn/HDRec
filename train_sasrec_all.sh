#!/bin/bash
# train_sasrec_all.sh
# 用法：
#   bash train_sasrec_all.sh <数据集编号> [GPU编号]
#
# 数据集编号对应关系:
#   0 -> Arts_Crafts_and_Sewing
#   1 -> Industrial_and_Scientific
#   2 -> Musical_Instruments
#   3 -> Office_Products
#   4 -> Prime_Pantry
#   5 -> Video_Games
#
# 示例（6个screen同时跑）:
#   screen -S ds0 -dm bash -c "conda activate recbole && bash /data/lgd/HDRec/train_sasrec_all.sh 0 0"
#   screen -S ds1 -dm bash -c "conda activate recbole && bash /data/lgd/HDRec/train_sasrec_all.sh 1 1"
#   screen -S ds2 -dm bash -c "conda activate recbole && bash /data/lgd/HDRec/train_sasrec_all.sh 2 2"
#   screen -S ds3 -dm bash -c "conda activate recbole && bash /data/lgd/HDRec/train_sasrec_all.sh 3 3"
#   screen -S ds4 -dm bash -c "conda activate recbole && bash /data/lgd/HDRec/train_sasrec_all.sh 4 4"
#   screen -S ds5 -dm bash -c "conda activate recbole && bash /data/lgd/HDRec/train_sasrec_all.sh 5 5"

set -e

DATASETS=(
    "Arts_Crafts_and_Sewing"      # 0
    "Industrial_and_Scientific"   # 1
    "Musical_Instruments"         # 2
    "Office_Products"             # 3
    "Prime_Pantry"                # 4
    "Video_Games"                 # 5
)

# ---------- 参数解析 ----------
DS_IDX="${1:-}"       # 第一个参数: 数据集编号 0-5
CUDA_GPU="${2:-${CUDA_GPU:-0}}"  # 第二个参数: GPU编号，默认 0

if [ -z "$DS_IDX" ]; then
    echo "用法: bash train_sasrec_all.sh <数据集编号 0-5> [GPU编号]"
    echo ""
    echo "数据集编号:"
    for i in "${!DATASETS[@]}"; do
        echo "  $i -> ${DATASETS[$i]}"
    done
    exit 1
fi

if [[ "$DS_IDX" -lt 0 || "$DS_IDX" -ge "${#DATASETS[@]}" ]]; then
    echo "错误: 数据集编号必须在 0-$((${#DATASETS[@]}-1)) 之间"
    exit 1
fi

DATASET="${DATASETS[$DS_IDX]}"

# ---------- 路径配置 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECBOLE_DIR="$SCRIPT_DIR/RecBole"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/temp/SASRec"
EMB_DIM=128

echo "============================================================"
echo ">>> 数据集编号: $DS_IDX  ->  $DATASET"
echo ">>> GPU: $CUDA_GPU"
echo "============================================================"

cd "$RECBOLE_DIR"

# Step 1: 将 .inter 文件软链接到 RecBole dataset 目录
RECBOLE_DS_DIR="$RECBOLE_DIR/dataset/$DATASET"
mkdir -p "$RECBOLE_DS_DIR"
if [ ! -f "$RECBOLE_DS_DIR/$DATASET.inter" ]; then
    ln -s "$DATA_DIR/$DATASET/$DATASET.inter" "$RECBOLE_DS_DIR/$DATASET.inter"
    echo "  已链接 $DATASET.inter"
fi

# Step 2: 训练 SASRec
SAVE_DIR="$RECBOLE_DIR/saved/$DATASET"
echo "  训练 SASRec... (GPU $CUDA_GPU, 保存至 $SAVE_DIR)"
python run_recbole.py \
    --model SASRec \
    --dataset "$DATASET" \
    --config_files sasrec_config.yaml \
    --gpu_id="$CUDA_GPU" \
    --checkpoint_dir="$SAVE_DIR"

# Step 3: 找到最新保存的模型文件
MODEL_PATH=$(ls -t "$SAVE_DIR/SASRec-"*.pth 2>/dev/null | head -1)
if [ -z "$MODEL_PATH" ]; then
    echo "  错误: 未找到 SASRec 模型文件"
    exit 1
fi
echo "  找到模型: $MODEL_PATH"

# Step 4: 提取 embedding
echo "  提取 item embedding..."
python "$SCRIPT_DIR/extract_sasrec_embeddings.py" \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --recbole_dataset_dir "$RECBOLE_DIR/dataset" \
    --smap_path "$DATA_DIR/$DATASET/smap.json" \
    --emb_dim "$EMB_DIM" \
    --output_dir "$OUTPUT_DIR"

echo "  完成: $OUTPUT_DIR/$DATASET-$EMB_DIM.pth"
