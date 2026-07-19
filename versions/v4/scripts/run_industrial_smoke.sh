#!/usr/bin/env bash
set -euo pipefail

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
SMOKE_DIR="${V4_DIR}/results/smoke/${DATASET}"
CACHE_PATH="${SMOKE_DIR}/valid_256.pt"
CAL_TRAIN="${SMOKE_DIR}/calibration_train.pt"
CAL_VALID="${SMOKE_DIR}/calibration_valid.pt"
CHECKPOINT="${PROJECT_DIR}/versions/v1/outputs/${DATASET}/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_all_klw0.7_fa0.5/pytorch_model.bin"

cd "${PROJECT_DIR}"
mkdir -p "${SMOKE_DIR}" "${V4_DIR}/logs"

echo "[$(date --iso-8601=seconds)] Industrial v4 smoke started on cuda:7"

"${PYTHON_BIN}" versions/v4/scripts/cache_fusion_data.py \
  --dataset "${DATASET}" \
  --split valid \
  --checkpoint_path "${CHECKPOINT}" \
  --cache_path "${CACHE_PATH}" \
  --max_samples 256 \
  --batch_size 4 \
  --device cuda:7 \
  --overwrite

"${PYTHON_BIN}" versions/v4/scripts/split_calibration.py \
  --input_cache "${CACHE_PATH}" \
  --train_output "${CAL_TRAIN}" \
  --valid_output "${CAL_VALID}" \
  --index_output "${SMOKE_DIR}/calibration_split_indices.pt" \
  --valid_ratio 0.2 \
  --seed 42 \
  --overwrite

"${PYTHON_BIN}" versions/v4/scripts/analyze_oracle.py \
  --cache_path "${CACHE_PATH}" \
  --output_json "${SMOKE_DIR}/oracle.json"

"${PYTHON_BIN}" versions/v4/scripts/analyze_reliability.py \
  --cache_path "${CAL_TRAIN}" \
  --valid_cache "${CAL_VALID}" \
  --output_json "${SMOKE_DIR}/utility.json" \
  --seed 42

echo "[$(date --iso-8601=seconds)] Industrial v4 smoke completed"
