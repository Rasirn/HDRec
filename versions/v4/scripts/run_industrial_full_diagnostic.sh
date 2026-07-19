#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
RESULT_DIR="${V4_DIR}/results/full_diagnostic/${DATASET}"
CACHE_PATH="${RESULT_DIR}/valid.pt"
CAL_TRAIN="${RESULT_DIR}/calibration_train.pt"
CAL_VALID="${RESULT_DIR}/calibration_valid.pt"
CHECKPOINT="${PROJECT_DIR}/versions/v1/outputs/${DATASET}/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_all_klw0.7_fa0.5/pytorch_model.bin"

cd "${PROJECT_DIR}"
mkdir -p "${RESULT_DIR}" "${V4_DIR}/logs"

echo "[$(date --iso-8601=seconds)] Industrial full validation diagnostic started on physical GPU 7"

"${PYTHON_BIN}" versions/v4/scripts/cache_fusion_data.py \
  --dataset "${DATASET}" \
  --split valid \
  --checkpoint_path "${CHECKPOINT}" \
  --cache_path "${CACHE_PATH}" \
  --batch_size 4 \
  --device cuda:0 \
  --overwrite

"${PYTHON_BIN}" versions/v4/scripts/analyze_oracle.py \
  --cache_path "${CACHE_PATH}" \
  --alpha_step 0.05 \
  --output_json "${RESULT_DIR}/validation_oracle.json"

"${PYTHON_BIN}" versions/v4/scripts/split_calibration.py \
  --input_cache "${CACHE_PATH}" \
  --train_output "${CAL_TRAIN}" \
  --valid_output "${CAL_VALID}" \
  --index_output "${RESULT_DIR}/calibration_split_indices.pt" \
  --valid_ratio 0.2 \
  --seed 42 \
  --overwrite

"${PYTHON_BIN}" versions/v4/scripts/analyze_reliability.py \
  --cache_path "${CAL_TRAIN}" \
  --valid_cache "${CAL_VALID}" \
  --output_json "${RESULT_DIR}/utility_analysis.json" \
  --seed 42

"${PYTHON_BIN}" versions/v4/scripts/analyze_oracle.py \
  --cache_path "${CAL_VALID}" \
  --alpha_step 0.05 \
  --output_json "${RESULT_DIR}/calibration_valid_oracle.json"

"${PYTHON_BIN}" versions/v4/scripts/summarize_diagnostics.py \
  --full_oracle "${RESULT_DIR}/validation_oracle.json" \
  --calibration_oracle "${RESULT_DIR}/calibration_valid_oracle.json" \
  --utility_analysis "${RESULT_DIR}/utility_analysis.json" \
  --output_json "${RESULT_DIR}/summary.json" \
  --output_markdown "${RESULT_DIR}/summary.md" \
  --memory_safe

echo "[$(date --iso-8601=seconds)] Industrial full validation diagnostic completed"
