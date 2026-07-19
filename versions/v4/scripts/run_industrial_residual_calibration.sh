#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
SOURCE_DIR="${V4_DIR}/results/full_diagnostic/${DATASET}"
ALPHA_DIR="${V4_DIR}/results/alpha_selector/${DATASET}"
RESULT_DIR="${V4_DIR}/results/residual_calibration/${DATASET}"

cd "${PROJECT_DIR}"
mkdir -p "${RESULT_DIR}" "${V4_DIR}/logs"

echo "[$(date --iso-8601=seconds)] Industrial residual calibration started on physical GPU 7"

"${PYTHON_BIN}" versions/v4/scripts/analyze_residual_scale.py \
  --train_cache "${SOURCE_DIR}/calibration_train.pt" \
  --valid_cache "${SOURCE_DIR}/calibration_valid.pt" \
  --train_targets "${ALPHA_DIR}/calibration_train_alpha_targets.pt" \
  --valid_targets "${ALPHA_DIR}/calibration_valid_alpha_targets.pt" \
  --train_features_output "${RESULT_DIR}/calibration_train_residual_features.pt" \
  --valid_features_output "${RESULT_DIR}/calibration_valid_residual_features.pt" \
  --output_json "${RESULT_DIR}/residual_scale_analysis.json" \
  --num_buckets 5 \
  --chunk_size 256 \
  --device cuda:0

"${PYTHON_BIN}" versions/v4/scripts/analyze_residual_calibration.py \
  --train_cache "${SOURCE_DIR}/calibration_train.pt" \
  --valid_cache "${SOURCE_DIR}/calibration_valid.pt" \
  --train_scale_features "${RESULT_DIR}/calibration_train_residual_features.pt" \
  --valid_scale_features "${RESULT_DIR}/calibration_valid_residual_features.pt" \
  --selected_config_json "${RESULT_DIR}/selected_train_config.json" \
  --output_json "${RESULT_DIR}/residual_calibration_results.json" \
  --seed 42 \
  --device cuda:0

echo "[$(date --iso-8601=seconds)] Industrial residual calibration completed"
