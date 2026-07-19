#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
SOURCE_DIR="${V4_DIR}/results/full_diagnostic/${DATASET}"
RESULT_DIR="${V4_DIR}/results/alpha_selector/${DATASET}"
TRAIN_CACHE="${SOURCE_DIR}/calibration_train.pt"
VALID_CACHE="${SOURCE_DIR}/calibration_valid.pt"
TRAIN_TARGETS="${RESULT_DIR}/calibration_train_alpha_targets.pt"
VALID_TARGETS="${RESULT_DIR}/calibration_valid_alpha_targets.pt"

cd "${PROJECT_DIR}"
mkdir -p "${RESULT_DIR}" "${V4_DIR}/logs"

echo "[$(date --iso-8601=seconds)] Industrial Alpha predictability diagnostic started on physical GPU 7"

"${PYTHON_BIN}" versions/v4/scripts/generate_alpha_targets.py \
  --cache_path "${TRAIN_CACHE}" \
  --output_path "${TRAIN_TARGETS}" \
  --alpha_step 0.05 \
  --alpha_max 1.0 \
  --tau_ce 0.5 \
  --tau_metric 0.1 \
  --beta_rr 0.1 \
  --beta_ce 0.05 \
  --chunk_size 256 \
  --device cuda:0 \
  --overwrite

"${PYTHON_BIN}" versions/v4/scripts/generate_alpha_targets.py \
  --cache_path "${VALID_CACHE}" \
  --output_path "${VALID_TARGETS}" \
  --alpha_step 0.05 \
  --alpha_max 1.0 \
  --tau_ce 0.5 \
  --tau_metric 0.1 \
  --beta_rr 0.1 \
  --beta_ce 0.05 \
  --chunk_size 256 \
  --device cuda:0 \
  --overwrite

"${PYTHON_BIN}" versions/v4/scripts/analyze_alpha_targets.py \
  --train_cache "${TRAIN_CACHE}" \
  --valid_cache "${VALID_CACHE}" \
  --train_targets "${TRAIN_TARGETS}" \
  --valid_targets "${VALID_TARGETS}" \
  --output_json "${RESULT_DIR}/alpha_predictability.json" \
  --epochs 200 \
  --hidden_dim 32 \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 42 \
  --device cuda:0

echo "[$(date --iso-8601=seconds)] Industrial Alpha predictability diagnostic completed"
