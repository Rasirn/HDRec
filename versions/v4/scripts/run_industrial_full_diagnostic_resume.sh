#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
RESULT_DIR="${V4_DIR}/results/full_diagnostic/${DATASET}"
CAL_TRAIN="${RESULT_DIR}/calibration_train.pt"
CAL_VALID="${RESULT_DIR}/calibration_valid.pt"

cd "${PROJECT_DIR}"

for required_file in "${CAL_TRAIN}" "${CAL_VALID}" "${RESULT_DIR}/validation_oracle.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing prerequisite: ${required_file}" >&2
    exit 1
  fi
done

echo "[$(date --iso-8601=seconds)] Resuming Industrial diagnostic after completed cache and split"

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

echo "[$(date --iso-8601=seconds)] Industrial full validation diagnostic resume completed"
