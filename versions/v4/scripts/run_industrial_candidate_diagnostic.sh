#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
SOURCE_DIR="${V4_DIR}/results/full_diagnostic/${DATASET}"
RESULT_DIR="${V4_DIR}/results/candidate_fusion/${DATASET}"

cd "${PROJECT_DIR}"
mkdir -p "${RESULT_DIR}" "${V4_DIR}/logs"

echo "[$(date --iso-8601=seconds)] Industrial candidate policy diagnostic started on physical GPU 7"

if [[ ! -f "${RESULT_DIR}/candidate_policy_results.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/analyze_candidate_policies.py \
    --train_cache "${SOURCE_DIR}/calibration_train.pt" \
    --valid_cache "${SOURCE_DIR}/calibration_valid.pt" \
    --data_root "${PROJECT_DIR}/data" \
    --selected_json "${RESULT_DIR}/selected_candidate_rule.json" \
    --output_json "${RESULT_DIR}/candidate_policy_results.json" \
    --batch_size 64 \
    --candidate_chunk 1024 \
    --seed 42 \
    --device cuda:0
  echo "[$(date --iso-8601=seconds)] Candidate policy diagnostic completed"
else
  echo "[$(date --iso-8601=seconds)] Reusing frozen candidate policy result; calibration-valid is not evaluated twice"
fi

GO_NO_GO_1="$("${PYTHON_BIN}" -c "import json; print(str(json.load(open('${RESULT_DIR}/candidate_policy_results.json'))['go_no_go_1']).lower())")"
if [[ "${GO_NO_GO_1}" != "true" ]]; then
  echo "[$(date --iso-8601=seconds)] Go/No-Go-1 failed; Candidate Gate training skipped"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] Go/No-Go-1 passed; Linear Candidate Gate training started"
"${PYTHON_BIN}" versions/v4/scripts/train_candidate_gate.py \
  --train_cache "${SOURCE_DIR}/calibration_train.pt" \
  --data_root "${PROJECT_DIR}/data" \
  --output_dir "${RESULT_DIR}/checkpoints" \
  --summary_json "${RESULT_DIR}/candidate_gate_training.json" \
  --batch_size 16 \
  --candidate_chunk 1024 \
  --epochs 6 \
  --patience 2 \
  --initial_gate_probability 0.01 \
  --seed 42 \
  --device cuda:0

echo "[$(date --iso-8601=seconds)] Frozen Candidate Gate independent validation started"
"${PYTHON_BIN}" versions/v4/scripts/evaluate_candidate_gate.py \
  --train_cache "${SOURCE_DIR}/calibration_train.pt" \
  --valid_cache "${SOURCE_DIR}/calibration_valid.pt" \
  --training_summary "${RESULT_DIR}/candidate_gate_training.json" \
  --policy_results "${RESULT_DIR}/candidate_policy_results.json" \
  --residual_results "${V4_DIR}/results/residual_calibration/${DATASET}/residual_calibration_results.json" \
  --data_root "${PROJECT_DIR}/data" \
  --output_json "${RESULT_DIR}/candidate_gate_evaluation.json" \
  --batch_size 16 \
  --candidate_chunk 1024 \
  --seed 42 \
  --device cuda:0

echo "[$(date --iso-8601=seconds)] Industrial candidate diagnostic completed"
