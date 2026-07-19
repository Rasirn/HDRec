#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Industrial_and_Scientific"
SOURCE_DIR="${V4_DIR}/results/full_diagnostic/${DATASET}"
CANDIDATE_DIR="${V4_DIR}/results/candidate_fusion/${DATASET}"
FINAL_DIR="${V4_DIR}/results/final_test/${DATASET}"
CHECKPOINT="${PROJECT_DIR}/versions/v1/outputs/${DATASET}/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_all_klw0.7_fa0.5/pytorch_model.bin"
TEST_CACHE="${FINAL_DIR}/test.pt"
FINAL_GATE="${FINAL_DIR}/final_linear_candidate_gate.pt"

cd "${PROJECT_DIR}"
mkdir -p "${FINAL_DIR}" "${V4_DIR}/logs"

echo "[$(date --iso-8601=seconds)] Industrial v4 final experiment started on physical GPU 7"

if [[ ! -f "${FINAL_GATE}" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/train_final_candidate_gate.py \
    --calibration_train "${SOURCE_DIR}/calibration_train.pt" \
    --calibration_valid "${SOURCE_DIR}/calibration_valid.pt" \
    --training_summary "${CANDIDATE_DIR}/candidate_gate_training.json" \
    --data_root "${PROJECT_DIR}/data" \
    --output_checkpoint "${FINAL_GATE}" \
    --output_manifest "${FINAL_DIR}/final_gate_manifest.json" \
    --batch_size 16 \
    --candidate_chunk 1024 \
    --seed 42 \
    --device cuda:0
else
  echo "[$(date --iso-8601=seconds)] Reusing frozen final Candidate Gate"
fi

if [[ ! -f "${TEST_CACHE}" ]]; then
  echo "[$(date --iso-8601=seconds)] Caching frozen v1 Industrial test logits"
  "${PYTHON_BIN}" versions/v4/scripts/cache_fusion_data.py \
    --dataset "${DATASET}" \
    --split test \
    --checkpoint_path "${CHECKPOINT}" \
    --cache_path "${TEST_CACHE}" \
    --batch_size 4 \
    --device cuda:0
else
  echo "[$(date --iso-8601=seconds)] Reusing Industrial test cache"
fi

if [[ ! -f "${FINAL_DIR}/test_oracle.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/analyze_oracle.py \
    --cache_path "${TEST_CACHE}" \
    --alpha_step 0.05 \
    --output_json "${FINAL_DIR}/test_oracle.json"
fi

if [[ ! -f "${FINAL_DIR}/final_test_results.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/evaluate_candidate_gate_test.py \
    --test_cache "${TEST_CACHE}" \
    --candidate_checkpoint "${FINAL_GATE}" \
    --oracle_json "${FINAL_DIR}/test_oracle.json" \
    --data_root "${PROJECT_DIR}/data" \
    --output_json "${FINAL_DIR}/final_test_results.json" \
    --output_markdown "${FINAL_DIR}/final_test_results.md" \
    --batch_size 16 \
    --candidate_chunk 1024 \
    --seed 42 \
    --device cuda:0
else
  echo "[$(date --iso-8601=seconds)] Final test result already exists; test is not evaluated twice"
fi

echo "[$(date --iso-8601=seconds)] Industrial v4 final experiment completed"
