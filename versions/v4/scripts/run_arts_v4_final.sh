#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="/home/lgd/.conda/envs/hdrec/bin/python"
DATASET="Arts_Crafts_and_Sewing"
RESULT_DIR="${V4_DIR}/results/arts_candidate/${DATASET}"
FINAL_DIR="${V4_DIR}/results/final_test/${DATASET}"
CHECKPOINT="${PROJECT_DIR}/versions/v1/outputs/${DATASET}/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_all_klw0.5_fa0.7/pytorch_model.bin"
VALID_CACHE="${RESULT_DIR}/valid.pt"
CAL_TRAIN="${RESULT_DIR}/calibration_train.pt"
CAL_VALID="${RESULT_DIR}/calibration_valid.pt"
TEST_CACHE="${FINAL_DIR}/test.pt"
FINAL_GATE="${FINAL_DIR}/final_candidate_gate.pt"

cd "${PROJECT_DIR}"
mkdir -p "${RESULT_DIR}" "${FINAL_DIR}" "${V4_DIR}/logs"
echo "[$(date --iso-8601=seconds)] Arts v4 experiment started on physical GPU 7"

if [[ ! -f "${VALID_CACHE}" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/cache_fusion_data.py \
    --dataset "${DATASET}" --split valid --checkpoint_path "${CHECKPOINT}" \
    --cache_path "${VALID_CACHE}" --batch_size 4 --device cuda:0
fi

if [[ ! -f "${CAL_TRAIN}" || ! -f "${CAL_VALID}" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/split_calibration.py \
    --input_cache "${VALID_CACHE}" \
    --train_output "${CAL_TRAIN}" --valid_output "${CAL_VALID}" \
    --index_output "${RESULT_DIR}/calibration_split_indices.pt" \
    --valid_ratio 0.2 --seed 42
fi

if [[ ! -f "${RESULT_DIR}/candidate_policy_results.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/analyze_candidate_policies.py \
    --train_cache "${CAL_TRAIN}" --valid_cache "${CAL_VALID}" \
    --data_root "${PROJECT_DIR}/data" \
    --selected_json "${RESULT_DIR}/selected_candidate_rule.json" \
    --output_json "${RESULT_DIR}/candidate_policy_results.json" \
    --batch_size 16 --candidate_chunk 512 --seed 42 --device cuda:0
fi

GO_NO_GO_1="$("${PYTHON_BIN}" -c "import json; print(str(json.load(open('${RESULT_DIR}/candidate_policy_results.json'))['go_no_go_1']).lower())")"
if [[ "${GO_NO_GO_1}" != "true" ]]; then
  echo "[$(date --iso-8601=seconds)] Arts Go/No-Go-1 failed; stopping before learnable Gate" >&2
  exit 2
fi

if [[ ! -f "${RESULT_DIR}/candidate_gate_training.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/train_candidate_gate.py \
    --train_cache "${CAL_TRAIN}" --data_root "${PROJECT_DIR}/data" \
    --output_dir "${RESULT_DIR}/checkpoints" \
    --summary_json "${RESULT_DIR}/candidate_gate_training.json" \
    --batch_size 8 --candidate_chunk 512 --epochs 6 --patience 2 \
    --initial_gate_probability 0.01 --seed 42 --device cuda:0
fi

if [[ ! -f "${RESULT_DIR}/candidate_gate_evaluation.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/evaluate_candidate_gate.py \
    --train_cache "${CAL_TRAIN}" --valid_cache "${CAL_VALID}" \
    --training_summary "${RESULT_DIR}/candidate_gate_training.json" \
    --policy_results "${RESULT_DIR}/candidate_policy_results.json" \
    --data_root "${PROJECT_DIR}/data" \
    --output_json "${RESULT_DIR}/candidate_gate_evaluation.json" \
    --batch_size 8 --candidate_chunk 512 --seed 42 --device cuda:0
fi

if [[ ! -f "${FINAL_GATE}" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/train_final_candidate_gate.py \
    --calibration_train "${CAL_TRAIN}" --calibration_valid "${CAL_VALID}" \
    --training_summary "${RESULT_DIR}/candidate_gate_training.json" \
    --data_root "${PROJECT_DIR}/data" \
    --output_checkpoint "${FINAL_GATE}" \
    --output_manifest "${FINAL_DIR}/final_gate_manifest.json" \
    --batch_size 8 --candidate_chunk 512 --seed 42 --device cuda:0
fi

if [[ ! -f "${TEST_CACHE}" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/cache_fusion_data.py \
    --dataset "${DATASET}" --split test --checkpoint_path "${CHECKPOINT}" \
    --cache_path "${TEST_CACHE}" --batch_size 4 --device cuda:0
fi

if [[ ! -f "${FINAL_DIR}/test_oracle.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/analyze_oracle.py \
    --cache_path "${TEST_CACHE}" --alpha_step 0.05 \
    --output_json "${FINAL_DIR}/test_oracle.json"
fi

if [[ ! -f "${FINAL_DIR}/final_test_results.json" ]]; then
  "${PYTHON_BIN}" versions/v4/scripts/evaluate_candidate_gate_test.py \
    --test_cache "${TEST_CACHE}" --candidate_checkpoint "${FINAL_GATE}" \
    --oracle_json "${FINAL_DIR}/test_oracle.json" --data_root "${PROJECT_DIR}/data" \
    --output_json "${FINAL_DIR}/final_test_results.json" \
    --output_markdown "${FINAL_DIR}/final_test_results.md" \
    --batch_size 8 --candidate_chunk 512 --seed 42 --device cuda:0
fi

echo "[$(date --iso-8601=seconds)] Arts v4 experiment completed"
