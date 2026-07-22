#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash versions/v4/run.sh --dataset DATASET --v1_checkpoint PATH [--gpu 7]
Runs the only official v4 definition: frozen v1 + Candidate-Level Reliability Gate.
EOF
}

DATASET=""
V1_CHECKPOINT=""
GPU="${GPU_IDS:-0}"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_ROOT=""
RESULT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --v1_checkpoint) V1_CHECKPOINT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --result_root) RESULT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$DATASET" && -n "$V1_CHECKPOINT" ]] || { usage >&2; exit 2; }
[[ -f "$V1_CHECKPOINT" ]] || { echo "v1 checkpoint not found: $V1_CHECKPOINT" >&2; exit 2; }

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${V4_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lgd/.conda/envs/hdrec/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
SHORT_SHA="$(sha256sum "$V1_CHECKPOINT" | awk '{print substr($1, 1, 12)}')"
RESULT_ROOT="${RESULT_ROOT:-${V4_DIR}/results/candidate/${DATASET}/${SHORT_SHA}}"
FINAL_DIR="${V4_DIR}/results/final_test/${DATASET}"
VALID_CACHE="${RESULT_ROOT}/valid.pt"
CAL_TRAIN="${RESULT_ROOT}/calibration_train.pt"
CAL_VALID="${RESULT_ROOT}/calibration_valid.pt"
TEST_CACHE="${FINAL_DIR}/test.pt"
FINAL_GATE="${FINAL_DIR}/final_candidate_gate.pt"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
mkdir -p "$RESULT_ROOT" "$FINAL_DIR/diagnostics"

"$PYTHON_BIN" versions/v4/scripts/write_v4_manifest.py \
  --dataset "$DATASET" --v1_checkpoint "$V1_CHECKPOINT" --output "${RESULT_ROOT}/run_manifest.json"

if [[ ! -f "$VALID_CACHE" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/cache_fusion_data.py \
    --dataset "$DATASET" --split valid --checkpoint_path "$V1_CHECKPOINT" \
    --cache_path "$VALID_CACHE" --model_name_or_path "$MODEL" --data_root "$DATA_ROOT" \
    --batch_size 4 --device cuda:0
fi
if [[ ! -f "$CAL_TRAIN" || ! -f "$CAL_VALID" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/split_calibration.py \
    --input_cache "$VALID_CACHE" --train_output "$CAL_TRAIN" --valid_output "$CAL_VALID" \
    --index_output "${RESULT_ROOT}/calibration_split_indices.pt" --valid_ratio 0.2 --seed 42
fi
if [[ ! -f "${RESULT_ROOT}/candidate_policy_results.json" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/analyze_candidate_policies.py \
    --train_cache "$CAL_TRAIN" --valid_cache "$CAL_VALID" --data_root "$DATA_ROOT" \
    --selected_json "${RESULT_ROOT}/selected_candidate_rule.json" \
    --output_json "${RESULT_ROOT}/candidate_policy_results.json" \
    --batch_size 16 --candidate_chunk 512 --seed 42 --device cuda:0
fi
if [[ ! -f "${RESULT_ROOT}/candidate_gate_training.json" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/train_candidate_gate.py \
    --train_cache "$CAL_TRAIN" --data_root "$DATA_ROOT" --output_dir "${RESULT_ROOT}/checkpoints" \
    --summary_json "${RESULT_ROOT}/candidate_gate_training.json" \
    --batch_size 8 --candidate_chunk 512 --epochs 6 --patience 2 --seed 42 --device cuda:0
fi
if [[ ! -f "${RESULT_ROOT}/candidate_gate_evaluation.json" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/evaluate_candidate_gate.py \
    --train_cache "$CAL_TRAIN" --valid_cache "$CAL_VALID" \
    --training_summary "${RESULT_ROOT}/candidate_gate_training.json" \
    --policy_results "${RESULT_ROOT}/candidate_policy_results.json" --data_root "$DATA_ROOT" \
    --output_json "${RESULT_ROOT}/candidate_gate_evaluation.json" \
    --batch_size 8 --candidate_chunk 512 --seed 42 --device cuda:0
fi
if [[ ! -f "$FINAL_GATE" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/train_final_candidate_gate.py \
    --calibration_train "$CAL_TRAIN" --calibration_valid "$CAL_VALID" \
    --training_summary "${RESULT_ROOT}/candidate_gate_training.json" --data_root "$DATA_ROOT" \
    --output_checkpoint "$FINAL_GATE" --output_manifest "${FINAL_DIR}/final_gate_manifest.json" \
    --batch_size 8 --candidate_chunk 512 --seed 42 --device cuda:0
fi
if [[ ! -f "$TEST_CACHE" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/cache_fusion_data.py \
    --dataset "$DATASET" --split test --checkpoint_path "$V1_CHECKPOINT" \
    --cache_path "$TEST_CACHE" --model_name_or_path "$MODEL" --data_root "$DATA_ROOT" \
    --batch_size 4 --device cuda:0
fi
if [[ ! -f "${FINAL_DIR}/diagnostics/oracle.json" ]]; then
  "$PYTHON_BIN" versions/v4/scripts/analyze_oracle.py --cache_path "$TEST_CACHE" \
    --alpha_step 0.05 --output_json "${FINAL_DIR}/diagnostics/oracle.json"
fi
"$PYTHON_BIN" versions/v4/scripts/evaluate_candidate_gate_test.py \
  --test_cache "$TEST_CACHE" --candidate_checkpoint "$FINAL_GATE" \
  --oracle_json "${FINAL_DIR}/diagnostics/oracle.json" --data_root "$DATA_ROOT" \
  --output_json "${FINAL_DIR}/v4_candidate_metrics.json" \
  --diagnostics_json "${FINAL_DIR}/diagnostics/comparison.json" \
  --official_log "${FINAL_DIR}/v4_candidate_test.log" --official-only \
  --batch_size 8 --candidate_chunk 512 --seed 42 --device cuda:0
"$PYTHON_BIN" versions/v4/scripts/write_v4_manifest.py \
  --dataset "$DATASET" --v1_checkpoint "$V1_CHECKPOINT" --candidate_checkpoint "$FINAL_GATE" \
  --metrics_json "${FINAL_DIR}/v4_candidate_metrics.json" --output "${FINAL_DIR}/run_manifest.json"
cp "${FINAL_DIR}/run_manifest.json" "${FINAL_DIR}/provenance.json"
