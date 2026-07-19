#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR=$1
VERSION_ID=$2
ENTRY_FILE=$3
DEFAULT_SUFFIX=$4
shift 4

ROOT_DIR="$(cd "${VERSION_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET=${1:-Industrial_and_Scientific}
MODEL=${2:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
SUFFIX=${3:-${DEFAULT_SUFFIX}}

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-6}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

case "${DATASET}" in
  Arts_Crafts_and_Sewing)
    EPOCHS=${EPOCHS:-10}
    LR=${LR:-1.6e-4}
    SCORE_DROPOUT=${SCORE_DROPOUT:-0.4}
    BATCH_SIZE=${BATCH_SIZE:-10}
    ALL_STEPS=${ALL_STEPS:-56083}
    KL_LOSS_WEIGHT=${KL_LOSS_WEIGHT:-0.5}
    FUSION_ALPHA=${FUSION_ALPHA:-0.7}
    FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}
    ;;
  Industrial_and_Scientific)
    EPOCHS=${EPOCHS:-10}
    LR=${LR:-1.5e-4}
    SCORE_DROPOUT=${SCORE_DROPOUT:-0.5}
    BATCH_SIZE=${BATCH_SIZE:-10}
    ALL_STEPS=${ALL_STEPS:-10975}
    KL_LOSS_WEIGHT=${KL_LOSS_WEIGHT:-0.7}
    FUSION_ALPHA=${FUSION_ALPHA:-0.5}
    FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}
    ;;
  Musical_Instruments)
    EPOCHS=${EPOCHS:-8}
    LR=${LR:-1.5e-4}
    SCORE_DROPOUT=${SCORE_DROPOUT:-0.4}
    BATCH_SIZE=${BATCH_SIZE:-10}
    ALL_STEPS=${ALL_STEPS:-27518}
    KL_LOSS_WEIGHT=${KL_LOSS_WEIGHT:-0.7}
    FUSION_ALPHA=${FUSION_ALPHA:-0.7}
    FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}
    ;;
  Prime_Pantry)
    EPOCHS=${EPOCHS:-10}
    LR=${LR:-1.5e-4}
    SCORE_DROPOUT=${SCORE_DROPOUT:-0.5}
    BATCH_SIZE=${BATCH_SIZE:-10}
    ALL_STEPS=${ALL_STEPS:-14177}
    KL_LOSS_WEIGHT=${KL_LOSS_WEIGHT:-0.3}
    FUSION_ALPHA=${FUSION_ALPHA:-0.3}
    FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.0}
    ;;
  Video_Games)
    EPOCHS=${EPOCHS:-12}
    LR=${LR:-1.5e-4}
    SCORE_DROPOUT=${SCORE_DROPOUT:-0.4}
    BATCH_SIZE=${BATCH_SIZE:-10}
    ALL_STEPS=${ALL_STEPS:-55126}
    KL_LOSS_WEIGHT=${KL_LOSS_WEIGHT:-0.3}
    FUSION_ALPHA=${FUSION_ALPHA:-0.5}
    FUSION_TEMPERATURE=${FUSION_TEMPERATURE:-1.2}
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

OUTPUT_DIR=${OUTPUT_DIR:-${VERSION_DIR}/outputs}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT:-0.05}
HD_FREQUENCY=${HD_FREQUENCY:-8}
ADAPTER_DROPOUT=${ADAPTER_DROPOUT:-0.5}
LORA_FREQUENCY=${LORA_FREQUENCY:-1}
MAX_ITEM_NUM=${MAX_ITEM_NUM:-10}
PATIENT=${PATIENT:-1}
SKIP_VALID=${SKIP_VALID:-0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
USE_GRAD_CKPT=${USE_GRAD_CKPT:-0}

GPU_NUM=${GPU_NUM:-$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}
WARMUP_STEPS=${WARMUP_STEPS:-$(awk "BEGIN {print int(${ALL_STEPS} / (${GPU_NUM} * ${BATCH_SIZE}) * ${EPOCHS} * 0.06 + 0.5)}")}

GRAD_CKPT_FLAG=()
if [[ "${USE_GRAD_CKPT}" == "1" ]]; then
  GRAD_CKPT_FLAG=(--gradient_checkpointing_enable)
fi

COMMON_ARGS=(
  --dataset "${DATASET}"
  --model_name_or_path "${MODEL}"
  --suffix "${SUFFIX}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --learning_rate "${LR}"
  --num_train_epochs "${EPOCHS}"
  --warmup_steps "${WARMUP_STEPS}"
  --skip_valid "${SKIP_VALID}"
  --patient "${PATIENT}"
  --weight_decay "${WEIGHT_DECAY}"
  --hidden_dropout "${HIDDEN_DROPOUT}"
  --hd_frequency "${HD_FREQUENCY}"
  --score_dropout "${SCORE_DROPOUT}"
  --adapter_dropout "${ADAPTER_DROPOUT}"
  --lora_frequency "${LORA_FREQUENCY}"
  --max_item_num "${MAX_ITEM_NUM}"
  --fix_backbone
  --fix_emb
  --use_small_model
  --use_gate
  --use_lora
  --fusion_alpha "${FUSION_ALPHA}"
  --fusion_temperature "${FUSION_TEMPERATURE}"
  --kl_loss_weight "${KL_LOSS_WEIGHT}"
  "${GRAD_CKPT_FLAG[@]}"
)

VERSION_ARGS=()
case "${VERSION_ID}" in
  v1)
    ;;
  v2)
    FLYLORA_R=${FLYLORA_R:-16}
    FLYLORA_K=${FLYLORA_K:-4}
    FLYLORA_ALPHA=${FLYLORA_ALPHA:-32}
    FLYLORA_SPARSITY_RATIO=${FLYLORA_SPARSITY_RATIO:-0.25}
    FLYLORA_BIAS_LR=${FLYLORA_BIAS_LR:-1e-3}
    VERSION_ARGS=(
      --use_flylora
      --flylora_r "${FLYLORA_R}"
      --flylora_k "${FLYLORA_K}"
      --flylora_alpha "${FLYLORA_ALPHA}"
      --flylora_sparsity_ratio "${FLYLORA_SPARSITY_RATIO}"
      --flylora_bias_lr "${FLYLORA_BIAS_LR}"
    )
    ;;
  v3)
    FLYLORA_R=${FLYLORA_R:-16}
    FLYLORA_K=${FLYLORA_K:-4}
    FLYLORA_ALPHA=${FLYLORA_ALPHA:-32}
    FLYLORA_SPARSITY_RATIO=${FLYLORA_SPARSITY_RATIO:-0.25}
    FLYLORA_BIAS_LR=${FLYLORA_BIAS_LR:-1e-3}
    FLYLORA_OUTPUT_MIX=${FLYLORA_OUTPUT_MIX:-${FUSION_ALPHA}}
    VERSION_ARGS=(
      --use_flylora_dual
      --flylora_r "${FLYLORA_R}"
      --flylora_k "${FLYLORA_K}"
      --flylora_alpha "${FLYLORA_ALPHA}"
      --flylora_sparsity_ratio "${FLYLORA_SPARSITY_RATIO}"
      --flylora_bias_lr "${FLYLORA_BIAS_LR}"
      --flylora_output_mix "${FLYLORA_OUTPUT_MIX}"
    )
    ;;
  v4)
    ;;
  *)
    echo "Unsupported VERSION_ID=${VERSION_ID}" >&2
    exit 1
    ;;
esac

echo "Version: ${VERSION_ID}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Output root: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec python "${ENTRY_FILE}" "${COMMON_ARGS[@]}" "${VERSION_ARGS[@]}"
