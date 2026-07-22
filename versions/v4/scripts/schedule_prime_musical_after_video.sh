#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V4_DIR="${PROJECT_DIR}/versions/v4"
GPU=7
VIDEO_SCREEN="hdrec_v1_video_src"
PRIME_SCREEN="hdrec_v4_prime_candidate"
MUSICAL_SCREEN="hdrec_v4_musical_candidate"
PRIME_CKPT="${PROJECT_DIR}/versions/v1/outputs/Prime_Pantry/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_all_klw0.3_fa0.3/pytorch_model.bin"
MUSICAL_CKPT="${PROJECT_DIR}/versions/v1/outputs/Musical_Instruments/deepseek-ai-DeepSeek-R1-Distill-Llama-8B_all_klw0.7_fa0.7/pytorch_model.bin"

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]$1[[:space:]]"
}

while screen_exists "$VIDEO_SCREEN"; do
  sleep 60
done

mkdir -p "${V4_DIR}/logs"
screen -L -Logfile "${V4_DIR}/logs/prime_candidate.log" -dmS "$PRIME_SCREEN" \
  bash "${V4_DIR}/run.sh" --dataset Prime_Pantry --v1_checkpoint "$PRIME_CKPT" --gpu "$GPU"
while screen_exists "$PRIME_SCREEN"; do
  sleep 60
done

screen -L -Logfile "${V4_DIR}/logs/musical_candidate.log" -dmS "$MUSICAL_SCREEN" \
  bash "${V4_DIR}/run.sh" --dataset Musical_Instruments --v1_checkpoint "$MUSICAL_CKPT" --gpu "$GPU"
