#!/usr/bin/env bash
set -euo pipefail

V1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_IDS=7
export PYTHONUNBUFFERED=1
exec bash "${V1_DIR}/run.sh" --dataset Video_Games --profile src_original --gpu 7
