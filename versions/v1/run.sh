#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lgd/.conda/envs/hdrec/bin/python}"
exec "${PYTHON_BIN}" "${VERSION_DIR}/scripts/run_profile.py" "$@"
