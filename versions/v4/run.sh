#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${VERSION_DIR}/scripts/run_v4_candidate.sh" "$@"
