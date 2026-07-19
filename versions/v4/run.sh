#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${VERSION_DIR}/../run_common.sh" "${VERSION_DIR}" v4 "${VERSION_DIR}/model/main.py" v4 "$@"
