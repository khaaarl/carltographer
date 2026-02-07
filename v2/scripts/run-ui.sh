#!/bin/bash
################################################################################
# run-ui.sh
#
# Runs the Carltographer UI from anywhere.
# Handles venv activation and directory setup automatically.
#
# Usage:
#   v2/scripts/run-ui.sh
#   /path/to/v2/scripts/run-ui.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${V2_DIR}/.env/bin/python"

# Check venv exists
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "Error: Python venv not found at $VENV_PYTHON" >&2
    echo "Run: cd $V2_DIR && python3 -m venv .env" >&2
    exit 1
fi

# Activate venv environment variables
export VIRTUAL_ENV="${V2_DIR}/.env"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Change to repo root (parent of v2) so v2 is recognized as a package
cd "$(dirname "$V2_DIR")"
export PYTHONPATH="${V2_DIR}:${PYTHONPATH:-}"
exec "$VENV_PYTHON" -m v2.frontend.app "$@"
