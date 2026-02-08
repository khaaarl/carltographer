#!/bin/bash
################################################################################
# run-coverage.sh
#
# Runs Python tests with coverage and generates reports.
# Produces a terminal summary and an HTML report in v2/htmlcov/.
#
# Usage:
#   v2/scripts/run-coverage.sh           # terminal + HTML report
#   v2/scripts/run-coverage.sh --html    # open HTML report in browser after
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_ACTIVATE="${V2_DIR}/.env/bin/activate"

# Check venv exists and activate it
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Error: Python venv not found at ${V2_DIR}/.env" >&2
    echo "Run: cd $V2_DIR && python3 -m venv .env" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

cd "$V2_DIR"

echo "Running tests with coverage..."
python -m coverage run -m pytest engine/ frontend/ "$@"

echo ""
echo "=== Coverage Report ==="
python -m coverage report

echo ""
echo "Generating HTML report..."
python -m coverage html

echo "HTML report: $V2_DIR/htmlcov/index.html"
