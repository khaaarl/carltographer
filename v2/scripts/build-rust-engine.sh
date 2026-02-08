#!/bin/bash

################################################################################
# build-rust-engine.sh
#
# Builds Rust engine and verifies it matches Python implementation:
# 1. Compiles Rust engine with maturin
# 2. Runs Python/Rust comparison tests
# 3. Validates hash manifest was written
#
# Can be run from any directory. Exits with:
#   0 = all tests passed, manifest written
#   1 = any step failed
#
# Usage:
#   ./build-rust-engine.sh           # uses verbose output
#   ./build-rust-engine.sh --quiet   # minimal output
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_DIR}/.." && pwd)"

QUIET="${1:-}"
VENV_PYTHON="${V2_DIR}/.env/bin/python"
MANIFEST_FILE="${V2_DIR}/.engine_parity_manifest.json"

################################################################################
# Logging helpers
################################################################################

log_info() {
    if [[ -z "$QUIET" ]]; then
        echo "✓ $*" >&2
    fi
}

log_step() {
    if [[ -z "$QUIET" ]]; then
        echo "" >&2
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
        echo "  $*" >&2
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    fi
}

log_error() {
    echo "" >&2
    echo "✗ ERROR: $*" >&2
    echo "" >&2
}

################################################################################
# Validation
################################################################################

check_venv() {
    log_step "Checking Python virtual environment"

    if [[ ! -f "$VENV_PYTHON" ]]; then
        log_error "Python venv not found at $VENV_PYTHON"
        log_error "Run: cd $V2_DIR && python3 -m venv .env"
        return 1
    fi

    log_info "Python venv found: $VENV_PYTHON"

    # Check if pip works
    if ! "$VENV_PYTHON" -m pip --version > /dev/null 2>&1; then
        log_error "pip is broken in venv"
        log_error "Try: cd $V2_DIR && .env/bin/python -m pip install --upgrade pip"
        return 1
    fi
}

check_rust() {
    log_step "Checking Rust toolchain"

    if ! command -v rustc &> /dev/null; then
        log_error "Rust toolchain not installed"
        log_error "Install from https://rustup.rs/"
        return 1
    fi

    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found in PATH"
        return 1
    fi

    log_info "Rust toolchain found: $(rustc --version)"
    log_info "Cargo found: $(cargo --version)"
}

check_maturin() {
    log_step "Checking maturin"

    if ! "$VENV_PYTHON" -c "import maturin" 2>/dev/null; then
        log_error "maturin not installed in venv"
        log_error "Run: cd $V2_DIR && .env/bin/pip install maturin"
        return 1
    fi

    log_info "maturin found"
}

################################################################################
# Build Rust engine
################################################################################

build_rust_engine() {
    log_step "Building Rust engine"

    if [[ ! -d "${V2_DIR}/engine_rs" ]]; then
        log_error "engine_rs directory not found at ${V2_DIR}/engine_rs"
        return 1
    fi

    cd "${V2_DIR}/engine_rs"

    # Activate venv so maturin knows where to install
    export VIRTUAL_ENV="${V2_DIR}/.env"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"

    # Attempt normal maturin develop
    local maturin_output
    if maturin_output=$("$VENV_PYTHON" -m maturin develop --release 2>&1); then
        # Success
        :
    elif echo "$maturin_output" | grep -q "Failed to find pip"; then
        # Try with --uv flag for uv-based venvs
        log_info "Standard pip not available, trying with --uv flag..."
        if ! "$VENV_PYTHON" -m maturin develop --release --uv; then
            log_error "maturin develop failed with --uv flag"
            if [[ -z "$QUIET" ]]; then
                echo "$maturin_output" >&2
            fi
            return 1
        fi
    else
        # Some other error
        log_error "maturin develop failed"
        if [[ -z "$QUIET" ]]; then
            echo "$maturin_output" >&2
        fi
        return 1
    fi

    log_info "Rust engine built and installed to venv"
}

################################################################################
# Run comparison tests
################################################################################

run_comparison_tests() {
    log_step "Running engine comparison tests"

    cd "${V2_DIR}"

    # Run compare.py which runs all scenarios and writes manifest on success
    if [[ -z "$QUIET" ]]; then
        if ! "$VENV_PYTHON" -m engine_cmp.compare --verbose; then
            log_error "Comparison tests failed"
            return 1
        fi
    else
        if ! "$VENV_PYTHON" -m engine_cmp.compare; then
            log_error "Comparison tests failed"
            return 1
        fi
    fi

    log_info "All comparison tests passed"
}

################################################################################
# Verify manifest
################################################################################

verify_manifest() {
    log_step "Verifying hash manifest"

    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_error "Hash manifest not found at $MANIFEST_FILE"
        log_error "Comparison tests may have passed, but manifest was not written"
        return 1
    fi

    # Validate it's valid JSON
    if ! "$VENV_PYTHON" -c "import json; json.load(open('$MANIFEST_FILE'))" 2>/dev/null; then
        log_error "Hash manifest exists but is not valid JSON"
        return 1
    fi

    # Show manifest info
    FILE_COUNT=$("$VENV_PYTHON" -c "import json; m=json.load(open('$MANIFEST_FILE')); print(len(m.get('engine_files', {})))" 2>/dev/null || echo "0")

    log_info "Hash manifest exists: $MANIFEST_FILE"
    log_info "Manifest contains hashes for $FILE_COUNT engine files"
}

################################################################################
# Main
################################################################################

main() {
    log_step "Engine Parity Verification"
    log_info "v2_dir: $V2_DIR"
    log_info "repo_root: $REPO_ROOT"

    if ! check_venv; then
        return 1
    fi

    if ! check_rust; then
        return 1
    fi

    if ! check_maturin; then
        return 1
    fi

    if ! build_rust_engine; then
        return 1
    fi

    if ! run_comparison_tests; then
        return 1
    fi

    if ! verify_manifest; then
        return 1
    fi

    log_step "✓ All checks passed!"
    log_info "Engines verified as bit-identical"
    log_info "Manifest written to: $MANIFEST_FILE"

    return 0
}

# Trap errors
trap 'log_error "Script failed at line $LINENO"; exit 1' ERR

main "$@"
