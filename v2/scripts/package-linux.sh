#!/bin/bash

################################################################################
# package-linux.sh
#
# Packages Carltographer as a single-file Linux executable using PyInstaller:
# 1. Validates prerequisites (venv, PyInstaller)
# 2. Builds Rust engine (unless --skip-rust-build)
# 3. Locates engine_rs in site-packages
# 4. Runs PyInstaller with --onefile
# 5. Verifies output exists, reports file size
#
# Can be run from any directory. Exits with:
#   0 = packaging succeeded
#   1 = any step failed
#
# Usage:
#   ./package-linux.sh                  # full build including Rust engine
#   ./package-linux.sh --skip-rust-build  # skip Rust engine rebuild
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${V2_DIR}/.." && pwd)"

VENV_PYTHON="${V2_DIR}/.env/bin/python"
SKIP_RUST_BUILD=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --skip-rust-build)
            SKIP_RUST_BUILD="1"
            ;;
    esac
done

################################################################################
# Logging helpers
################################################################################

log_info() {
    echo "✓ $*" >&2
}

log_step() {
    echo "" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
    echo "  $*" >&2
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
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
}

ensure_pyinstaller() {
    log_step "Checking PyInstaller"

    if "$VENV_PYTHON" -c "import PyInstaller" 2>/dev/null; then
        log_info "PyInstaller already installed"
        return 0
    fi

    log_info "PyInstaller not found, installing..."
    if ! "$VENV_PYTHON" -m pip install pyinstaller; then
        log_error "Failed to install PyInstaller"
        return 1
    fi

    log_info "PyInstaller installed successfully"
}

################################################################################
# Build Rust engine
################################################################################

build_rust() {
    if [[ -n "$SKIP_RUST_BUILD" ]]; then
        log_step "Skipping Rust engine build (--skip-rust-build)"
        return 0
    fi

    log_step "Building Rust engine"

    if ! "${SCRIPT_DIR}/build-rust-engine.sh" --quiet; then
        log_error "Rust engine build/verification failed"
        return 1
    fi

    log_info "Rust engine built and verified"
}

################################################################################
# Locate engine_rs
################################################################################

find_site_packages() {
    SITE_PACKAGES=$("$VENV_PYTHON" -c "import site; print(site.getsitepackages()[0])")

    if [[ ! -d "$SITE_PACKAGES" ]]; then
        log_error "site-packages directory not found: $SITE_PACKAGES"
        return 1
    fi

    if [[ ! -d "${SITE_PACKAGES}/engine_rs" ]]; then
        log_error "engine_rs not found in site-packages at ${SITE_PACKAGES}/engine_rs"
        log_error "Run: v2/scripts/build-rust-engine.sh"
        return 1
    fi

    log_info "site-packages: $SITE_PACKAGES"
    log_info "engine_rs found: ${SITE_PACKAGES}/engine_rs"
}

################################################################################
# Run PyInstaller
################################################################################

run_pyinstaller() {
    log_step "Running PyInstaller"

    cd "${REPO_ROOT}"

    "$VENV_PYTHON" -m PyInstaller \
        --onefile \
        --name carltographer-linux-x86_64 \
        --strip \
        --paths "${REPO_ROOT}" \
        --paths "${SITE_PACKAGES}" \
        --hidden-import PIL._tkinter_finder \
        --collect-all engine_rs \
        --collect-submodules v2 \
        --distpath "${V2_DIR}/dist" \
        --workpath "${V2_DIR}/build/pyinstaller" \
        --specpath "${V2_DIR}/build/pyinstaller" \
        v2/packaging/entry_point.py

    log_info "PyInstaller completed"
}

################################################################################
# Verify output
################################################################################

verify_output() {
    log_step "Verifying output"

    local OUTPUT="${V2_DIR}/dist/carltographer-linux-x86_64"

    if [[ ! -f "$OUTPUT" ]]; then
        log_error "Expected output not found: $OUTPUT"
        return 1
    fi

    local SIZE
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    log_info "Output: $OUTPUT"
    log_info "Size: $SIZE"

    # Verify it's executable
    if [[ ! -x "$OUTPUT" ]]; then
        log_error "Output exists but is not executable"
        return 1
    fi

    log_info "Executable verified"
}

################################################################################
# Main
################################################################################

main() {
    log_step "Packaging Carltographer for Linux"
    log_info "v2_dir: $V2_DIR"
    log_info "repo_root: $REPO_ROOT"

    if ! check_venv; then
        return 1
    fi

    if ! ensure_pyinstaller; then
        return 1
    fi

    if ! build_rust; then
        return 1
    fi

    if ! find_site_packages; then
        return 1
    fi

    if ! run_pyinstaller; then
        return 1
    fi

    if ! verify_output; then
        return 1
    fi

    log_step "✓ Packaging complete!"
    log_info "Run: ${V2_DIR}/dist/carltographer-linux-x86_64"

    return 0
}

# Trap errors
trap 'log_error "Script failed at line $LINENO"; exit 1' ERR

main "$@"
