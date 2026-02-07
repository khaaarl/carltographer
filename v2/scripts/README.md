# Carltographer v2 Build Scripts

Utility scripts for building, testing, and verifying the Rust engine.

## build-rust-engine.sh

Builds the Rust engine and verifies it matches the Python implementation.

**What it does:**
1. Validates Python venv is available and functional
2. Validates Rust toolchain is installed
3. Validates maturin is available
4. Compiles Rust engine with `maturin develop`
5. Runs Python/Rust comparison tests (12 parity scenarios)
6. Verifies hash manifest was written
7. Reports success or fails loudly

**Usage:**

```bash
# From any directory - verbose output (recommended for first runs)
/home/carlx/prog/carltographer/v2/scripts/build-rust-engine.sh

# Quiet mode - minimal output, good for CI/CD
/home/carlx/prog/carltographer/v2/scripts/build-rust-engine.sh --quiet

# From within the repo
cd /home/carlx/prog/carltographer/v2/scripts
./build-rust-engine.sh
```

**Exit codes:**
- `0` - All checks passed, engines are verified identical, manifest written
- `1` - Any step failed (venv broken, Rust missing, build failed, tests failed, etc.)

**Prerequisites:**
- Python 3.12+ with venv at `v2/.env/`
- Rust toolchain (rustc, cargo)
- maturin installed in venv
- All Python test dependencies

**What gets written:**
- Rebuilds and reinstalls engine_rs Python binding
- Writes/updates `.engine_parity_manifest.json` with SHA256 hashes of all Python engine source files
  - Used by frontend to auto-select which engine to use
  - Certifies Rust engine has been verified bit-identical to Python

**Example output:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Engine Parity Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Python venv found: /home/carlx/prog/carltographer/v2/.env/bin/python
✓ Rust toolchain found: rustc 1.93.0
✓ maturin found
✓ Rust engine built and installed to venv
✓ All comparison tests passed
✓ Hash manifest exists: /home/carlx/prog/carltographer/v2/.engine_parity_manifest.json
✓ Manifest contains hashes for 5 engine files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ All checks passed!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Troubleshooting

**"pip is broken in venv"**
- Symptom: Script fails at venv check
- Fix: Reinstall pip: `cd v2 && python -m ensurepip --upgrade`
- Or: Clear and reinstall: `rm -rf v2/.env/lib/python3.12/site-packages/pip* && python -m ensurepip`

**"Rust toolchain not found"**
- Install from https://rustup.rs/

**"maturin develop failed"**
- Try: `cd v2 && .env/bin/pip install --upgrade maturin`

**Tests fail with "engine_rs not found"**
- The Rust build may have failed silently
- Check the full output (remove `--quiet` flag)
- Verify Rust code compiles: `cd v2/engine_rs && cargo build`
