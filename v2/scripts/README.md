# Carltographer v2 Build Scripts

Utility scripts for building, testing, and verifying the Rust engine.

## build_rust_engine.py (recommended)

Cross-platform Python script that builds the Rust engine and verifies parity.
Run from `v2/` with the venv activated:

```bash
python scripts/build_rust_engine.py          # verbose (default)
python scripts/build_rust_engine.py --quiet  # minimal output
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
