#!/usr/bin/env python3
"""Build Rust engine and verify parity with Python engine.

Cross-platform replacement for build-rust-engine.sh. Can be run from any
directory.

Steps:
  1. Validate Python venv exists
  2. Check Rust toolchain (rustc, cargo)
  3. Check maturin is installed
  4. Build Rust engine with maturin develop --release
  5. Run Python/Rust comparison tests
  6. Verify hash manifest was written

Usage:
  python build_rust_engine.py           # verbose output
  python build_rust_engine.py --quiet   # minimal output

Exit codes:
  0 = all tests passed, manifest written
  1 = any step failed
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
REPO_ROOT = V2_DIR.parent
MANIFEST_FILE = V2_DIR / ".engine_parity_manifest.json"

IS_WINDOWS = platform.system() == "Windows"
VENV_PYTHON = (
    V2_DIR / ".env" / "Scripts" / "python.exe"
    if IS_WINDOWS
    else V2_DIR / ".env" / "bin" / "python"
)


class Logger:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def info(self, msg: str) -> None:
        if not self.quiet:
            print(f"\u2713 {msg}", file=sys.stderr)

    def step(self, msg: str) -> None:
        if not self.quiet:
            print(file=sys.stderr)
            print(
                "\u2501" * 47,
                file=sys.stderr,
            )
            print(f"  {msg}", file=sys.stderr)
            print(
                "\u2501" * 47,
                file=sys.stderr,
            )

    def error(self, msg: str) -> None:
        print(file=sys.stderr)
        print(f"\u2717 ERROR: {msg}", file=sys.stderr)
        print(file=sys.stderr)


log = Logger()


def run(  # type: ignore[no-any-explicit]
    args: list[str], **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Run a command, letting stdout/stderr pass through unless quiet."""
    defaults: dict[str, Any] = {
        "check": True,
    }
    if log.quiet:
        defaults["capture_output"] = True
    defaults.update(kwargs)
    return subprocess.run(args, **defaults)


def check_venv() -> bool:
    log.step("Checking Python virtual environment")

    if not VENV_PYTHON.exists():
        log.error(f"Python venv not found at {VENV_PYTHON}")
        log.error(f"Run: cd {V2_DIR} && python3 -m venv .env")
        return False

    log.info(f"Python venv found: {VENV_PYTHON}")

    # Check pip works
    try:
        run([str(VENV_PYTHON), "-m", "pip", "--version"], capture_output=True)
    except subprocess.CalledProcessError:
        log.error("pip is broken in venv")
        log.error(
            f"Try: cd {V2_DIR} && .env/bin/python -m pip install --upgrade pip"
        )
        return False

    return True


def check_rust() -> bool:
    log.step("Checking Rust toolchain")

    for cmd in ("rustc", "cargo"):
        try:
            result = subprocess.run(
                [cmd, "--version"], capture_output=True, text=True, check=True
            )
            log.info(f"{cmd} found: {result.stdout.strip()}")
        except FileNotFoundError:
            log.error(f"{cmd} not found in PATH")
            log.error("Install from https://rustup.rs/")
            return False
        except subprocess.CalledProcessError:
            log.error(f"{cmd} --version failed")
            return False

    return True


def check_maturin() -> bool:
    log.step("Checking maturin")

    try:
        run(
            [str(VENV_PYTHON), "-c", "import maturin"],
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        log.error("maturin not installed in venv")
        log.error(f"Run: cd {V2_DIR} && .env/bin/pip install maturin")
        return False

    log.info("maturin found")
    return True


def build_rust_engine() -> bool:
    log.step("Building Rust engine")

    engine_rs_dir = V2_DIR / "engine_rs"
    if not engine_rs_dir.is_dir():
        log.error(f"engine_rs directory not found at {engine_rs_dir}")
        return False

    full_env = {
        **os.environ,
        "VIRTUAL_ENV": str(V2_DIR / ".env"),
        "PATH": str(VENV_PYTHON.parent)
        + (";" if IS_WINDOWS else ":")
        + os.environ.get("PATH", ""),
    }

    try:
        run(
            [str(VENV_PYTHON), "-m", "maturin", "develop", "--release"],
            cwd=str(engine_rs_dir),
            env=full_env,
        )
    except subprocess.CalledProcessError:
        # Try with --uv flag
        log.info("Standard pip not available, trying with --uv flag...")
        try:
            run(
                [
                    str(VENV_PYTHON),
                    "-m",
                    "maturin",
                    "develop",
                    "--release",
                    "--uv",
                ],
                cwd=str(engine_rs_dir),
                env=full_env,
            )
        except subprocess.CalledProcessError:
            log.error("maturin develop failed")
            return False

    log.info("Rust engine built and installed to venv")
    return True


def run_comparison_tests(
    newest_first: bool = False,
    fail_fast: bool = False,
) -> bool:
    log.step("Running engine comparison tests")

    cmd = [str(VENV_PYTHON), "-m", "engine_cmp.compare"]
    if not log.quiet:
        cmd.append("--verbose")
    if newest_first:
        cmd.append("--newest-first")
    if fail_fast:
        cmd.append("--fail-fast")

    try:
        run(cmd, cwd=str(V2_DIR))
    except subprocess.CalledProcessError:
        log.error("Comparison tests failed")
        return False

    log.info("All comparison tests passed")
    return True


def verify_manifest() -> bool:
    log.step("Verifying hash manifest")

    if not MANIFEST_FILE.exists():
        log.error(f"Hash manifest not found at {MANIFEST_FILE}")
        log.error(
            "Comparison tests may have passed, but manifest was not written"
        )
        return False

    # Validate JSON
    try:
        with open(MANIFEST_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        log.error("Hash manifest exists but is not valid JSON")
        return False

    file_count = len(data.get("engine_files", {}))
    log.info(f"Hash manifest exists: {MANIFEST_FILE}")
    log.info(f"Manifest contains hashes for {file_count} engine files")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Rust engine and verify parity with Python engine"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--newest-first",
        action="store_true",
        help="Run comparison scenarios newest first",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first comparison failure",
    )
    args = parser.parse_args()

    global log
    log = Logger(quiet=args.quiet)

    log.step("Engine Parity Verification")
    log.info(f"v2_dir: {V2_DIR}")
    log.info(f"repo_root: {REPO_ROOT}")

    steps: list = [
        check_venv,
        check_rust,
        check_maturin,
        build_rust_engine,
        lambda: run_comparison_tests(
            newest_first=args.newest_first,
            fail_fast=args.fail_fast,
        ),
        verify_manifest,
    ]

    for step in steps:
        if not step():
            return 1

    log.step("\u2713 All checks passed!")
    log.info("Engines verified as bit-identical")
    log.info(f"Manifest written to: {MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
