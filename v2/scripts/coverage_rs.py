#!/usr/bin/env python3
"""Run Rust unit tests with coverage via cargo-tarpaulin.

Produces a terminal summary and an HTML report.
Works on Linux (tarpaulin requirement). macOS/Windows users should use
cargo-llvm-cov instead.

Prerequisites:
    cargo install cargo-tarpaulin

Usage:
    python scripts/coverage_rs.py            # terminal + HTML report
    python scripts/coverage_rs.py --html     # also open HTML report in browser
"""

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
ENGINE_RS_DIR = V2_DIR / "engine_rs"
COV_DIR = V2_DIR / "coverage_rs"


def main() -> None:
    if not shutil.which("cargo"):
        print(
            "Error: cargo not found. Install Rust: https://rustup.rs/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check cargo-tarpaulin is installed
    result = subprocess.run(
        ["cargo", "tarpaulin", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            "Error: cargo-tarpaulin not found.\n"
            "Install it with: cargo install cargo-tarpaulin",
            file=sys.stderr,
        )
        sys.exit(1)

    COV_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Rust unit tests with coverage...")
    cmd = [
        "cargo",
        "tarpaulin",
        "--out",
        "Html",
        "--out",
        "Stdout",
        "--output-dir",
        str(COV_DIR),
    ]
    result = subprocess.run(cmd, cwd=str(ENGINE_RS_DIR))
    if result.returncode != 0:
        sys.exit(result.returncode)

    index = COV_DIR / "tarpaulin-report.html"
    print(f"\nHTML report: {index}")

    if "--html" in sys.argv[1:]:
        import webbrowser

        webbrowser.open(index.as_uri())


if __name__ == "__main__":
    main()
