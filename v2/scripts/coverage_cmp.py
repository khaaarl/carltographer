#!/usr/bin/env python3
"""Run engine parity comparison tests with coverage.

Measures how much of the Python engine code is exercised by the 30 parity
scenarios in engine_cmp/. This shows which code paths the parity tests
cover vs. what only unit tests reach.

TODO: Add Rust-side coverage via cargo-llvm-cov. Build the PyO3 .so with
      RUSTFLAGS="-C instrument-coverage", run the parity tests, then
      post-process the .profraw with llvm-profdata/llvm-cov to produce a
      Rust coverage report alongside the Python one.

Works on Linux, macOS, and Windows.

Usage:
    python scripts/coverage_cmp.py            # terminal + HTML report
    python scripts/coverage_cmp.py --html     # also open HTML report in browser
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
COV_DIR = V2_DIR / "coverage_cmp"


def find_venv_python() -> Path:
    """Return the venv Python interpreter path, or exit with an error."""
    candidates = [
        V2_DIR / ".env" / "bin" / "python",
        V2_DIR / ".env" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    print(
        f"Error: Python venv not found at {V2_DIR / '.env'}", file=sys.stderr
    )
    print(f"Run: cd {V2_DIR} && python3 -m venv .env", file=sys.stderr)
    sys.exit(1)


def run(python: Path, *args: str) -> None:
    """Run a command via the venv Python, exiting on failure."""
    result = subprocess.run([str(python), *args], cwd=str(V2_DIR))
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    venv_python = find_venv_python()
    data_file = str(COV_DIR / ".coverage")

    print("Running parity comparison tests with coverage...")
    run(
        venv_python,
        "-m",
        "coverage",
        "run",
        f"--data-file={data_file}",
        "--source=engine",
        "-m",
        "pytest",
        "engine_cmp/",
    )

    print("\n=== Coverage Report (parity tests → Python engine) ===")
    run(venv_python, "-m", "coverage", "report", f"--data-file={data_file}")

    html_dir = str(COV_DIR / "html")
    print("\nGenerating HTML report...")
    run(
        venv_python,
        "-m",
        "coverage",
        "html",
        f"--data-file={data_file}",
        f"--directory={html_dir}",
    )

    index = COV_DIR / "html" / "index.html"
    print(f"HTML report: {index}")

    if "--html" in sys.argv[1:]:
        import webbrowser

        webbrowser.open(index.as_uri())


if __name__ == "__main__":
    main()
