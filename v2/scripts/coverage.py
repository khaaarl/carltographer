#!/usr/bin/env python3
"""Run Python tests with coverage and generate reports.

Produces a terminal summary and an HTML report in v2/htmlcov/.
Works on Linux, macOS, and Windows.

Usage:
    python v2/scripts/run_coverage.py            # terminal + HTML report
    python v2/scripts/run_coverage.py --html     # open HTML report in browser after
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent


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
    result = subprocess.run(
        [str(python), *args],
        cwd=str(V2_DIR),
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    venv_python = find_venv_python()

    print("Running tests with coverage...")
    run(
        venv_python,
        "-m",
        "coverage",
        "run",
        "-m",
        "pytest",
        "engine/",
        "frontend/",
    )

    print("\n=== Coverage Report ===")
    run(venv_python, "-m", "coverage", "report")

    print("\nGenerating HTML report...")
    run(venv_python, "-m", "coverage", "html")

    htmlcov = V2_DIR / "htmlcov" / "index.html"
    print(f"HTML report: {htmlcov}")

    if "--html" in sys.argv[1:]:
        import webbrowser

        webbrowser.open(htmlcov.as_uri())


if __name__ == "__main__":
    main()
