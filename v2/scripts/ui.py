#!/usr/bin/env python3
"""Run the Carltographer UI from anywhere.

Handles venv discovery and directory setup automatically.
Works on Linux, macOS, and Windows.

Usage:
    python v2/scripts/run_ui.py
    v2/scripts/run_ui.py          # if executable
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent


def find_venv_python() -> Path:
    """Return the venv Python interpreter path, or exit with an error."""
    # Windows: Scripts/python.exe, Unix: bin/python
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


def main() -> None:
    venv_python = find_venv_python()
    venv_root = venv_python.parent.parent  # .env/

    repo_root = V2_DIR.parent

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_root)
    env["PYTHONPATH"] = str(V2_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    # Prepend venv bin dir to PATH
    venv_bin = str(venv_python.parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    os.chdir(repo_root)
    os.execve(
        str(venv_python),
        [str(venv_python), "-m", "v2.frontend.app", *sys.argv[1:]],
        env,
    )


if __name__ == "__main__":
    main()
