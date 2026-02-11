"""PyInstaller entry point for Carltographer.

PyInstaller needs a concrete .py file — can't use `python -m`.

On Windows with --windowed, sys.stdout/stderr are None (no console).
Redirect to devnull so print() calls don't crash the app.
"""

import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

from v2.frontend.app import main

main()
