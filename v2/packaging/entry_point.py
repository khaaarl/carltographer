"""PyInstaller entry point for Carltographer.

PyInstaller needs a concrete .py file — can't use `python -m`.
"""

from v2.frontend.app import main

main()
