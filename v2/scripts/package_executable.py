#!/usr/bin/env python3
"""Package Carltographer as a single-file executable using PyInstaller.

Cross-platform replacement for package-linux.sh. Detects the current OS and
architecture to name the output appropriately.

Steps:
  1. Validate Python venv exists
  2. Ensure PyInstaller is installed (auto-installs if missing)
  3. Build Rust engine via build_rust_engine.py (unless --skip-rust-build)
  4. Locate engine_rs in site-packages
  5. Run PyInstaller with --onefile
  6. Verify output exists, report file size

Output naming (without --version):
  carltographer-linux-x86_64
  carltographer-mac-arm64
  carltographer-windows-x86_64.exe

Output naming (with --version v20260211-143025):
  carltographer-v20260211-143025-linux-x86_64
  carltographer-v20260211-143025-mac-arm64
  carltographer-v20260211-143025-windows-x86_64.exe

Usage:
  python package_executable.py                                    # full build
  python package_executable.py --skip-rust-build                  # skip Rust rebuild
  python package_executable.py --version v20260211-143025         # versioned name

Exit codes:
  0 = packaging succeeded
  1 = any step failed
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
REPO_ROOT = V2_DIR.parent

IS_WINDOWS = platform.system() == "Windows"
VENV_PYTHON = (
    V2_DIR / ".env" / "Scripts" / "python.exe"
    if IS_WINDOWS
    else V2_DIR / ".env" / "bin" / "python"
)


def _normalize_arch(machine: str) -> str:
    """Normalize platform.machine() to a consistent architecture name."""
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    return m


def _os_name() -> str:
    s = platform.system()
    if s == "Linux":
        return "linux"
    if s == "Darwin":
        return "mac"
    if s == "Windows":
        return "windows"
    return s.lower()


def executable_name(version: str | None = None) -> str:
    """Build the output executable name for this platform.

    If *version* is given (e.g. "v20260211-143025"), the name becomes
    ``carltographer-v20260211-143025-linux-x86_64``.  Without a version it
    stays ``carltographer-linux-x86_64`` for backward-compatibility.
    """
    parts = ["carltographer"]
    if version:
        parts.append(version)
    parts.append(f"{_os_name()}-{_normalize_arch(platform.machine())}")
    name = "-".join(parts)
    if IS_WINDOWS:
        name += ".exe"
    return name


class Logger:
    def info(self, msg: str) -> None:
        print(f"\u2713 {msg}", file=sys.stderr)

    def step(self, msg: str) -> None:
        print(file=sys.stderr)
        print("\u2501" * 47, file=sys.stderr)
        print(f"  {msg}", file=sys.stderr)
        print("\u2501" * 47, file=sys.stderr)

    def error(self, msg: str) -> None:
        print(file=sys.stderr)
        print(f"\u2717 ERROR: {msg}", file=sys.stderr)
        print(file=sys.stderr)


log = Logger()


def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    defaults: dict[str, Any] = {"check": True}
    defaults.update(kwargs)
    return subprocess.run(args, **defaults)


def check_venv() -> bool:
    log.step("Checking Python virtual environment")

    if not VENV_PYTHON.exists():
        log.error(f"Python venv not found at {VENV_PYTHON}")
        log.error(f"Run: cd {V2_DIR} && python3 -m venv .env")
        return False

    log.info(f"Python venv found: {VENV_PYTHON}")
    return True


def ensure_pyinstaller() -> bool:
    log.step("Checking PyInstaller")

    try:
        run(
            [str(VENV_PYTHON), "-c", "import PyInstaller"],
            capture_output=True,
        )
        log.info("PyInstaller already installed")
        return True
    except subprocess.CalledProcessError:
        pass

    log.info("PyInstaller not found, installing...")
    try:
        run([str(VENV_PYTHON), "-m", "pip", "install", "pyinstaller"])
    except subprocess.CalledProcessError:
        log.error("Failed to install PyInstaller")
        return False

    log.info("PyInstaller installed successfully")
    return True


def build_rust(skip: bool) -> bool:
    if skip:
        log.step("Skipping Rust engine build (--skip-rust-build)")
        return True

    log.step("Building Rust engine")

    build_script = SCRIPT_DIR / "build_rust_engine.py"
    try:
        run([str(VENV_PYTHON), str(build_script), "--quiet"])
    except subprocess.CalledProcessError:
        log.error("Rust engine build/verification failed")
        return False

    log.info("Rust engine built and verified")
    return True


def check_engine_rs() -> bool:
    log.step("Checking engine_rs")

    try:
        run(
            [str(VENV_PYTHON), "-c", "import engine_rs"],
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        log.error("engine_rs not importable from the venv")
        log.error("Run: python build_rust_engine.py")
        return False

    log.info("engine_rs is importable")
    return True


def run_pyinstaller(name: str) -> bool:
    log.step("Running PyInstaller")

    # Strip .exe suffix for PyInstaller --name (it adds .exe on Windows)
    pyinstaller_name = name.removesuffix(".exe")

    # --add-data separator is OS-dependent
    data_sep = ";" if IS_WINDOWS else ":"
    catalogs_src = str(V2_DIR / "catalogs")
    catalogs_data = f"{catalogs_src}{data_sep}v2/catalogs"

    cmd = [
        str(VENV_PYTHON),
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        pyinstaller_name,
        "--paths",
        str(REPO_ROOT),
        "--hidden-import",
        "PIL._tkinter_finder",
        "--collect-all",
        "engine_rs",
        "--collect-submodules",
        "v2",
        "--add-data",
        catalogs_data,
        "--distpath",
        str(V2_DIR / "dist"),
        "--workpath",
        str(V2_DIR / "build" / "pyinstaller"),
        "--specpath",
        str(V2_DIR / "build" / "pyinstaller"),
    ]

    if IS_WINDOWS:
        # Suppress console window for GUI app
        cmd.append("--windowed")
    else:
        # --strip is only available on Unix
        cmd.append("--strip")

    cmd.append(str(REPO_ROOT / "v2" / "packaging" / "entry_point.py"))

    try:
        run(cmd, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError:
        log.error("PyInstaller failed")
        return False

    log.info("PyInstaller completed")
    return True


def verify_output(name: str) -> bool:
    log.step("Verifying output")

    output = V2_DIR / "dist" / name

    if not output.exists():
        log.error(f"Expected output not found: {output}")
        return False

    size_bytes = output.stat().st_size
    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{size_bytes / 1024:.1f} KB"

    log.info(f"Output: {output}")
    log.info(f"Size: {size_str}")

    # On Unix, verify it's executable
    if not IS_WINDOWS and not os.access(output, os.X_OK):
        log.error("Output exists but is not executable")
        return False

    log.info("Executable verified")
    return True


def run_smoke_test(name: str) -> bool:
    log.step("Running smoke test")

    output = V2_DIR / "dist" / name
    try:
        run([str(output), "--smoke-test"])
    except subprocess.CalledProcessError:
        log.error("Smoke test failed")
        return False

    log.info("Smoke test passed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package Carltographer as a single-file executable"
    )
    parser.add_argument(
        "--skip-rust-build",
        action="store_true",
        help="Skip Rust engine rebuild",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version tag to embed in the executable name (e.g. v20260211-143025)",
    )
    args = parser.parse_args()

    name = executable_name(args.version)
    log.step(f"Packaging Carltographer: {name}")
    log.info(f"v2_dir: {V2_DIR}")
    log.info(f"repo_root: {REPO_ROOT}")

    if not check_venv():
        return 1

    if not ensure_pyinstaller():
        return 1

    if not build_rust(skip=args.skip_rust_build):
        return 1

    if not check_engine_rs():
        return 1

    if not run_pyinstaller(name):
        return 1

    if not verify_output(name):
        return 1

    if not run_smoke_test(name):
        return 1

    log.step("\u2713 Packaging complete!")
    log.info(f"Run: {V2_DIR / 'dist' / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
