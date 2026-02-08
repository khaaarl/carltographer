# Cross-Platform Packaging with GitHub Actions

## Overview

Use a GitHub Actions workflow with `workflow_dispatch` to build executables
for all platforms on demand. Click "Run workflow" in the GitHub UI, and it
spins up parallel runners for each OS/architecture, runs PyInstaller, and
uploads the binaries as artifacts (or attaches them to a release).

## Workflow Matrix

```yaml
name: Package Executables

on:
  workflow_dispatch:  # manual trigger via GitHub UI
  release:
    types: [created]  # also run on new releases

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            name: linux-x86_64
          - os: ubuntu-24.04-arm
            name: linux-arm64
          - os: macos-latest
            name: mac-arm64
          - os: macos-13
            name: mac-x86_64
          - os: windows-latest
            name: windows-x86_64

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - uses: dtolnay/rust-toolchain@stable

      - name: Install dependencies
        run: |
          python -m venv v2/.env
          v2/.env/bin/python -m pip install -r v2/requirements.txt
          v2/.env/bin/python -m pip install -r v2/requirements-dev.txt

      - name: Package executable
        run: v2/.env/bin/python v2/scripts/package_executable.py

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: carltographer-${{ matrix.name }}
          path: v2/dist/carltographer-*
```

## Notes

### Runner minutes (private repos)

GitHub free tier: 2,000 minutes/month for private repos.
Minute multipliers by OS:
- Linux: 1x
- Windows: 2x
- macOS: 10x

A full 5-platform build might use ~50 real minutes = ~80 billed minutes.
Public repos get unlimited minutes.

### macOS code signing

Without signing, macOS Gatekeeper blocks downloaded executables.
Users would need to right-click → Open → confirm, or `xattr -d com.apple.quarantine`.

For proper signing:
1. Apple Developer account ($99/year)
2. Store signing certificate + password as GitHub Secrets
3. Add a signing step after PyInstaller:
   `codesign --deep --force --sign "$CERT_ID" v2/dist/carltographer-mac-*`
4. Optionally notarize with `xcrun notarytool`

This can be deferred — unsigned binaries work, they just show a warning.

### Windows considerations

- No code signing friction like macOS (SmartScreen may warn, but users can click through)
- Output is `.exe` automatically
- MSVC runtime is pre-installed on Windows 10+

### Release automation

To auto-attach binaries to a GitHub Release:
1. Create a release (tag push or manual)
2. The workflow triggers on `release: [created]`
3. Add a step using `gh release upload` or `softprops/action-gh-release@v2`

```yaml
      - name: Upload to release
        if: github.event_name == 'release'
        uses: softprops/action-gh-release@v2
        with:
          files: v2/dist/carltographer-*
```

### arm64 Linux

`ubuntu-24.04-arm` runners are available but may have limited availability.
Alternative: use QEMU emulation via `docker/setup-qemu-action`, but it's
much slower (~5-10x). For a project this size it's probably fine.
