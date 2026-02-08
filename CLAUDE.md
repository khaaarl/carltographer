# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Carltographer generates random terrain layouts for Warhammer 40k using a genetic algorithm. It evolves a population of candidate table layouts through mutation (add/remove/move/rotate/modify terrain pieces), scores them on fitness criteria (piece counts, gap enforcement, no overlaps), and selects the best.

**v1/** contains the original Lua implementation (~1900 lines) that ran inside Tabletop Simulator. It is kept as reference but is not actively developed.

**v2/** is the active rewrite. See `v2/PLAN.md` for detailed architecture and design notes.

## v2 Architecture

The v2 code is split into two layers:

- **Engine** (deterministic, portable): Terrain data model, collision/validation, genetic algorithm. Takes a terrain catalog + generation params, produces a layout. All randomness from a seeded PRNG. This layer exists in both Python (`v2/engine/`) and Rust (`v2/engine_rs/`) with bit-identical output for the same seed.

  - **v2/engine/** (Python): Reference implementation, primary development target
    - `generate.py`: Main generation loop with mutation actions (add/move/delete)
    - `collision.py`: OBB-based collision detection, gap validation, distance calculations
    - `types.py`: Data model (Terrain Catalog, Layout, EngineParams, etc.)
    - `prng.py`: PCG32 PRNG for deterministic randomness

  - **v2/engine_rs/** (Rust): Feature-complete parity with Python, built with PyO3 for Python integration
    - Identical feature set: quantization, gap checking, feature count preferences, weighted action selection
    - Callable from Python: `import engine_rs; engine_rs.generate_json(params_json) -> result_json`
    - Build: `cd v2/engine_rs && maturin develop`

  - **v2/engine_cmp/** (Comparison Tool): Validates Python and Rust engines produce identical output
    - `compare.py`: 12 test scenarios covering all features, tolerance-based comparison
    - `compare_test.py`: pytest integration for CI/CD
    - Run: `python -m engine_cmp.compare --verbose` or `pytest engine_cmp/`
    - **Status**: All 12 scenarios pass ✓ (bit-identical parity achieved)

- **Frontend** (Python only): Terrain collection management (what pieces a player owns), UI, orchestration. Calls the engine via `engine.generate()` or (when built) `engine_rs.generate_json()`.

JSON specifications are the interchange format: a **terrain catalog** describes available piece types and their geometry, and a **layout** describes a specific arrangement of pieces on a table.

## Development Environment

**IMPORTANT: All Python commands should run from `v2/` as the root directory, not from the repo root.**

Python virtual environment: `v2/.env/` (Python 3.12).

Setup:
```bash
cd v2
source .env/bin/activate
```

Toolchain (always run from v2/):
- **Run UI**: `./scripts/run-ui.sh` (works from any directory)
- **Build Rust engine**: `python scripts/build_rust_engine.py` (cross-platform, works from v2/)
- **Package executable**: `python scripts/package_executable.py` (cross-platform, works from v2/)
- **pytest** for tests: `python -m pytest engine/`
- **ruff format**: `ruff format .` (auto-fixes)
- **isort** for import sorting: `isort .` (auto-fixes)

## Formatting and Linting (AUTONOMOUS, NO PERMISSION NEEDED)

**CRITICAL: After modifying ANY Python files, immediately and automatically run formatters WITHOUT asking.**

From v2/ directory:
```bash
isort .
ruff format .
```

**Do NOT ask for permission.** This must happen automatically after every Python file edit/write. Ensures code stays clean and pre-commit checks pass on first try.

## Branching (CRITICAL — DO THIS FIRST)

**NEVER make code changes directly on `main`.** Before writing ANY code, you MUST:

1. Create a feature branch: `git checkout -b <descriptive-branch-name>`
2. Verify you are on the feature branch: `git branch --show-current`
3. ONLY THEN start making changes

**This is non-negotiable.** If you realize you are on `main` and have already made changes, STOP immediately and ask the user how to proceed — do NOT commit to `main`.

The only exception is editing `CLAUDE.md` itself, which can be done on `main` if explicitly requested.

## Committing Code

ALWAYS ASK FOR PERMISSION BEFORE COMMITTING TO MAIN/MASTER, BUT COMMITTING TO FEATURE BRANCHES DOES NOT REQUIRE PERMISSION.

Before committing, run the full validation pipeline (from repo root):

```bash
# Step 1: Auto-format code (isort + ruff format) - from v2/
cd v2
isort .
ruff format .

# Step 2: Run full pre-commit hooks (from repo root)
cd ..
pre-commit run --all-files

# Step 3: If all hooks pass, commit
git add .
git commit -m "Your commit message"
```

If any hook fails in Step 2:
- Fix the issues autonomously
- Re-run `pre-commit run --all-files`
- Once all pass, proceed to commit

The repository is configured with:
- **isort**: Import organization (auto-fixes)
- **ruff format**: Code formatting (auto-fixes)
- **ruff (legacy)**: Linting validation (E/F/W rules)
- **type checker**: Type annotation validation
- **pytest**: Unit tests

**CRITICAL: Always ask user permission before committing. Do not commit autonomously.**

When ready to commit, inform the user of the changes and ask approval. The validation pipeline (formatting + pre-commit checks) should run automatically, but the actual `git commit` requires explicit user permission.

## Merging to Main

When the user asks to merge a feature branch to main, follow this procedure:

```bash
# 1. Pull latest main
git checkout main && git pull

# 2. Rebase feature branch onto latest main (CRITICAL: conflict detection happens here)
git checkout feature/my-branch
git rebase main
# If conflicts arise, resolve them manually, then: git rebase --continue

# 3. Create a temporary branch for squashing (preserves the original feature branch)
git checkout -b feature/my-branch-rebase feature/my-branch

# 4. Squash into a single commit
git reset --soft main
git commit -m "Unified commit message describing the feature"

# 5. Fast-forward merge into main
git checkout main
git merge feature/my-branch-rebase

# 6. Push and clean up temporary branch
git push
git branch -d feature/my-branch-rebase
```

**WARNING: Never skip step 2 (rebase).** `git reset --soft` does NOT detect conflicts. Without the rebase, it silently reverts any changes made to main after the feature branch was created. The rebase step ensures proper 3-way merge conflict detection before squashing.

The squashed commit message should summarize the entire feature, not repeat individual commit messages. Always ask the user before pushing to main.

## Key Constraints

- **Determinism**: The engine must produce identical results given the same seed. No hash-order dependence, no set iteration, no stdlib PRNG (use a portable PRNG like PCG/xoshiro implemented from scratch). This enables cross-language verification between Python and Rust implementations.

- **Quantization**: All terrain positions snap to 0.1" increments, all rotations snap to 15° increments. This is applied in both Add and Move actions.

- **Gap Enforcement**: Tall terrain (height >= 1") must maintain:
  - Minimum distance from other tall terrain (min_feature_gap_inches)
  - Minimum distance from table edges (min_edge_gap_inches)
  - Both are optional (None by default, skipped if <= 0)

- **Feature Count Preferences**: The engine supports min/max constraints on feature types via weighted action selection. Currently supports "obstacle" type; extensible to other types.

- **Engine purity**: The engine has no UI concerns, no asset URLs, no TTS-specific logic. It works purely with geometric abstractions.

## Maintaining Engine Parity (Critical for Rust Updates)

**MUST-DO when updating the Rust engine with new features:**

1. **Python engine first**: Implement and test new features in Python engine (`v2/engine/`) first. Ensure all Python tests pass: `python -m pytest v2/engine/ -v`

2. **Add comparison scenarios**: Add new test scenarios to `v2/engine_cmp/compare.py` that exercise the new feature. Each scenario should be added to the `TEST_SCENARIOS` list with representative values.
   - Example: If adding a new feature "X", create `TestScenario("with_X", seed=42, num_steps=50, ...)`
   - Make sure the scenario exercises both the feature enabled and disabled (if applicable)

3. **Implement in Rust**: Port the feature to Rust engine (`v2/engine_rs/src/`):
   - `types.rs`: Add any new data types or fields to EngineParams
   - `collision.rs`: Add any new collision/validation logic
   - `generate.rs`: Update the main loop and action handlers
   - Keep implementation as close to Python version as possible for maintainability

4. **Run comparison tests**: After implementing in Rust, MUST verify parity using the build script:
   ```bash
   # Automated: builds engine, runs tests, validates manifest (from v2/)
   python scripts/build_rust_engine.py

   # Or manually:
   cd v2/engine_rs && maturin develop
   python -m pytest engine_cmp/ -v
   ```
   - All tests must pass (18 scenarios currently, more if you added new ones)
   - Look for "18 passed, 0 failed" in output
   - If any test fails, inspect the diff output to find where engines diverge
   - Common issues: floating-point order, randomness, quantization rounding

5. **Update unit tests**: Ensure new Rust tests are added to cover the new feature:
   - Add to `v2/engine_rs/src/collision.rs` tests if collision-related
   - Add to `v2/engine_rs/src/generate.rs` tests if generation-related
   - Run: `cd v2/engine_rs && cargo test`

6. **Verify no regressions**:
   - Python: `python -m pytest v2/engine/ -v` (should still have 44+ tests)
   - Rust: `cargo test` (should still have 28+ tests)
   - Comparison: `python -m engine_cmp.compare` (18+ tests, all passing)

**Do NOT commit Rust engine changes without confirming parity.** The comparison tool is your verification that the engines are in sync.

## Build and Verification Scripts

**After any changes to Rust engine code or verification code, ALWAYS run (from v2/):**

```bash
python scripts/build_rust_engine.py
```

**What this script does:**
1. Validates Python venv and Rust toolchain are available
2. Compiles Rust engine with `maturin develop`
3. Runs all 18 engine parity comparison tests
4. Verifies hash manifest was written to `.engine_parity_manifest.json`
5. Exits with code 0 (success) or 1 (failure)

**Usage:**
- **Default (verbose)**: `python scripts/build_rust_engine.py`
- **Quiet mode**: `python scripts/build_rust_engine.py --quiet`

**Packaging (from v2/):**
```bash
python scripts/package_executable.py                   # full build
python scripts/package_executable.py --skip-rust-build # skip Rust rebuild
```

Auto-detects OS and architecture, names output accordingly (e.g. `carltographer-linux-x86_64`, `carltographer-windows-x86_64.exe`, `carltographer-mac-arm64`). Both scripts are cross-platform (Linux, macOS, Windows).
