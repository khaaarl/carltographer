# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Carltographer generates random terrain layouts for Warhammer 40k. It uses parallel tempering (simulated annealing across multiple temperature chains) to evolve terrain layouts through mutation (add/move/delete/replace/rotate pieces), scoring them on feature counts, gap enforcement, and line-of-sight visibility metrics.

**v1/** contains the original Lua implementation (~1900 lines) that ran inside Tabletop Simulator. It is kept as reference but is not actively developed.

**v2/** is the active rewrite. `v2/PLAN.md` is the original design document (historical).

## v2 Architecture

The v2 code is split into two layers:

- **Engine** (deterministic, portable): Terrain data model, collision/validation, genetic algorithm. Takes a terrain catalog + generation params, produces a layout. All randomness from a seeded PRNG. This layer exists in both Python (`v2/engine/`) and Rust (`v2/engine_rs/`) with bit-identical output for the same seed.

  - **v2/engine/** (Python): Reference implementation, primary development target
    - `generate.py`: Main generation loop with 5 mutation actions (add/move/delete/replace/rotate), two-phase scoring, parallel tempering integration
    - `collision.py`: OBB-based collision detection, gap validation, distance calculations
    - `visibility.py`: Angular-sweep line-of-sight analysis with caching (overall, per-DZ, cross-DZ, objective hidability)
    - `tempering.py`: Parallel tempering with simulated annealing and replica exchange
    - `types.py`: Data model (Terrain Catalog, Layout, EngineParams, Mission, ScoringTargets, etc.)
    - `prng.py`: PCG32 PRNG for deterministic randomness

  - **v2/engine_rs/** (Rust): Feature-complete parity with Python, built with PyO3 for Python integration
    - Identical feature set: all mutations, quantization, gap checking, feature count preferences, visibility scoring, parallel tempering, tile-biased placement
    - Callable from Python: `import engine_rs; engine_rs.generate_json(params_json) -> result_json`
    - Build: `cd v2/engine_rs && maturin develop`

  - **v2/engine_cmp/** (Comparison Tool): Validates Python and Rust engines produce identical output
    - `compare.py`: 30 test scenarios covering all features
    - `compare_test.py`: pytest integration for CI/CD
    - Run: `python -m engine_cmp.compare --verbose` or `pytest engine_cmp/`
    - **Status**: All 30 scenarios pass (bit-identical parity achieved)

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
v2/.env/bin/isort .
v2/.env/bin/ruff format .
```

**Do NOT ask for permission.** This must happen automatically after every Python file edit/write. Ensures code stays clean and pre-commit checks pass on first try.

**Rust files:** After modifying Rust files in `v2/engine_rs/`, run from that directory:
```bash
cd v2/engine_rs && cargo fmt
```

The Rust engine is configured with `warnings = "deny"` and `clippy::all = "deny"` in `Cargo.toml [lints]`, so compiler warnings and clippy lints are compile errors. Pre-commit hooks also run `cargo fmt --check`, `cargo clippy`, and `cargo test` on Rust file changes.

## Branching (CRITICAL — DO THIS FIRST)

**NEVER make code changes directly on `main`.** Before writing ANY code, you MUST:

1. Create a feature branch: `git checkout -b <descriptive-branch-name>`
2. Verify you are on the feature branch: `git branch --show-current`
3. ONLY THEN start making changes

**This is non-negotiable.** If you realize you are on `main` and have already made changes, STOP immediately and ask the user how to proceed — do NOT commit to `main`.

The only exception is editing `CLAUDE.md` itself, which can be done on `main` if explicitly requested. However, do NOT commit or push CLAUDE.md changes until the user explicitly says to — they may want to review or iterate first.

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
# 1. Create a temporary branch and squash all feature commits into one
#    (This way conflicts only need to be resolved once, not per-commit)
#    IMPORTANT: The REAL commit message goes HERE — step 4 is a fast-forward
#    merge which does NOT create a new commit, so any -m there is ignored.
git checkout -b feature/my-branch-rebase feature/my-branch
git reset --soft $(git merge-base main feature/my-branch-rebase)
git commit -m "Your descriptive commit message here"

# 2. Pull latest main
git checkout main && git pull

# 3. Rebase the single squashed commit onto main (conflict detection here)
git checkout feature/my-branch-rebase
git rebase main
# If conflicts arise, resolve them carefully, then: git add <files> && git rebase --continue

# 4. Fast-forward merge into main (no new commit — just moves the pointer)
git checkout main
git merge feature/my-branch-rebase

# 5. Push and clean up
git push
git branch -d feature/my-branch-rebase
```

**Why squash first, then rebase?** Rebasing a multi-commit branch onto main can require resolving the same conflict repeatedly (once per commit). By squashing into one commit first, you only resolve conflicts once. The `git reset --soft $(git merge-base ...)` in step 1 is safe — it collapses our own feature commits back to the branch point, without touching main's state. The rebase in step 3 then does proper 3-way conflict detection against latest main.

**Handling rebase conflicts:** When `git rebase main` reports conflicts:
1. Run `git status` to see which files conflict
2. Read the conflicting files — look for `<<<<<<<`, `=======`, `>>>>>>>` markers
3. Resolve by editing to keep the correct version of each section
4. `git add <resolved-files> && git rebase --continue`
5. After rebase completes, verify the code still works (run tests)
6. **If conflicts required non-trivial edits** (e.g., integrating two features that touch the same code), ask the user for permission before completing the merge. Truly trivial conflicts (e.g., both sides added adjacent lines with no semantic interaction) can be resolved and merged without asking.

The squashed commit message should summarize the entire feature, not repeat individual commit messages. Always ask the user before pushing to main.

## Key Constraints

- **Determinism**: The engine must produce identical results given the same seed. No hash-order dependence, no set iteration, no stdlib PRNG (use a portable PRNG like PCG/xoshiro implemented from scratch). This enables cross-language verification between Python and Rust implementations.

- **Quantization**: All terrain positions snap to 0.1" increments, all rotations snap to 15° increments. Applied in Add, Move, Replace, and Rotate actions.

- **Gap Enforcement**: Tall terrain (height >= 1") must maintain:
  - Minimum distance from other tall terrain (min_feature_gap_inches)
  - Minimum distance from table edges (min_edge_gap_inches)
  - Both are optional (None by default, skipped if <= 0)

- **Feature Count Preferences**: Min/max constraints on feature types (obstacle, ruins) via weighted action selection. Soft constraints — the engine biases mutation probabilities rather than hard-rejecting.

- **Visibility Scoring**: Two-phase scoring: phase 1 drives toward target feature counts, phase 2 optimizes line-of-sight metrics (overall visibility %, per-DZ visibility, cross-DZ sightlines, objective hidability). Configurable targets with weights.

- **Parallel Tempering**: Multiple replicas at different temperatures with periodic replica exchange. Cold chain does greedy hill-climbing; hot chains explore freely. Temperature-aware move distances (small at cold, large at hot).

- **Rotational Symmetry**: Optional mode where features placed off-center are mirrored at 180° for balanced gameplay. Mirror features count toward preferences.

- **Engine purity**: The engine has no UI concerns, no asset URLs, no TTS-specific logic. It works purely with geometric abstractions.

## Engine Development: Test-Driven Workflow (CRITICAL)

**Applies to:** Bug fixes and new features that affect engine behavior. Both Python and Rust engines must be updated together using TDD.

**Does NOT apply to:** Performance optimizations that only touch one language (e.g., optimizing a hot loop in Rust without changing behavior). Those can be developed and tested in isolation — just verify existing parity tests still pass afterward.

### Phase 1: Python (Red → Green)

All commands run from `v2/`.

1. **Write a failing Python unit test** in `v2/engine/` that captures the bug or specifies the new behavior.
   ```bash
   python -m pytest engine/ -v
   ```
   Confirm the new test **fails**.

2. **Write Python code** to make the test pass.
   ```bash
   python -m pytest engine/ -v
   ```
   Confirm the new test **passes** and no existing tests regress.

3. Repeat steps 1–2 as needed until the Python side is complete.

### Phase 2: Parity Comparison (Red)

4. **Add new comparison scenarios** to `v2/engine_cmp/compare.py` that exercise the new/fixed behavior. Each scenario goes in the `TEST_SCENARIOS` list.
   - Example: `TestScenario("with_X", seed=42, num_steps=50, ...)`
   - Cover both enabled and disabled states if applicable.
   - **Skip visibility checks** (`check_visibility=False`) if the change is not visibility-related — this keeps parity tests fast and focused.

5. **Build the Rust engine and run parity tests** — confirm the new scenarios **fail** (Rust doesn't have the change yet):
   ```bash
   python scripts/build_rust_engine.py
   ```
   New scenarios should fail; existing scenarios should still pass.

### Phase 3: Rust (Red → Green)

6. **Write a failing Rust unit test** that mirrors the Python test:
   - `v2/engine_rs/src/collision.rs` tests for collision/validation logic
   - `v2/engine_rs/src/generate.rs` tests for generation logic
   ```bash
   cd v2/engine_rs && cargo test
   ```
   Confirm the new test **fails**.

7. **Write Rust code** to make the test pass. Keep implementation close to the Python version for maintainability.
   - Key files: `types.rs` (data model), `collision.rs` (collision/validation), `generate.rs` (main loop/actions)
   ```bash
   cd v2/engine_rs && cargo test
   ```
   Confirm the new test **passes** and no existing tests regress.

8. Repeat steps 6–7 as needed.

### Phase 4: Verify Full Parity (Green)

9. **Run the full build-and-verify pipeline** (from v2/):
   ```bash
   python scripts/build_rust_engine.py
   ```
   ALL parity scenarios must pass (existing + new). If any fail, inspect the diff output to find where engines diverge. Common issues: floating-point order, randomness, quantization rounding.

10. **Final regression check** (from v2/):
    ```bash
    python -m pytest engine/ -v                # Python engine tests
    cd engine_rs && cargo test && cd ..         # Rust engine tests
    python -m pytest engine_cmp/ -v            # Parity comparison tests
    ```

**Do NOT commit engine changes without confirming parity.** The comparison tool is your verification that the engines are in sync.

## Build and Verification Scripts

**After any changes to Rust engine code or verification code, ALWAYS run (from v2/):**

```bash
python scripts/build_rust_engine.py
```

**What this script does:**
1. Validates Python venv and Rust toolchain are available
2. Compiles Rust engine with `maturin develop`
3. Runs all 30 engine parity comparison tests
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
