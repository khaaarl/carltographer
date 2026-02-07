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

Python virtual environment: `v2/.env/` (Python 3.12).

Activate: `source v2/.env/bin/activate`

Intended toolchain (not all configured yet):
- **pytest** for tests: `python -m pytest v2/`
- **black** for formatting: `python -m black v2/`
- **ruff** for linting: `ruff check v2/`
- **isort** for import sorting: `python -m isort v2/`

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

4. **Run comparison tests**: After implementing in Rust, MUST verify parity:
   ```bash
   # Build Rust engine
   cd v2/engine_rs && maturin develop

   # Run all comparison tests
   python -m engine_cmp.compare --verbose

   # Run pytest version for CI/CD
   python -m pytest engine_cmp/ -v
   ```
   - All tests must pass (12 scenarios currently, more if you added new ones)
   - Look for "12 passed, 0 failed" in output
   - If any test fails, inspect the diff output to find where engines diverge
   - Common issues: floating-point order, randomness, quantization rounding

5. **Update unit tests**: Ensure new Rust tests are added to cover the new feature:
   - Add to `v2/engine_rs/src/collision.rs` tests if collision-related
   - Add to `v2/engine_rs/src/generate.rs` tests if generation-related
   - Run: `cd v2/engine_rs && cargo test`

6. **Verify no regressions**:
   - Python: `python -m pytest v2/engine/ -v` (should still have 44+ tests)
   - Rust: `cargo test` (should still have 28+ tests)
   - Comparison: `python -m engine_cmp.compare` (12+ tests, all passing)

**Do NOT commit Rust engine changes without confirming parity.** The comparison tool is your verification that the engines are in sync.
