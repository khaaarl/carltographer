# Code Review Notes (2026-02-11)

Findings from full read-throughs of the engine, comparison tool, and frontend.

## Fixed (round 1)

**engine_cmp/compare.py — scenario name/parameter mismatch:**
`"basic_50_steps"` and `"basic_100_steps"` both had `num_steps=10`. Renamed to
`"basic_10_steps_no_vis"` and `"basic_10_steps_with_vis"`.

**engine_cmp/compare.py — dead functions:**
`quantize_position()` and `quantize_angle()` were defined but never called.
Removed.

**engine_rs/src/visibility.rs — stale comment (line 158):**
Referenced `pip_zsorted_update_seen` which no longer exists. Removed from comment.

**engine_rs/src/visibility.rs — inaccurate comment (line 23):**
Said `VisBuffers` contains "sorted events". Updated to match actual fields.

**engine_rs/src/generate.rs — comment numbering gap (line 151):**
Scoring steps went from "2." to "4.". Renumbered to "3.".

**engine/generate.py — missing deepcopy (line 494):**
`best_layout = replica.layout` was a direct reference, while all other
best_layout assignments used `copy.deepcopy()`. Added deepcopy for consistency.

**engine/types.py — docstring overclaim (lines 46-48):**
Said "Every dataclass has `from_dict` / `to_dict` methods" but several
internal-only dataclasses only have `from_dict()`. Updated to reflect reality.

**frontend/catalogs.py — docstring omission (line 11):**
Said feature types are "obstacle or obscuring" but the catalog also defines
`"woods"` features. Updated to include "woods".

## Not changed (round 1)

**engine/visibility.py — `_point_near_any_polygon`:**
Defined in visibility.py but only called from tests. Left in place since it
depends on `point_to_segment_distance_squared` from collision.py and moving it
to the test file would just shuffle the dependency. Small pure function that
serves as a test utility.

---

# Round 2 Findings (2026-02-11)

## Fixed (round 2)

**engine_rs/src/generate.rs — tautological assertion:**
`assert!(count >= 3 || count <= 10)` → `assert!(count >= 3 && count <= 10)`.

**engine/visibility.py — unused params in `_merge_dual_pass_results`:**
Removed `standard_height` and `infantry_height` parameters (and updated call
site).

**engine/tempering.py — `@runtime_checkable` on `TemperingCandidate`:**
Removed unused decorator and `runtime_checkable` import.

**engine/prng.py — unused `from __future__ import annotations`.** Removed.

**frontend/app.py — `TABLE_BG` and `TABLE_GRID` constants:** Removed.

**frontend/app.py — `hasattr(self, "_popup_window_id")` always True:**
Removed dead guard, now calls `_dismiss_popup()` directly.

**frontend/app.py — legacy variable names:**
Renamed `min_crates_var`/`max_crates_var` → `min_obstacles_var`/`max_obstacles_var`.

**frontend/app.py — `_on_move_resume` name:**
Renamed to `_on_move_cancel_btn`.

**frontend/app.py — redundant exception catch:**
`except (ValueError, Exception)` → `except Exception`.

**engine/mutation.py docstring — Move PRNG consumption:**
Updated to say "consumes exactly 4 PRNG values when selected feature is not
locked; returns immediately (consuming only 1) if locked."

**engine/mutation.py docstring — displacement range:**
Fixed `+-min_move_range/2` → `+-min_move_range`.

**engine/collision.py docstring — claims `app.py` uses collision functions:**
Removed `app.py` claim, kept `mutation.py`.

**engine/collision.py docstring — `point_in_polygon` attribution:**
Removed specific `visibility.py` attribution; now describes generic purpose.

**engine/generate.py — misleading variable name `needs_dz`:**
Renamed to `needs_detailed_vis`.

**engine/types.py docstring — `CatalogFeature` claim:**
Removed `CatalogFeature` from the "only have from_dict" list.

**engine_rs/src/collision.rs test — "2 inch gap" comment:**
Fixed to "4 inch gap".

**engine_rs/src/collision.rs test — `point_to_segment_endpoint`:**
Renamed to `point_to_segment_on_segment`, updated comment.

**engine_rs/src/collision.rs — `polygons_overlap` doc:**
Updated to "share any interior area or touch".

**engine_rs/src/visibility.rs — stale `batch_point_in_polygon` docstring:**
Updated to say both functions are test-only; production uses
`polygons_overlap_aabb`.

**engine_cmp/compare.py — `basic_10_steps_no_vis` misnomer:**
Added `skip_visibility=True` so the name matches behavior.

**engine_cmp/compare.py — stale `opacity_height_inches` references:**
Updated docstrings and comments to reference `is_footprint` instead.

**engine_cmp/compare.py — stale "bypass height check" comment:**
Removed stale "currently bypass" language; describes current behavior.

**engine_cmp/compare.py — `step_multiplier` mutates global objects:**
Changed `list(TEST_SCENARIOS)` to `[copy.copy(s) for s in TEST_SCENARIOS]`.

**engine_cmp/compare_test.py — unreachable assert:**
Removed dead `assert success` after `pytest.fail()`.

**engine_cmp/compare_test.py — `visibility_tolerance` not passed through:**
Added `visibility_tolerance=scenario.visibility_tolerance` to `run_comparison`.

## Not changed (round 2)

**engine_rs/src/tempering.rs — `best_candidate` never updated in single-replica path:**
`&mut Some(best_candidate.clone())` creates a temporary that's dropped. Mitigated
because the real engine uses `generate_tempering` not `run_tempering`, so only
the abstract tempering module's tests are affected. Needs careful redesign.

**engine_rs/src/generate.rs — `vis_cache` not passed to initial `compute_score`
in `generate_tempering`:**
Passes `None` where Python passes `vis_cache`. Harmless for empty initial layouts
but inconsistent. Left for a targeted parity fix.

**engine_rs/src/visibility.rs — `merge_dual_pass_results` asymmetric fallthrough:**
When a section key exists in `standard` but not `infantry`, Rust copies it;
Python omits it. Unlikely to trigger but is a latent divergence. Needs parity
analysis.

**engine_rs/src/visibility.rs — missing `overall_only` optimization:**
Rust always computes DZ/objective metrics. Performance difference, not
correctness. Would be a feature addition to the Rust engine.

**engine_rs/src/visibility.rs — infantry pass doesn't reuse visibility cache:**
Python reuses; Rust recomputes. Performance difference.

**engine/collision.py — `segments_intersect_inclusive`:**
Only called from tests, not production. Left in place as test utility (same
rationale as `_point_near_any_polygon`).

**engine/collision.py — inconsistent touching semantics:**
Rect touching is NOT overlap; polygon touching IS overlap. Behavioral choice
that affects Rust parity — not a quick fix.

**frontend/app.py — variable shadowing in `_on_load`:**
`path` reused for different types. Minor; left to avoid churn in a large method.

**engine_cmp/compare.py — `_require_mission` table size mismatch:**
Builds 60x44 missions but some scenarios use 44x30 tables. Parity holds but
tests may not exercise intended behavior. Needs scenario-level analysis.

---

# Round 3 Findings (2026-02-11)

## Fixed (round 3)

**scripts/bench_python_visibility.py — crash at runtime:**
Referenced `"basic_50_steps"` which no longer exists. Updated to
`"basic_10_steps"`.

**engine/collision_test.py — redundant `import math`:**
Removed local imports from two test functions (already imported at module level).

**scripts/tuning_scenarios.py — dead `_num_steps` metadata:**
Removed from `build_scenarios()` and the corresponding `pop()` in
`tune_hyperparams_multi.py`.

**engine_cmp/hash_manifest.py — dead `if file_path.exists()` guard:**
Removed (paths come from `rglob`, always exist).

**engine_cmp/hash_manifest.py — stale `build-rust-engine.sh` reference:**
Updated to `build_rust_engine.py`.

**engine/generate_test.py — `test_rotate_deterministic`:**
Replaced with `test_rotate_occurs` that verifies at least some features have
non-zero rotation after 200 steps.

**engine/generate_test.py — `test_multi_mutation_at_high_temperature`:**
Renamed to `test_max_extra_mutations_constant` to accurately describe what it
tests (the arithmetic formula, not engine behavior).

**engine/generate_test.py — `test_score_phase1_with_zero_steps`:**
Renamed to `test_score_phase1_empty_layout_with_deficit`.

**engine/tempering_test.py — `test_swap_occurs` docstring:**
Updated to "best score improves beyond worst initial" (what it actually checks).

**engine/tempering_test.py — `MultimodalCandidate` docstring:**
Updated to "Unimodal landscape: single peak at 0, small Gaussian bumps at ±5."

**engine/tempering_test.py — `test_best_tracking` docstring:**
Updated to "Final best score is at least as good as the initial best."

**engine/visibility_test.py — `test_fringe_hiding_square` tautological assert:**
Removed `assert safe_count >= 0` (always true).

**engine/visibility_test.py — `test_fringe_corners_tracked`:**
Rewrote docstring to describe actual behavior (single hidden point near boundary,
no valid hiding square). Removed vacuous `if needed:` guard — now always iterates
`needed` (loop is a no-op when empty).

**engine/visibility_test.py — `TestObscuringHeightFiltering` class docstring:**
Removed stale "Currently" language. Now describes the correct behavior.

**engine/visibility_test.py — `test_obscuring_above_threshold_blocks`:**
Fixed docstring: "static segments that block LOS" (not "back-facing edges").

**engine/visibility_test.py — `test_obscuring_ruin_at_objective_no_hiding`:**
Rewrote docstring to describe the actual mechanism (terrain-intersection
rejection). Fixed math: `0.75 + 3.0 + sqrt(2) ≈ 5.16` (was `3.0 + sqrt(2)`).

**scripts/ui.py — wrong filename:** Updated `run_ui.py` → `ui.py`.

**scripts/build_rust_engine.py — hardcoded Unix paths:**
Error messages now use `VENV_PYTHON` variable for cross-platform correctness.

**scripts/profile_engine.py — wrong CLI syntax:**
Updated docstring from `--profile` etc. to subcommand syntax (`profile` etc.).
Also updated stale scenario names.

**scripts/coverage_cmp.py — stale scenario count:**
Removed hardcoded "30" count; now says "the parity scenarios".

## Clean Files (round 3)

`collision_distance_test.py`, `mutation_test.py`, `app_test.py`,
`layout_io_test.py`, `missions_test.py`, `coverage_py.py`, `coverage_rs.py`,
`package_executable.py`, `entry_point.py`, `tune_hyperparams.py`,
`catalog_io.py`, `prng.rs`, `generate_bench.rs`.
