# Code Review Notes (2026-02-11)

Findings from a full read-through of the engine, comparison tool, and frontend.
Nothing here was an active correctness bug in production, but they ranged from
misleading to fragile. All items have been addressed.

## Fixed

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

## Not changed

**engine/visibility.py — `_point_near_any_polygon`:**
Defined in visibility.py but only called from tests. Left in place since it
depends on `point_to_segment_distance_squared` from collision.py and moving it
to the test file would just shuffle the dependency. It's a small pure function
that serves as a test utility.
