# Code Review Notes (2026-02-11)

Findings from a full read-through of the engine, comparison tool, and frontend.
Nothing here is an active correctness bug in production, but they range from
misleading to fragile.

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

## Remaining (not yet addressed)

**engine/generate.py — missing deepcopy (line 494):**
`best_layout = replica.layout` is a direct reference, while lines 409, 413,
and 474 all use `copy.deepcopy(replica.layout)`. Not a current bug (replicas
aren't used after this point), but inconsistent — if code is added later that
touches replicas, `best_layout` would alias one of them.

**engine/visibility.py — `_point_near_any_polygon` (line 727):**
Defined but never called from production code. Only called from tests.

**engine/types.py — docstring overclaim (lines 46-48):**
Says "Every dataclass has `from_dict` / `to_dict` methods" but 5 dataclasses
(`CatalogObject`, `CatalogFeature`, `TerrainCatalog`, `FeatureCountPreference`,
`EngineParams`) only have `from_dict()`, no `to_dict()`.

**frontend/catalogs.py — docstring omission (line 11):**
Says feature types are "obstacle or obscuring" but the catalog also defines
`"woods"` features (kidney bean woods, industrial tank).
