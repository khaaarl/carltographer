# Plan: Make Obscuring Features Respect `min_blocking_height` [DONE]

## Problem

Obscuring features (ruins) bypass the `min_blocking_height` filter in `_precompute_segments`. All their shapes block LOS regardless of height. This means:

- A 3.0" WTC short ruin blocks LOS identically at both the 4.0" (standard) and 2.2" (infantry) thresholds
- `_has_intermediate_shapes` skips obscuring features entirely, so the infantry dual-pass never triggers
- The frontend never shows infantry visibility data when only obscuring terrain has intermediate heights

## Desired Behavior

Obscuring shapes should be filtered by `min_blocking_height` the same way obstacle shapes are. A 3.0" ruin wall should:
- **Standard pass (4.0")**: Not block (3.0 < 4.0) — observers can see over it
- **Infantry pass (2.2")**: Block via back-facing edges (3.0 >= 2.2) — infantry can't see over it

## Failing Tests (Already Written)

These tests are on the `feature/show-infantry-visibility` branch and assert the desired behavior:

**Python unit tests** (`v2/engine/visibility_test.py`):
- `TestHasIntermediateShapes::test_obscuring_intermediate_detected` — asserts `_has_intermediate_shapes` returns True for 3.0" obscuring feature
- `TestObscuringHeightFiltering::test_obscuring_below_threshold_no_block` — asserts 100% visibility for 3.0" obscuring shape at 4.0" threshold
- `TestObscuringHeightFiltering::test_obscuring_above_threshold_blocks` — asserts <100% visibility for 3.0" obscuring shape at 2.2" threshold (already passes by accident)
- `TestObscuringHeightFiltering::test_obscuring_dual_pass_produces_different_values` — asserts dual-pass sub-dicts exist with different standard/infantry values

**Engine comparison test** (`v2/engine_cmp/compare.py`):
- `infantry_vis_obscuring_dual_pass` scenario with `_validate_infantry_obscuring_dual_pass` — checks both engines produce `standard`/`infantry` sub-dicts with correct values

**Also on the branch**: `compare_test.py` fix to pass `validate_fn` through to `run_comparison` (was silently dropped before).

## Changes Required

### Step 1: Fix Python `_precompute_segments` (`v2/engine/visibility.py`, lines 237–262)

The obscuring branch currently calls `_get_footprint_corners(pf, objects_by_id)` which returns ALL shapes with no height filtering. Replace it with per-component, per-shape iteration that mirrors the non-obscuring path's structure but produces `obscuring_shapes` entries instead of `static_segments`.

**Current** (lines 237–262):
```python
if is_obscuring:
    all_corners = _get_footprint_corners(pf, objects_by_id)
    if not all_corners:
        continue
    for corners in all_corners:
        # ... compute edges with normals ...
        obscuring_shapes.append((corners, edges))
```

**After**:
```python
if is_obscuring:
    for comp in pf.feature.components:
        obj = objects_by_id.get(comp.object_id)
        if obj is None:
            continue
        comp_t = comp.transform or Transform()
        for shape in obj.shapes:
            if shape.effective_opacity_height() < min_blocking_height:
                continue
            shape_t = shape.offset or Transform()
            world = compose_transform(
                compose_transform(shape_t, comp_t),
                pf.transform,
            )
            corners = _shape_world_corners(shape, world)
            # ... compute edges with normals (same as before) ...
            obscuring_shapes.append((corners, edges))
```

The edge-normal computation (center, edges loop, dot_center check) stays identical — only the outer iteration changes from "all footprints at once" to "per-component, per-shape with height filter".

### Step 2: Fix Python `_has_intermediate_shapes` (`v2/engine/visibility.py`, lines 920–922)

Remove the `if pf.feature.feature_type == "obscuring": continue` skip. Now that obscuring shapes respect height, intermediate-height obscuring shapes DO produce different results between passes.

### Step 3: Update Python test `test_obscuring_ignored` (`v2/engine/visibility_test.py`, line 929)

This test asserts the OLD behavior (obscuring features are skipped). Either:
- Delete it (the new `test_obscuring_intermediate_detected` replaces it), or
- Invert its assertion and update the docstring

### Step 4: Verify Python tests pass

```bash
source v2/.env/bin/activate && cd v2 && python -m pytest engine/visibility_test.py -v
```

All 4 new tests should pass. The existing `test_obscuring_above_threshold_blocks` already passes. No other tests should break — obscuring features at standard heights (e.g., 9.0" three-storey ruins) are above both thresholds and will still block in both passes.

### Step 5: Fix Rust `has_intermediate_shapes` (`v2/engine_rs/src/visibility.rs`, lines 1068–1070)

Remove the `if pf.feature.feature_type == "obscuring" { continue; }` skip. Identical to the Python fix.

### Step 6: Fix Rust `precompute_segments` (`v2/engine_rs/src/visibility.rs`, lines 214–247)

Same structural change as Python: replace `get_footprint_corners` call with per-component, per-shape iteration using `shape.effective_opacity_height() < min_blocking_height` filtering. The edge/normal computation stays the same.

**Current** (lines 214–247):
```rust
if is_obscuring {
    let all_corners = get_footprint_corners(pf, objects_by_id);
    for corners in &all_corners {
        // ... compute edges with normals ...
        obscuring_shapes.push(ObscuringShape { corners: verts, edges });
    }
}
```

**After**:
```rust
if is_obscuring {
    for comp in &pf.feature.components {
        let obj = match objects_by_id.get(&comp.object_id) {
            Some(o) => o,
            None => continue,
        };
        let comp_t = comp.transform.as_ref().unwrap_or(&default_t);
        for shape in &obj.shapes {
            if shape.effective_opacity_height() < min_blocking_height {
                continue;
            }
            let shape_t = shape.offset.as_ref().unwrap_or(&default_t);
            let world = compose_transform(
                &compose_transform(shape_t, comp_t), &pf.transform
            );
            let corners = shape_world_corners(shape, &world);
            // ... compute edges with normals (same as before) ...
            obscuring_shapes.push(ObscuringShape { corners: verts, edges });
        }
    }
}
```

### Step 7: Verify Rust tests pass

```bash
cd v2/engine_rs && cargo test
```

### Step 8: Full parity verification

```bash
source v2/.env/bin/activate && cd v2 && python scripts/build_rust_engine.py
```

All 42 comparison scenarios must pass (41 existing + 1 new).

### Step 9: Final regression sweep

```bash
source v2/.env/bin/activate && cd v2 && python -m pytest engine/ -v
cd v2/engine_rs && cargo test
source v2/.env/bin/activate && cd v2 && python -m pytest engine_cmp/ -v
```

## Things NOT Affected

- **VisibilityCache**: Uses `get_tall_world_obbs(min_height=1.0)` for grid-point-inside-terrain exclusion. Separate concern from LOS blocking height — no change needed.
- **DZ hideability / objective hidability**: These use the same `_precompute_segments` path, so they automatically benefit from the fix.
- **Tall obscuring features**: e.g., 9.0" WTC three-storey ruins are above both thresholds (4.0" and 2.2") and will continue to block in both passes. No behavior change for them.
- **`_get_footprint_corners` / `get_world_obbs`**: These functions remain unchanged. They're still used elsewhere (collision, rendering). The visibility code just stops calling `_get_footprint_corners` in the obscuring path.

## Frontend Change (Already Done)

The `_update_visibility_display` change in `v2/frontend/app.py` is already on the branch — it shows `"72.53% (inf: 69.99%)"` when dual-pass data is present. Once the engine fix lands, this will automatically display infantry info for layouts with intermediate-height obscuring features.
