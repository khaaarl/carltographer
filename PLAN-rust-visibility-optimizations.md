# Plan: Optimize Rust Visibility Engine (Single-Threaded)

## Context

The `generate_with_mission_hna` criterion benchmark takes **203ms** for 50 steps — 500x slower than the non-mission `vis_50` benchmark (403µs). The goal is to optimize the Rust visibility code without introducing multithreading.

**Profiling analysis** shows `fraction_of_dz_visible` dominates at **~77%** of per-call time for the mission benchmark. This function tests each of ~731 DZ sample points against the visibility polygon (~50-84 vertices) via `point_in_polygon`, and it's called ~1,146 times per visibility call (once per DZ per outside-DZ observer).

The remaining ~23% breaks down as: angular sweep 13%, cross-DZ seen tracking 7%, objective hidability 2%, misc 1%.

**On the "numpy equivalent" question**: Libraries like `ndarray` won't help here — Rust is already native code, so the Python→C speedup that makes NumPy valuable doesn't apply. The problem sizes are also too small (~84 segments, ~731 samples) for array creation overhead to be amortized. The better Rust equivalent of "use NumPy" is: restructure data layout for cache locality and let LLVM autovectorize the inner loops.

## Baselines (single-threaded, seed=42)

| Benchmark | Time |
|---|---|
| `generate_with_visibility_50` | 403 µs |
| `generate_with_visibility_100` | 884 µs |
| `generate_with_mission_hna` | 203 ms |

## Changes (ordered by expected impact)

All changes are to **`v2/engine_rs/src/visibility.rs`** unless noted.

### 1. Batch PIP for `fraction_of_dz_visible` (HIGH — targets 77% of mission cost)

Replace the current point-first PIP iteration with edge-first ("batch PIP"):

**Current** (731 calls to `point_in_polygon`, each iterating ~84 edges):
```rust
let count = dz_sample_points.iter()
    .filter(|&&(px, pz)| point_in_polygon(px, pz, vis_poly))
    .count();
```

**New** (iterate 84 edges, each scanning all 731 points):
```rust
fn batch_point_in_polygon(points: &[(f64, f64)], polygon: &[(f64, f64)], inside: &mut Vec<bool>) {
    inside.clear();
    inside.resize(points.len(), false);
    let n = polygon.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = polygon[i];
        let (xj, zj) = polygon[j];
        // Inner loop over all points — LLVM can autovectorize this
        for (k, &(px, pz)) in points.iter().enumerate() {
            if (zi > pz) != (zj > pz) {
                let intersect_x = (xj - xi) * (pz - zi) / (zj - zi) + xi;
                if px < intersect_x {
                    inside[k] = !inside[k];
                }
            }
        }
        j = i;
    }
}
```

Why this helps: The edge data (xi, zi, xj, zj) stays in registers while the inner loop linearly scans the contiguous points array (~12KB, fits L1 cache). This is much more cache-friendly and autovectorization-amenable than 731 separate `point_in_polygon` calls each re-traversing the polygon.

Apply the same pattern to:
- `fraction_of_dz_visible` (main target — 731 DZ samples vs vis_poly)
- Cross-DZ seen tracking (731 samples, but with already-seen skip)
- Objective hidability (200 samples, but with already-seen skip)

For the seen-tracking paths, pre-filter unseen points into a compact buffer, then batch-PIP the buffer, then scatter results back. This avoids branching in the inner loop.

### 2. Vec<bool> for seen tracking (MEDIUM — targets 7% of mission cost)

Replace `HashSet<usize>` with `Vec<bool>` for cross-DZ seen sets and objective seen sets.

- `dz_cross_seen: HashMap<String, HashSet<usize>>` → `Vec<Vec<bool>>` (flat-indexed by `target_dz * num_dzs + source_dz`)
- `obj_seen_from_dz: HashMap<String, Vec<HashSet<usize>>>` → `Vec<Vec<Vec<bool>>>` (by `dz_idx -> obj_idx -> sample_idx`)

Benefits:
- `seen[i]` is O(1) direct indexed vs O(1) amortized with ~20-30ns hash overhead
- "Full visibility" path: `seen.fill(true)` vs 731 individual `HashSet::insert` calls
- `!seen[i]` check in inner loop is branch-predictor-friendly (sequential access)

### 3. Index-based DZ keys (MEDIUM — eliminates ~1200 format!() per visibility call)

Replace `HashMap<String, ...>` accumulators with `Vec<...>` indexed by DZ position.

- `dz_vis_accum: HashMap<String, (f64, i64)>` → `Vec<(f64, i64)>` indexed by `dz_idx`
- `dz_cross_obs_count: HashMap<String, i64>` → `Vec<i64>` indexed by pair index
- Eliminate all `format!("{}_from_{}", ...)` calls in the hot loop
- Convert back to string keys only when building the output JSON (cold path)

### 4. Precompute observer DZ membership (LOW — small savings per observer)

Before the main observer loop, build a `Vec<Option<usize>>` mapping each observer to its DZ index (or None). This eliminates per-DZ `point_in_any_polygon` calls inside the loop.

```rust
let observer_dz: Vec<Option<usize>> = sample_points.iter()
    .map(|&(sx, sz)| dz_data.iter().position(|dd| point_in_any_polygon(sx, sz, &dd.polys)))
    .collect();
```

### 5. Precompute segment deltas (LOW — targets vis_50/vis_100)

In `PrecomputedSegments`, store `(sx, sz)` = `(x2-x1, z2-z1)` alongside each segment. This saves 2 subtractions per `ray_segment_intersection` call. Change `Segment` from `(f64, f64, f64, f64)` to a struct with 6 fields or a 6-tuple.

Impact is small (~5-10% improvement on vis_50/vis_100) because there are relatively few segments. But it's a clean optimization.

## Parity

All changes are internal to visibility computation — same algorithm, same results, just faster data structures and iteration order. Engine parity tests must still pass (24 scenarios, bit-identical).

The batch PIP produces identical results to individual PIP calls (same ray-casting algorithm, same edge iteration). The only requirement is that the `inside` toggle is initialized to `false` and each edge toggles it independently — which is exactly what the batch version does.

## Verification

1. `cd v2/engine_rs && cargo test` — all 51 unit tests pass
2. `python scripts/build_rust_engine.py` — build + 24 parity tests pass
3. `cd v2/engine_rs && cargo bench` — compare against baselines above
4. `pre-commit run --all-files` — formatting/linting/type checks pass

## Expected Results

| Optimization | Target benchmark | Expected improvement |
|---|---|---|
| Batch PIP | mission_hna | 30-50% faster (~100-140ms) |
| Vec<bool> + index keys | mission_hna | 5-10% additional |
| Segment deltas | vis_50, vis_100 | 5-10% faster |
| **Combined** | **mission_hna** | **~40-55% faster (~90-120ms)** |
| **Combined** | **vis_50/vis_100** | **~5-10% faster** |

Note: autovectorization impact is hard to predict precisely — depends on LLVM's ability to vectorize the batch PIP inner loop. The branch on `(zi > pz) != (zj > pz)` may limit vectorization. If so, a branchless formulation using conditional moves may be needed.

## Actual Results (implemented changes 1-4, skipped 5)

| Benchmark | Before | After | Speedup |
|---|---|---|---|
| `generate_with_visibility_50` | 470 µs | 459 µs | ~2% (within noise) |
| `generate_with_visibility_100` | 928 µs | 836 µs | **10%** |
| **`generate_with_mission_hna`** | **199 ms** | **4.92 ms** | **40.4x (97.5% reduction)** |

The mission benchmark improvement of 40x far exceeded the predicted 40-55%. The batch PIP, Vec-indexed accumulators, and precomputed observer DZ membership combine synergistically. Change 5 (segment deltas) was not implemented as the impact on vis_50/vis_100 is marginal.
