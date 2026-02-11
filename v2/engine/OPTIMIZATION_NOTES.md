# Python Engine Optimization Notes

Status: **3 optimizations committed** — ~10% serial parity suite speedup

## Context

The Python engine is the reference implementation for terrain layout generation. It must produce **bit-identical** output to the Rust engine for the same seed. Any optimization must preserve this invariant.

The primary motivation for Python engine optimization is **parity test throughput**. The parity test suite (46 scenarios as of February 2026) runs both Python and Rust engines and compares output. The Rust engine is ~10x faster, so the Python engine dominates total test time. As more scenarios are added, this becomes the bottleneck.

Current parity suite runtime: **~27s** (46 scenarios, serial execution). Was ~30s before optimizations.

### Key constraint: determinism and Rust parity

Every optimization must preserve:
- Identical PRNG call sequence (same number of calls, same order)
- Identical floating-point results (IEEE 754 operation order matters)
- Identical mutation acceptance/rejection decisions
- Identical final layout for any given seed

Internal-only changes (caching, data structures, parallelism in non-deterministic-order-dependent code) are safe. Any change to arithmetic expression order or loop structure that affects FP results will break parity.

## Profiling Results (February 2026)

### Methodology

Profiled with `cProfile` on two representative parity scenarios:
- **polygon_symmetric**: 50 steps, symmetric, polygon terrain (wtcPoly catalog), with mission — the slowest parity scenario at ~6.3s
- **scoring_targets_with_mission**: 50 steps, rect-only terrain, with mission — the slowest rect-only scenario at ~2.6s

### polygon_symmetric (7.6s total, 6.2M function calls)

| Function | tottime | % | calls | Description |
|---|---|---|---|---|
| `_compute_visibility_polygon` | 2.29s | **30%** | 3,817 | Angular sweep raycasting (NumPy vectorized) |
| `_vectorized_pip_mask` | 1.81s | **24%** | 1,991 | Batch point-in-polygon for grid/objective samples |
| `numpy.ufunc.reduce` | 0.87s | 11% | 599,430 | Internal NumPy reduction (called by `np.any`, `np.min`, etc.) |
| `segments_intersect_inclusive` | 0.57s | 7% | 2,417,564 | Pure Python edge-edge intersection test |
| `polygons_overlap` | 0.48s | 6% | 440 | Pure Python polygon overlap (calls `segments_intersect_inclusive`) |
| `np.any` dispatch | 0.48s | 6% | 595,611 | Python→C function call overhead for `np.any()` |
| `compute_layout_visibility` | 0.28s | 4% | 7 | Top-level visibility orchestration |
| `numpy.array` | 0.13s | 2% | 7,648 | Array construction overhead |
| `_polygon_area` | 0.11s | 1% | 3,817 | Shoelace formula (NumPy) |

### scoring_targets_with_mission (3.4s total, 3.2M function calls)

| Function | tottime | % | calls | Description |
|---|---|---|---|---|
| `_vectorized_pip_mask` | 1.34s | **39%** | 10,223 | Batch point-in-polygon |
| `numpy.ufunc.reduce` | 0.46s | 13% | 319,793 | NumPy reduction overhead |
| `_compute_visibility_polygon` | 0.37s | **11%** | 4,607 | Angular sweep raycasting |
| `np.any` dispatch | 0.27s | 8% | 315,160 | Python→C call overhead |
| `segments_intersect_inclusive` | 0.25s | 7% | 1,071,890 | Edge-edge intersection |
| `polygons_overlap` | 0.22s | 6% | 2,048 | Polygon overlap testing |

### Key observations

1. **Visibility dominates**: `_compute_visibility_polygon` + `_vectorized_pip_mask` account for **54%** of polygon scenario time and **50%** of rect-only scenario time. Both are already NumPy-vectorized.

2. **NumPy call overhead is significant**: `np.any()` is called ~600K times in the polygon scenario. Each call costs ~1.5μs in Python→C dispatch overhead. The actual reduction work is fast; it's the per-call overhead that adds up. This accounts for ~17% of total time (`ufunc.reduce` + `any` dispatch).

3. **Pure Python collision loops are expensive with polygons**: `segments_intersect_inclusive` is called 2.4M times in the polygon scenario (7% of time) — all from `polygons_overlap` which does O(E_a × E_b) edge-pair tests. With 24-gon tanks, that's up to 24×24=576 tests per pair.

4. **Visibility is already well-vectorized**: The raycasting in `_compute_visibility_polygon` uses NumPy broadcasting for the R×S intersection matrix. The main cost is inherent — ~3,800 observer calls × ~50 segments each.

5. **Collision/mutation path is cheap for rects**: `is_valid_placement` doesn't appear in the top functions for rect-only scenarios. It only becomes significant with polygon terrain (via `polygons_overlap`).

### Where the Python engine differs from Rust

The Rust engine has several optimizations that don't exist in Python:
- Angular bucketing (reduces ray-segment work from O(R×S) to O(R×k))
- FxHash for endpoint deduplication
- Pseudoangle for bucket assignment (eliminates atan2)
- sin_cos() fusion
- Rayon parallelism for the observer loop

The Python engine uses NumPy vectorization instead — different approach to the same problem. NumPy broadcasting handles the R×S matrix in C, which is effective but has per-call overhead that Rust avoids.

## Future Optimization Ideas

### Tier 1: Reduce NumPy call overhead in visibility

#### 1b. Reduce `numpy.array` construction in hot loops
7,648 `numpy.array()` calls in the polygon scenario. Some may be constructing small arrays per-observer that could be pre-allocated and reused via slice assignment.

**Complexity**: Low-medium (need to identify which calls are per-observer vs one-time).
**Expected gain**: 2-5%.
**Parity risk**: None.

### Tier 3: Visibility micro-optimizations

#### 3a. Pre-allocate per-observer arrays
`_compute_visibility_polygon` constructs several NumPy arrays per observer call (ray angles, directions, intersection results). Pre-allocating max-sized buffers and slicing could reduce `numpy.array` construction overhead.

**Complexity**: Medium (need to thread buffer objects through the call chain).
**Expected gain**: 2-5% (array construction is ~2% of total time).
**Parity risk**: None.

#### 3b. Fuse `_polygon_area` into visibility polygon construction
`_polygon_area` (shoelace formula) is called once per observer after constructing the visibility polygon. The shoelace sum could be accumulated during polygon construction rather than requiring a second pass. Saves one array traversal per observer.

**Complexity**: Low.
**Expected gain**: ~1% (area computation is only 1.5% of total time).
**Parity risk**: Medium — accumulating the sum incrementally may produce different FP results than the sequential Python loop. Would need parity verification.

### Tier 4: Algorithmic changes (higher complexity)

#### 4a. Angular bucketing in Python (mirror Rust optimization)
The Rust engine partitions segments into angular buckets so each ray only tests segments in its bucket. The Python engine tests all segments against all rays via NumPy broadcasting (R×S matrix). For small segment counts (~40-80), the broadcast approach is competitive because NumPy's C loops are fast. But for polygon-heavy layouts with more segments, bucketing could help.

**Complexity**: High (significant restructuring of `_compute_visibility_polygon`).
**Expected gain**: Uncertain — NumPy broadcasting may already be faster than Python-level bucketing with its attendant overhead.
**Parity risk**: None (same rays, same segments, same intersections — just processed in different order internally).

## Committed Optimizations

### 0a. Parallel parity test runner
Added `-j`/`--parallel` flag to `engine_cmp/compare.py` using `ProcessPoolExecutor`. The build script (`build_rust_engine.py`) now runs with `-j -1` (auto CPU count) by default. Also added `pytest-xdist` to `requirements-dev.txt`.

**Result**: 46 scenarios serial ~31s → parallel ~12s wall-clock (2.5x speedup on 11-core machine).

### 1a-revised. Scalar z-range guard in `_vectorized_pip_mask`
Replaced `np.any(crosses)` with a pure-Python scalar comparison: precompute `pts_z_min/max`, then check if the edge's z-range overlaps before computing the crossing mask. Same skip semantics as `np.any(crosses)`, but avoids ~600K numpy dispatch calls per visibility computation.

**Note**: The original 1a idea (simply removing the `np.any()` guard) was a **30% regression** — the guard saves far more `np.where` computation than the dispatch costs. The scalar z-range check preserves the skip benefit while eliminating the numpy overhead.

**Result**: ~10-25% speedup on visibility-heavy scenarios.

| Scenario | Before | After | Speedup |
|---|---|---|---|
| `scoring_targets_with_mission` | 2.35s | 1.79s | **24%** |
| `infantry_vis_dual_pass` | 1.07s | 0.69s | **36%** |
| `infantry_vis_no_intermediate` | 0.47s | 0.28s | **40%** |
| `polygon_symmetric` | 7.43s | 5.88s | **21%** |
| `polygon_only_generation` | 3.14s | 2.35s | **25%** |

### 2a. Vectorized edge-edge intersection in `polygons_overlap`
Replaced the nested Python loop calling `segments_intersect_inclusive` O(E_a × E_b) times with NumPy broadcasting. Edge arrays are built as (E, 4) numpy arrays, and all cross products are computed in one broadcast operation.

**Result**: Marginal overall (polygon collision is dominated by AABB early-exit), but eliminates worst-case degradation for overlapping polygons.

### 2b. AABB early-exit in `polygons_overlap`
Added axis-aligned bounding box check before the expensive edge-intersection and vertex-containment tests. For distant polygons (the common case in placement validation), the 4-comparison AABB check skips all O(E²) work.

**Result**: ~10-17% speedup on polygon scenarios.

### Misc. Skip list copy in `_get_observer_segments`
When there are no obscuring shapes (common for obstacle-only layouts), return the static segments list directly instead of copying it per-observer.

**Result**: Minor (avoids ~500 list copies per visibility evaluation in non-ruins layouts).

## Attempted But Abandoned

### 1a-original. Remove `np.any()` guard entirely from `_vectorized_pip_mask`
Hypothesis: eliminating ~600K `np.any()` calls (each ~1.5μs dispatch overhead) would save ~15%. Reality: the `np.any(crosses)` guard skips expensive `np.where` + division + comparison operations for edges that don't cross any sample point's z-coordinate. Without the guard, every edge triggers the full computation, costing far more than the dispatch overhead saved.

**Result**: **30% regression** (39.27s vs 30.11s baseline). The guard is net positive.
**Lesson**: NumPy dispatch overhead matters, but vectorized computation overhead matters more. A cheap Python-level pre-filter (scalar z-range comparison) is the right approach — see 1a-revised.

### 3b-attempt. Vectorize `_polygon_area` with numpy
Attempted to replace the sequential shoelace loop with `np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)`. Reverted immediately — `np.sum` uses different accumulation order than sequential addition, producing different FP results and breaking parity.

**Result**: Not tested (reverted before running).
**Lesson**: Any change to arithmetic accumulation order breaks parity. Sequential sums ≠ numpy reduction sums due to FP associativity.

## Recommended Next Steps

1. **Re-profile** after committed optimizations to see updated bottleneck distribution.
2. **Try 1b (reduce numpy.array construction)** — identify per-observer allocations that could be pre-allocated.
3. **Try 3a (pre-allocate per-observer arrays)** in `_compute_visibility_polygon` — the R×S matrix dominates, but array construction overhead may be more visible now that PIP overhead is reduced.
4. Remaining bottleneck is likely `_compute_visibility_polygon` itself (the R×S matrix), which is already NumPy-vectorized and hard to improve further without algorithmic changes (Tier 4).
