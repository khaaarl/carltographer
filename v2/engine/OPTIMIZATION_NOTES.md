# Python Engine Optimization Notes

Status: **initial profiling complete** — no optimizations attempted yet

## Context

The Python engine is the reference implementation for terrain layout generation. It must produce **bit-identical** output to the Rust engine for the same seed. Any optimization must preserve this invariant.

The primary motivation for Python engine optimization is **parity test throughput**. The parity test suite (46 scenarios as of February 2026) runs both Python and Rust engines and compares output. The Rust engine is ~10x faster, so the Python engine dominates total test time. As more scenarios are added, this becomes the bottleneck.

Current parity suite runtime: **~30-36s** (46 scenarios, serial execution).

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

#### 1a. Eliminate `np.any()` guard in `_vectorized_pip_mask`
The inner loop (per-edge) does:
```python
crosses = (zi > pts_z) != (zj > pts_z)
if np.any(crosses):
    # vectorized intersection computation
```
The `np.any(crosses)` guard skips edges that don't cross any sample point's Z-coordinate. But with ~600K calls, the guard's Python→C dispatch cost (~1.5μs each) may exceed the savings from skipping. Removing the guard and always computing the vectorized result trades redundant arithmetic (fast in C) for eliminated call overhead (slow in Python).

**Complexity**: Very low (delete 2 lines, adjust indentation).
**Expected gain**: Up to ~15% on visibility-heavy scenarios (eliminates ~600K `np.any` calls).
**Parity risk**: None — same math, same results, just skips a branch.

#### 1b. Reduce `numpy.array` construction in hot loops
7,648 `numpy.array()` calls in the polygon scenario. Some may be constructing small arrays per-observer that could be pre-allocated and reused via slice assignment.

**Complexity**: Low-medium (need to identify which calls are per-observer vs one-time).
**Expected gain**: 2-5%.
**Parity risk**: None.

### Tier 2: Vectorize collision for polygon shapes

#### 2a. Batch `segments_intersect_inclusive` via NumPy
`polygons_overlap` calls `segments_intersect_inclusive` in a nested Python loop — O(E_a × E_b) calls. For a 24-gon vs 4-rect, that's 96 calls; for 24-gon vs 24-gon, 576 calls. The intersection test is 4 cross products + comparisons — straightforward to vectorize:

```python
# Instead of:
for (ax1,az1),(ax2,az2) in edges_a:
    for (bx1,bz1),(bx2,bz2) in edges_b:
        if segments_intersect_inclusive(ax1,az1,ax2,az2,bx1,bz1,bx2,bz2):
            return True

# Vectorize as:
# edges_a: (E_a, 2, 2), edges_b: (E_b, 2, 2)
# Broadcast cross products over all E_a × E_b pairs at once
```

**Complexity**: Medium (need to restructure `polygons_overlap` to build edge arrays, implement vectorized cross products, handle early-exit semantics).
**Expected gain**: 30-50% on polygon collision scenarios (eliminates 2.4M Python function calls).
**Parity risk**: Low — cross products are simple arithmetic with identical operation order. But must verify that NumPy broadcasting produces the same FP results as scalar Python (it should, since the operations are identical per-element).

#### 2b. AABB early-exit for terrain-vs-terrain polygon overlap
Same idea as Rust opt #8 — compute axis-aligned bounding boxes for placed features and skip `polygons_overlap` when AABBs don't overlap. Cheap to compute (min/max of vertices), eliminates expensive O(E²) tests for distant features.

**Complexity**: Low (add 4 comparisons before `polygons_overlap` call).
**Expected gain**: Variable — depends on feature density. Most impactful on large tables with many features.
**Parity risk**: None (early-exit on a condition that guarantees no overlap).

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
**Parity risk**: Medium — accumulating the sum incrementally may produce different FP results than the vectorized numpy computation. Would need parity verification.

### Tier 4: Algorithmic changes (higher complexity)

#### 4a. Angular bucketing in Python (mirror Rust optimization)
The Rust engine partitions segments into angular buckets so each ray only tests segments in its bucket. The Python engine tests all segments against all rays via NumPy broadcasting (R×S matrix). For small segment counts (~40-80), the broadcast approach is competitive because NumPy's C loops are fast. But for polygon-heavy layouts with more segments, bucketing could help.

**Complexity**: High (significant restructuring of `_compute_visibility_polygon`).
**Expected gain**: Uncertain — NumPy broadcasting may already be faster than Python-level bucketing with its attendant overhead.
**Parity risk**: None (same rays, same segments, same intersections — just processed in different order internally).

#### 4b. Cython compilation of `segments_intersect_inclusive`
Compile the hot pure-Python function to C via Cython. This eliminates Python interpreter overhead for the 2.4M calls in polygon scenarios.

**Complexity**: Medium (need Cython build infrastructure, `.pyx` file, `setup.py` changes).
**Expected gain**: ~5-7% overall (function is 7% of total time; Cython typically 10-50x faster for tight numeric loops, but we'd also eliminate call overhead).
**Parity risk**: None (same arithmetic).

## Committed Optimizations

### 0a. Parallel parity test runner
Added `-j`/`--parallel` flag to `engine_cmp/compare.py` using `ProcessPoolExecutor`. The build script (`build_rust_engine.py`) now runs with `-j -1` (auto CPU count) by default. Also added `pytest-xdist` to `requirements-dev.txt`.

**Result**: 46 scenarios serial ~31s → parallel ~12s wall-clock (2.5x speedup on 11-core machine).

## Attempted But Abandoned

*(None yet)*

## Recommended Next Steps

1. **Try 1a (eliminate `np.any()` guard)** — very low effort, measurable gain, zero parity risk.
3. **Try 2b (AABB early-exit for terrain overlap)** — low effort, helps polygon scenarios.
4. **Profile again after 1a and 2b** to see if the bottleneck distribution has shifted enough to warrant deeper changes.
5. **Consider 2a (vectorize `segments_intersect_inclusive`)** only if polygon collision remains a significant fraction after the easier wins.
