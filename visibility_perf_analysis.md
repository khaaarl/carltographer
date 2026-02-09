# Visibility Engine — Performance Analysis

Analysis of visibility computation in both Rust (`v2/engine_rs/src/visibility.rs`) and Python (`v2/engine/visibility.py`) engines.

---

## Architecture overview

The main function `compute_layout_visibility` loops over ~600 observer sample points (on a 60x44 table with 2" spacing). For each observer it:

1. Calls `extract_blocking_segments` — builds effective features list, iterates shapes, produces segment list
2. Calls `compute_visibility_polygon` — angular sweep: collects endpoints, generates 3 rays per endpoint, sorts rays, tests every ray against every segment
3. Computes `polygon_area` on the result
4. If DZs present: accumulates DZ visibility (PIP tests on DZ samples), cross-DZ seen sets, objective hidability

The dominant cost is step 2. With ~20 terrain shapes producing ~80 segments + 4 table boundary = ~84 segments, there are ~168 endpoints, ~504 rays, and 504x84 = ~42k ray-segment intersection tests **per observer**. Over 600 observers that's ~25M intersection tests.

---

## Part 1: Rust engine optimizations

### Implemented

#### 1. Precompute static segments once (DONE — 7.4x speedup alone)

Split `extract_blocking_segments` into:
- **`precompute_segments`** (called once): builds the effective features list, computes all static segments from regular obstacles, and precomputes obscuring shape footprints with edge normals.
- **`get_observer_segments`** (called per observer): starts from the precomputed static segments, then adds back-facing edges for obscuring features using precomputed normals.

This eliminated per-observer cloning of `PlacedFeature`, repeated `mirror_placed_feature` calls, and redundant `compose_transform`/`obb_corners` computations.

#### 5. Remove `corners_to_vec` allocation (DONE)

`Corners` (`[(f64, f64); 4]`) coerces to `&[(f64, f64)]` via slice, so the heap allocation was unnecessary. Removed.

#### 9. Avoid segment Vec copy in `compute_visibility_polygon` (DONE)

Replaced `segments.to_vec()` + push with `segments.iter().chain(table_boundary.iter())`. Table boundary is precomputed as `[Segment; 4]` outside the observer loop.

#### 10. Reuse allocations across observer iterations (DONE)

Introduced `VisBuffers` struct holding `endpoints`, `endpoint_seen`, `rays`, and `polygon` Vecs/HashSets. Allocated once before the loop, cleared with `.clear()` each iteration. `vis_poly` result Vec is also reused.

#### 14. `sort_unstable_by` for ray sorting (DONE)

Duplicate ray angles have no meaningful ordering, so stable sort was unnecessary overhead.

### Measured results

Criterion benchmarks, seed=42, single-threaded:

| Benchmark | Before | After | Speedup |
|---|---|---|---|
| generate_with_visibility_50 | 5.6 ms | 0.68 ms | **8.2x** |
| generate_with_visibility_100 | 9.5 ms | 1.35 ms | **7.0x** |

### Remaining Rust opportunities (not yet implemented)

#### 4. Parallelism (rayon) — Very high impact

The observer loop is embarrassingly parallel. DZ accumulation has shared mutable state but can use per-thread accumulators merged afterward. Near-linear speedup with core count. **Deferred for now** per user preference.

#### 3. Reduce ray count — Medium-high impact

Currently 3 rays per endpoint (angle, angle +/- epsilon). Could reduce to ~1.5x by skipping epsilon rays for table boundary endpoints and deduplicating near-identical rays.

#### 7. Index-based DZ keys instead of strings — Medium impact

`format!("{}_from_{}", ...)` called per DZ pair per observer (1200+ string allocations). Could use `(usize, usize)` tuple keys instead.

#### 8. BitVec for seen-tracking — Medium impact

`HashSet<usize>` for DZ sample tracking could be `Vec<bool>` for denser, faster membership tests. "Full visibility" path would become a `memset` instead of N individual inserts.

#### 6. Single-pass AABB — Low impact

4 separate iterator passes for min/max x/z could be a single fold.

#### 2. Spatial indexing — Low impact at current scale

Brute force ray-segment intersection is fine for typical ~84 segments. Would matter if layouts grow much larger.

---

## Part 2: Python engine optimizations

Baseline: **162 ms** (100 steps, seed 42, median of 5 runs).

### Implemented

| # | Optimization | Before | After | Speedup |
|---|---|---|---|---|
| P3 | `itertools.chain` instead of list copy | 162 ms | 135 ms | 16.7% |
| P1+P2 | Precompute static segments + cache transforms | 135 ms | 99.5 ms | 26.3% |
| P7 | Reuse normalized vector for direct ray, cache `math.*` as locals | 99.5 ms | 94.4 ms | 5.1% |
| P6 | NumPy vectorized observer filtering + inlined ray intersection | 94.4 ms | 65.7 ms | 30.4% |
| Cache | Incremental tall-footprint bitmask across visibility calls | 65.7 ms | 26.8 ms | 59.2% |
| | **Total** | **162 ms** | **26.8 ms** | **6.0x** |

### Key finding: observer-tall-footprint filtering is the real bottleneck

Profiling after P1-P3+P7 revealed that the ray-segment intersection was **not** the dominant cost. The actual bottleneck was the observer filtering step that removes sample points inside tall terrain:

```python
sample_points = [(x, z) for x, z in sample_points
                 if not any(_point_in_polygon(x, z, fp) for fp in tall_footprints)]
```

This accounted for **~46% of total visibility time** — ~600 observer points × N tall footprints × Python-level PIP calls. NumPy vectorized ray-casting PIP (testing all 600 points against each footprint in one pass) cut this dramatically.

Meanwhile, NumPy for the ray-segment intersection matrix (the original P6 plan) **broke even** at this problem scale (~500 rays × ~80 segments). Array creation overhead cancelled out the computation savings.

### Incremental tall-footprint cache (implemented)

`VisibilityCache` precomputes the observer sample grid once (constant for a given table + mission) and maintains a per-feature blocked-point count array. On each `compute_layout_visibility` call it diffs the current placed features against cached state — only recomputing PIP masks for added/removed features. Since each generation step changes at most one feature, the diff is typically 1 feature's mask vs re-testing all ~600 points against all N footprints.

### Remaining opportunities

- **P4**: Cache DZ membership per observer — marginal gain (only one code path executes)
- **P5**: Consolidate duplicate full-visibility logic — clarity only
- **P8**: `operator.itemgetter(0)` for ray sort — negligible
