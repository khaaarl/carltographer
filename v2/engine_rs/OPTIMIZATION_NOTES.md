# Rust Visibility Optimization Notes

Status: **merged through Z-sorted PIP** — future work TBD

## Context

The visibility polygon computation (`compute_visibility_polygon` in `visibility.rs`) is the dominant cost during generation. It uses angular-sweep raycasting: cast rays toward segment endpoints (±epsilon), find the nearest intersection per ray, form the visibility polygon. The hot loop is O(R × S) where R = rays, S = segments.

Real-world workloads: ~100-200 unique endpoints → ~300-600 rays, ~40-80 terrain segments + 4 table boundary segments.

## Benchmark Setup (IMPORTANT)

The benchmark fixtures in `benches/generate_bench.rs` were originally using **2" tall crates**, which are below the 4" `min_blocking_height` threshold. This meant the "visibility" benchmarks weren't actually testing the intersection loop at all — every observer hit the fast path (0 segments). The fixtures were fixed to use **5" tall crates** as part of this work.

Baseline numbers (with corrected 5" crates, before any optimization):
- `visibility_50`: 872 ms
- `visibility_100`: 3.80 s
- `mission_hna`: 5.70 s

## Committed Optimizations (3 commits)

### 1. Angular bucketing (commit `47e451c`)
Partition segments by angular extent into 64 buckets. Each ray only tests segments in its bucket, reducing work from O(R×S) to O(R×k) where k ≈ average segments per bucket.

- visibility_50: 872ms → 688ms (**-21%**)
- visibility_100: 3.80s → 2.19s (**-42%**)
- mission_hna: 5.70s → 4.76s (**-16%**)

### 2. Trig reduction in ray generation (commit `43c1bfa`)
Replace 6 cos/sin calls per endpoint with 1 sqrt (normalization) + small-angle rotation for ±epsilon rays. Still uses atan2 for the sort key.

- visibility_50: 688ms → 548ms (**-20%**)
- visibility_100: 2.19s → 2.17s (~same, intersection-dominated)
- mission_hna: 4.76s → 4.82s (~same)

### 3. Inlined intersection + removed polygon buffer (commit `dfd7351`)
Inline `ray_segment_intersection` in the hot loop to enable early exit when `t >= min_t` (skips u computation). Eliminate intermediate `polygon` Vec, write directly to result.

- visibility_50: 548ms → 476ms (**-13%**)
- visibility_100: 2.17s → 1.68s (**-23%**)
- mission_hna: 4.82s → 4.93s (~same)

### Cumulative improvement (single-threaded optimizations)
| Benchmark | Before | After | Improvement |
|---|---|---|---|
| visibility_50 | 872 ms | 476 ms | **-45%** |
| visibility_100 | 3.80 s | 1.68 s | **-56%** |
| mission_hna | 5.70 s | 4.93 s | **-14%** |

### 4. Rayon parallel observer loop (commit `3f719bc`)
Parallelize the observer loop in `compute_layout_visibility` using `rayon::par_iter().fold().reduce()`. Each thread gets a `Box<ThreadAccum>` with its own working buffers (segments, vis_bufs, vis_poly, pip_buf, etc.) and accumulators (total_ratio, dz_vis_accum, dz_cross_seen, obj_seen_from_dz). Merge via sum for ratios/counts and OR for boolean seen arrays.

Note: rayon parallelism changes the order observers are processed, but the final result is mathematically identical (addition is commutative). Parity with Python engine is maintained.

- visibility_50: 476ms → 178ms (**-63%**)
- visibility_100: 1.68s → 551ms (**-67%**)
- mission_hna: 4.93s → 1.92s (**-61%**)

### Cumulative improvement (optimizations 1-4: ST opts + parallelism)
| Benchmark | Original | After ST opts | After rayon | Improvement |
|---|---|---|---|---|
| visibility_50 | 872 ms | 476 ms | 178 ms | **-80%** |
| visibility_100 | 3.80 s | 1.68 s | 551 ms | **-85%** |
| mission_hna | 5.70 s | 4.93 s | 1.92 s | **-66%** |

### 5. Z-sorted binary search for dz_vis PIP

The profiling data (see below) showed PIP operations dominate at ~94% of mission workload time. The `dz_vis` phase (60% alone) calls `fraction_of_dz_visible_batch` which runs `batch_point_in_polygon` testing ~792 DZ sample points against each observer's visibility polygon.

**Approach**: Pre-sort DZ sample points by Z-coordinate once during DZ setup (before the observer loop). In the PIP edge loop, replace the full O(P) point scan with a binary search per edge to find only points whose Z-coordinate falls in the edge's crossing range [min(zi, zj), max(zi, zj)). This exploits the fact that the edge-crossing test `(zi > pz) != (zj > pz)` can only be true when `pz` is between the edge's Z endpoints.

Data structure: `DzSortedSamples` holds `Vec<(z, x, original_index)>` sorted by z. Uses `slice::partition_point` (binary search) to find the start/end of each edge's Z-range.

Additional micro-optimization: precompute `1.0 / (zj - zi)` once per edge instead of dividing in the inner loop. Skip horizontal edges (zi == zj) entirely since they have no crossings.

Only applied to the `dz_vis` path (`fraction_of_dz_visible_zsorted`). The `cross_dz` and `obj_hide` paths continue using the original `batch_point_in_polygon` because their point sets change per-observer (unseen filtering).

- visibility_50: ~178ms → ~189ms (no DZs — unaffected, within noise)
- visibility_100: ~551ms → ~582ms (no DZs — unaffected, within noise)
- **mission_hna: ~1.92s → ~985ms (-49%)**

### Cumulative improvement (all optimizations including Z-sorted PIP)
| Benchmark | Original | After ST opts | After rayon | After Z-sorted | Total improvement |
|---|---|---|---|---|---|
| visibility_50 | 872 ms | 476 ms | 178 ms | ~178 ms | **-80%** |
| visibility_100 | 3.80 s | 1.68 s | 551 ms | ~551 ms | **-85%** |
| mission_hna | 5.70 s | 4.93 s | 1.92 s | ~985 ms | **-83%** |

## Attempted But Abandoned

### Segment-first loop reordering
Flip ray-outer/segment-inner to segment-outer/ray-inner with a `min_t[]` array (like `batch_point_in_polygon` does for PIP). The idea was to keep segment data in registers while scanning rays linearly.

**Result**: No clear improvement (±3% noise). The array-based `min_t` prevents register allocation of the per-ray minimum, and LLVM already does a good job with the original loop.

### Precomputed segment data
Precompute `(sx, sz, d_x1, d_z1, num_t)` per segment before the ray loop to avoid redundant arithmetic.

**Result**: Made things **worse** (~+15%). The 40-byte precomputed tuples increased the inner loop's memory footprint. The original 32-byte segment tuples with recomputed arithmetic were faster due to better cache behavior.

### AABB pre-filter on batch_point_in_polygon (Tier 1a from roadmap)
Compute the bounding box of the visibility polygon and build a candidates list of points inside the AABB. Only run the edge-crossing loop on candidates. The idea was to skip 60-80% of DZ points when the vis polygon is small.

**Result**: No improvement on mission_hna. With the benchmark's sparse terrain (5 small crates on a 60×44" table), each observer's visibility polygon covers nearly the entire table. The vis polygon AABB therefore overlaps almost all DZ sample points (>95%), so the pre-filter skips very few points while adding overhead for building the candidates list and introducing index indirection in the inner loop. The optimization would be effective with denser terrain creating more occlusion, but the benchmark's sparsity makes it counterproductive.

The AABB filter has fundamentally wrong assumptions about the problem structure: in visibility analysis, most observers see most of the table. The vis polygon is usually large (80-95% of table area), making the AABB nearly the full table. A per-edge approach (like Z-sorted binary search) is more effective because it operates on the polygon's edges rather than its bounding box.

### Pseudoangle (replacing atan2)
Replace atan2 with a cheap pseudoangle function `p = dx / (|dx| + |dz|)` that maps monotonically to [0, 4). Eliminates all atan2 calls (ray generation + bucket assignment).

**Result**: Mixed. Helped visibility_50 (-19%) and visibility_100 (-14%), but mission_hna regressed (+12%). The pseudoangle maps non-linearly to real angles, causing uneven bucket distribution. Buckets near the cardinal axes become wider (more segments), creating load imbalance. Needs more investigation — the regression may have been noise (other workloads were running on the machine during benchmarking). **Worth retrying with a clean machine.**

## Profiling Results (mission_hna, post-rayon, pre-Z-sorted)

Instrumented `compute_layout_visibility` with per-phase timing (sum of all thread-nanoseconds). Late-game steps (more terrain, ~730 observers, ~20 segments):

| Phase | Thread-ms | % of observer loop |
|---|---|---|
| `get_observer_segments` | 0.2 | ~0.04% |
| `compute_visibility_polygon` | 35-48 | **~8%** |
| `dz_vis` (fraction_of_dz_visible_batch) | 270-330 | **~60%** |
| `cross_dz` (batch_point_in_polygon) | 120-148 | **~27%** |
| `obj_hide` (batch_point_in_polygon) | 30-47 | **~7%** |
| `precompute_segments` | 0.0 | negligible |

**Key finding**: PIP (point-in-polygon) operations now dominate at **~94%** of observer-loop time. The raycasting intersection loop (`vis_polygon`) is only ~8%. For mission workloads, optimizing `batch_point_in_polygon` and `fraction_of_dz_visible_batch` is the highest-leverage target.

The `dz_vis` phase tests ~792 DZ sample points against each observer's visibility polygon (2 DZs × ~730 observers). The `cross_dz` phase tests the same DZ points but with an unseen-only filter that reduces work as more points get marked seen.

## Future Optimization Ideas (Not Yet Tried)

Organized by expected impact. Updated after Z-sorted PIP optimization.

### Tier 1: Remaining PIP optimization (cross_dz + obj_hide paths)

The Z-sorted binary search addressed the `dz_vis` path (~60% of pre-optimization observer time). The `cross_dz` (~27%) and `obj_hide` (~7%) paths still use the original O(E×P) `batch_point_in_polygon`. However, these paths already shrink their point sets via unseen filtering, so the effective P decreases over time. Fresh profiling post-Z-sorted would clarify the new bottleneck distribution.

#### 1a. Z-sorted binary search for cross_dz/obj_hide

Apply the same Z-sort technique to the cross_dz and obj_hide paths. Challenge: these paths work on `unseen_points` which change per-observer (points are filtered as they become seen). Options:
- Pre-sort the full DZ points once, then maintain a parallel `unseen` bitmask instead of building `unseen_points` Vec. The Z-sorted PIP would iterate the sorted array, skip seen entries, and only process unseen ones in the Z-range.
- Or: accept the O(P log P) sort cost per-call. With P shrinking quickly (few hundred → few dozen), the sort is cheap.

**Expected impact**: Moderate. The cross_dz/obj_hide paths already benefit from unseen filtering. The Z-sort would help most in early iterations when most points are still unseen.

#### 1b. Horizontal slab decomposition for PIP

Preprocess the visibility polygon into horizontal slabs by sorting its vertices by Z coordinate. For each test point, binary-search to find its slab in O(log E), then test against only the 1-2 edges in that slab instead of all E edges. Turns PIP from O(E) per point to O(log E) per point, with O(E log E) setup per observer.

With E=20-40, this turns ~30 edge comparisons into ~5. This is complementary to (not competing with) the Z-sorted approach — Z-sorting reduces P per edge, slab decomposition reduces E per point.

**Expected impact**: Moderate improvement on cross_dz/obj_hide. For dz_vis, the Z-sorted approach already handles the P dimension; slab decomposition would help reduce the remaining edge work but with diminishing returns.

### Tier 2: Raycasting refinements (addresses ~8% of mission, more on non-mission)

These would have been high priority before the profiling showed PIP dominance. Still worth pursuing for `visibility_50` and `visibility_100` benchmarks where DZ scoring is absent.

#### 2a. Pseudoangle hybrid
Use pseudoangle `p = dx / (|dx| + |dz|)` for ray sorting (avoids atan2 in ray generation) but keep atan2 for bucket assignment (uniform bucket distribution). Gets most of the atan2 savings without the uneven-bucket regression that the full pseudoangle approach showed. The pure pseudoangle version improved visibility_50 by 19% and visibility_100 by 14% — this hybrid should capture most of that while avoiding the mission_hna regression.

#### 2b. Faster endpoint deduplication
The `HashSet<(u64, u64)>` for endpoint dedup uses SipHash (DoS-resistant but slow). With rayon, each thread has its own HashSet, so cost is multiplied by thread count. Options:
- Replace with `FxHashSet` from `rustc-hash` crate (fast hash, would add a dependency)
- Implement a simple custom hasher (XOR + multiply, no dependency)
- Skip dedup entirely (costs ~80% more rays but avoids hashing overhead — net effect unclear)

#### 2c. SIMD intersection
Manually vectorize the ray-segment intersection using `std::arch` SIMD intrinsics (SSE2/AVX2). Process 2-4 rays simultaneously against one segment. Requires branchless arithmetic (replace conditionals with masks). Complex to implement and maintain. Probably not worthwhile unless raycasting becomes the bottleneck again.

#### 2d. Adaptive bucket count
Currently fixed at 64 buckets. Could scale with segment count: more segments → more buckets. But 64 seems reasonable for typical workloads (40-80 segments). Low priority.

### Tier 3: Collision / mutation path (affects all workloads)

These don't affect the visibility-heavy benchmarks much, but they matter for the mutation loop (which runs every step regardless of visibility). The `basic_100` and `all_features` benchmarks (~380-590µs) are bottlenecked here.

#### 3a. Redundant OBB computation in is_valid_placement
`get_world_obbs()` and `get_tall_world_obbs()` are called multiple times for the *same* features during a single `is_valid_placement()` call — once for overlap checking, again for all-feature gap, again for tall-feature gap. Each call recomputes cos/sin and allocates a new Vec. Computing OBBs once per feature and filtering by height would eliminate redundant trig and allocation. Called on every mutation attempt.

#### 3b. Mirror feature cloning in hot paths
In both `is_valid_placement()` and `precompute_segments()`, mirror features for rotationally symmetric layouts are cloned into a fresh Vec on every call. This doubles allocations. Could precompute mirrors once per step, or use lazy iterator adapters that yield mirrors without allocating.

### Tier 4: Tempering / allocation overhead

Low individual impact, but easy wins that reduce overhead across the entire run.

#### 4a. Pre-allocate sub_undos
`Vec::with_capacity(num_mutations)` is allocated fresh for every step of every replica (~2000+ allocations per run). Could pre-allocate once per replica and reuse with `clear()`.

#### 4b. Layout cloning on swap and best-tracking
When tracking best layouts per replica and doing replica swaps, the full `TerrainLayout` (including all PlacedFeature vecs) gets cloned. With many replicas and frequent improvements this adds up. Could use COW semantics or only clone the diff.

#### 4c. Single-pass polygon bounds
`sample_points_in_polygon()` does 4 separate fold operations (min_x, max_x, min_z, max_z) over the polygon vertices. Trivial to merge into a single pass. Minor but free.

### Tier 5: Algorithmic / architectural (high complexity, speculative)

These are bigger changes that could yield large improvements but would require significant implementation effort and may affect engine parity.

#### 5a. Incremental visibility
Instead of recomputing the full visibility from scratch after each mutation, incrementally update. When a feature moves, only observers near it have their visibility changed. Track which sample points are "affected" by each feature (based on angular extent from observer) and only recompute those observers. Could reduce per-step visibility cost from O(observers × segments) to O(affected_observers × segments).

**Caveat**: Extremely complex to implement correctly, especially with the DZ/cross-DZ/objective accumulation. Would not maintain parity with Python engine (Rust-only optimization). Risk of subtle correctness bugs.

#### 5b. Reduced-density adaptive sampling
Use coarser grid in open areas, finer near terrain or DZ boundaries. Would change scoring semantics and require matching in Python, so probably not worth the complexity for a Rust-only optimization.

### Recommended next steps

1. **Re-profile mission_hna** to see the new bottleneck distribution post-Z-sorted. The dz_vis path should be dramatically faster; cross_dz and vis_polygon may now be the top targets.
2. **Try 2a (pseudoangle hybrid)** for the non-mission benchmarks (visibility_50/100). This addresses the raycasting path which is still the bottleneck when DZs are absent.
3. **Try 1a (Z-sorted cross_dz)** — apply Z-sorted binary search to the cross_dz path using a bitmask approach (skip seen points in the sorted array rather than building unseen_points).
4. **3a (OBB caching)** is worth doing opportunistically — it's a clean code improvement regardless of performance impact.
5. **Add a denser benchmark fixture** — the current mission_hna uses sparse 5×2.5" crates. A fixture with 10+ large ruins/obstacles would better represent real-world usage and stress different code paths.
