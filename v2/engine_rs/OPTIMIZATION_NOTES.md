# Rust Visibility Optimization Notes

Status: **merged through rayon parallelism** — future work TBD

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

### Cumulative improvement (all optimizations including parallelism)
| Benchmark | Original | After ST opts | After rayon | Total improvement |
|---|---|---|---|---|
| visibility_50 | 872 ms | 476 ms | 178 ms | **-80%** |
| visibility_100 | 3.80 s | 1.68 s | 551 ms | **-85%** |
| mission_hna | 5.70 s | 4.93 s | 1.92 s | **-66%** |

## Attempted But Abandoned

### Segment-first loop reordering
Flip ray-outer/segment-inner to segment-outer/ray-inner with a `min_t[]` array (like `batch_point_in_polygon` does for PIP). The idea was to keep segment data in registers while scanning rays linearly.

**Result**: No clear improvement (±3% noise). The array-based `min_t` prevents register allocation of the per-ray minimum, and LLVM already does a good job with the original loop.

### Precomputed segment data
Precompute `(sx, sz, d_x1, d_z1, num_t)` per segment before the ray loop to avoid redundant arithmetic.

**Result**: Made things **worse** (~+15%). The 40-byte precomputed tuples increased the inner loop's memory footprint. The original 32-byte segment tuples with recomputed arithmetic were faster due to better cache behavior.

### Pseudoangle (replacing atan2)
Replace atan2 with a cheap pseudoangle function `p = dx / (|dx| + |dz|)` that maps monotonically to [0, 4). Eliminates all atan2 calls (ray generation + bucket assignment).

**Result**: Mixed. Helped visibility_50 (-19%) and visibility_100 (-14%), but mission_hna regressed (+12%). The pseudoangle maps non-linearly to real angles, causing uneven bucket distribution. Buckets near the cardinal axes become wider (more segments), creating load imbalance. Needs more investigation — the regression may have been noise (other workloads were running on the machine during benchmarking). **Worth retrying with a clean machine.**

## Profiling Results (mission_hna, post-rayon)

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

Organized by expected impact, informed by the profiling results above.

### Tier 1: PIP optimization (addresses ~94% of mission workload)

The current `batch_point_in_polygon` is a straightforward edge-crossing algorithm: for each polygon edge, iterate all test points and toggle their inside/outside bit. This is O(E × P) where E = visibility polygon edges (typically 10-40 after raycasting) and P = test points (~792 per DZ). It's called multiple times per observer:
- `dz_vis`: once per DZ (2 calls × ~792 pts each = ~1584 PIP tests per observer)
- `cross_dz`: once per cross-DZ pair, but only for unseen points (shrinks over time)
- `obj_hide`: once per objective per DZ, only for unseen points (~50-100 pts per objective)

#### 1a. AABB pre-filter on batch_point_in_polygon

**Simplest, try first.** Before the O(E×P) loop, compute the bounding box of the visibility polygon in O(E). Then skip the full PIP test for any point outside the AABB — those are guaranteed outside. Cost: O(E + P) for the filter, plus O(E × P') for the remaining points P' ⊆ P.

Effectiveness depends on how much of the DZ the visibility polygon covers. When terrain blocks most sightlines, the vis polygon is small and the AABB eliminates most DZ points cheaply. In the Hammer and Anvil mission, each DZ is an 18×44" rectangle; a partially-occluded visibility polygon might cover only 20-40% of a DZ's AABB, meaning 60-80% of points could be skipped.

**Expected impact**: 30-50% reduction in PIP time for mission workloads, or roughly **15-30% on mission_hna overall**. Minimal code change, no parity concerns (Rust-only optimization within a Rust-only parallel code path).

#### 1b. AABB pre-filter on the DZ sample points themselves

Rather than filtering per-observer, precompute a spatial index of DZ sample points (e.g., sort by Z coordinate). Then for each observer's vis polygon AABB, binary-search to find the Z-range of candidate points and only iterate those. This turns the O(P) scan into O(log P + P'), and combined with the X bounds check becomes very tight.

**Expected impact**: Potentially better than 1a for large DZs with small vis polygons. Slightly more setup code.

#### 1c. Horizontal slab decomposition for PIP

Preprocess the visibility polygon into horizontal slabs by sorting its vertices by Z coordinate. For each test point, binary-search to find its slab in O(log E), then test against only the 1-2 edges in that slab instead of all E edges. Turns PIP from O(E) per point to O(log E) per point, with O(E log E) setup per observer.

With E=20-40, this turns ~30 edge comparisons into ~5. Combined with AABB pre-filter, this could reduce PIP time by 80-90%.

**Expected impact**: Significant, but more complex to implement. Worth trying if 1a alone isn't sufficient. Setup cost (O(E log E) per observer) needs to be amortized over enough points — with ~792 DZ points per call this should pay for itself easily.

#### 1d. Restructure dz_vis to track "already fully visible" DZs

Currently `fraction_of_dz_visible_batch` tests all ~792 DZ points for every observer. But once a DZ point has been confirmed visible from enough observers, its contribution to the running average is well-established. We can't use the cross-DZ "unseen" trick directly (since `dz_vis` needs a per-observer fraction, not a boolean), but we could:
- Track running min/max fraction bounds per DZ
- Skip DZ fraction computation entirely for observers that can't change the outcome
- Or: precompute which DZ sample points are definitely inside the vis polygon AABB and only test the borderline ones

This is more speculative — the math is trickier than cross-DZ's boolean tracking.

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

1. **Start with 1a (AABB pre-filter)** — simplest change, highest expected impact. Add AABB computation to `batch_point_in_polygon` (or a wrapper) and skip points outside. Benchmark on mission_hna.
2. If 1a gives good results, try **1b (sorted DZ points)** for further refinement.
3. If PIP is still the bottleneck after 1a+1b, try **1c (slab decomposition)**.
4. Once PIP is under control, revisit **2a (pseudoangle hybrid)** for the non-mission benchmarks.
5. **3a (OBB caching)** is worth doing opportunistically — it's a clean code improvement regardless of performance impact.
