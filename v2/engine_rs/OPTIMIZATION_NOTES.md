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

### Benchmark fixtures

| Benchmark | Table | Catalog | Mission | Steps | Visibility |
|---|---|---|---|---|---|
| `basic_100` | 60×44 | crates (2") | none | 100 | off |
| `all_features` | 60×44 | crates (2") | none | 100 | off |
| `visibility_50` | 60×44 | crates (5") | none | 50 | on |
| `visibility_100` | 60×44 | crates (5") | none | 100 | on |
| `mission_hna` | 60×44 | crates (5") | HnA | 50 | on + DZ |
| `mission_ruins` | 44×30 | crates + ruins + walls | HnA | 50 | on + DZ |

The `mission_ruins` fixture uses the WTC-style catalog matching the parity test default: 5" crates, 10×6" ruin bases (obscuring), and 6×0.5×5" opaque walls within ruins. This exercises multi-component features, obscuring segment generation, and a denser terrain mix on a smaller table.

## Committed Optimizations (3 commits)

### 1. Angular bucketing (commit `47e451c`)
Partition segments by angular extent into 64 buckets. Each ray only tests segments in its bucket, reducing work from O(R×S) to O(R×k) where k ≈ average segments per bucket.

- visibility_50: 872ms → 688ms (**-21%**)
- visibility_100: 3.80s → 2.19s (**-42%**)
- mission_hna: 5.70s → 4.76s (**-16%**)

### 2. ~~Trig reduction in ray generation~~ (commit `43c1bfa`, **reverted**)
Originally replaced 6 cos/sin calls per endpoint with 1 sqrt + small-angle rotation matrix for ±epsilon rays. **Reverted for FP parity**: the rotation matrix `ndx * cos_eps ± ndz * sin_eps` produces different IEEE 754 results than Python's `cos(angle ± eps)` / `sin(angle ± eps)`, causing visibility polygon vertex differences that propagated into DZ PIP boundary tests. The code now uses `cos(angle ± eps)` / `sin(angle ± eps)` (4 trig calls per endpoint) to match Python exactly.

### 3. Inlined intersection + removed polygon buffer (commit `dfd7351`)
Inline `ray_segment_intersection` in the hot loop to enable early exit when `t >= min_t` (skips u computation). Eliminate intermediate `polygon` Vec, write directly to result.

- visibility_50: 688ms → 583ms (**-15%**)
- visibility_100: 2.19s → 1.84s (**-16%**)
- mission_hna: 4.76s → 4.88s (~same)

*Note: numbers updated to reflect opt #2 revert (baseline is post-bucketing, not post-trig-reduction).*

### Cumulative improvement (single-threaded optimizations)
| Benchmark | Before | After | Improvement |
|---|---|---|---|
| visibility_50 | 872 ms | ~583 ms | **-33%** |
| visibility_100 | 3.80 s | ~1.84 s | **-52%** |
| mission_hna | 5.70 s | ~4.88 s | **-14%** |

*Note: cumulative ST improvement is lower than originally reported because opt #2 (trig reduction) was reverted.*

### 4. Rayon parallel observer loop (commit `3f719bc`)
Parallelize the observer loop in `compute_layout_visibility` using `rayon::par_iter().fold().reduce()`. Each thread gets a `Box<ThreadAccum>` with its own working buffers (segments, vis_bufs, vis_poly, pip_buf, etc.) and accumulators (total_ratio, dz_vis_accum, dz_cross_seen, obj_seen_from_dz). Merge via sum for ratios/counts and OR for boolean seen arrays.

Note: rayon parallelism changes the order observers are processed, but the final result is mathematically identical (addition is commutative). Parity with Python engine is maintained.

*Numbers below estimated relative to the adjusted ST baseline (post opt #2 revert):*

- visibility_50: ~583ms → ~218ms (**-63%**)
- visibility_100: ~1.84s → ~606ms (**-67%**)
- mission_hna: ~4.88s → ~1.90s (**-61%**)

### Cumulative improvement (optimizations 1, 3, 4: ST opts + parallelism)
| Benchmark | Original | After ST opts | After rayon | Improvement |
|---|---|---|---|---|
| visibility_50 | 872 ms | ~583 ms | ~218 ms | **-75%** |
| visibility_100 | 3.80 s | ~1.84 s | ~606 ms | **-84%** |
| mission_hna | 5.70 s | ~4.88 s | ~1.90 s | **-67%** |

*Note: opt #2 (trig reduction) was reverted for FP parity, so ST improvement is slightly less than originally reported.*

### 5. Z-sorted binary search for dz_vis PIP

The profiling data (see below) showed PIP operations dominate at ~94% of mission workload time. The `dz_vis` phase (60% alone) calls `fraction_of_dz_visible_batch` which runs `batch_point_in_polygon` testing ~792 DZ sample points against each observer's visibility polygon.

**Approach**: Pre-sort DZ sample points by Z-coordinate once during DZ setup (before the observer loop). In the PIP edge loop, replace the full O(P) point scan with a binary search per edge to find only points whose Z-coordinate falls in the edge's crossing range [min(zi, zj), max(zi, zj)). This exploits the fact that the edge-crossing test `(zi > pz) != (zj > pz)` can only be true when `pz` is between the edge's Z endpoints.

Data structure: `DzSortedSamples` holds `Vec<(z, x, original_index)>` sorted by z. Uses `slice::partition_point` (binary search) to find the start/end of each edge's Z-range.

~~Additional micro-optimization: precompute `1.0 / (zj - zi)` once per edge instead of dividing in the inner loop.~~ **Reverted**: the precomputed `1.0 / dz` then `dx * (pz - zi) * inv_dz` produces different IEEE 754 results than direct division `dx * (pz - zi) / dz` for boundary points, causing PIP in/out flips. Now uses direct division to match Python and the batch PIP path. Skip horizontal edges (zi == zj) entirely since they have no crossings.

Initially applied only to the `dz_vis` path (`fraction_of_dz_visible_zsorted`). The `cross_dz` and `obj_hide` paths continued using the original `batch_point_in_polygon`.

- visibility_50: ~178ms → ~189ms (no DZs — unaffected, within noise)
- visibility_100: ~551ms → ~582ms (no DZs — unaffected, within noise)
- **mission_hna: ~1.92s → ~985ms (-49%)**

### 6. Z-sorted PIP for cross_dz and obj_hide (skip-seen)

Post-Z-sorted profiling showed the new bottleneck distribution:
- cross_dz: **~45-50%** (was 27%, now dominant)
- obj_hide: **~25-30%** (was 7%)
- vis_poly: **~15-18%** (was 8%)
- dz_vis: **~9-12%** (was 60%, dramatically reduced by opt #5)

Applied the same Z-sorted binary search to cross_dz and obj_hide paths via a new `pip_zsorted_update_seen()` function. This function:
- Uses the same pre-sorted DzSortedSamples (DZ points for cross_dz, objective sample points for obj_hide)
- Skips already-seen entries in the inner loop (`if seen[oidx] { continue; }`) — avoids building the `unseen_points` Vec entirely
- Directly updates the `seen` boolean array after the edge loop

Objective sample points are also pre-sorted during setup (`obj_sorted_samples`). Removed the now-unused `unseen_indices` and `unseen_points` working buffers from ThreadAccum.

- visibility_50: ~178ms (unaffected)
- visibility_100: ~551ms (unaffected)
- **mission_hna: ~985ms → ~584ms (-41%)**

### Cumulative improvement (all optimizations)
| Benchmark | Original | Current | Total improvement |
|---|---|---|---|
| visibility_50 | 872 ms | **~201 ms** | **-77%** |
| visibility_100 | 3.80 s | **~719 ms** | **-81%** |
| mission_hna | 5.70 s | **~629 ms** | **-89%** |

*Measured February 2026 after all optimizations including FP parity reversions.*

A new `mission_ruins` benchmark was added with the WTC-style catalog (crates + ruins + opaque walls, 44×30" table, Hammer and Anvil mission). Baseline: **~371 ms** (50 steps). This exercises multi-component obscuring features and a denser, more realistic terrain mix than the crate-only `mission_hna`.

### FP parity reversions (post-optimization correctness fixes)

Three micro-optimizations were reverted because they produced different IEEE 754 floating-point results than the Python engine, breaking bit-identical parity:

1. **Trig reduction (opt #2)**: Rotation matrix for ±eps rays → reverted to `cos(angle ± eps)` / `sin(angle ± eps)`. Cost: +4 trig calls per endpoint.
2. **Precomputed inv_dz (opt #5 micro-opt)**: `1.0 / (zj - zi)` then multiply → reverted to direct division `/ (zj - zi)`. Cost: 1 division per edge per Z-range point instead of 1 multiply.
3. **Ray normalization inv_len**: `dx * (1.0 / len)` → reverted to `dx / len`. Cost: negligible (1 division vs 1 multiply, once per endpoint).

Net performance impact of reversions: ~8% regression on visibility_50 (201ms vs 178ms), ~23% on visibility_100 (719ms vs 586ms), ~8% on mission_hna (629ms vs 584ms). The visibility_100 regression is largest because raycasting (where trig reduction helped most) is a bigger fraction of non-mission workloads.

**Lesson learned**: Any arithmetic expression that produces values used in boundary tests (PIP edge crossings, visibility polygon vertices fed to DZ PIP) must use the exact same FP expression order as Python. `a * b * (1/c)` ≠ `a * b / c` and `cos(atan2(y,x) ± eps)` ≠ rotation matrix at IEEE 754 level.

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

## Profiling Results

### Post-Z-sorted-dz_vis, pre-Z-sorted-cross_dz (mission_hna)

Late-game steps (~760 observers, ~20 segments):

| Phase | Thread-ms | % of observer loop |
|---|---|---|
| `compute_visibility_polygon` | 30-50 | **~15-18%** |
| `dz_vis` (Z-sorted) | 20-30 | **~9-12%** |
| `cross_dz` (batch PIP) | 100-130 | **~45-50%** |
| `obj_hide` (batch PIP) | 55-80 | **~25-30%** |

cross_dz became the dominant bottleneck after dz_vis was optimized. This motivated extending Z-sorted PIP to cross_dz and obj_hide paths.

### Post-rayon, pre-Z-sorted (mission_hna, historical)

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

### Tier 1: Further PIP optimization

All three PIP paths (dz_vis, cross_dz, obj_hide) now use Z-sorted binary search. The remaining PIP optimization ideas target algorithmic improvements or further reducing per-point work.

#### 1a. Horizontal slab decomposition for PIP

Preprocess the visibility polygon into horizontal slabs by sorting its vertices by Z coordinate. For each test point, binary-search to find its slab in O(log E), then test against only the 1-2 edges in that slab instead of all E edges. Turns PIP from O(E) per point to O(log E) per point, with O(E log E) setup per observer.

With E=20-40, this turns ~30 edge comparisons into ~5. This is complementary to (not competing with) the Z-sorted approach — Z-sorting reduces P per edge, slab decomposition reduces E per point.

**Expected impact**: Moderate improvement on cross_dz/obj_hide. For dz_vis, the Z-sorted approach already handles the P dimension; slab decomposition would help reduce the remaining edge work but with diminishing returns.

### Tier 2: Raycasting refinements (addresses ~8% of mission, more on non-mission)

These would have been high priority before the profiling showed PIP dominance. Still worth pursuing for `visibility_50` and `visibility_100` benchmarks where DZ scoring is absent.

#### 2a. Pseudoangle hybrid
Use pseudoangle `p = dx / (|dx| + |dz|)` for ray sorting (avoids atan2 in ray generation) but keep atan2 for bucket assignment (uniform bucket distribution). Gets most of the atan2 savings without the uneven-bucket regression that the full pseudoangle approach showed.

**Note**: Since the trig reduction (opt #2) was reverted, ray generation now does `atan2 + 4×cos/sin + 1×sqrt + 2×division` per endpoint. Pseudoangle could replace the atan2 for sort-key computation (but the 4 cos/sin calls for ±eps rays must stay for FP parity). The potential savings are smaller than originally estimated since atan2 is now a smaller fraction of ray generation cost.

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

1. **Re-profile** post-parity-reversions to see updated bottleneck distribution. The trig revert likely shifted more time back into ray generation (vis_polygon phase).
2. **Try 1a (slab decomposition)** — the Z-sorted PIP paths are the proven winner; slab decomposition would further reduce edge work per point and is complementary.
3. **3a (OBB caching)** is worth doing opportunistically — it's a clean code improvement regardless of performance impact.
4. **Use the `mission_ruins` benchmark** for more realistic profiling — it has obscuring features with multi-component terrain (ruins + walls) and a smaller table (44×30"), matching the updated parity test catalog.
