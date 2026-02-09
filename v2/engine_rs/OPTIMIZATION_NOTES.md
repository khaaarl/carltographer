# Rust Visibility Optimization Notes

Status: **in progress** on branch `feature/rust-visibility-optimization`

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

### Cumulative improvement
| Benchmark | Before | After | Improvement |
|---|---|---|---|
| visibility_50 | 872 ms | 476 ms | **-45%** |
| visibility_100 | 3.80 s | 1.68 s | **-56%** |
| mission_hna | 5.70 s | 4.93 s | **-14%** |

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

## Future Optimization Ideas (Not Yet Tried)

### Pseudoangle hybrid
Use pseudoangle for ray sorting (avoids atan2 in ray generation) but keep atan2 for bucket assignment (uniform bucket distribution). Gets most of the savings with uniform buckets.

### Faster endpoint deduplication
The `HashSet<(u64, u64)>` for endpoint dedup uses SipHash (DoS-resistant but slow). Options:
- Replace with `FxHashSet` from `rustc-hash` crate (would add a dependency)
- Implement a simple custom hasher (XOR + multiply)
- Skip dedup entirely (costs ~80% more rays but avoids hashing overhead — net effect unclear)

### Adaptive bucket count
Currently fixed at 64 buckets. Could scale with segment count: more segments → more buckets. But 64 seems reasonable for typical workloads (40-80 segments).

### DZ/objective PIP optimization
For the mission benchmark, a large fraction of time is spent in `batch_point_in_polygon` for DZ visibility, cross-DZ tracking, and objective hidability — not in the visibility polygon computation. Optimizing PIP (e.g., spatial partitioning of the visibility polygon) could help mission-heavy workloads.

### Parallelism (rayon)
The observer loop in `compute_layout_visibility` is embarrassingly parallel — each observer's visibility polygon is independent. Using `rayon::par_iter` could give near-linear speedup with core count. Concerns:
- Synchronization cost for DZ/cross-DZ/objective accumulators
- Would need atomic or per-thread accumulation with final merge
- Wouldn't affect Python engine parity (Rust-only optimization)
- User expressed caution about synchronization overhead

### SIMD intersection
Manually vectorize the ray-segment intersection using `std::arch` SIMD intrinsics (SSE2/AVX2). Process 2-4 rays simultaneously against one segment. Requires branchless arithmetic (replace conditionals with masks). Complex to implement and maintain.
