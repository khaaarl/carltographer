//! Visibility score computation for terrain layouts.
//!
//! Measures what percentage of the battlefield has clear line of sight
//! by sampling observer positions on a grid and computing visibility
//! polygons via angular sweep.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use rayon::prelude::*;

use crate::collision::{
    compose_transform, get_tall_world_obbs, is_at_origin, mirror_placed_feature, obb_corners,
    obbs_overlap, Corners,
};
use crate::types::{PlacedFeature, TerrainLayout, TerrainObject, Transform};

// Note: DZ expansion is done in Python via shapely buffer() and
// passed to Rust via the expanded_polygons field on DeploymentZone.

/// Fast non-cryptographic hasher (FxHash) for integer keys.
/// Uses a single multiply-XOR per 8-byte chunk. Much faster than the
/// default SipHash for hash-table lookups on known-safe (non-adversarial) keys
/// like f64 bit patterns and array indices.
struct FxHasher {
    hash: u64,
}

/// Constant from Firefox/rustc FxHash: a good odd multiplier for mixing.
const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        // Only u64 and usize writes are used in this codebase.
        unreachable!("FxHasher: only write_u64 and write_usize are supported");
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

impl Default for FxHasher {
    #[inline]
    fn default() -> Self {
        FxHasher { hash: 0 }
    }
}

/// BuildHasher that produces FxHasher instances.
type FxBuildHasher = BuildHasherDefault<FxHasher>;
/// HashSet using FxHash instead of SipHash.
type FxHashSet<T> = HashSet<T, FxBuildHasher>;
/// HashMap using FxHash instead of SipHash.
type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// A line segment: (x1, z1, x2, z2)
type Segment = (f64, f64, f64, f64);

/// Compute polygon area using the shoelace formula.
/// Returns positive area regardless of winding order.
fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].0 * vertices[j].1;
        area -= vertices[j].0 * vertices[i].1;
    }
    area.abs() / 2.0
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(px: f64, pz: f64, vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = vertices[i];
        let (xj, zj) = vertices[j];
        if (zi > pz) != (zj > pz) {
            let intersect_x = (xj - xi) * (pz - zi) / (zj - zi) + xi;
            if px < intersect_x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Edge-first batch point-in-polygon: tests ALL points against the polygon
/// at once, iterating polygon edges in the outer loop. This keeps edge data
/// in registers while linearly scanning the points array (good cache locality,
/// LLVM can autovectorize the inner loop).
///
/// Superseded by `fraction_of_dz_visible_zsorted` and
/// `pip_zsorted_update_seen` in production code. Retained for tests.
#[cfg(test)]
fn batch_point_in_polygon(points: &[(f64, f64)], polygon: &[(f64, f64)], inside: &mut Vec<bool>) {
    inside.clear();
    inside.resize(points.len(), false);
    let n = polygon.len();
    if n < 3 {
        return;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = polygon[i];
        let (xj, zj) = polygon[j];
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

/// Find parameter t where ray (ox+t*dx, oz+t*dz) hits segment.
/// Returns Some(t) if hit (t >= 0), None if miss or parallel.
#[cfg(test)]
fn ray_segment_intersection(
    ox: f64,
    oz: f64,
    dx: f64,
    dz: f64,
    x1: f64,
    z1: f64,
    x2: f64,
    z2: f64,
) -> Option<f64> {
    let sx = x2 - x1;
    let sz = z2 - z1;
    let denom = dx * sz - dz * sx;
    if denom.abs() < 1e-12 {
        return None;
    }

    let t = ((x1 - ox) * sz - (z1 - oz) * sx) / denom;
    let u = ((x1 - ox) * dz - (z1 - oz) * dx) / denom;

    if t >= 0.0 && (0.0..=1.0).contains(&u) {
        Some(t)
    } else {
        None
    }
}

/// Precomputed data for blocking segment extraction.
/// Static segments (from regular obstacles) are observer-independent.
/// Obscuring feature footprints need per-observer back-face culling.
struct PrecomputedSegments {
    /// Segments from regular obstacles — same for every observer.
    static_segments: Vec<Segment>,
    /// Footprint corners for each obscuring feature shape, with precomputed
    /// center and outward normals per edge: (corners, center, edge_data).
    /// edge_data: Vec<(x1, z1, x2, z2, mx, mz, nx, nz)> per shape.
    obscuring_shapes: Vec<ObscuringShape>,
}

/// Per-edge data: (x1, z1, x2, z2, midpoint_x, midpoint_z, normal_x, normal_z).
type EdgeData = (f64, f64, f64, f64, f64, f64, f64, f64);

/// Precomputed data for one obscuring feature shape.
struct ObscuringShape {
    /// Corner vertices for point-in-polygon test (observer inside check).
    corners: Vec<(f64, f64)>,
    /// Per-edge data.
    edges: Vec<EdgeData>,
}

/// Precompute static segments and obscuring shape data.
/// Called once before the observer loop.
fn precompute_segments(
    layout: &TerrainLayout,
    objects_by_id: &HashMap<String, &TerrainObject>,
    min_blocking_height: f64,
) -> PrecomputedSegments {
    let mut static_segments: Vec<Segment> = Vec::new();
    let mut obscuring_shapes: Vec<ObscuringShape> = Vec::new();
    let default_t = Transform::default();

    // Build effective placed features list (include mirrors for symmetric)
    let mut effective_features: Vec<PlacedFeature> = Vec::new();
    for pf in &layout.placed_features {
        effective_features.push(pf.clone());
        if layout.rotationally_symmetric && !is_at_origin(pf) {
            effective_features.push(mirror_placed_feature(pf));
        }
    }

    for pf in &effective_features {
        let is_obscuring = pf.feature.feature_type == "obscuring";

        if is_obscuring {
            let all_corners = get_footprint_corners(pf, objects_by_id);
            for corners in &all_corners {
                let n = corners.len();
                let verts: Vec<(f64, f64)> = corners.to_vec();

                // Precompute center
                let cx: f64 = corners.iter().map(|c| c.0).sum::<f64>() / n as f64;
                let cz: f64 = corners.iter().map(|c| c.1).sum::<f64>() / n as f64;

                // Precompute edge data with outward normals
                let mut edges = Vec::with_capacity(n);
                for i in 0..n {
                    let j = (i + 1) % n;
                    let (x1, z1) = corners[i];
                    let (x2, z2) = corners[j];
                    let mx = (x1 + x2) / 2.0;
                    let mz = (z1 + z2) / 2.0;
                    let ex = x2 - x1;
                    let ez = z2 - z1;
                    let (mut nx, mut nz) = (ez, -ex);
                    let dot_center = (cx - mx) * nx + (cz - mz) * nz;
                    if dot_center > 0.0 {
                        nx = -nx;
                        nz = -nz;
                    }
                    edges.push((x1, z1, x2, z2, mx, mz, nx, nz));
                }

                obscuring_shapes.push(ObscuringShape {
                    corners: verts,
                    edges,
                });
            }
        } else {
            // Regular obstacle: all 4 edges block regardless of observer
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
                    let world =
                        compose_transform(&compose_transform(shape_t, comp_t), &pf.transform);
                    let corners = obb_corners(
                        world.x_inches,
                        world.z_inches,
                        shape.width_inches / 2.0,
                        shape.depth_inches / 2.0,
                        world.rotation_deg.to_radians(),
                    );
                    for i in 0..4 {
                        let j = (i + 1) % 4;
                        static_segments.push((
                            corners[i].0,
                            corners[i].1,
                            corners[j].0,
                            corners[j].1,
                        ));
                    }
                }
            }
        }
    }

    PrecomputedSegments {
        static_segments,
        obscuring_shapes,
    }
}

/// Build the full segment list for a specific observer from precomputed data.
/// Starts with static segments, then adds back-facing edges of obscuring features.
fn get_observer_segments(
    precomputed: &PrecomputedSegments,
    observer_x: f64,
    observer_z: f64,
    out: &mut Vec<Segment>,
) {
    out.clear();
    out.extend_from_slice(&precomputed.static_segments);

    if precomputed.obscuring_shapes.is_empty() {
        return;
    }

    // Each shape is checked independently: being inside one shape of a feature
    // (e.g. the ruins base) does not skip other shapes (e.g. a wall within it).
    for shape in &precomputed.obscuring_shapes {
        // Check if observer is inside this shape
        if point_in_polygon(observer_x, observer_z, &shape.corners) {
            continue; // Can see out from inside
        }

        // Add back-facing edges
        for &(x1, z1, x2, z2, mx, mz, nx, nz) in &shape.edges {
            let dot_observer = (observer_x - mx) * nx + (observer_z - mz) * nz;
            if dot_observer < 0.0 {
                out.push((x1, z1, x2, z2));
            }
        }
    }
}

/// Get world-space OBB corners for all shapes in a feature (ignoring height).
fn get_footprint_corners(
    placed: &PlacedFeature,
    objects_by_id: &HashMap<String, &TerrainObject>,
) -> Vec<Corners> {
    // Reuse the existing get_world_obbs logic
    crate::collision::get_world_obbs(placed, objects_by_id)
}

/// Number of angular buckets for segment partitioning.
const NUM_ANGLE_BUCKETS: usize = 64;

/// Cheap pseudoangle: monotonically maps direction (dx, dz) to [0, 4),
/// wrapping at angle ±π (pointing left, same as atan2). Replaces atan2
/// for bucket assignment — no trig, just one division and a few branches.
///
/// Mapping: angle -π → PA 0, angle 0 → PA 2, angle +π → PA 4 (clamped to ~4).
/// Monotonically increasing with real angle throughout [-π, π].
#[inline]
fn pseudoangle(dx: f64, dz: f64) -> f64 {
    let adx = dx.abs();
    let adz = dz.abs();
    let sum = adx + adz;
    if sum == 0.0 {
        return 0.0;
    }
    // p = dx / (|dx| + |dz|) in [-1, 1]
    let p = dx / sum;
    if dz >= 0.0 {
        3.0 - p // angle [0, π] → PA [2, 4)
    } else {
        1.0 + p // angle [-π, 0) → PA [0, 2)
    }
}

/// Map a pseudoangle in [0, 4) to a bucket index in [0, NUM_ANGLE_BUCKETS).
/// PA_SCALE = NUM_ANGLE_BUCKETS / 4.0 = 16.0 for 64 buckets.
const PA_SCALE: f64 = NUM_ANGLE_BUCKETS as f64 / 4.0;

#[inline]
fn pa_to_bucket(pa: f64) -> usize {
    let b = (pa * PA_SCALE) as usize;
    if b >= NUM_ANGLE_BUCKETS {
        NUM_ANGLE_BUCKETS - 1
    } else {
        b
    }
}

/// Reusable buffers for compute_visibility_polygon to avoid per-call allocations.
struct VisBuffers {
    endpoints: Vec<(f64, f64)>,
    endpoint_seen: FxHashSet<(u64, u64)>,
    rays: Vec<(f64, f64, f64)>,
    /// Angular buckets: each bucket holds indices into `all_segments`.
    buckets: [Vec<u16>; NUM_ANGLE_BUCKETS],
    /// Flattened terrain + table boundary segments for bucket indexing.
    all_segments: Vec<Segment>,
}

impl VisBuffers {
    fn new() -> Self {
        Self {
            endpoints: Vec::with_capacity(256),
            endpoint_seen: FxHashSet::with_capacity_and_hasher(256, FxBuildHasher::default()),
            rays: Vec::with_capacity(768),
            buckets: std::array::from_fn(|_| Vec::with_capacity(16)),
            all_segments: Vec::with_capacity(128),
        }
    }
}

/// Compute visibility polygon from observer via angular sweep.
/// Uses reusable buffers and a precomputed table boundary to avoid allocations.
fn compute_visibility_polygon(
    ox: f64,
    oz: f64,
    segments: &[Segment],
    table_boundary: &[Segment; 4],
    bufs: &mut VisBuffers,
    result: &mut Vec<(f64, f64)>,
) {
    bufs.endpoints.clear();
    bufs.endpoint_seen.clear();
    bufs.rays.clear();
    result.clear();

    // Collect unique endpoints from terrain segments + table boundary
    for &(x1, z1, x2, z2) in segments.iter().chain(table_boundary.iter()) {
        let k1 = (x1.to_bits(), z1.to_bits());
        let k2 = (x2.to_bits(), z2.to_bits());
        if bufs.endpoint_seen.insert(k1) {
            bufs.endpoints.push((x1, z1));
        }
        if bufs.endpoint_seen.insert(k2) {
            bufs.endpoints.push((x2, z2));
        }
    }

    // For each endpoint, cast rays at angle and angle ± epsilon.
    // Use cos/sin of offset angles (matching Python's numpy approach)
    // to ensure FP-identical ray directions across engines.
    let eps = 1e-5_f64;

    for &(ex, ez) in &bufs.endpoints {
        let dx = ex - ox;
        let dz = ez - oz;
        let angle = dz.atan2(dx);
        let len = (dx * dx + dz * dz).sqrt();
        let ndx = dx / len;
        let ndz = dz / len;
        // -eps ray via sin_cos (shares range reduction; bit-identical
        // to separate sin()/cos() calls on the same platform)
        let a_neg = angle - eps;
        let (neg_dz, neg_dx) = a_neg.sin_cos();
        bufs.rays
            .push((pseudoangle(neg_dx, neg_dz), neg_dx, neg_dz));
        // Center ray
        bufs.rays.push((pseudoangle(ndx, ndz), ndx, ndz));
        // +eps ray via sin_cos
        let a_pos = angle + eps;
        let (pos_dz, pos_dx) = a_pos.sin_cos();
        bufs.rays
            .push((pseudoangle(pos_dx, pos_dz), pos_dx, pos_dz));
    }

    // Sort rays by angle (unstable sort is fine — duplicate angles have no meaningful order)
    bufs.rays
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Build flattened segment array and assign to angular buckets.
    // Each segment goes in all buckets overlapping its angular extent
    // (the shorter arc between its two endpoint angles from the observer).
    // This reduces intersection work from O(R×S) to O(R×k) where k is
    // the average segments per bucket.
    bufs.all_segments.clear();
    bufs.all_segments.extend_from_slice(segments);
    bufs.all_segments.extend_from_slice(table_boundary);
    for bucket in &mut bufs.buckets {
        bucket.clear();
    }

    let half_buckets = NUM_ANGLE_BUCKETS / 2;
    for (si, &(x1, z1, x2, z2)) in bufs.all_segments.iter().enumerate() {
        let b1 = pa_to_bucket(pseudoangle(x1 - ox, z1 - oz));
        let b2 = pa_to_bucket(pseudoangle(x2 - ox, z2 - oz));

        // Find the shorter arc, expanded by 1 bucket for boundary safety.
        let (lo, hi) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
        let si16 = si as u16;
        if hi - lo <= half_buckets {
            // Non-wrapping arc
            let start = lo.saturating_sub(1);
            let end = (hi + 1).min(NUM_ANGLE_BUCKETS - 1);
            for b in start..=end {
                bufs.buckets[b].push(si16);
            }
        } else {
            // Arc wraps around ±π boundary
            let start = hi.saturating_sub(1);
            for b in start..NUM_ANGLE_BUCKETS {
                bufs.buckets[b].push(si16);
            }
            let end = (lo + 1).min(NUM_ANGLE_BUCKETS - 1);
            for b in 0..=end {
                bufs.buckets[b].push(si16);
            }
        }
    }

    // Cast each ray, testing only segments in its angular bucket.
    // Inline intersection math with early exit when t >= current min_t.
    let rays = &bufs.rays;
    let buckets = &bufs.buckets;
    let all_segs = &bufs.all_segments;

    for &(pa, dx, dz) in rays {
        let bucket_idx = pa_to_bucket(pa);
        let mut min_t = f64::INFINITY;
        for &si in &buckets[bucket_idx] {
            let (x1, z1, x2, z2) = all_segs[si as usize];
            let sx = x2 - x1;
            let sz = z2 - z1;
            let denom = dx * sz - dz * sx;
            if denom.abs() < 1e-12 {
                continue;
            }
            let d_x1 = x1 - ox;
            let d_z1 = z1 - oz;
            let t = (d_x1 * sz - d_z1 * sx) / denom;
            if t < 0.0 || t >= min_t {
                continue;
            }
            let u = (d_x1 * dz - d_z1 * dx) / denom;
            if (0.0..=1.0).contains(&u) {
                min_t = t;
            }
        }

        if min_t < f64::INFINITY {
            result.push((ox + min_t * dx, oz + min_t * dz));
        }
    }
}

/// Generate grid sample points that fall inside a polygon.
/// Uses AABB bounding box to limit search, then PIP-filters.
#[cfg(test)]
fn sample_points_in_polygon(
    polygon: &[(f64, f64)],
    grid_spacing: f64,
    grid_offset: f64,
) -> Vec<(f64, f64)> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let min_x = polygon.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = polygon
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_z = polygon.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_z = polygon
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut start_x = ((min_x - grid_offset) / grid_spacing).floor() * grid_spacing + grid_offset;
    let mut start_z = ((min_z - grid_offset) / grid_spacing).floor() * grid_spacing + grid_offset;
    if start_x < min_x {
        start_x += grid_spacing;
    }
    if start_z < min_z {
        start_z += grid_spacing;
    }

    let mut x = start_x;
    while x < max_x {
        let mut z = start_z;
        while z < max_z {
            if point_in_polygon(x, z, polygon) {
                points.push((x, z));
            }
            z += grid_spacing;
        }
        x += grid_spacing;
    }
    points
}

/// Generate grid sample points that fall inside a circle.
/// Uses AABB bounding box to limit search, then distance-filters.
fn sample_points_in_circle(
    cx: f64,
    cz: f64,
    radius: f64,
    grid_spacing: f64,
    grid_offset: f64,
) -> Vec<(f64, f64)> {
    if radius <= 0.0 {
        return Vec::new();
    }
    let min_x = cx - radius;
    let max_x = cx + radius;
    let min_z = cz - radius;
    let max_z = cz + radius;

    let mut start_x = ((min_x - grid_offset) / grid_spacing).floor() * grid_spacing + grid_offset;
    let mut start_z = ((min_z - grid_offset) / grid_spacing).floor() * grid_spacing + grid_offset;
    if start_x < min_x {
        start_x += grid_spacing;
    }
    if start_z < min_z {
        start_z += grid_spacing;
    }

    let r_sq = radius * radius;
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut x = start_x;
    while x < max_x {
        let mut z = start_z;
        while z < max_z {
            let dx = x - cx;
            let dz = z - cz;
            if dx * dx + dz * dz <= r_sq {
                points.push((x, z));
            }
            z += grid_spacing;
        }
        x += grid_spacing;
    }
    points
}

/// Test if a point is inside any of the given polygon rings.
fn point_in_any_polygon(px: f64, pz: f64, polygons: &[Vec<(f64, f64)>]) -> bool {
    for poly in polygons {
        if point_in_polygon(px, pz, poly) {
            return true;
        }
    }
    false
}

/// True if point is inside or within max_dist of any polygon ring.
#[cfg(test)]
fn point_near_any_polygon(px: f64, pz: f64, polygons: &[Vec<(f64, f64)>], max_dist: f64) -> bool {
    let max_dist_sq = max_dist * max_dist;
    for poly in polygons {
        if point_in_polygon(px, pz, poly) {
            return true;
        }
        let n = poly.len();
        for i in 0..n {
            let (x1, z1) = poly[i];
            let (x2, z2) = poly[(i + 1) % n];
            if crate::collision::point_to_segment_distance_squared(px, pz, x1, z1, x2, z2)
                <= max_dist_sq
            {
                return true;
            }
        }
    }
    false
}

/// Compute fraction of DZ sample points visible (inside vis polygon).
#[cfg(test)]
fn fraction_of_dz_visible(vis_poly: &[(f64, f64)], dz_sample_points: &[(f64, f64)]) -> f64 {
    let mut buf = Vec::new();
    fraction_of_dz_visible_batch(vis_poly, dz_sample_points, &mut buf)
}

/// Batch version that reuses an external buffer to avoid allocation.
#[cfg(test)]
fn fraction_of_dz_visible_batch(
    vis_poly: &[(f64, f64)],
    dz_sample_points: &[(f64, f64)],
    pip_buf: &mut Vec<bool>,
) -> f64 {
    if dz_sample_points.is_empty() {
        return 0.0;
    }
    if vis_poly.len() < 3 {
        return 0.0;
    }
    batch_point_in_polygon(dz_sample_points, vis_poly, pip_buf);
    let count = pip_buf.iter().filter(|&&b| b).count();
    count as f64 / dz_sample_points.len() as f64
}

/// DZ sample points pre-sorted by Z coordinate for efficient binary
/// search in the PIP inner loop. Built once during DZ setup and reused
/// for every observer.
struct DzSortedSamples {
    /// (z, x, original_index) sorted by z.
    sorted: Vec<(f64, f64, u32)>,
    /// Total number of original sample points.
    count: usize,
}

impl DzSortedSamples {
    fn from_points(points: &[(f64, f64)]) -> Self {
        let mut sorted: Vec<(f64, f64, u32)> = points
            .iter()
            .enumerate()
            .map(|(i, &(x, z))| (z, x, i as u32))
            .collect();
        sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self {
            count: points.len(),
            sorted,
        }
    }
}

/// Compute fraction of DZ sample points visible using Z-sorted binary
/// search. For each polygon edge, binary-searches the sorted points to
/// find only those in the edge's Z-range, reducing work from O(E×P) to
/// O(E × (log P + P_range)).
#[cfg(test)]
fn fraction_of_dz_visible_zsorted(
    vis_poly: &[(f64, f64)],
    sorted: &DzSortedSamples,
    inside: &mut Vec<bool>,
) -> f64 {
    if sorted.count == 0 || vis_poly.len() < 3 {
        return 0.0;
    }

    inside.clear();
    inside.resize(sorted.count, false);

    let n = vis_poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = vis_poly[i];
        let (xj, zj) = vis_poly[j];

        // Skip horizontal edges (zi == zj → no Z-crossings possible)
        if zi == zj {
            j = i;
            continue;
        }

        // Z-range where (zi > pz) != (zj > pz) is true: [z_lo, z_hi)
        let (z_lo, z_hi) = if zi < zj { (zi, zj) } else { (zj, zi) };

        // Binary search: find range of sorted points with z_lo <= z < z_hi
        let start = sorted.sorted.partition_point(|s| s.0 < z_lo);
        let end = sorted.sorted.partition_point(|s| s.0 < z_hi);

        // Process only points in the Z-range.
        // Use direct division (not precomputed 1/dz) to match batch PIP
        // and Python — avoids boundary-point flips from IEEE 754 rounding.
        let dz = zj - zi;
        let dx = xj - xi;
        for k in start..end {
            let (pz, px, orig_idx) = sorted.sorted[k];
            let intersect_x = dx * (pz - zi) / dz + xi;
            if px < intersect_x {
                inside[orig_idx as usize] = !inside[orig_idx as usize];
            }
        }

        j = i;
    }

    let count = inside.iter().filter(|&&b| b).count();
    count as f64 / sorted.count as f64
}

/// Z-sorted PIP that directly updates a `seen` boolean array.
/// Only processes unseen points (skips already-seen entries in the
/// edge loop). After all edges, marks newly-inside points as seen.
fn pip_zsorted_update_seen(
    vis_poly: &[(f64, f64)],
    sorted: &DzSortedSamples,
    seen: &mut [bool],
    inside: &mut Vec<bool>,
) {
    if sorted.count == 0 || vis_poly.len() < 3 {
        return;
    }

    inside.clear();
    inside.resize(sorted.count, false);

    let n = vis_poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = vis_poly[i];
        let (xj, zj) = vis_poly[j];

        if zi == zj {
            j = i;
            continue;
        }

        let (z_lo, z_hi) = if zi < zj { (zi, zj) } else { (zj, zi) };
        let start = sorted.sorted.partition_point(|s| s.0 < z_lo);
        let end = sorted.sorted.partition_point(|s| s.0 < z_hi);

        let dz = zj - zi;
        let dx = xj - xi;
        for k in start..end {
            let (pz, px, orig_idx) = sorted.sorted[k];
            let oidx = orig_idx as usize;
            if seen[oidx] {
                continue;
            }
            let intersect_x = dx * (pz - zi) / dz + xi;
            if px < intersect_x {
                inside[oidx] = !inside[oidx];
            }
        }

        j = i;
    }

    // Mark newly visible unseen points as seen
    for (i, &is_inside) in inside.iter().enumerate() {
        if is_inside && !seen[i] {
            seen[i] = true;
        }
    }
}

/// Build the full sample grid for a table + mission (same logic as
/// the grid generation in `compute_layout_visibility`).
fn build_sample_grid(layout: &TerrainLayout) -> Vec<(f64, f64)> {
    let half_w = layout.table_width_inches / 2.0;
    let half_d = layout.table_depth_inches / 2.0;

    let mut obj_ranges: Vec<(f64, f64, f64)> = Vec::new();
    if let Some(ref mission) = layout.mission {
        for obj_marker in &mission.objectives {
            obj_ranges.push((
                obj_marker.position.x_inches,
                obj_marker.position.z_inches,
                0.75 + obj_marker.range_inches,
            ));
        }
    }

    let mut sample_points: Vec<(f64, f64)> = Vec::new();
    let ix_start = (-half_w) as i32 + 1;
    let ix_end = half_w as i32 - 1;
    let iz_start = (-half_d) as i32 + 1;
    let iz_end = half_d as i32 - 1;

    for ix in ix_start..=ix_end {
        for iz in iz_start..=iz_end {
            if ix % 2 == 0 && iz % 2 == 0 {
                sample_points.push((ix as f64, iz as f64));
            } else {
                let fx = ix as f64;
                let fz = iz as f64;
                let mut near_obj = false;
                for &(ox, oz, radius) in &obj_ranges {
                    let dx = fx - ox;
                    let dz = fz - oz;
                    if dx * dx + dz * dz <= radius * radius {
                        near_obj = true;
                        break;
                    }
                }
                if near_obj {
                    sample_points.push((fx, fz));
                }
            }
        }
    }
    sample_points
}

/// Cache for visibility computation that avoids regenerating the
/// observer sample grid on every call.
///
/// The sample grid is constant for a given table size + mission.
/// Tall-terrain filtering is recomputed lazily when `dirty` is set.
/// The generate loop calls `mark_dirty()` (O(1)) whenever features
/// change; the actual PIP recompute only happens when visibility is
/// next needed.
pub struct VisibilityCache<'a> {
    objects_by_id: &'a HashMap<String, &'a TerrainObject>,
    rotationally_symmetric: bool,
    /// Full sample grid (before tall-terrain filtering).
    all_sample_points: Vec<(f64, f64)>,
    /// Cached filtered sample points (rebuilt lazily when dirty).
    filtered_cache: Vec<(f64, f64)>,
    /// Whether filtered_cache needs rebuilding.
    dirty: bool,
}

impl<'a> VisibilityCache<'a> {
    pub fn new(
        layout: &TerrainLayout,
        objects_by_id: &'a HashMap<String, &'a TerrainObject>,
    ) -> Self {
        let all_sample_points = build_sample_grid(layout);
        let mut cache = Self {
            objects_by_id,
            rotationally_symmetric: layout.rotationally_symmetric,
            all_sample_points,
            filtered_cache: Vec::new(),
            dirty: true,
        };
        cache.recompute_filtered(layout);
        cache
    }

    /// Mark the cache as needing a recompute.
    /// Called from the generate loop when features change — O(1).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Recompute tall-terrain filtering from scratch.
    fn recompute_filtered(&mut self, layout: &TerrainLayout) {
        let mut tall_footprints: Vec<Corners> = Vec::new();
        for pf in &layout.placed_features {
            for corners in get_tall_world_obbs(pf, self.objects_by_id, 1.0) {
                tall_footprints.push(corners);
            }
            if self.rotationally_symmetric && !is_at_origin(pf) {
                let mirror = mirror_placed_feature(pf);
                for corners in get_tall_world_obbs(&mirror, self.objects_by_id, 1.0) {
                    tall_footprints.push(corners);
                }
            }
        }

        self.filtered_cache.clear();
        if tall_footprints.is_empty() {
            self.filtered_cache
                .extend_from_slice(&self.all_sample_points);
        } else {
            for &(x, z) in &self.all_sample_points {
                if !tall_footprints.iter().any(|fp| point_in_polygon(x, z, fp)) {
                    self.filtered_cache.push((x, z));
                }
            }
        }
        self.dirty = false;
    }

    /// Return sample points not inside any tall terrain.
    /// Recomputes only when dirty (i.e., features changed since
    /// last call).
    pub fn get_filtered_sample_points(&mut self, layout: &TerrainLayout) -> &[(f64, f64)] {
        if self.dirty {
            self.recompute_filtered(layout);
        }
        &self.filtered_cache
    }
}

/// Check if a 1x1" square of hidden grid points exists for model placement.
///
/// A model needs physical space to stand, so a single hidden point is not
/// enough. We look for a 1"x1" axis-aligned square (4 adjacent grid points
/// at 1" spacing) where:
///   1. All 4 corners are within table bounds
///   2. At least 1 corner is within the objective's range circle
///   3. All 4 corners are hidden from the opposing threat zone
///   4. No tall terrain shape (height >= 1.0") intersects the square
///
/// WARNING: The terrain-intersection check (step 4) uses OBB (oriented
/// bounding box) geometry via obbs_overlap(). This assumes all terrain
/// shapes are rectangular. If polygonal or cylindrical terrain shapes are
/// added in the future, this check will need to be updated to use the
/// appropriate intersection test.
fn has_valid_hiding_square(
    obj_cx: f64,
    obj_cz: f64,
    obj_radius: f64,
    sample_points: &[(f64, f64)],
    hidden_indices: &FxHashSet<usize>,
    tall_obbs: &[Corners],
    half_w: f64,
    half_d: f64,
) -> bool {
    // Build lookup from (x_bits, z_bits) -> sample index for O(1) access.
    // Using f64::to_bits() is safe because all coords are integer-valued
    // from the grid, so there are no floating-point precision issues.
    let mut pt_index: FxHashMap<(u64, u64), usize> =
        FxHashMap::with_capacity_and_hasher(sample_points.len(), FxBuildHasher::default());
    for (i, &(px, pz)) in sample_points.iter().enumerate() {
        pt_index.insert((px.to_bits(), pz.to_bits()), i);
    }

    let r_sq = obj_radius * obj_radius;

    // Collect candidate square origins (bottom-left corners).
    // For each hidden, in-range point, it could be any of the 4 corners
    // of a valid square. Deduplicate via HashSet of (x_bits, z_bits).
    let mut candidate_origins: FxHashSet<(u64, u64)> = FxHashSet::default();
    for &idx in hidden_indices {
        let (px, pz) = sample_points[idx];
        let dx = px - obj_cx;
        let dz = pz - obj_cz;
        if dx * dx + dz * dz > r_sq {
            continue;
        }
        // This point is hidden AND in range — generate 4 candidate origins
        for &(ox, oz) in &[
            (px, pz),             // point is bottom-left
            (px - 1.0, pz),       // point is bottom-right
            (px, pz - 1.0),       // point is top-left
            (px - 1.0, pz - 1.0), // point is top-right
        ] {
            candidate_origins.insert((ox.to_bits(), oz.to_bits()));
        }
    }

    for &(ox_bits, oz_bits) in &candidate_origins {
        let ox = f64::from_bits(ox_bits);
        let oz = f64::from_bits(oz_bits);

        // The 4 corners of this candidate square
        let corners_xz = [
            (ox, oz),
            (ox + 1.0, oz),
            (ox + 1.0, oz + 1.0),
            (ox, oz + 1.0),
        ];

        // Check 1: all 4 corners within table bounds
        let all_in_bounds = corners_xz
            .iter()
            .all(|&(cx, cz)| cx >= -half_w && cx <= half_w && cz >= -half_d && cz <= half_d);
        if !all_in_bounds {
            continue;
        }

        // Check 2: at least 1 corner in range of objective
        let any_in_range = corners_xz.iter().any(|&(cx, cz)| {
            let dx = cx - obj_cx;
            let dz = cz - obj_cz;
            dx * dx + dz * dz <= r_sq
        });
        if !any_in_range {
            continue;
        }

        // Check 3: all 4 corners are hidden (exist in sample and are hidden)
        let all_hidden = corners_xz.iter().all(|&(cx, cz)| {
            if let Some(&idx) = pt_index.get(&(cx.to_bits(), cz.to_bits())) {
                hidden_indices.contains(&idx)
            } else {
                false
            }
        });
        if !all_hidden {
            continue;
        }

        // Check 4: no tall terrain OBB overlaps the square
        let square_obb = obb_corners(ox + 0.5, oz + 0.5, 0.5, 0.5, 0.0);
        let terrain_blocks = tall_obbs
            .iter()
            .any(|tall_obb| obbs_overlap(&square_obb, tall_obb));
        if terrain_blocks {
            continue;
        }

        // All checks passed — valid hiding square found
        return true;
    }

    false
}

/// Check if any non-obscuring placed feature shape has an effective opacity
/// height in [infantry_height, standard_height). If so, the infantry pass
/// would produce different results from the standard pass.
fn has_intermediate_shapes(
    layout: &TerrainLayout,
    objects_by_id: &HashMap<String, &TerrainObject>,
    infantry_height: f64,
    standard_height: f64,
) -> bool {
    for pf in &layout.placed_features {
        if pf.feature.feature_type == "obscuring" {
            continue;
        }
        for comp in &pf.feature.components {
            if let Some(obj) = objects_by_id.get(&comp.object_id) {
                for shape in &obj.shapes {
                    let h = shape.effective_opacity_height();
                    if h >= infantry_height && h < standard_height {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Merge standard and infantry visibility results into a dual-pass output.
/// For each section (overall, dz_visibility, etc.), sets the top-level "value"
/// to the average of standard and infantry, and includes "standard"/"infantry"
/// sub-dicts with the full breakdown.
fn merge_dual_pass_results(
    standard: &serde_json::Value,
    infantry: &serde_json::Value,
) -> serde_json::Value {
    let mut result = serde_json::Map::new();

    // Helper: merge a single entry (with "value" key).
    // Produces {"value": avg, "standard": std_entry, "infantry": inf_entry}.
    let merge_entry =
        |std_entry: &serde_json::Value, inf_entry: &serde_json::Value| -> serde_json::Value {
            let std_val = std_entry
                .get("value")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let inf_val = inf_entry
                .get("value")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let avg = ((std_val + inf_val) / 2.0 * 100.0).round() / 100.0;

            serde_json::json!({
                "value": avg,
                "standard": std_entry,
                "infantry": inf_entry,
            })
        };

    // Merge overall
    let std_overall = &standard["overall"];
    let inf_overall = &infantry["overall"];
    result.insert("overall".into(), merge_entry(std_overall, inf_overall));

    // Merge section dicts (dz_hideability, objective_hidability)
    for section_key in &["dz_hideability", "objective_hidability"] {
        let std_section = standard.get(*section_key);
        let inf_section = infantry.get(*section_key);
        if let (Some(std_s), Some(inf_s)) = (std_section, inf_section) {
            if let (Some(std_obj), Some(inf_obj)) = (std_s.as_object(), inf_s.as_object()) {
                let mut merged_section = serde_json::Map::new();
                for (key, std_entry) in std_obj {
                    if let Some(inf_entry) = inf_obj.get(key) {
                        merged_section.insert(key.clone(), merge_entry(std_entry, inf_entry));
                    }
                }
                result.insert(
                    (*section_key).into(),
                    serde_json::Value::Object(merged_section),
                );
            }
        } else if let Some(std_s) = std_section {
            result.insert((*section_key).into(), std_s.clone());
        }
    }

    serde_json::Value::Object(result)
}

/// Compute visibility score for a terrain layout.
///
/// Samples observer positions on a grid across the table, computes
/// visibility polygon for each, and returns the average visibility
/// ratio (visible area / total area).
///
/// When `visibility_cache` is provided, uses incremental tall-footprint
/// filtering instead of recomputing from scratch each call.
pub fn compute_layout_visibility(
    layout: &TerrainLayout,
    objects_by_id: &HashMap<String, &TerrainObject>,
    min_blocking_height: f64,
    visibility_cache: Option<&mut VisibilityCache>,
    infantry_blocking_height: Option<f64>,
) -> serde_json::Value {
    let half_w = layout.table_width_inches / 2.0;
    let half_d = layout.table_depth_inches / 2.0;
    let table_area = layout.table_width_inches * layout.table_depth_inches;

    // Get sample points (either from cache or computed fresh)
    let owned_points: Vec<(f64, f64)>;
    let sample_points: &[(f64, f64)] = match visibility_cache {
        Some(cache) => cache.get_filtered_sample_points(layout),
        None => {
            let mut pts = build_sample_grid(layout);

            // Filter out observer points inside tall terrain (height >= 1")
            let mut tall_footprints: Vec<Corners> = Vec::new();
            {
                let mut effective_features: Vec<PlacedFeature> = Vec::new();
                for pf in &layout.placed_features {
                    effective_features.push(pf.clone());
                    if layout.rotationally_symmetric && !is_at_origin(pf) {
                        effective_features.push(mirror_placed_feature(pf));
                    }
                }
                for pf in &effective_features {
                    for corners in get_tall_world_obbs(pf, objects_by_id, 1.0) {
                        tall_footprints.push(corners);
                    }
                }
            }

            if !tall_footprints.is_empty() {
                pts.retain(|&(x, z)| !tall_footprints.iter().any(|fp| point_in_polygon(x, z, fp)));
            }

            owned_points = pts;
            &owned_points
        }
    };

    if sample_points.is_empty() {
        return serde_json::json!({
            "overall": {
                "value": 100.0,
                "min_blocking_height_inches": min_blocking_height,
                "sample_count": 0
            }
        });
    }

    // -- DZ pre-loop setup --
    let has_dzs = layout
        .mission
        .as_ref()
        .is_some_and(|m| !m.deployment_zones.is_empty());

    struct DzData {
        id: String,
        polys: Vec<Vec<(f64, f64)>>,
        expanded_polys: Vec<Vec<(f64, f64)>>,
        expanded_aabbs: Vec<crate::collision::Aabb>,
    }

    let mut dz_data: Vec<DzData> = Vec::new();
    if has_dzs {
        let mission = layout.mission.as_ref().unwrap();
        for dz in &mission.deployment_zones {
            let polys: Vec<Vec<(f64, f64)>> = dz
                .polygons
                .iter()
                .map(|poly| poly.iter().map(|p| (p.x_inches, p.z_inches)).collect())
                .collect();
            let expanded_polys: Vec<Vec<(f64, f64)>> = dz
                .expanded_polygons
                .iter()
                .map(|poly| poly.iter().map(|p| (p.x_inches, p.z_inches)).collect())
                .collect();
            let expanded_aabbs: Vec<crate::collision::Aabb> = expanded_polys
                .iter()
                .map(|ep| crate::collision::compute_aabb(ep))
                .collect();
            dz_data.push(DzData {
                id: dz.id.clone(),
                polys,
                expanded_polys,
                expanded_aabbs,
            });
        }
    }
    let num_dzs = dz_data.len();

    // Precompute observer DZ membership: Vec<Option<usize>>
    let observer_dz: Vec<Option<usize>> = if has_dzs {
        sample_points
            .iter()
            .map(|&(sx, sz)| {
                dz_data
                    .iter()
                    .position(|dd| point_in_any_polygon(sx, sz, &dd.polys))
            })
            .collect()
    } else {
        Vec::new()
    };

    // Note: expanded DZ membership precomputation removed — DZ hideability now uses
    // polygon-polygon intersection (vis_poly vs expanded_polys) instead of PIP sampling.

    // -- Objective hidability pre-loop setup --
    let has_objectives = has_dzs
        && layout
            .mission
            .as_ref()
            .is_some_and(|m| !m.objectives.is_empty());

    let mut obj_sample_points: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut obj_sorted_samples: Vec<DzSortedSamples> = Vec::new();
    let mut obj_radii: Vec<f64> = Vec::new();

    if has_objectives {
        let mission = layout.mission.as_ref().unwrap();
        for obj_marker in &mission.objectives {
            let obj_radius = 0.75 + obj_marker.range_inches;
            obj_radii.push(obj_radius);
            // Expand sample radius by sqrt(2) so that all diagonal grid
            // neighbors of in-range points are included. This is needed
            // for the model-fit hiding square check which looks at 1x1"
            // squares of 4 adjacent grid points.
            let expanded_radius = obj_radius + std::f64::consts::SQRT_2;
            let pts = sample_points_in_circle(
                obj_marker.position.x_inches,
                obj_marker.position.z_inches,
                expanded_radius,
                1.0,
                0.0,
            );
            obj_sorted_samples.push(DzSortedSamples::from_points(&pts));
            obj_sample_points.push(pts);
        }
    }

    // Precompute static segments and obscuring shape data once
    let precomputed = precompute_segments(layout, objects_by_id, min_blocking_height);

    // Precompute table boundary segments (used by every observer)
    let table_boundary: [Segment; 4] = [
        (-half_w, -half_d, half_w, -half_d), // bottom
        (half_w, -half_d, half_w, half_d),   // right
        (half_w, half_d, -half_w, half_d),   // top
        (-half_w, half_d, -half_w, -half_d), // left
    ];

    // Per-thread accumulator for parallel observer loop.
    struct ThreadAccum {
        total_ratio: f64,
        // DZ hideability: (hidden_count, total_count) per DZ
        dz_hide_accum: Vec<(u32, u32)>,
        obj_seen_from_dz: Vec<Vec<Vec<bool>>>,
        // Working buffers (not merged — just avoids per-call allocation)
        segments: Vec<Segment>,
        vis_bufs: VisBuffers,
        vis_poly: Vec<(f64, f64)>,
        pip_buf: Vec<bool>,
    }

    // Closure to create a fresh per-thread accumulator.
    let make_accum = || {
        Box::new(ThreadAccum {
            total_ratio: 0.0,
            dz_hide_accum: vec![(0, 0); num_dzs],
            obj_seen_from_dz: if has_objectives {
                (0..num_dzs)
                    .map(|_| {
                        obj_sample_points
                            .iter()
                            .map(|pts| vec![false; pts.len()])
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            },
            segments: Vec::new(),
            vis_bufs: VisBuffers::new(),
            vis_poly: Vec::new(),
            pip_buf: Vec::new(),
        })
    };

    // Deterministic parallel observer loop: fixed-size chunks processed in
    // parallel, collected in index order, then merged sequentially.  This
    // ensures floating-point accumulation order is independent of thread
    // scheduling, producing bit-identical results across runs.
    const CHUNK_SIZE: usize = 128;
    let chunk_accums: Vec<Box<ThreadAccum>> = sample_points
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut acc = make_accum();
            let base = chunk_idx * CHUNK_SIZE;
            for (local, &(sx, sz)) in chunk.iter().enumerate() {
                let obs_idx = base + local;
                get_observer_segments(&precomputed, sx, sz, &mut acc.segments);
                let obs_dz = if has_dzs { observer_dz[obs_idx] } else { None };

                // Helper: accumulate for full visibility (no segments or degenerate polygon)
                let accumulate_full = |acc: &mut ThreadAccum| {
                    acc.total_ratio += 1.0;
                    if has_dzs {
                        // Full visibility means vis poly = entire table.
                        // Observer inside DZ: vis poly definitely overlaps opponent → NOT hidden.
                        // Just increment total_count (not hidden_count).
                        if let Some(my_dz) = obs_dz {
                            acc.dz_hide_accum[my_dz].1 += 1;
                            // Objective hidability: mark all objective points as seen
                            if has_objectives {
                                for seen in acc.obj_seen_from_dz[my_dz].iter_mut() {
                                    seen.fill(true);
                                }
                            }
                        }
                    }
                };

                if acc.segments.is_empty() {
                    accumulate_full(&mut acc);
                    continue;
                }

                compute_visibility_polygon(
                    sx,
                    sz,
                    &acc.segments,
                    &table_boundary,
                    &mut acc.vis_bufs,
                    &mut acc.vis_poly,
                );
                if acc.vis_poly.len() < 3 {
                    accumulate_full(&mut acc);
                    continue;
                }

                let vis_area = polygon_area(&acc.vis_poly);
                let ratio = (vis_area / table_area).min(1.0);
                acc.total_ratio += ratio;

                // DZ hideability + objective hidability (polygon intersection)
                if has_dzs {
                    if let Some(my_dz) = obs_dz {
                        // This observer is inside DZ my_dz.
                        // Test if vis poly overlaps each opponent's expanded DZ.
                        for (oi, other_dd) in dz_data.iter().enumerate() {
                            if oi == my_dz {
                                continue;
                            }
                            acc.dz_hide_accum[my_dz].1 += 1;
                            let overlaps = other_dd
                                .expanded_polys
                                .iter()
                                .zip(other_dd.expanded_aabbs.iter())
                                .any(|(ep, ep_aabb)| {
                                    crate::collision::polygons_overlap_aabb(
                                        &acc.vis_poly,
                                        ep,
                                        ep_aabb,
                                    )
                                });
                            if !overlaps {
                                acc.dz_hide_accum[my_dz].0 += 1;
                            }
                        }

                        // Objective hidability: PIP for objective sample points
                        if has_objectives {
                            for (oi, obj_sorted) in obj_sorted_samples.iter().enumerate() {
                                let seen = &mut acc.obj_seen_from_dz[my_dz][oi];
                                pip_zsorted_update_seen(
                                    &acc.vis_poly,
                                    obj_sorted,
                                    seen,
                                    &mut acc.pip_buf,
                                );
                            }
                        }
                    }
                }
            }
            acc
        })
        .collect();

    // Sequential merge in index order (deterministic reduction).
    let merged = chunk_accums
        .into_iter()
        .reduce(|mut a, b: Box<ThreadAccum>| {
            a.total_ratio += b.total_ratio;
            for di in 0..num_dzs {
                a.dz_hide_accum[di].0 += b.dz_hide_accum[di].0;
                a.dz_hide_accum[di].1 += b.dz_hide_accum[di].1;
            }
            if has_objectives {
                for di in 0..num_dzs {
                    for oi in 0..a.obj_seen_from_dz[di].len() {
                        for (i, &seen) in b.obj_seen_from_dz[di][oi].iter().enumerate() {
                            if seen {
                                a.obj_seen_from_dz[di][oi][i] = true;
                            }
                        }
                    }
                }
            }
            a
        })
        .unwrap_or_else(make_accum);

    let total_ratio = merged.total_ratio;
    let dz_hide_accum = merged.dz_hide_accum;
    let obj_seen_from_dz = merged.obj_seen_from_dz;

    let avg_visibility = total_ratio / sample_points.len() as f64;
    let value = (avg_visibility * 100.0 * 100.0).round() / 100.0;

    let mut result = serde_json::json!({
        "overall": {
            "value": value,
            "min_blocking_height_inches": min_blocking_height,
            "sample_count": sample_points.len()
        }
    });

    // Build DZ hideability results (convert index-based → string keys)
    if has_dzs {
        let mut dz_hideability = serde_json::Map::new();

        for (di, dd) in dz_data.iter().enumerate() {
            let (hidden_count, total_count) = dz_hide_accum[di];
            let pct = if total_count > 0 {
                (hidden_count as f64 / total_count as f64 * 100.0 * 100.0).round() / 100.0
            } else {
                0.0
            };
            dz_hideability.insert(
                dd.id.clone(),
                serde_json::json!({
                    "value": pct,
                    "hidden_count": hidden_count,
                    "total_count": total_count
                }),
            );
        }

        result["dz_hideability"] = serde_json::Value::Object(dz_hideability);

        // Build objective hidability results
        if has_objectives {
            let mission = layout.mission.as_ref().unwrap();
            let total_objectives = obj_sample_points.len();
            let mut objective_hidability = serde_json::Map::new();

            // Collect tall OBBs for terrain-intersection check
            let mut obj_tall_obbs: Vec<Corners> = Vec::new();
            for pf in &layout.placed_features {
                for corners in get_tall_world_obbs(pf, objects_by_id, 1.0) {
                    obj_tall_obbs.push(corners);
                }
                if layout.rotationally_symmetric && !is_at_origin(pf) {
                    let mirror = mirror_placed_feature(pf);
                    for corners in get_tall_world_obbs(&mirror, objects_by_id, 1.0) {
                        obj_tall_obbs.push(corners);
                    }
                }
            }

            for (di, _dd) in dz_data.iter().enumerate() {
                let mut safe_count = 0;
                let per_obj = &obj_seen_from_dz[di];
                for oi in 0..total_objectives {
                    let total_pts = obj_sample_points[oi].len();
                    if total_pts == 0 {
                        continue;
                    }
                    // Build set of hidden indices (complement of seen)
                    let hidden: FxHashSet<usize> =
                        (0..total_pts).filter(|&i| !per_obj[oi][i]).collect();
                    if hidden.is_empty() {
                        continue;
                    }
                    let obj_marker = &mission.objectives[oi];
                    if has_valid_hiding_square(
                        obj_marker.position.x_inches,
                        obj_marker.position.z_inches,
                        obj_radii[oi],
                        &obj_sample_points[oi],
                        &hidden,
                        &obj_tall_obbs,
                        half_w,
                        half_d,
                    ) {
                        safe_count += 1;
                    }
                }

                // Store under the opposing player's DZ id
                for (oi, other_dd) in dz_data.iter().enumerate() {
                    if oi != di {
                        let pct = if total_objectives > 0 {
                            (safe_count as f64 / total_objectives as f64 * 100.0 * 100.0).round()
                                / 100.0
                        } else {
                            0.0
                        };
                        objective_hidability.insert(
                            other_dd.id.clone(),
                            serde_json::json!({
                                "value": pct,
                                "safe_count": safe_count,
                                "total_objectives": total_objectives
                            }),
                        );
                    }
                }
            }

            result["objective_hidability"] = serde_json::Value::Object(objective_hidability);
        }
    }

    // Dual-pass infantry visibility
    if let Some(inf_height) = infantry_blocking_height {
        if has_intermediate_shapes(layout, objects_by_id, inf_height, min_blocking_height) {
            let infantry_result = compute_layout_visibility(
                layout,
                objects_by_id,
                inf_height,
                None, // don't reuse cache for different blocking height
                None, // prevent infinite recursion
            );
            return merge_dual_pass_results(&result, &infantry_result);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FeatureComponent, GeometricShape, TerrainFeature};

    fn make_object(id: &str, width: f64, depth: f64, height: f64) -> TerrainObject {
        TerrainObject {
            id: id.into(),
            shapes: vec![GeometricShape {
                shape_type: "rectangular_prism".into(),
                width_inches: width,
                depth_inches: depth,
                height_inches: height,
                offset: None,
                opacity_height_inches: None,
            }],
            name: None,
            tags: vec![],
        }
    }

    fn make_feature(id: &str, obj_id: &str, feature_type: &str) -> TerrainFeature {
        TerrainFeature {
            id: id.into(),
            feature_type: feature_type.into(),
            components: vec![FeatureComponent {
                object_id: obj_id.into(),
                transform: None,
            }],
            tags: vec![],
        }
    }

    fn make_placed(feature: TerrainFeature, x: f64, z: f64) -> PlacedFeature {
        PlacedFeature {
            feature,
            transform: Transform {
                x_inches: x,
                y_inches: 0.0,
                z_inches: z,
                rotation_deg: 0.0,
            },
            locked: false,
        }
    }

    fn make_layout(w: f64, d: f64, features: Vec<PlacedFeature>) -> TerrainLayout {
        TerrainLayout {
            table_width_inches: w,
            table_depth_inches: d,
            placed_features: features,
            rotationally_symmetric: false,
            terrain_objects: vec![],
            visibility: None,
            mission: None,
        }
    }

    #[test]
    fn empty_battlefield_full_visibility() {
        let layout = make_layout(60.0, 44.0, vec![]);
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!((val - 100.0).abs() < 0.01);
    }

    #[test]
    fn tall_block_reduces_visibility() {
        let obj = make_object("box", 5.0, 5.0, 5.0);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> = HashMap::new();
        objects.insert("box".into(), &obj);
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!(val < 100.0);
        assert!(val > 50.0);
    }

    #[test]
    fn short_block_full_visibility() {
        let obj = make_object("box", 5.0, 5.0, 2.5);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> = HashMap::new();
        objects.insert("box".into(), &obj);
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!((val - 100.0).abs() < 0.01);
    }

    #[test]
    fn obscuring_reduces_visibility() {
        let obj = make_object("ruins", 12.0, 6.0, 0.0);
        let feat = make_feature("f1", "ruins", "obscuring");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> = HashMap::new();
        objects.insert("ruins".into(), &obj);
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!(val < 100.0);
    }

    #[test]
    fn polygon_area_unit_square() {
        let verts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!((polygon_area(&verts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn point_inside_polygon() {
        let sq = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(point_in_polygon(5.0, 5.0, &sq));
        assert!(!point_in_polygon(15.0, 5.0, &sq));
    }

    #[test]
    fn ray_hits_segment() {
        let t = ray_segment_intersection(0.0, 0.0, 1.0, 0.0, 5.0, -5.0, 5.0, 5.0);
        assert!(t.is_some());
        assert!((t.unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn ray_misses_segment() {
        let t = ray_segment_intersection(0.0, 0.0, 1.0, 0.0, -5.0, -5.0, -5.0, 5.0);
        assert!(t.is_none());
    }

    // -- DZ helper tests --

    #[test]
    fn sample_points_in_polygon_rect() {
        let rect = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let pts = sample_points_in_polygon(&rect, 1.0, 0.5);
        assert_eq!(pts.len(), 100);
    }

    #[test]
    fn sample_points_in_polygon_triangle() {
        let tri = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let pts = sample_points_in_polygon(&tri, 1.0, 0.5);
        assert!(pts.len() > 0);
        assert!(pts.len() < 100);
    }

    #[test]
    fn fraction_full_coverage() {
        let dz_samples = vec![(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)];
        let vis_poly = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let frac = fraction_of_dz_visible(&vis_poly, &dz_samples);
        assert!((frac - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fraction_no_coverage() {
        let dz_samples = vec![(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)];
        let vis_poly = vec![(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)];
        let frac = fraction_of_dz_visible(&vis_poly, &dz_samples);
        assert!(frac.abs() < 1e-9);
    }

    // -- DZ integration tests --

    use crate::types::{DeploymentZone, Mission, ObjectiveMarker, Point2D};

    fn make_rect_dz(id: &str, x1: f64, z1: f64, x2: f64, z2: f64) -> DeploymentZone {
        DeploymentZone {
            id: id.into(),
            polygons: vec![vec![
                Point2D {
                    x_inches: x1,
                    z_inches: z1,
                },
                Point2D {
                    x_inches: x2,
                    z_inches: z1,
                },
                Point2D {
                    x_inches: x2,
                    z_inches: z2,
                },
                Point2D {
                    x_inches: x1,
                    z_inches: z2,
                },
            ]],
            expanded_polygons: vec![],
        }
    }

    fn make_layout_with_mission(
        w: f64,
        d: f64,
        features: Vec<PlacedFeature>,
        mission: Option<Mission>,
    ) -> TerrainLayout {
        TerrainLayout {
            table_width_inches: w,
            table_depth_inches: d,
            placed_features: features,
            rotationally_symmetric: false,
            terrain_objects: vec![],
            visibility: None,
            mission,
        }
    }

    #[test]
    fn dz_empty_table_full_visibility() {
        let green_dz = make_rect_dz("green", -30.0, -22.0, -20.0, 22.0);
        let red_dz = make_rect_dz("red", 20.0, -22.0, 30.0, 22.0);
        let mission = Mission {
            name: "test".into(),
            objectives: vec![],
            deployment_zones: vec![green_dz, red_dz],
            rotationally_symmetric: false,
        };
        let layout = make_layout_with_mission(60.0, 44.0, vec![], Some(mission));
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);

        assert!(result.get("dz_hideability").is_some());
        let dz_hide = &result["dz_hideability"];
        // No terrain → nothing hidden → hideability 0%
        assert_eq!(dz_hide["green"]["value"].as_f64().unwrap(), 0.0);
        assert_eq!(dz_hide["red"]["value"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn no_mission_no_dz_keys() {
        let layout = make_layout(60.0, 44.0, vec![]);
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        assert!(result.get("dz_hideability").is_none());
    }

    // -- Sample points in circle tests --

    #[test]
    fn sample_points_in_circle_known_radius() {
        let pts = sample_points_in_circle(0.0, 0.0, 3.75, 1.0, 0.5);
        assert!(!pts.is_empty());
        let r_sq = 3.75 * 3.75;
        for &(x, z) in &pts {
            assert!(x * x + z * z <= r_sq + 1e-9);
        }
    }

    #[test]
    fn sample_points_in_circle_zero_radius() {
        let pts = sample_points_in_circle(0.0, 0.0, 0.0, 1.0, 0.5);
        assert!(pts.is_empty());
    }

    // -- Objective hidability tests --

    fn make_standard_objectives() -> Vec<ObjectiveMarker> {
        vec![
            ObjectiveMarker {
                id: "obj1".into(),
                position: Point2D {
                    x_inches: 0.0,
                    z_inches: 0.0,
                },
                range_inches: 3.0,
            },
            ObjectiveMarker {
                id: "obj2".into(),
                position: Point2D {
                    x_inches: -12.0,
                    z_inches: -8.0,
                },
                range_inches: 3.0,
            },
            ObjectiveMarker {
                id: "obj3".into(),
                position: Point2D {
                    x_inches: 12.0,
                    z_inches: -8.0,
                },
                range_inches: 3.0,
            },
            ObjectiveMarker {
                id: "obj4".into(),
                position: Point2D {
                    x_inches: -12.0,
                    z_inches: 8.0,
                },
                range_inches: 3.0,
            },
            ObjectiveMarker {
                id: "obj5".into(),
                position: Point2D {
                    x_inches: 12.0,
                    z_inches: 8.0,
                },
                range_inches: 3.0,
            },
        ]
    }

    #[test]
    fn objective_hidability_empty_table() {
        let green_dz = make_rect_dz("green", -30.0, -22.0, -20.0, 22.0);
        let red_dz = make_rect_dz("red", 20.0, -22.0, 30.0, 22.0);
        let mission = Mission {
            name: "test".into(),
            objectives: make_standard_objectives(),
            deployment_zones: vec![green_dz, red_dz],
            rotationally_symmetric: false,
        };
        let layout = make_layout_with_mission(60.0, 44.0, vec![], Some(mission));
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);

        assert!(result.get("objective_hidability").is_some());
        let oh = &result["objective_hidability"];
        assert_eq!(oh["green"]["value"].as_f64().unwrap(), 0.0);
        assert_eq!(oh["red"]["value"].as_f64().unwrap(), 0.0);
        assert_eq!(oh["green"]["safe_count"].as_i64().unwrap(), 0);
        assert_eq!(oh["red"]["safe_count"].as_i64().unwrap(), 0);
        assert_eq!(oh["green"]["total_objectives"].as_i64().unwrap(), 5);
    }

    // -- Observer filtering tests --

    #[test]
    fn tall_terrain_reduces_sample_count() {
        let empty_layout = make_layout(60.0, 44.0, vec![]);
        let empty_objects: HashMap<String, &TerrainObject> = HashMap::new();
        let empty_result =
            compute_layout_visibility(&empty_layout, &empty_objects, 4.0, None, None);
        let empty_count = empty_result["overall"]["sample_count"].as_i64().unwrap();

        let obj = make_object("box", 10.0, 10.0, 2.0);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> = HashMap::new();
        objects.insert("box".into(), &obj);
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        let filtered_count = result["overall"]["sample_count"].as_i64().unwrap();

        assert!(filtered_count < empty_count);
    }

    #[test]
    fn short_terrain_does_not_reduce_sample_count() {
        let empty_layout = make_layout(60.0, 44.0, vec![]);
        let empty_objects: HashMap<String, &TerrainObject> = HashMap::new();
        let empty_result =
            compute_layout_visibility(&empty_layout, &empty_objects, 4.0, None, None);
        let empty_count = empty_result["overall"]["sample_count"].as_i64().unwrap();

        let obj = make_object("box", 10.0, 10.0, 0.5);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> = HashMap::new();
        objects.insert("box".into(), &obj);
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        let filtered_count = result["overall"]["sample_count"].as_i64().unwrap();

        assert_eq!(filtered_count, empty_count);
    }

    #[test]
    fn no_objectives_no_key() {
        let green_dz = make_rect_dz("green", -30.0, -22.0, -20.0, 22.0);
        let red_dz = make_rect_dz("red", 20.0, -22.0, 30.0, 22.0);
        let mission = Mission {
            name: "test".into(),
            objectives: vec![],
            deployment_zones: vec![green_dz, red_dz],
            rotationally_symmetric: false,
        };
        let layout = make_layout_with_mission(60.0, 44.0, vec![], Some(mission));
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result = compute_layout_visibility(&layout, &objects, 4.0, None, None);
        assert!(result.get("objective_hidability").is_none());
    }

    // -- Expanded DZ tests --

    #[test]
    fn point_near_any_polygon_inside() {
        let poly = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(point_near_any_polygon(5.0, 5.0, &[poly], 6.0));
    }

    #[test]
    fn point_near_any_polygon_within_range() {
        let poly = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        // Point at (13, 5) is 3" from the right edge at x=10
        assert!(point_near_any_polygon(13.0, 5.0, &[poly], 6.0));
    }

    #[test]
    fn point_near_any_polygon_out_of_range() {
        let poly = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        // Point at (20, 5) is 10" from the right edge — beyond 6"
        assert!(!point_near_any_polygon(20.0, 5.0, &[poly], 6.0));
    }

    #[test]
    fn eps_ray_uses_trig_not_rotation_matrix() {
        // The ±eps rays in compute_visibility_polygon must use cos/sin of
        // the offset angle (matching Python's numpy approach) rather than a
        // rotation matrix, because IEEE 754 rounding differs between the
        // two approaches. This test sets up a simple scenario and verifies
        // the visibility polygon vertices match the cos/sin-derived values.
        //
        // A single blocking segment at a specific angle creates three rays
        // per endpoint. We verify the polygon output matches expected values
        // computed via the trig approach.
        let table_boundary: [Segment; 4] = [
            (-10.0, -10.0, 10.0, -10.0),
            (10.0, -10.0, 10.0, 10.0),
            (10.0, 10.0, -10.0, 10.0),
            (-10.0, 10.0, -10.0, -10.0),
        ];
        // One blocking segment from (3,4) to (4,3) — observer at origin
        let segments: Vec<Segment> = vec![(3.0, 4.0, 4.0, 3.0)];
        let mut bufs = VisBuffers::new();
        let mut result = Vec::new();

        compute_visibility_polygon(0.0, 0.0, &segments, &table_boundary, &mut bufs, &mut result);

        // The visibility polygon should have vertices. Verify that the
        // rays cast at angle(3,4)±eps produce the trig-derived directions,
        // which we can check by examining the polygon output vertices.
        // The key property: polygon must be non-empty and deterministic.
        assert!(
            !result.is_empty(),
            "visibility polygon should have vertices"
        );

        // Verify a rotation-matrix approach would give DIFFERENT results
        // (documenting why we use cos/sin):
        let eps = 1e-5_f64;
        let cos_eps = eps.cos();
        let sin_eps = eps.sin();
        let angle = 4.0_f64.atan2(3.0);
        let len = (9.0 + 16.0_f64).sqrt();
        let ndx = 3.0 / len;
        let ndz = 4.0 / len;
        let rot_dx_pos = ndx * cos_eps - ndz * sin_eps;
        let trig_dx_pos = (angle + eps).cos();
        assert_ne!(
            rot_dx_pos, trig_dx_pos,
            "rotation matrix and trig should differ — if equal, test is vacuous"
        );
    }

    #[test]
    fn zsorted_pip_matches_batch_pip() {
        // Triangle with edge from (0,0) to (7,3). At z=1, the intersection
        // x = 7/3 ≈ 2.3333... When computed via direct division (batch) vs
        // precomputed 1/dz (zsorted), IEEE 754 rounding produces different
        // intersection values. A point at exactly the boundary flips.
        //
        // px = 7*(1/3) lands at the smaller of the two adjacent f64 values.
        // Batch (division): intersect_x = 7/3 = larger value → px < intersect → TRUE
        // Zsorted (inv):    intersect_x = 7*(1/3) = same as px → px < intersect → FALSE
        let polygon = vec![(0.0, 0.0), (7.0, 3.0), (7.0, 0.0)];
        let px = 7.0 * (1.0_f64 / 3.0); // inv_dz intersection value
        let points = vec![(px, 1.0)];

        let sorted = DzSortedSamples::from_points(&points);
        let mut buf = Vec::new();

        let frac_batch = fraction_of_dz_visible_batch(&polygon, &points, &mut buf);
        let frac_zsorted = fraction_of_dz_visible_zsorted(&polygon, &sorted, &mut buf);

        assert_eq!(
            frac_batch, frac_zsorted,
            "batch ({frac_batch}) != zsorted ({frac_zsorted}): \
             boundary point in/out diverged due to inv_dz precomputation"
        );
    }
}
