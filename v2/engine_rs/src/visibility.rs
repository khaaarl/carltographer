//! Visibility score computation for terrain layouts.
//!
//! Measures what percentage of the battlefield has clear line of sight
//! by sampling observer positions on a grid and computing visibility
//! polygons via angular sweep.

use std::collections::{HashMap, HashSet};

use crate::collision::{
    compose_transform, get_tall_world_obbs, is_at_origin,
    mirror_placed_feature, obb_corners, point_to_segment_distance_squared,
    Corners,
};
use crate::types::{
    DeploymentZone, PlacedFeature, TerrainLayout, TerrainObject, Transform,
};

/// Expansion distance for DZ threat zone (inches).
const DZ_EXPANSION_INCHES: f64 = 6.0;

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
fn batch_point_in_polygon(
    points: &[(f64, f64)],
    polygon: &[(f64, f64)],
    inside: &mut Vec<bool>,
) {
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
                let intersect_x =
                    (xj - xi) * (pz - zi) / (zj - zi) + xi;
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

/// Precomputed data for one obscuring feature shape.
struct ObscuringShape {
    /// Corner vertices for point-in-polygon test (observer inside check).
    corners: Vec<(f64, f64)>,
    /// Per-edge data: (x1, z1, x2, z2, midpoint_x, midpoint_z, normal_x, normal_z).
    edges: Vec<(f64, f64, f64, f64, f64, f64, f64, f64)>,
    /// Index of the obscuring feature this shape belongs to (for grouping).
    feature_idx: usize,
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

    for (feat_idx, pf) in effective_features.iter().enumerate() {
        let is_obscuring = pf.feature.feature_type == "obscuring";

        if is_obscuring {
            let all_corners = get_footprint_corners(pf, objects_by_id);
            for corners in &all_corners {
                let n = corners.len();
                let verts: Vec<(f64, f64)> = corners.to_vec();

                // Precompute center
                let cx: f64 =
                    corners.iter().map(|c| c.0).sum::<f64>() / n as f64;
                let cz: f64 =
                    corners.iter().map(|c| c.1).sum::<f64>() / n as f64;

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
                    feature_idx: feat_idx,
                });
            }
        } else {
            // Regular obstacle: all 4 edges block regardless of observer
            for comp in &pf.feature.components {
                let obj = match objects_by_id.get(&comp.object_id) {
                    Some(o) => o,
                    None => continue,
                };
                let comp_t =
                    comp.transform.as_ref().unwrap_or(&default_t);
                for shape in &obj.shapes {
                    if shape.height_inches < min_blocking_height {
                        continue;
                    }
                    let shape_t =
                        shape.offset.as_ref().unwrap_or(&default_t);
                    let world = compose_transform(
                        &compose_transform(shape_t, comp_t),
                        &pf.transform,
                    );
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

    // Group shapes by feature_idx to check observer-inside per feature
    // Track which features have the observer inside
    let mut skip_feature: Option<usize> = None;
    let mut last_feat_idx: Option<usize> = None;

    for shape in &precomputed.obscuring_shapes {
        // Reset skip when we move to a new feature
        if last_feat_idx != Some(shape.feature_idx) {
            skip_feature = None;
            last_feat_idx = Some(shape.feature_idx);
        }

        // If already determined observer is inside this feature, skip all its shapes
        if skip_feature == Some(shape.feature_idx) {
            continue;
        }

        // Check if observer is inside this shape
        if point_in_polygon(observer_x, observer_z, &shape.corners) {
            skip_feature = Some(shape.feature_idx);
            // Remove any segments already added from earlier shapes of same feature
            // (This is rare — most features have 1 shape)
            continue;
        }

        // Add back-facing edges
        for &(x1, z1, x2, z2, mx, mz, nx, nz) in &shape.edges {
            let dot_observer =
                (observer_x - mx) * nx + (observer_z - mz) * nz;
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
const BUCKET_WIDTH: f64 = 2.0 * std::f64::consts::PI / NUM_ANGLE_BUCKETS as f64;

/// Map an angle in [-π, π] to a bucket index in [0, NUM_ANGLE_BUCKETS).
#[inline]
fn angle_to_bucket(angle: f64) -> usize {
    let normalized = angle + std::f64::consts::PI; // [0, 2π)
    let b = (normalized / BUCKET_WIDTH) as usize;
    if b >= NUM_ANGLE_BUCKETS {
        NUM_ANGLE_BUCKETS - 1
    } else {
        b
    }
}

/// Reusable buffers for compute_visibility_polygon to avoid per-call allocations.
struct VisBuffers {
    endpoints: Vec<(f64, f64)>,
    endpoint_seen: HashSet<(u64, u64)>,
    rays: Vec<(f64, f64, f64)>,
    polygon: Vec<(f64, f64, f64)>,
    /// Angular buckets: each bucket holds indices into `all_segments`.
    buckets: [Vec<u16>; NUM_ANGLE_BUCKETS],
    /// Flattened terrain + table boundary segments for bucket indexing.
    all_segments: Vec<Segment>,
}

impl VisBuffers {
    fn new() -> Self {
        Self {
            endpoints: Vec::with_capacity(256),
            endpoint_seen: HashSet::with_capacity(256),
            rays: Vec::with_capacity(768),
            polygon: Vec::with_capacity(768),
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
    bufs.polygon.clear();
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

    // For each endpoint, cast rays at angle and angle +/- epsilon
    let eps = 1e-5_f64;

    for &(ex, ez) in &bufs.endpoints {
        let dx = ex - ox;
        let dz = ez - oz;
        let angle = dz.atan2(dx);
        for &a in &[angle - eps, angle, angle + eps] {
            bufs.rays.push((a, a.cos(), a.sin()));
        }
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
        let a1 = (z1 - oz).atan2(x1 - ox);
        let a2 = (z2 - oz).atan2(x2 - ox);
        let b1 = angle_to_bucket(a1);
        let b2 = angle_to_bucket(a2);

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
    let rays = &bufs.rays;
    let buckets = &bufs.buckets;
    let all_segs = &bufs.all_segments;
    let polygon = &mut bufs.polygon;

    for &(angle, dx, dz) in rays {
        let bucket_idx = angle_to_bucket(angle);
        let mut min_t = f64::INFINITY;
        for &si in &buckets[bucket_idx] {
            let (x1, z1, x2, z2) = all_segs[si as usize];
            if let Some(t) =
                ray_segment_intersection(ox, oz, dx, dz, x1, z1, x2, z2)
            {
                if t < min_t {
                    min_t = t;
                }
            }
        }

        if min_t < f64::INFINITY {
            polygon.push((angle, ox + min_t * dx, oz + min_t * dz));
        }
    }

    // Extract just the (x, z) coordinates
    result.extend(bufs.polygon.iter().map(|p| (p.1, p.2)));
}

/// Generate grid sample points that fall inside a polygon.
/// Uses AABB bounding box to limit search, then PIP-filters.
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
    let mut start_x = ((min_x - grid_offset) / grid_spacing).floor()
        * grid_spacing
        + grid_offset;
    let mut start_z = ((min_z - grid_offset) / grid_spacing).floor()
        * grid_spacing
        + grid_offset;
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

/// Generate grid sample points across all polygon rings of a deployment zone.
fn sample_points_in_dz(
    dz: &DeploymentZone,
    grid_spacing: f64,
    grid_offset: f64,
) -> Vec<(f64, f64)> {
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let mut points: Vec<(f64, f64)> = Vec::new();
    for poly in &dz.polygons {
        let poly_tuples: Vec<(f64, f64)> =
            poly.iter().map(|p| (p.x_inches, p.z_inches)).collect();
        for pt in sample_points_in_polygon(
            &poly_tuples,
            grid_spacing,
            grid_offset,
        ) {
            let key = (pt.0.to_bits(), pt.1.to_bits());
            if seen.insert(key) {
                points.push(pt);
            }
        }
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

    let mut start_x = ((min_x - grid_offset) / grid_spacing).floor()
        * grid_spacing
        + grid_offset;
    let mut start_z = ((min_z - grid_offset) / grid_spacing).floor()
        * grid_spacing
        + grid_offset;
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
fn point_in_any_polygon(
    px: f64,
    pz: f64,
    polygons: &[Vec<(f64, f64)>],
) -> bool {
    for poly in polygons {
        if point_in_polygon(px, pz, poly) {
            return true;
        }
    }
    false
}

/// True if point is inside or within max_dist of any polygon ring.
fn point_near_any_polygon(
    px: f64,
    pz: f64,
    polygons: &[Vec<(f64, f64)>],
    max_dist: f64,
) -> bool {
    let max_dist_sq = max_dist * max_dist;
    for poly in polygons {
        if point_in_polygon(px, pz, poly) {
            return true;
        }
        let n = poly.len();
        for i in 0..n {
            let (x1, z1) = poly[i];
            let (x2, z2) = poly[(i + 1) % n];
            if point_to_segment_distance_squared(px, pz, x1, z1, x2, z2)
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
fn fraction_of_dz_visible(
    vis_poly: &[(f64, f64)],
    dz_sample_points: &[(f64, f64)],
) -> f64 {
    let mut buf = Vec::new();
    fraction_of_dz_visible_batch(vis_poly, dz_sample_points, &mut buf)
}

/// Batch version that reuses an external buffer to avoid allocation.
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
    fn recompute_filtered(
        &mut self,
        layout: &TerrainLayout,
    ) {
        let mut tall_footprints: Vec<Corners> = Vec::new();
        for pf in &layout.placed_features {
            for corners in
                get_tall_world_obbs(pf, self.objects_by_id, 1.0)
            {
                tall_footprints.push(corners);
            }
            if self.rotationally_symmetric && !is_at_origin(pf) {
                let mirror = mirror_placed_feature(pf);
                for corners in get_tall_world_obbs(
                    &mirror,
                    self.objects_by_id,
                    1.0,
                ) {
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
                if !tall_footprints
                    .iter()
                    .any(|fp| point_in_polygon(x, z, fp))
                {
                    self.filtered_cache.push((x, z));
                }
            }
        }
        self.dirty = false;
    }

    /// Return sample points not inside any tall terrain.
    /// Recomputes only when dirty (i.e., features changed since
    /// last call).
    pub fn get_filtered_sample_points(
        &mut self,
        layout: &TerrainLayout,
    ) -> &[(f64, f64)] {
        if self.dirty {
            self.recompute_filtered(layout);
        }
        &self.filtered_cache
    }
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
) -> serde_json::Value {
    let half_w = layout.table_width_inches / 2.0;
    let half_d = layout.table_depth_inches / 2.0;
    let table_area =
        layout.table_width_inches * layout.table_depth_inches;

    // Get sample points (either from cache or computed fresh)
    let owned_points: Vec<(f64, f64)>;
    let sample_points: &[(f64, f64)] = match visibility_cache {
        Some(cache) => cache.get_filtered_sample_points(layout),
        None => {
            let mut pts = build_sample_grid(layout);

            // Filter out observer points inside tall terrain (height >= 1")
            let mut tall_footprints: Vec<Corners> = Vec::new();
            {
                let mut effective_features: Vec<PlacedFeature> =
                    Vec::new();
                for pf in &layout.placed_features {
                    effective_features.push(pf.clone());
                    if layout.rotationally_symmetric
                        && !is_at_origin(pf)
                    {
                        effective_features
                            .push(mirror_placed_feature(pf));
                    }
                }
                for pf in &effective_features {
                    for corners in
                        get_tall_world_obbs(pf, objects_by_id, 1.0)
                    {
                        tall_footprints.push(corners);
                    }
                }
            }

            if !tall_footprints.is_empty() {
                pts.retain(|&(x, z)| {
                    !tall_footprints
                        .iter()
                        .any(|fp| point_in_polygon(x, z, fp))
                });
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
        .map_or(false, |m| !m.deployment_zones.is_empty());

    struct DzData {
        id: String,
        polys: Vec<Vec<(f64, f64)>>,
        samples: Vec<(f64, f64)>,
    }

    let mut dz_data: Vec<DzData> = Vec::new();
    if has_dzs {
        let mission = layout.mission.as_ref().unwrap();
        for dz in &mission.deployment_zones {
            let polys: Vec<Vec<(f64, f64)>> = dz
                .polygons
                .iter()
                .map(|poly| {
                    poly.iter()
                        .map(|p| (p.x_inches, p.z_inches))
                        .collect()
                })
                .collect();
            let samples = sample_points_in_dz(dz, 1.0, 0.0);
            dz_data.push(DzData {
                id: dz.id.clone(),
                polys,
                samples,
            });
        }
    }
    let num_dzs = dz_data.len();

    // Index-based accumulators (no HashMap<String, ...> in hot loop)
    let mut dz_vis_accum: Vec<(f64, i64)> = vec![(0.0, 0); num_dzs];
    // Cross-DZ seen: flat Vec indexed by target_idx * num_dzs + source_idx
    // Each entry is Vec<bool> with len = target DZ's sample count
    let mut dz_cross_seen: Vec<Vec<bool>> = (0..num_dzs * num_dzs)
        .map(|flat| {
            let target = flat / num_dzs;
            let source = flat % num_dzs;
            if target != source {
                vec![false; dz_data[target].samples.len()]
            } else {
                Vec::new() // diagonal: unused
            }
        })
        .collect();
    let mut dz_cross_obs_count: Vec<i64> =
        vec![0; num_dzs * num_dzs];

    // Precompute observer DZ membership: Vec<Option<usize>>
    let observer_dz: Vec<Option<usize>> = if has_dzs {
        sample_points
            .iter()
            .map(|&(sx, sz)| {
                dz_data
                    .iter()
                    .position(|dd| {
                        point_in_any_polygon(sx, sz, &dd.polys)
                    })
            })
            .collect()
    } else {
        Vec::new()
    };

    // Precompute expanded DZ membership (within DZ_EXPANSION_INCHES of any DZ edge)
    let observer_expanded_dz: Vec<Option<usize>> = if has_dzs {
        sample_points
            .iter()
            .enumerate()
            .map(|(i, &(sx, sz))| {
                // If already inside a DZ, reuse that result
                if let Some(dz_idx) = observer_dz[i] {
                    return Some(dz_idx);
                }
                // Otherwise check if near any DZ
                dz_data
                    .iter()
                    .position(|dd| {
                        point_near_any_polygon(
                            sx,
                            sz,
                            &dd.polys,
                            DZ_EXPANSION_INCHES,
                        )
                    })
            })
            .collect()
    } else {
        Vec::new()
    };

    // -- Objective hidability pre-loop setup --
    let has_objectives = has_dzs
        && layout
            .mission
            .as_ref()
            .map_or(false, |m| !m.objectives.is_empty());

    let mut obj_sample_points: Vec<Vec<(f64, f64)>> = Vec::new();
    // Per-DZ, per-objective, Vec<bool> of seen sample indices
    let mut obj_seen_from_dz: Vec<Vec<Vec<bool>>> = Vec::new();

    if has_objectives {
        let mission = layout.mission.as_ref().unwrap();
        for obj_marker in &mission.objectives {
            let obj_radius = 0.75 + obj_marker.range_inches;
            let pts = sample_points_in_circle(
                obj_marker.position.x_inches,
                obj_marker.position.z_inches,
                obj_radius,
                1.0,
                0.0,
            );
            obj_sample_points.push(pts);
        }

        for _ in 0..num_dzs {
            let per_obj: Vec<Vec<bool>> = obj_sample_points
                .iter()
                .map(|pts| vec![false; pts.len()])
                .collect();
            obj_seen_from_dz.push(per_obj);
        }
    }

    // Helper: handle DZ accumulation when visibility is full
    // (no segments or < 3 vis_poly vertices). Uses precomputed obs_dz + obs_expanded_dz.
    let accumulate_dz_full =
        |obs_dz: Option<usize>,
         obs_expanded_dz: Option<usize>,
         num_dzs: usize,
         dz_vis_accum: &mut [(f64, i64)],
         dz_cross_seen: &mut [Vec<bool>],
         dz_cross_obs_count: &mut [i64],
         obj_seen_from_dz: &mut [Vec<Vec<bool>>],
         has_objectives: bool| {
            for di in 0..num_dzs {
                // dz_visibility: uses original boundary
                if obs_dz != Some(di) {
                    dz_vis_accum[di].0 += 1.0;
                    dz_vis_accum[di].1 += 1;
                }
                // cross-DZ + objective: uses expanded boundary
                if obs_expanded_dz == Some(di) {
                    for oi in 0..num_dzs {
                        if oi != di {
                            let pair = oi * num_dzs + di;
                            dz_cross_obs_count[pair] += 1;
                            dz_cross_seen[pair].fill(true);
                        }
                    }
                    if has_objectives {
                        for seen in
                            obj_seen_from_dz[di].iter_mut()
                        {
                            seen.fill(true);
                        }
                    }
                }
            }
        };

    // Precompute static segments and obscuring shape data once
    let precomputed =
        precompute_segments(layout, objects_by_id, min_blocking_height);

    // Precompute table boundary segments (used by every observer)
    let table_boundary: [Segment; 4] = [
        (-half_w, -half_d, half_w, -half_d), // bottom
        (half_w, -half_d, half_w, half_d),   // right
        (half_w, half_d, -half_w, half_d),   // top
        (-half_w, half_d, -half_w, -half_d), // left
    ];

    let mut total_ratio = 0.0;
    let mut segments: Vec<Segment> = Vec::new();
    let mut vis_bufs = VisBuffers::new();
    let mut vis_poly: Vec<(f64, f64)> = Vec::new();
    // Reusable buffers for batch PIP
    let mut pip_buf: Vec<bool> = Vec::new();
    let mut unseen_indices: Vec<usize> = Vec::new();
    let mut unseen_points: Vec<(f64, f64)> = Vec::new();

    for (obs_idx, &(sx, sz)) in sample_points.iter().enumerate()
    {
        get_observer_segments(&precomputed, sx, sz, &mut segments);
        let obs_dz = if has_dzs {
            observer_dz[obs_idx]
        } else {
            None
        };
        let obs_expanded_dz = if has_dzs {
            observer_expanded_dz[obs_idx]
        } else {
            None
        };

        if segments.is_empty() {
            total_ratio += 1.0;
            if has_dzs {
                accumulate_dz_full(
                    obs_dz,
                    obs_expanded_dz,
                    num_dzs,
                    &mut dz_vis_accum,
                    &mut dz_cross_seen,
                    &mut dz_cross_obs_count,
                    &mut obj_seen_from_dz,
                    has_objectives,
                );
            }
            continue;
        }

        compute_visibility_polygon(
            sx, sz, &segments, &table_boundary, &mut vis_bufs,
            &mut vis_poly,
        );
        if vis_poly.len() < 3 {
            total_ratio += 1.0;
            if has_dzs {
                accumulate_dz_full(
                    obs_dz,
                    obs_expanded_dz,
                    num_dzs,
                    &mut dz_vis_accum,
                    &mut dz_cross_seen,
                    &mut dz_cross_obs_count,
                    &mut obj_seen_from_dz,
                    has_objectives,
                );
            }
            continue;
        }

        let vis_area = polygon_area(&vis_poly);
        let ratio = (vis_area / table_area).min(1.0);
        total_ratio += ratio;

        // DZ visibility accumulation (index-based + batch PIP)
        if has_dzs {
            // Step 1: dz_visibility — uses original DZ boundary
            for (di, dd) in dz_data.iter().enumerate() {
                if obs_dz != Some(di) {
                    let frac = fraction_of_dz_visible_batch(
                        &vis_poly,
                        &dd.samples,
                        &mut pip_buf,
                    );
                    dz_vis_accum[di].0 += frac;
                    dz_vis_accum[di].1 += 1;
                }
            }

            // Step 2: cross-DZ + objective — uses expanded DZ boundary
            if let Some(my_dz) = obs_expanded_dz {
                // Cross-DZ seen tracking: batch PIP unseen samples
                for (oi, other_dd) in
                    dz_data.iter().enumerate()
                {
                    if oi == my_dz {
                        continue;
                    }
                    let pair =
                        oi * num_dzs + my_dz;
                    dz_cross_obs_count[pair] += 1;
                    let seen =
                        &mut dz_cross_seen[pair];
                    unseen_indices.clear();
                    unseen_points.clear();
                    for (i, &pt) in
                        other_dd.samples.iter().enumerate()
                    {
                        if !seen[i] {
                            unseen_indices.push(i);
                            unseen_points.push(pt);
                        }
                    }
                    if !unseen_points.is_empty() {
                        batch_point_in_polygon(
                            &unseen_points,
                            &vis_poly,
                            &mut pip_buf,
                        );
                        for (k, &idx) in
                            unseen_indices.iter().enumerate()
                        {
                            if pip_buf[k] {
                                seen[idx] = true;
                            }
                        }
                    }
                }
                // Objective hidability: batch PIP unseen samples
                if has_objectives {
                    for (oi, obj_pts) in
                        obj_sample_points.iter().enumerate()
                    {
                        let seen =
                            &mut obj_seen_from_dz[my_dz][oi];
                        unseen_indices.clear();
                        unseen_points.clear();
                        for (i, &pt) in
                            obj_pts.iter().enumerate()
                        {
                            if !seen[i] {
                                unseen_indices.push(i);
                                unseen_points.push(pt);
                            }
                        }
                        if !unseen_points.is_empty() {
                            batch_point_in_polygon(
                                &unseen_points,
                                &vis_poly,
                                &mut pip_buf,
                            );
                            for (k, &idx) in unseen_indices
                                .iter()
                                .enumerate()
                            {
                                if pip_buf[k] {
                                    seen[idx] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let avg_visibility = total_ratio / sample_points.len() as f64;
    let value =
        (avg_visibility * 100.0 * 100.0).round() / 100.0;

    let mut result = serde_json::json!({
        "overall": {
            "value": value,
            "min_blocking_height_inches": min_blocking_height,
            "sample_count": sample_points.len()
        }
    });

    // Build DZ visibility results (convert index-based → string keys)
    if has_dzs {
        let mut dz_visibility = serde_json::Map::new();
        let mut dz_to_dz_visibility = serde_json::Map::new();

        for (di, dd) in dz_data.iter().enumerate() {
            let (total_frac, obs_count) = dz_vis_accum[di];
            let avg = if obs_count > 0 {
                total_frac / obs_count as f64
            } else {
                0.0
            };
            dz_visibility.insert(
                dd.id.clone(),
                serde_json::json!({
                    "value": (avg * 100.0 * 100.0).round() / 100.0,
                    "dz_sample_count": dd.samples.len(),
                    "observer_count": obs_count
                }),
            );
        }

        for target in 0..num_dzs {
            for source in 0..num_dzs {
                if target == source {
                    continue;
                }
                let pair = target * num_dzs + source;
                let seen = &dz_cross_seen[pair];
                let target_count =
                    dz_data[target].samples.len();
                let seen_count =
                    seen.iter().filter(|&&b| b).count();
                let hidden_count = target_count - seen_count;
                let hidden_pct = if target_count > 0 {
                    (hidden_count as f64
                        / target_count as f64
                        * 100.0
                        * 100.0)
                        .round()
                        / 100.0
                } else {
                    0.0
                };
                let key = format!(
                    "{}_from_{}",
                    dz_data[target].id, dz_data[source].id
                );
                dz_to_dz_visibility.insert(
                    key,
                    serde_json::json!({
                        "value": hidden_pct,
                        "hidden_count": hidden_count,
                        "target_sample_count": target_count,
                        "observer_count": dz_cross_obs_count[pair]
                    }),
                );
            }
        }

        result["dz_visibility"] =
            serde_json::Value::Object(dz_visibility);
        result["dz_to_dz_visibility"] =
            serde_json::Value::Object(dz_to_dz_visibility);

        // Build objective hidability results
        if has_objectives {
            let total_objectives = obj_sample_points.len();
            let mut objective_hidability = serde_json::Map::new();

            for (di, _dd) in dz_data.iter().enumerate() {
                let mut safe_count = 0;
                let per_obj = &obj_seen_from_dz[di];
                for oi in 0..total_objectives {
                    let total_pts = obj_sample_points[oi].len();
                    if total_pts == 0 {
                        continue;
                    }
                    let seen_count = per_obj[oi]
                        .iter()
                        .filter(|&&b| b)
                        .count();
                    if seen_count < total_pts {
                        safe_count += 1;
                    }
                }

                // Store under the opposing player's DZ id
                for (oi, other_dd) in
                    dz_data.iter().enumerate()
                {
                    if oi != di {
                        let pct = if total_objectives > 0 {
                            (safe_count as f64
                                / total_objectives as f64
                                * 100.0
                                * 100.0)
                                .round()
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

            result["objective_hidability"] =
                serde_json::Value::Object(objective_hidability);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        FeatureComponent, GeometricShape, TerrainFeature,
    };

    fn make_object(
        id: &str,
        width: f64,
        depth: f64,
        height: f64,
    ) -> TerrainObject {
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

    fn make_feature(
        id: &str,
        obj_id: &str,
        feature_type: &str,
    ) -> TerrainFeature {
        TerrainFeature {
            id: id.into(),
            feature_type: feature_type.into(),
            components: vec![FeatureComponent {
                object_id: obj_id.into(),
                transform: None,
            }],
        }
    }

    fn make_placed(
        feature: TerrainFeature,
        x: f64,
        z: f64,
    ) -> PlacedFeature {
        PlacedFeature {
            feature,
            transform: Transform {
                x_inches: x,
                y_inches: 0.0,
                z_inches: z,
                rotation_deg: 0.0,
            },
        }
    }

    fn make_layout(
        w: f64,
        d: f64,
        features: Vec<PlacedFeature>,
    ) -> TerrainLayout {
        TerrainLayout {
            table_width_inches: w,
            table_depth_inches: d,
            placed_features: features,
            rotationally_symmetric: false,
            visibility: None,
            mission: None,
        }
    }

    #[test]
    fn empty_battlefield_full_visibility() {
        let layout = make_layout(60.0, 44.0, vec![]);
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!((val - 100.0).abs() < 0.01);
    }

    #[test]
    fn tall_block_reduces_visibility() {
        let obj = make_object("box", 5.0, 5.0, 5.0);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        objects.insert("box".into(), &obj);
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
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
        let mut objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        objects.insert("box".into(), &obj);
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!((val - 100.0).abs() < 0.01);
    }

    #[test]
    fn obscuring_reduces_visibility() {
        let obj = make_object("ruins", 12.0, 6.0, 0.0);
        let feat = make_feature("f1", "ruins", "obscuring");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        objects.insert("ruins".into(), &obj);
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        let val = result["overall"]["value"].as_f64().unwrap();
        assert!(val < 100.0);
    }

    #[test]
    fn polygon_area_unit_square() {
        let verts = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ];
        assert!((polygon_area(&verts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn point_inside_polygon() {
        let sq = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];
        assert!(point_in_polygon(5.0, 5.0, &sq));
        assert!(!point_in_polygon(15.0, 5.0, &sq));
    }

    #[test]
    fn ray_hits_segment() {
        let t = ray_segment_intersection(
            0.0, 0.0, 1.0, 0.0, 5.0, -5.0, 5.0, 5.0,
        );
        assert!(t.is_some());
        assert!((t.unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn ray_misses_segment() {
        let t = ray_segment_intersection(
            0.0, 0.0, 1.0, 0.0, -5.0, -5.0, -5.0, 5.0,
        );
        assert!(t.is_none());
    }

    // -- DZ helper tests --

    #[test]
    fn sample_points_in_polygon_rect() {
        let rect =
            vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
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
        let dz_samples =
            vec![(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)];
        let vis_poly =
            vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let frac = fraction_of_dz_visible(&vis_poly, &dz_samples);
        assert!((frac - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fraction_no_coverage() {
        let dz_samples =
            vec![(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)];
        let vis_poly = vec![
            (20.0, 20.0),
            (30.0, 20.0),
            (30.0, 30.0),
            (20.0, 30.0),
        ];
        let frac = fraction_of_dz_visible(&vis_poly, &dz_samples);
        assert!(frac.abs() < 1e-9);
    }

    // -- DZ integration tests --

    use crate::types::{
        DeploymentZone, Mission, ObjectiveMarker, Point2D,
    };

    fn make_rect_dz(
        id: &str,
        x1: f64,
        z1: f64,
        x2: f64,
        z2: f64,
    ) -> DeploymentZone {
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
        let layout =
            make_layout_with_mission(60.0, 44.0, vec![], Some(mission));
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);

        assert!(result.get("dz_visibility").is_some());
        let dz_vis = &result["dz_visibility"];
        assert_eq!(
            dz_vis["green"]["value"].as_f64().unwrap(),
            100.0
        );
        assert_eq!(
            dz_vis["red"]["value"].as_f64().unwrap(),
            100.0
        );
    }

    #[test]
    fn dz_to_dz_empty_table() {
        let green_dz = make_rect_dz("green", -30.0, -22.0, -20.0, 22.0);
        let red_dz = make_rect_dz("red", 20.0, -22.0, 30.0, 22.0);
        let mission = Mission {
            name: "test".into(),
            objectives: vec![],
            deployment_zones: vec![green_dz, red_dz],
            rotationally_symmetric: false,
        };
        let layout =
            make_layout_with_mission(60.0, 44.0, vec![], Some(mission));
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);

        let cross = &result["dz_to_dz_visibility"];
        // No terrain -> nothing hidden -> 0%
        assert_eq!(
            cross["red_from_green"]["value"].as_f64().unwrap(),
            0.0
        );
        assert_eq!(
            cross["green_from_red"]["value"].as_f64().unwrap(),
            0.0
        );
        assert_eq!(
            cross["red_from_green"]["hidden_count"]
                .as_i64()
                .unwrap(),
            0
        );
        assert_eq!(
            cross["green_from_red"]["hidden_count"]
                .as_i64()
                .unwrap(),
            0
        );
    }

    #[test]
    fn no_mission_no_dz_keys() {
        let layout = make_layout(60.0, 44.0, vec![]);
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        assert!(result.get("dz_visibility").is_none());
        assert!(result.get("dz_to_dz_visibility").is_none());
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
        let green_dz =
            make_rect_dz("green", -30.0, -22.0, -20.0, 22.0);
        let red_dz = make_rect_dz("red", 20.0, -22.0, 30.0, 22.0);
        let mission = Mission {
            name: "test".into(),
            objectives: make_standard_objectives(),
            deployment_zones: vec![green_dz, red_dz],
            rotationally_symmetric: false,
        };
        let layout = make_layout_with_mission(
            60.0,
            44.0,
            vec![],
            Some(mission),
        );
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);

        assert!(result.get("objective_hidability").is_some());
        let oh = &result["objective_hidability"];
        assert_eq!(oh["green"]["value"].as_f64().unwrap(), 0.0);
        assert_eq!(oh["red"]["value"].as_f64().unwrap(), 0.0);
        assert_eq!(oh["green"]["safe_count"].as_i64().unwrap(), 0);
        assert_eq!(oh["red"]["safe_count"].as_i64().unwrap(), 0);
        assert_eq!(
            oh["green"]["total_objectives"].as_i64().unwrap(),
            5
        );
    }

    // -- Observer filtering tests --

    #[test]
    fn tall_terrain_reduces_sample_count() {
        let empty_layout = make_layout(60.0, 44.0, vec![]);
        let empty_objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        let empty_result = compute_layout_visibility(
            &empty_layout,
            &empty_objects,
            4.0,
            None,
        );
        let empty_count =
            empty_result["overall"]["sample_count"].as_i64().unwrap();

        let obj = make_object("box", 10.0, 10.0, 2.0);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        objects.insert("box".into(), &obj);
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        let filtered_count =
            result["overall"]["sample_count"].as_i64().unwrap();

        assert!(filtered_count < empty_count);
    }

    #[test]
    fn short_terrain_does_not_reduce_sample_count() {
        let empty_layout = make_layout(60.0, 44.0, vec![]);
        let empty_objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        let empty_result = compute_layout_visibility(
            &empty_layout,
            &empty_objects,
            4.0,
            None,
        );
        let empty_count =
            empty_result["overall"]["sample_count"].as_i64().unwrap();

        let obj = make_object("box", 10.0, 10.0, 0.5);
        let feat = make_feature("f1", "box", "obstacle");
        let pf = make_placed(feat, 0.0, 0.0);
        let layout = make_layout(60.0, 44.0, vec![pf]);
        let mut objects: HashMap<String, &TerrainObject> =
            HashMap::new();
        objects.insert("box".into(), &obj);
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        let filtered_count =
            result["overall"]["sample_count"].as_i64().unwrap();

        assert_eq!(filtered_count, empty_count);
    }

    #[test]
    fn no_objectives_no_key() {
        let green_dz =
            make_rect_dz("green", -30.0, -22.0, -20.0, 22.0);
        let red_dz = make_rect_dz("red", 20.0, -22.0, 30.0, 22.0);
        let mission = Mission {
            name: "test".into(),
            objectives: vec![],
            deployment_zones: vec![green_dz, red_dz],
            rotationally_symmetric: false,
        };
        let layout = make_layout_with_mission(
            60.0,
            44.0,
            vec![],
            Some(mission),
        );
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 4.0, None);
        assert!(result.get("objective_hidability").is_none());
    }

    // -- Expanded DZ tests --

    #[test]
    fn point_near_any_polygon_inside() {
        let poly = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];
        assert!(point_near_any_polygon(5.0, 5.0, &[poly], 6.0));
    }

    #[test]
    fn point_near_any_polygon_within_range() {
        let poly = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];
        // Point at (13, 5) is 3" from the right edge at x=10
        assert!(point_near_any_polygon(13.0, 5.0, &[poly], 6.0));
    }

    #[test]
    fn point_near_any_polygon_out_of_range() {
        let poly = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];
        // Point at (20, 5) is 10" from the right edge — beyond 6"
        assert!(!point_near_any_polygon(20.0, 5.0, &[poly], 6.0));
    }
}
