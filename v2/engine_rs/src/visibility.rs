//! Visibility score computation for terrain layouts.
//!
//! Measures what percentage of the battlefield has clear line of sight
//! by sampling observer positions on a grid and computing visibility
//! polygons via angular sweep.

use std::collections::{HashMap, HashSet};

use crate::collision::{
    compose_transform, get_tall_world_obbs, is_at_origin,
    mirror_placed_feature, obb_corners, Corners,
};
use crate::types::{
    DeploymentZone, PlacedFeature, TerrainLayout, TerrainObject, Transform,
};

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

/// Convert Corners (fixed-size array) to Vec of tuples for polygon ops.
fn corners_to_vec(corners: &Corners) -> Vec<(f64, f64)> {
    corners.to_vec()
}

/// Extract line segments that block LoS from the given observer position.
fn extract_blocking_segments(
    layout: &TerrainLayout,
    objects_by_id: &HashMap<String, &TerrainObject>,
    observer_x: f64,
    observer_z: f64,
    min_blocking_height: f64,
) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
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
            // Get all shape footprints regardless of height
            let all_corners = get_footprint_corners(pf, objects_by_id);
            if all_corners.is_empty() {
                continue;
            }

            // Check if observer is inside any shape of this feature
            let mut observer_inside = false;
            for corners in &all_corners {
                let verts = corners_to_vec(corners);
                if point_in_polygon(observer_x, observer_z, &verts) {
                    observer_inside = true;
                    break;
                }
            }

            if observer_inside {
                continue; // Can see out from inside
            }

            // Observer outside: only back-facing edges block
            for corners in &all_corners {
                let n = corners.len();
                // Compute polygon center
                let cx: f64 =
                    corners.iter().map(|c| c.0).sum::<f64>() / n as f64;
                let cz: f64 =
                    corners.iter().map(|c| c.1).sum::<f64>() / n as f64;

                for i in 0..n {
                    let j = (i + 1) % n;
                    let (x1, z1) = corners[i];
                    let (x2, z2) = corners[j];

                    // Edge midpoint
                    let mx = (x1 + x2) / 2.0;
                    let mz = (z1 + z2) / 2.0;

                    // Outward normal
                    let ex = x2 - x1;
                    let ez = z2 - z1;
                    // Normal candidate: (ez, -ex)
                    let (mut nx, mut nz) = (ez, -ex);
                    // If center is on the same side as normal, flip
                    let dot_center = (cx - mx) * nx + (cz - mz) * nz;
                    if dot_center > 0.0 {
                        nx = -nx;
                        nz = -nz;
                    }

                    // Back-facing: outward normal points away from observer
                    let dot_observer =
                        (observer_x - mx) * nx + (observer_z - mz) * nz;
                    if dot_observer < 0.0 {
                        segments.push((x1, z1, x2, z2));
                    }
                }
            }
        } else {
            // Regular obstacle: only shapes with height >= min_blocking_height
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
                    // All 4 edges block
                    for i in 0..4 {
                        let j = (i + 1) % 4;
                        segments.push((
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

    segments
}

/// Get world-space OBB corners for all shapes in a feature (ignoring height).
fn get_footprint_corners(
    placed: &PlacedFeature,
    objects_by_id: &HashMap<String, &TerrainObject>,
) -> Vec<Corners> {
    // Reuse the existing get_world_obbs logic
    crate::collision::get_world_obbs(placed, objects_by_id)
}

/// Compute visibility polygon from observer via angular sweep.
fn compute_visibility_polygon(
    ox: f64,
    oz: f64,
    segments: &[Segment],
    table_half_w: f64,
    table_half_d: f64,
) -> Vec<(f64, f64)> {
    // Add table boundary segments
    let mut all_segments = segments.to_vec();
    let tw = table_half_w;
    let td = table_half_d;
    all_segments.push((-tw, -td, tw, -td)); // bottom
    all_segments.push((tw, -td, tw, td)); // right
    all_segments.push((tw, td, -tw, td)); // top
    all_segments.push((-tw, td, -tw, -td)); // left

    // Collect unique endpoints
    let mut endpoints: Vec<(f64, f64)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &(x1, z1, x2, z2) in &all_segments {
        // Use bit representation for exact dedup
        let k1 = (x1.to_bits(), z1.to_bits());
        let k2 = (x2.to_bits(), z2.to_bits());
        if seen.insert(k1) {
            endpoints.push((x1, z1));
        }
        if seen.insert(k2) {
            endpoints.push((x2, z2));
        }
    }

    // For each endpoint, cast rays at angle and angle +/- epsilon
    let eps = 1e-5_f64;
    let mut rays: Vec<(f64, f64, f64)> = Vec::new(); // (angle, dx, dz)

    for &(ex, ez) in &endpoints {
        let dx = ex - ox;
        let dz = ez - oz;
        let angle = dz.atan2(dx);
        for &a in &[angle - eps, angle, angle + eps] {
            rays.push((a, a.cos(), a.sin()));
        }
    }

    // Sort rays by angle
    rays.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Cast each ray, find nearest intersection
    let mut polygon: Vec<(f64, f64, f64)> = Vec::new(); // (angle, hit_x, hit_z)

    for &(angle, dx, dz) in &rays {
        let mut min_t = f64::INFINITY;
        for &(x1, z1, x2, z2) in &all_segments {
            if let Some(t) =
                ray_segment_intersection(ox, oz, dx, dz, x1, z1, x2, z2)
            {
                if t < min_t {
                    min_t = t;
                }
            }
        }

        if min_t < f64::INFINITY {
            let hit_x = ox + min_t * dx;
            let hit_z = oz + min_t * dz;
            polygon.push((angle, hit_x, hit_z));
        }
    }

    // Extract just the (x, z) coordinates
    polygon.iter().map(|p| (p.1, p.2)).collect()
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

/// Compute fraction of DZ sample points visible (inside vis polygon).
fn fraction_of_dz_visible(
    vis_poly: &[(f64, f64)],
    dz_sample_points: &[(f64, f64)],
) -> f64 {
    if dz_sample_points.is_empty() {
        return 0.0;
    }
    if vis_poly.len() < 3 {
        return 0.0;
    }
    let count = dz_sample_points
        .iter()
        .filter(|&&(px, pz)| point_in_polygon(px, pz, vis_poly))
        .count();
    count as f64 / dz_sample_points.len() as f64
}

/// Compute visibility score for a terrain layout.
///
/// Samples observer positions on a grid across the table, computes
/// visibility polygon for each, and returns the average visibility
/// ratio (visible area / total area).
pub fn compute_layout_visibility(
    layout: &TerrainLayout,
    objects_by_id: &HashMap<String, &TerrainObject>,
    min_blocking_height: f64,
) -> serde_json::Value {
    let half_w = layout.table_width_inches / 2.0;
    let half_d = layout.table_depth_inches / 2.0;
    let table_area =
        layout.table_width_inches * layout.table_depth_inches;

    // Build objective ranges for proximity check
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

    // Generate grid sample points: integer coords, skip edges, 2" spacing
    // except near objectives (1" spacing)
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
                // Check proximity to any objective
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

    // Filter out observer points inside tall terrain (height >= 1")
    let mut tall_footprints: Vec<Vec<(f64, f64)>> = Vec::new();
    {
        let mut effective_features: Vec<PlacedFeature> = Vec::new();
        for pf in &layout.placed_features {
            effective_features.push(pf.clone());
            if layout.rotationally_symmetric && !is_at_origin(pf) {
                effective_features
                    .push(mirror_placed_feature(pf));
            }
        }
        for pf in &effective_features {
            for corners in
                get_tall_world_obbs(pf, objects_by_id, 1.0)
            {
                tall_footprints.push(corners_to_vec(&corners));
            }
        }
    }

    if !tall_footprints.is_empty() {
        sample_points.retain(|&(x, z)| {
            !tall_footprints
                .iter()
                .any(|fp| point_in_polygon(x, z, fp))
        });
    }

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

    // (dz_id, polygon_tuples, sample_points)
    struct DzData {
        id: String,
        polys: Vec<Vec<(f64, f64)>>,
        samples: Vec<(f64, f64)>,
    }

    let mut dz_data: Vec<DzData> = Vec::new();
    // {dz_id: [total_fraction, observer_count]}
    let mut dz_vis_accum: HashMap<String, (f64, i64)> = HashMap::new();
    // Cross-DZ: track which target sample indices have been seen by ANY observer
    let mut dz_cross_seen: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut dz_cross_obs_count: HashMap<String, i64> = HashMap::new();

    if has_dzs {
        let mission = layout.mission.as_ref().unwrap();
        let dzs = &mission.deployment_zones;

        for dz in dzs {
            let polys: Vec<Vec<(f64, f64)>> = dz
                .polygons
                .iter()
                .map(|poly| {
                    poly.iter()
                        .map(|p| (p.x_inches, p.z_inches))
                        .collect()
                })
                .collect();
            let samples =
                sample_points_in_dz(dz, 1.0, 0.0);
            dz_vis_accum.insert(dz.id.clone(), (0.0, 0));
            for other_dz in dzs {
                if other_dz.id != dz.id {
                    let key =
                        format!("{}_from_{}", dz.id, other_dz.id);
                    dz_cross_seen.insert(key.clone(), HashSet::new());
                    dz_cross_obs_count.insert(key, 0);
                }
            }
            dz_data.push(DzData {
                id: dz.id.clone(),
                polys,
                samples,
            });
        }
    }

    // -- Objective hidability pre-loop setup --
    let has_objectives = has_dzs
        && layout
            .mission
            .as_ref()
            .map_or(false, |m| !m.objectives.is_empty());

    // Per-objective sample points
    let mut obj_sample_points: Vec<Vec<(f64, f64)>> = Vec::new();
    // {dz_id: {obj_index: HashSet of seen sample indices}}
    let mut obj_seen_from_dz: HashMap<String, Vec<HashSet<usize>>> =
        HashMap::new();

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

        for dz in &mission.deployment_zones {
            let mut per_obj: Vec<HashSet<usize>> = Vec::new();
            for _ in 0..mission.objectives.len() {
                per_obj.push(HashSet::new());
            }
            obj_seen_from_dz.insert(dz.id.clone(), per_obj);
        }
    }

    // Helper closure to handle DZ accumulation for a given observer
    // when visibility is full (no segments or < 3 vis_poly vertices)
    let accumulate_dz_full =
        |dz_data: &[DzData],
         dz_vis_accum: &mut HashMap<String, (f64, i64)>,
         dz_cross_seen: &mut HashMap<String, HashSet<usize>>,
         dz_cross_obs_count: &mut HashMap<String, i64>,
         obj_seen_from_dz: &mut HashMap<String, Vec<HashSet<usize>>>,
         obj_sample_points: &[Vec<(f64, f64)>],
         has_objectives: bool,
         sx: f64,
         sz: f64| {
            for dd in dz_data {
                let observer_in_dz =
                    point_in_any_polygon(sx, sz, &dd.polys);
                if !observer_in_dz {
                    if let Some(acc) =
                        dz_vis_accum.get_mut(&dd.id)
                    {
                        acc.0 += 1.0;
                        acc.1 += 1;
                    }
                } else {
                    // Full visibility: all target samples are seen
                    for other_dd in dz_data {
                        if other_dd.id != dd.id {
                            let key = format!(
                                "{}_from_{}",
                                other_dd.id, dd.id
                            );
                            if let Some(count) =
                                dz_cross_obs_count.get_mut(&key)
                            {
                                *count += 1;
                            }
                            if let Some(seen) =
                                dz_cross_seen.get_mut(&key)
                            {
                                for i in 0..other_dd.samples.len() {
                                    seen.insert(i);
                                }
                            }
                        }
                    }
                    // Full vis: all objective samples seen from this DZ
                    if has_objectives {
                        if let Some(per_obj) =
                            obj_seen_from_dz.get_mut(&dd.id)
                        {
                            for (oi, obj_pts) in
                                obj_sample_points.iter().enumerate()
                            {
                                for i in 0..obj_pts.len() {
                                    per_obj[oi].insert(i);
                                }
                            }
                        }
                    }
                }
            }
        };

    let mut total_ratio = 0.0;

    for &(sx, sz) in &sample_points {
        let segments = extract_blocking_segments(
            layout,
            objects_by_id,
            sx,
            sz,
            min_blocking_height,
        );

        if segments.is_empty() {
            total_ratio += 1.0;
            if has_dzs {
                accumulate_dz_full(
                    &dz_data,
                    &mut dz_vis_accum,
                    &mut dz_cross_seen,
                    &mut dz_cross_obs_count,
                    &mut obj_seen_from_dz,
                    &obj_sample_points,
                    has_objectives,
                    sx,
                    sz,
                );
            }
            continue;
        }

        let vis_poly = compute_visibility_polygon(
            sx, sz, &segments, half_w, half_d,
        );
        if vis_poly.len() < 3 {
            total_ratio += 1.0;
            if has_dzs {
                accumulate_dz_full(
                    &dz_data,
                    &mut dz_vis_accum,
                    &mut dz_cross_seen,
                    &mut dz_cross_obs_count,
                    &mut obj_seen_from_dz,
                    &obj_sample_points,
                    has_objectives,
                    sx,
                    sz,
                );
            }
            continue;
        }

        let vis_area = polygon_area(&vis_poly);
        let ratio = (vis_area / table_area).min(1.0);
        total_ratio += ratio;

        // DZ visibility accumulation
        if has_dzs {
            for dd in &dz_data {
                let observer_in_dz =
                    point_in_any_polygon(sx, sz, &dd.polys);
                if !observer_in_dz {
                    let frac = fraction_of_dz_visible(
                        &vis_poly,
                        &dd.samples,
                    );
                    if let Some(acc) =
                        dz_vis_accum.get_mut(&dd.id)
                    {
                        acc.0 += frac;
                        acc.1 += 1;
                    }
                } else {
                    // Observer inside this DZ: mark visible target samples
                    for other_dd in &dz_data {
                        if other_dd.id != dd.id {
                            let key = format!(
                                "{}_from_{}",
                                other_dd.id, dd.id
                            );
                            if let Some(count) =
                                dz_cross_obs_count.get_mut(&key)
                            {
                                *count += 1;
                            }
                            if let Some(seen) =
                                dz_cross_seen.get_mut(&key)
                            {
                                for (i, &(px, pz)) in
                                    other_dd.samples.iter().enumerate()
                                {
                                    if !seen.contains(&i) {
                                        if point_in_polygon(
                                            px, pz, &vis_poly,
                                        ) {
                                            seen.insert(i);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Objective hidability: mark seen objective samples
                    if has_objectives {
                        if let Some(per_obj) =
                            obj_seen_from_dz.get_mut(&dd.id)
                        {
                            for (oi, obj_pts) in
                                obj_sample_points.iter().enumerate()
                            {
                                for (i, &(px, pz)) in
                                    obj_pts.iter().enumerate()
                                {
                                    if !per_obj[oi].contains(&i) {
                                        if point_in_polygon(
                                            px, pz, &vis_poly,
                                        ) {
                                            per_obj[oi].insert(i);
                                        }
                                    }
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

    // Build DZ visibility results
    if has_dzs {
        let mut dz_visibility = serde_json::Map::new();
        let mut dz_to_dz_visibility = serde_json::Map::new();

        for (dz_id, &(total_frac, obs_count)) in &dz_vis_accum {
            let avg = if obs_count > 0 {
                total_frac / obs_count as f64
            } else {
                0.0
            };
            let dz_samples_count = dz_data
                .iter()
                .find(|dd| dd.id == *dz_id)
                .map_or(0, |dd| dd.samples.len());
            dz_visibility.insert(
                dz_id.clone(),
                serde_json::json!({
                    "value": (avg * 100.0 * 100.0).round() / 100.0,
                    "dz_sample_count": dz_samples_count,
                    "observer_count": obs_count
                }),
            );
        }

        for (key, seen_set) in &dz_cross_seen {
            let target_id = key.split("_from_").next().unwrap_or("");
            let target_samples_count = dz_data
                .iter()
                .find(|dd| dd.id == target_id)
                .map_or(0, |dd| dd.samples.len());
            let hidden_count =
                target_samples_count - seen_set.len();
            let hidden_pct = if target_samples_count > 0 {
                (hidden_count as f64 / target_samples_count as f64
                    * 100.0
                    * 100.0)
                    .round()
                    / 100.0
            } else {
                0.0
            };
            let obs_count =
                dz_cross_obs_count.get(key).copied().unwrap_or(0);
            dz_to_dz_visibility.insert(
                key.clone(),
                serde_json::json!({
                    "value": hidden_pct,
                    "hidden_count": hidden_count,
                    "target_sample_count": target_samples_count,
                    "observer_count": obs_count
                }),
            );
        }

        result["dz_visibility"] =
            serde_json::Value::Object(dz_visibility);
        result["dz_to_dz_visibility"] =
            serde_json::Value::Object(dz_to_dz_visibility);

        // Build objective hidability results
        if has_objectives {
            let mission = layout.mission.as_ref().unwrap();
            let dzs = &mission.deployment_zones;
            let total_objectives = mission.objectives.len();
            let mut objective_hidability = serde_json::Map::new();

            for dz in dzs {
                // Count objectives where at least 1 sample NOT seen
                let mut safe_count = 0;
                if let Some(per_obj) = obj_seen_from_dz.get(&dz.id) {
                    for oi in 0..total_objectives {
                        let total_pts = obj_sample_points[oi].len();
                        if total_pts == 0 {
                            continue;
                        }
                        let seen_count = per_obj[oi].len();
                        if seen_count < total_pts {
                            safe_count += 1;
                        }
                    }
                }

                // Store under the opposing player's DZ id
                for other_dz in dzs {
                    if other_dz.id != dz.id {
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
                            other_dz.id.clone(),
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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);

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
            compute_layout_visibility(&layout, &objects, 4.0);

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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);

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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);
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
            compute_layout_visibility(&layout, &objects, 4.0);
        assert!(result.get("objective_hidability").is_none());
    }
}
