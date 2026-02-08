//! Visibility score computation for terrain layouts.
//!
//! Measures what percentage of the battlefield has clear line of sight
//! by sampling observer positions on a grid and computing visibility
//! polygons via angular sweep.

use std::collections::HashMap;

use crate::collision::{
    compose_transform, is_at_origin, mirror_placed_feature, obb_corners,
    Corners,
};
use crate::types::{PlacedFeature, TerrainLayout, TerrainObject, Transform};

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

/// Compute visibility score for a terrain layout.
///
/// Samples observer positions on a grid across the table, computes
/// visibility polygon for each, and returns the average visibility
/// ratio (visible area / total area).
pub fn compute_layout_visibility(
    layout: &TerrainLayout,
    objects_by_id: &HashMap<String, &TerrainObject>,
    grid_spacing: f64,
    grid_offset: f64,
    min_blocking_height: f64,
) -> serde_json::Value {
    let half_w = layout.table_width_inches / 2.0;
    let half_d = layout.table_depth_inches / 2.0;
    let table_area =
        layout.table_width_inches * layout.table_depth_inches;

    // Generate grid sample points
    let mut sample_points: Vec<(f64, f64)> = Vec::new();
    let mut x = -half_w + grid_offset;
    while x < half_w {
        let mut z = -half_d + grid_offset;
        while z < half_d {
            sample_points.push((x, z));
            z += grid_spacing;
        }
        x += grid_spacing;
    }

    if sample_points.is_empty() {
        return serde_json::json!({
            "overall": {
                "value": 100.0,
                "grid_spacing_inches": grid_spacing,
                "grid_offset_inches": grid_offset,
                "min_blocking_height_inches": min_blocking_height,
                "sample_count": 0
            }
        });
    }

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
            continue;
        }

        let vis_poly = compute_visibility_polygon(
            sx, sz, &segments, half_w, half_d,
        );
        if vis_poly.len() < 3 {
            total_ratio += 1.0;
            continue;
        }

        let vis_area = polygon_area(&vis_poly);
        let ratio = (vis_area / table_area).min(1.0);
        total_ratio += ratio;
    }

    let avg_visibility = total_ratio / sample_points.len() as f64;
    let value =
        (avg_visibility * 100.0 * 100.0).round() / 100.0; // round to 2 decimal places

    serde_json::json!({
        "overall": {
            "value": value,
            "grid_spacing_inches": grid_spacing,
            "grid_offset_inches": grid_offset,
            "min_blocking_height_inches": min_blocking_height,
            "sample_count": sample_points.len()
        }
    })
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
        }
    }

    #[test]
    fn empty_battlefield_full_visibility() {
        let layout = make_layout(60.0, 44.0, vec![]);
        let objects: HashMap<String, &TerrainObject> = HashMap::new();
        let result =
            compute_layout_visibility(&layout, &objects, 1.0, 0.5, 4.0);
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
            compute_layout_visibility(&layout, &objects, 1.0, 0.5, 4.0);
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
            compute_layout_visibility(&layout, &objects, 1.0, 0.5, 4.0);
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
            compute_layout_visibility(&layout, &objects, 1.0, 0.5, 4.0);
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
}
