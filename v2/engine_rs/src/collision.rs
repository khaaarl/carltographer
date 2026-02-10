//! Oriented bounding box (OBB) collision and bounds checking.
//!
//! Uses the Separating Axis Theorem for overlap detection between
//! rotated rectangles on the 2D table surface.

use std::collections::HashMap;

use crate::types::{PlacedFeature, TerrainObject, Transform};

pub type Corners = [(f64, f64); 4];

/// Create the 180-degree rotational mirror of a placed feature: (-x, -z, rot+180).
pub fn mirror_placed_feature(pf: &PlacedFeature) -> PlacedFeature {
    PlacedFeature {
        feature: pf.feature.clone(),
        transform: Transform {
            x_inches: -pf.transform.x_inches,
            y_inches: 0.0,
            z_inches: -pf.transform.z_inches,
            rotation_deg: pf.transform.rotation_deg + 180.0,
        },
    }
}

/// Check if a placed feature is at the table origin (0, 0).
pub fn is_at_origin(pf: &PlacedFeature) -> bool {
    pf.transform.x_inches == 0.0 && pf.transform.z_inches == 0.0
}

pub fn compose_transform(inner: &Transform, outer: &Transform) -> Transform {
    let cos_o = outer.rotation_deg.to_radians().cos();
    let sin_o = outer.rotation_deg.to_radians().sin();
    Transform {
        x_inches: outer.x_inches + inner.x_inches * cos_o - inner.z_inches * sin_o,
        y_inches: 0.0,
        z_inches: outer.z_inches + inner.x_inches * sin_o + inner.z_inches * cos_o,
        rotation_deg: inner.rotation_deg + outer.rotation_deg,
    }
}

pub fn obb_corners(cx: f64, cz: f64, half_w: f64, half_d: f64, rot_rad: f64) -> Corners {
    let cos_r = rot_rad.cos();
    let sin_r = rot_rad.sin();
    const SIGNS: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let mut corners = [(0.0, 0.0); 4];
    for (i, &(sx, sz)) in SIGNS.iter().enumerate() {
        let lx = sx * half_w;
        let lz = sz * half_d;
        corners[i] = (cx + lx * cos_r - lz * sin_r, cz + lx * sin_r + lz * cos_r);
    }
    corners
}

fn project(corners: &Corners, ax: f64, az: f64) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(cx, cz) in corners {
        let dot = cx * ax + cz * az;
        if dot < lo {
            lo = dot;
        }
        if dot > hi {
            hi = dot;
        }
    }
    (lo, hi)
}

/// True if the interiors of two OBBs overlap.
/// Touching (shared edge or corner) is NOT counted as overlap.
pub fn obbs_overlap(a: &Corners, b: &Corners) -> bool {
    for corners in [a, b] {
        // Only need 2 edge normals per rectangle (opposite
        // edges are parallel and give the same axis).
        for i in 0..2 {
            let j = (i + 1) % 4;
            let ex = corners[j].0 - corners[i].0;
            let ez = corners[j].1 - corners[i].1;
            let (ax, az) = (-ez, ex);
            let (min_a, max_a) = project(a, ax, az);
            let (min_b, max_b) = project(b, ax, az);
            if max_a <= min_b || max_b <= min_a {
                return false;
            }
        }
    }
    true
}

/// True if all corners are within (or on) the table edges.
pub fn obb_in_bounds(corners: &Corners, table_width: f64, table_depth: f64) -> bool {
    let half_w = table_width / 2.0;
    let half_d = table_depth / 2.0;
    corners
        .iter()
        .all(|&(cx, cz)| -half_w <= cx && cx <= half_w && -half_d <= cz && cz <= half_d)
}

/// Compute squared distance from point to line segment.
///
/// Uses vector projection. If projection falls within segment,
/// returns perpendicular distance. Otherwise returns distance
/// to nearest endpoint.
pub fn point_to_segment_distance_squared(
    px: f64,
    pz: f64,
    x1: f64,
    z1: f64,
    x2: f64,
    z2: f64,
) -> f64 {
    let seg_len_sq = (x2 - x1).powi(2) + (z2 - z1).powi(2);
    if seg_len_sq == 0.0 {
        return (px - x1).powi(2) + (pz - z1).powi(2);
    }

    let t = ((px - x1) * (x2 - x1) + (pz - z1) * (z2 - z1)) / seg_len_sq;

    if 0.0 < t && t < 1.0 {
        let proj_x = x1 + t * (x2 - x1);
        let proj_z = z1 + t * (z2 - z1);
        (px - proj_x).powi(2) + (pz - proj_z).powi(2)
    } else {
        let dist_to_start = (px - x1).powi(2) + (pz - z1).powi(2);
        let dist_to_end = (px - x2).powi(2) + (pz - z2).powi(2);
        dist_to_start.min(dist_to_end)
    }
}

/// Test if two line segments intersect (excluding endpoints).
///
/// Uses parametric line intersection algorithm.
pub fn segments_intersect(
    x1: f64,
    z1: f64,
    x2: f64,
    z2: f64,
    x3: f64,
    z3: f64,
    x4: f64,
    z4: f64,
) -> bool {
    let denominator = (z4 - z3) * (x2 - x1) - (x4 - x3) * (z2 - z1);

    if denominator == 0.0 {
        return false;
    }

    let ua = ((x4 - x3) * (z1 - z3) - (z4 - z3) * (x1 - x3)) / denominator;
    if ua <= 0.0 || ua >= 1.0 {
        return false;
    }

    let ub = ((x2 - x1) * (z1 - z3) - (z2 - z1) * (x1 - x3)) / denominator;
    0.0 < ub && ub < 1.0
}

/// Compute minimum distance between two oriented bounding boxes.
///
/// Returns 0 if rectangles intersect or touch. Otherwise returns
/// minimum distance between any corner and edge.
pub fn obb_distance(corners_a: &Corners, corners_b: &Corners) -> f64 {
    // Check edge intersections
    for i in 0..4 {
        let (ax1, az1) = corners_a[i];
        let (ax2, az2) = corners_a[(i + 1) % 4];
        for j in 0..4 {
            let (bx1, bz1) = corners_b[j];
            let (bx2, bz2) = corners_b[(j + 1) % 4];
            if segments_intersect(ax1, az1, ax2, az2, bx1, bz1, bx2, bz2) {
                return 0.0;
            }
        }
    }

    let mut min_dist_sq = f64::INFINITY;

    // Check corners of A against edges of B
    for (corner_x, corner_z) in corners_a {
        for i in 0..4 {
            let (bx1, bz1) = corners_b[i];
            let (bx2, bz2) = corners_b[(i + 1) % 4];
            let dist_sq =
                point_to_segment_distance_squared(*corner_x, *corner_z, bx1, bz1, bx2, bz2);
            min_dist_sq = min_dist_sq.min(dist_sq);
        }
    }

    // Check corners of B against edges of A
    for (corner_x, corner_z) in corners_b {
        for i in 0..4 {
            let (ax1, az1) = corners_a[i];
            let (ax2, az2) = corners_a[(i + 1) % 4];
            let dist_sq =
                point_to_segment_distance_squared(*corner_x, *corner_z, ax1, az1, ax2, az2);
            min_dist_sq = min_dist_sq.min(dist_sq);
        }
    }

    min_dist_sq.sqrt()
}

/// Compute minimum distance from OBB to nearest table edge.
///
/// Returns 0 if any corner is outside or on table boundary.
pub fn obb_to_table_edge_distance(corners: &Corners, table_width: f64, table_depth: f64) -> f64 {
    let half_w = table_width / 2.0;
    let half_d = table_depth / 2.0;

    let mut min_dist = f64::INFINITY;

    for (cx, cz) in corners {
        let dist_to_right = half_w - cx;
        let dist_to_left = half_w + cx;
        let dist_to_top = half_d - cz;
        let dist_to_bottom = half_d + cz;

        if dist_to_right <= 0.0
            || dist_to_left <= 0.0
            || dist_to_top <= 0.0
            || dist_to_bottom <= 0.0
        {
            return 0.0;
        }

        min_dist = min_dist.min(dist_to_right);
        min_dist = min_dist.min(dist_to_left);
        min_dist = min_dist.min(dist_to_top);
        min_dist = min_dist.min(dist_to_bottom);
    }

    min_dist
}

/// Compute world-space OBB corners for every shape in a placed
/// feature, composing shape offset, component transform, and
/// feature table transform.
pub fn get_world_obbs(
    placed: &PlacedFeature,
    objects_by_id: &HashMap<String, &TerrainObject>,
) -> Vec<Corners> {
    let default_t = Transform::default();
    let mut result = Vec::new();
    for comp in &placed.feature.components {
        let obj = match objects_by_id.get(&comp.object_id) {
            Some(o) => o,
            None => continue,
        };
        let comp_t = comp.transform.as_ref().unwrap_or(&default_t);
        for shape in &obj.shapes {
            let shape_t = shape.offset.as_ref().unwrap_or(&default_t);
            let world = compose_transform(&compose_transform(shape_t, comp_t), &placed.transform);
            let corners = obb_corners(
                world.x_inches,
                world.z_inches,
                shape.width_inches / 2.0,
                shape.depth_inches / 2.0,
                world.rotation_deg.to_radians(),
            );
            result.push(corners);
        }
    }
    result
}

/// Extract world-space OBB corners for shapes with height >= min_height.
///
/// Similar to get_world_obbs() but filters by shape height.
pub fn get_tall_world_obbs(
    placed: &PlacedFeature,
    objects_by_id: &HashMap<String, &TerrainObject>,
    min_height: f64,
) -> Vec<Corners> {
    let default_t = Transform::default();
    let mut result = Vec::new();
    for comp in &placed.feature.components {
        let obj = match objects_by_id.get(&comp.object_id) {
            Some(o) => o,
            None => continue,
        };
        let comp_t = comp.transform.as_ref().unwrap_or(&default_t);
        for shape in &obj.shapes {
            if shape.height_inches < min_height {
                continue;
            }

            let shape_t = shape.offset.as_ref().unwrap_or(&default_t);
            let world = compose_transform(&compose_transform(shape_t, comp_t), &placed.transform);
            let corners = obb_corners(
                world.x_inches,
                world.z_inches,
                shape.width_inches / 2.0,
                shape.depth_inches / 2.0,
                world.rotation_deg.to_radians(),
            );
            result.push(corners);
        }
    }
    result
}

/// Check that the feature at check_idx is within table bounds
/// and does not overlap any other placed feature.
///
/// Validates:
/// 1. All shapes within table bounds
/// 2. No overlap with other features (including mirrors if symmetric)
/// 3. Tall shapes (height >= 1") respect min_feature_gap
/// 4. Tall shapes (height >= 1") respect min_edge_gap
pub fn is_valid_placement(
    placed_features: &[PlacedFeature],
    check_idx: usize,
    table_width: f64,
    table_depth: f64,
    objects_by_id: &HashMap<String, &TerrainObject>,
    min_feature_gap: Option<f64>,
    min_edge_gap: Option<f64>,
    rotationally_symmetric: bool,
    min_all_feature_gap: Option<f64>,
    min_all_edge_gap: Option<f64>,
) -> bool {
    let check_pf = &placed_features[check_idx];
    let check_obbs = get_world_obbs(check_pf, objects_by_id);

    // 1. Check table bounds
    for corners in &check_obbs {
        if !obb_in_bounds(corners, table_width, table_depth) {
            return false;
        }
    }

    // Build expanded "other features" list including mirrors when symmetric
    let mut other_features: Vec<PlacedFeature> = Vec::new();
    for (i, pf) in placed_features.iter().enumerate() {
        if i == check_idx {
            continue;
        }
        other_features.push(pf.clone());
        if rotationally_symmetric && !is_at_origin(pf) {
            other_features.push(mirror_placed_feature(pf));
        }
    }

    // Also check feature's own mirror (can't overlap itself)
    if rotationally_symmetric && !is_at_origin(check_pf) {
        other_features.push(mirror_placed_feature(check_pf));
    }

    // 2. Check overlap with other features
    for pf in &other_features {
        let other_obbs = get_world_obbs(pf, objects_by_id);
        for ca in &check_obbs {
            for cb in &other_obbs {
                if obbs_overlap(ca, cb) {
                    return false;
                }
            }
        }
    }

    // 2b. All-feature edge gap (applies to all shapes, not just tall)
    if let Some(gap) = min_all_edge_gap {
        if gap > 0.0 {
            for corners in &check_obbs {
                let dist = obb_to_table_edge_distance(corners, table_width, table_depth);
                if dist < gap {
                    return false;
                }
            }
        }
    }

    // 2c. All-feature gap (applies to all shapes, not just tall)
    if let Some(gap) = min_all_feature_gap {
        if gap > 0.0 {
            for pf in &other_features {
                let other_obbs = get_world_obbs(pf, objects_by_id);
                for ca in &check_obbs {
                    for cb in &other_obbs {
                        let dist = obb_distance(ca, cb);
                        if dist < gap {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // Gap checking only for tall geometries (height >= 1")
    let check_tall = get_tall_world_obbs(check_pf, objects_by_id, 1.0);

    if check_tall.is_empty() {
        return true;
    }

    // 3. Check edge gap
    if let Some(gap) = min_edge_gap {
        if gap > 0.0 {
            for corners in &check_tall {
                let dist = obb_to_table_edge_distance(corners, table_width, table_depth);
                if dist < gap {
                    return false;
                }
            }
        }
    }

    // 4. Check feature gap
    if let Some(gap) = min_feature_gap {
        if gap > 0.0 {
            for pf in &other_features {
                let other_tall = get_tall_world_obbs(pf, objects_by_id, 1.0);
                for ca in &check_tall {
                    for cb in &other_tall {
                        let dist = obb_distance(ca, cb);
                        if dist < gap {
                            return false;
                        }
                    }
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CatalogFeature, CatalogObject, FeatureComponent, GeometricShape, TerrainCatalog,
        TerrainFeature, TerrainObject,
    };

    #[test]
    fn separated_no_overlap() {
        let a = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        let b = obb_corners(10.0, 0.0, 2.5, 1.25, 0.0);
        assert!(!obbs_overlap(&a, &b));
    }

    #[test]
    fn overlapping() {
        let a = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        let b = obb_corners(3.0, 0.0, 2.5, 1.25, 0.0);
        assert!(obbs_overlap(&a, &b));
    }

    #[test]
    fn touching_no_overlap() {
        let a = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        let b = obb_corners(5.0, 0.0, 2.5, 1.25, 0.0);
        assert!(!obbs_overlap(&a, &b));
    }

    #[test]
    fn touching_corner_no_overlap() {
        let a = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        let b = obb_corners(5.0, 2.5, 2.5, 1.25, 0.0);
        assert!(!obbs_overlap(&a, &b));
    }

    #[test]
    fn rotated_same_center_overlap() {
        let a = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        let b = obb_corners(0.0, 0.0, 2.5, 1.25, std::f64::consts::FRAC_PI_4);
        assert!(obbs_overlap(&a, &b));
    }

    #[test]
    fn in_bounds() {
        let c = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        assert!(obb_in_bounds(&c, 60.0, 44.0));
    }

    #[test]
    fn out_of_bounds() {
        let c = obb_corners(29.0, 0.0, 2.5, 1.25, 0.0);
        assert!(!obb_in_bounds(&c, 60.0, 44.0));
    }

    #[test]
    fn touching_edge_in_bounds() {
        let c = obb_corners(27.5, 0.0, 2.5, 1.25, 0.0);
        assert!(obb_in_bounds(&c, 60.0, 44.0));
    }

    #[test]
    fn point_to_segment_perpendicular() {
        // Point directly above segment midpoint
        let dist_sq = point_to_segment_distance_squared(0.0, 2.0, -1.0, 0.0, 1.0, 0.0);
        assert!((dist_sq - 4.0).abs() < 1e-10);
    }

    #[test]
    fn point_to_segment_endpoint() {
        // Point near segment endpoint
        let dist_sq = point_to_segment_distance_squared(0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        assert!(dist_sq < 1e-10);
    }

    #[test]
    fn segments_do_intersect() {
        // Two crossing segments
        let intersects = segments_intersect(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0);
        assert!(intersects);
    }

    #[test]
    fn segments_dont_intersect() {
        // Parallel segments
        let intersects = segments_intersect(-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0);
        assert!(!intersects);
    }

    #[test]
    fn obb_distance_separated() {
        let a = obb_corners(0.0, 0.0, 2.5, 1.25, 0.0);
        let b = obb_corners(10.0, 0.0, 2.5, 1.25, 0.0);
        let dist = obb_distance(&a, &b);
        assert!(dist > 0.0);
    }

    #[test]
    fn gap_checking_respects_edge_gap() {
        // Test that features must maintain distance from table edges
        let catalog = crate_catalog();
        let mut features = vec![];
        let objs = build_object_index(&catalog);

        // Place a feature close to edge (should fail with edge gap)
        features.push(PlacedFeature {
            feature: TerrainFeature {
                id: "feature_1".into(),
                feature_type: "obstacle".into(),
                components: vec![FeatureComponent {
                    object_id: "crate_5x2.5".into(),
                    transform: None,
                }],
                tags: vec![],
            },
            transform: Transform {
                x_inches: 27.0, // Close to edge at 30.0
                y_inches: 0.0,
                z_inches: 0.0,
                rotation_deg: 0.0,
            },
        });

        // This should fail with min_edge_gap_inches=2.0
        let valid = is_valid_placement(
            &features,
            0,
            60.0,
            44.0,
            &objs,
            None,
            Some(2.0),
            false,
            None,
            None,
        );
        assert!(!valid);

        // But pass without gap constraint
        let valid = is_valid_placement(
            &features, 0, 60.0, 44.0, &objs, None, None, false, None, None,
        );
        assert!(valid);
    }

    #[test]
    fn gap_checking_respects_feature_gap() {
        // Test that tall features must maintain distance from each other
        let catalog = crate_catalog();
        let mut features = vec![];
        let objs = build_object_index(&catalog);

        // Place two features close together (2 inches gap)
        features.push(PlacedFeature {
            feature: TerrainFeature {
                id: "feature_1".into(),
                feature_type: "obstacle".into(),
                components: vec![FeatureComponent {
                    object_id: "crate_5x2.5".into(),
                    transform: None,
                }],
                tags: vec![],
            },
            transform: Transform {
                x_inches: -5.0,
                y_inches: 0.0,
                z_inches: 0.0,
                rotation_deg: 0.0,
            },
        });

        features.push(PlacedFeature {
            feature: TerrainFeature {
                id: "feature_2".into(),
                feature_type: "obstacle".into(),
                components: vec![FeatureComponent {
                    object_id: "crate_5x2.5".into(),
                    transform: None,
                }],
                tags: vec![],
            },
            transform: Transform {
                x_inches: 4.0, // 2 inch gap: (-5+2.5) to (4-2.5) = -2.5 to 1.5
                y_inches: 0.0,
                z_inches: 0.0,
                rotation_deg: 0.0,
            },
        });

        // Gap is ~4 inches, so should fail with min_feature_gap_inches=5.0
        let valid = is_valid_placement(
            &features,
            1,
            60.0,
            44.0,
            &objs,
            Some(5.0),
            None,
            false,
            None,
            None,
        );
        assert!(!valid);

        // But pass without gap constraint
        let valid = is_valid_placement(
            &features, 1, 60.0, 44.0, &objs, None, None, false, None, None,
        );
        assert!(valid);
    }

    /// Catalog with short features (height 0.5") for all-feature gap tests.
    fn short_catalog() -> TerrainCatalog {
        TerrainCatalog {
            objects: vec![CatalogObject {
                item: TerrainObject {
                    id: "short_box".into(),
                    shapes: vec![GeometricShape {
                        shape_type: "rectangular_prism".into(),
                        width_inches: 5.0,
                        depth_inches: 2.5,
                        height_inches: 0.5,
                        offset: None,
                        opacity_height_inches: None,
                    }],
                    name: None,
                    tags: vec![],
                },
                quantity: None,
            }],
            features: vec![CatalogFeature {
                item: TerrainFeature {
                    id: "short_crate".into(),
                    feature_type: "obstacle".into(),
                    components: vec![FeatureComponent {
                        object_id: "short_box".into(),
                        transform: None,
                    }],
                    tags: vec![],
                },
                quantity: None,
            }],
            name: None,
        }
    }

    #[test]
    fn all_feature_edge_gap_works_on_short_features() {
        // Short features (height < 1.0) are ignored by tall-only gap checks
        // but should be caught by all-feature edge gap.
        let catalog = short_catalog();
        let objs = build_object_index(&catalog);
        let features = vec![PlacedFeature {
            feature: TerrainFeature {
                id: "feature_1".into(),
                feature_type: "obstacle".into(),
                components: vec![FeatureComponent {
                    object_id: "short_box".into(),
                    transform: None,
                }],
                tags: vec![],
            },
            transform: Transform {
                x_inches: 27.0, // Close to edge at 30.0
                y_inches: 0.0,
                z_inches: 0.0,
                rotation_deg: 0.0,
            },
        }];

        // Tall-only edge gap should pass (feature is short)
        let valid = is_valid_placement(
            &features,
            0,
            60.0,
            44.0,
            &objs,
            None,
            Some(5.0),
            false,
            None,
            None,
        );
        assert!(valid, "Short feature should pass tall-only edge gap");

        // All-feature edge gap should fail (feature is close to edge)
        let valid = is_valid_placement(
            &features,
            0,
            60.0,
            44.0,
            &objs,
            None,
            None,
            false,
            None,
            Some(5.0),
        );
        assert!(!valid, "Short feature should fail all-feature edge gap");

        // All-feature edge gap with smaller value should pass
        let valid = is_valid_placement(
            &features,
            0,
            60.0,
            44.0,
            &objs,
            None,
            None,
            false,
            None,
            Some(0.1),
        );
        assert!(
            valid,
            "Short feature should pass small all-feature edge gap"
        );
    }

    #[test]
    fn all_feature_gap_works_on_short_features() {
        // Short features are ignored by tall-only feature gap checks
        // but should be caught by all-feature gap.
        let catalog = short_catalog();
        let objs = build_object_index(&catalog);
        let features = vec![
            PlacedFeature {
                feature: TerrainFeature {
                    id: "feature_1".into(),
                    feature_type: "obstacle".into(),
                    components: vec![FeatureComponent {
                        object_id: "short_box".into(),
                        transform: None,
                    }],
                    tags: vec![],
                },
                transform: Transform {
                    x_inches: -5.0,
                    y_inches: 0.0,
                    z_inches: 0.0,
                    rotation_deg: 0.0,
                },
            },
            PlacedFeature {
                feature: TerrainFeature {
                    id: "feature_2".into(),
                    feature_type: "obstacle".into(),
                    components: vec![FeatureComponent {
                        object_id: "short_box".into(),
                        transform: None,
                    }],
                    tags: vec![],
                },
                transform: Transform {
                    x_inches: 4.0, // ~4 inch gap between features
                    y_inches: 0.0,
                    z_inches: 0.0,
                    rotation_deg: 0.0,
                },
            },
        ];

        // Tall-only feature gap should pass (features are short)
        let valid = is_valid_placement(
            &features,
            1,
            60.0,
            44.0,
            &objs,
            Some(5.0),
            None,
            false,
            None,
            None,
        );
        assert!(valid, "Short features should pass tall-only feature gap");

        // All-feature gap should fail (features are close)
        let valid = is_valid_placement(
            &features,
            1,
            60.0,
            44.0,
            &objs,
            None,
            None,
            false,
            Some(5.0),
            None,
        );
        assert!(!valid, "Short features should fail all-feature gap");

        // All-feature gap with smaller value should pass
        let valid = is_valid_placement(
            &features,
            1,
            60.0,
            44.0,
            &objs,
            None,
            None,
            false,
            Some(1.0),
            None,
        );
        assert!(valid, "Short features should pass small all-feature gap");
    }

    fn crate_catalog() -> TerrainCatalog {
        TerrainCatalog {
            objects: vec![CatalogObject {
                item: TerrainObject {
                    id: "crate_5x2.5".into(),
                    shapes: vec![GeometricShape {
                        shape_type: "rectangular_prism".into(),
                        width_inches: 5.0,
                        depth_inches: 2.5,
                        height_inches: 2.0,
                        offset: None,
                        opacity_height_inches: None,
                    }],
                    name: None,
                    tags: vec![],
                },
                quantity: None,
            }],
            features: vec![CatalogFeature {
                item: TerrainFeature {
                    id: "crate".into(),
                    feature_type: "obstacle".into(),
                    components: vec![FeatureComponent {
                        object_id: "crate_5x2.5".into(),
                        transform: None,
                    }],
                    tags: vec![],
                },
                quantity: None,
            }],
            name: None,
        }
    }

    fn build_object_index(catalog: &TerrainCatalog) -> HashMap<String, &TerrainObject> {
        let mut index = HashMap::new();
        for co in &catalog.objects {
            index.insert(co.item.id.clone(), &co.item);
        }
        index
    }
}
