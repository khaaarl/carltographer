//! Oriented bounding box (OBB) collision and bounds checking.
//!
//! Uses the Separating Axis Theorem for overlap detection between
//! rotated rectangles on the 2D table surface.

use std::collections::HashMap;

use crate::types::{PlacedFeature, TerrainObject, Transform};

pub type Corners = [(f64, f64); 4];

pub fn compose_transform(inner: &Transform, outer: &Transform) -> Transform {
    let cos_o = outer.rotation_deg.to_radians().cos();
    let sin_o = outer.rotation_deg.to_radians().sin();
    Transform {
        x_inches: outer.x_inches + inner.x_inches * cos_o
            - inner.z_inches * sin_o,
        y_inches: 0.0,
        z_inches: outer.z_inches + inner.x_inches * sin_o
            + inner.z_inches * cos_o,
        rotation_deg: inner.rotation_deg + outer.rotation_deg,
    }
}

pub fn obb_corners(
    cx: f64,
    cz: f64,
    half_w: f64,
    half_d: f64,
    rot_rad: f64,
) -> Corners {
    let cos_r = rot_rad.cos();
    let sin_r = rot_rad.sin();
    const SIGNS: [(f64, f64); 4] =
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let mut corners = [(0.0, 0.0); 4];
    for (i, &(sx, sz)) in SIGNS.iter().enumerate() {
        let lx = sx * half_w;
        let lz = sz * half_d;
        corners[i] = (
            cx + lx * cos_r - lz * sin_r,
            cz + lx * sin_r + lz * cos_r,
        );
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
pub fn obb_in_bounds(
    corners: &Corners,
    table_width: f64,
    table_depth: f64,
) -> bool {
    let half_w = table_width / 2.0;
    let half_d = table_depth / 2.0;
    corners
        .iter()
        .all(|&(cx, cz)| {
            -half_w <= cx
                && cx <= half_w
                && -half_d <= cz
                && cz <= half_d
        })
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
            let world = compose_transform(
                &compose_transform(shape_t, comp_t),
                &placed.transform,
            );
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
pub fn is_valid_placement(
    placed_features: &[PlacedFeature],
    check_idx: usize,
    table_width: f64,
    table_depth: f64,
    objects_by_id: &HashMap<String, &TerrainObject>,
) -> bool {
    let check_obbs =
        get_world_obbs(&placed_features[check_idx], objects_by_id);
    for corners in &check_obbs {
        if !obb_in_bounds(corners, table_width, table_depth) {
            return false;
        }
    }
    for (i, pf) in placed_features.iter().enumerate() {
        if i == check_idx {
            continue;
        }
        let other_obbs = get_world_obbs(pf, objects_by_id);
        for ca in &check_obbs {
            for cb in &other_obbs {
                if obbs_overlap(ca, cb) {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let b = obb_corners(
            0.0,
            0.0,
            2.5,
            1.25,
            std::f64::consts::FRAC_PI_4,
        );
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
}
