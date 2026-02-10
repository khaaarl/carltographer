//! Terrain layout mutation actions and undo logic.

use std::collections::HashMap;

use crate::collision::{get_world_obbs, is_at_origin, is_valid_placement, mirror_placed_feature};
use crate::prng::Pcg32;
use crate::types::{
    EngineParams, FeatureCountPreference, PlacedFeature, TerrainFeature, TerrainLayout,
    TerrainObject, Transform,
};

/// Undo token for reverting a single mutation step.
#[derive(Debug)]
pub(crate) enum StepUndo {
    Noop,
    Add { index: usize },
    Move { index: usize, old: PlacedFeature },
    Delete { index: usize, saved: PlacedFeature },
    Replace { index: usize, old: PlacedFeature },
    Rotate { index: usize, old: PlacedFeature },
}

pub(crate) fn quantize_position(value: f64) -> f64 {
    // Quantize position to nearest 0.1 inch.
    (value / 0.1).round() * 0.1
}

pub(crate) fn quantize_angle(value: f64, granularity: f64) -> f64 {
    // Quantize angle to nearest multiple of granularity degrees.
    (value / granularity).round() * granularity
}

pub(crate) fn count_features_by_type(
    features: &[PlacedFeature],
    rotationally_symmetric: bool,
) -> HashMap<String, u32> {
    // Count how many of each feature_type are visible on the table.
    // When rotationally_symmetric, non-origin features count as 2
    // (canonical + mirror). Origin features count as 1.
    let mut counts = HashMap::new();
    for pf in features {
        let ft = pf.feature.feature_type.clone();
        let inc = if rotationally_symmetric
            && (pf.transform.x_inches != 0.0 || pf.transform.z_inches != 0.0)
        {
            2
        } else {
            1
        };
        *counts.entry(ft).or_insert(0) += inc;
    }
    counts
}

pub(crate) fn weighted_choice(rng: &mut Pcg32, weights: &[f64]) -> Option<usize> {
    // Select index with probability proportional to weights.
    // Uses PCG32 for determinism. Returns None if all weights are 0.
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let r = rng.next_float() * total;
    let mut cumulative = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w;
        if r < cumulative {
            return Some(i);
        }
    }
    Some(weights.len() - 1)
}

fn count_placed_per_template(
    placed_features: &[PlacedFeature],
    catalog_features: &[&TerrainFeature],
    rotationally_symmetric: bool,
) -> Vec<u32> {
    // Count how many placed features match each catalog template.
    // Matches by comparing component object_id tuples.
    // When rotationally symmetric, non-origin features count as 2.
    let template_keys: Vec<Vec<&str>> = catalog_features
        .iter()
        .map(|cf| cf.components.iter().map(|c| c.object_id.as_str()).collect())
        .collect();
    let mut counts = vec![0u32; catalog_features.len()];
    for pf in placed_features {
        let pf_key: Vec<&str> = pf
            .feature
            .components
            .iter()
            .map(|c| c.object_id.as_str())
            .collect();
        let increment = if rotationally_symmetric
            && (pf.transform.x_inches != 0.0 || pf.transform.z_inches != 0.0)
        {
            2
        } else {
            1
        };
        for (i, tk) in template_keys.iter().enumerate() {
            if pf_key == *tk {
                counts[i] += increment;
                break;
            }
        }
    }
    counts
}

fn compute_template_weights(
    catalog_features: &[&TerrainFeature],
    feature_counts: &HashMap<String, u32>,
    preferences: &[FeatureCountPreference],
    catalog_quantities: &[Option<u32>],
    placed_features: &[PlacedFeature],
    rotationally_symmetric: bool,
    shortage_boost: f64,
    penalty_factor: f64,
) -> Vec<f64> {
    // Compute weights for selecting which catalog feature to add.
    // Boosts weight for types below their min, reduces for types at/above max.
    // Sets weight to 0 for templates that have reached their catalog quantity limit.
    let pref_by_type: HashMap<&str, &FeatureCountPreference> = preferences
        .iter()
        .map(|p| (p.feature_type.as_str(), p))
        .collect();
    let placed_counts =
        count_placed_per_template(placed_features, catalog_features, rotationally_symmetric);
    let mut weights = Vec::with_capacity(catalog_features.len());
    for (i, cf) in catalog_features.iter().enumerate() {
        if let Some(qty) = catalog_quantities[i] {
            if placed_counts[i] >= qty {
                weights.push(0.0);
                continue;
            }
        }
        let mut w = 1.0;
        if let Some(pref) = pref_by_type.get(cf.feature_type.as_str()) {
            let current = feature_counts.get(&cf.feature_type).copied().unwrap_or(0);
            if current < pref.min {
                w = 1.0 + (pref.min - current) as f64 * shortage_boost;
            } else if let Some(max) = pref.max {
                if current >= max {
                    w = penalty_factor;
                }
            }
        }
        weights.push(w);
    }
    weights
}

fn compute_delete_weights(
    features: &[PlacedFeature],
    feature_counts: &HashMap<String, u32>,
    preferences: &[FeatureCountPreference],
    excess_boost: f64,
    penalty_factor: f64,
) -> Vec<f64> {
    // Compute weights for selecting which feature to delete.
    // Boosts weight for types above their max, reduces for types at/below min.
    let pref_by_type: HashMap<&str, &FeatureCountPreference> = preferences
        .iter()
        .map(|p| (p.feature_type.as_str(), p))
        .collect();
    let mut weights = Vec::with_capacity(features.len());
    for pf in features {
        let mut w = 1.0;
        if let Some(pref) = pref_by_type.get(pf.feature.feature_type.as_str()) {
            let current = feature_counts
                .get(&pf.feature.feature_type)
                .copied()
                .unwrap_or(0);
            if let Some(max) = pref.max {
                if current > max {
                    w = 1.0 + (current - max) as f64 * excess_boost;
                } else if current <= pref.min {
                    w = penalty_factor;
                }
            } else if current <= pref.min {
                w = penalty_factor;
            }
        }
        weights.push(w);
    }
    weights
}

/// Compute per-tile placement weights inversely proportional to occupancy.
///
/// Divides table into ~tile_size tiles and counts how many feature OBBs
/// overlap each tile. Weight = 1/(1+count), biasing placement toward
/// empty areas.
///
/// Could be cached and invalidated on layout changes (similar to
/// VisibilityCache pattern) if performance becomes a concern.
pub(crate) fn compute_tile_weights(
    placed_features: &[PlacedFeature],
    objects_by_id: &HashMap<String, &TerrainObject>,
    table_width: f64,
    table_depth: f64,
    rotationally_symmetric: bool,
    tile_size: f64,
) -> (Vec<f64>, usize, usize, f64, f64) {
    let nx = (table_width / tile_size).round().max(1.0) as usize;
    let nz = (table_depth / tile_size).round().max(1.0) as usize;
    let tile_w = table_width / nx as f64;
    let tile_d = table_depth / nz as f64;
    let half_w = table_width / 2.0;
    let half_d = table_depth / 2.0;

    let mut counts = vec![0u32; nx * nz];

    for pf in placed_features {
        let mut obbs = get_world_obbs(pf, objects_by_id);
        if rotationally_symmetric && !is_at_origin(pf) {
            let mirror = mirror_placed_feature(pf);
            obbs.extend(get_world_obbs(&mirror, objects_by_id));
        }
        for corners in &obbs {
            // Compute AABB from corners
            let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
            let max_x = corners
                .iter()
                .map(|c| c.0)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_z = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
            let max_z = corners
                .iter()
                .map(|c| c.1)
                .fold(f64::NEG_INFINITY, f64::max);
            // Convert to tile indices
            let ix_lo = ((min_x + half_w) / tile_w) as isize;
            let ix_hi = ((max_x + half_w) / tile_w) as isize;
            let iz_lo = ((min_z + half_d) / tile_d) as isize;
            let iz_hi = ((max_z + half_d) / tile_d) as isize;
            let ix_lo = ix_lo.max(0) as usize;
            let ix_hi = (ix_hi as usize).min(nx - 1);
            let iz_lo = iz_lo.max(0) as usize;
            let iz_hi = (iz_hi as usize).min(nz - 1);
            for iz in iz_lo..=iz_hi {
                for ix in ix_lo..=ix_hi {
                    counts[iz * nx + ix] += 1;
                }
            }
        }
    }

    let weights: Vec<f64> = counts.iter().map(|&c| 1.0 / (1 + c) as f64).collect();
    (weights, nx, nz, tile_w, tile_d)
}

fn instantiate_feature(template: &TerrainFeature, feature_id: u32) -> TerrainFeature {
    TerrainFeature {
        id: format!("feature_{}", feature_id),
        feature_type: template.feature_type.clone(),
        components: template.components.clone(),
        tags: template.tags.clone(),
    }
}

/// Generate a temperature-aware move transform.
///
/// At t_factor=0, displacement is small (±min_move_range/2 inches).
/// At t_factor=1, displacement spans the full table.
/// Always consumes exactly 4 PRNG values.
pub(crate) fn temperature_move(
    rng: &mut Pcg32,
    old_transform: &Transform,
    table_width: f64,
    table_depth: f64,
    t_factor: f64,
    rotation_granularity: f64,
    min_move_range: f64,
    rotate_on_move_prob: f64,
) -> Transform {
    let max_dim = table_width.max(table_depth);
    let move_range = min_move_range + t_factor * (max_dim - min_move_range);

    let dx = (rng.next_float() - 0.5) * 2.0 * move_range;
    let dz = (rng.next_float() - 0.5) * 2.0 * move_range;
    let rotate_check = rng.next_float();
    let rot_angle_raw = rng.next_float();

    let new_x = quantize_position(old_transform.x_inches + dx);
    let new_z = quantize_position(old_transform.z_inches + dz);

    // Rotation: 0% chance at t=0, rotate_on_move_prob chance at t=1
    let rot = if rotate_check < rotate_on_move_prob * t_factor {
        quantize_angle(rot_angle_raw * 360.0, rotation_granularity)
    } else {
        old_transform.rotation_deg
    };

    Transform {
        x_inches: new_x,
        y_inches: 0.0,
        z_inches: new_z,
        rotation_deg: rot,
    }
}

/// Attempt one mutation action. Returns (undo, new_next_id) or None on failure.
fn try_single_action(
    layout: &mut TerrainLayout,
    rng: &mut Pcg32,
    t_factor: f64,
    next_id: u32,
    catalog_features: &[&TerrainFeature],
    has_catalog: bool,
    objects_by_id: &HashMap<String, &TerrainObject>,
    params: &EngineParams,
    prefs: &[FeatureCountPreference],
    catalog_quantities: &[Option<u32>],
    index_in_chain: u32,
    chain_length: u32,
) -> Option<(StepUndo, u32)> {
    let has_features = !layout.placed_features.is_empty();
    let feature_counts =
        count_features_by_type(&layout.placed_features, params.rotationally_symmetric);
    let tuning = params.tuning();

    // Compute action weights: [add, move, delete, replace, rotate]
    // Non-final mutations in a chain get full delete weight (clearing space
    // for subsequent add/move), while final/standalone mutations use the
    // reduced base weight to avoid wasting scoring compute on doomed deletes.
    let mut add_weight: f64 = if has_catalog { 1.0 } else { 0.0 };
    let move_weight: f64 = if has_features { 1.0 } else { 0.0 };
    let is_last = index_in_chain >= chain_length - 1;
    let mut delete_weight: f64 = if !has_features {
        0.0
    } else if is_last {
        tuning.delete_weight_last
    } else {
        1.0
    };
    let replace_weight: f64 = if has_features && has_catalog {
        1.0
    } else {
        0.0
    };
    let rotate_weight: f64 = if has_features { 1.0 } else { 0.0 };

    // Preference biasing on add/delete only
    for pref in prefs {
        let current = feature_counts.get(&pref.feature_type).copied().unwrap_or(0);
        if current < pref.min {
            let shortage = pref.min - current;
            add_weight *= 1.0 + shortage as f64 * tuning.shortage_boost;
            delete_weight *= tuning.penalty_factor;
        } else if let Some(max) = pref.max {
            if current > max {
                let excess = current - max;
                delete_weight *= 1.0 + excess as f64 * tuning.excess_boost;
                add_weight *= tuning.penalty_factor;
            }
        }
    }

    let weights = vec![
        add_weight,
        move_weight,
        delete_weight,
        replace_weight,
        rotate_weight,
    ];
    let action = match weighted_choice(rng, &weights) {
        Some(idx) => idx as u32,
        None => return None,
    };

    match action {
        0 => {
            // Add
            let template_weights = compute_template_weights(
                catalog_features,
                &feature_counts,
                prefs,
                catalog_quantities,
                &layout.placed_features,
                params.rotationally_symmetric,
                tuning.shortage_boost,
                tuning.penalty_factor,
            );
            let tidx = weighted_choice(rng, &template_weights)?;
            let template = catalog_features[tidx];
            let new_feat = instantiate_feature(template, next_id);
            let half_w = params.table_width_inches / 2.0;
            let half_d = params.table_depth_inches / 2.0;
            let (tile_weights, nx, _nz, tile_w, tile_d) = compute_tile_weights(
                &layout.placed_features,
                objects_by_id,
                params.table_width_inches,
                params.table_depth_inches,
                params.rotationally_symmetric,
                tuning.tile_size,
            );
            let tile_idx = weighted_choice(rng, &tile_weights)?;
            let tile_iz = tile_idx / nx;
            let tile_ix = tile_idx % nx;
            let tile_x_min = -half_w + tile_ix as f64 * tile_w;
            let tile_z_min = -half_d + tile_iz as f64 * tile_d;
            let x = quantize_position(tile_x_min + rng.next_float() * tile_w);
            let z = quantize_position(tile_z_min + rng.next_float() * tile_d);
            let rot = quantize_angle(rng.next_float() * 360.0, params.rotation_granularity_deg);
            layout.placed_features.push(PlacedFeature {
                feature: new_feat,
                transform: Transform {
                    x_inches: x,
                    y_inches: 0.0,
                    z_inches: z,
                    rotation_deg: rot,
                },
            });
            let idx = layout.placed_features.len() - 1;
            if is_valid_placement(
                &layout.placed_features,
                idx,
                params.table_width_inches,
                params.table_depth_inches,
                objects_by_id,
                params.min_feature_gap_inches,
                params.min_edge_gap_inches,
                params.rotationally_symmetric,
                params.min_all_feature_gap_inches,
                params.min_all_edge_gap_inches,
            ) {
                Some((StepUndo::Add { index: idx }, next_id + 1))
            } else {
                layout.placed_features.pop();
                None
            }
        }
        1 => {
            // Move (temperature-aware)
            let idx = rng.next_int(0, layout.placed_features.len() as u32 - 1) as usize;
            let old = layout.placed_features[idx].clone();
            let new_transform = temperature_move(
                rng,
                &old.transform,
                params.table_width_inches,
                params.table_depth_inches,
                t_factor,
                params.rotation_granularity_deg,
                tuning.min_move_range,
                tuning.rotate_on_move_prob,
            );
            layout.placed_features[idx].transform = new_transform;
            if is_valid_placement(
                &layout.placed_features,
                idx,
                params.table_width_inches,
                params.table_depth_inches,
                objects_by_id,
                params.min_feature_gap_inches,
                params.min_edge_gap_inches,
                params.rotationally_symmetric,
                params.min_all_feature_gap_inches,
                params.min_all_edge_gap_inches,
            ) {
                Some((StepUndo::Move { index: idx, old }, next_id))
            } else {
                layout.placed_features[idx] = old;
                None
            }
        }
        2 => {
            // Delete
            let delete_weights = compute_delete_weights(
                &layout.placed_features,
                &feature_counts,
                prefs,
                tuning.excess_boost,
                tuning.penalty_factor,
            );
            let idx = weighted_choice(rng, &delete_weights)?;
            let saved = layout.placed_features.remove(idx);
            Some((StepUndo::Delete { index: idx, saved }, next_id))
        }
        3 => {
            // Replace: remove feature, add different template at same position
            let delete_weights = compute_delete_weights(
                &layout.placed_features,
                &feature_counts,
                prefs,
                tuning.excess_boost,
                tuning.penalty_factor,
            );
            let idx = weighted_choice(rng, &delete_weights)?;
            let template_weights = compute_template_weights(
                catalog_features,
                &feature_counts,
                prefs,
                catalog_quantities,
                &layout.placed_features,
                params.rotationally_symmetric,
                tuning.shortage_boost,
                tuning.penalty_factor,
            );
            let tidx = weighted_choice(rng, &template_weights)?;
            let template = catalog_features[tidx];
            let old = layout.placed_features[idx].clone();
            let new_feat = instantiate_feature(template, next_id);
            layout.placed_features[idx] = PlacedFeature {
                feature: new_feat,
                transform: old.transform.clone(),
            };
            if is_valid_placement(
                &layout.placed_features,
                idx,
                params.table_width_inches,
                params.table_depth_inches,
                objects_by_id,
                params.min_feature_gap_inches,
                params.min_edge_gap_inches,
                params.rotationally_symmetric,
                params.min_all_feature_gap_inches,
                params.min_all_edge_gap_inches,
            ) {
                Some((StepUndo::Replace { index: idx, old }, next_id + 1))
            } else {
                layout.placed_features[idx] = old;
                None
            }
        }
        4 => {
            // Rotate: pick random feature, assign new quantized angle
            let idx = rng.next_int(0, layout.placed_features.len() as u32 - 1) as usize;
            let old = layout.placed_features[idx].clone();
            let new_rot = quantize_angle(rng.next_float() * 360.0, params.rotation_granularity_deg);
            layout.placed_features[idx].transform = Transform {
                x_inches: old.transform.x_inches,
                y_inches: 0.0,
                z_inches: old.transform.z_inches,
                rotation_deg: new_rot,
            };
            if is_valid_placement(
                &layout.placed_features,
                idx,
                params.table_width_inches,
                params.table_depth_inches,
                objects_by_id,
                params.min_feature_gap_inches,
                params.min_edge_gap_inches,
                params.rotationally_symmetric,
                params.min_all_feature_gap_inches,
                params.min_all_edge_gap_inches,
            ) {
                Some((StepUndo::Rotate { index: idx, old }, next_id))
            } else {
                layout.placed_features[idx] = old;
                None
            }
        }
        _ => None,
    }
}

/// Try mutations with decaying temperature until one succeeds or retries exhausted.
pub(crate) fn perform_step(
    layout: &mut TerrainLayout,
    rng: &mut Pcg32,
    t_factor: f64,
    next_id: u32,
    catalog_features: &[&TerrainFeature],
    has_catalog: bool,
    objects_by_id: &HashMap<String, &TerrainObject>,
    params: &EngineParams,
    prefs: &[FeatureCountPreference],
    catalog_quantities: &[Option<u32>],
    index_in_chain: u32,
    chain_length: u32,
) -> (StepUndo, u32) {
    let tuning = params.tuning();
    let mut effective_t = t_factor;
    for _ in 0..tuning.max_retries {
        if let Some(result) = try_single_action(
            layout,
            rng,
            effective_t,
            next_id,
            catalog_features,
            has_catalog,
            objects_by_id,
            params,
            prefs,
            catalog_quantities,
            index_in_chain,
            chain_length,
        ) {
            return result;
        }
        effective_t *= tuning.retry_decay;
    }
    (StepUndo::Noop, next_id)
}

/// Revert a mutation using its undo token.
pub(crate) fn undo_step(layout: &mut TerrainLayout, undo: StepUndo) {
    match undo {
        StepUndo::Noop => {}
        StepUndo::Add { index, .. } => {
            layout.placed_features.remove(index);
        }
        StepUndo::Move { index, old } => {
            layout.placed_features[index] = old;
        }
        StepUndo::Delete { index, saved } => {
            layout.placed_features.insert(index, saved);
        }
        StepUndo::Replace { index, old, .. } => {
            layout.placed_features[index] = old;
        }
        StepUndo::Rotate { index, old } => {
            layout.placed_features[index] = old;
        }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Pcg32;
    use crate::types::{
        CatalogFeature, CatalogObject, EngineParams, FeatureComponent, GeometricShape,
        TerrainCatalog, TerrainFeature, Transform, TuningParams,
    };

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

    fn make_params(seed: u64, num_steps: u32) -> EngineParams {
        EngineParams {
            seed,
            table_width_inches: 60.0,
            table_depth_inches: 44.0,
            catalog: crate_catalog(),
            num_steps: Some(num_steps),
            initial_layout: None,
            feature_count_preferences: None,
            min_feature_gap_inches: None,
            min_edge_gap_inches: None,
            min_all_feature_gap_inches: None,
            min_all_edge_gap_inches: None,
            rotation_granularity_deg: 15.0,
            rotationally_symmetric: false,
            mission: None,
            skip_visibility: false,
            scoring_targets: None,
            num_replicas: None,
            swap_interval: 20,
            max_temperature: 50.0,
            tuning: None,
        }
    }

    #[test]
    fn temperature_move_small_t() {
        let defaults = TuningParams::default();
        let mut rng = Pcg32::new(42, 0);
        let old = Transform {
            x_inches: 0.0,
            y_inches: 0.0,
            z_inches: 0.0,
            rotation_deg: 90.0,
        };
        for _ in 0..100 {
            let t = temperature_move(
                &mut rng,
                &old,
                60.0,
                44.0,
                0.0,
                15.0,
                defaults.min_move_range,
                defaults.rotate_on_move_prob,
            );
            assert!((t.x_inches - old.x_inches).abs() <= defaults.min_move_range + 0.1);
            assert!((t.z_inches - old.z_inches).abs() <= defaults.min_move_range + 0.1);
            assert_eq!(t.rotation_deg, 90.0); // no rotation at t=0
        }
    }

    #[test]
    fn temperature_move_consumes_4_prng() {
        let defaults = TuningParams::default();
        let mut rng1 = Pcg32::new(99, 0);
        let mut rng2 = Pcg32::new(99, 0);
        let old = Transform {
            x_inches: 5.0,
            y_inches: 0.0,
            z_inches: 3.0,
            rotation_deg: 30.0,
        };
        temperature_move(
            &mut rng1,
            &old,
            60.0,
            44.0,
            0.5,
            15.0,
            defaults.min_move_range,
            defaults.rotate_on_move_prob,
        );
        for _ in 0..4 {
            rng2.next_float();
        }
        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }

    #[test]
    fn tile_weights_empty_table() {
        let defaults = TuningParams::default();
        let catalog = crate_catalog();
        let objs = crate::generate::build_object_index(&catalog);
        let (weights, nx, nz, _, _) =
            compute_tile_weights(&[], &objs, 60.0, 44.0, false, defaults.tile_size);
        assert_eq!(nx, 30); // 60/2
        assert_eq!(nz, 22); // 44/2
        assert!(weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn tile_weights_occupied_lower() {
        let defaults = TuningParams::default();
        let params = make_params(42, 200);
        let result = crate::generate::generate(&params);
        let objs = crate::generate::build_object_index(&params.catalog);
        let (weights, _, _, _, _) = compute_tile_weights(
            &result.layout.placed_features,
            &objs,
            params.table_width_inches,
            params.table_depth_inches,
            false,
            defaults.tile_size,
        );
        let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min_w < max_w);
        assert_eq!(max_w, 1.0);
        assert!(min_w < 1.0);
    }
}
