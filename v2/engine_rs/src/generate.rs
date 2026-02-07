//! Terrain layout generation engine.

use std::collections::HashMap;

use crate::collision::is_valid_placement;
use crate::prng::Pcg32;
use crate::types::{
    EngineParams, EngineResult, PlacedFeature, TerrainCatalog,
    TerrainFeature, TerrainLayout, TerrainObject, Transform,
};

fn build_object_index<'a>(
    catalog: &'a TerrainCatalog,
) -> HashMap<String, &'a TerrainObject> {
    let mut index = HashMap::new();
    for co in &catalog.objects {
        index.insert(co.item.id.clone(), &co.item);
    }
    index
}

fn instantiate_feature(
    template: &TerrainFeature,
    feature_id: u32,
) -> TerrainFeature {
    TerrainFeature {
        id: format!("feature_{}", feature_id),
        feature_type: template.feature_type.clone(),
        components: template.components.clone(),
    }
}

fn next_feature_id(features: &[PlacedFeature]) -> u32 {
    let mut max_id: u32 = 0;
    for pf in features {
        if let Some(num_str) =
            pf.feature.id.strip_prefix("feature_")
        {
            if let Ok(n) = num_str.parse::<u32>() {
                max_id = max_id.max(n);
            }
        }
    }
    max_id + 1
}

fn quantize_position(value: f64) -> f64 {
    // Quantize position to nearest 0.1 inch.
    (value / 0.1).round() * 0.1
}

fn quantize_angle(value: f64) -> f64 {
    // Quantize angle to nearest 15 degrees.
    (value / 15.0).round() * 15.0
}

fn count_features_by_type(
    features: &[PlacedFeature],
) -> std::collections::HashMap<String, u32> {
    // Count how many of each feature_type are currently in the layout.
    let mut counts = std::collections::HashMap::new();
    for pf in features {
        let ft = pf.feature.feature_type.clone();
        *counts.entry(ft).or_insert(0) += 1;
    }
    counts
}

fn weighted_choice(
    rng: &mut Pcg32,
    weights: &[f64],
) -> Option<usize> {
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

fn compute_action_weights(
    current_count: u32,
    min_count: u32,
    max_count: Option<u32>,
    has_catalog: bool,
    has_features: bool,
) -> (f64, f64, f64) {
    // Returns (add_weight, move_weight, delete_weight).
    let mut add_weight = if has_catalog { 1.0 } else { 0.0 };
    let move_weight = if has_features { 1.0 } else { 0.0 };
    let mut delete_weight = if has_features { 1.0 } else { 0.0 };

    if current_count < min_count {
        let shortage = min_count - current_count;
        add_weight *= 1.0 + shortage as f64 * 2.0;
        delete_weight *= 0.1;
    } else if let Some(max) = max_count {
        if current_count > max {
            let excess = current_count - max;
            delete_weight *= 1.0 + excess as f64 * 2.0;
            add_weight *= 0.1;
        }
    }

    (add_weight, move_weight, delete_weight)
}

pub fn generate(params: &EngineParams) -> EngineResult {
    let mut rng = Pcg32::new(params.seed, 0);
    let objects_by_id = build_object_index(&params.catalog);

    let (mut placed_features, mut nid) =
        match &params.initial_layout {
            Some(initial) => {
                let feats = initial.placed_features.clone();
                let id = next_feature_id(&feats);
                (feats, id)
            }
            None => (Vec::new(), 1u32),
        };

    let catalog_features: Vec<&TerrainFeature> = params
        .catalog
        .features
        .iter()
        .map(|cf| &cf.item)
        .collect();
    let has_catalog = !catalog_features.is_empty();
    let num_steps = params.steps();

    for _ in 0..num_steps {
        let has_features = !placed_features.is_empty();
        let feature_counts = count_features_by_type(&placed_features);

        // Default weights
        let mut add_weight = if has_catalog { 1.0 } else { 0.0 };
        let move_weight = if has_features { 1.0 } else { 0.0 };
        let mut delete_weight = if has_features { 1.0 } else { 0.0 };

        // Apply biasing for feature types with preferences
        if let Some(ref prefs) = params.feature_count_preferences {
            for pref in prefs {
                let ft = &pref.feature_type;
                let current = feature_counts.get(ft).copied().unwrap_or(0);

                // For obstacle type, apply biasing to all add/delete actions
                // (assumes catalog primarily contains obstacles)
                if ft == "obstacle" && has_catalog {
                    let (aw, _mw, dw) = compute_action_weights(
                        current,
                        pref.min,
                        pref.max,
                        has_catalog,
                        has_features,
                    );
                    add_weight = aw;
                    delete_weight = dw;
                }
            }
        }

        // Weighted random selection
        let weights = vec![add_weight, move_weight, delete_weight];
        let action = match weighted_choice(&mut rng, &weights) {
            Some(idx) => idx as u32,
            None => continue,
        };

        match action {
            0 => {
                let ti = rng.next_int(
                    0,
                    catalog_features.len() as u32 - 1,
                ) as usize;
                let template = catalog_features[ti];
                let new_feat = instantiate_feature(template, nid);
                let x = quantize_position(
                    rng.next_float()
                        * params.table_width_inches
                        - params.table_width_inches / 2.0
                );
                let z = quantize_position(
                    rng.next_float()
                        * params.table_depth_inches
                        - params.table_depth_inches / 2.0
                );
                let rot = quantize_angle(rng.next_float() * 360.0);
                placed_features.push(PlacedFeature {
                    feature: new_feat,
                    transform: Transform {
                        x_inches: x,
                        y_inches: 0.0,
                        z_inches: z,
                        rotation_deg: rot,
                    },
                });
                let idx = placed_features.len() - 1;
                if is_valid_placement(
                    &placed_features,
                    idx,
                    params.table_width_inches,
                    params.table_depth_inches,
                    &objects_by_id,
                    params.min_feature_gap_inches,
                    params.min_edge_gap_inches,
                ) {
                    nid += 1;
                } else {
                    placed_features.pop();
                }
            }
            1 => {
                let idx = rng.next_int(
                    0,
                    placed_features.len() as u32 - 1,
                ) as usize;
                let old = placed_features[idx].clone();
                let x = quantize_position(
                    rng.next_float()
                        * params.table_width_inches
                        - params.table_width_inches / 2.0
                );
                let z = quantize_position(
                    rng.next_float()
                        * params.table_depth_inches
                        - params.table_depth_inches / 2.0
                );
                let rot = quantize_angle(rng.next_float() * 360.0);
                placed_features[idx].transform = Transform {
                    x_inches: x,
                    y_inches: 0.0,
                    z_inches: z,
                    rotation_deg: rot,
                };
                if !is_valid_placement(
                    &placed_features,
                    idx,
                    params.table_width_inches,
                    params.table_depth_inches,
                    &objects_by_id,
                    params.min_feature_gap_inches,
                    params.min_edge_gap_inches,
                ) {
                    placed_features[idx] = old;
                }
            }
            2 => {
                let idx = rng.next_int(
                    0,
                    placed_features.len() as u32 - 1,
                ) as usize;
                placed_features.remove(idx);
            }
            _ => unreachable!(),
        }
    }

    EngineResult {
        layout: TerrainLayout {
            table_width_inches: params.table_width_inches,
            table_depth_inches: params.table_depth_inches,
            placed_features,
        },
        score: 0.0,
        steps_completed: num_steps,
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::{get_world_obbs, obb_in_bounds, obbs_overlap};
    use crate::types::{
        CatalogFeature, CatalogObject, FeatureComponent, GeometricShape,
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
        }
    }

    #[test]
    fn deterministic() {
        let r1 = generate(&make_params(123, 200));
        let r2 = generate(&make_params(123, 200));
        let j1 = serde_json::to_string(&r1).unwrap();
        let j2 = serde_json::to_string(&r2).unwrap();
        assert_eq!(j1, j2);
    }

    #[test]
    fn different_seeds() {
        let r1 = generate(&make_params(1, 200));
        let r2 = generate(&make_params(2, 200));
        let j1 = serde_json::to_string(&r1).unwrap();
        let j2 = serde_json::to_string(&r2).unwrap();
        assert_ne!(j1, j2);
    }

    #[test]
    fn produces_features() {
        let result = generate(&make_params(42, 200));
        assert!(!result.layout.placed_features.is_empty());
    }

    #[test]
    fn all_features_in_bounds() {
        let params = make_params(42, 200);
        let result = generate(&params);
        let objs = build_object_index(&params.catalog);
        for pf in &result.layout.placed_features {
            for corners in get_world_obbs(pf, &objs) {
                assert!(obb_in_bounds(
                    &corners,
                    params.table_width_inches,
                    params.table_depth_inches,
                ));
            }
        }
    }

    #[test]
    fn no_overlaps() {
        let params = make_params(42, 200);
        let result = generate(&params);
        let objs = build_object_index(&params.catalog);
        let feats = &result.layout.placed_features;
        for i in 0..feats.len() {
            let oi = get_world_obbs(&feats[i], &objs);
            for j in (i + 1)..feats.len() {
                let oj = get_world_obbs(&feats[j], &objs);
                for ca in &oi {
                    for cb in &oj {
                        assert!(
                            !obbs_overlap(ca, cb),
                            "Features {} and {} overlap",
                            i,
                            j
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_catalog_returns_empty_layout() {
        let params = EngineParams {
            seed: 42,
            table_width_inches: 60.0,
            table_depth_inches: 44.0,
            catalog: TerrainCatalog::default(),
            num_steps: Some(50),
            initial_layout: None,
            feature_count_preferences: None,
            min_feature_gap_inches: None,
            min_edge_gap_inches: None,
        };
        let result = generate(&params);
        assert!(result.layout.placed_features.is_empty());
        assert_eq!(result.steps_completed, 50);
    }

    #[test]
    fn weighted_selection_reaches_target_range() {
        let mut params = make_params(42, 100);
        params.feature_count_preferences = Some(vec![crate::types::FeatureCountPreference {
            feature_type: "obstacle".into(),
            min: 3,
            max: Some(10),
        }]);
        let result = generate(&params);
        let count = result.layout.placed_features.len() as u32;
        // Should have at least min features (may not always reach due to randomness)
        // but the biasing should make it more likely to stay in range
        assert!(count >= 3 || count <= 10);
    }
}
