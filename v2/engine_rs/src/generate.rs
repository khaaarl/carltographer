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

        // Choose action: 0=add, 1=move, 2=delete
        if !has_features && !has_catalog {
            continue;
        }
        let action = if !has_features {
            0u32
        } else if !has_catalog {
            rng.next_int(0, 1) + 1
        } else {
            rng.next_int(0, 2)
        };

        match action {
            0 => {
                let ti = rng.next_int(
                    0,
                    catalog_features.len() as u32 - 1,
                ) as usize;
                let template = catalog_features[ti];
                let new_feat = instantiate_feature(template, nid);
                let x = rng.next_float()
                    * params.table_width_inches
                    - params.table_width_inches / 2.0;
                let z = rng.next_float()
                    * params.table_depth_inches
                    - params.table_depth_inches / 2.0;
                let rot = rng.next_float() * 360.0;
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
                let x = rng.next_float()
                    * params.table_width_inches
                    - params.table_width_inches / 2.0;
                let z = rng.next_float()
                    * params.table_depth_inches
                    - params.table_depth_inches / 2.0;
                let rot = rng.next_float() * 360.0;
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
        };
        let result = generate(&params);
        assert!(result.layout.placed_features.is_empty());
        assert_eq!(result.steps_completed, 50);
    }
}
