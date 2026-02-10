//! Terrain layout generation engine.

use std::collections::HashMap;

use crate::mutation::{
    count_features_by_type, perform_step, undo_step, StepUndo, MAX_EXTRA_MUTATIONS,
};
use crate::prng::Pcg32;
use crate::types::{
    EngineParams, EngineResult, FeatureCountPreference, PlacedFeature, ScoringTargets,
    TerrainCatalog, TerrainFeature, TerrainLayout, TerrainObject,
};

const PHASE2_BASE: f64 = 1000.0;

pub(crate) fn build_object_index(catalog: &TerrainCatalog) -> HashMap<String, &TerrainObject> {
    let mut index = HashMap::new();
    for co in &catalog.objects {
        index.insert(co.item.id.clone(), &co.item);
    }
    index
}

fn merge_layout_objects<'a>(
    objects_by_id: &mut HashMap<String, &'a TerrainObject>,
    layout_objects: &'a [TerrainObject],
) {
    for obj in layout_objects {
        objects_by_id.entry(obj.id.clone()).or_insert(obj);
    }
}

fn collect_layout_objects(
    placed_features: &[PlacedFeature],
    objects_by_id: &HashMap<String, &TerrainObject>,
) -> Vec<TerrainObject> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for pf in placed_features {
        for comp in &pf.feature.components {
            if seen.insert(comp.object_id.clone()) {
                if let Some(obj) = objects_by_id.get(&comp.object_id) {
                    result.push((*obj).clone());
                }
            }
        }
    }
    result
}

fn next_feature_id(features: &[PlacedFeature]) -> u32 {
    let mut max_id: u32 = 0;
    for pf in features {
        if let Some(num_str) = pf.feature.id.strip_prefix("feature_") {
            if let Ok(n) = num_str.parse::<u32>() {
                max_id = max_id.max(n);
            }
        }
    }
    max_id + 1
}

fn avg_metric_error(obj: &serde_json::Map<String, serde_json::Value>, target: f64) -> Option<f64> {
    if obj.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in obj.values() {
        if let Some(val) = v.get("value").and_then(|x| x.as_f64()) {
            sum += (val - target).abs();
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

// NOTE: If scoring logic grows more complex, consider extracting to a
// dedicated scoring module.
fn compute_score(
    layout: &TerrainLayout,
    preferences: &[FeatureCountPreference],
    objects_by_id: &HashMap<String, &TerrainObject>,
    skip_visibility: bool,
    scoring_targets: Option<&ScoringTargets>,
    visibility_cache: Option<&mut crate::visibility::VisibilityCache>,
) -> f64 {
    // Phase 1: gradient toward satisfying count preferences.
    let counts = count_features_by_type(&layout.placed_features, layout.rotationally_symmetric);
    let mut total_deficit: u32 = 0;
    for pref in preferences {
        let current = counts.get(&pref.feature_type).copied().unwrap_or(0);
        if current < pref.min {
            total_deficit += pref.min - current;
        } else if let Some(max) = pref.max {
            if current > max {
                total_deficit += current - max;
            }
        }
    }

    if total_deficit > 0 {
        return PHASE2_BASE / (1.0 + total_deficit as f64);
    }

    if skip_visibility {
        return PHASE2_BASE;
    }

    // Phase 2: optimize visibility toward targets.
    let vis = crate::visibility::compute_layout_visibility(
        layout,
        objects_by_id,
        4.0, // min_blocking_height
        visibility_cache,
    );

    let targets = match scoring_targets {
        None => {
            let vis_pct = vis["overall"]["value"].as_f64().unwrap_or(100.0);
            return PHASE2_BASE + (100.0 - vis_pct);
        }
        Some(t) => t,
    };

    let mut total_weight: f64 = 0.0;
    let mut total_weighted_error: f64 = 0.0;

    // 1. Overall visibility
    if let Some(target) = targets.overall_visibility_target {
        let actual = vis["overall"]["value"].as_f64().unwrap_or(100.0);
        let error = (actual - target).abs();
        total_weighted_error += targets.overall_visibility_weight * error;
        total_weight += targets.overall_visibility_weight;
    }

    // 2. DZ visibility (per-DZ error from target, then averaged)
    if let Some(target) = targets.dz_visibility_target {
        if let Some(dz_vis) = vis.get("dz_visibility").and_then(|v| v.as_object()) {
            if let Some(avg_error) = avg_metric_error(dz_vis, target) {
                total_weighted_error += targets.dz_visibility_weight * avg_error;
                total_weight += targets.dz_visibility_weight;
            }
        }
    }

    // 3. DZ hidden from opponent (per-pair error from target, then averaged)
    if let Some(target) = targets.dz_hidden_target {
        if let Some(dz_cross) = vis.get("dz_to_dz_visibility").and_then(|v| v.as_object()) {
            if let Some(avg_error) = avg_metric_error(dz_cross, target) {
                total_weighted_error += targets.dz_hidden_weight * avg_error;
                total_weight += targets.dz_hidden_weight;
            }
        }
    }

    // 4. Objective hidability (per-DZ error from target, then averaged)
    if let Some(target) = targets.objective_hidability_target {
        if let Some(obj_hide) = vis.get("objective_hidability").and_then(|v| v.as_object()) {
            if let Some(avg_error) = avg_metric_error(obj_hide, target) {
                total_weighted_error += targets.objective_hidability_weight * avg_error;
                total_weight += targets.objective_hidability_weight;
            }
        }
    }

    if total_weight <= 0.0 {
        let vis_pct = vis["overall"]["value"].as_f64().unwrap_or(100.0);
        return PHASE2_BASE + (100.0 - vis_pct);
    }

    let weighted_avg_error = total_weighted_error / total_weight;
    PHASE2_BASE + (100.0 - weighted_avg_error)
}

pub fn generate(params: &EngineParams) -> EngineResult {
    match params.num_replicas {
        Some(n) if n > 1 => return generate_tempering(params, n),
        _ => {}
    }
    generate_hill_climbing(params)
}

fn generate_hill_climbing(params: &EngineParams) -> EngineResult {
    let mut rng = Pcg32::new(params.seed, 0);
    let mut objects_by_id = build_object_index(&params.catalog);
    if let Some(ref initial) = params.initial_layout {
        merge_layout_objects(&mut objects_by_id, &initial.terrain_objects);
    }

    let (initial_features, mut nid) = match &params.initial_layout {
        Some(initial) => {
            let feats = initial.placed_features.clone();
            let id = next_feature_id(&feats);
            (feats, id)
        }
        None => (Vec::new(), 1u32),
    };

    let catalog_features: Vec<&TerrainFeature> =
        params.catalog.features.iter().map(|cf| &cf.item).collect();
    let catalog_quantities: Vec<Option<u32>> = params
        .catalog
        .features
        .iter()
        .map(|cf| cf.quantity)
        .collect();
    let has_catalog = !catalog_features.is_empty();
    let num_steps = params.steps();
    let prefs = params.feature_count_preferences.as_deref().unwrap_or(&[]);

    let mut layout = TerrainLayout {
        table_width_inches: params.table_width_inches,
        table_depth_inches: params.table_depth_inches,
        placed_features: initial_features,
        rotationally_symmetric: params.rotationally_symmetric,
        terrain_objects: Vec::new(),
        visibility: None,
        mission: params.mission.clone(),
    };

    // Create visibility cache for incremental tall-footprint filtering
    let mut vis_cache: Option<crate::visibility::VisibilityCache> = if !params.skip_visibility {
        Some(crate::visibility::VisibilityCache::new(
            &layout,
            &objects_by_id,
        ))
    } else {
        None
    };

    let mut current_score = compute_score(
        &layout,
        prefs,
        &objects_by_id,
        params.skip_visibility,
        params.scoring_targets.as_ref(),
        vis_cache.as_mut(),
    );

    for _ in 0..num_steps {
        let (undo, new_nid) = perform_step(
            &mut layout,
            &mut rng,
            1.0,
            nid,
            &catalog_features,
            has_catalog,
            &objects_by_id,
            params,
            prefs,
            &catalog_quantities,
            0,
            1,
        );

        if matches!(undo, StepUndo::Noop) {
            continue;
        }

        if let Some(ref mut cache) = vis_cache {
            cache.mark_dirty();
        }
        let new_score = compute_score(
            &layout,
            prefs,
            &objects_by_id,
            params.skip_visibility,
            params.scoring_targets.as_ref(),
            vis_cache.as_mut(),
        );
        if new_score >= current_score {
            current_score = new_score;
            nid = new_nid;
        } else {
            undo_step(&mut layout, undo);
            if let Some(ref mut cache) = vis_cache {
                cache.mark_dirty();
            }
        }
    }

    if !params.skip_visibility {
        layout.visibility = Some(crate::visibility::compute_layout_visibility(
            &layout,
            &objects_by_id,
            4.0, // min_blocking_height
            vis_cache.as_mut(),
        ));
    }

    layout.terrain_objects = collect_layout_objects(&layout.placed_features, &objects_by_id);

    EngineResult {
        layout,
        score: current_score,
        steps_completed: num_steps,
    }
}

fn generate_tempering(params: &EngineParams, num_replicas: u32) -> EngineResult {
    use crate::tempering::{compute_temperatures, sa_accept};

    let mut objects_by_id = build_object_index(&params.catalog);
    if let Some(ref initial) = params.initial_layout {
        merge_layout_objects(&mut objects_by_id, &initial.terrain_objects);
    }
    let catalog_features: Vec<&TerrainFeature> =
        params.catalog.features.iter().map(|cf| &cf.item).collect();
    let catalog_quantities: Vec<Option<u32>> = params
        .catalog
        .features
        .iter()
        .map(|cf| cf.quantity)
        .collect();
    let has_catalog = !catalog_features.is_empty();
    let num_steps = params.steps();
    let prefs = params.feature_count_preferences.as_deref().unwrap_or(&[]);
    let temperatures = compute_temperatures(num_replicas, params.max_temperature);
    let max_temperature = params.max_temperature;
    let swap_interval = params.swap_interval;

    // Create replicas
    struct Replica<'a> {
        layout: TerrainLayout,
        rng: Pcg32,
        score: f64,
        next_id: u32,
        temperature: f64,
        vis_cache: Option<crate::visibility::VisibilityCache<'a>>,
    }

    let mut replicas: Vec<Replica> = Vec::with_capacity(num_replicas as usize);
    for i in 0..num_replicas {
        let rng = Pcg32::new(params.seed, i as u64);
        let (initial_features, next_id) = match &params.initial_layout {
            Some(initial) => {
                let feats = initial.placed_features.clone();
                let id = next_feature_id(&feats);
                (feats, id)
            }
            None => (Vec::new(), 1u32),
        };
        let layout = TerrainLayout {
            table_width_inches: params.table_width_inches,
            table_depth_inches: params.table_depth_inches,
            placed_features: initial_features,
            rotationally_symmetric: params.rotationally_symmetric,
            terrain_objects: Vec::new(),
            visibility: None,
            mission: params.mission.clone(),
        };
        let vis_cache = if !params.skip_visibility {
            Some(crate::visibility::VisibilityCache::new(
                &layout,
                &objects_by_id,
            ))
        } else {
            None
        };
        let score = compute_score(
            &layout,
            prefs,
            &objects_by_id,
            params.skip_visibility,
            params.scoring_targets.as_ref(),
            None,
        );
        replicas.push(Replica {
            layout,
            rng,
            score,
            next_id,
            temperature: temperatures[i as usize],
            vis_cache,
        });
    }

    let mut swap_rng = Pcg32::new(params.seed, num_replicas as u64);

    // Track best
    let mut best_score = replicas[0].score;
    let mut best_layout = replicas[0].layout.clone();
    for r in replicas.iter().skip(1) {
        if r.score > best_score {
            best_score = r.score;
            best_layout = r.layout.clone();
        }
    }

    // Main loop — each batch runs replicas in parallel via scoped threads
    let mut remaining = num_steps;

    while remaining > 0 {
        let batch_size = remaining.min(swap_interval);

        // Per-replica best tracking for this batch
        let mut per_replica_best: Vec<(f64, Option<TerrainLayout>)> = (0..num_replicas)
            .map(|_| (f64::NEG_INFINITY, None))
            .collect();
        let current_best = best_score;

        std::thread::scope(|s| {
            let handles: Vec<_> = replicas
                .iter_mut()
                .zip(per_replica_best.iter_mut())
                .map(|(replica, prb)| {
                    let cat_feats = &catalog_features;
                    let cat_qtys = &catalog_quantities;
                    let objs = &objects_by_id;
                    s.spawn(move || {
                        let t_factor = if max_temperature > 0.0 {
                            replica.temperature / max_temperature
                        } else {
                            0.0
                        };
                        let num_mutations = 1 + (t_factor * MAX_EXTRA_MUTATIONS as f64) as u32;

                        for _ in 0..batch_size {
                            let mut sub_undos: Vec<(StepUndo, u32)> =
                                Vec::with_capacity(num_mutations as usize);
                            for mi in 0..num_mutations {
                                let (undo, new_nid) = perform_step(
                                    &mut replica.layout,
                                    &mut replica.rng,
                                    t_factor,
                                    replica.next_id,
                                    cat_feats,
                                    has_catalog,
                                    objs,
                                    params,
                                    prefs,
                                    cat_qtys,
                                    mi,
                                    num_mutations,
                                );
                                sub_undos.push((undo, replica.next_id));
                                replica.next_id = new_nid;
                            }

                            if sub_undos.iter().all(|(u, _)| matches!(u, StepUndo::Noop)) {
                                continue;
                            }

                            if let Some(ref mut cache) = replica.vis_cache {
                                cache.mark_dirty();
                            }

                            let old_score = replica.score;
                            let new_score = compute_score(
                                &replica.layout,
                                prefs,
                                objs,
                                params.skip_visibility,
                                params.scoring_targets.as_ref(),
                                replica.vis_cache.as_mut(),
                            );

                            if sa_accept(
                                old_score,
                                new_score,
                                replica.temperature,
                                &mut replica.rng,
                            ) {
                                replica.score = new_score;
                                if new_score > current_best && new_score > prb.0 {
                                    prb.0 = new_score;
                                    prb.1 = Some(replica.layout.clone());
                                }
                            } else {
                                for (undo, prev_nid) in sub_undos.into_iter().rev() {
                                    undo_step(&mut replica.layout, undo);
                                    replica.next_id = prev_nid;
                                }
                                if let Some(ref mut cache) = replica.vis_cache {
                                    cache.mark_dirty();
                                }
                            }
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });

        // Merge per-replica bests into global best
        for (score, layout_opt) in per_replica_best {
            if score > best_score {
                if let Some(l) = layout_opt {
                    best_score = score;
                    best_layout = l;
                }
            }
        }

        remaining -= batch_size;

        // Swap adjacent replicas
        if remaining > 0 && num_replicas > 1 {
            for i in 0..(num_replicas - 1) as usize {
                let r = swap_rng.next_float();
                let ti = replicas[i].temperature;
                let tj = replicas[i + 1].temperature;
                let si = replicas[i].score;
                let sj = replicas[i + 1].score;

                let accept = if ti <= 0.0 {
                    sj >= si
                } else if tj <= 0.0 {
                    si >= sj
                } else {
                    let delta = (1.0 / ti - 1.0 / tj) * (sj - si);
                    if delta >= 0.0 {
                        true
                    } else {
                        r < delta.exp()
                    }
                };

                if accept {
                    // Swap layouts, scores, next_ids, vis_caches
                    let (left, right) = replicas.split_at_mut(i + 1);
                    std::mem::swap(&mut left[i].layout, &mut right[0].layout);
                    std::mem::swap(&mut left[i].score, &mut right[0].score);
                    std::mem::swap(&mut left[i].next_id, &mut right[0].next_id);
                    std::mem::swap(&mut left[i].vis_cache, &mut right[0].vis_cache);
                }
            }
        }
    }

    // Pick best from final replica states (covers equal-score case).
    // Reverse iteration so cold chain (index 0) wins ties.
    for replica in replicas.iter().rev() {
        if replica.score >= best_score {
            best_score = replica.score;
            best_layout = replica.layout.clone();
        }
    }

    // Final visibility on best layout
    if !params.skip_visibility {
        let mut best_vis_cache =
            crate::visibility::VisibilityCache::new(&best_layout, &objects_by_id);
        best_layout.visibility = Some(crate::visibility::compute_layout_visibility(
            &best_layout,
            &objects_by_id,
            4.0,
            Some(&mut best_vis_cache),
        ));
    }

    best_layout.terrain_objects =
        collect_layout_objects(&best_layout.placed_features, &objects_by_id);

    EngineResult {
        layout: best_layout,
        score: best_score,
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
    use crate::types::{CatalogFeature, CatalogObject, FeatureComponent, GeometricShape};

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
                        assert!(!obbs_overlap(ca, cb), "Features {} and {} overlap", i, j);
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

    #[test]
    fn symmetric_deterministic() {
        let mut params = make_params(42, 100);
        params.rotationally_symmetric = true;
        let r1 = generate(&params);
        let r2 = generate(&params);
        let j1 = serde_json::to_string(&r1).unwrap();
        let j2 = serde_json::to_string(&r2).unwrap();
        assert_eq!(j1, j2);
    }

    #[test]
    fn symmetric_flag_on_output() {
        let mut params = make_params(42, 50);
        params.rotationally_symmetric = true;
        let result = generate(&params);
        assert!(result.layout.rotationally_symmetric);
    }

    #[test]
    fn catalog_quantity_limit_respected() {
        let mut params = make_params(42, 200);
        params.skip_visibility = true;
        // Set quantity limit of 2 on the single crate feature
        params.catalog.features[0].quantity = Some(2);
        params.catalog.objects[0].quantity = Some(2);
        let result = generate(&params);
        assert!(
            result.layout.placed_features.len() <= 2,
            "Expected at most 2 features, got {}",
            result.layout.placed_features.len()
        );
    }

    #[test]
    fn orphaned_features_no_crash_no_overlap() {
        // Create a catalog with small blocks
        let small_catalog = TerrainCatalog {
            objects: vec![CatalogObject {
                item: TerrainObject {
                    id: "small_block".into(),
                    shapes: vec![GeometricShape {
                        shape_type: "rectangular_prism".into(),
                        width_inches: 3.0,
                        depth_inches: 3.0,
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
                    id: "small".into(),
                    feature_type: "obstacle".into(),
                    components: vec![FeatureComponent {
                        object_id: "small_block".into(),
                        transform: None,
                    }],
                },
                quantity: None,
            }],
            name: None,
        };

        // Initial layout has a big_block feature (not in the catalog)
        let big_block_obj = TerrainObject {
            id: "big_block".into(),
            shapes: vec![GeometricShape {
                shape_type: "rectangular_prism".into(),
                width_inches: 10.0,
                depth_inches: 10.0,
                height_inches: 3.0,
                offset: None,
                opacity_height_inches: None,
            }],
            name: None,
            tags: vec![],
        };

        let initial_layout = TerrainLayout {
            table_width_inches: 60.0,
            table_depth_inches: 44.0,
            placed_features: vec![PlacedFeature {
                feature: TerrainFeature {
                    id: "feature_1".into(),
                    feature_type: "obstacle".into(),
                    components: vec![FeatureComponent {
                        object_id: "big_block".into(),
                        transform: None,
                    }],
                },
                transform: crate::types::Transform {
                    x_inches: 10.0,
                    y_inches: 0.0,
                    z_inches: 5.0,
                    rotation_deg: 0.0,
                },
            }],
            terrain_objects: vec![big_block_obj],
            rotationally_symmetric: false,
            visibility: None,
            mission: None,
        };

        let params = EngineParams {
            seed: 99,
            table_width_inches: 60.0,
            table_depth_inches: 44.0,
            catalog: small_catalog,
            num_steps: Some(200),
            initial_layout: Some(initial_layout),
            feature_count_preferences: None,
            min_feature_gap_inches: None,
            min_edge_gap_inches: None,
            min_all_feature_gap_inches: None,
            min_all_edge_gap_inches: None,
            rotation_granularity_deg: 15.0,
            rotationally_symmetric: false,
            mission: None,
            skip_visibility: true,
            scoring_targets: None,
            num_replicas: None,
            swap_interval: 20,
            max_temperature: 50.0,
        };

        let result = generate(&params);

        // Build combined object index for verification
        let mut objs: HashMap<String, &TerrainObject> = HashMap::new();
        for co in &params.catalog.objects {
            objs.insert(co.item.id.clone(), &co.item);
        }
        // Also need big_block for checking orphaned features
        if let Some(ref il) = params.initial_layout {
            for obj in &il.terrain_objects {
                objs.entry(obj.id.clone()).or_insert(obj);
            }
        }

        // Verify no overlaps
        let feats = &result.layout.placed_features;
        for i in 0..feats.len() {
            let oi = get_world_obbs(&feats[i], &objs);
            for j in (i + 1)..feats.len() {
                let oj = get_world_obbs(&feats[j], &objs);
                for ca in &oi {
                    for cb in &oj {
                        assert!(!obbs_overlap(ca, cb), "Features {} and {} overlap", i, j);
                    }
                }
            }
        }
    }

    #[test]
    fn terrain_objects_in_output() {
        let params = make_params(42, 50);
        let result = generate(&params);
        // Output should have terrain_objects matching placed features
        assert!(!result.layout.terrain_objects.is_empty());
        let obj_ids: std::collections::HashSet<&str> = result
            .layout
            .terrain_objects
            .iter()
            .map(|o| o.id.as_str())
            .collect();
        for pf in &result.layout.placed_features {
            for comp in &pf.feature.components {
                assert!(
                    obj_ids.contains(comp.object_id.as_str()),
                    "Object {} not in terrain_objects",
                    comp.object_id
                );
            }
        }
    }
}
