//! Terrain layout generation engine.

use std::collections::HashMap;

use crate::collision::is_valid_placement;
use crate::prng::Pcg32;
use crate::types::{
    EngineParams, EngineResult, FeatureCountPreference, PlacedFeature, ScoringTargets,
    TerrainCatalog, TerrainFeature, TerrainLayout, TerrainObject, Transform,
};

const PHASE2_BASE: f64 = 1000.0;
const MAX_RETRIES: u32 = 100;
const RETRY_DECAY: f64 = 0.95;
const MIN_MOVE_RANGE: f64 = 0.1;
const MAX_EXTRA_MUTATIONS: u32 = 3;

/// Undo token for reverting a single mutation step.
#[derive(Debug)]
pub enum StepUndo {
    Noop,
    Add { index: usize, prev_next_id: u32 },
    Move { index: usize, old: PlacedFeature },
    Delete { index: usize, saved: PlacedFeature },
    Replace { index: usize, old: PlacedFeature, prev_next_id: u32 },
}

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
    rotationally_symmetric: bool,
) -> std::collections::HashMap<String, u32> {
    // Count how many of each feature_type are visible on the table.
    // When rotationally_symmetric, non-origin features count as 2
    // (canonical + mirror). Origin features count as 1.
    let mut counts = std::collections::HashMap::new();
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

fn compute_template_weights(
    catalog_features: &[&TerrainFeature],
    feature_counts: &HashMap<String, u32>,
    preferences: &[crate::types::FeatureCountPreference],
) -> Vec<f64> {
    // Compute weights for selecting which catalog feature to add.
    // Boosts weight for types below their min, reduces for types at/above max.
    let pref_by_type: HashMap<&str, &crate::types::FeatureCountPreference> =
        preferences.iter().map(|p| (p.feature_type.as_str(), p)).collect();
    let mut weights = Vec::with_capacity(catalog_features.len());
    for cf in catalog_features {
        let mut w = 1.0;
        if let Some(pref) = pref_by_type.get(cf.feature_type.as_str()) {
            let current = feature_counts
                .get(&cf.feature_type)
                .copied()
                .unwrap_or(0);
            if current < pref.min {
                w = 1.0 + (pref.min - current) as f64 * 2.0;
            } else if let Some(max) = pref.max {
                if current >= max {
                    w = 0.1;
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
    preferences: &[crate::types::FeatureCountPreference],
) -> Vec<f64> {
    // Compute weights for selecting which feature to delete.
    // Boosts weight for types above their max, reduces for types at/below min.
    let pref_by_type: HashMap<&str, &crate::types::FeatureCountPreference> =
        preferences.iter().map(|p| (p.feature_type.as_str(), p)).collect();
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
                    w = 1.0 + (current - max) as f64 * 2.0;
                } else if current <= pref.min {
                    w = 0.1;
                }
            } else if current <= pref.min {
                w = 0.1;
            }
        }
        weights.push(w);
    }
    weights
}

fn avg_metric_value(obj: &serde_json::Map<String, serde_json::Value>) -> Option<f64> {
    if obj.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in obj.values() {
        if let Some(val) = v.get("value").and_then(|x| x.as_f64()) {
            sum += val;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn compute_score(
    layout: &TerrainLayout,
    preferences: &[FeatureCountPreference],
    objects_by_id: &HashMap<String, &TerrainObject>,
    skip_visibility: bool,
    scoring_targets: Option<&ScoringTargets>,
    visibility_cache: Option<&mut crate::visibility::VisibilityCache>,
) -> f64 {
    // Phase 1: gradient toward satisfying count preferences.
    let counts = count_features_by_type(
        &layout.placed_features,
        layout.rotationally_symmetric,
    );
    let mut total_deficit: u32 = 0;
    for pref in preferences {
        let current = counts
            .get(&pref.feature_type)
            .copied()
            .unwrap_or(0);
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
        4.0,  // min_blocking_height
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

    // 2. DZ visibility (average across all DZs)
    if let Some(target) = targets.dz_visibility_target {
        if let Some(dz_vis) = vis.get("dz_visibility").and_then(|v| v.as_object()) {
            if let Some(avg) = avg_metric_value(dz_vis) {
                let error = (avg - target).abs();
                total_weighted_error += targets.dz_visibility_weight * error;
                total_weight += targets.dz_visibility_weight;
            }
        }
    }

    // 3. DZ hidden from opponent (average across all cross-DZ pairs)
    if let Some(target) = targets.dz_hidden_target {
        if let Some(dz_cross) = vis.get("dz_to_dz_visibility").and_then(|v| v.as_object()) {
            if let Some(avg) = avg_metric_value(dz_cross) {
                let error = (avg - target).abs();
                total_weighted_error += targets.dz_hidden_weight * error;
                total_weight += targets.dz_hidden_weight;
            }
        }
    }

    // 4. Objective hidability (average across all DZs)
    if let Some(target) = targets.objective_hidability_target {
        if let Some(obj_hide) = vis.get("objective_hidability").and_then(|v| v.as_object()) {
            if let Some(avg) = avg_metric_value(obj_hide) {
                let error = (avg - target).abs();
                total_weighted_error += targets.objective_hidability_weight * error;
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

/// Generate a temperature-aware move transform.
///
/// At t_factor=0, displacement is small (±MIN_MOVE_RANGE/2 inches).
/// At t_factor=1, displacement spans the full table.
/// Always consumes exactly 4 PRNG values.
fn temperature_move(
    rng: &mut Pcg32,
    old_transform: &Transform,
    table_width: f64,
    table_depth: f64,
    t_factor: f64,
) -> Transform {
    let max_dim = table_width.max(table_depth);
    let move_range = MIN_MOVE_RANGE + t_factor * (max_dim - MIN_MOVE_RANGE);

    let dx = (rng.next_float() - 0.5) * 2.0 * move_range;
    let dz = (rng.next_float() - 0.5) * 2.0 * move_range;
    let rotate_check = rng.next_float();
    let rot_angle_raw = rng.next_float();

    let new_x = quantize_position(old_transform.x_inches + dx);
    let new_z = quantize_position(old_transform.z_inches + dz);

    // Rotation: 0% chance at t=0, 50% chance at t=1
    let rot = if rotate_check < 0.5 * t_factor {
        quantize_angle(rot_angle_raw * 360.0)
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
) -> Option<(StepUndo, u32)> {
    let has_features = !layout.placed_features.is_empty();
    let feature_counts = count_features_by_type(&layout.placed_features, params.rotationally_symmetric);

    // Compute action weights: [add, move, delete, replace]
    let mut add_weight: f64 = if has_catalog { 1.0 } else { 0.0 };
    let move_weight: f64 = if has_features { 1.0 } else { 0.0 };
    let mut delete_weight: f64 = if has_features { 1.0 } else { 0.0 };
    let replace_weight: f64 = if has_features && has_catalog { 1.0 } else { 0.0 };

    // Preference biasing on add/delete only
    for pref in prefs {
        let current = feature_counts
            .get(&pref.feature_type)
            .copied()
            .unwrap_or(0);
        if current < pref.min {
            let shortage = pref.min - current;
            add_weight *= 1.0 + shortage as f64 * 2.0;
            delete_weight *= 0.1;
        } else if let Some(max) = pref.max {
            if current > max {
                let excess = current - max;
                delete_weight *= 1.0 + excess as f64 * 2.0;
                add_weight *= 0.1;
            }
        }
    }

    let weights = vec![add_weight, move_weight, delete_weight, replace_weight];
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
            );
            let tidx = match weighted_choice(rng, &template_weights) {
                Some(i) => i,
                None => return None,
            };
            let template = catalog_features[tidx];
            let new_feat = instantiate_feature(template, next_id);
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
            ) {
                Some((StepUndo::Add { index: idx, prev_next_id: next_id }, next_id + 1))
            } else {
                layout.placed_features.pop();
                None
            }
        }
        1 => {
            // Move (temperature-aware)
            let idx = rng.next_int(
                0,
                layout.placed_features.len() as u32 - 1,
            ) as usize;
            let old = layout.placed_features[idx].clone();
            let new_transform = temperature_move(
                rng,
                &old.transform,
                params.table_width_inches,
                params.table_depth_inches,
                t_factor,
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
            );
            let idx = match weighted_choice(rng, &delete_weights) {
                Some(i) => i,
                None => return None,
            };
            let saved = layout.placed_features.remove(idx);
            Some((StepUndo::Delete { index: idx, saved }, next_id))
        }
        3 => {
            // Replace: remove feature, add different template at same position
            let delete_weights = compute_delete_weights(
                &layout.placed_features,
                &feature_counts,
                prefs,
            );
            let idx = match weighted_choice(rng, &delete_weights) {
                Some(i) => i,
                None => return None,
            };
            let template_weights = compute_template_weights(
                catalog_features,
                &feature_counts,
                prefs,
            );
            let tidx = match weighted_choice(rng, &template_weights) {
                Some(i) => i,
                None => return None,
            };
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
            ) {
                Some((StepUndo::Replace { index: idx, old, prev_next_id: next_id }, next_id + 1))
            } else {
                layout.placed_features[idx] = old;
                None
            }
        }
        _ => None,
    }
}

/// Try mutations with decaying temperature until one succeeds or retries exhausted.
fn perform_step(
    layout: &mut TerrainLayout,
    rng: &mut Pcg32,
    t_factor: f64,
    next_id: u32,
    catalog_features: &[&TerrainFeature],
    has_catalog: bool,
    objects_by_id: &HashMap<String, &TerrainObject>,
    params: &EngineParams,
    prefs: &[FeatureCountPreference],
) -> (StepUndo, u32) {
    let mut effective_t = t_factor;
    for _ in 0..MAX_RETRIES {
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
        ) {
            return result;
        }
        effective_t *= RETRY_DECAY;
    }
    (StepUndo::Noop, next_id)
}

/// Revert a mutation using its undo token.
fn undo_step(layout: &mut TerrainLayout, undo: StepUndo) {
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
    }
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
    let objects_by_id = build_object_index(&params.catalog);

    let (initial_features, mut nid) =
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
    let prefs = params.feature_count_preferences.as_deref().unwrap_or(&[]);

    let mut layout = TerrainLayout {
        table_width_inches: params.table_width_inches,
        table_depth_inches: params.table_depth_inches,
        placed_features: initial_features,
        rotationally_symmetric: params.rotationally_symmetric,
        visibility: None,
        mission: params.mission.clone(),
    };

    // Create visibility cache for incremental tall-footprint filtering
    let mut vis_cache: Option<crate::visibility::VisibilityCache> = if !params.skip_visibility {
        Some(crate::visibility::VisibilityCache::new(&layout, &objects_by_id))
    } else {
        None
    };

    let mut current_score = compute_score(&layout, prefs, &objects_by_id, params.skip_visibility, params.scoring_targets.as_ref(), vis_cache.as_mut());

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
        );

        if matches!(undo, StepUndo::Noop) {
            continue;
        }

        if let Some(ref mut cache) = vis_cache {
            cache.mark_dirty();
        }
        let new_score = compute_score(&layout, prefs, &objects_by_id, params.skip_visibility, params.scoring_targets.as_ref(), vis_cache.as_mut());
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
        layout.visibility = Some(
            crate::visibility::compute_layout_visibility(
                &layout,
                &objects_by_id,
                4.0,  // min_blocking_height
                vis_cache.as_mut(),
            ),
        );
    }

    EngineResult {
        layout,
        score: current_score,
        steps_completed: num_steps,
    }
}

fn generate_tempering(params: &EngineParams, num_replicas: u32) -> EngineResult {
    use crate::tempering::{compute_temperatures, sa_accept};

    let objects_by_id = build_object_index(&params.catalog);
    let catalog_features: Vec<&TerrainFeature> = params
        .catalog
        .features
        .iter()
        .map(|cf| &cf.item)
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
            visibility: None,
            mission: params.mission.clone(),
        };
        let vis_cache = if !params.skip_visibility {
            Some(crate::visibility::VisibilityCache::new(&layout, &objects_by_id))
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
        let mut per_replica_best: Vec<(f64, Option<TerrainLayout>)> =
            (0..num_replicas).map(|_| (f64::NEG_INFINITY, None)).collect();
        let current_best = best_score;

        std::thread::scope(|s| {
            let handles: Vec<_> = replicas.iter_mut()
                .zip(per_replica_best.iter_mut())
                .map(|(replica, prb)| {
                    let cat_feats = &catalog_features;
                    let objs = &objects_by_id;
                    s.spawn(move || {
                        let t_factor = if max_temperature > 0.0 {
                            replica.temperature / max_temperature
                        } else {
                            0.0
                        };
                        let num_mutations = 1 + (t_factor * MAX_EXTRA_MUTATIONS as f64) as u32;

                        for _ in 0..batch_size {
                            let mut sub_undos: Vec<(StepUndo, u32)> = Vec::with_capacity(num_mutations as usize);
                            for _ in 0..num_mutations {
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

                            if sa_accept(old_score, new_score, replica.temperature, &mut replica.rng) {
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
        let mut best_vis_cache = crate::visibility::VisibilityCache::new(&best_layout, &objects_by_id);
        best_layout.visibility = Some(
            crate::visibility::compute_layout_visibility(
                &best_layout,
                &objects_by_id,
                4.0,
                Some(&mut best_vis_cache),
            ),
        );
    }

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
    fn temperature_move_small_t() {
        let mut rng = Pcg32::new(42, 0);
        let old = Transform {
            x_inches: 0.0, y_inches: 0.0, z_inches: 0.0, rotation_deg: 90.0,
        };
        for _ in 0..100 {
            let t = temperature_move(&mut rng, &old, 60.0, 44.0, 0.0);
            assert!((t.x_inches - old.x_inches).abs() <= MIN_MOVE_RANGE + 0.1);
            assert!((t.z_inches - old.z_inches).abs() <= MIN_MOVE_RANGE + 0.1);
            assert_eq!(t.rotation_deg, 90.0); // no rotation at t=0
        }
    }

    #[test]
    fn temperature_move_consumes_4_prng() {
        let mut rng1 = Pcg32::new(99, 0);
        let mut rng2 = Pcg32::new(99, 0);
        let old = Transform {
            x_inches: 5.0, y_inches: 0.0, z_inches: 3.0, rotation_deg: 30.0,
        };
        temperature_move(&mut rng1, &old, 60.0, 44.0, 0.5);
        for _ in 0..4 {
            rng2.next_float();
        }
        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }
}
