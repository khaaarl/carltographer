"""Terrain layout generation engine."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .collision import is_valid_placement
from .prng import PCG32
from .tempering import compute_temperatures, sa_accept
from .types import (
    EngineParams,
    EngineResult,
    FeatureCountPreference,
    PlacedFeature,
    ScoringTargets,
    TerrainCatalog,
    TerrainFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)
from .visibility import VisibilityCache, compute_layout_visibility

PHASE2_BASE = 1000.0
MAX_RETRIES = 100
RETRY_DECAY = 0.95
MIN_MOVE_RANGE = 0.1
MAX_EXTRA_MUTATIONS = 3


@dataclass
class StepUndo:
    action: str  # "noop", "add", "move", "delete", "replace"
    index: int = -1
    old_feature: PlacedFeature | None = None
    prev_next_id: int = 0


def _quantize_position(value: float) -> float:
    """Quantize position to nearest 0.1 inch."""
    return round(value / 0.1) * 0.1


def _quantize_angle(value: float) -> float:
    """Quantize angle to nearest 15 degrees."""
    return round(value / 15.0) * 15.0


def _count_features_by_type(layout: TerrainLayout) -> dict[str, int]:
    """Count how many of each feature_type are visible on the table.

    When rotationally_symmetric, non-origin features count as 2
    (canonical + mirror). Origin features count as 1.
    """
    counts: dict[str, int] = {}
    for pf in layout.placed_features:
        ft = pf.feature.feature_type
        if layout.rotationally_symmetric and (
            pf.transform.x != 0.0 or pf.transform.z != 0.0
        ):
            counts[ft] = counts.get(ft, 0) + 2
        else:
            counts[ft] = counts.get(ft, 0) + 1
    return counts


def _weighted_choice(rng: PCG32, weights: list[float]) -> int:
    """Select index with probability proportional to weights.
    Uses PCG32 for determinism. Returns -1 if all weights are 0."""
    total = sum(weights)
    if total <= 0:
        return -1
    r = rng.next_float() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return i
    return len(weights) - 1


def _compute_template_weights(
    catalog_features: list[TerrainFeature],
    feature_counts: dict[str, int],
    preferences: list,
) -> list[float]:
    """Compute weights for selecting which catalog feature to add.

    Boosts weight for types below their min, reduces for types at/above max.
    """
    pref_by_type = {p.feature_type: p for p in preferences}
    weights = []
    for cf in catalog_features:
        pref = pref_by_type.get(cf.feature_type)
        w = 1.0
        if pref is not None:
            current = feature_counts.get(cf.feature_type, 0)
            if current < pref.min:
                w = 1.0 + (pref.min - current) * 2.0
            elif pref.max is not None and current >= pref.max:
                w = 0.1
        weights.append(w)
    return weights


def _compute_delete_weights(
    features: list[PlacedFeature],
    feature_counts: dict[str, int],
    preferences: list,
) -> list[float]:
    """Compute weights for selecting which feature to delete.

    Boosts weight for types above their max, reduces for types at/below min.
    """
    pref_by_type = {p.feature_type: p for p in preferences}
    weights = []
    for pf in features:
        pref = pref_by_type.get(pf.feature.feature_type)
        w = 1.0
        if pref is not None:
            current = feature_counts.get(pf.feature.feature_type, 0)
            if pref.max is not None and current > pref.max:
                w = 1.0 + (current - pref.max) * 2.0
            elif current <= pref.min:
                w = 0.1
        weights.append(w)
    return weights


def _build_object_index(
    catalog: TerrainCatalog,
) -> dict[str, TerrainObject]:
    return {co.item.id: co.item for co in catalog.objects}


def _compute_score(
    layout: TerrainLayout,
    feature_count_preferences: list[FeatureCountPreference],
    objects_by_id: dict[str, TerrainObject],
    skip_visibility: bool = False,
    scoring_targets: ScoringTargets | None = None,
    visibility_cache: VisibilityCache | None = None,
) -> float:
    """Compute fitness score for hill-climbing.

    Phase 1 (score < PHASE2_BASE): gradient toward satisfying count preferences.
    Phase 2 (score >= PHASE2_BASE): optimize visibility toward targets.
    """
    counts = _count_features_by_type(layout)
    total_deficit = 0
    for pref in feature_count_preferences:
        current = counts.get(pref.feature_type, 0)
        if current < pref.min:
            total_deficit += pref.min - current
        elif pref.max is not None and current > pref.max:
            total_deficit += current - pref.max

    if total_deficit > 0:
        return PHASE2_BASE / (1.0 + total_deficit)

    if skip_visibility:
        return PHASE2_BASE

    vis = compute_layout_visibility(
        layout, objects_by_id, visibility_cache=visibility_cache
    )

    if scoring_targets is None:
        vis_pct = vis["overall"]["value"]
        return PHASE2_BASE + (100.0 - vis_pct)

    total_weight = 0.0
    total_weighted_error = 0.0

    if scoring_targets.overall_visibility_target is not None:
        actual = vis["overall"]["value"]
        error = abs(actual - scoring_targets.overall_visibility_target)
        w = scoring_targets.overall_visibility_weight
        total_weighted_error += w * error
        total_weight += w

    if scoring_targets.dz_visibility_target is not None:
        dz_vis = vis.get("dz_visibility")
        if dz_vis and len(dz_vis) > 0:
            avg = sum(d["value"] for d in dz_vis.values()) / len(dz_vis)
            error = abs(avg - scoring_targets.dz_visibility_target)
            w = scoring_targets.dz_visibility_weight
            total_weighted_error += w * error
            total_weight += w

    if scoring_targets.dz_hidden_target is not None:
        dz_cross = vis.get("dz_to_dz_visibility")
        if dz_cross and len(dz_cross) > 0:
            avg = sum(d["value"] for d in dz_cross.values()) / len(dz_cross)
            error = abs(avg - scoring_targets.dz_hidden_target)
            w = scoring_targets.dz_hidden_weight
            total_weighted_error += w * error
            total_weight += w

    if scoring_targets.objective_hidability_target is not None:
        obj_hide = vis.get("objective_hidability")
        if obj_hide and len(obj_hide) > 0:
            avg = sum(d["value"] for d in obj_hide.values()) / len(obj_hide)
            error = abs(avg - scoring_targets.objective_hidability_target)
            w = scoring_targets.objective_hidability_weight
            total_weighted_error += w * error
            total_weight += w

    if total_weight <= 0:
        vis_pct = vis["overall"]["value"]
        return PHASE2_BASE + (100.0 - vis_pct)

    weighted_avg_error = total_weighted_error / total_weight
    return PHASE2_BASE + (100.0 - weighted_avg_error)


def _instantiate_feature(
    template: TerrainFeature, feature_id: int
) -> TerrainFeature:
    """Create a new feature instance from a catalog template."""
    return TerrainFeature(
        id=f"feature_{feature_id}",
        feature_type=template.feature_type,
        components=list(template.components),
    )


def _next_feature_id(layout: TerrainLayout) -> int:
    """Find the next unused feature_N id number."""
    max_id = 0
    for pf in layout.placed_features:
        parts = pf.feature.id.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            max_id = max(max_id, int(parts[1]))
    return max_id + 1


def _temperature_move(
    rng: PCG32,
    old_transform: Transform,
    table_width: float,
    table_depth: float,
    t_factor: float,
) -> Transform:
    """Generate a temperature-aware move transform.

    At t_factor=0, displacement is small (±MIN_MOVE_RANGE/2 inches).
    At t_factor=1, displacement spans the full table.
    Always consumes exactly 4 PRNG values.
    """
    max_dim = max(table_width, table_depth)
    move_range = MIN_MOVE_RANGE + t_factor * (max_dim - MIN_MOVE_RANGE)

    dx = (rng.next_float() - 0.5) * 2.0 * move_range
    dz = (rng.next_float() - 0.5) * 2.0 * move_range
    rotate_check = rng.next_float()
    rot_angle_raw = rng.next_float()

    new_x = _quantize_position(old_transform.x + dx)
    new_z = _quantize_position(old_transform.z + dz)

    # Rotation: 0% chance at t=0, 50% chance at t=1
    if rotate_check < 0.5 * t_factor:
        rot = _quantize_angle(rot_angle_raw * 360.0)
    else:
        rot = old_transform.rotation_deg

    return Transform(new_x, new_z, rot)


def _try_single_action(
    layout: TerrainLayout,
    rng: PCG32,
    t_factor: float,
    next_id: int,
    catalog_features: list[TerrainFeature],
    has_catalog: bool,
    objects_by_id: dict[str, TerrainObject],
    params: EngineParams,
) -> tuple[StepUndo, int] | None:
    """Attempt one mutation action. Returns (undo, new_next_id) or None on failure."""
    features = layout.placed_features
    has_features = len(features) > 0
    feature_counts = _count_features_by_type(layout)

    # Compute action weights: [add, move, delete, replace]
    add_weight = 1.0 if has_catalog else 0.0
    move_weight = 1.0 if has_features else 0.0
    delete_weight = 1.0 if has_features else 0.0
    replace_weight = 1.0 if (has_features and has_catalog) else 0.0

    # Preference biasing on add/delete only
    for pref in params.feature_count_preferences:
        current = feature_counts.get(pref.feature_type, 0)
        if current < pref.min:
            shortage = pref.min - current
            add_weight *= 1.0 + shortage * 2.0
            delete_weight *= 0.1
        elif pref.max is not None and current > pref.max:
            excess = current - pref.max
            delete_weight *= 1.0 + excess * 2.0
            add_weight *= 0.1

    weights = [add_weight, move_weight, delete_weight, replace_weight]
    action = _weighted_choice(rng, weights)

    if action < 0:
        return None

    if action == 0:
        # Add
        template_weights = _compute_template_weights(
            catalog_features,
            feature_counts,
            params.feature_count_preferences,
        )
        tidx = _weighted_choice(rng, template_weights)
        if tidx < 0:
            return None
        template = catalog_features[tidx]
        new_feat = _instantiate_feature(template, next_id)
        x = _quantize_position(
            rng.next_float() * params.table_width - params.table_width / 2
        )
        z = _quantize_position(
            rng.next_float() * params.table_depth - params.table_depth / 2
        )
        rot = _quantize_angle(rng.next_float() * 360.0)
        placed = PlacedFeature(new_feat, Transform(x, z, rot))
        features.append(placed)
        idx = len(features) - 1
        if is_valid_placement(
            features,
            idx,
            params.table_width,
            params.table_depth,
            objects_by_id,
            min_feature_gap=params.min_feature_gap_inches,
            min_edge_gap=params.min_edge_gap_inches,
            rotationally_symmetric=params.rotationally_symmetric,
        ):
            return (
                StepUndo(action="add", index=idx, prev_next_id=next_id),
                next_id + 1,
            )
        else:
            features.pop()
            return None

    elif action == 1:
        # Move (temperature-aware)
        idx = rng.next_int(0, len(features) - 1)
        old = features[idx]
        new_transform = _temperature_move(
            rng,
            old.transform,
            params.table_width,
            params.table_depth,
            t_factor,
        )
        features[idx] = PlacedFeature(old.feature, new_transform)
        if is_valid_placement(
            features,
            idx,
            params.table_width,
            params.table_depth,
            objects_by_id,
            min_feature_gap=params.min_feature_gap_inches,
            min_edge_gap=params.min_edge_gap_inches,
            rotationally_symmetric=params.rotationally_symmetric,
        ):
            return (
                StepUndo(action="move", index=idx, old_feature=old),
                next_id,
            )
        else:
            features[idx] = old
            return None

    elif action == 2:
        # Delete
        delete_weights = _compute_delete_weights(
            features, feature_counts, params.feature_count_preferences
        )
        idx = _weighted_choice(rng, delete_weights)
        if idx < 0:
            return None
        saved = features.pop(idx)
        return (
            StepUndo(action="delete", index=idx, old_feature=saved),
            next_id,
        )

    elif action == 3:
        # Replace: remove feature, add different template at same position
        delete_weights = _compute_delete_weights(
            features, feature_counts, params.feature_count_preferences
        )
        idx = _weighted_choice(rng, delete_weights)
        if idx < 0:
            return None
        template_weights = _compute_template_weights(
            catalog_features,
            feature_counts,
            params.feature_count_preferences,
        )
        tidx = _weighted_choice(rng, template_weights)
        if tidx < 0:
            return None
        template = catalog_features[tidx]
        old = features[idx]
        new_feat = _instantiate_feature(template, next_id)
        features[idx] = PlacedFeature(new_feat, old.transform)
        if is_valid_placement(
            features,
            idx,
            params.table_width,
            params.table_depth,
            objects_by_id,
            min_feature_gap=params.min_feature_gap_inches,
            min_edge_gap=params.min_edge_gap_inches,
            rotationally_symmetric=params.rotationally_symmetric,
        ):
            return (
                StepUndo(
                    action="replace",
                    index=idx,
                    old_feature=old,
                    prev_next_id=next_id,
                ),
                next_id + 1,
            )
        else:
            features[idx] = old
            return None

    return None


def _perform_step(
    layout: TerrainLayout,
    rng: PCG32,
    t_factor: float,
    next_id: int,
    catalog_features: list[TerrainFeature],
    has_catalog: bool,
    objects_by_id: dict[str, TerrainObject],
    params: EngineParams,
) -> tuple[StepUndo, int]:
    """Try mutations with decaying temperature until one succeeds or retries exhausted."""
    effective_t = t_factor
    for _ in range(MAX_RETRIES):
        result = _try_single_action(
            layout,
            rng,
            effective_t,
            next_id,
            catalog_features,
            has_catalog,
            objects_by_id,
            params,
        )
        if result is not None:
            return result
        effective_t *= RETRY_DECAY
    return StepUndo(action="noop"), next_id


def _undo_step(layout: TerrainLayout, undo: StepUndo) -> None:
    """Revert a mutation using its undo token."""
    features = layout.placed_features
    if undo.action == "noop":
        return
    elif undo.action == "add":
        features.pop(undo.index)
    elif undo.action == "move":
        assert undo.old_feature is not None
        features[undo.index] = undo.old_feature
    elif undo.action == "delete":
        assert undo.old_feature is not None
        features.insert(undo.index, undo.old_feature)
    elif undo.action == "replace":
        assert undo.old_feature is not None
        features[undo.index] = undo.old_feature


def _create_layout(params: EngineParams) -> tuple[TerrainLayout, int]:
    """Create initial layout and next_id from params."""
    if params.initial_layout is not None:
        layout = TerrainLayout(
            table_width=params.table_width,
            table_depth=params.table_depth,
            placed_features=list(params.initial_layout.placed_features),
            rotationally_symmetric=params.rotationally_symmetric,
            mission=params.mission,
        )
        next_id = _next_feature_id(layout)
    else:
        layout = TerrainLayout(
            table_width=params.table_width,
            table_depth=params.table_depth,
            rotationally_symmetric=params.rotationally_symmetric,
            mission=params.mission,
        )
        next_id = 1
    return layout, next_id


def _generate_hill_climbing(params: EngineParams) -> EngineResult:
    """Single-replica hill climbing (original algorithm)."""
    rng = PCG32(params.seed)
    objects_by_id = _build_object_index(params.catalog)
    layout, next_id = _create_layout(params)

    catalog_features = [cf.item for cf in params.catalog.features]
    has_catalog = len(catalog_features) > 0

    vis_cache: VisibilityCache | None = None
    if not params.skip_visibility:
        vis_cache = VisibilityCache(layout, objects_by_id)

    current_score = _compute_score(
        layout,
        params.feature_count_preferences,
        objects_by_id,
        params.skip_visibility,
        params.scoring_targets,
        visibility_cache=vis_cache,
    )

    for _ in range(params.num_steps):
        undo, new_next_id = _perform_step(
            layout,
            rng,
            1.0,
            next_id,
            catalog_features,
            has_catalog,
            objects_by_id,
            params,
        )
        if undo.action == "noop":
            continue
        new_score = _compute_score(
            layout,
            params.feature_count_preferences,
            objects_by_id,
            params.skip_visibility,
            params.scoring_targets,
            visibility_cache=vis_cache,
        )
        if new_score >= current_score:
            current_score = new_score
            next_id = new_next_id
        else:
            _undo_step(layout, undo)

    if not params.skip_visibility:
        layout.visibility = compute_layout_visibility(
            layout, objects_by_id, visibility_cache=vis_cache
        )

    return EngineResult(
        layout=layout,
        score=current_score,
        steps_completed=params.num_steps,
    )


@dataclass
class _TemperingReplica:
    layout: TerrainLayout
    rng: PCG32
    score: float
    next_id: int
    temperature: float
    vis_cache: VisibilityCache | None


def _attempt_replica_swap(
    ri: _TemperingReplica,
    rj: _TemperingReplica,
    swap_rng: PCG32,
) -> bool:
    """Attempt replica exchange. Always consumes one PRNG value."""
    r = swap_rng.next_float()

    ti = ri.temperature
    tj = rj.temperature
    si = ri.score
    sj = rj.score

    if ti <= 0.0:
        accept = sj >= si
    elif tj <= 0.0:
        accept = si >= sj
    else:
        delta = (1.0 / ti - 1.0 / tj) * (sj - si)
        if delta >= 0.0:
            accept = True
        else:
            accept = r < math.exp(delta)

    if accept:
        ri.layout, rj.layout = rj.layout, ri.layout
        ri.score, rj.score = rj.score, ri.score
        ri.next_id, rj.next_id = rj.next_id, ri.next_id
        ri.vis_cache, rj.vis_cache = rj.vis_cache, ri.vis_cache

    return accept


def _generate_tempering(
    params: EngineParams, num_replicas: int
) -> EngineResult:
    """Multi-replica parallel tempering with SA acceptance."""
    objects_by_id = _build_object_index(params.catalog)
    catalog_features = [cf.item for cf in params.catalog.features]
    has_catalog = len(catalog_features) > 0
    temperatures = compute_temperatures(num_replicas, params.max_temperature)

    # Create replicas
    replicas: list[_TemperingReplica] = []
    for i in range(num_replicas):
        rng = PCG32(params.seed, seq=i)
        layout, next_id = _create_layout(params)
        vis_cache: VisibilityCache | None = None
        if not params.skip_visibility:
            vis_cache = VisibilityCache(layout, objects_by_id)
        score = _compute_score(
            layout,
            params.feature_count_preferences,
            objects_by_id,
            params.skip_visibility,
            params.scoring_targets,
            visibility_cache=vis_cache,
        )
        replicas.append(
            _TemperingReplica(
                layout=layout,
                rng=rng,
                score=score,
                next_id=next_id,
                temperature=temperatures[i],
                vis_cache=vis_cache,
            )
        )

    swap_rng = PCG32(params.seed, seq=num_replicas)

    # Track best
    best_score = replicas[0].score
    best_layout = copy.deepcopy(replicas[0].layout)
    for r in replicas[1:]:
        if r.score > best_score:
            best_score = r.score
            best_layout = copy.deepcopy(r.layout)

    # Main loop
    remaining = params.num_steps
    swap_interval = params.swap_interval
    max_temperature = params.max_temperature

    while remaining > 0:
        batch_size = min(swap_interval, remaining)

        for replica in replicas:
            t_factor = (
                replica.temperature / max_temperature
                if max_temperature > 0
                else 0.0
            )
            num_mutations = 1 + int(t_factor * MAX_EXTRA_MUTATIONS)

            for _ in range(batch_size):
                # Apply multiple mutations
                sub_undos: list[tuple[StepUndo, int]] = []
                for _ in range(num_mutations):
                    undo, new_nid = _perform_step(
                        replica.layout,
                        replica.rng,
                        t_factor,
                        replica.next_id,
                        catalog_features,
                        has_catalog,
                        objects_by_id,
                        params,
                    )
                    sub_undos.append((undo, replica.next_id))
                    replica.next_id = new_nid

                # Skip scoring if all noops
                if all(u.action == "noop" for u, _ in sub_undos):
                    continue

                old_score = replica.score
                new_score = _compute_score(
                    replica.layout,
                    params.feature_count_preferences,
                    objects_by_id,
                    params.skip_visibility,
                    params.scoring_targets,
                    visibility_cache=replica.vis_cache,
                )

                if sa_accept(
                    old_score, new_score, replica.temperature, replica.rng
                ):
                    replica.score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_layout = copy.deepcopy(replica.layout)
                else:
                    # Undo all mutations in reverse
                    for undo, prev_nid in reversed(sub_undos):
                        _undo_step(replica.layout, undo)
                        replica.next_id = prev_nid

        remaining -= batch_size

        # Swap adjacent replicas
        if remaining > 0 and num_replicas > 1:
            for i in range(num_replicas - 1):
                _attempt_replica_swap(replicas[i], replicas[i + 1], swap_rng)

    # Pick best from final replica states (covers equal-score case where
    # best_layout was never updated during the loop). Reverse iteration
    # so cold chain (index 0) wins ties.
    for replica in reversed(replicas):
        if replica.score >= best_score:
            best_score = replica.score
            best_layout = replica.layout

    # Final visibility on best layout
    if not params.skip_visibility:
        best_vis_cache = VisibilityCache(best_layout, objects_by_id)
        best_layout.visibility = compute_layout_visibility(
            best_layout, objects_by_id, visibility_cache=best_vis_cache
        )

    return EngineResult(
        layout=best_layout,
        score=best_score,
        steps_completed=params.num_steps,
    )


def generate(params: EngineParams) -> EngineResult:
    """Run terrain generation. Dispatches to hill climbing or tempering."""
    num_replicas = params.num_replicas
    if num_replicas is None or num_replicas <= 1:
        return _generate_hill_climbing(params)
    return _generate_tempering(params, num_replicas)


def generate_json(params_dict: dict) -> dict:
    """JSON-dict in, JSON-dict out wrapper."""
    params = EngineParams.from_dict(params_dict)
    result = generate(params)
    return result.to_dict()
