"""Terrain layout generation engine."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .mutation import (
    MAX_EXTRA_MUTATIONS,
    StepUndo,
    _count_features_by_type,
    _perform_step,
    _undo_step,
)
from .prng import PCG32
from .tempering import compute_temperatures, sa_accept
from .types import (
    EngineParams,
    EngineResult,
    FeatureCountPreference,
    ScoringTargets,
    TerrainCatalog,
    TerrainLayout,
    TerrainObject,
)
from .visibility import VisibilityCache, compute_layout_visibility

PHASE2_BASE = 1000.0


def _build_object_index(
    catalog: TerrainCatalog,
) -> dict[str, TerrainObject]:
    return {co.item.id: co.item for co in catalog.objects}


# NOTE: If scoring logic grows more complex, consider extracting to a
# dedicated scoring module.
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


def _next_feature_id(layout: TerrainLayout) -> int:
    """Find the next unused feature_N id number."""
    max_id = 0
    for pf in layout.placed_features:
        parts = pf.feature.id.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            max_id = max(max_id, int(parts[1]))
    return max_id + 1


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
    catalog_quantities = [cf.quantity for cf in params.catalog.features]
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
            catalog_quantities,
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
    catalog_quantities = [cf.quantity for cf in params.catalog.features]
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
                for mi in range(num_mutations):
                    undo, new_nid = _perform_step(
                        replica.layout,
                        replica.rng,
                        t_factor,
                        replica.next_id,
                        catalog_features,
                        has_catalog,
                        objects_by_id,
                        params,
                        catalog_quantities,
                        index_in_chain=mi,
                        chain_length=num_mutations,
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
