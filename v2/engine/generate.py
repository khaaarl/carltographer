"""Terrain layout generation engine.

Orchestrates the full optimization loop that turns a terrain catalog into a
scored layout. Two strategies are available:

  * **Hill-climbing** (single replica) — a simple simulated-annealing loop
    used when ``num_replicas <= 1``.
  * **Parallel tempering** (multiple replicas at different temperatures with
    periodic replica exchange) — the default. Cold replicas exploit good
    solutions while hot ones explore freely, with temperature-aware mutation
    distances so hot chains make large moves.

Both strategies share the same two-phase scoring function:

  1. **Phase 1 (feature counts):** Score is driven by how close the layout is
     to satisfying feature-count preferences (min/max obstacles, ruins, etc.).
     No visibility computation happens here—mutations are cheap.
  2. **Phase 2 (visibility):** Once counts are satisfied, score shifts to a
     weighted combination of line-of-sight metrics (overall visibility %,
     per-deployment-zone hideability, objective hidability). Each metric has a
     user-specified target and weight.

This module is the orchestrator—it owns the outer loop and scoring but
delegates heavily:

  * ``mutation.py`` — the five mutation actions (add/move/delete/replace/
    rotate) and their undo logic. Rejected mutations are rolled back
    in-place via ``StepUndo`` rather than cloning the layout.
  * ``tempering.py`` — temperature schedule computation and the Metropolis
    accept/reject criterion (``sa_accept``). This module builds the replica
    array and drives the swap logic, but the SA math lives there.
  * ``visibility.py`` — line-of-sight analysis (angular-sweep algorithm with
    caching). Called during phase-2 scoring to evaluate overall, per-DZ,
    and objective hidability metrics.
  * ``collision.py`` — OBB overlap and gap validation, used by mutations to
    check placement legality.

**Determinism and Rust parity:** This entire ``engine/`` package has a
bit-identical Rust twin in ``engine_rs/``. Given the same seed, both must
produce exactly the same layout. All randomness flows through a seeded PCG32
PRNG (``prng.py``), so the order of PRNG calls matters—reordering loops,
adding early exits, or using nondeterministic iteration (``set``, ``dict``
before 3.7) will silently break parity. After any behavioral change here,
run ``engine_cmp/`` to verify the engines still agree.

The public API is ``generate(params)`` which returns an ``EngineResult``,
and ``generate_json(dict)`` for the Rust-engine-compatible JSON interface.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .mutation import (
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


def _build_object_index(
    catalog: TerrainCatalog,
) -> dict[str, TerrainObject]:
    return {co.item.id: co.item for co in catalog.objects}


def _merge_layout_objects(
    objects_by_id: dict[str, TerrainObject],
    layout: TerrainLayout | None,
) -> None:
    """Merge object definitions from an initial layout into objects_by_id.

    Only adds objects whose IDs are not already present (catalog wins).
    """
    if layout is None:
        return
    for obj in layout.terrain_objects:
        if obj.id not in objects_by_id:
            objects_by_id[obj.id] = obj


def _collect_layout_objects(
    layout: TerrainLayout,
    objects_by_id: dict[str, TerrainObject],
) -> list[TerrainObject]:
    """Collect all unique TerrainObjects referenced by placed features."""
    seen: set[str] = set()
    result: list[TerrainObject] = []
    for pf in layout.placed_features:
        for comp in pf.feature.components:
            if comp.object_id not in seen:
                seen.add(comp.object_id)
                obj = objects_by_id.get(comp.object_id)
                if obj is not None:
                    result.append(obj)
    return result


# NOTE: If scoring logic grows more complex, consider extracting to a
# dedicated scoring module.
def _compute_score(
    layout: TerrainLayout,
    feature_count_preferences: list[FeatureCountPreference],
    objects_by_id: dict[str, TerrainObject],
    skip_visibility: bool = False,
    scoring_targets: ScoringTargets | None = None,
    visibility_cache: VisibilityCache | None = None,
    phase2_base: float = 10.0,
    standard_blocking_height: float = 4.0,
    infantry_blocking_height: float | None = 2.2,
) -> float:
    """Compute fitness score for hill-climbing.

    Phase 1 (score < phase2_base): gradient toward satisfying count preferences.
    Phase 2 (score >= phase2_base): optimize visibility toward targets.
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
        return phase2_base - total_deficit * 0.01

    if skip_visibility:
        return phase2_base

    # Skip expensive DZ/objective metrics during scoring when those
    # metrics aren't needed.
    needs_dz = scoring_targets is not None and (
        scoring_targets.dz_hideability_target is not None
        or scoring_targets.objective_hidability_target is not None
    )

    vis = compute_layout_visibility(
        layout,
        objects_by_id,
        min_blocking_height=standard_blocking_height,
        visibility_cache=visibility_cache,
        infantry_blocking_height=infantry_blocking_height,
        overall_only=not needs_dz,
    )

    if scoring_targets is None:
        vis_pct = vis["overall"]["value"]
        return phase2_base + (100.0 - vis_pct)

    total_weight = 0.0
    total_weighted_error = 0.0

    if scoring_targets.overall_visibility_target is not None:
        actual = vis["overall"]["value"]
        error = abs(actual - scoring_targets.overall_visibility_target)
        w = scoring_targets.overall_visibility_weight
        total_weighted_error += w * error
        total_weight += w

    if scoring_targets.dz_hideability_target is not None:
        dz_hide = vis.get("dz_hideability")
        if dz_hide and len(dz_hide) > 0:
            target = scoring_targets.dz_hideability_target
            avg_error = sum(
                abs(d["value"] - target) for d in dz_hide.values()
            ) / len(dz_hide)
            w = scoring_targets.dz_hideability_weight
            total_weighted_error += w * avg_error
            total_weight += w

    if scoring_targets.objective_hidability_target is not None:
        obj_hide = vis.get("objective_hidability")
        if obj_hide and len(obj_hide) > 0:
            target = scoring_targets.objective_hidability_target
            avg_error = sum(
                abs(d["value"] - target) for d in obj_hide.values()
            ) / len(obj_hide)
            w = scoring_targets.objective_hidability_weight
            total_weighted_error += w * avg_error
            total_weight += w

    if total_weight <= 0:
        vis_pct = vis["overall"]["value"]
        return phase2_base + (100.0 - vis_pct)

    weighted_avg_error = total_weighted_error / total_weight
    return phase2_base + (100.0 - weighted_avg_error)


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
    _merge_layout_objects(objects_by_id, params.initial_layout)
    layout, next_id = _create_layout(params)
    tuning = params.get_tuning()

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
        phase2_base=tuning.phase2_base,
        standard_blocking_height=params.standard_blocking_height,
        infantry_blocking_height=params.infantry_blocking_height,
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
            phase2_base=tuning.phase2_base,
            standard_blocking_height=params.standard_blocking_height,
            infantry_blocking_height=params.infantry_blocking_height,
        )
        if new_score >= current_score:
            current_score = new_score
            next_id = new_next_id
        else:
            _undo_step(layout, undo)

    if not params.skip_visibility:
        layout.visibility = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=params.standard_blocking_height,
            visibility_cache=vis_cache,
            infantry_blocking_height=params.infantry_blocking_height,
        )

    layout.terrain_objects = _collect_layout_objects(layout, objects_by_id)

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
    _merge_layout_objects(objects_by_id, params.initial_layout)
    catalog_features = [cf.item for cf in params.catalog.features]
    catalog_quantities = [cf.quantity for cf in params.catalog.features]
    has_catalog = len(catalog_features) > 0
    tuning = params.get_tuning()
    temperatures = compute_temperatures(
        num_replicas,
        params.max_temperature,
        temp_ladder_min_ratio=tuning.temp_ladder_min_ratio,
    )

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
            phase2_base=tuning.phase2_base,
            standard_blocking_height=params.standard_blocking_height,
            infantry_blocking_height=params.infantry_blocking_height,
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
            num_mutations = 1 + int(t_factor * tuning.max_extra_mutations)

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
                    phase2_base=tuning.phase2_base,
                    standard_blocking_height=params.standard_blocking_height,
                    infantry_blocking_height=params.infantry_blocking_height,
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
            best_layout,
            objects_by_id,
            min_blocking_height=params.standard_blocking_height,
            visibility_cache=best_vis_cache,
            infantry_blocking_height=params.infantry_blocking_height,
        )

    best_layout.terrain_objects = _collect_layout_objects(
        best_layout, objects_by_id
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
