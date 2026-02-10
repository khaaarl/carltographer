"""Terrain layout mutation actions and undo logic.

Each optimization step in ``generate.py`` calls ``_perform_step`` to mutate
the layout, then either keeps or reverts the change via ``_undo_step``.
There are five mutation actions:

  * **Add** — pick a catalog template (weighted by feature-count preferences
    and catalog quantity limits), place it in a tile-biased random position,
    accept if collision/gap checks pass.
  * **Move** — nudge a random feature by a temperature-scaled displacement
    (small at cold temperatures, table-spanning at hot). Always consumes
    exactly 4 PRNG values regardless of outcome to keep the random stream
    aligned.
  * **Delete** — remove a feature, weighted toward types that exceed their
    max count preference.
  * **Replace** — swap a feature's template in-place (same position, new
    geometry). Combines delete-weighting for the victim with add-weighting
    for the replacement.
  * **Rotate** — assign a new quantized angle to a random feature.

All mutations operate in-place on ``layout.placed_features`` and return a
``StepUndo`` token that records exactly what changed. Reverting is O(1) — no
deep copies of the layout are needed. If a mutation fails validation (e.g.,
collision), it is rolled back immediately and returns ``None``, so the caller
never sees an invalid state.

When ``_perform_step`` fails to find a valid mutation, it retries with
exponentially decaying temperature (``retry_decay``) up to ``max_retries``
times before returning a no-op.

Uses ``collision.py`` for placement validation and ``prng.py`` for all
randomness. Subject to the determinism/Rust-parity constraint — the number
and order of PRNG calls per action must match ``engine_rs`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .collision import (
    _is_at_origin,
    _mirror_placed_feature,
    get_world_obbs,
    is_valid_placement,
)
from .prng import PCG32
from .types import (
    EngineParams,
    PlacedFeature,
    TerrainFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)


@dataclass
class StepUndo:
    action: str  # "noop", "add", "move", "delete", "replace", "rotate"
    index: int = -1
    old_feature: PlacedFeature | None = None
    prev_next_id: int = 0


def _quantize_position(value: float) -> float:
    """Quantize position to nearest 0.1 inch."""
    return round(value / 0.1) * 0.1


def _quantize_angle(value: float, granularity: float = 15.0) -> float:
    """Quantize angle to nearest multiple of granularity degrees."""
    return round(value / granularity) * granularity


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


def _count_placed_per_template(
    placed_features: list[PlacedFeature],
    catalog_features: list[TerrainFeature],
    rotationally_symmetric: bool,
) -> list[int]:
    """Count how many placed features match each catalog template.

    Matches by comparing component object_id tuples. When rotationally
    symmetric, non-origin features count as 2 (canonical + mirror).
    """
    template_keys = [
        tuple(c.object_id for c in cf.components) for cf in catalog_features
    ]
    counts = [0] * len(catalog_features)
    for pf in placed_features:
        pf_key = tuple(c.object_id for c in pf.feature.components)
        increment = 1
        if rotationally_symmetric and (
            pf.transform.x != 0.0 or pf.transform.z != 0.0
        ):
            increment = 2
        for i, tk in enumerate(template_keys):
            if pf_key == tk:
                counts[i] += increment
                break
    return counts


def _compute_template_weights(
    catalog_features: list[TerrainFeature],
    feature_counts: dict[str, int],
    preferences: list,
    catalog_quantities: list[int | None],
    placed_features: list[PlacedFeature],
    rotationally_symmetric: bool,
    shortage_boost: float = 2.0,
    penalty_factor: float = 0.1,
) -> list[float]:
    """Compute weights for selecting which catalog feature to add.

    Boosts weight for types below their min, reduces for types at/above max.
    Sets weight to 0 for templates that have reached their catalog quantity limit.
    """
    pref_by_type = {p.feature_type: p for p in preferences}
    placed_counts = _count_placed_per_template(
        placed_features, catalog_features, rotationally_symmetric
    )
    weights = []
    for i, cf in enumerate(catalog_features):
        qty = catalog_quantities[i]
        if qty is not None and placed_counts[i] >= qty:
            weights.append(0.0)
            continue
        pref = pref_by_type.get(cf.feature_type)
        w = 1.0
        if pref is not None:
            current = feature_counts.get(cf.feature_type, 0)
            if current < pref.min:
                w = 1.0 + (pref.min - current) * shortage_boost
            elif pref.max is not None and current >= pref.max:
                w = penalty_factor
        weights.append(w)
    return weights


def _compute_delete_weights(
    features: list[PlacedFeature],
    feature_counts: dict[str, int],
    preferences: list,
    excess_boost: float = 2.0,
    penalty_factor: float = 0.1,
) -> list[float]:
    """Compute weights for selecting which feature to delete.

    Boosts weight for types above their max, reduces for types at/below min.
    """
    pref_by_type = {p.feature_type: p for p in preferences}
    weights = []
    for pf in features:
        if pf.locked:
            weights.append(0.0)
            continue
        pref = pref_by_type.get(pf.feature.feature_type)
        w = 1.0
        if pref is not None:
            current = feature_counts.get(pf.feature.feature_type, 0)
            if pref.max is not None and current > pref.max:
                w = 1.0 + (current - pref.max) * excess_boost
            elif current <= pref.min:
                w = penalty_factor
        weights.append(w)
    return weights


def _compute_tile_weights(
    placed_features: list[PlacedFeature],
    objects_by_id: dict[str, TerrainObject],
    table_width: float,
    table_depth: float,
    rotationally_symmetric: bool,
    tile_size: float = 2.0,
) -> tuple[list[float], int, int, float, float]:
    """Compute per-tile placement weights inversely proportional to occupancy.

    Divides table into ~tile_size tiles and counts how many feature OBBs
    overlap each tile. Weight = 1/(1+count), biasing placement toward
    empty areas.

    Could be cached and invalidated on layout changes (similar to
    VisibilityCache pattern) if performance becomes a concern.

    Returns (weights, nx, nz, tile_w, tile_d).
    """
    nx = max(1, round(table_width / tile_size))
    nz = max(1, round(table_depth / tile_size))
    tile_w = table_width / nx
    tile_d = table_depth / nz
    half_w = table_width / 2.0
    half_d = table_depth / 2.0

    counts = [0] * (nx * nz)

    for pf in placed_features:
        obbs = get_world_obbs(pf, objects_by_id)
        if rotationally_symmetric and not _is_at_origin(pf):
            mirror = _mirror_placed_feature(pf)
            obbs.extend(get_world_obbs(mirror, objects_by_id))
        for corners in obbs:
            # Compute AABB from corners
            min_x = min(c[0] for c in corners)
            max_x = max(c[0] for c in corners)
            min_z = min(c[1] for c in corners)
            max_z = max(c[1] for c in corners)
            # Convert to tile indices
            ix_lo = max(0, int((min_x + half_w) / tile_w))
            ix_hi = min(nx - 1, int((max_x + half_w) / tile_w))
            iz_lo = max(0, int((min_z + half_d) / tile_d))
            iz_hi = min(nz - 1, int((max_z + half_d) / tile_d))
            for iz in range(iz_lo, iz_hi + 1):
                for ix in range(ix_lo, ix_hi + 1):
                    counts[iz * nx + ix] += 1

    weights = [1.0 / (1 + c) for c in counts]
    return weights, nx, nz, tile_w, tile_d


def _instantiate_feature(
    template: TerrainFeature, feature_id: int
) -> TerrainFeature:
    """Create a new feature instance from a catalog template."""
    return TerrainFeature(
        id=f"feature_{feature_id}",
        feature_type=template.feature_type,
        components=list(template.components),
        tags=list(template.tags),
    )


def _temperature_move(
    rng: PCG32,
    old_transform: Transform,
    table_width: float,
    table_depth: float,
    t_factor: float,
    rotation_granularity: float = 15.0,
    min_move_range: float = 2.0,
    rotate_on_move_prob: float = 0.5,
) -> Transform:
    """Generate a temperature-aware move transform.

    At t_factor=0, displacement is small (±min_move_range/2 inches).
    At t_factor=1, displacement spans the full table.
    Always consumes exactly 4 PRNG values.
    """
    max_dim = max(table_width, table_depth)
    move_range = min_move_range + t_factor * (max_dim - min_move_range)

    dx = (rng.next_float() - 0.5) * 2.0 * move_range
    dz = (rng.next_float() - 0.5) * 2.0 * move_range
    rotate_check = rng.next_float()
    rot_angle_raw = rng.next_float()

    new_x = _quantize_position(old_transform.x + dx)
    new_z = _quantize_position(old_transform.z + dz)

    # Rotation: 0% chance at t=0, rotate_on_move_prob chance at t=1
    if rotate_check < rotate_on_move_prob * t_factor:
        rot = _quantize_angle(rot_angle_raw * 360.0, rotation_granularity)
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
    catalog_quantities: list[int | None],
    index_in_chain: int = 0,
    chain_length: int = 1,
) -> tuple[StepUndo, int] | None:
    """Attempt one mutation action. Returns (undo, new_next_id) or None on failure."""
    features = layout.placed_features
    has_features = len(features) > 0
    feature_counts = _count_features_by_type(layout)
    tuning = params.get_tuning()

    # Compute action weights: [add, move, delete, replace, rotate]
    # Non-final mutations in a chain get full delete weight (clearing space
    # for subsequent add/move), while final/standalone mutations use the
    # reduced base weight to avoid wasting scoring compute on doomed deletes.
    add_weight = 1.0 if has_catalog else 0.0
    move_weight = 1.0 if has_features else 0.0
    is_last = index_in_chain >= chain_length - 1
    delete_weight = (
        tuning.delete_weight_last
        if has_features and is_last
        else 1.0
        if has_features
        else 0.0
    )
    replace_weight = 1.0 if (has_features and has_catalog) else 0.0
    rotate_weight = 1.0 if has_features else 0.0

    # Preference biasing on add/delete only
    for pref in params.feature_count_preferences:
        current = feature_counts.get(pref.feature_type, 0)
        if current < pref.min:
            shortage = pref.min - current
            add_weight *= 1.0 + shortage * tuning.shortage_boost
            delete_weight *= tuning.penalty_factor
        elif pref.max is not None and current > pref.max:
            excess = current - pref.max
            delete_weight *= 1.0 + excess * tuning.excess_boost
            add_weight *= tuning.penalty_factor

    weights = [
        add_weight,
        move_weight,
        delete_weight,
        replace_weight,
        rotate_weight,
    ]
    action = _weighted_choice(rng, weights)

    if action < 0:
        return None

    if action == 0:
        # Add
        template_weights = _compute_template_weights(
            catalog_features,
            feature_counts,
            params.feature_count_preferences,
            catalog_quantities,
            features,
            layout.rotationally_symmetric,
            shortage_boost=tuning.shortage_boost,
            penalty_factor=tuning.penalty_factor,
        )
        tidx = _weighted_choice(rng, template_weights)
        if tidx < 0:
            return None
        template = catalog_features[tidx]
        new_feat = _instantiate_feature(template, next_id)
        half_w = params.table_width / 2.0
        half_d = params.table_depth / 2.0
        tile_weights, nx, nz, tile_w, tile_d = _compute_tile_weights(
            features,
            objects_by_id,
            params.table_width,
            params.table_depth,
            params.rotationally_symmetric,
            tile_size=tuning.tile_size,
        )
        tile_idx = _weighted_choice(rng, tile_weights)
        if tile_idx < 0:
            return None
        tile_iz = tile_idx // nx
        tile_ix = tile_idx % nx
        tile_x_min = -half_w + tile_ix * tile_w
        tile_z_min = -half_d + tile_iz * tile_d
        x = _quantize_position(tile_x_min + rng.next_float() * tile_w)
        z = _quantize_position(tile_z_min + rng.next_float() * tile_d)
        rot = _quantize_angle(
            rng.next_float() * 360.0, params.rotation_granularity_deg
        )
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
            min_all_feature_gap=params.min_all_feature_gap_inches,
            min_all_edge_gap=params.min_all_edge_gap_inches,
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
        if old.locked:
            return None
        new_transform = _temperature_move(
            rng,
            old.transform,
            params.table_width,
            params.table_depth,
            t_factor,
            rotation_granularity=params.rotation_granularity_deg,
            min_move_range=tuning.min_move_range,
            rotate_on_move_prob=tuning.rotate_on_move_prob,
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
            min_all_feature_gap=params.min_all_feature_gap_inches,
            min_all_edge_gap=params.min_all_edge_gap_inches,
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
            features,
            feature_counts,
            params.feature_count_preferences,
            excess_boost=tuning.excess_boost,
            penalty_factor=tuning.penalty_factor,
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
            features,
            feature_counts,
            params.feature_count_preferences,
            excess_boost=tuning.excess_boost,
            penalty_factor=tuning.penalty_factor,
        )
        idx = _weighted_choice(rng, delete_weights)
        if idx < 0:
            return None
        template_weights = _compute_template_weights(
            catalog_features,
            feature_counts,
            params.feature_count_preferences,
            catalog_quantities,
            features,
            layout.rotationally_symmetric,
            shortage_boost=tuning.shortage_boost,
            penalty_factor=tuning.penalty_factor,
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
            min_all_feature_gap=params.min_all_feature_gap_inches,
            min_all_edge_gap=params.min_all_edge_gap_inches,
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

    elif action == 4:
        # Rotate: pick random feature, assign new quantized angle
        idx = rng.next_int(0, len(features) - 1)
        old = features[idx]
        if old.locked:
            return None
        new_rot = _quantize_angle(
            rng.next_float() * 360.0, params.rotation_granularity_deg
        )
        new_transform = Transform(old.transform.x, old.transform.z, new_rot)
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
            min_all_feature_gap=params.min_all_feature_gap_inches,
            min_all_edge_gap=params.min_all_edge_gap_inches,
        ):
            return (
                StepUndo(action="rotate", index=idx, old_feature=old),
                next_id,
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
    catalog_quantities: list[int | None] | None = None,
    index_in_chain: int = 0,
    chain_length: int = 1,
) -> tuple[StepUndo, int]:
    """Try mutations with decaying temperature until one succeeds or retries exhausted."""
    if catalog_quantities is None:
        catalog_quantities = [None] * len(catalog_features)
    tuning = params.get_tuning()
    effective_t = t_factor
    for _ in range(tuning.max_retries):
        result = _try_single_action(
            layout,
            rng,
            effective_t,
            next_id,
            catalog_features,
            has_catalog,
            objects_by_id,
            params,
            catalog_quantities,
            index_in_chain=index_in_chain,
            chain_length=chain_length,
        )
        if result is not None:
            return result
        effective_t *= tuning.retry_decay
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
    elif undo.action == "rotate":
        assert undo.old_feature is not None
        features[undo.index] = undo.old_feature
