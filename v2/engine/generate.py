"""Terrain layout generation engine."""

from __future__ import annotations

from .collision import is_valid_placement
from .prng import PCG32
from .types import (
    EngineParams,
    EngineResult,
    PlacedFeature,
    TerrainCatalog,
    TerrainFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)


def _quantize_position(value: float) -> float:
    """Quantize position to nearest 0.1 inch."""
    return round(value / 0.1) * 0.1


def _quantize_angle(value: float) -> float:
    """Quantize angle to nearest 15 degrees."""
    return round(value / 15.0) * 15.0


def _count_features_by_type(layout: TerrainLayout) -> dict[str, int]:
    """Count how many of each feature_type are currently in the layout."""
    counts: dict[str, int] = {}
    for pf in layout.placed_features:
        ft = pf.feature.feature_type
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


def _compute_action_weights(
    current_count: int,
    min_count: int,
    max_count: int | None,
    has_catalog: bool,
    has_features: bool,
) -> tuple[float, float, float]:
    """Returns (add_weight, move_weight, delete_weight)."""
    add_weight = 1.0 if has_catalog else 0.0
    move_weight = 1.0 if has_features else 0.0
    delete_weight = 1.0 if has_features else 0.0

    if current_count < min_count:
        shortage = min_count - current_count
        add_weight *= 1.0 + shortage * 2.0
        delete_weight *= 0.1
    elif max_count is not None and current_count > max_count:
        excess = current_count - max_count
        delete_weight *= 1.0 + excess * 2.0
        add_weight *= 0.1

    return (add_weight, move_weight, delete_weight)


def _build_object_index(
    catalog: TerrainCatalog,
) -> dict[str, TerrainObject]:
    return {co.item.id: co.item for co in catalog.objects}


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


def generate(params: EngineParams) -> EngineResult:
    rng = PCG32(params.seed)
    objects_by_id = _build_object_index(params.catalog)
    if params.initial_layout is not None:
        layout = TerrainLayout(
            table_width=params.table_width,
            table_depth=params.table_depth,
            placed_features=list(params.initial_layout.placed_features),
        )
        next_id = _next_feature_id(layout)
    else:
        layout = TerrainLayout(
            table_width=params.table_width,
            table_depth=params.table_depth,
        )
        next_id = 1

    catalog_features = [cf.item for cf in params.catalog.features]
    has_catalog = len(catalog_features) > 0

    for _ in range(params.num_steps):
        features = layout.placed_features
        has_features = len(features) > 0
        feature_counts = _count_features_by_type(layout)

        # Default weights
        add_weight = 1.0 if has_catalog else 0.0
        move_weight = 1.0 if has_features else 0.0
        delete_weight = 1.0 if has_features else 0.0

        # Apply biasing for feature types with preferences
        for pref in params.feature_count_preferences:
            ft = pref.feature_type
            current = feature_counts.get(ft, 0)

            # For obstacle type, apply biasing to all add/delete actions
            # (assumes catalog primarily contains obstacles)
            if ft == "obstacle" and has_catalog:
                add_weight, move_weight, delete_weight = (
                    _compute_action_weights(
                        current, pref.min, pref.max, has_catalog, has_features
                    )
                )

        # Weighted random selection
        weights = [add_weight, move_weight, delete_weight]
        action = _weighted_choice(rng, weights)

        if action < 0:
            continue  # All weights were 0

        if action == 0:
            template = catalog_features[
                rng.next_int(0, len(catalog_features) - 1)
            ]
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
            if is_valid_placement(
                features,
                len(features) - 1,
                params.table_width,
                params.table_depth,
                objects_by_id,
                min_feature_gap=params.min_feature_gap_inches,
                min_edge_gap=params.min_edge_gap_inches,
            ):
                next_id += 1
            else:
                features.pop()

        elif action == 1:
            idx = rng.next_int(0, len(features) - 1)
            old = features[idx]
            x = _quantize_position(
                rng.next_float() * params.table_width - params.table_width / 2
            )
            z = _quantize_position(
                rng.next_float() * params.table_depth - params.table_depth / 2
            )
            rot = _quantize_angle(rng.next_float() * 360.0)
            features[idx] = PlacedFeature(old.feature, Transform(x, z, rot))
            if not is_valid_placement(
                features,
                idx,
                params.table_width,
                params.table_depth,
                objects_by_id,
                min_feature_gap=params.min_feature_gap_inches,
                min_edge_gap=params.min_edge_gap_inches,
            ):
                features[idx] = old

        elif action == 2:
            idx = rng.next_int(0, len(features) - 1)
            features.pop(idx)

    return EngineResult(
        layout=layout,
        steps_completed=params.num_steps,
    )


def generate_json(params_dict: dict) -> dict:
    """JSON-dict in, JSON-dict out wrapper."""
    params = EngineParams.from_dict(params_dict)
    result = generate(params)
    return result.to_dict()
