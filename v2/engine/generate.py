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

        # Choose action: 0=add, 1=move, 2=delete
        if not has_features and not has_catalog:
            continue
        elif not has_features:
            action = 0
        elif not has_catalog:
            action = rng.next_int(0, 1) + 1
        else:
            action = rng.next_int(0, 2)

        if action == 0:
            template = catalog_features[
                rng.next_int(0, len(catalog_features) - 1)
            ]
            new_feat = _instantiate_feature(template, next_id)
            x = rng.next_float() * params.table_width - params.table_width / 2
            z = rng.next_float() * params.table_depth - params.table_depth / 2
            rot = rng.next_float() * 360.0
            placed = PlacedFeature(new_feat, Transform(x, z, rot))
            features.append(placed)
            if is_valid_placement(
                features,
                len(features) - 1,
                params.table_width,
                params.table_depth,
                objects_by_id,
            ):
                next_id += 1
            else:
                features.pop()

        elif action == 1:
            idx = rng.next_int(0, len(features) - 1)
            old = features[idx]
            x = rng.next_float() * params.table_width - params.table_width / 2
            z = rng.next_float() * params.table_depth - params.table_depth / 2
            rot = rng.next_float() * 360.0
            features[idx] = PlacedFeature(old.feature, Transform(x, z, rot))
            if not is_valid_placement(
                features,
                idx,
                params.table_width,
                params.table_depth,
                objects_by_id,
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
