"""Engine comparison tool for Python and Rust parity validation.

Validates that Python and Rust engines produce identical layouts for the same seed.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Add parent directory to path to import engine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.generate import generate as py_generate
from engine.types import (
    CatalogFeature,
    CatalogObject,
    EngineParams,
    FeatureComponent,
    FeatureCountPreference,
    Mission,
    PlacedFeature,
    ScoringTargets,
    Shape,
    TerrainCatalog,
    TerrainFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
    TuningParams,
)
from frontend.missions import get_mission

from .hash_manifest import compute_engine_hashes, write_manifest


def quantize_position(value: float) -> float:
    """Quantize position to nearest 0.1 inch (matches engine)."""
    return round(value / 0.1) * 0.1


def quantize_angle(value: float) -> float:
    """Quantize angle to nearest 15 degrees (matches engine)."""
    return round(value / 15.0) * 15.0


def positions_match(p1: float, p2: float, tolerance: float = 0.1) -> bool:
    """Compare positions with tolerance (matches quantization grid)."""
    return abs(p1 - p2) < tolerance


def angles_match(a1: float, a2: float, tolerance: float = 15.0) -> bool:
    """Compare angles with wraparound handling."""
    # Normalize both to [0, 360)
    a1 = a1 % 360.0
    a2 = a2 % 360.0
    # Check direct difference and wraparound
    diff = abs(a1 - a2)
    return diff < tolerance or abs(diff - 360.0) < tolerance


def compare_transforms(t1: dict, t2: dict) -> tuple[bool, list[str]]:
    """Deep comparison of position/rotation transforms.

    Returns (match: bool, diffs: list of error messages).
    """
    diffs = []

    x1 = t1.get("x_inches", 0.0)
    x2 = t2.get("x_inches", 0.0)
    if not positions_match(x1, x2):
        diffs.append(f"x_inches: {x1} vs {x2}")

    z1 = t1.get("z_inches", 0.0)
    z2 = t2.get("z_inches", 0.0)
    if not positions_match(z1, z2):
        diffs.append(f"z_inches: {z1} vs {z2}")

    r1 = t1.get("rotation_deg", 0.0)
    r2 = t2.get("rotation_deg", 0.0)
    if not angles_match(r1, r2):
        diffs.append(f"rotation_deg: {r1} vs {r2}")

    return len(diffs) == 0, diffs


def compare_layouts(layout1: dict, layout2: dict) -> tuple[bool, list[str]]:
    """Deep comparison of feature arrays.

    Returns (match: bool, diffs: list of error messages).
    """
    diffs = []

    features1 = layout1.get("placed_features", [])
    features2 = layout2.get("placed_features", [])

    if len(features1) != len(features2):
        diffs.append(f"Feature count: {len(features1)} vs {len(features2)}")
        return False, diffs

    for i, (f1, f2) in enumerate(zip(features1, features2)):
        # Compare feature ID
        id1 = f1.get("feature", {}).get("id", "")
        id2 = f2.get("feature", {}).get("id", "")
        if id1 != id2:
            diffs.append(f"Feature {i} id: {id1} vs {id2}")

        # Compare feature type
        ft1 = f1.get("feature", {}).get("feature_type", "")
        ft2 = f2.get("feature", {}).get("feature_type", "")
        if ft1 != ft2:
            diffs.append(f"Feature {i} type: {ft1} vs {ft2}")

        # Compare feature tags
        tags1 = sorted(f1.get("feature", {}).get("tags", []))
        tags2 = sorted(f2.get("feature", {}).get("tags", []))
        if tags1 != tags2:
            diffs.append(f"Feature {i} tags: {tags1} vs {tags2}")

        # Compare locked state
        locked1 = f1.get("locked", False)
        locked2 = f2.get("locked", False)
        if locked1 != locked2:
            diffs.append(f"Feature {i} locked: {locked1} vs {locked2}")

        # Compare transform
        t1 = f1.get("transform", {})
        t2 = f2.get("transform", {})
        match, transform_diffs = compare_transforms(t1, t2)
        if not match:
            for diff in transform_diffs:
                diffs.append(f"Feature {i} transform: {diff}")

    return len(diffs) == 0, diffs


def _compare_sub_passes(
    entry1: dict, entry2: dict, prefix: str, tolerance: float = 0.01
) -> list[str]:
    """Compare standard/infantry sub-dicts within a visibility entry."""
    diffs = []
    for pass_name in ("standard", "infantry"):
        sub1 = entry1.get(pass_name)
        sub2 = entry2.get(pass_name)
        if (sub1 is None) != (sub2 is None):
            diffs.append(
                f"{prefix} {pass_name}: one is None, other is not "
                f"(py={sub1 is not None}, rs={sub2 is not None})"
            )
        elif sub1 is not None and sub2 is not None:
            v1 = sub1.get("value", 0.0)
            v2 = sub2.get("value", 0.0)
            if abs(v1 - v2) > tolerance:
                diffs.append(f"{prefix} {pass_name} value: {v1} vs {v2}")
            sc1 = sub1.get("sample_count", 0)
            sc2 = sub2.get("sample_count", 0)
            if sc1 != sc2:
                diffs.append(
                    f"{prefix} {pass_name} sample_count: {sc1} vs {sc2}"
                )
    return diffs


def compare_visibility(
    vis1: dict | None,
    vis2: dict | None,
    tolerance: float = 0.01,
) -> tuple[bool, list[str]]:
    """Compare visibility results between engines.

    Returns (match: bool, diffs: list of error messages).
    """
    diffs = []

    if vis1 is None and vis2 is None:
        return True, []

    if vis1 is None or vis2 is None:
        diffs.append("Visibility: one is None, other is not")
        return False, diffs

    o1 = vis1.get("overall", {})
    o2 = vis2.get("overall", {})

    v1 = o1.get("value", 0.0)
    v2 = o2.get("value", 0.0)
    if abs(v1 - v2) > tolerance:
        diffs.append(f"Visibility value: {v1} vs {v2}")

    s1 = o1.get("sample_count", 0)
    s2 = o2.get("sample_count", 0)
    if s1 != s2:
        diffs.append(f"Visibility sample_count: {s1} vs {s2}")

    # Compare standard/infantry sub-passes on overall
    diffs.extend(_compare_sub_passes(o1, o2, "overall", tolerance))

    # Compare DZ visibility
    dz1 = vis1.get("dz_visibility")
    dz2 = vis2.get("dz_visibility")
    if (dz1 is None) != (dz2 is None):
        diffs.append(
            f"dz_visibility: one is None, other is not "
            f"(py={dz1 is not None}, rs={dz2 is not None})"
        )
    elif dz1 is not None and dz2 is not None:
        for key in set(list(dz1.keys()) + list(dz2.keys())):
            if key not in dz1:
                diffs.append(f"dz_visibility[{key}]: missing in Python")
            elif key not in dz2:
                diffs.append(f"dz_visibility[{key}]: missing in Rust")
            else:
                dv1 = dz1[key].get("value", 0.0)
                dv2 = dz2[key].get("value", 0.0)
                if abs(dv1 - dv2) > tolerance:
                    diffs.append(f"dz_visibility[{key}] value: {dv1} vs {dv2}")
                diffs.extend(
                    _compare_sub_passes(
                        dz1[key], dz2[key], f"dz_visibility[{key}]", tolerance
                    )
                )

    # Compare DZ-to-DZ visibility
    cross1 = vis1.get("dz_to_dz_visibility")
    cross2 = vis2.get("dz_to_dz_visibility")
    if (cross1 is None) != (cross2 is None):
        diffs.append(
            f"dz_to_dz_visibility: one is None, other is not "
            f"(py={cross1 is not None}, rs={cross2 is not None})"
        )
    elif cross1 is not None and cross2 is not None:
        for key in set(list(cross1.keys()) + list(cross2.keys())):
            if key not in cross1:
                diffs.append(f"dz_to_dz_visibility[{key}]: missing in Python")
            elif key not in cross2:
                diffs.append(f"dz_to_dz_visibility[{key}]: missing in Rust")
            else:
                cv1 = cross1[key].get("value", 0.0)
                cv2 = cross2[key].get("value", 0.0)
                if abs(cv1 - cv2) > tolerance:
                    diffs.append(
                        f"dz_to_dz_visibility[{key}] value: {cv1} vs {cv2}"
                    )
                diffs.extend(
                    _compare_sub_passes(
                        cross1[key],
                        cross2[key],
                        f"dz_to_dz_visibility[{key}]",
                        tolerance,
                    )
                )

    # Compare objective hidability
    oh1 = vis1.get("objective_hidability")
    oh2 = vis2.get("objective_hidability")
    if (oh1 is None) != (oh2 is None):
        diffs.append(
            f"objective_hidability: one is None, other is not "
            f"(py={oh1 is not None}, rs={oh2 is not None})"
        )
    elif oh1 is not None and oh2 is not None:
        for key in set(list(oh1.keys()) + list(oh2.keys())):
            if key not in oh1:
                diffs.append(f"objective_hidability[{key}]: missing in Python")
            elif key not in oh2:
                diffs.append(f"objective_hidability[{key}]: missing in Rust")
            else:
                ov1 = oh1[key].get("value", 0.0)
                ov2 = oh2[key].get("value", 0.0)
                if abs(ov1 - ov2) > tolerance:
                    diffs.append(
                        f"objective_hidability[{key}] value: {ov1} vs {ov2}"
                    )
                sc1 = oh1[key].get("safe_count", 0)
                sc2 = oh2[key].get("safe_count", 0)
                if sc1 != sc2:
                    diffs.append(
                        f"objective_hidability[{key}] safe_count: {sc1} vs {sc2}"
                    )
                diffs.extend(
                    _compare_sub_passes(
                        oh1[key],
                        oh2[key],
                        f"objective_hidability[{key}]",
                        tolerance,
                    )
                )

    return len(diffs) == 0, diffs


def compare_missions(
    m1: dict | None, m2: dict | None
) -> tuple[bool, list[str]]:
    """Compare mission data between engines.

    Returns (match: bool, diffs: list of error messages).
    """
    diffs = []

    if m1 is None and m2 is None:
        return True, []

    if m1 is None or m2 is None:
        diffs.append(
            f"Mission: one is None, other is not (py={m1 is not None}, rs={m2 is not None})"
        )
        return False, diffs

    if m1.get("name") != m2.get("name"):
        diffs.append(f"Mission name: {m1.get('name')} vs {m2.get('name')}")

    if m1.get("rotationally_symmetric") != m2.get("rotationally_symmetric"):
        diffs.append(
            f"Mission rotationally_symmetric: {m1.get('rotationally_symmetric')} vs {m2.get('rotationally_symmetric')}"
        )

    obj1 = m1.get("objectives", [])
    obj2 = m2.get("objectives", [])
    if len(obj1) != len(obj2):
        diffs.append(f"Mission objectives count: {len(obj1)} vs {len(obj2)}")

    dz1 = m1.get("deployment_zones", [])
    dz2 = m2.get("deployment_zones", [])
    if len(dz1) != len(dz2):
        diffs.append(
            f"Mission deployment_zones count: {len(dz1)} vs {len(dz2)}"
        )

    return len(diffs) == 0, diffs


def compare_results(
    result1: dict, result2: dict, visibility_tolerance: float = 0.01
) -> tuple[bool, list[str]]:
    """Compare full EngineResult objects.

    Returns (match: bool, diffs: list of error messages).
    """
    diffs = []

    layout1 = result1.get("layout", {})
    layout2 = result2.get("layout", {})
    match, layout_diffs = compare_layouts(layout1, layout2)
    if not match:
        diffs.extend(layout_diffs)

    # Compare visibility
    vis1 = layout1.get("visibility")
    vis2 = layout2.get("visibility")
    vis_match, vis_diffs = compare_visibility(
        vis1, vis2, tolerance=visibility_tolerance
    )
    if not vis_match:
        diffs.extend(vis_diffs)

    # Compare mission
    mission1 = layout1.get("mission")
    mission2 = layout2.get("mission")
    mission_match, mission_diffs = compare_missions(mission1, mission2)
    if not mission_match:
        diffs.extend(mission_diffs)

    # Compare terrain_objects
    to1 = sorted(layout1.get("terrain_objects", []), key=lambda o: o["id"])
    to2 = sorted(layout2.get("terrain_objects", []), key=lambda o: o["id"])
    if len(to1) != len(to2):
        diffs.append(f"terrain_objects count: {len(to1)} vs {len(to2)}")
    else:
        for o1, o2 in zip(to1, to2):
            if o1["id"] != o2["id"]:
                diffs.append(f"terrain_objects id: {o1['id']} vs {o2['id']}")

    # Compare score
    score1 = result1.get("score", 0.0)
    score2 = result2.get("score", 0.0)
    if abs(score1 - score2) > 0.01:
        diffs.append(f"Score: {score1} vs {score2}")

    return len(diffs) == 0, diffs


def make_test_catalog() -> TerrainCatalog:
    """Standard test catalog with WTC-style terrain.

    Includes:
    - Crate (double-stack): 5×2.5×5" obstacle, above blocking threshold
    - Bare ruin: 10×6" footprint, height 0 (obscuring, no LOS blocking)
    - Ruin with wall: 10×6" footprint + 6×0.5×5" opaque wall (blocks LOS)
    """
    return TerrainCatalog(
        objects=[
            CatalogObject(
                item=TerrainObject(
                    id="crate_5x2.5",
                    shapes=[
                        Shape(
                            width=5.0,
                            depth=2.5,
                            height=5.0,
                            offset=None,
                        )
                    ],
                    name="Crate (double-stack)",
                    tags=["container"],
                ),
                quantity=None,
            ),
            CatalogObject(
                item=TerrainObject(
                    id="ruins_10x6",
                    shapes=[
                        Shape(
                            width=10.0,
                            depth=6.0,
                            height=0.0,
                            offset=None,
                        )
                    ],
                    name="Ruins (base)",
                    tags=["ruins"],
                ),
                quantity=None,
            ),
            CatalogObject(
                item=TerrainObject(
                    id="opaque_wall_6x0.5",
                    shapes=[
                        Shape(
                            width=6.0,
                            depth=0.5,
                            height=5.0,
                            offset=None,
                        )
                    ],
                    name="Opaque Wall",
                    tags=["wall"],
                ),
                quantity=None,
            ),
        ],
        features=[
            CatalogFeature(
                item=TerrainFeature(
                    id="crate",
                    feature_type="obstacle",
                    components=[
                        FeatureComponent(
                            object_id="crate_5x2.5",
                            transform=None,
                        )
                    ],
                    tags=["obstacle"],
                ),
                quantity=None,
            ),
            CatalogFeature(
                item=TerrainFeature(
                    id="bare_ruin",
                    feature_type="obscuring",
                    components=[
                        FeatureComponent(
                            object_id="ruins_10x6",
                            transform=None,
                        )
                    ],
                    tags=["ruins", "obscuring"],
                ),
                quantity=None,
            ),
            CatalogFeature(
                item=TerrainFeature(
                    id="ruin_with_wall",
                    feature_type="obscuring",
                    components=[
                        FeatureComponent(
                            object_id="ruins_10x6",
                            transform=None,
                        ),
                        FeatureComponent(
                            object_id="opaque_wall_6x0.5",
                            transform=Transform(
                                x=2.0, z=0.0, rotation_deg=0.0
                            ),
                        ),
                    ],
                    tags=["ruins", "obscuring"],
                ),
                quantity=None,
            ),
        ],
        name="Test Catalog",
    )


def make_multi_type_catalog() -> TerrainCatalog:
    """Test catalog with both crates (obstacle) and ruins (obscuring)."""
    return TerrainCatalog(
        objects=[
            CatalogObject(
                item=TerrainObject(
                    id="crate_5x2.5",
                    shapes=[
                        Shape(
                            width=5.0,
                            depth=2.5,
                            height=5.0,
                            offset=None,
                        )
                    ],
                    name="Crate (double-stack)",
                    tags=["container"],
                ),
                quantity=None,
            ),
            CatalogObject(
                item=TerrainObject(
                    id="ruins_12x6",
                    shapes=[
                        Shape(
                            width=12.0,
                            depth=6.0,
                            height=0.0,
                            offset=None,
                        )
                    ],
                    name="Ruins",
                    tags=["ruins"],
                ),
                quantity=None,
            ),
        ],
        features=[
            CatalogFeature(
                item=TerrainFeature(
                    id="crate",
                    feature_type="obstacle",
                    components=[
                        FeatureComponent(
                            object_id="crate_5x2.5",
                            transform=None,
                        )
                    ],
                    tags=["obstacle"],
                ),
                quantity=None,
            ),
            CatalogFeature(
                item=TerrainFeature(
                    id="ruins",
                    feature_type="obscuring",
                    components=[
                        FeatureComponent(
                            object_id="ruins_12x6",
                            transform=None,
                        )
                    ],
                    tags=["ruins", "obscuring"],
                ),
                quantity=None,
            ),
        ],
        name="Multi-type Test Catalog",
    )


def make_quantity_limited_catalog() -> TerrainCatalog:
    """Test catalog with a single crate feature limited to quantity=2."""
    return TerrainCatalog(
        objects=[
            CatalogObject(
                item=TerrainObject(
                    id="crate_5x2.5",
                    shapes=[
                        Shape(
                            width=5.0,
                            depth=2.5,
                            height=5.0,
                            offset=None,
                        )
                    ],
                    name="Crate (double-stack)",
                    tags=["container"],
                ),
                quantity=2,
            )
        ],
        features=[
            CatalogFeature(
                item=TerrainFeature(
                    id="crate",
                    feature_type="obstacle",
                    components=[
                        FeatureComponent(
                            object_id="crate_5x2.5",
                            transform=None,
                        )
                    ],
                    tags=["obstacle"],
                ),
                quantity=2,
            )
        ],
        name="Quantity-limited Test Catalog",
    )


def _make_wall_catalog(include_short: bool = True) -> TerrainCatalog:
    """Catalog for infantry visibility tests.

    Contains a tall wall (5.0") that blocks both standard and infantry LOS,
    and optionally a short wall (2.5") that blocks infantry only.
    """
    objects = [
        CatalogObject(
            item=TerrainObject(
                id="tall_wall",
                shapes=[Shape(width=1.0, depth=24.0, height=5.0)],
                name="Tall Wall",
                tags=["obstacle"],
            ),
            quantity=None,
        ),
    ]
    features = [
        CatalogFeature(
            item=TerrainFeature(
                id="tall_wall",
                feature_type="obstacle",
                components=[FeatureComponent(object_id="tall_wall")],
                tags=["obstacle"],
            ),
            quantity=None,
        ),
    ]
    if include_short:
        objects.append(
            CatalogObject(
                item=TerrainObject(
                    id="short_wall",
                    shapes=[Shape(width=1.0, depth=24.0, height=2.5)],
                    name="Short Wall",
                    tags=["obstacle"],
                ),
                quantity=None,
            ),
        )
        features.append(
            CatalogFeature(
                item=TerrainFeature(
                    id="short_wall",
                    feature_type="obstacle",
                    components=[FeatureComponent(object_id="short_wall")],
                    tags=["obstacle"],
                ),
                quantity=None,
            ),
        )
    return TerrainCatalog(
        objects=objects,
        features=features,
        name="Wall Catalog",
    )


def _validate_infantry_dual_pass(py_dict: dict, rs_dict: dict) -> list[str]:
    """Validate dual-pass infantry visibility produces correct hidability.

    With a tall wall (5") on objective 2 and a short wall (2.5") on objective 3:
    - Standard pass: 1 of 5 objectives hidden (only behind the tall wall)
    - Infantry pass: 2 of 5 objectives hidden (behind both walls)
    """
    diffs = []
    for engine_name, result in [("Python", py_dict), ("Rust", rs_dict)]:
        vis = result.get("layout", {}).get("visibility", {})
        oh = vis.get("objective_hidability", {})
        if not oh:
            diffs.append(f"{engine_name}: missing objective_hidability")
            continue
        for dz_id in ("green", "red"):
            entry = oh.get(dz_id, {})
            std = entry.get("standard")
            inf = entry.get("infantry")
            if std is None:
                diffs.append(
                    f"{engine_name} {dz_id}: missing 'standard' sub-dict"
                )
                continue
            if inf is None:
                diffs.append(
                    f"{engine_name} {dz_id}: missing 'infantry' sub-dict"
                )
                continue
            std_safe = std.get("safe_count")
            if std_safe != 1:
                diffs.append(
                    f"{engine_name} {dz_id} standard safe_count: "
                    f"{std_safe} (expected 1)"
                )
            inf_safe = inf.get("safe_count")
            if inf_safe != 2:
                diffs.append(
                    f"{engine_name} {dz_id} infantry safe_count: "
                    f"{inf_safe} (expected 2)"
                )
    return diffs


def _validate_infantry_no_intermediate(
    py_dict: dict, rs_dict: dict
) -> list[str]:
    """Validate that without intermediate shapes, no infantry sub-dicts exist.

    With only a tall wall (5") which is above both thresholds, there are no
    shapes in the [infantry, standard) range, so the infantry pass is skipped.
    """
    diffs = []
    for engine_name, result in [("Python", py_dict), ("Rust", rs_dict)]:
        vis = result.get("layout", {}).get("visibility", {})
        oh = vis.get("objective_hidability", {})
        if not oh:
            diffs.append(f"{engine_name}: missing objective_hidability")
            continue
        for dz_id in ("green", "red"):
            entry = oh.get(dz_id, {})
            safe = entry.get("safe_count")
            if safe != 1:
                diffs.append(
                    f"{engine_name} {dz_id} safe_count: {safe} (expected 1)"
                )
            if "standard" in entry:
                diffs.append(
                    f"{engine_name} {dz_id}: unexpected 'standard' sub-dict"
                )
            if "infantry" in entry:
                diffs.append(
                    f"{engine_name} {dz_id}: unexpected 'infantry' sub-dict"
                )
    return diffs


def make_test_params(
    seed: int = 42,
    num_steps: int = 100,
    table_width: float = 44.0,
    table_depth: float = 30.0,
    min_feature_gap_inches: Optional[float] = None,
    min_edge_gap_inches: Optional[float] = None,
    feature_count_preferences: Optional[list[FeatureCountPreference]] = None,
    catalog: Optional[TerrainCatalog] = None,
    rotationally_symmetric: bool = False,
    mission: Optional[Mission] = None,
    skip_visibility: bool = False,
    scoring_targets: Optional[ScoringTargets] = None,
    num_replicas: int = 2,
    swap_interval: int = 20,
    max_temperature: float = 50.0,
    min_all_feature_gap_inches: Optional[float] = None,
    min_all_edge_gap_inches: Optional[float] = None,
    rotation_granularity_deg: float = 15.0,
    initial_layout: Optional[TerrainLayout] = None,
    tuning: Optional[TuningParams] = None,
    standard_blocking_height: float = 4.0,
    infantry_blocking_height: Optional[float] = 2.2,
) -> EngineParams:
    """Helper to build test params."""
    return EngineParams(
        seed=seed,
        table_width=table_width,
        table_depth=table_depth,
        catalog=catalog if catalog is not None else make_test_catalog(),
        num_steps=num_steps,
        initial_layout=initial_layout,
        feature_count_preferences=feature_count_preferences or [],
        min_feature_gap_inches=min_feature_gap_inches,
        min_edge_gap_inches=min_edge_gap_inches,
        min_all_feature_gap_inches=min_all_feature_gap_inches,
        min_all_edge_gap_inches=min_all_edge_gap_inches,
        rotation_granularity_deg=rotation_granularity_deg,
        rotationally_symmetric=rotationally_symmetric,
        mission=mission,
        skip_visibility=skip_visibility,
        scoring_targets=scoring_targets,
        num_replicas=num_replicas,
        swap_interval=swap_interval,
        max_temperature=max_temperature,
        tuning=tuning,
        standard_blocking_height=standard_blocking_height,
        infantry_blocking_height=infantry_blocking_height,
    )


@dataclass
class TestScenario:
    """One test scenario for comparison."""

    name: str
    seed: int
    num_steps: int
    table_width: float = 44.0
    table_depth: float = 30.0
    min_feature_gap_inches: Optional[float] = None
    min_edge_gap_inches: Optional[float] = None
    feature_count_preferences: Optional[list[FeatureCountPreference]] = None
    catalog: Optional[TerrainCatalog] = None
    rotationally_symmetric: bool = False
    mission: Optional[Mission] = None
    skip_visibility: bool = False
    scoring_targets: Optional[ScoringTargets] = None
    num_replicas: int = 2
    swap_interval: int = 20
    max_temperature: float = 50.0
    min_all_feature_gap_inches: Optional[float] = None
    min_all_edge_gap_inches: Optional[float] = None
    rotation_granularity_deg: float = 15.0
    initial_layout: Optional[TerrainLayout] = None
    tuning: Optional[TuningParams] = None
    standard_blocking_height: float = 4.0
    infantry_blocking_height: Optional[float] = 2.2
    validate_fn: Optional[Callable[[dict, dict], list[str]]] = None
    visibility_tolerance: float = 0.01

    def make_params(self) -> EngineParams:
        """Build EngineParams for this scenario."""
        return make_test_params(
            seed=self.seed,
            num_steps=self.num_steps,
            table_width=self.table_width,
            table_depth=self.table_depth,
            min_feature_gap_inches=self.min_feature_gap_inches,
            min_edge_gap_inches=self.min_edge_gap_inches,
            feature_count_preferences=self.feature_count_preferences,
            catalog=self.catalog,
            rotationally_symmetric=self.rotationally_symmetric,
            mission=self.mission,
            skip_visibility=self.skip_visibility,
            scoring_targets=self.scoring_targets,
            num_replicas=self.num_replicas,
            swap_interval=self.swap_interval,
            max_temperature=self.max_temperature,
            min_all_feature_gap_inches=self.min_all_feature_gap_inches,
            min_all_edge_gap_inches=self.min_all_edge_gap_inches,
            rotation_granularity_deg=self.rotation_granularity_deg,
            initial_layout=self.initial_layout,
            tuning=self.tuning,
            standard_blocking_height=self.standard_blocking_height,
            infantry_blocking_height=self.infantry_blocking_height,
        )


def _require_mission(deployment_name: str) -> dict:
    """Look up a CA2025-26 mission by deployment name, raising if not found."""
    m = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        deployment_name,
    )
    if m is None:
        raise ValueError(f"Mission {deployment_name!r} not found")
    return m


# Test scenarios
TEST_SCENARIOS = [
    TestScenario(
        "basic_10_steps", seed=42, num_steps=10, skip_visibility=False
    ),
    TestScenario("basic_50_steps", seed=42, num_steps=10),
    TestScenario(
        "basic_100_steps", seed=42, num_steps=10, skip_visibility=False
    ),
    TestScenario("seed_1", seed=1, num_steps=10, skip_visibility=False),
    TestScenario("seed_999", seed=999, num_steps=10, skip_visibility=False),
    TestScenario(
        "small_table",
        seed=42,
        num_steps=10,
        table_width=30.0,
        table_depth=22.0,
        skip_visibility=False,
    ),
    TestScenario(
        "large_table",
        seed=42,
        num_steps=50,
        table_width=120.0,
        table_depth=88.0,
        skip_visibility=True,
    ),
    TestScenario(
        "with_edge_gap",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        skip_visibility=True,
    ),
    TestScenario(
        "with_feature_gap",
        seed=42,
        num_steps=50,
        min_feature_gap_inches=3.0,
        skip_visibility=True,
    ),
    TestScenario(
        "with_both_gaps",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        min_feature_gap_inches=3.0,
        skip_visibility=True,
    ),
    TestScenario(
        "with_preferences",
        seed=42,
        num_steps=10,
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=2,
                max=5,
            )
        ],
        skip_visibility=False,
    ),
    TestScenario(
        "all_features",
        seed=42,
        num_steps=10,
        min_edge_gap_inches=1.0,
        min_feature_gap_inches=2.0,
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=3,
                max=10,
            )
        ],
        skip_visibility=False,
    ),
    TestScenario(
        "multi_type_no_prefs",
        seed=42,
        num_steps=10,
        catalog=make_multi_type_catalog(),
    ),
    TestScenario(
        "multi_type_with_prefs",
        seed=42,
        num_steps=10,
        catalog=make_multi_type_catalog(),
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=2,
                max=5,
            ),
            FeatureCountPreference(
                feature_type="obscuring",
                min=1,
                max=3,
            ),
        ],
        skip_visibility=False,
    ),
    TestScenario(
        "multi_type_one_pref",
        seed=99,
        num_steps=10,
        catalog=make_multi_type_catalog(),
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obscuring",
                min=2,
                max=4,
            ),
        ],
        skip_visibility=False,
    ),
    TestScenario(
        "symmetric_basic",
        seed=42,
        num_steps=10,
        rotationally_symmetric=True,
        skip_visibility=False,
    ),
    TestScenario(
        "symmetric_with_gaps",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        min_feature_gap_inches=3.0,
        rotationally_symmetric=True,
        skip_visibility=True,
    ),
    TestScenario(
        "symmetric_multi_type",
        seed=42,
        num_steps=10,
        catalog=make_multi_type_catalog(),
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=2,
                max=5,
            ),
            FeatureCountPreference(
                feature_type="obscuring",
                min=1,
                max=3,
            ),
        ],
        rotationally_symmetric=True,
        skip_visibility=False,
    ),
    TestScenario(
        "with_mission_hna",
        seed=42,
        num_steps=10,
        mission=Mission.from_dict(_require_mission("Hammer and Anvil")),
    ),
    TestScenario(
        "with_mission_dow",
        seed=99,
        num_steps=10,
        mission=Mission.from_dict(_require_mission("Dawn of War")),
    ),
    TestScenario(
        "scoring_with_prefs",
        seed=42,
        num_steps=10,
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=3,
                max=8,
            )
        ],
    ),
    TestScenario(
        "scoring_no_prefs",
        seed=99,
        num_steps=10,
    ),
    TestScenario(
        "scoring_targets_overall_only",
        seed=42,
        num_steps=10,
        scoring_targets=ScoringTargets(
            overall_visibility_target=30.0,
        ),
    ),
    TestScenario(
        "scoring_targets_with_mission",
        seed=42,
        num_steps=10,
        mission=Mission.from_dict(_require_mission("Hammer and Anvil")),
        scoring_targets=ScoringTargets(
            overall_visibility_target=30.0,
            dz_visibility_target=20.0,
            dz_hidden_target=40.0,
            objective_hidability_target=40.0,
        ),
    ),
    # -- Rotate action ---
    TestScenario(
        "rotate_action_basic",
        seed=77,
        num_steps=10,
        skip_visibility=False,
    ),
    # -- Tempering scenarios ---
    TestScenario(
        "tempering_basic",
        seed=42,
        num_steps=100,
        num_replicas=3,
        skip_visibility=True,
    ),
    TestScenario(
        "tempering_with_visibility",
        seed=42,
        num_steps=10,
        num_replicas=2,
    ),
    TestScenario(
        "tempering_with_gaps",
        seed=42,
        num_steps=100,
        num_replicas=3,
        min_feature_gap_inches=2.0,
        min_edge_gap_inches=3.0,
        skip_visibility=True,
    ),
    TestScenario(
        "tempering_with_preferences",
        seed=42,
        num_steps=100,
        num_replicas=3,
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=3,
                max=8,
            )
        ],
        skip_visibility=True,
    ),
    TestScenario(
        "tile_biased_small_table",
        seed=42,
        num_steps=100,
        table_width=20.0,
        table_depth=20.0,
        skip_visibility=True,
    ),
    TestScenario(
        "catalog_quantity_limit",
        seed=42,
        num_steps=100,
        catalog=make_quantity_limited_catalog(),
        skip_visibility=True,
    ),
    TestScenario(
        "rotation_granularity_90",
        seed=42,
        num_steps=100,
        rotation_granularity_deg=90.0,
        skip_visibility=True,
    ),
    TestScenario(
        "rotation_granularity_45",
        seed=77,
        num_steps=100,
        rotation_granularity_deg=45.0,
        skip_visibility=True,
    ),
    TestScenario(
        "all_feature_gaps",
        seed=42,
        num_steps=100,
        min_all_feature_gap_inches=1.5,
        min_all_edge_gap_inches=1.0,
        skip_visibility=True,
    ),
    # -- Tuning params ---
    TestScenario(
        "tuning_explicit_defaults",
        seed=42,
        num_steps=50,
        tuning=TuningParams(),
        skip_visibility=True,
    ),
    TestScenario(
        "tuning_custom_move_params",
        seed=42,
        num_steps=100,
        tuning=TuningParams(
            min_move_range=4.0,
            rotate_on_move_prob=0.8,
            tile_size=4.0,
        ),
        skip_visibility=True,
    ),
    TestScenario(
        "tuning_custom_weights",
        seed=77,
        num_steps=100,
        feature_count_preferences=[
            FeatureCountPreference(
                feature_type="obstacle",
                min=3,
                max=8,
            )
        ],
        tuning=TuningParams(
            shortage_boost=4.0,
            excess_boost=4.0,
            penalty_factor=0.2,
            delete_weight_last=0.5,
        ),
        skip_visibility=True,
    ),
    TestScenario(
        "orphaned_features",
        seed=99,
        num_steps=100,
        catalog=make_test_catalog(),
        skip_visibility=True,
        initial_layout=TerrainLayout(
            table_width=44.0,
            table_depth=30.0,
            placed_features=[
                PlacedFeature(
                    feature=TerrainFeature(
                        id="feature_1",
                        feature_type="obstacle",
                        components=[
                            FeatureComponent(object_id="big_block"),
                        ],
                    ),
                    transform=Transform(x=10.0, z=5.0, rotation_deg=0.0),
                ),
            ],
            terrain_objects=[
                TerrainObject(
                    id="big_block",
                    shapes=[
                        Shape(width=10.0, depth=10.0, height=3.0),
                    ],
                ),
            ],
        ),
    ),
    # -- Infantry visibility ---
    # Focused zero-step tests with hand-crafted layouts for correctness checks.
    # Tall wall (5.0"h) at objective 2 (0, -16): blocks both standard + infantry.
    # Short wall (2.5"h) at objective 3 (0, 16): blocks infantry only (2.5 < 4.0).
    # Walls are 1" wide × 24" deep to fully shadow the objective from both DZs.
    TestScenario(
        "infantry_vis_dual_pass",
        seed=42,
        num_steps=0,
        num_replicas=1,
        table_width=60.0,
        table_depth=44.0,
        catalog=_make_wall_catalog(include_short=True),
        mission=Mission.from_dict(_require_mission("Hammer and Anvil")),
        initial_layout=TerrainLayout(
            table_width=60.0,
            table_depth=44.0,
            placed_features=[
                PlacedFeature(
                    feature=TerrainFeature(
                        id="tall_wall",
                        feature_type="obstacle",
                        components=[FeatureComponent(object_id="tall_wall")],
                    ),
                    transform=Transform(x=0.0, z=-16.0, rotation_deg=0.0),
                ),
                PlacedFeature(
                    feature=TerrainFeature(
                        id="short_wall",
                        feature_type="obstacle",
                        components=[FeatureComponent(object_id="short_wall")],
                    ),
                    transform=Transform(x=0.0, z=16.0, rotation_deg=0.0),
                ),
            ],
            terrain_objects=[
                TerrainObject(
                    id="tall_wall",
                    shapes=[Shape(width=1.0, depth=24.0, height=5.0)],
                ),
                TerrainObject(
                    id="short_wall",
                    shapes=[Shape(width=1.0, depth=24.0, height=2.5)],
                ),
            ],
        ),
        validate_fn=_validate_infantry_dual_pass,
    ),
    # Same mission, only the tall wall — no intermediate shapes exist,
    # so the infantry pass should be skipped entirely.
    TestScenario(
        "infantry_vis_no_intermediate",
        seed=42,
        num_steps=0,
        num_replicas=1,
        table_width=60.0,
        table_depth=44.0,
        catalog=_make_wall_catalog(include_short=False),
        mission=Mission.from_dict(_require_mission("Hammer and Anvil")),
        initial_layout=TerrainLayout(
            table_width=60.0,
            table_depth=44.0,
            placed_features=[
                PlacedFeature(
                    feature=TerrainFeature(
                        id="tall_wall",
                        feature_type="obstacle",
                        components=[FeatureComponent(object_id="tall_wall")],
                    ),
                    transform=Transform(x=0.0, z=-16.0, rotation_deg=0.0),
                ),
            ],
            terrain_objects=[
                TerrainObject(
                    id="tall_wall",
                    shapes=[Shape(width=1.0, depth=24.0, height=5.0)],
                ),
            ],
        ),
        validate_fn=_validate_infantry_no_intermediate,
    ),
    # -- Feature locking ---
    TestScenario(
        "locked_features",
        seed=42,
        num_steps=100,
        catalog=make_test_catalog(),
        skip_visibility=True,
        initial_layout=TerrainLayout(
            table_width=44.0,
            table_depth=30.0,
            placed_features=[
                PlacedFeature(
                    feature=TerrainFeature(
                        id="feature_locked",
                        feature_type="obstacle",
                        components=[
                            FeatureComponent(object_id="crate_5x2.5"),
                        ],
                    ),
                    transform=Transform(x=5.0, z=5.0, rotation_deg=0.0),
                    locked=True,
                ),
                PlacedFeature(
                    feature=TerrainFeature(
                        id="feature_unlocked",
                        feature_type="obstacle",
                        components=[
                            FeatureComponent(object_id="crate_5x2.5"),
                        ],
                    ),
                    transform=Transform(x=-10.0, z=-5.0, rotation_deg=45.0),
                ),
            ],
        ),
    ),
]


def params_to_json_dict(params: EngineParams) -> dict:
    """Convert EngineParams to the JSON-compatible dict expected by the Rust engine."""
    return {
        "seed": params.seed,
        "table_width_inches": params.table_width,
        "table_depth_inches": params.table_depth,
        "catalog": {
            "objects": [
                {
                    "item": {
                        "id": obj.item.id,
                        "shapes": [
                            {
                                "shape_type": "rectangular_prism",
                                "width_inches": shape.width,
                                "depth_inches": shape.depth,
                                "height_inches": shape.height,
                                **(
                                    {"offset": shape.offset.to_dict()}
                                    if shape.offset
                                    else {}
                                ),
                                **(
                                    {
                                        "opacity_height_inches": shape.opacity_height_inches
                                    }
                                    if shape.opacity_height_inches is not None
                                    else {}
                                ),
                            }
                            for shape in obj.item.shapes
                        ],
                        **({"name": obj.item.name} if obj.item.name else {}),
                        **({"tags": obj.item.tags} if obj.item.tags else {}),
                    },
                    **({"quantity": obj.quantity} if obj.quantity else {}),
                }
                for obj in params.catalog.objects
            ],
            "features": [
                {
                    "item": {
                        "id": feat.item.id,
                        "feature_type": feat.item.feature_type,
                        "components": [
                            {
                                "object_id": comp.object_id,
                                **(
                                    {"transform": comp.transform.to_dict()}
                                    if comp.transform
                                    else {}
                                ),
                            }
                            for comp in feat.item.components
                        ],
                        **({"tags": feat.item.tags} if feat.item.tags else {}),
                    },
                    **({"quantity": feat.quantity} if feat.quantity else {}),
                }
                for feat in params.catalog.features
            ],
            **({"name": params.catalog.name} if params.catalog.name else {}),
        },
        "num_steps": params.num_steps,
        **(
            {"initial_layout": params.initial_layout.to_dict()}
            if params.initial_layout
            else {}
        ),
        **(
            {
                "feature_count_preferences": [
                    {
                        "feature_type": p.feature_type,
                        "min": p.min,
                        "max": p.max,
                    }
                    for p in params.feature_count_preferences
                ]
            }
            if params.feature_count_preferences
            else {}
        ),
        **(
            {"min_feature_gap_inches": params.min_feature_gap_inches}
            if params.min_feature_gap_inches is not None
            else {}
        ),
        **(
            {"min_edge_gap_inches": params.min_edge_gap_inches}
            if params.min_edge_gap_inches is not None
            else {}
        ),
        **(
            {"min_all_feature_gap_inches": params.min_all_feature_gap_inches}
            if params.min_all_feature_gap_inches is not None
            else {}
        ),
        **(
            {"min_all_edge_gap_inches": params.min_all_edge_gap_inches}
            if params.min_all_edge_gap_inches is not None
            else {}
        ),
        **(
            {"rotation_granularity_deg": params.rotation_granularity_deg}
            if params.rotation_granularity_deg != 15.0
            else {}
        ),
        **(
            {"rotationally_symmetric": True}
            if params.rotationally_symmetric
            else {}
        ),
        **(
            {"mission": params.mission.to_dict()}
            if params.mission is not None
            else {}
        ),
        **({"skip_visibility": True} if params.skip_visibility else {}),
        **(
            {"scoring_targets": params.scoring_targets.to_dict()}
            if params.scoring_targets is not None
            else {}
        ),
        **(
            {"num_replicas": params.num_replicas}
            if params.num_replicas is not None
            else {}
        ),
        "swap_interval": params.swap_interval,
        "max_temperature": params.max_temperature,
        **(
            {"tuning": params.tuning.to_dict()}
            if params.tuning is not None
            else {}
        ),
        **(
            {
                "standard_blocking_height_inches": params.standard_blocking_height
            }
            if params.standard_blocking_height != 4.0
            else {}
        ),
        **(
            {
                "infantry_blocking_height_inches": params.infantry_blocking_height
            }
            if params.infantry_blocking_height != 2.2
            else {}
        ),
    }


@dataclass
class ComparisonTiming:
    """Timing breakdown for a single comparison run."""

    python_secs: float
    rust_secs: float

    @property
    def total_secs(self) -> float:
        return self.python_secs + self.rust_secs


def run_comparison(
    params: EngineParams,
    verbose: bool = False,
    validate_fn: Optional[Callable[[dict, dict], list[str]]] = None,
    visibility_tolerance: float = 0.01,
) -> tuple[bool, list[str], ComparisonTiming | None]:
    """Execute both engines and compare results.

    Args:
        params: Engine parameters
        verbose: Print detailed comparison output
        validate_fn: Optional callback to validate correctness of results.
            Receives (py_dict, rs_dict) and returns list of error messages.
        visibility_tolerance: Tolerance for visibility value comparisons.

    Returns:
        (success: bool, diffs: list of error messages, timing or None on error)
    """
    diffs = []

    # Run Python engine
    try:
        t0 = time.perf_counter()
        py_result = py_generate(params)
        py_dict = py_result.to_dict()
        py_secs = time.perf_counter() - t0
    except Exception as e:
        diffs.append(f"Python engine failed: {e}")
        return False, diffs, None

    # Run Rust engine (via Python PyO3 binding)
    try:
        try:
            import engine_rs
        except ImportError:
            diffs.append(
                "Rust engine not available - need to build with: "
                "cd v2/engine_rs && maturin develop"
            )
            return False, diffs, None

        params_dict = params_to_json_dict(params)

        # Call Rust engine via PyO3
        t0 = time.perf_counter()
        rs_json_str = engine_rs.generate_json(  # type: ignore[attr-defined]
            json.dumps(params_dict)
        )
        rs_dict = json.loads(rs_json_str)
        rs_secs = time.perf_counter() - t0

    except ImportError as e:
        diffs.append(f"Failed to import Rust engine: {e}")
        return False, diffs, None
    except json.JSONDecodeError as e:
        diffs.append(f"Rust engine output not valid JSON: {e}")
        return False, diffs, None
    except Exception as e:
        diffs.append(f"Rust engine error: {e}")
        return False, diffs, None

    timing = ComparisonTiming(python_secs=py_secs, rust_secs=rs_secs)

    # Compare results
    match, compare_diffs = compare_results(
        py_dict, rs_dict, visibility_tolerance=visibility_tolerance
    )
    if not match:
        diffs.extend(compare_diffs)

    # Run correctness validation if provided
    if validate_fn is not None:
        validation_diffs = validate_fn(py_dict, rs_dict)
        diffs.extend(validation_diffs)

    if verbose and diffs:
        print("\nDifferences found:")
        for diff in diffs:
            print(f"  - {diff}")

    return len(diffs) == 0, diffs, timing


def main():
    """CLI entry point with pytest-compatible exit codes."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Python and Rust engine outputs"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run specific scenario by name",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed comparison output",
    )
    parser.add_argument(
        "--newest-first",
        action="store_true",
        help="Run scenarios in reverse order (newest first)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )

    args = parser.parse_args()

    scenarios = list(TEST_SCENARIOS)
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found")
            return 1

    if args.newest_first:
        scenarios = list(reversed(scenarios))

    passed = 0
    failed = 0
    total_time = 0.0

    for scenario in scenarios:
        params = scenario.make_params()
        success, diffs, timing = run_comparison(
            params,
            verbose=args.verbose,
            validate_fn=scenario.validate_fn,
            visibility_tolerance=scenario.visibility_tolerance,
        )

        if timing:
            time_str = (
                f"  ({timing.total_secs:.2f}s"
                f" — py {timing.python_secs:.2f}s"
                f", rs {timing.rust_secs:.2f}s)"
            )
            total_time += timing.total_secs
        else:
            time_str = ""

        if success:
            print(f"✓ {scenario.name}{time_str}")
            passed += 1
        else:
            print(f"✗ {scenario.name}{time_str}")
            if args.verbose:
                for diff in diffs:
                    print(f"    {diff}")
            failed += 1
            if args.fail_fast:
                print(f"\n{passed} passed, {failed} failed (stopped early)")
                return 1

    print(f"\n{passed} passed, {failed} failed ({total_time:.2f}s total)")

    # If all tests passed and we ran the full suite, write certification manifest
    if failed == 0 and not args.scenario:
        hashes = compute_engine_hashes()
        write_manifest(hashes)
        print(
            "\n✓ All tests passed! Engine parity manifest written to "
            ".engine_parity_manifest.json"
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
