"""Engine comparison tool for Python and Rust parity validation.

Validates that Python and Rust engines produce identical layouts for the same seed.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    ScoringTargets,
    Shape,
    TerrainCatalog,
    TerrainFeature,
    TerrainObject,
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

        # Compare transform
        t1 = f1.get("transform", {})
        t2 = f2.get("transform", {})
        match, transform_diffs = compare_transforms(t1, t2)
        if not match:
            for diff in transform_diffs:
                diffs.append(f"Feature {i} transform: {diff}")

    return len(diffs) == 0, diffs


def compare_visibility(
    vis1: dict | None, vis2: dict | None
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
    if abs(v1 - v2) > 0.01:
        diffs.append(f"Visibility value: {v1} vs {v2}")

    s1 = o1.get("sample_count", 0)
    s2 = o2.get("sample_count", 0)
    if s1 != s2:
        diffs.append(f"Visibility sample_count: {s1} vs {s2}")

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
                if abs(dv1 - dv2) > 0.01:
                    diffs.append(f"dz_visibility[{key}] value: {dv1} vs {dv2}")

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
                if abs(cv1 - cv2) > 0.01:
                    diffs.append(
                        f"dz_to_dz_visibility[{key}] value: {cv1} vs {cv2}"
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
                if abs(ov1 - ov2) > 0.01:
                    diffs.append(
                        f"objective_hidability[{key}] value: {ov1} vs {ov2}"
                    )
                sc1 = oh1[key].get("safe_count", 0)
                sc2 = oh2[key].get("safe_count", 0)
                if sc1 != sc2:
                    diffs.append(
                        f"objective_hidability[{key}] safe_count: {sc1} vs {sc2}"
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


def compare_results(result1: dict, result2: dict) -> tuple[bool, list[str]]:
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
    vis_match, vis_diffs = compare_visibility(vis1, vis2)
    if not vis_match:
        diffs.extend(vis_diffs)

    # Compare mission
    mission1 = layout1.get("mission")
    mission2 = layout2.get("mission")
    mission_match, mission_diffs = compare_missions(mission1, mission2)
    if not mission_match:
        diffs.extend(mission_diffs)

    # Compare score
    score1 = result1.get("score", 0.0)
    score2 = result2.get("score", 0.0)
    if abs(score1 - score2) > 0.01:
        diffs.append(f"Score: {score1} vs {score2}")

    return len(diffs) == 0, diffs


def make_test_catalog() -> TerrainCatalog:
    """Standard test catalog with crates."""
    return TerrainCatalog(
        objects=[
            CatalogObject(
                item=TerrainObject(
                    id="crate_5x2.5",
                    shapes=[
                        Shape(
                            width=5.0,
                            depth=2.5,
                            height=2.0,
                            offset=None,
                        )
                    ],
                    name="Crate",
                    tags=[],
                ),
                quantity=None,
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
                ),
                quantity=None,
            )
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
                            height=2.0,
                            offset=None,
                        )
                    ],
                    name="Crate",
                    tags=[],
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
                    tags=[],
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
                            height=2.0,
                            offset=None,
                        )
                    ],
                    name="Crate",
                    tags=[],
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
                ),
                quantity=2,
            )
        ],
        name="Quantity-limited Test Catalog",
    )


def make_test_params(
    seed: int = 42,
    num_steps: int = 100,
    table_width: float = 60.0,
    table_depth: float = 44.0,
    min_feature_gap_inches: Optional[float] = None,
    min_edge_gap_inches: Optional[float] = None,
    feature_count_preferences: Optional[list[FeatureCountPreference]] = None,
    catalog: Optional[TerrainCatalog] = None,
    rotationally_symmetric: bool = False,
    mission: Optional[Mission] = None,
    skip_visibility: bool = False,
    scoring_targets: Optional[ScoringTargets] = None,
    num_replicas: Optional[int] = None,
    swap_interval: int = 20,
    max_temperature: float = 50.0,
) -> EngineParams:
    """Helper to build test params."""
    return EngineParams(
        seed=seed,
        table_width=table_width,
        table_depth=table_depth,
        catalog=catalog if catalog is not None else make_test_catalog(),
        num_steps=num_steps,
        initial_layout=None,
        feature_count_preferences=feature_count_preferences or [],
        min_feature_gap_inches=min_feature_gap_inches,
        min_edge_gap_inches=min_edge_gap_inches,
        rotationally_symmetric=rotationally_symmetric,
        mission=mission,
        skip_visibility=skip_visibility,
        scoring_targets=scoring_targets,
        num_replicas=num_replicas,
        swap_interval=swap_interval,
        max_temperature=max_temperature,
    )


@dataclass
class TestScenario:
    """One test scenario for comparison."""

    name: str
    seed: int
    num_steps: int
    table_width: float = 60.0
    table_depth: float = 44.0
    min_feature_gap_inches: Optional[float] = None
    min_edge_gap_inches: Optional[float] = None
    feature_count_preferences: Optional[list[FeatureCountPreference]] = None
    catalog: Optional[TerrainCatalog] = None
    rotationally_symmetric: bool = False
    mission: Optional[Mission] = None
    skip_visibility: bool = False
    scoring_targets: Optional[ScoringTargets] = None
    num_replicas: Optional[int] = None
    swap_interval: int = 20
    max_temperature: float = 50.0

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
    TestScenario("basic_50_steps", seed=42, num_steps=50),
    TestScenario(
        "basic_100_steps", seed=42, num_steps=100, skip_visibility=False
    ),
    TestScenario("seed_1", seed=1, num_steps=100, skip_visibility=False),
    TestScenario("seed_999", seed=999, num_steps=100, skip_visibility=False),
    TestScenario(
        "small_table",
        seed=42,
        num_steps=50,
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
        skip_visibility=False,
    ),
    TestScenario(
        "with_edge_gap",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        skip_visibility=False,
    ),
    TestScenario(
        "with_feature_gap",
        seed=42,
        num_steps=50,
        min_feature_gap_inches=3.0,
        skip_visibility=False,
    ),
    TestScenario(
        "with_both_gaps",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        min_feature_gap_inches=3.0,
        skip_visibility=False,
    ),
    TestScenario(
        "with_preferences",
        seed=42,
        num_steps=50,
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
        num_steps=100,
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
        num_steps=50,
        catalog=make_multi_type_catalog(),
    ),
    TestScenario(
        "multi_type_with_prefs",
        seed=42,
        num_steps=100,
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
        num_steps=100,
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
        num_steps=50,
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
        skip_visibility=False,
    ),
    TestScenario(
        "symmetric_multi_type",
        seed=42,
        num_steps=100,
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
        num_steps=50,
        mission=Mission.from_dict(_require_mission("Hammer and Anvil")),
    ),
    TestScenario(
        "with_mission_dow",
        seed=99,
        num_steps=50,
        mission=Mission.from_dict(_require_mission("Dawn of War")),
    ),
    TestScenario(
        "scoring_with_prefs",
        seed=42,
        num_steps=50,
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
        num_steps=50,
    ),
    TestScenario(
        "scoring_targets_overall_only",
        seed=42,
        num_steps=50,
        scoring_targets=ScoringTargets(
            overall_visibility_target=30.0,
        ),
    ),
    TestScenario(
        "scoring_targets_with_mission",
        seed=42,
        num_steps=50,
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
        num_steps=100,
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
        num_steps=50,
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
    }


def run_comparison(
    params: EngineParams,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Execute both engines and compare results.

    Args:
        params: Engine parameters
        verbose: Print detailed comparison output

    Returns:
        (success: bool, diffs: list of error messages)
    """
    diffs = []

    # Run Python engine
    try:
        py_result = py_generate(params)
        py_dict = py_result.to_dict()
    except Exception as e:
        diffs.append(f"Python engine failed: {e}")
        return False, diffs

    # Run Rust engine (via Python PyO3 binding)
    try:
        try:
            import engine_rs
        except ImportError:
            diffs.append(
                "Rust engine not available - need to build with: "
                "cd v2/engine_rs && maturin develop"
            )
            return False, diffs

        params_dict = params_to_json_dict(params)

        # Call Rust engine via PyO3
        rs_json_str = engine_rs.generate_json(  # type: ignore[attr-defined]
            json.dumps(params_dict)
        )
        rs_dict = json.loads(rs_json_str)

    except ImportError as e:
        diffs.append(f"Failed to import Rust engine: {e}")
        return False, diffs
    except json.JSONDecodeError as e:
        diffs.append(f"Rust engine output not valid JSON: {e}")
        return False, diffs
    except Exception as e:
        diffs.append(f"Rust engine error: {e}")
        return False, diffs

    # Compare results
    match, compare_diffs = compare_results(py_dict, rs_dict)
    if not match:
        diffs.extend(compare_diffs)

    if verbose and diffs:
        print("\nDifferences found:")
        for diff in diffs:
            print(f"  - {diff}")

    return len(diffs) == 0, diffs


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

    args = parser.parse_args()

    scenarios = TEST_SCENARIOS
    if args.scenario:
        scenarios = [s for s in TEST_SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"Scenario '{args.scenario}' not found")
            return 1

    passed = 0
    failed = 0

    for scenario in scenarios:
        params = scenario.make_params()
        success, diffs = run_comparison(params, verbose=args.verbose)

        if success:
            print(f"✓ {scenario.name}")
            passed += 1
        else:
            print(f"✗ {scenario.name}")
            if args.verbose:
                for diff in diffs:
                    print(f"    {diff}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")

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
