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
    Shape,
    TerrainCatalog,
    TerrainFeature,
    TerrainObject,
)

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
        )


# 12 diverse test scenarios
TEST_SCENARIOS = [
    TestScenario("basic_10_steps", seed=42, num_steps=10),
    TestScenario("basic_50_steps", seed=42, num_steps=50),
    TestScenario("basic_100_steps", seed=42, num_steps=100),
    TestScenario("seed_1", seed=1, num_steps=100),
    TestScenario("seed_999", seed=999, num_steps=100),
    TestScenario(
        "small_table",
        seed=42,
        num_steps=50,
        table_width=30.0,
        table_depth=22.0,
    ),
    TestScenario(
        "large_table",
        seed=42,
        num_steps=50,
        table_width=120.0,
        table_depth=88.0,
    ),
    TestScenario(
        "with_edge_gap", seed=42, num_steps=50, min_edge_gap_inches=2.0
    ),
    TestScenario(
        "with_feature_gap", seed=42, num_steps=50, min_feature_gap_inches=3.0
    ),
    TestScenario(
        "with_both_gaps",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        min_feature_gap_inches=3.0,
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
    ),
    TestScenario(
        "symmetric_basic",
        seed=42,
        num_steps=50,
        rotationally_symmetric=True,
    ),
    TestScenario(
        "symmetric_with_gaps",
        seed=42,
        num_steps=50,
        min_edge_gap_inches=2.0,
        min_feature_gap_inches=3.0,
        rotationally_symmetric=True,
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
    ),
]


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

        # Convert params to dict for JSON serialization
        params_dict = {
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
                            **(
                                {"name": obj.item.name}
                                if obj.item.name
                                else {}
                            ),
                            **(
                                {"tags": obj.item.tags}
                                if obj.item.tags
                                else {}
                            ),
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
                        **(
                            {"quantity": feat.quantity}
                            if feat.quantity
                            else {}
                        ),
                    }
                    for feat in params.catalog.features
                ],
                **(
                    {"name": params.catalog.name}
                    if params.catalog.name
                    else {}
                ),
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
        }

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
