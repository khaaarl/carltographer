#!/usr/bin/env python3
"""Generate benchmark fixture JSON files for the Rust engine.

Uses pairwise (all-pairs) testing to cover all 2-way parameter interactions
with 28 test cases across 9 parameters:

  - Table size: 44x30 (incursion), 60x44 (strike), 90x44 (onslaught)
  - Symmetry: off, on
  - Mission: none + 6 deployment types
  - Terrain: crates-only (5" tall), WTC mixed (crates + ruins + walls)
  - Steps: 10, 20, 50, 100
  - min_feature_gap_inches: 0, 5.2
  - min_edge_gap_inches: 0, 5.2
  - min_all_feature_gap_inches: 0, 3
  - min_all_edge_gap_inches: 0, 3

All cases include scoring_targets matching the UI defaults:
  overall_visibility_target=30 (weight 1), dz_hideability_target=70 (weight 5),
  objective_hidability_target=50 (weight 5).

Constraint: mission=none -> symmetry forced to off.

Run from v2/:
    python engine_rs/benches/generate_fixtures.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

# Ensure v2/ is on the path so we can import engine + frontend
v2_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(v2_dir))

from engine.visibility import (  # noqa: E402
    DZ_EXPANSION_INCHES,
    _expand_dz_polygons,
)
from frontend.missions import (  # noqa: E402
    _crucible_of_battle,
    _dawn_of_war,
    _hammer_and_anvil,
    _search_and_destroy,
    _sweeping_engagement,
    _tipping_point,
)

# ---------------------------------------------------------------------------
# Terrain catalogs
# ---------------------------------------------------------------------------

CRATES_CATALOG: dict[str, Any] = {
    "objects": [
        {
            "item": {
                "id": "crate_5x2.5",
                "shapes": [
                    {
                        "shape_type": "rectangular_prism",
                        "width_inches": 5.0,
                        "depth_inches": 2.5,
                        "height_inches": 5.0,
                    }
                ],
                "name": "Crate (double-stack)",
            }
        }
    ],
    "features": [
        {
            "item": {
                "id": "crate",
                "feature_type": "obstacle",
                "components": [{"object_id": "crate_5x2.5"}],
            }
        }
    ],
    "name": "Crates Only",
}

WTC_CATALOG: dict[str, Any] = {
    "objects": [
        {
            "item": {
                "id": "crate_5x2.5",
                "shapes": [
                    {
                        "shape_type": "rectangular_prism",
                        "width_inches": 5.0,
                        "depth_inches": 2.5,
                        "height_inches": 5.0,
                    }
                ],
                "name": "Crate (double-stack)",
            }
        },
        {
            "item": {
                "id": "ruins_10x6",
                "shapes": [
                    {
                        "shape_type": "rectangular_prism",
                        "width_inches": 10.0,
                        "depth_inches": 6.0,
                        "height_inches": 0.0,
                    }
                ],
                "name": "Ruins (base)",
            }
        },
        {
            "item": {
                "id": "opaque_wall_6x0.5",
                "shapes": [
                    {
                        "shape_type": "rectangular_prism",
                        "width_inches": 6.0,
                        "depth_inches": 0.5,
                        "height_inches": 5.0,
                    }
                ],
                "name": "Opaque Wall",
            }
        },
    ],
    "features": [
        {
            "item": {
                "id": "crate",
                "feature_type": "obstacle",
                "components": [{"object_id": "crate_5x2.5"}],
            }
        },
        {
            "item": {
                "id": "bare_ruin",
                "feature_type": "obscuring",
                "components": [{"object_id": "ruins_10x6"}],
            }
        },
        {
            "item": {
                "id": "ruin_with_wall",
                "feature_type": "obscuring",
                "components": [
                    {"object_id": "ruins_10x6"},
                    {
                        "object_id": "opaque_wall_6x0.5",
                        "transform": {
                            "x_inches": 2.0,
                            "y_inches": 0.0,
                            "z_inches": 0.0,
                            "rotation_deg": 0.0,
                        },
                    },
                ],
            }
        },
    ],
    "name": "WTC-style Catalog",
}

# Scoring targets matching UI defaults (fixed, not varied)
SCORING_TARGETS: dict[str, Any] = {
    "overall_visibility_target": 30.0,
    "overall_visibility_weight": 1.0,
    "dz_hideability_target": 70.0,
    "dz_hideability_weight": 5.0,
    "objective_hidability_target": 50.0,
    "objective_hidability_weight": 5.0,
}

# ---------------------------------------------------------------------------
# Mission builders (keyed by short name used in the covering array)
# ---------------------------------------------------------------------------

MISSION_BUILDERS = {
    "HnA": _hammer_and_anvil,
    "DoW": _dawn_of_war,
    "TipPt": _tipping_point,
    "SwpEng": _sweeping_engagement,
    "Crucible": _crucible_of_battle,
    "SnD": _search_and_destroy,
}


def _build_mission_json(
    short_name: str, tw: float, td: float, symmetric: bool
) -> dict[str, Any]:
    """Build a mission dict with expanded_polygons computed via shapely."""
    builder = MISSION_BUILDERS[short_name]
    mission = builder(tw, td)
    mission["rotationally_symmetric"] = symmetric

    # Compute expanded DZ polygons (same as compare.py / app.py)
    for dz in mission["deployment_zones"]:
        polys_tuples = [
            [(p["x_inches"], p["z_inches"]) for p in ring]
            for ring in dz["polygons"]
        ]
        expanded = _expand_dz_polygons(polys_tuples, DZ_EXPANSION_INCHES)
        dz["expanded_polygons"] = [
            [{"x_inches": x, "z_inches": z} for x, z in ring]
            for ring in expanded
        ]

    return mission


# ---------------------------------------------------------------------------
# 28-case covering array (9 parameters, pairwise coverage)
# ---------------------------------------------------------------------------

# Each tuple:
#   (table_w, table_d, symmetric, mission_key, catalog_key, steps,
#    min_feature_gap, min_edge_gap, min_all_feature_gap, min_all_edge_gap)
#
# Gap columns encode as binary flags in the benchmark name suffix "_gFEAA":
#   F = min_feature_gap (0=off, 1=5.2")
#   E = min_edge_gap (0=off, 1=5.2")
#   A = min_all_feature_gap (0=off, 1=3")
#   A = min_all_edge_gap (0=off, 1=3")

CaseRow = tuple[float, float, bool, str, str, int, float, float, float, float]

COVERING_ARRAY: list[CaseRow] = [
    # fmt: off
    #    tw  td   sym    mission     catalog  steps  fg   eg   afg  aeg
    (44, 30, False, "none", "crates", 10, 0, 0, 0, 0),  # 01
    (60, 44, False, "none", "wtc", 20, 5.2, 0, 3, 0),  # 02
    (90, 44, False, "none", "crates", 50, 5.2, 5.2, 0, 3),  # 03
    (44, 30, False, "none", "wtc", 100, 0, 5.2, 3, 3),  # 04
    (60, 44, False, "HnA", "crates", 10, 5.2, 0, 0, 3),  # 05
    (90, 44, True, "HnA", "wtc", 20, 0, 5.2, 0, 0),  # 06
    (44, 30, False, "HnA", "wtc", 50, 0, 5.2, 3, 0),  # 07
    (60, 44, True, "HnA", "crates", 100, 5.2, 0, 3, 3),  # 08
    (90, 44, False, "DoW", "wtc", 10, 0, 5.2, 3, 0),  # 09
    (44, 30, True, "DoW", "crates", 20, 5.2, 0, 0, 3),  # 10
    (60, 44, False, "DoW", "crates", 50, 0, 0, 3, 3),  # 11
    (90, 44, True, "DoW", "wtc", 100, 5.2, 5.2, 0, 0),  # 12
    (44, 30, True, "TipPt", "wtc", 10, 5.2, 0, 3, 0),  # 13
    (60, 44, False, "TipPt", "crates", 20, 0, 5.2, 0, 3),  # 14
    (90, 44, True, "TipPt", "crates", 50, 5.2, 0, 0, 3),  # 15
    (44, 30, False, "TipPt", "wtc", 100, 0, 5.2, 3, 0),  # 16
    (60, 44, True, "SwpEng", "wtc", 10, 0, 5.2, 0, 3),  # 17
    (90, 44, False, "SwpEng", "crates", 20, 5.2, 0, 3, 0),  # 18
    (44, 30, True, "SwpEng", "crates", 50, 5.2, 5.2, 0, 3),  # 19
    (60, 44, False, "SwpEng", "wtc", 100, 0, 0, 3, 0),  # 20
    (90, 44, True, "Crucible", "crates", 10, 5.2, 0, 0, 0),  # 21
    (44, 30, False, "Crucible", "wtc", 20, 0, 5.2, 3, 3),  # 22
    (60, 44, True, "Crucible", "wtc", 50, 0, 5.2, 3, 0),  # 23
    (90, 44, False, "Crucible", "crates", 100, 5.2, 0, 0, 3),  # 24
    (44, 30, False, "SnD", "crates", 10, 0, 5.2, 3, 3),  # 25
    (60, 44, True, "SnD", "wtc", 20, 5.2, 0, 0, 0),  # 26
    (90, 44, False, "SnD", "wtc", 50, 0, 0, 3, 0),  # 27
    (44, 30, True, "SnD", "crates", 100, 5.2, 5.2, 0, 3),  # 28
    # fmt: on
]


def _gap_code(fg: float, eg: float, afg: float, aeg: float) -> str:
    """Encode gap params as 4-bit string for benchmark name suffix."""
    return (
        f"{'1' if fg > 0 else '0'}"
        f"{'1' if eg > 0 else '0'}"
        f"{'1' if afg > 0 else '0'}"
        f"{'1' if aeg > 0 else '0'}"
    )


def _bench_name(row: CaseRow) -> str:
    """Generate a short, descriptive benchmark name."""
    tw, td, sym, mission, catalog, steps, fg, eg, afg, aeg = row
    table = f"{int(tw)}x{int(td)}"
    sym_str = "sym" if sym else "nosym"
    gc = _gap_code(fg, eg, afg, aeg)
    return f"{table}_{catalog}_{mission}_{sym_str}_{steps}_g{gc}"


def generate_fixture(row: CaseRow) -> dict[str, Any]:
    """Generate a single benchmark fixture as a JSON-compatible dict."""
    tw, td, symmetric, mission_key, catalog_key, steps, fg, eg, afg, aeg = row
    catalog = CRATES_CATALOG if catalog_key == "crates" else WTC_CATALOG

    params: dict[str, Any] = {
        "seed": 42,
        "table_width_inches": tw,
        "table_depth_inches": td,
        "catalog": catalog,
        "num_steps": steps,
        "scoring_targets": SCORING_TARGETS,
    }

    # Gap params (only include non-zero values)
    if fg > 0:
        params["min_feature_gap_inches"] = fg
    if eg > 0:
        params["min_edge_gap_inches"] = eg
    if afg > 0:
        params["min_all_feature_gap_inches"] = afg
    if aeg > 0:
        params["min_all_edge_gap_inches"] = aeg

    if mission_key != "none":
        params["mission"] = _build_mission_json(mission_key, tw, td, symmetric)
        if symmetric:
            params["rotationally_symmetric"] = True

    return params


def verify_pairwise_coverage() -> list[str]:
    """Verify all pairwise interactions are covered. Returns list of gaps."""
    # Extract parameter columns with labels
    param_names = [
        "table",
        "sym",
        "mission",
        "terrain",
        "steps",
        "feat_gap",
        "edge_gap",
        "all_feat_gap",
        "all_edge_gap",
    ]

    rows: list[list[Any]] = []
    for (
        tw,
        td,
        sym,
        mission,
        catalog,
        steps,
        fg,
        eg,
        afg,
        aeg,
    ) in COVERING_ARRAY:
        rows.append(
            [
                f"{int(tw)}x{int(td)}",
                sym,
                mission,
                catalog,
                steps,
                fg,
                eg,
                afg,
                aeg,
            ]
        )

    # Known intentional constraint: mission=none forces symmetry=off
    allowed_gaps = {
        ("sym", "mission"): {(True, "none")},
    }

    gaps: list[str] = []
    for i, j in combinations(range(len(param_names)), 2):
        # Collect all observed (level_i, level_j) pairs
        observed = set()
        for row in rows:
            observed.add((row[i], row[j]))
        # Compute expected pairs
        levels_i = sorted(set(row[i] for row in rows), key=str)
        levels_j = sorted(set(row[j] for row in rows), key=str)
        expected = {(a, b) for a in levels_i for b in levels_j}
        missing = expected - observed
        # Remove known intentional gaps
        pair_key = (param_names[i], param_names[j])
        if pair_key in allowed_gaps:
            missing -= allowed_gaps[pair_key]
        if missing:
            gaps.append(
                f"  {param_names[i]} x {param_names[j]}: "
                f"missing {len(missing)} pair(s): {missing}"
            )

    return gaps


def generate_readme() -> str:
    """Generate README.md content documenting the covering array."""
    lines = [
        "# Benchmark Fixtures",
        "",
        "Generated by `generate_fixtures.py`. Do not edit manually.",
        "",
        "## Pairwise covering array (28 cases, 9 parameters)",
        "",
        "Parameters:",
        "- **Table**: 44x30 (incursion), 60x44 (strike), 90x44 (onslaught)",
        "- **Symmetry**: off, on (forced off when mission=none)",
        "- **Mission**: none, HnA, DoW, TipPt, SwpEng, Crucible, SnD",
        '- **Terrain**: crates (5" tall), WTC (crates + ruins + walls)',
        "- **Steps**: 10, 20, 50, 100",
        '- **Feature gap**: 0 or 5.2" (min_feature_gap_inches)',
        '- **Edge gap**: 0 or 5.2" (min_edge_gap_inches)',
        '- **All-feature gap**: 0 or 3" (min_all_feature_gap_inches)',
        '- **All-edge gap**: 0 or 3" (min_all_edge_gap_inches)',
        "",
        "All cases use seed=42, visibility enabled, scoring_targets matching",
        "UI defaults (overall=30%/w1, dz_hide=70%/w5, obj_hide=50%/w5).",
        'Mission cases include `expanded_polygons` (6" DZ expansion via shapely).',
        "",
        "Gap suffix `_gFEAA` encodes gap params as 4-bit flags:",
        "  F=feature_gap, E=edge_gap, A=all_feature_gap, A=all_edge_gap",
        '  (0=off, 1=on: 5.2" for F/E, 3" for A/A)',
        "",
        "| # | File | Table | Sym | Mission | Terrain | Steps | Gaps |",
        "|---|------|-------|-----|---------|---------|-------|------|",
    ]

    for i, row in enumerate(COVERING_ARRAY):
        num = i + 1
        tw, td, sym, mission, catalog, steps, fg, eg, afg, aeg = row
        name = _bench_name(row)
        table = f"{int(tw)}x{int(td)}"
        sym_str = "on" if sym else "off"
        gc = _gap_code(fg, eg, afg, aeg)
        lines.append(
            f"| {num:02d} | `{name}.json` | {table} | {sym_str} "
            f"| {mission} | {catalog} | {steps} | {gc} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    # Verify pairwise coverage first
    print("Verifying pairwise coverage...")
    gaps = verify_pairwise_coverage()
    if gaps:
        print("COVERAGE GAPS FOUND:")
        for gap in gaps:
            print(gap)
        sys.exit(1)
    print("  All pairwise interactions covered.")

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    n = len(COVERING_ARRAY)
    print(f"\nGenerating {n} benchmark fixtures...")

    for i, row in enumerate(COVERING_ARRAY):
        name = _bench_name(row)
        fixture = generate_fixture(row)
        path = fixtures_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(fixture, f, indent=2)
        print(f"  [{i + 1:02d}/{n}] {path.name}")

    # Write README
    readme_path = fixtures_dir / "README.md"
    readme_path.write_text(generate_readme())
    print("  README.md written")

    print(f"\nDone. {n} fixtures in {fixtures_dir}/")


if __name__ == "__main__":
    main()
