"""Tests for mission data definitions and engine pass-through."""

from engine.generate import generate
from engine.types import (
    DeploymentZone,
    EngineParams,
    Mission,
    ObjectiveMarker,
    Point2D,
    TerrainCatalog,
)
from frontend.missions import EDITIONS, find_mission_path, get_mission

# ---------------------------------------------------------------------------
# Mission data structure validity
# ---------------------------------------------------------------------------

# Default table size used for structural tests
_TW, _TD = 60.0, 44.0


def _build_all_missions():
    """Yield (ed_name, pack_name, dep_name, mission_dict) for every mission."""
    for ed_name, packs in EDITIONS.items():
        for pack_name, deployments in packs.items():
            for dep_name, builder in deployments.items():
                yield ed_name, pack_name, dep_name, builder(_TW, _TD)


def test_all_missions_have_required_fields():
    """Every mission in EDITIONS has name, deployment_zones, objectives."""
    for ed_name, pack_name, dep_name, mission in _build_all_missions():
        assert "name" in mission, f"{ed_name}/{pack_name}/{dep_name}"
        assert "deployment_zones" in mission
        assert "objectives" in mission
        assert "rotationally_symmetric" in mission
        assert isinstance(mission["rotationally_symmetric"], bool)


def test_all_deployment_zones_have_polygons():
    """Each DZ has an id and at least one polygon with at least 3 vertices."""
    for _ed, _pk, _dep, mission in _build_all_missions():
        dz_list: list[dict] = mission["deployment_zones"]
        for dz in dz_list:
            assert "id" in dz
            assert "polygons" in dz
            for poly in dz["polygons"]:
                assert len(poly) >= 3
                for pt in poly:
                    assert "x_inches" in pt
                    assert "z_inches" in pt


def test_all_objectives_have_position_and_range():
    """Each objective has id, position, and range_inches."""
    for _ed, _pk, _dep, mission in _build_all_missions():
        obj_list: list[dict] = mission["objectives"]
        for obj in obj_list:
            assert "id" in obj
            assert "position" in obj
            pos: dict = obj["position"]
            assert "x_inches" in pos
            assert "z_inches" in pos
            assert "range_inches" in obj
            range_val: float = obj["range_inches"]
            assert range_val > 0


def test_get_mission_found():
    m = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Hammer and Anvil",
    )
    assert m is not None
    assert m["name"] == "Hammer and Anvil"


def test_get_mission_not_found():
    assert get_mission("99th Edition", "Fake Pack", "Fake") is None


def test_find_mission_path_found():
    path = find_mission_path("Dawn of War")
    assert path is not None
    ed, pk, dep = path
    assert ed == "10th Edition"
    assert dep == "Dawn of War"


def test_find_mission_path_not_found():
    assert find_mission_path("Nonexistent Mission") is None


# ---------------------------------------------------------------------------
# Parametric scaling tests
# ---------------------------------------------------------------------------


def test_default_table_size_matches_original():
    """Spot-check known 60x44 values to ensure builders match original static data."""
    m = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Hammer and Anvil",
    )
    assert m is not None
    # Green DZ: left 18" strip
    green_poly = m["deployment_zones"][0]["polygons"][0]
    assert green_poly[0] == {"x_inches": -30.0, "z_inches": -22.0}
    assert green_poly[1] == {"x_inches": -12.0, "z_inches": -22.0}
    assert green_poly[2] == {"x_inches": -12.0, "z_inches": 22.0}
    assert green_poly[3] == {"x_inches": -30.0, "z_inches": 22.0}
    # Obj 4: 10" from left edge
    assert m["objectives"][3]["position"] == {
        "x_inches": -20.0,
        "z_inches": 0.0,
    }


def test_missions_scale_with_table_dimensions():
    """All missions produce valid structure at non-standard 80x60 size."""
    for _ed, packs in EDITIONS.items():
        for _pk, deployments in packs.items():
            for dep_name, builder in deployments.items():
                mission = builder(80.0, 60.0)
                assert mission["name"] == dep_name
                assert len(mission["deployment_zones"]) == 2
                assert len(mission["objectives"]) == 5


def test_hammer_and_anvil_scales_dz_width():
    """DZ outer edges track table boundary; interior stays 18" from edge."""
    m = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Hammer and Anvil",
        table_width=72.0,
        table_depth=48.0,
    )
    assert m is not None
    # hw=36, hd=24
    green_poly = m["deployment_zones"][0]["polygons"][0]
    assert green_poly[0]["x_inches"] == -36.0  # outer edge = -hw
    assert green_poly[1]["x_inches"] == -18.0  # inner edge = -hw+18
    assert green_poly[0]["z_inches"] == -24.0  # top = -hd
    assert green_poly[2]["z_inches"] == 24.0  # bottom = hd

    # Objectives scale too
    obj4 = m["objectives"][3]["position"]
    assert obj4["x_inches"] == -26.0  # -hw+10 = -36+10


def test_center_objective_unchanged_across_sizes():
    """Center objective stays at (0, 0) regardless of table size."""
    for tw, td in [(60, 44), (80, 60), (40, 30), (100, 72)]:
        for _ed, packs in EDITIONS.items():
            for _pk, deployments in packs.items():
                for _dep, builder in deployments.items():
                    mission = builder(tw, td)
                    center = mission["objectives"][0]["position"]
                    assert center == {"x_inches": 0.0, "z_inches": 0.0}


def test_search_and_destroy_arc_fixed():
    """9\" arc points around center stay center-relative (don't scale)."""
    m80 = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Search and Destroy",
        table_width=80.0,
        table_depth=60.0,
    )
    m60 = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Search and Destroy",
        table_width=60.0,
        table_depth=44.0,
    )
    assert m80 is not None and m60 is not None
    # Arc points (indices 2-8 in green polygon) should be identical
    green80 = m80["deployment_zones"][0]["polygons"][0]
    green60 = m60["deployment_zones"][0]["polygons"][0]
    for i in range(2, 9):
        assert green80[i] == green60[i], f"Arc point {i} differs"

    # But corner points should differ (they use hw/hd)
    assert green80[0]["x_inches"] != green60[0]["x_inches"]


def test_dawn_of_war_dz_depth_scales():
    """Dawn of War DZ strips track long edges."""
    m = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Dawn of War",
        table_width=72.0,
        table_depth=52.0,
    )
    assert m is not None
    # hd=26, red DZ: z from -26 to -16
    red_poly = m["deployment_zones"][0]["polygons"][0]
    assert red_poly[0]["z_inches"] == -26.0  # -hd
    assert red_poly[2]["z_inches"] == -16.0  # -hd+10


# ---------------------------------------------------------------------------
# Mission from_dict / to_dict roundtrip
# ---------------------------------------------------------------------------


def test_point2d_roundtrip():
    d = {"x_inches": 1.5, "z_inches": -3.0}
    p = Point2D.from_dict(d)
    assert p.x == 1.5
    assert p.z == -3.0
    assert p.to_dict() == d


def test_objective_marker_roundtrip():
    d = {
        "id": "A",
        "position": {"x_inches": 0.0, "z_inches": 0.0},
        "range_inches": 3.0,
    }
    o = ObjectiveMarker.from_dict(d)
    assert o.id == "A"
    assert o.range_inches == 3.0
    assert o.to_dict() == d


def test_objective_marker_default_range():
    d = {"id": "B", "position": {"x_inches": 1.0, "z_inches": 2.0}}
    o = ObjectiveMarker.from_dict(d)
    assert o.range_inches == 3.0


def test_deployment_zone_roundtrip():
    d = {
        "id": "red",
        "polygons": [
            [
                {"x_inches": -30.0, "z_inches": -22.0},
                {"x_inches": -12.0, "z_inches": -22.0},
                {"x_inches": -12.0, "z_inches": 22.0},
                {"x_inches": -30.0, "z_inches": 22.0},
            ]
        ],
    }
    dz = DeploymentZone.from_dict(d)
    assert dz.id == "red"
    assert len(dz.polygons) == 1
    assert len(dz.polygons[0]) == 4
    assert dz.to_dict() == d


def test_mission_roundtrip():
    d = {
        "name": "Test Mission",
        "rotationally_symmetric": True,
        "objectives": [
            {
                "id": "1",
                "position": {"x_inches": 0.0, "z_inches": 0.0},
                "range_inches": 3.0,
            }
        ],
        "deployment_zones": [
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": -30.0, "z_inches": -22.0},
                        {"x_inches": -12.0, "z_inches": -22.0},
                        {"x_inches": -12.0, "z_inches": 22.0},
                        {"x_inches": -30.0, "z_inches": 22.0},
                    ]
                ],
            }
        ],
    }
    m = Mission.from_dict(d)
    assert m.name == "Test Mission"
    assert m.rotationally_symmetric is True
    assert len(m.objectives) == 1
    assert len(m.deployment_zones) == 1
    assert m.to_dict() == d


def test_mission_from_missions_module_roundtrip():
    """Mission data from EDITIONS survives from_dict/to_dict."""
    raw = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Hammer and Anvil",
    )
    assert raw is not None
    m = Mission.from_dict(raw)
    roundtripped = m.to_dict()
    assert roundtripped["name"] == raw["name"]
    assert len(roundtripped["objectives"]) == len(raw["objectives"])
    assert len(roundtripped["deployment_zones"]) == len(
        raw["deployment_zones"]
    )


# ---------------------------------------------------------------------------
# Engine pass-through
# ---------------------------------------------------------------------------


def _crate_catalog_dict():
    return {
        "objects": [
            {
                "item": {
                    "id": "crate",
                    "shapes": [
                        {
                            "shape_type": "rectangular_prism",
                            "width_inches": 5.0,
                            "depth_inches": 2.5,
                            "height_inches": 2.0,
                        }
                    ],
                },
            }
        ],
        "features": [
            {
                "item": {
                    "id": "crate",
                    "feature_type": "obstacle",
                    "components": [{"object_id": "crate"}],
                },
            }
        ],
    }


def test_engine_passthrough_with_mission():
    """Mission in params appears in output layout unchanged."""
    mission_dict = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Hammer and Anvil",
    )
    assert mission_dict is not None
    params = EngineParams(
        seed=42,
        table_width=60.0,
        table_depth=44.0,
        catalog=TerrainCatalog.from_dict(_crate_catalog_dict()),
        num_steps=10,
        mission=Mission.from_dict(mission_dict),
    )
    result = generate(params)
    assert result.layout.mission is not None
    assert result.layout.mission.name == "Hammer and Anvil"
    assert len(result.layout.mission.objectives) == 5
    assert len(result.layout.mission.deployment_zones) == 2


def test_engine_passthrough_no_mission():
    """Without mission, layout.mission is None."""
    params = EngineParams(
        seed=42,
        table_width=60.0,
        table_depth=44.0,
        catalog=TerrainCatalog.from_dict(_crate_catalog_dict()),
        num_steps=10,
    )
    result = generate(params)
    assert result.layout.mission is None


def test_layout_to_dict_includes_mission():
    """Mission survives layout.to_dict()."""
    mission_dict = get_mission(
        "10th Edition",
        "Matched Play: Chapter Approved 2025-26",
        "Dawn of War",
    )
    assert mission_dict is not None
    params = EngineParams(
        seed=42,
        table_width=60.0,
        table_depth=44.0,
        catalog=TerrainCatalog.from_dict(_crate_catalog_dict()),
        num_steps=5,
        mission=Mission.from_dict(mission_dict),
    )
    result = generate(params)
    d = result.layout.to_dict()
    assert "mission" in d
    assert d["mission"]["name"] == "Dawn of War"


def test_layout_to_dict_no_mission():
    """Without mission, to_dict omits the mission key."""
    params = EngineParams(
        seed=42,
        table_width=60.0,
        table_depth=44.0,
        catalog=TerrainCatalog.from_dict(_crate_catalog_dict()),
        num_steps=5,
    )
    result = generate(params)
    d = result.layout.to_dict()
    assert "mission" not in d
