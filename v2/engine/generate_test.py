import math

from engine.collision import (
    get_tall_world_obbs,
    get_world_obbs,
    obb_corners,
    obb_distance,
    obb_in_bounds,
    obb_to_table_edge_distance,
    obbs_overlap,
)
from engine.generate import generate, generate_json
from engine.prng import PCG32
from engine.types import EngineParams


def _crate_catalog_dict():
    return {
        "objects": [
            {
                "item": {
                    "id": "crate_5x2.5",
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
                    "components": [{"object_id": "crate_5x2.5"}],
                },
            }
        ],
    }


def _make_params_dict(seed=42, num_steps=50, table_w=60.0, table_d=44.0):
    return {
        "seed": seed,
        "table_width_inches": table_w,
        "table_depth_inches": table_d,
        "catalog": _crate_catalog_dict(),
        "num_steps": num_steps,
    }


# -- PCG32 ---------------------------------------------------------


class TestPCG32:
    def test_reference_values(self):
        """PCG32(seed=42, seq=54) matches C reference output."""
        rng = PCG32(seed=42, seq=54)
        expected = [
            0xA15C02B7,
            0x7B47F409,
            0xBA1D3330,
            0x83D2F293,
            0xBFA4784B,
        ]
        for exp in expected:
            assert rng.next_u32() == exp

    def test_next_float_range(self):
        rng = PCG32(seed=1)
        for _ in range(1000):
            f = rng.next_float()
            assert 0.0 <= f < 1.0

    def test_next_int_range(self):
        rng = PCG32(seed=1)
        for _ in range(1000):
            v = rng.next_int(3, 7)
            assert 3 <= v <= 7


# -- Collision ------------------------------------------------------


class TestCollision:
    def test_separated_no_overlap(self):
        a = obb_corners(0, 0, 2.5, 1.25, 0)
        b = obb_corners(10, 0, 2.5, 1.25, 0)
        assert not obbs_overlap(a, b)

    def test_overlapping(self):
        a = obb_corners(0, 0, 2.5, 1.25, 0)
        b = obb_corners(3, 0, 2.5, 1.25, 0)
        assert obbs_overlap(a, b)

    def test_touching_no_overlap(self):
        """Crates sharing an edge do not count as overlapping."""
        a = obb_corners(0, 0, 2.5, 1.25, 0)
        b = obb_corners(5, 0, 2.5, 1.25, 0)
        assert not obbs_overlap(a, b)

    def test_touching_corner_no_overlap(self):
        """Crates sharing only a corner do not overlap."""
        a = obb_corners(0, 0, 2.5, 1.25, 0)
        b = obb_corners(5, 2.5, 2.5, 1.25, 0)
        assert not obbs_overlap(a, b)

    def test_rotated_same_center_overlap(self):
        a = obb_corners(0, 0, 2.5, 1.25, 0)
        b = obb_corners(0, 0, 2.5, 1.25, math.radians(45))
        assert obbs_overlap(a, b)

    def test_in_bounds(self):
        corners = obb_corners(0, 0, 2.5, 1.25, 0)
        assert obb_in_bounds(corners, 60, 44)

    def test_out_of_bounds(self):
        # Center at x=29, half_w=2.5 → right edge at 31.5
        corners = obb_corners(29, 0, 2.5, 1.25, 0)
        assert not obb_in_bounds(corners, 60, 44)

    def test_touching_edge_in_bounds(self):
        """Corner exactly on the table edge is valid."""
        # Center at x=27.5, half_w=2.5 → right edge at 30
        corners = obb_corners(27.5, 0, 2.5, 1.25, 0)
        assert obb_in_bounds(corners, 60, 44)


# -- Generate -------------------------------------------------------


class TestGenerate:
    def test_deterministic(self):
        """Same seed produces identical output."""
        p1 = EngineParams.from_dict(_make_params_dict(seed=123, num_steps=200))
        p2 = EngineParams.from_dict(_make_params_dict(seed=123, num_steps=200))
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() == r2.layout.to_dict()

    def test_different_seeds(self):
        """Different seeds produce different layouts."""
        p1 = EngineParams.from_dict(_make_params_dict(seed=1, num_steps=200))
        p2 = EngineParams.from_dict(_make_params_dict(seed=2, num_steps=200))
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() != r2.layout.to_dict()

    def test_produces_features(self):
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200)
        )
        result = generate(params)
        assert len(result.layout.placed_features) > 0

    def test_all_features_in_bounds(self):
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200)
        )
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}
        for pf in result.layout.placed_features:
            obbs = get_world_obbs(pf, objects_by_id)
            for corners in obbs:
                assert obb_in_bounds(
                    corners,
                    params.table_width,
                    params.table_depth,
                )

    def test_no_overlaps(self):
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200)
        )
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}
        features = result.layout.placed_features
        for i in range(len(features)):
            obbs_i = get_world_obbs(features[i], objects_by_id)
            for j in range(i + 1, len(features)):
                obbs_j = get_world_obbs(features[j], objects_by_id)
                for ca in obbs_i:
                    for cb in obbs_j:
                        assert not obbs_overlap(ca, cb), (
                            f"Features {i} and {j} overlap"
                        )

    def test_empty_catalog(self):
        """Empty catalog produces empty layout."""
        params_dict = {
            "seed": 42,
            "table_width_inches": 60,
            "table_depth_inches": 44,
            "catalog": {},
            "num_steps": 50,
        }
        result = generate_json(params_dict)
        assert result["layout"]["placed_features"] == []

    def test_edge_gap_enforcement(self):
        """All tall shapes respect edge gap constraint."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200)
        )
        params.min_edge_gap_inches = 3.0
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}

        for pf in result.layout.placed_features:
            tall_obbs = get_tall_world_obbs(pf, objects_by_id, min_height=1.0)
            for corners in tall_obbs:
                dist = obb_to_table_edge_distance(
                    corners,
                    params.table_width,
                    params.table_depth,
                )
                assert dist >= params.min_edge_gap_inches - 1e-6, (
                    f"Shape too close to edge: dist={dist}, min={params.min_edge_gap_inches}"
                )

    def test_feature_gap_enforcement(self):
        """All tall shape pairs respect feature gap constraint."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200)
        )
        params.min_feature_gap_inches = 2.0
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}

        features = result.layout.placed_features
        for i in range(len(features)):
            tall_i = get_tall_world_obbs(
                features[i], objects_by_id, min_height=1.0
            )
            for j in range(i + 1, len(features)):
                tall_j = get_tall_world_obbs(
                    features[j], objects_by_id, min_height=1.0
                )
                for ca in tall_i:
                    for cb in tall_j:
                        dist = obb_distance(ca, cb)
                        assert dist >= params.min_feature_gap_inches - 1e-6, (
                            f"Features {i} and {j} too close: dist={dist}, min={params.min_feature_gap_inches}"
                        )

    def test_short_shapes_ignored_in_gap_checks(self):
        """Shapes with height < 1" do not participate in gap checks."""
        # Create a catalog with mixed heights
        catalog_dict = {
            "objects": [
                {
                    "item": {
                        "id": "tall_crate",
                        "shapes": [
                            {
                                "shape_type": "rectangular_prism",
                                "width_inches": 5.0,
                                "depth_inches": 2.5,
                                "height_inches": 2.0,  # Tall
                            }
                        ],
                    },
                },
                {
                    "item": {
                        "id": "short_scatter",
                        "shapes": [
                            {
                                "shape_type": "rectangular_prism",
                                "width_inches": 3.0,
                                "depth_inches": 3.0,
                                "height_inches": 0.5,  # Short (< 1")
                            }
                        ],
                    },
                },
            ],
            "features": [
                {
                    "item": {
                        "id": "tall",
                        "feature_type": "obstacle",
                        "components": [{"object_id": "tall_crate"}],
                    },
                },
                {
                    "item": {
                        "id": "short",
                        "feature_type": "scatter",
                        "components": [{"object_id": "short_scatter"}],
                    },
                },
            ],
        }
        params_dict = {
            "seed": 42,
            "table_width_inches": 60,
            "table_depth_inches": 44,
            "catalog": catalog_dict,
            "num_steps": 200,
            "min_feature_gap_inches": 10.0,  # Large gap
            "min_edge_gap_inches": 5.0,
        }
        result = generate_json(params_dict)
        # With large gaps, should still produce features since short
        # shapes don't participate in gap checks
        assert len(result["layout"]["placed_features"]) > 0

    def test_determinism_with_gaps(self):
        """Same seed with gaps produces identical layouts."""
        params1 = EngineParams.from_dict(
            _make_params_dict(seed=123, num_steps=200)
        )
        params1.min_feature_gap_inches = 2.0
        params1.min_edge_gap_inches = 3.0

        params2 = EngineParams.from_dict(
            _make_params_dict(seed=123, num_steps=200)
        )
        params2.min_feature_gap_inches = 2.0
        params2.min_edge_gap_inches = 3.0

        result1 = generate(params1)
        result2 = generate(params2)
        assert result1.layout.to_dict() == result2.layout.to_dict()

    def test_no_gaps_specified(self):
        """With no gaps specified, generation works normally."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200)
        )
        # min_feature_gap_inches and min_edge_gap_inches are None by default
        result = generate(params)
        assert len(result.layout.placed_features) > 0

    def test_symmetric_deterministic(self):
        """Symmetric mode is deterministic."""
        pd = _make_params_dict(seed=42, num_steps=100)
        pd["rotationally_symmetric"] = True
        p1 = EngineParams.from_dict(pd)
        p2 = EngineParams.from_dict(pd)
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() == r2.layout.to_dict()

    def test_symmetric_flag_on_output(self):
        """Output layout has rotationally_symmetric flag."""
        pd = _make_params_dict(seed=42, num_steps=50)
        pd["rotationally_symmetric"] = True
        params = EngineParams.from_dict(pd)
        result = generate(params)
        assert result.layout.rotationally_symmetric is True
        d = result.layout.to_dict()
        assert d["rotationally_symmetric"] is True

    def test_symmetric_no_self_overlap(self):
        """Symmetric features don't overlap their own mirrors."""
        pd = _make_params_dict(seed=42, num_steps=200)
        pd["rotationally_symmetric"] = True
        params = EngineParams.from_dict(pd)
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}

        from engine.collision import _is_at_origin, _mirror_placed_feature

        for pf in result.layout.placed_features:
            if _is_at_origin(pf):
                continue
            mirror = _mirror_placed_feature(pf)
            obbs_orig = get_world_obbs(pf, objects_by_id)
            obbs_mirror = get_world_obbs(mirror, objects_by_id)
            for ca in obbs_orig:
                for cb in obbs_mirror:
                    assert not obbs_overlap(ca, cb), (
                        f"Feature at ({pf.transform.x}, {pf.transform.z}) "
                        f"overlaps its mirror"
                    )


# -- JSON -----------------------------------------------------------


class TestJSON:
    def test_round_trip(self):
        params = _make_params_dict(seed=42, num_steps=50)
        result = generate_json(params)
        assert "layout" in result
        layout = result["layout"]
        assert "placed_features" in layout
        assert layout["table_width_inches"] == 60.0
        assert layout["table_depth_inches"] == 44.0
        assert result["steps_completed"] == 50

    def test_features_have_schema_fields(self):
        params = _make_params_dict(seed=42, num_steps=200)
        result = generate_json(params)
        for pf in result["layout"]["placed_features"]:
            assert "feature" in pf
            assert "transform" in pf
            feat = pf["feature"]
            assert "id" in feat
            assert "feature_type" in feat
            assert "components" in feat
            t = pf["transform"]
            assert "x_inches" in t
            assert "z_inches" in t
            assert "rotation_deg" in t
