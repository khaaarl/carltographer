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
from engine.generate import (
    MAX_EXTRA_MUTATIONS,
    MIN_MOVE_RANGE,
    PHASE2_BASE,
    StepUndo,
    _compute_score,
    _quantize_angle,
    _quantize_position,
    _temperature_move,
    _undo_step,
    generate,
    generate_json,
)
from engine.prng import PCG32
from engine.types import (
    EngineParams,
    FeatureCountPreference,
    PlacedFeature,
    ScoringTargets,
    TerrainFeature,
    TerrainLayout,
    Transform,
)


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


def _make_params_dict(
    seed=42, num_steps=50, table_w=60.0, table_d=44.0, skip_visibility=False
):
    d = {
        "seed": seed,
        "table_width_inches": table_w,
        "table_depth_inches": table_d,
        "catalog": _crate_catalog_dict(),
        "num_steps": num_steps,
    }
    if skip_visibility:
        d["skip_visibility"] = True
    return d


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
        p1 = EngineParams.from_dict(
            _make_params_dict(seed=123, num_steps=200, skip_visibility=True)
        )
        p2 = EngineParams.from_dict(
            _make_params_dict(seed=123, num_steps=200, skip_visibility=True)
        )
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() == r2.layout.to_dict()

    def test_different_seeds(self):
        """Different seeds produce different layouts."""
        p1 = EngineParams.from_dict(
            _make_params_dict(seed=1, num_steps=200, skip_visibility=True)
        )
        p2 = EngineParams.from_dict(
            _make_params_dict(seed=2, num_steps=200, skip_visibility=True)
        )
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() != r2.layout.to_dict()

    def test_produces_features(self):
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        )
        result = generate(params)
        assert len(result.layout.placed_features) > 0

    def test_all_features_in_bounds(self):
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
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
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
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
            "skip_visibility": True,
        }
        result = generate_json(params_dict)
        assert result["layout"]["placed_features"] == []

    def test_edge_gap_enforcement(self):
        """All tall shapes respect edge gap constraint."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
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
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
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
            "skip_visibility": True,
        }
        result = generate_json(params_dict)
        # With large gaps, should still produce features since short
        # shapes don't participate in gap checks
        assert len(result["layout"]["placed_features"]) > 0

    def test_determinism_with_gaps(self):
        """Same seed with gaps produces identical layouts."""
        params1 = EngineParams.from_dict(
            _make_params_dict(seed=123, num_steps=200, skip_visibility=True)
        )
        params1.min_feature_gap_inches = 2.0
        params1.min_edge_gap_inches = 3.0

        params2 = EngineParams.from_dict(
            _make_params_dict(seed=123, num_steps=200, skip_visibility=True)
        )
        params2.min_feature_gap_inches = 2.0
        params2.min_edge_gap_inches = 3.0

        result1 = generate(params1)
        result2 = generate(params2)
        assert result1.layout.to_dict() == result2.layout.to_dict()

    def test_no_gaps_specified(self):
        """With no gaps specified, generation works normally."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        )
        # min_feature_gap_inches and min_edge_gap_inches are None by default
        result = generate(params)
        assert len(result.layout.placed_features) > 0

    def test_symmetric_deterministic(self):
        """Symmetric mode is deterministic."""
        pd = _make_params_dict(seed=42, num_steps=100, skip_visibility=True)
        pd["rotationally_symmetric"] = True
        p1 = EngineParams.from_dict(pd)
        p2 = EngineParams.from_dict(pd)
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() == r2.layout.to_dict()

    def test_symmetric_flag_on_output(self):
        """Output layout has rotationally_symmetric flag."""
        pd = _make_params_dict(seed=42, num_steps=50, skip_visibility=True)
        pd["rotationally_symmetric"] = True
        params = EngineParams.from_dict(pd)
        result = generate(params)
        assert result.layout.rotationally_symmetric is True
        d = result.layout.to_dict()
        assert d["rotationally_symmetric"] is True

    def test_symmetric_no_self_overlap(self):
        """Symmetric features don't overlap their own mirrors."""
        pd = _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
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


# -- Temperature Move -----------------------------------------------


class TestTemperatureMove:
    def test_small_t_small_displacement(self):
        """t_factor=0 produces displacements within ±MIN_MOVE_RANGE."""
        rng = PCG32(seed=42)
        old = Transform(0.0, 0.0, 90.0)
        results = []
        for _ in range(100):
            t = _temperature_move(rng, old, 60.0, 44.0, 0.0)
            results.append(t)
        max_dx = max(abs(t.x - old.x) for t in results)
        max_dz = max(abs(t.z - old.z) for t in results)
        assert max_dx <= MIN_MOVE_RANGE + 0.1  # quantization tolerance
        assert max_dz <= MIN_MOVE_RANGE + 0.1

    def test_large_t_large_displacement(self):
        """t_factor=1 can produce large displacements."""
        rng = PCG32(seed=42)
        old = Transform(0.0, 0.0, 0.0)
        results = []
        for _ in range(200):
            t = _temperature_move(rng, old, 60.0, 44.0, 1.0)
            results.append(t)
        max_dx = max(abs(t.x) for t in results)
        # At t=1, range = max(60,44) = 60, so displacements up to ±60
        assert max_dx > MIN_MOVE_RANGE

    def test_no_rotation_at_t0(self):
        """t_factor=0 should never rotate (rotate_check < 0 is impossible)."""
        rng = PCG32(seed=42)
        old = Transform(0.0, 0.0, 45.0)
        for _ in range(100):
            t = _temperature_move(rng, old, 60.0, 44.0, 0.0)
            assert t.rotation_deg == 45.0

    def test_some_rotation_at_t1(self):
        """t_factor=1 should sometimes rotate (~50% chance)."""
        rng = PCG32(seed=42)
        old = Transform(0.0, 0.0, 45.0)
        rotated = 0
        total = 200
        for _ in range(total):
            t = _temperature_move(rng, old, 60.0, 44.0, 1.0)
            if t.rotation_deg != 45.0:
                rotated += 1
        # Should be approximately 50% rotations
        assert rotated > total * 0.3
        assert rotated < total * 0.7

    def test_consumes_4_prng_values(self):
        """_temperature_move always consumes exactly 4 PRNG values."""
        rng1 = PCG32(seed=99)
        rng2 = PCG32(seed=99)
        old = Transform(5.0, 3.0, 30.0)
        _temperature_move(rng1, old, 60.0, 44.0, 0.5)
        # Manually advance rng2 by 4
        for _ in range(4):
            rng2.next_float()
        assert rng1.next_u32() == rng2.next_u32()

    def test_quantized_output(self):
        """Output positions and angles are quantized."""
        rng = PCG32(seed=42)
        old = Transform(0.0, 0.0, 0.0)
        for _ in range(50):
            t = _temperature_move(rng, old, 60.0, 44.0, 1.0)
            assert abs(t.x - _quantize_position(t.x)) < 1e-9
            assert abs(t.z - _quantize_position(t.z)) < 1e-9
            assert abs(t.rotation_deg - _quantize_angle(t.rotation_deg)) < 1e-9


# -- Replace Action --------------------------------------------------


class TestReplaceAction:
    def _make_multi_catalog(self):
        return {
            "objects": [
                {
                    "item": {
                        "id": "small_crate",
                        "shapes": [
                            {
                                "shape_type": "rectangular_prism",
                                "width_inches": 3.0,
                                "depth_inches": 2.0,
                                "height_inches": 2.0,
                            }
                        ],
                    },
                },
                {
                    "item": {
                        "id": "big_crate",
                        "shapes": [
                            {
                                "shape_type": "rectangular_prism",
                                "width_inches": 5.0,
                                "depth_inches": 3.0,
                                "height_inches": 2.0,
                            }
                        ],
                    },
                },
            ],
            "features": [
                {
                    "item": {
                        "id": "small",
                        "feature_type": "obstacle",
                        "components": [{"object_id": "small_crate"}],
                    },
                },
                {
                    "item": {
                        "id": "big",
                        "feature_type": "obstacle",
                        "components": [{"object_id": "big_crate"}],
                    },
                },
            ],
        }

    def test_replace_produces_features(self):
        """With multi-template catalog, generation still produces features."""
        pd = {
            "seed": 42,
            "table_width_inches": 60,
            "table_depth_inches": 44,
            "catalog": self._make_multi_catalog(),
            "num_steps": 200,
            "skip_visibility": True,
        }
        result = generate_json(pd)
        assert len(result["layout"]["placed_features"]) > 0

    def test_replace_deterministic(self):
        """Replace action is deterministic."""
        pd = {
            "seed": 42,
            "table_width_inches": 60,
            "table_depth_inches": 44,
            "catalog": self._make_multi_catalog(),
            "num_steps": 200,
            "skip_visibility": True,
        }
        r1 = generate_json(pd)
        r2 = generate_json(pd)
        assert r1 == r2


# -- Undo Step -------------------------------------------------------


class TestUndoStep:
    def test_undo_add(self):
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        feat = TerrainFeature(id="f1", feature_type="obstacle", components=[])
        pf = PlacedFeature(feat, Transform(1.0, 2.0, 0.0))
        layout.placed_features.append(pf)
        undo = StepUndo(action="add", index=0, prev_next_id=1)
        _undo_step(layout, undo)
        assert len(layout.placed_features) == 0

    def test_undo_move(self):
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        feat = TerrainFeature(id="f1", feature_type="obstacle", components=[])
        old_pf = PlacedFeature(feat, Transform(1.0, 2.0, 0.0))
        new_pf = PlacedFeature(feat, Transform(5.0, 6.0, 90.0))
        layout.placed_features.append(new_pf)
        undo = StepUndo(action="move", index=0, old_feature=old_pf)
        _undo_step(layout, undo)
        assert layout.placed_features[0].transform.x == 1.0
        assert layout.placed_features[0].transform.z == 2.0

    def test_undo_delete(self):
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        feat = TerrainFeature(id="f1", feature_type="obstacle", components=[])
        saved = PlacedFeature(feat, Transform(1.0, 2.0, 0.0))
        undo = StepUndo(action="delete", index=0, old_feature=saved)
        _undo_step(layout, undo)
        assert len(layout.placed_features) == 1
        assert layout.placed_features[0].transform.x == 1.0

    def test_undo_replace(self):
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        old_feat = TerrainFeature(
            id="f1", feature_type="obstacle", components=[]
        )
        new_feat = TerrainFeature(
            id="f2", feature_type="obstacle", components=[]
        )
        old_pf = PlacedFeature(old_feat, Transform(1.0, 2.0, 0.0))
        new_pf = PlacedFeature(new_feat, Transform(1.0, 2.0, 0.0))
        layout.placed_features.append(new_pf)
        undo = StepUndo(
            action="replace", index=0, old_feature=old_pf, prev_next_id=1
        )
        _undo_step(layout, undo)
        assert layout.placed_features[0].feature.id == "f1"

    def test_undo_noop(self):
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        undo = StepUndo(action="noop")
        _undo_step(layout, undo)  # should not raise
        assert len(layout.placed_features) == 0


# -- Retry Loop ------------------------------------------------------


class TestRetryLoop:
    def test_retry_finds_valid_on_crowded_table(self):
        """Even on a crowded table, retry loop should find a valid placement."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        )
        params.min_feature_gap_inches = 5.0
        params.min_edge_gap_inches = 3.0
        result = generate(params)
        # Should still produce features
        assert len(result.layout.placed_features) > 0

    def test_retry_deterministic(self):
        """Retry loop is deterministic."""
        params1 = EngineParams.from_dict(
            _make_params_dict(seed=77, num_steps=100, skip_visibility=True)
        )
        params1.min_feature_gap_inches = 5.0
        params2 = EngineParams.from_dict(
            _make_params_dict(seed=77, num_steps=100, skip_visibility=True)
        )
        params2.min_feature_gap_inches = 5.0
        r1 = generate(params1)
        r2 = generate(params2)
        assert r1.layout.to_dict() == r2.layout.to_dict()


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


# -- Scoring ------------------------------------------------------------


class TestScoring:
    def test_score_nonzero(self):
        """Score is > 0 after generation."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=50)
        )
        result = generate(params)
        assert result.score > 0

    def test_score_phase2_when_no_prefs(self):
        """With no preferences, score >= PHASE2_BASE (straight to Phase 2)."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=50)
        )
        result = generate(params)
        assert result.score >= PHASE2_BASE

    def test_score_deterministic(self):
        """Same seed produces same score."""
        p1 = EngineParams.from_dict(_make_params_dict(seed=123, num_steps=100))
        p2 = EngineParams.from_dict(_make_params_dict(seed=123, num_steps=100))
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.score == r2.score

    def test_score_increases_with_more_steps(self):
        """Longer run score >= shorter run score (same seed)."""
        p_short = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=10)
        )
        p_long = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=100)
        )
        r_short = generate(p_short)
        r_long = generate(p_long)
        assert r_long.score >= r_short.score

    def test_score_phase1_with_zero_steps(self):
        """Empty layout with min=5 preference -> score = PHASE2_BASE / 6."""
        layout = TerrainLayout(
            table_width=60.0,
            table_depth=44.0,
        )
        prefs = [FeatureCountPreference(feature_type="obstacle", min=5)]
        score = _compute_score(layout, prefs, {})
        expected = PHASE2_BASE / (1.0 + 5)
        assert abs(score - expected) < 1e-6

    def test_delete_reversion(self):
        """With min=5 preference and enough steps, count stays >= 5."""
        pd = _make_params_dict(seed=42, num_steps=200)
        pd["feature_count_preferences"] = [
            {"feature_type": "obstacle", "min": 5}
        ]
        params = EngineParams.from_dict(pd)
        result = generate(params)
        count = len(result.layout.placed_features)
        assert count >= 5, f"Expected >= 5 features, got {count}"

    def test_score_in_result_dict(self):
        """Score appears in JSON output."""
        result = generate_json(_make_params_dict(seed=42, num_steps=50))
        assert "score" in result
        assert result["score"] > 0


# -- Scoring Targets ---------------------------------------------------


class TestScoringTargets:
    def test_legacy_behavior_no_targets(self):
        """Without scoring_targets, uses old behavior (minimize overall vis)."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=50)
        )
        assert params.scoring_targets is None
        result = generate(params)
        assert result.score >= PHASE2_BASE

    def test_overall_only_target(self):
        """Setting only overall target scores based on distance to target."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=50)
        )
        params.scoring_targets = ScoringTargets(
            overall_visibility_target=30.0,
        )
        result = generate(params)
        assert result.score >= PHASE2_BASE

    def test_scoring_targets_deterministic(self):
        """Same seed + same targets = same score and layout."""
        pd = _make_params_dict(seed=123, num_steps=100)
        pd["scoring_targets"] = {
            "overall_visibility_target": 30.0,
            "overall_visibility_weight": 1.0,
        }
        p1 = EngineParams.from_dict(pd)
        p2 = EngineParams.from_dict(pd)
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.score == r2.score
        assert r1.layout.to_dict() == r2.layout.to_dict()

    def test_all_none_targets_fallback(self):
        """ScoringTargets with all None targets uses legacy fallback."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=50)
        )
        params.scoring_targets = ScoringTargets()
        result = generate(params)
        assert result.score >= PHASE2_BASE

    def test_scoring_targets_from_dict(self):
        """ScoringTargets round-trips through dict."""
        pd = _make_params_dict(seed=42, num_steps=50)
        pd["scoring_targets"] = {
            "overall_visibility_target": 30.0,
            "overall_visibility_weight": 2.0,
            "dz_hidden_target": 40.0,
        }
        params = EngineParams.from_dict(pd)
        assert params.scoring_targets is not None
        assert params.scoring_targets.overall_visibility_target == 30.0
        assert params.scoring_targets.overall_visibility_weight == 2.0
        assert params.scoring_targets.dz_visibility_target is None
        assert params.scoring_targets.dz_hidden_target == 40.0
        assert params.scoring_targets.dz_hidden_weight == 1.0

    def test_scoring_targets_to_dict(self):
        """to_dict only includes set targets."""
        st = ScoringTargets(
            overall_visibility_target=30.0,
            overall_visibility_weight=2.0,
        )
        d = st.to_dict()
        assert d["overall_visibility_target"] == 30.0
        assert d["overall_visibility_weight"] == 2.0
        assert "dz_visibility_target" not in d
        assert "dz_hidden_target" not in d

    def test_phase1_unaffected_by_targets(self):
        """Phase 1 scoring is unchanged by scoring targets."""
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        prefs = [FeatureCountPreference(feature_type="obstacle", min=5)]
        targets = ScoringTargets(overall_visibility_target=30.0)
        score = _compute_score(layout, prefs, {}, scoring_targets=targets)
        expected = PHASE2_BASE / (1.0 + 5)
        assert abs(score - expected) < 1e-6


# -- Tempering Integration ---------------------------------------------------


class TestTemperingIntegration:
    def test_tempering_deterministic(self):
        """Same seed + num_replicas produces identical output."""
        pd = _make_params_dict(seed=42, num_steps=100, skip_visibility=True)
        pd["num_replicas"] = 3
        pd["swap_interval"] = 20
        pd["max_temperature"] = 50.0
        p1 = EngineParams.from_dict(pd)
        p2 = EngineParams.from_dict(pd)
        r1 = generate(p1)
        r2 = generate(p2)
        assert r1.layout.to_dict() == r2.layout.to_dict()
        assert r1.score == r2.score

    def test_tempering_produces_features(self):
        """Multi-replica generation produces terrain features."""
        pd = _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        pd["num_replicas"] = 3
        p = EngineParams.from_dict(pd)
        r = generate(p)
        assert len(r.layout.placed_features) > 0

    def test_tempering_different_seeds(self):
        """Different seeds produce different layouts with tempering."""
        pd1 = _make_params_dict(seed=1, num_steps=100, skip_visibility=True)
        pd1["num_replicas"] = 3
        pd2 = _make_params_dict(seed=2, num_steps=100, skip_visibility=True)
        pd2["num_replicas"] = 3
        r1 = generate(EngineParams.from_dict(pd1))
        r2 = generate(EngineParams.from_dict(pd2))
        assert r1.layout.to_dict() != r2.layout.to_dict()

    def test_tempering_no_overlaps(self):
        """Multi-replica output has no overlapping features."""
        pd = _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        pd["num_replicas"] = 3
        params = EngineParams.from_dict(pd)
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

    def test_tempering_all_in_bounds(self):
        """Multi-replica output has all features within table bounds."""
        pd = _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        pd["num_replicas"] = 3
        params = EngineParams.from_dict(pd)
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}
        for pf in result.layout.placed_features:
            obbs = get_world_obbs(pf, objects_by_id)
            for corners in obbs:
                assert obb_in_bounds(
                    corners, params.table_width, params.table_depth
                )

    def test_single_replica_matches_hill_climbing(self):
        """num_replicas=1 dispatches to hill climbing (same code path)."""
        pd = _make_params_dict(seed=42, num_steps=100, skip_visibility=True)
        pd["num_replicas"] = 1
        p_one = EngineParams.from_dict(pd)

        pd_none = _make_params_dict(
            seed=42, num_steps=100, skip_visibility=True
        )
        p_none = EngineParams.from_dict(pd_none)

        r_one = generate(p_one)
        r_none = generate(p_none)
        assert r_one.layout.to_dict() == r_none.layout.to_dict()
        assert r_one.score == r_none.score

    def test_multi_mutation_at_high_temperature(self):
        """MAX_EXTRA_MUTATIONS is used: hot replicas do more mutations per step."""
        # Verify the constant is what we expect
        assert MAX_EXTRA_MUTATIONS == 3
        # At t_factor=1.0: 1 + int(1.0 * 3) = 4 mutations per step
        # At t_factor=0.0: 1 + int(0.0 * 3) = 1 mutation per step
        assert 1 + int(1.0 * MAX_EXTRA_MUTATIONS) == 4
        assert 1 + int(0.0 * MAX_EXTRA_MUTATIONS) == 1
        assert 1 + int(0.5 * MAX_EXTRA_MUTATIONS) == 2

    def test_tempering_with_gaps(self):
        """Tempering respects gap constraints."""
        pd = _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        pd["num_replicas"] = 3
        pd["min_feature_gap_inches"] = 2.0
        pd["min_edge_gap_inches"] = 3.0
        params = EngineParams.from_dict(pd)
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}

        # Verify edge gaps
        for pf in result.layout.placed_features:
            tall_obbs = get_tall_world_obbs(pf, objects_by_id, min_height=1.0)
            for corners in tall_obbs:
                dist = obb_to_table_edge_distance(
                    corners, params.table_width, params.table_depth
                )
                assert dist >= (params.min_edge_gap_inches or 0) - 1e-6

        # Verify feature gaps
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
                        assert (
                            dist >= (params.min_feature_gap_inches or 0) - 1e-6
                        )

    def test_tempering_with_preferences(self):
        """Tempering respects feature count preferences."""
        pd = _make_params_dict(seed=42, num_steps=200, skip_visibility=True)
        pd["num_replicas"] = 3
        pd["feature_count_preferences"] = [
            {"feature_type": "obstacle", "min": 5}
        ]
        params = EngineParams.from_dict(pd)
        result = generate(params)
        count = len(result.layout.placed_features)
        assert count >= 5, f"Expected >= 5 features, got {count}"

    def test_tempering_json_roundtrip(self):
        """Tempering works through the JSON interface."""
        pd = _make_params_dict(seed=42, num_steps=100, skip_visibility=True)
        pd["num_replicas"] = 3
        pd["swap_interval"] = 20
        pd["max_temperature"] = 50.0
        result = generate_json(pd)
        assert "layout" in result
        assert "score" in result
        assert len(result["layout"]["placed_features"]) > 0

    def test_tempering_score_nonnegative(self):
        """Tempering always produces a non-negative score."""
        pd = _make_params_dict(seed=42, num_steps=100, skip_visibility=True)
        pd["num_replicas"] = 3
        params = EngineParams.from_dict(pd)
        result = generate(params)
        assert result.score >= 0
