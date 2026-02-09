from engine.generate import generate
from engine.mutation import (
    MIN_MOVE_RANGE,
    TILE_SIZE,
    StepUndo,
    _compute_tile_weights,
    _quantize_angle,
    _quantize_position,
    _temperature_move,
    _undo_step,
)
from engine.prng import PCG32
from engine.types import (
    EngineParams,
    FeatureComponent,
    PlacedFeature,
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

    def test_undo_rotate(self):
        """Undo rotate restores original transform."""
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        feat = TerrainFeature(id="f1", feature_type="obstacle", components=[])
        old_pf = PlacedFeature(feat, Transform(1.0, 2.0, 45.0))
        new_pf = PlacedFeature(feat, Transform(1.0, 2.0, 90.0))
        layout.placed_features.append(new_pf)
        undo = StepUndo(action="rotate", index=0, old_feature=old_pf)
        _undo_step(layout, undo)
        assert layout.placed_features[0].transform.rotation_deg == 45.0
        assert layout.placed_features[0].transform.x == 1.0
        assert layout.placed_features[0].transform.z == 2.0

    def test_undo_noop(self):
        layout = TerrainLayout(table_width=60.0, table_depth=44.0)
        undo = StepUndo(action="noop")
        _undo_step(layout, undo)  # should not raise
        assert len(layout.placed_features) == 0


# -- Tile-Biased Placement -----------------------------------------------


class TestTileBiasedPlacement:
    def test_tile_weights_empty_table(self):
        """Empty table has all tile weights equal to 1.0."""
        objects_by_id = {
            "crate_5x2.5": _crate_catalog_dict()["objects"][0]["item"]
        }
        from engine.types import TerrainObject

        obj = TerrainObject.from_dict(objects_by_id["crate_5x2.5"])
        objs = {"crate_5x2.5": obj}

        weights, nx, nz, tw, td = _compute_tile_weights(
            [], objs, 60.0, 44.0, False
        )
        assert all(w == 1.0 for w in weights)
        assert nx == round(60.0 / TILE_SIZE)
        assert nz == round(44.0 / TILE_SIZE)

    def test_tile_weights_occupied_lower(self):
        """Tiles containing features have lower weight than empty tiles."""
        params = EngineParams.from_dict(
            _make_params_dict(seed=42, num_steps=100, skip_visibility=True)
        )
        result = generate(params)
        objects_by_id = {co.item.id: co.item for co in params.catalog.objects}

        weights, nx, nz, tw, td = _compute_tile_weights(
            result.layout.placed_features,
            objects_by_id,
            params.table_width,
            params.table_depth,
            False,
        )
        min_w = min(weights)
        max_w = max(weights)
        # Features present means some tiles occupied → lower weight
        assert min_w < max_w
        # Max weight should be 1.0 (empty tiles)
        assert max_w == 1.0
        # Min weight should be < 1.0 (occupied tiles)
        assert min_w < 1.0

    def test_tile_weights_symmetric_mirrors(self):
        """In symmetric mode, mirrored features also penalize tiles."""
        feat = TerrainFeature(
            id="f1",
            feature_type="obstacle",
            components=[
                FeatureComponent(
                    object_id="crate_5x2.5",
                    transform=None,
                )
            ],
        )
        placed = PlacedFeature(feat, Transform(10.0, 5.0, 0.0))

        from engine.types import Shape, TerrainObject

        obj = TerrainObject(
            id="crate_5x2.5",
            shapes=[Shape(width=5.0, depth=2.5, height=2.0)],
        )
        objs = {"crate_5x2.5": obj}

        # Non-symmetric: only affects tiles near (10, 5)
        w_nonsym, nx, nz, tw, td = _compute_tile_weights(
            [placed], objs, 60.0, 44.0, False
        )
        # Symmetric: also affects tiles near (-10, -5)
        w_sym, _, _, _, _ = _compute_tile_weights(
            [placed], objs, 60.0, 44.0, True
        )

        # Count how many tiles are occupied (weight < 1.0)
        nonsym_occupied = sum(1 for w in w_nonsym if w < 1.0)
        sym_occupied = sum(1 for w in w_sym if w < 1.0)
        # Symmetric should have more occupied tiles due to mirror
        assert sym_occupied > nonsym_occupied
