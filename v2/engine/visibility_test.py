"""Tests for visibility score computation."""

from engine.types import (
    FeatureComponent,
    PlacedFeature,
    Shape,
    TerrainFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)
from engine.visibility import (
    _extract_blocking_segments,
    _point_in_polygon,
    _polygon_area,
    _ray_segment_intersection,
    compute_layout_visibility,
)


def _make_object(obj_id, width, depth, height):
    return TerrainObject(
        id=obj_id,
        shapes=[Shape(width=width, depth=depth, height=height)],
    )


def _make_feature(feat_id, obj_id, feature_type="obstacle"):
    return TerrainFeature(
        id=feat_id,
        feature_type=feature_type,
        components=[FeatureComponent(object_id=obj_id)],
    )


def _place(feature, x, z, rot=0.0):
    return PlacedFeature(
        feature=feature, transform=Transform(x=x, z=z, rotation_deg=rot)
    )


def _make_layout(table_w, table_d, placed_features, symmetric=False):
    return TerrainLayout(
        table_width=table_w,
        table_depth=table_d,
        placed_features=placed_features,
        rotationally_symmetric=symmetric,
    )


# -- Helper function tests --


class TestPolygonArea:
    def test_unit_square(self):
        # CCW square: (0,0), (1,0), (1,1), (0,1)
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert abs(_polygon_area(verts) - 1.0) < 1e-9

    def test_triangle(self):
        verts = [(0, 0), (4, 0), (0, 3)]
        assert abs(_polygon_area(verts) - 6.0) < 1e-9

    def test_degenerate(self):
        verts = [(0, 0), (1, 0)]
        assert abs(_polygon_area(verts)) < 1e-9


class TestPointInPolygon:
    def test_inside_square(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(5, 5, sq) is True

    def test_outside_square(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(15, 5, sq) is False

    def test_on_edge(self):
        """Points on the boundary are implementation-defined; just don't crash."""
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        _point_in_polygon(0, 5, sq)  # Should not raise


class TestRaySegmentIntersection:
    def test_hit(self):
        # Ray from origin going right, segment is vertical at x=5
        t = _ray_segment_intersection(0, 0, 1, 0, 5, -5, 5, 5)
        assert t is not None
        assert abs(t - 5.0) < 1e-9

    def test_miss(self):
        # Ray from origin going right, segment is behind
        t = _ray_segment_intersection(0, 0, 1, 0, -5, -5, -5, 5)
        assert t is None

    def test_parallel(self):
        # Ray along x-axis, segment also along x-axis
        t = _ray_segment_intersection(0, 0, 1, 0, 1, 0, 5, 0)
        assert t is None


# -- Blocking segment extraction tests --


class TestExtractBlockingSegments:
    def test_tall_obstacle_blocks(self):
        """A 5x5x5 obstacle at center produces 4 blocking edges."""
        obj = _make_object("box", 5, 5, 5)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        # Observer far away from center
        segs = _extract_blocking_segments(
            layout, objects_by_id, 25, 0, min_blocking_height=4.0
        )
        assert len(segs) == 4

    def test_short_obstacle_no_blocking(self):
        """A 5x5x2.5 obstacle (below 4" threshold) does not block."""
        obj = _make_object("box", 5, 5, 2.5)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        segs = _extract_blocking_segments(
            layout, objects_by_id, 25, 0, min_blocking_height=4.0
        )
        assert len(segs) == 0

    def test_obscuring_blocks_regardless_of_height(self):
        """An obscuring feature blocks even with 0 height shapes."""
        obj = _make_object("ruins", 12, 6, 0)
        feat = _make_feature("f1", "ruins", "obscuring")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruins": obj}

        # Observer outside the footprint
        segs = _extract_blocking_segments(
            layout, objects_by_id, 25, 0, min_blocking_height=4.0
        )
        # Obscuring: only back-facing edges block (should be 1-2 edges)
        assert len(segs) > 0
        assert len(segs) < 4  # Not all edges, only back-facing

    def test_obscuring_observer_inside_no_blocking(self):
        """Observer inside obscuring footprint gets no blocking from it."""
        obj = _make_object("ruins", 12, 6, 0)
        feat = _make_feature("f1", "ruins", "obscuring")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruins": obj}

        # Observer at center of the 12x6 feature at origin
        segs = _extract_blocking_segments(
            layout, objects_by_id, 0, 0, min_blocking_height=4.0
        )
        assert len(segs) == 0


# -- Full visibility computation tests --


class TestComputeLayoutVisibility:
    def test_empty_battlefield(self):
        """Empty table → 100% visibility."""
        layout = _make_layout(60, 44, [])
        objects_by_id = {}
        result = compute_layout_visibility(layout, objects_by_id)
        assert result["overall"]["value"] == 100.0

    def test_single_tall_block_center(self):
        """A 5x5x5 block at center reduces visibility below 100%."""
        obj = _make_object("box", 5, 5, 5)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        val = result["overall"]["value"]
        assert val < 100.0
        assert val > 50.0  # Single small block shouldn't block half the table

    def test_short_block_full_visibility(self):
        """A 5x5x2.5 block (below 4" threshold) → 100% visibility."""
        obj = _make_object("box", 5, 5, 2.5)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        assert result["overall"]["value"] == 100.0

    def test_obscuring_feature_reduces_visibility(self):
        """An obscuring feature (even with height 0) reduces visibility."""
        obj = _make_object("ruins", 12, 6, 0)
        feat = _make_feature("f1", "ruins", "obscuring")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruins": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        val = result["overall"]["value"]
        assert val < 100.0

    def test_multiple_pieces_additive_occlusion(self):
        """More terrain pieces → lower visibility."""
        obj = _make_object("box", 5, 5, 5)
        feat1 = _make_feature("f1", "box", "obstacle")
        feat2 = _make_feature("f2", "box", "obstacle")
        feat3 = _make_feature("f3", "box", "obstacle")

        single = _make_layout(60, 44, [_place(feat1, -10, 0)])
        multi = _make_layout(
            60,
            44,
            [
                _place(feat1, -10, 0),
                _place(feat2, 10, 0),
                _place(feat3, 0, 10),
            ],
        )
        objects_by_id = {"box": obj}

        r_single = compute_layout_visibility(single, objects_by_id)
        r_multi = compute_layout_visibility(multi, objects_by_id)
        assert r_multi["overall"]["value"] < r_single["overall"]["value"]

    def test_symmetric_layout_includes_mirrors(self):
        """Symmetric layout should include mirrored features in occlusion."""
        obj = _make_object("box", 5, 5, 5)
        feat = _make_feature("f1", "box", "obstacle")
        # Place at (10, 10) — mirror at (-10, -10)
        pf = _place(feat, 10, 10)
        sym_layout = _make_layout(60, 44, [pf], symmetric=True)
        nonsym_layout = _make_layout(60, 44, [pf], symmetric=False)
        objects_by_id = {"box": obj}

        r_sym = compute_layout_visibility(sym_layout, objects_by_id)
        r_nonsym = compute_layout_visibility(nonsym_layout, objects_by_id)
        # Symmetric should have more occlusion (2 blocks vs 1)
        assert r_sym["overall"]["value"] < r_nonsym["overall"]["value"]

    def test_result_format(self):
        """Result dict has expected keys and types."""
        layout = _make_layout(60, 44, [])
        objects_by_id = {}
        result = compute_layout_visibility(layout, objects_by_id)
        assert "overall" in result
        overall = result["overall"]
        assert "value" in overall
        assert "grid_spacing_inches" in overall
        assert "grid_offset_inches" in overall
        assert "min_blocking_height_inches" in overall
        assert "sample_count" in overall
        assert isinstance(overall["value"], float)
        assert isinstance(overall["sample_count"], int)

    def test_feature_on_table_edge(self):
        """Feature near table edge still computes without error."""
        obj = _make_object("box", 5, 5, 5)
        feat = _make_feature("f1", "box", "obstacle")
        # Place near edge
        pf = _place(feat, 27, 19)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        assert 0 <= result["overall"]["value"] <= 100.0
