"""Unit tests for algorithmically complex helpers in frontend/app.py."""

import math

import pytest

from .app import (
    BattlefieldRenderer,
    _build_object_index,
    _compose,
    _get_tf,
    _mirror_pf_dict,
)

# ---------------------------------------------------------------------------
# _get_tf
# ---------------------------------------------------------------------------


class TestGetTf:
    def test_none_returns_zeros(self):
        assert _get_tf(None) == (0.0, 0.0, 0.0)

    def test_empty_dict_returns_zeros(self):
        assert _get_tf({}) == (0.0, 0.0, 0.0)

    def test_partial_dict_fills_defaults(self):
        assert _get_tf({"x_inches": 3.0}) == (3.0, 0.0, 0.0)
        assert _get_tf({"z_inches": -2.0}) == (0.0, -2.0, 0.0)
        assert _get_tf({"rotation_deg": 90.0}) == (0.0, 0.0, 90.0)

    def test_full_dict(self):
        d = {"x_inches": 1.5, "z_inches": -3.0, "rotation_deg": 45.0}
        assert _get_tf(d) == (1.5, -3.0, 45.0)


# ---------------------------------------------------------------------------
# _compose
# ---------------------------------------------------------------------------


class TestCompose:
    def test_identity_outer(self):
        """Composing with identity outer leaves inner unchanged."""
        inner = (3.0, 4.0, 30.0)
        result = _compose((0.0, 0.0, 0.0), inner)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(4.0)
        assert result[2] == pytest.approx(30.0)

    def test_identity_inner(self):
        """Composing with identity inner leaves outer unchanged."""
        outer = (5.0, -2.0, 60.0)
        result = _compose(outer, (0.0, 0.0, 0.0))
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(-2.0)
        assert result[2] == pytest.approx(60.0)

    def test_pure_translation(self):
        """Two translations (no rotation) add."""
        result = _compose((1.0, 2.0, 0.0), (3.0, 4.0, 0.0))
        assert result[0] == pytest.approx(4.0)
        assert result[1] == pytest.approx(6.0)
        assert result[2] == pytest.approx(0.0)

    def test_rotation_then_translate(self):
        """90° outer rotation rotates the inner offset."""
        # Outer is at origin rotated 90°. Inner is (1, 0, 0).
        # After 90° rotation: inner (1,0) -> (0, 1)
        result = _compose((0.0, 0.0, 90.0), (1.0, 0.0, 0.0))
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(90.0)

    def test_180_rotation(self):
        """180° outer rotation negates inner offsets."""
        result = _compose((0.0, 0.0, 180.0), (3.0, 4.0, 0.0))
        assert result[0] == pytest.approx(-3.0, abs=1e-10)
        assert result[1] == pytest.approx(-4.0, abs=1e-10)
        assert result[2] == pytest.approx(180.0)

    def test_rotation_accumulates(self):
        """Rotations from outer and inner add."""
        result = _compose((0.0, 0.0, 45.0), (0.0, 0.0, 30.0))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(75.0)

    def test_translate_and_rotate(self):
        """Outer has both translation and rotation."""
        # Outer: translate (10, 0), rotate 90°
        # Inner: (2, 0, 0)
        # Inner rotated by 90°: (0, 2). Then translated: (10, 2).
        result = _compose((10.0, 0.0, 90.0), (2.0, 0.0, 0.0))
        assert result[0] == pytest.approx(10.0, abs=1e-10)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# _mirror_pf_dict
# ---------------------------------------------------------------------------


class TestMirrorPfDict:
    def test_basic_mirror(self):
        pf = {
            "feature": {"id": "crate"},
            "transform": {
                "x_inches": 5.0,
                "y_inches": 0.0,
                "z_inches": 3.0,
                "rotation_deg": 45.0,
            },
        }
        mirrored = _mirror_pf_dict(pf)
        tf = mirrored["transform"]
        assert tf["x_inches"] == -5.0
        assert tf["y_inches"] == 0.0
        assert tf["z_inches"] == -3.0
        assert tf["rotation_deg"] == 225.0

    def test_double_mirror_is_identity(self):
        """Mirroring twice returns to the original position (mod 360)."""
        pf = {
            "feature": {"id": "wall"},
            "transform": {
                "x_inches": 7.0,
                "y_inches": 1.0,
                "z_inches": -2.0,
                "rotation_deg": 120.0,
            },
        }
        double = _mirror_pf_dict(_mirror_pf_dict(pf))
        orig_tf = pf["transform"]
        dbl_tf = double["transform"]
        assert dbl_tf["x_inches"] == pytest.approx(orig_tf["x_inches"])
        assert dbl_tf["y_inches"] == pytest.approx(orig_tf["y_inches"])
        assert dbl_tf["z_inches"] == pytest.approx(orig_tf["z_inches"])
        assert dbl_tf["rotation_deg"] % 360 == pytest.approx(
            orig_tf["rotation_deg"] % 360
        )

    def test_origin_mirror(self):
        """Feature at origin: position stays (0,0), rotation changes by 180."""
        pf = {
            "feature": {"id": "x"},
            "transform": {
                "x_inches": 0.0,
                "y_inches": 0.0,
                "z_inches": 0.0,
                "rotation_deg": 0.0,
            },
        }
        mirrored = _mirror_pf_dict(pf)
        tf = mirrored["transform"]
        assert tf["x_inches"] == 0.0
        assert tf["z_inches"] == 0.0
        assert tf["rotation_deg"] == 180.0

    def test_preserves_non_transform_keys(self):
        """Non-transform keys are preserved in the output."""
        pf = {
            "feature": {"id": "ruin", "components": []},
            "locked": True,
            "transform": {
                "x_inches": 1.0,
                "y_inches": 0.0,
                "z_inches": 2.0,
                "rotation_deg": 0.0,
            },
        }
        mirrored = _mirror_pf_dict(pf)
        assert mirrored["feature"] == pf["feature"]
        assert mirrored["locked"] is True


# ---------------------------------------------------------------------------
# BattlefieldRenderer._point_in_polygon
# ---------------------------------------------------------------------------

# Alias for brevity
_pip = BattlefieldRenderer._point_in_polygon


class TestPointInPolygon:
    """Tests for the ray-casting point-in-polygon algorithm."""

    UNIT_SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_inside_square(self):
        assert _pip(0.5, 0.5, self.UNIT_SQUARE) is True

    def test_outside_square(self):
        assert _pip(2.0, 0.5, self.UNIT_SQUARE) is False
        assert _pip(-0.5, 0.5, self.UNIT_SQUARE) is False
        assert _pip(0.5, 2.0, self.UNIT_SQUARE) is False
        assert _pip(0.5, -1.0, self.UNIT_SQUARE) is False

    def test_inside_triangle(self):
        tri = [(0, 0), (4, 0), (2, 3)]
        assert _pip(2.0, 1.0, tri) is True

    def test_outside_triangle(self):
        tri = [(0, 0), (4, 0), (2, 3)]
        assert _pip(0.0, 3.0, tri) is False

    def test_concave_polygon(self):
        """L-shaped concave polygon: inside the notch should be outside."""
        # L-shape: bottom-left 2x2 plus top-left 1x1
        poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        assert _pip(0.5, 0.5, poly) is True  # inside bottom
        assert _pip(0.5, 1.5, poly) is True  # inside top-left arm
        assert _pip(1.5, 1.5, poly) is False  # in the notch

    def test_large_polygon(self):
        """Regular hexagon centered at origin."""
        hex_poly = [
            (math.cos(math.radians(a)), math.sin(math.radians(a)))
            for a in range(0, 360, 60)
        ]
        assert _pip(0.0, 0.0, hex_poly) is True
        assert _pip(2.0, 0.0, hex_poly) is False

    def test_negative_coordinates(self):
        poly = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        assert _pip(0, 0, poly) is True
        assert _pip(-4.9, -4.9, poly) is True
        assert _pip(6.0, 0.0, poly) is False


# ---------------------------------------------------------------------------
# BattlefieldRenderer._line_polygon_intersections_z
# ---------------------------------------------------------------------------

_intersect_z = BattlefieldRenderer._line_polygon_intersections_z
_intersect_x = BattlefieldRenderer._line_polygon_intersections_x


class TestLinePolygonIntersectionsZ:
    """Vertical line x=x_val intersected with polygon edges."""

    UNIT_SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_through_center(self):
        """x=0.5 through unit square should give z=0 and z=1."""
        zs = sorted(_intersect_z(0.5, self.UNIT_SQUARE))
        assert len(zs) == 2
        assert zs[0] == pytest.approx(0.0)
        assert zs[1] == pytest.approx(1.0)

    def test_outside_polygon(self):
        """x=2.0 misses the unit square entirely."""
        zs = _intersect_z(2.0, self.UNIT_SQUARE)
        assert len(zs) == 0

    def test_negative_x(self):
        """x=-1 misses the unit square."""
        zs = _intersect_z(-1.0, self.UNIT_SQUARE)
        assert len(zs) == 0

    def test_triangle(self):
        """Line through a triangle apex may produce duplicate at shared vertex."""
        tri = [(0, 0), (4, 0), (2, 4)]
        zs = sorted(_intersect_z(2.0, tri))
        # Both edges meeting at apex (2,4) report z=4; caller deduplicates via set
        unique = sorted(set(round(z, 10) for z in zs))
        assert len(unique) == 2
        assert unique[0] == pytest.approx(0.0)
        assert unique[1] == pytest.approx(4.0)

    def test_at_edge(self):
        """x=0 along a vertical edge — vertical edges are skipped."""
        zs = _intersect_z(0.0, self.UNIT_SQUARE)
        # The left edge (x=0) is vertical, so it's skipped.
        # But the top and bottom edges at z=0 and z=1 still cross x=0 at their endpoints.
        # Edge (0,0)-(1,0): t=(0-0)/(1-0)=0 => z=0
        # Edge (0,1)-(0,0): vertical, skipped
        # Edge (1,0)-(1,1): vertical, skipped
        # Edge (1,1)-(0,1): t=(0-1)/(0-1)=1 => z=1
        assert len(zs) == 2


class TestLinePolygonIntersectionsX:
    """Horizontal line z=z_val intersected with polygon edges."""

    UNIT_SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_through_center(self):
        """z=0.5 through unit square should give x=0 and x=1."""
        xs = sorted(_intersect_x(0.5, self.UNIT_SQUARE))
        assert len(xs) == 2
        assert xs[0] == pytest.approx(0.0)
        assert xs[1] == pytest.approx(1.0)

    def test_outside_polygon(self):
        """z=2.0 misses the unit square entirely."""
        xs = _intersect_x(2.0, self.UNIT_SQUARE)
        assert len(xs) == 0

    def test_triangle(self):
        """Line through a triangle."""
        tri = [(0, 0), (4, 0), (2, 4)]
        xs = sorted(_intersect_x(2.0, tri))
        # At z=2: left edge (0,0)-(2,4): t=2/4=0.5 => x=1.0
        # right edge (4,0)-(2,4): t=2/4=0.5 => x=3.0
        assert len(xs) == 2
        assert xs[0] == pytest.approx(1.0)
        assert xs[1] == pytest.approx(3.0)

    def test_at_horizontal_edge(self):
        """z=0 along the bottom edge — horizontal edges are skipped."""
        xs = _intersect_x(0.0, self.UNIT_SQUARE)
        # Bottom edge (0,0)-(1,0): horizontal, skipped
        # Left edge (0,1)-(0,0): t=(0-1)/(0-1)=1 => x=0
        # Right edge (1,0)-(1,1): t=(0-0)/(1-0)=0 => x=1
        # Top edge (1,1)-(0,1): horizontal z=1, not relevant
        assert len(xs) == 2


# ---------------------------------------------------------------------------
# _build_object_index
# ---------------------------------------------------------------------------


class TestBuildObjectIndex:
    def test_indexes_top_level_objects(self):
        catalog = {
            "objects": [
                {"item": {"id": "crate_5x2.5", "shapes": []}},
                {"item": {"id": "wall_6x1", "shapes": []}},
            ]
        }
        index = _build_object_index(catalog)
        assert "crate_5x2.5" in index
        assert "wall_6x1" in index
        assert len(index) == 2

    def test_indexes_objects_inside_features(self):
        catalog = {
            "objects": [],
            "features": [
                {
                    "item": {
                        "id": "ruin_L",
                        "components": [
                            {
                                "object_id": "wall_a",
                                "object": {
                                    "id": "wall_a",
                                    "shapes": [],
                                },
                            },
                            {
                                "object_id": "wall_b",
                                "object": {
                                    "id": "wall_b",
                                    "shapes": [],
                                },
                            },
                        ],
                    }
                }
            ],
        }
        index = _build_object_index(catalog)
        assert "wall_a" in index
        assert "wall_b" in index

    def test_both_sources_combined(self):
        catalog = {
            "objects": [
                {"item": {"id": "standalone", "shapes": []}},
            ],
            "features": [
                {
                    "item": {
                        "id": "feat",
                        "components": [
                            {
                                "object_id": "nested",
                                "object": {"id": "nested", "shapes": []},
                            }
                        ],
                    }
                }
            ],
        }
        index = _build_object_index(catalog)
        assert "standalone" in index
        assert "nested" in index

    def test_empty_catalog(self):
        assert _build_object_index({}) == {}
        assert _build_object_index({"objects": [], "features": []}) == {}

    def test_feature_without_inline_object(self):
        """Components that reference objects by ID only (no inline 'object') are skipped."""
        catalog = {
            "objects": [{"item": {"id": "wall", "shapes": []}}],
            "features": [
                {
                    "item": {
                        "id": "feat",
                        "components": [{"object_id": "wall"}],
                    }
                }
            ],
        }
        index = _build_object_index(catalog)
        # Only the top-level object is indexed
        assert list(index.keys()) == ["wall"]


# ---------------------------------------------------------------------------
# BattlefieldRenderer._to_px
# ---------------------------------------------------------------------------


class TestToPx:
    def _make_renderer(self, tw=60.0, td=44.0, ppi=10.0):
        return BattlefieldRenderer(tw, td, ppi, {})

    def test_center_of_table(self):
        """Table center (0, 0) maps to pixel center."""
        r = self._make_renderer(tw=60.0, td=44.0, ppi=10.0)
        px, py = r._to_px(0.0, 0.0)
        assert px == pytest.approx(300.0)  # 60/2 * 10
        assert py == pytest.approx(220.0)  # 44/2 * 10

    def test_top_left_corner(self):
        """Top-left corner (-tw/2, -td/2) maps to pixel (0, 0)."""
        r = self._make_renderer(tw=60.0, td=44.0, ppi=10.0)
        px, py = r._to_px(-30.0, -22.0)
        assert px == pytest.approx(0.0)
        assert py == pytest.approx(0.0)

    def test_bottom_right_corner(self):
        """Bottom-right corner (tw/2, td/2) maps to (tw*ppi, td*ppi)."""
        r = self._make_renderer(tw=60.0, td=44.0, ppi=10.0)
        px, py = r._to_px(30.0, 22.0)
        assert px == pytest.approx(600.0)
        assert py == pytest.approx(440.0)

    def test_different_ppi(self):
        """Higher PPI scales pixel coordinates proportionally."""
        r = self._make_renderer(tw=10.0, td=10.0, ppi=20.0)
        px, py = r._to_px(0.0, 0.0)
        assert px == pytest.approx(100.0)  # 10/2 * 20
        assert py == pytest.approx(100.0)

    def test_asymmetric_table(self):
        r = self._make_renderer(tw=20.0, td=10.0, ppi=5.0)
        # Center
        px, py = r._to_px(0.0, 0.0)
        assert px == pytest.approx(50.0)  # 20/2 * 5
        assert py == pytest.approx(25.0)  # 10/2 * 5
        # One inch right and down from center
        px, py = r._to_px(1.0, 1.0)
        assert px == pytest.approx(55.0)
        assert py == pytest.approx(30.0)
