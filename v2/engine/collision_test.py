"""Tests for polygon overlap detection and polygon terrain shapes."""

import math

from engine.collision import (
    _shape_world_corners,
    _transform_polygon,
    get_world_obbs,
    is_valid_placement,
    obb_distance,
    obb_in_bounds,
    point_in_polygon,
    polygons_overlap,
    segments_intersect_inclusive,
)
from engine.types import (
    FeatureComponent,
    PlacedFeature,
    Shape,
    TerrainFeature,
    TerrainObject,
    Transform,
)


class TestSegmentsIntersectInclusive:
    def test_crossing(self):
        """Two crossing segments."""
        assert segments_intersect_inclusive(0, 0, 10, 10, 0, 10, 10, 0)

    def test_no_crossing(self):
        """Parallel segments."""
        assert not segments_intersect_inclusive(0, 0, 10, 0, 0, 5, 10, 5)

    def test_shared_endpoint(self):
        """Segments sharing an endpoint."""
        assert segments_intersect_inclusive(0, 0, 5, 5, 5, 5, 10, 0)

    def test_t_intersection(self):
        """One segment endpoint touches the other's interior."""
        assert segments_intersect_inclusive(0, 5, 10, 5, 5, 0, 5, 5)

    def test_collinear_non_overlapping(self):
        """Collinear segments that don't overlap."""
        assert not segments_intersect_inclusive(0, 0, 3, 0, 5, 0, 8, 0)


class TestPointInPolygon:
    def test_inside(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon(5, 5, sq) is True

    def test_outside(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon(15, 5, sq) is False


class TestPolygonsOverlap:
    def test_non_overlapping(self):
        """Two separate squares."""
        a = [(0, 0), (5, 0), (5, 5), (0, 5)]
        b = [(10, 10), (15, 10), (15, 15), (10, 15)]
        assert not polygons_overlap(a, b)

    def test_edge_crossing(self):
        """Two squares that overlap via crossing edges."""
        a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        b = [(5, 5), (15, 5), (15, 15), (5, 15)]
        assert polygons_overlap(a, b)

    def test_full_containment_a_inside_b(self):
        """Small polygon fully inside large polygon."""
        a = [(3, 3), (7, 3), (7, 7), (3, 7)]
        b = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert polygons_overlap(a, b)

    def test_full_containment_b_inside_a(self):
        """Large polygon fully contains small polygon."""
        a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        b = [(3, 3), (7, 3), (7, 7), (3, 7)]
        assert polygons_overlap(a, b)

    def test_shared_edge(self):
        """Two squares sharing an edge."""
        a = [(0, 0), (5, 0), (5, 5), (0, 5)]
        b = [(5, 0), (10, 0), (10, 5), (5, 5)]
        assert polygons_overlap(a, b)

    def test_shared_vertex(self):
        """Two squares sharing only a vertex."""
        a = [(0, 0), (5, 0), (5, 5), (0, 5)]
        b = [(5, 5), (10, 5), (10, 10), (5, 10)]
        assert polygons_overlap(a, b)

    def test_vis_poly_vs_small_dz(self):
        """Many-vertex vis polygon vs 4-vertex DZ polygon."""
        # Simulate a large vis polygon (circle approximation)
        import math

        n = 32
        vis_poly = [
            (
                20 * math.cos(2 * math.pi * i / n),
                20 * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
        # Small DZ rectangle inside the vis polygon
        dz = [(5, 5), (8, 5), (8, 8), (5, 8)]
        assert polygons_overlap(vis_poly, dz)

    def test_vis_poly_no_overlap_with_distant_dz(self):
        """Vis polygon doesn't overlap distant DZ."""
        import math

        n = 32
        vis_poly = [
            (
                5 * math.cos(2 * math.pi * i / n),
                5 * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
        dz = [(20, 20), (25, 20), (25, 25), (20, 25)]
        assert not polygons_overlap(vis_poly, dz)

    def test_degenerate_polygon(self):
        """Polygons with < 3 vertices return False."""
        assert not polygons_overlap([(0, 0), (1, 0)], [(0, 0), (1, 0), (1, 1)])
        assert not polygons_overlap([(0, 0), (1, 0), (1, 1)], [(0, 0)])


# ---------------------------------------------------------------------------
# Shape polygon support
# ---------------------------------------------------------------------------


def _make_triangle_shape(height=5.0):
    """Create a triangular polygon shape for testing."""
    return Shape(
        width=4.0,
        depth=3.0,
        height=height,
        vertices=[(-2.0, -1.5), (2.0, -1.5), (0.0, 1.5)],
    )


def _make_hex_shape(radius=2.5, height=5.0):
    """Create a regular hexagonal polygon shape."""
    verts = [
        (
            round(radius * math.cos(2 * math.pi * i / 6), 4),
            round(radius * math.sin(2 * math.pi * i / 6), 4),
        )
        for i in range(6)
    ]
    xs = [v[0] for v in verts]
    zs = [v[1] for v in verts]
    return Shape(
        width=max(xs) - min(xs),
        depth=max(zs) - min(zs),
        height=height,
        vertices=verts,
    )


class TestShapePolygonRoundTrip:
    def test_polygon_from_dict(self):
        """Shape.from_dict with shape_type=polygon reads vertices and computes width/depth."""
        d = {
            "shape_type": "polygon",
            "vertices": [[-2, -1], [2, -1], [0, 1]],
            "height_inches": 3.0,
        }
        s = Shape.from_dict(d)
        assert s.vertices == [(-2, -1), (2, -1), (0, 1)]
        assert s.width == 4.0
        assert s.depth == 2.0
        assert s.height == 3.0

    def test_polygon_to_dict(self):
        """Shape.to_dict with vertices emits shape_type=polygon."""
        s = Shape(
            width=4.0,
            depth=2.0,
            height=3.0,
            vertices=[(-2, -1), (2, -1), (0, 1)],
        )
        d = s.to_dict()
        assert d["shape_type"] == "polygon"
        assert d["vertices"] == [[-2, -1], [2, -1], [0, 1]]
        assert d["height_inches"] == 3.0
        assert d["width_inches"] == 4.0
        assert d["depth_inches"] == 2.0

    def test_rect_to_dict_unchanged(self):
        """Shape.to_dict without vertices still emits rectangular_prism."""
        s = Shape(width=5.0, depth=2.5, height=5.0)
        d = s.to_dict()
        assert d["shape_type"] == "rectangular_prism"
        assert d["width_inches"] == 5.0

    def test_round_trip(self):
        """from_dict(to_dict(shape)) preserves polygon vertices."""
        original = Shape(
            width=4.0,
            depth=2.0,
            height=3.0,
            vertices=[(-2, -1), (2, -1), (0, 1)],
        )
        restored = Shape.from_dict(original.to_dict())
        assert restored.vertices == original.vertices
        assert restored.width == original.width
        assert restored.depth == original.depth
        assert restored.height == original.height


class TestTransformPolygon:
    def test_identity(self):
        """No rotation or translation returns original vertices."""
        verts = [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)]
        result = _transform_polygon(verts, 0.0, 0.0, 0.0)
        for (rx, rz), (vx, vz) in zip(result, verts):
            assert abs(rx - vx) < 1e-10
            assert abs(rz - vz) < 1e-10

    def test_translation(self):
        """Pure translation shifts all vertices."""
        verts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        result = _transform_polygon(verts, 5.0, 3.0, 0.0)
        assert abs(result[0][0] - 5.0) < 1e-10
        assert abs(result[0][1] - 3.0) < 1e-10
        assert abs(result[1][0] - 6.0) < 1e-10
        assert abs(result[1][1] - 3.0) < 1e-10

    def test_rotation_90(self):
        """90 degree rotation transforms correctly."""
        verts = [(1.0, 0.0)]
        result = _transform_polygon(verts, 0.0, 0.0, math.radians(90))
        assert abs(result[0][0] - 0.0) < 1e-10
        assert abs(result[0][1] - 1.0) < 1e-10


class TestShapeWorldCorners:
    def test_rect_shape(self):
        """Rectangular shape returns 4 OBB corners."""
        shape = Shape(width=4.0, depth=2.0, height=5.0)
        world = Transform(x=0.0, z=0.0, rotation_deg=0.0)
        corners = _shape_world_corners(shape, world)
        assert len(corners) == 4

    def test_polygon_shape(self):
        """Polygon shape returns N transformed vertices."""
        shape = _make_triangle_shape()
        world = Transform(x=0.0, z=0.0, rotation_deg=0.0)
        corners = _shape_world_corners(shape, world)
        assert len(corners) == 3

    def test_polygon_translated(self):
        """Polygon shape vertices are translated."""
        shape = _make_triangle_shape()
        world = Transform(x=10.0, z=5.0, rotation_deg=0.0)
        corners = _shape_world_corners(shape, world)
        # First vertex was (-2, -1.5), should now be (8, 3.5)
        assert abs(corners[0][0] - 8.0) < 1e-10
        assert abs(corners[0][1] - 3.5) < 1e-10


class TestGetWorldObbsPolygon:
    def test_polygon_object(self):
        """get_world_obbs returns polygon corners for polygon shapes."""
        shape = _make_triangle_shape()
        obj = TerrainObject(id="tri", shapes=[shape])
        feature = TerrainFeature(
            id="tri_feat",
            feature_type="obstacle",
            components=[FeatureComponent(object_id="tri")],
        )
        pf = PlacedFeature(
            feature=feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        corners_list = get_world_obbs(pf, {"tri": obj})
        assert len(corners_list) == 1
        assert len(corners_list[0]) == 3  # triangle

    def test_mixed_shapes(self):
        """Object with both rect and polygon shapes returns both."""
        rect_shape = Shape(width=4.0, depth=2.0, height=5.0)
        poly_shape = _make_triangle_shape()
        obj = TerrainObject(id="mixed", shapes=[rect_shape, poly_shape])
        feature = TerrainFeature(
            id="mixed_feat",
            feature_type="obstacle",
            components=[FeatureComponent(object_id="mixed")],
        )
        pf = PlacedFeature(
            feature=feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        corners_list = get_world_obbs(pf, {"mixed": obj})
        assert len(corners_list) == 2
        assert len(corners_list[0]) == 4  # rectangle
        assert len(corners_list[1]) == 3  # triangle


class TestObbDistancePolygon:
    def test_polygon_vs_polygon_separated(self):
        """Two separated polygon shapes have positive distance."""
        tri_a = [(-2, -1), (2, -1), (0, 1)]
        tri_b = [(5, -1), (9, -1), (7, 1)]
        dist = obb_distance(tri_a, tri_b)
        assert dist > 0

    def test_polygon_vs_rect_touching(self):
        """Polygon touching rectangle has distance 0."""
        tri = [(-2, 0), (0, 2), (2, 0)]
        rect = [(-1, -2), (1, -2), (1, 0), (-1, 0)]
        dist = obb_distance(tri, rect)
        assert dist == 0.0

    def test_polygon_vs_polygon_crossing(self):
        """Two polygons with crossing edges have distance 0."""
        tri_a = [(-2, -1), (2, -1), (0, 2)]
        tri_b = [(-2, 0), (2, 0), (0, -2)]
        dist = obb_distance(tri_a, tri_b)
        assert dist == 0.0


class TestObbInBoundsPolygon:
    def test_polygon_in_bounds(self):
        """Polygon vertices within table bounds."""
        tri = [(-2, -1), (2, -1), (0, 1)]
        assert obb_in_bounds(tri, 60.0, 44.0)

    def test_polygon_out_of_bounds(self):
        """Polygon with vertex outside table bounds."""
        tri = [(-31, 0), (0, 0), (0, 1)]
        assert not obb_in_bounds(tri, 60.0, 44.0)


class TestIsValidPlacementPolygon:
    def _make_polygon_catalog(self):
        """Create a catalog with a polygon shape for is_valid_placement testing."""
        shape = _make_hex_shape(radius=2.5, height=5.0)
        obj = TerrainObject(id="hex", shapes=[shape])
        feature = TerrainFeature(
            id="hex_feat",
            feature_type="obstacle",
            components=[FeatureComponent(object_id="hex")],
        )
        return obj, feature

    def test_valid_polygon_placement(self):
        """Polygon feature placed in valid position passes validation."""
        obj, feature = self._make_polygon_catalog()
        pf = PlacedFeature(
            feature=feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        assert is_valid_placement([pf], 0, 60.0, 44.0, {"hex": obj})

    def test_polygon_overlaps_rect(self):
        """Polygon overlapping a rectangle fails validation."""
        hex_obj, hex_feature = self._make_polygon_catalog()
        rect_shape = Shape(width=4.0, depth=2.0, height=5.0)
        rect_obj = TerrainObject(id="rect", shapes=[rect_shape])
        rect_feature = TerrainFeature(
            id="rect_feat",
            feature_type="obstacle",
            components=[FeatureComponent(object_id="rect")],
        )
        pf_hex = PlacedFeature(
            feature=hex_feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        pf_rect = PlacedFeature(
            feature=rect_feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        objects = {"hex": hex_obj, "rect": rect_obj}
        assert not is_valid_placement(
            [pf_hex, pf_rect], 0, 60.0, 44.0, objects
        )

    def test_polygon_out_of_bounds_fails(self):
        """Polygon feature placed outside table bounds fails validation."""
        obj, feature = self._make_polygon_catalog()
        pf = PlacedFeature(
            feature=feature,
            transform=Transform(x=29.0, z=0.0, rotation_deg=0.0),
        )
        assert not is_valid_placement([pf], 0, 60.0, 44.0, {"hex": obj})
