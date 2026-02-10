"""Tests for polygon overlap detection."""

from engine.collision import (
    point_in_polygon,
    polygons_overlap,
    segments_intersect_inclusive,
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
