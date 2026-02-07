"""Unit tests for distance calculation functions in collision.py"""

from __future__ import annotations

import math

from engine.collision import (
    obb_corners,
    obb_distance,
    obb_to_table_edge_distance,
    point_to_segment_distance_squared,
    segments_intersect,
)


class TestPointToSegmentDistance:
    """Tests for point_to_segment_distance_squared."""

    def test_point_on_segment(self):
        """Point on segment returns 0."""
        dist_sq = point_to_segment_distance_squared(0, 0, -1, 0, 1, 0)
        assert dist_sq == 0.0

    def test_point_perpendicular_to_segment(self):
        """Point perpendicular to segment returns perpendicular distance squared."""
        # Point at (0, 1), segment from (-1, 0) to (1, 0)
        dist_sq = point_to_segment_distance_squared(0, 1, -1, 0, 1, 0)
        assert abs(dist_sq - 1.0) < 1e-9

    def test_point_nearest_to_start_endpoint(self):
        """Point nearest to start endpoint returns distance to start."""
        # Point at (-2, 0), segment from (-1, 0) to (1, 0)
        dist_sq = point_to_segment_distance_squared(-2, 0, -1, 0, 1, 0)
        assert abs(dist_sq - 1.0) < 1e-9

    def test_point_nearest_to_end_endpoint(self):
        """Point nearest to end endpoint returns distance to end."""
        # Point at (2, 0), segment from (-1, 0) to (1, 0)
        dist_sq = point_to_segment_distance_squared(2, 0, -1, 0, 1, 0)
        assert abs(dist_sq - 1.0) < 1e-9

    def test_degenerate_segment_zero_length(self):
        """Degenerate segment (point) returns distance to that point."""
        # Segment from (0, 0) to (0, 0), point at (3, 4)
        dist_sq = point_to_segment_distance_squared(3, 4, 0, 0, 0, 0)
        assert abs(dist_sq - 25.0) < 1e-9


class TestSegmentIntersection:
    """Tests for segments_intersect."""

    def test_crossing_segments(self):
        """Two crossing segments return True."""
        # Segment 1: (-1, 0) to (1, 0)
        # Segment 2: (0, -1) to (0, 1)
        assert segments_intersect(-1, 0, 1, 0, 0, -1, 0, 1)

    def test_non_intersecting_parallel(self):
        """Two parallel non-intersecting segments return False."""
        # Segment 1: (-1, 0) to (1, 0)
        # Segment 2: (-1, 1) to (1, 1)
        assert not segments_intersect(-1, 0, 1, 0, -1, 1, 1, 1)

    def test_non_intersecting_perpendicular(self):
        """Two perpendicular but non-intersecting segments return False."""
        # Segment 1: (0, 0) to (1, 0)
        # Segment 2: (2, 0) to (2, 1)
        assert not segments_intersect(0, 0, 1, 0, 2, 0, 2, 1)

    def test_touching_endpoint(self):
        """Segments touching at endpoint return False (endpoints excluded)."""
        # Segment 1: (-1, 0) to (0, 0)
        # Segment 2: (0, 0) to (1, 0)
        assert not segments_intersect(-1, 0, 0, 0, 0, 0, 1, 0)

    def test_t_intersection(self):
        """T-intersection (one segment endpoint on other) returns False."""
        # Segment 1: (-1, 0) to (1, 0)
        # Segment 2: (0, 0) to (0, 1) - touches at (0, 0) which is endpoint
        assert not segments_intersect(-1, 0, 1, 0, 0, 0, 0, 1)


class TestOBBDistance:
    """Tests for obb_distance."""

    def test_overlapping_returns_zero(self):
        """Overlapping rectangles return distance 0."""
        # Rectangle 1: corners at (-1, -1), (1, -1), (1, 1), (-1, 1)
        # Rectangle 2: corners at (0, 0), (2, 0), (2, 2), (0, 2)
        corners_a = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        corners_b = [(0, 0), (2, 0), (2, 2), (0, 2)]
        assert obb_distance(corners_a, corners_b) == 0.0

    def test_touching_returns_zero(self):
        """Touching rectangles (shared edge) return distance 0."""
        # Rectangle 1: (-2, -1) to (0, 1)
        # Rectangle 2: (0, -1) to (2, 1)
        corners_a = [(-2, -1), (0, -1), (0, 1), (-2, 1)]
        corners_b = [(0, -1), (2, -1), (2, 1), (0, 1)]
        assert obb_distance(corners_a, corners_b) == 0.0

    def test_separated_horizontal(self):
        """Separated rectangles return positive distance."""
        # Rectangle 1: (-2, -1) to (0, 1)
        # Rectangle 2: (1, -1) to (3, 1)
        corners_a = [(-2, -1), (0, -1), (0, 1), (-2, 1)]
        corners_b = [(1, -1), (3, -1), (3, 1), (1, 1)]
        dist = obb_distance(corners_a, corners_b)
        assert abs(dist - 1.0) < 1e-9

    def test_separated_diagonal(self):
        """Diagonally separated rectangles return correct distance."""
        # Rectangle 1: (-1, -1) to (0, 0)
        # Rectangle 2: (1, 1) to (2, 2)
        corners_a = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        corners_b = [(1, 1), (2, 1), (2, 2), (1, 2)]
        dist = obb_distance(corners_a, corners_b)
        # Diagonal distance: sqrt(2)
        assert abs(dist - math.sqrt(2)) < 1e-9

    def test_rotated_rectangles(self):
        """Rotated rectangles calculate distance correctly."""
        # Create two 45-degree rotated squares
        corners_a = obb_corners(0, 0, 1, 1, math.radians(45))
        corners_b = obb_corners(3, 0, 1, 1, math.radians(45))
        dist = obb_distance(corners_a, corners_b)
        # Distance should be positive
        assert dist > 0


class TestOBBToTableEdgeDistance:
    """Tests for obb_to_table_edge_distance."""

    def test_centered_rectangle(self):
        """Rectangle centered on table returns distance to nearest edge."""
        # Rectangle centered at (0, 0) with corners at (±1, ±1)
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        # Table 10x10, so nearest edge is at ±5
        dist = obb_to_table_edge_distance(corners, 10, 10)
        assert abs(dist - 4.0) < 1e-9

    def test_touching_edge(self):
        """Rectangle touching edge returns distance 0."""
        # Rectangle with corner touching right edge at x = 5
        corners = [(4, 0), (5, 0), (5, 1), (4, 1)]
        dist = obb_to_table_edge_distance(corners, 10, 10)
        assert dist == 0.0

    def test_outside_bounds(self):
        """Rectangle outside bounds returns distance 0."""
        # Rectangle with x > 5 (outside table)
        corners = [(5.1, 0), (6, 0), (6, 1), (5.1, 1)]
        dist = obb_to_table_edge_distance(corners, 10, 10)
        assert dist == 0.0

    def test_near_corner(self):
        """Rectangle near table corner returns minimum distance."""
        # Rectangle at (4, 4) with size 1x1
        corners = [(3.5, 3.5), (4.5, 3.5), (4.5, 4.5), (3.5, 4.5)]
        # Table 10x10, so edges at ±5
        # Distance to nearest edge: 5 - 4.5 = 0.5
        dist = obb_to_table_edge_distance(corners, 10, 10)
        assert abs(dist - 0.5) < 1e-9

    def test_rotated_rectangle(self):
        """Rotated rectangle calculates edge distance correctly."""
        corners = obb_corners(0, 0, 1, 1, math.radians(45))
        dist = obb_to_table_edge_distance(corners, 10, 10)
        # Center at (0, 0), rotated, nearest edge at 5
        # Rotated square extends to approx ±sqrt(2) ≈ ±1.414
        assert dist > 0
        assert abs(dist - (5 - math.sqrt(2))) < 1e-9


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
