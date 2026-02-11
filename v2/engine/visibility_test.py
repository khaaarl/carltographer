"""Tests for visibility score computation."""

from engine.collision import obb_corners
from engine.types import (
    DeploymentZone,
    FeatureComponent,
    Mission,
    ObjectiveMarker,
    PlacedFeature,
    Point2D,
    Shape,
    TerrainFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)
from engine.visibility import (
    DZ_EXPANSION_INCHES,
    _extract_blocking_segments,
    _fraction_of_dz_visible,
    _has_valid_hiding_square,
    _point_in_polygon,
    _point_near_any_polygon,
    _polygon_area,
    _ray_segment_intersection,
    _sample_points_in_circle,
    _sample_points_in_polygon,
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

    def test_wall_inside_obscuring_blocks_from_inside(self):
        """A tall wall inside an obscuring ruin blocks LoS even for observers
        inside the ruin footprint.

        The ruin base (12x6, low) grants the observer see-through on the
        outer footprint, but the internal wall (0.5x4, tall) is a separate
        shape that the observer is NOT inside, so its back-facing edges
        still block.
        """
        base = TerrainObject(
            id="ruin_base",
            shapes=[Shape(width=12, depth=6, height=0.1)],
        )
        wall = TerrainObject(
            id="ruin_wall",
            shapes=[Shape(width=0.5, depth=4, height=5)],
        )
        feat = TerrainFeature(
            id="ruin_with_wall",
            feature_type="obscuring",
            components=[
                FeatureComponent(object_id="ruin_base"),
                # Wall offset 2" to the right of center
                FeatureComponent(
                    object_id="ruin_wall",
                    transform=Transform(x=2, z=0),
                ),
            ],
        )
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruin_base": base, "ruin_wall": wall}

        # Observer inside the ruin base but to the LEFT of the wall
        segs = _extract_blocking_segments(
            layout, objects_by_id, -2, 0, min_blocking_height=4.0
        )
        # Base: observer is inside → no blocking from base (correct)
        # Wall: observer is outside the 0.5"-wide wall → back-faces block
        assert len(segs) > 0, "Wall inside ruin should block from inside ruin"

        # Observer OUTSIDE the ruin entirely (further left) should see
        # both the ruin back-faces and the wall back-faces
        segs_outside = _extract_blocking_segments(
            layout, objects_by_id, -25, 0, min_blocking_height=4.0
        )
        assert len(segs_outside) > len(segs), (
            "Observer outside ruin should see more blocking edges "
            "(ruin back-faces + wall back-faces)"
        )


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


# -- DZ helper function tests --


def _make_simple_mission(dz_list, symmetric=False, objectives=None):
    """Create a Mission with the given deployment zones."""
    return Mission(
        name="test",
        objectives=objectives or [],
        deployment_zones=dz_list,
        rotationally_symmetric=symmetric,
    )


def _make_rect_dz(dz_id, x1, z1, x2, z2):
    """Create a rectangular DeploymentZone from corner coords."""
    return DeploymentZone(
        id=dz_id,
        polygons=[
            [
                Point2D(x=x1, z=z1),
                Point2D(x=x2, z=z1),
                Point2D(x=x2, z=z2),
                Point2D(x=x1, z=z2),
            ]
        ],
    )


class TestSamplePointsInPolygon:
    def test_rect(self):
        """Known rectangle: 10x10 at origin, 1" grid, 0.5 offset."""
        rect = [(0, 0), (10, 0), (10, 10), (0, 10)]
        pts = _sample_points_in_polygon(rect, 1.0, 0.5)
        # Grid points: 0.5, 1.5, ..., 9.5 in each axis = 10x10 = 100
        assert len(pts) == 100

    def test_triangle(self):
        """Non-rectangular shape has fewer points than bounding rect."""
        tri = [(0, 0), (10, 0), (5, 10)]
        pts = _sample_points_in_polygon(tri, 1.0, 0.5)
        # Should have some points but fewer than 100 (the bounding rect)
        assert len(pts) > 0
        assert len(pts) < 100


class TestFractionOfDzVisible:
    def test_full_coverage(self):
        """Vis polygon covers DZ entirely -> ~1.0."""
        dz_samples = [(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)]
        # Vis polygon is a large square covering everything
        vis_poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        frac = _fraction_of_dz_visible(vis_poly, dz_samples)
        assert abs(frac - 1.0) < 1e-9

    def test_no_coverage(self):
        """Vis polygon away from DZ -> 0.0."""
        dz_samples = [(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)]
        vis_poly = [(20, 20), (30, 20), (30, 30), (20, 30)]
        frac = _fraction_of_dz_visible(vis_poly, dz_samples)
        assert abs(frac) < 1e-9


# -- DZ integration tests --


class TestDzHideability:
    def test_empty_table_hideability_zero(self):
        """No terrain: all DZ observers can see opponent DZ -> 0% hidden."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz])

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[],
            mission=mission,
        )
        result = compute_layout_visibility(layout, {})

        assert "dz_hideability" in result
        assert "green" in result["dz_hideability"]
        assert "red" in result["dz_hideability"]
        # With no terrain, nothing hidden -> 0%
        assert result["dz_hideability"]["green"]["value"] == 0.0
        assert result["dz_hideability"]["red"]["value"] == 0.0

    def test_blocker_increases_hideability(self):
        """Tall terrain between DZs increases hideability."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz])

        # Place a tall wall between the DZs
        obj = _make_object("wall", 2, 40, 5)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, -18, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        # Wall near green DZ should hide some green observers from red
        assert result["dz_hideability"]["green"]["value"] > 0.0

    def test_no_mission_no_dz_keys(self):
        """When no mission, DZ keys should be absent."""
        layout = _make_layout(60, 44, [])
        result = compute_layout_visibility(layout, {})
        assert "dz_hideability" not in result

    def test_asymmetric_hideability_differs(self):
        """Hideability is asymmetric with off-center terrain.

        Wall near green DZ blocks some green observers' view of red DZ,
        but red observers can still see green DZ mostly unobstructed.
        """
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz])

        # Wall close to green DZ, NOT mirrored
        obj = _make_object("wall", 2, 40, 5)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, -18, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            rotationally_symmetric=False,
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        hide = result["dz_hideability"]

        green_hide = hide["green"]["value"]
        red_hide = hide["red"]["value"]

        # Wall near green DZ blocks green observers' view of red DZ
        # -> green hideability should be substantial
        assert green_hide > 0.0, f"green_hide ({green_hide}) should be > 0"
        # Hideability should be asymmetric
        assert green_hide != red_hide, (
            f"green_hide ({green_hide}) should differ from red_hide ({red_hide})"
        )

    def test_symmetric_dz_values_close(self):
        """With symmetric layout + mission, both DZ hideability values are close."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz], symmetric=True)

        # Symmetric layout with terrain on one side (mirrored)
        obj = _make_object("box", 5, 5, 5)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 10, 5)  # Will have a mirror at (-10, -5)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            rotationally_symmetric=True,
            mission=mission,
        )
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        green_val = result["dz_hideability"]["green"]["value"]
        red_val = result["dz_hideability"]["red"]["value"]
        # Values should be close but not necessarily identical due to
        # floating-point differences in observer iteration order
        assert abs(green_val - red_val) < 1.0


# -- Sample points in circle tests --


class TestSamplePointsInCircle:
    def test_known_radius(self):
        """Circle at origin with radius 3.75, 1" grid, 0.5 offset."""
        pts = _sample_points_in_circle(0.0, 0.0, 3.75, 1.0, 0.5)
        # Grid points from -3.5 to 3.5 in each axis that fit inside r=3.75
        assert len(pts) > 0
        # All points should be within the circle
        for x, z in pts:
            assert x * x + z * z <= 3.75**2 + 1e-9

    def test_zero_radius(self):
        """Zero radius yields no points."""
        pts = _sample_points_in_circle(0.0, 0.0, 0.0, 1.0, 0.5)
        assert len(pts) == 0

    def test_offset_center(self):
        """Circle at non-origin position still works."""
        pts = _sample_points_in_circle(10.0, 5.0, 3.75, 1.0, 0.5)
        assert len(pts) > 0
        for x, z in pts:
            dx = x - 10.0
            dz = z - 5.0
            assert dx * dx + dz * dz <= 3.75**2 + 1e-9


# -- Objective hidability tests --


def _make_standard_objectives():
    """Create 5 objectives at standard positions for a 60x44 table."""
    return [
        ObjectiveMarker(id="obj1", position=Point2D(x=0.0, z=0.0)),
        ObjectiveMarker(id="obj2", position=Point2D(x=-12.0, z=-8.0)),
        ObjectiveMarker(id="obj3", position=Point2D(x=12.0, z=-8.0)),
        ObjectiveMarker(id="obj4", position=Point2D(x=-12.0, z=8.0)),
        ObjectiveMarker(id="obj5", position=Point2D(x=12.0, z=8.0)),
    ]


class TestObjectiveHidability:
    def test_empty_table_no_hiding(self):
        """No terrain: all objectives fully visible -> 0% safe."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        objectives = _make_standard_objectives()
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[],
            mission=mission,
        )
        result = compute_layout_visibility(layout, {})

        assert "objective_hidability" in result
        oh = result["objective_hidability"]
        assert oh["green"]["value"] == 0.0
        assert oh["red"]["value"] == 0.0
        assert oh["green"]["safe_count"] == 0
        assert oh["red"]["safe_count"] == 0
        assert oh["green"]["total_objectives"] == 5
        assert oh["red"]["total_objectives"] == 5

    def test_blocker_creates_hiding(self):
        """Wall near objectives creates hiding spots -> > 0% safe."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        # Single objective at center
        objectives = [
            ObjectiveMarker(id="obj1", position=Point2D(x=0.0, z=0.0))
        ]
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        # Tall wall right next to the objective, blocking view from red DZ
        obj = _make_object("wall", 2, 8, 5)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, 2, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        oh = result["objective_hidability"]
        # Green player can hide from red DZ observers (wall blocks red's view)
        assert oh["green"]["safe_count"] >= 1

    def test_asymmetric_hiding(self):
        """Wall near one DZ creates asymmetric hiding values."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        objectives = _make_standard_objectives()
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        # Wall close to red DZ side, blocking red's view of objectives
        obj = _make_object("wall", 2, 40, 5)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, 15, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        oh = result["objective_hidability"]
        # Green player benefits more from wall near red DZ
        # (wall blocks red's view, creating hiding spots for green)
        assert oh["green"]["safe_count"] >= oh["red"]["safe_count"]

    def test_no_objectives_no_key(self):
        """Mission with DZs but no objectives -> key absent."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz])

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[],
            mission=mission,
        )
        result = compute_layout_visibility(layout, {})
        assert "objective_hidability" not in result

    def test_no_mission_no_key(self):
        """No mission at all -> key absent."""
        layout = _make_layout(60, 44, [])
        result = compute_layout_visibility(layout, {})
        assert "objective_hidability" not in result


class TestHasValidHidingSquare:
    """Tests for the model-fit hiding square check."""

    def _grid_points(self, x_range, z_range):
        """Generate grid points at integer spacing."""
        pts = []
        for x in x_range:
            for z in z_range:
                pts.append((float(x), float(z)))
        return pts

    def test_basic_valid_square(self):
        """4 adjacent hidden points form a valid hiding square."""
        # Objective at origin, radius 5
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        # Hide a 2x2 block of points near origin
        hidden = set()
        for i, (x, z) in enumerate(pts):
            if 0 <= x <= 1 and 0 <= z <= 1:
                hidden.add(i)
        assert _has_valid_hiding_square(0.0, 0.0, 5.0, pts, hidden, [], 30, 22)

    def test_no_hidden_points(self):
        """No hidden points -> no valid square."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        hidden: set[int] = set()
        assert not _has_valid_hiding_square(
            0.0, 0.0, 5.0, pts, hidden, [], 30, 22
        )

    def test_single_hidden_point_not_enough(self):
        """A single hidden point cannot form a 1x1 square."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        # Hide only one point
        hidden = set()
        for i, (x, z) in enumerate(pts):
            if x == 0 and z == 0:
                hidden.add(i)
                break
        assert not _has_valid_hiding_square(
            0.0, 0.0, 5.0, pts, hidden, [], 30, 22
        )

    def test_terrain_blocks_square(self):
        """Tall terrain overlapping the square invalidates it."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        # Hide a 2x2 block
        hidden = set()
        for i, (x, z) in enumerate(pts):
            if 0 <= x <= 1 and 0 <= z <= 1:
                hidden.add(i)
        # Place tall terrain OBB covering exactly the hiding square
        tall_obb = obb_corners(0.5, 0.5, 0.6, 0.6, 0.0)
        assert not _has_valid_hiding_square(
            0.0, 0.0, 5.0, pts, hidden, [tall_obb], 30, 22
        )

    def test_out_of_range(self):
        """Hidden points far from objective don't count."""
        pts = self._grid_points(range(-10, 11), range(-10, 11))
        # Hide a 2x2 block far from origin (at x=8..9, z=8..9)
        hidden = set()
        for i, (x, z) in enumerate(pts):
            if 8 <= x <= 9 and 8 <= z <= 9:
                hidden.add(i)
        # Objective radius 3 — hidden points are way outside
        assert not _has_valid_hiding_square(
            0.0, 0.0, 3.0, pts, hidden, [], 30, 22
        )

    def test_out_of_bounds(self):
        """Hidden points outside table bounds don't form valid square."""
        # Table is 10x10 (half_w=5, half_d=5)
        pts = self._grid_points(range(-6, 7), range(-6, 7))
        # Hide points at the table edge where square would go OOB
        hidden = set()
        for i, (x, z) in enumerate(pts):
            if 5 <= x <= 6 and 0 <= z <= 1:
                hidden.add(i)
        # Objective at (5, 0) with radius 2 — hidden points at edge
        assert not _has_valid_hiding_square(
            5.0, 0.0, 2.0, pts, hidden, [], 5, 5
        )

    def test_terrain_adjacent_but_not_overlapping(self):
        """Tall terrain adjacent to (but not overlapping) square is fine."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        hidden = set()
        for i, (x, z) in enumerate(pts):
            if 0 <= x <= 1 and 0 <= z <= 1:
                hidden.add(i)
        # Terrain right next to the square but not overlapping
        tall_obb = obb_corners(2.5, 0.5, 0.5, 0.5, 0.0)
        assert _has_valid_hiding_square(
            0.0, 0.0, 5.0, pts, hidden, [tall_obb], 30, 22
        )


# -- Observer filtering tests --


class TestObserverFilteringInsideTallTerrain:
    def test_tall_terrain_reduces_sample_count(self):
        """A tall terrain piece (height >= 1") should reduce sample_count."""
        empty_layout = _make_layout(60, 44, [])
        empty_result = compute_layout_visibility(empty_layout, {})
        empty_count = empty_result["overall"]["sample_count"]

        # Place a large tall piece at center
        obj = _make_object("box", 10, 10, 2.0)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}
        result = compute_layout_visibility(layout, objects_by_id)
        filtered_count = result["overall"]["sample_count"]

        assert filtered_count < empty_count

    def test_short_terrain_does_not_reduce_sample_count(self):
        """A short terrain piece (height < 1") should NOT reduce sample_count."""
        empty_layout = _make_layout(60, 44, [])
        empty_result = compute_layout_visibility(empty_layout, {})
        empty_count = empty_result["overall"]["sample_count"]

        # Place a short piece at center
        obj = _make_object("box", 10, 10, 0.5)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}
        result = compute_layout_visibility(layout, objects_by_id)
        filtered_count = result["overall"]["sample_count"]

        assert filtered_count == empty_count


# -- Expanded DZ tests --


class TestExpandedDZ:
    def test_point_near_polygon_inside(self):
        """Point inside polygon returns True."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_near_any_polygon(5.0, 5.0, [poly], 6.0) is True

    def test_point_near_polygon_within_range(self):
        """Point outside but within max_dist of polygon edge returns True."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Point at (13, 5) is 3" from the right edge at x=10
        assert _point_near_any_polygon(13.0, 5.0, [poly], 6.0) is True

    def test_point_near_polygon_out_of_range(self):
        """Point far from polygon returns False."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Point at (20, 5) is 10" from the right edge — beyond 6"
        assert _point_near_any_polygon(20.0, 5.0, [poly], 6.0) is False

    def test_point_near_polygon_corner(self):
        """Point near corner (within diagonal distance) returns True."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Point at (14, 14) — distance from corner (10,10) = sqrt(32) ≈ 5.66"
        assert _point_near_any_polygon(14.0, 14.0, [poly], 6.0) is True

    def test_expansion_constant(self):
        """DZ_EXPANSION_INCHES is 6.0."""
        assert DZ_EXPANSION_INCHES == 6.0

    def test_wall_between_dzs_increases_hideability(self):
        """A wall between DZs increases hideability above zero."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz])

        # Wall in the center
        obj = _make_object("wall", 2, 40, 5)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, 0, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        hide = result["dz_hideability"]

        # Both DZs should have some hideability from the center wall
        assert hide["green"]["total_count"] > 0
        assert hide["red"]["total_count"] > 0


# -- Infantry visibility (dual-height LOS) tests --


class TestEffectiveOpacityHeight:
    def test_default_uses_physical_height(self):
        """When opacity_height_inches is None, use physical height."""
        s = Shape(width=5, depth=5, height=6.0)
        assert s.effective_opacity_height() == 6.0

    def test_override_uses_opacity(self):
        """When opacity_height_inches is set, use it."""
        s = Shape(width=5, depth=5, height=6.0, opacity_height_inches=3.0)
        assert s.effective_opacity_height() == 3.0

    def test_zero_opacity(self):
        """Zero opacity_height_inches means transparent."""
        s = Shape(width=5, depth=5, height=6.0, opacity_height_inches=0.0)
        assert s.effective_opacity_height() == 0.0


class TestHasIntermediateShapes:
    def test_shape_in_range(self):
        """A 3" tall shape is in [2.2, 4.0) -> True."""
        from engine.visibility import _has_intermediate_shapes

        obj = _make_object("box", 5, 5, 3.0)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        assert _has_intermediate_shapes(layout, {"box": obj}, 2.2, 4.0)

    def test_all_above_standard(self):
        """A 5" tall shape is above 4.0 -> False."""
        from engine.visibility import _has_intermediate_shapes

        obj = _make_object("box", 5, 5, 5.0)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        assert not _has_intermediate_shapes(layout, {"box": obj}, 2.2, 4.0)

    def test_all_below_infantry(self):
        """A 2.0" tall shape is below 2.2 -> False."""
        from engine.visibility import _has_intermediate_shapes

        obj = _make_object("box", 5, 5, 2.0)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        assert not _has_intermediate_shapes(layout, {"box": obj}, 2.2, 4.0)

    def test_opacity_override_in_range(self):
        """Shape with height=6, opacity_height=3 -> in [2.2, 4.0) -> True."""
        from engine.visibility import _has_intermediate_shapes

        obj = TerrainObject(
            id="box",
            shapes=[
                Shape(width=5, depth=5, height=6.0, opacity_height_inches=3.0)
            ],
        )
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        assert _has_intermediate_shapes(layout, {"box": obj}, 2.2, 4.0)

    def test_obscuring_ignored(self):
        """Obscuring features are excluded from intermediate check."""
        from engine.visibility import _has_intermediate_shapes

        obj = _make_object("ruins", 12, 6, 3.0)
        feat = _make_feature("f1", "ruins", "obscuring")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        assert not _has_intermediate_shapes(layout, {"ruins": obj}, 2.2, 4.0)


class TestInfantryVisibility:
    def test_infantry_disabled_no_sub_dicts(self):
        """infantry_blocking_height=None -> standard-only format."""
        obj = _make_object("box", 5, 5, 3.0)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=None
        )
        overall = result["overall"]
        assert "standard" not in overall
        assert "infantry" not in overall
        assert "value" in overall

    def test_no_intermediate_shapes_no_sub_dicts(self):
        """All shapes above standard height -> infantry pass skipped."""
        obj = _make_object("box", 5, 5, 5.0)  # 5" > 4.0"
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=2.2
        )
        overall = result["overall"]
        # No intermediate shapes -> no infantry pass -> standard format
        assert "standard" not in overall
        assert "infantry" not in overall

    def test_intermediate_shapes_trigger_dual_pass(self):
        """Shape at 3" with infantry_blocking_height=2.2 -> dual pass."""
        obj = _make_object("box", 5, 5, 3.0)  # 3" in [2.2, 4.0)
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=2.2
        )
        overall = result["overall"]
        assert "standard" in overall
        assert "infantry" in overall
        assert "value" in overall
        # Standard pass: 3" shape is below 4.0 -> doesn't block -> ~100%
        assert overall["standard"]["value"] == 100.0
        # Infantry pass: 3" shape blocks at 2.2 -> < 100%
        assert overall["infantry"]["value"] < 100.0
        # Average should be between the two
        avg = (overall["standard"]["value"] + overall["infantry"]["value"]) / 2
        assert abs(overall["value"] - round(avg, 2)) < 0.01

    def test_opacity_override_affects_blocking(self):
        """Shape with height=6, opacity_height=1 should NOT block at 2.2."""
        obj = TerrainObject(
            id="box",
            shapes=[
                Shape(width=5, depth=5, height=6.0, opacity_height_inches=1.0)
            ],
        )
        feat = _make_feature("f1", "box", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"box": obj}

        # opacity_height=1.0 < infantry_height=2.2 < standard=4.0
        # -> NOT an intermediate shape -> no infantry pass
        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=2.2
        )
        assert "standard" not in result["overall"]

        # Also check that the standard pass treats it correctly:
        # opacity_height=1.0 < standard=4.0, so it doesn't block at all
        assert result["overall"]["value"] == 100.0

    def test_opacity_affects_segments(self):
        """Shape with height=6, opacity_height=3 blocks infantry but not standard."""
        obj = TerrainObject(
            id="wall",
            shapes=[
                Shape(width=2, depth=10, height=6.0, opacity_height_inches=3.0)
            ],
        )
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"wall": obj}

        # Standard pass (4.0): opacity 3.0 < 4.0 -> doesn't block
        result_standard = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=4.0,
            infantry_blocking_height=None,
        )
        assert result_standard["overall"]["value"] == 100.0

        # Infantry pass (2.2): opacity 3.0 >= 2.2 -> blocks
        result_infantry = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=2.2,
            infantry_blocking_height=None,
        )
        assert result_infantry["overall"]["value"] < 100.0

    def test_infantry_with_dz_hideability(self):
        """Infantry pass with DZs produces sub-dicts in dz_hideability."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        mission = _make_simple_mission([green_dz, red_dz])

        # 3" tall wall -> intermediate shape -> triggers infantry pass
        obj = _make_object("wall", 2, 20, 3.0)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, 0, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=2.2
        )

        # Overall should have sub-dicts
        assert "standard" in result["overall"]
        assert "infantry" in result["overall"]

        # DZ hideability should also have sub-dicts
        assert "dz_hideability" in result
        for dz_id in ["green", "red"]:
            dz_entry = result["dz_hideability"][dz_id]
            assert "standard" in dz_entry
            assert "infantry" in dz_entry
            assert "value" in dz_entry


# -- Polygon terrain shape visibility tests --


def _make_polygon_object(obj_id, vertices, height):
    """Create a TerrainObject with a polygon shape."""
    return TerrainObject(
        id=obj_id,
        shapes=[
            Shape(
                width=max(v[0] for v in vertices)
                - min(v[0] for v in vertices),
                depth=max(v[1] for v in vertices)
                - min(v[1] for v in vertices),
                height=height,
                vertices=list(vertices),
            )
        ],
    )


class TestPolygonVisibility:
    def test_tall_polygon_blocks_los(self):
        """A tall polygon shape (e.g., tank) should block line of sight."""
        import math

        # 24-gon approximating a circle of radius 2.5
        radius = 2.5
        verts = [
            (
                radius * math.cos(2 * math.pi * i / 24),
                radius * math.sin(2 * math.pi * i / 24),
            )
            for i in range(24)
        ]
        tank_obj = _make_polygon_object("tank", verts, 5.0)
        tank_feat = _make_feature("tank_feat", "tank", "obstacle")
        pf = _place(tank_feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"tank": tank_obj}

        result = compute_layout_visibility(layout, objects_by_id)
        # A tall polygon should reduce visibility below 100%
        assert result["overall"]["value"] < 100.0

    def test_flat_polygon_does_not_block_los(self):
        """A flat polygon shape (height=0, e.g., woods) does NOT block LoS."""
        verts = [(-4, -2.5), (4, -2.5), (4, 2.5), (-4, 2.5)]
        woods_obj = _make_polygon_object("woods", verts, 0.0)
        woods_feat = _make_feature("woods_feat", "woods", "obstacle")
        pf = _place(woods_feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"woods": woods_obj}

        result = compute_layout_visibility(layout, objects_by_id)
        # height=0 -> does not block at standard height (4.0") -> 100%
        assert result["overall"]["value"] == 100.0

    def test_polygon_segments_count(self):
        """Polygon obstacle generates correct number of blocking segments."""
        # Triangle: 3 edges -> 3 segments
        verts = [(-2, -1), (2, -1), (0, 1)]
        obj = _make_polygon_object("tri", verts, 5.0)
        feat = _make_feature("tri_feat", "tri", "obstacle")
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"tri": obj}

        segments = _extract_blocking_segments(layout, objects_by_id, 0, 0)
        # Should have exactly 3 segments from the triangle
        assert len(segments) == 3
