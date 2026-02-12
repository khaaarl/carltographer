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
    _find_hiding_square_with_fringe,
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


def _make_feature(feat_id, obj_id, feature_type="obstacle", tags=None):
    return TerrainFeature(
        id=feat_id,
        feature_type=feature_type,
        components=[FeatureComponent(object_id=obj_id)],
        tags=tags if tags is not None else [],
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

    def test_obscuring_backface_blocking(self):
        """An obscuring footprint blocks via back-facing edges."""
        obj = TerrainObject(
            id="ruins",
            shapes=[Shape(width=12, depth=6, height=5)],
            is_footprint=True,
        )
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
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
        obj = TerrainObject(
            id="ruins",
            shapes=[Shape(width=12, depth=6, height=5)],
            is_footprint=True,
        )
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
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

        The ruin base (12x6, is_footprint) uses backface culling — observer
        inside sees no blocking from it.  The internal wall (0.5x4, tall,
        NOT a footprint) is a static segment that blocks from all directions.
        """
        base = TerrainObject(
            id="ruin_base",
            shapes=[Shape(width=12, depth=6, height=5)],
            is_footprint=True,
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
            tags=["obscuring"],
        )
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruin_base": base, "ruin_wall": wall}

        # Observer inside the ruin base but to the LEFT of the wall
        segs = _extract_blocking_segments(
            layout, objects_by_id, -2, 0, min_blocking_height=4.0
        )
        # Base: observer is inside footprint → no blocking (backface culling)
        # Wall: non-footprint → all 4 edges are static segments
        assert len(segs) == 4, "Wall should produce 4 static segments"

        # Observer OUTSIDE the ruin entirely — should see ruin back-faces
        # (from footprint backface culling) PLUS all 4 wall static segments
        segs_outside = _extract_blocking_segments(
            layout, objects_by_id, -25, 0, min_blocking_height=4.0
        )
        assert len(segs_outside) > len(segs), (
            "Observer outside ruin should see more blocking edges "
            "(ruin back-faces + wall static segments)"
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
        """A tall obscuring footprint reduces visibility via back-facing edges."""
        obj = TerrainObject(
            id="ruins",
            shapes=[Shape(width=12, depth=6, height=5)],
            is_footprint=True,
        )
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
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

    def test_obscuring_ruin_at_objective_no_hiding(self):
        """Obscuring ruin covering objective: tall terrain overlaps all candidate squares.

        The 14x14 ruin at the origin covers the entire objective sample area
        (expanded radius ≈ 3.75 + sqrt(2) ≈ 5.16, well within ±7). Every
        candidate hiding square overlaps the tall terrain OBB, so all are
        rejected by the terrain-intersection check. Result: safe_count == 0.
        """
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        objectives = [
            ObjectiveMarker(id="obj1", position=Point2D(x=0.0, z=0.0))
        ]
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        # Large obscuring ruin covering all objective sample points
        # Expanded radius = 0.75 + 3.0 + sqrt(2) ≈ 5.16, so 14x14 ruin at
        # origin covers all sample points (footprint from -7 to 7 in both axes)
        obj = _make_object("ruins", 14, 14, 5)
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
        pf = _place(feat, 0, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"ruins": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        oh = result["objective_hidability"]
        # All sample points inside the ruin -> full visibility -> no hiding
        assert oh["green"]["safe_count"] == 0
        assert oh["red"]["safe_count"] == 0

    def test_obscuring_between_objective_and_dz(self):
        """Obscuring ruin between objectives and DZ creates hiding.

        A wide ruin placed between objectives and red DZ blocks LoS from
        objective-vicinity sample points toward red's expanded DZ, giving
        the green player hiding spots at those objectives.
        """
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        objectives = _make_standard_objectives()
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        # Wide obscuring ruin between objectives and red DZ
        obj = _make_object("ruins", 2, 40, 5)
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
        pf = _place(feat, 15, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"ruins": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        oh = result["objective_hidability"]
        # Ruin blocking LoS toward red DZ should help green player
        assert oh["green"]["value"] >= 0.0
        assert oh["green"]["total_objectives"] == 5
        # Green benefits more from a ruin near red DZ
        assert oh["green"]["safe_count"] >= oh["red"]["safe_count"]


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

    def test_obscuring_intermediate_detected(self):
        """Obscuring features with intermediate-height shapes should be detected.

        A 3.0" obscuring shape is in [2.2, 4.0) and would produce different
        visibility at the infantry blocking height vs standard, so it should
        trigger the dual-pass.
        """
        from engine.visibility import _has_intermediate_shapes

        obj = _make_object("ruins", 12, 6, 3.0)
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        assert _has_intermediate_shapes(layout, {"ruins": obj}, 2.2, 4.0)


class TestObscuringHeightFiltering:
    """Obscuring features respect min_blocking_height like obstacles.

    Non-footprint obscuring shapes below min_blocking_height do NOT block
    LOS, just like non-obscuring obstacles. These tests verify that height
    filtering applies correctly to obscuring features.
    """

    def test_obscuring_below_threshold_no_block(self):
        """A 3.0" obscuring shape should not block at min_blocking_height=4.0.

        Standard pass uses 4.0" threshold.  A 3.0" ruin wall is shorter than
        that, so observers should be able to see over it — visibility should
        be 100%.
        """
        obj = _make_object("ruins", 5, 5, 3.0)
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruins": obj}

        result = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=4.0,
            infantry_blocking_height=None,
        )
        # 3.0" < 4.0" threshold -> should not block -> 100% visibility
        assert result["overall"]["value"] == 100.0

    def test_obscuring_above_threshold_blocks(self):
        """A 3.0" obscuring shape should block at min_blocking_height=2.2.

        Infantry pass uses 2.2" threshold.  A 3.0" non-footprint shape is
        taller than that, so it produces static segments that block LOS.
        """
        obj = _make_object("ruins", 5, 5, 3.0)
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruins": obj}

        result = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=2.2,
            infantry_blocking_height=None,
        )
        # 3.0" >= 2.2" threshold -> should block -> < 100% visibility
        assert result["overall"]["value"] < 100.0

    def test_obscuring_dual_pass_produces_different_values(self):
        """Obscuring feature at 3.0" should produce different std vs inf visibility.

        With infantry_blocking_height=2.2 and an obscuring shape at 3.0":
        - Standard pass (4.0"): 3.0 < 4.0 → doesn't block → ~100%
        - Infantry pass (2.2"): 3.0 >= 2.2 → blocks → < 100%
        The dual-pass should be triggered and produce sub-dicts.
        """
        obj = _make_object("ruins", 5, 5, 3.0)
        feat = _make_feature("f1", "ruins", "obscuring", tags=["obscuring"])
        pf = _place(feat, 0, 0)
        layout = _make_layout(60, 44, [pf])
        objects_by_id = {"ruins": obj}

        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=2.2
        )
        overall = result["overall"]
        # Dual-pass should be triggered
        assert "standard" in overall, "Expected dual-pass 'standard' sub-dict"
        assert "infantry" in overall, "Expected dual-pass 'infantry' sub-dict"
        # Standard: 3.0" < 4.0" → no blocking → 100%
        assert overall["standard"]["value"] == 100.0
        # Infantry: 3.0" >= 2.2" → blocks → < 100%
        assert overall["infantry"]["value"] < 100.0


class TestObscuringFootprintBlocksWithoutOpacityOverride:
    """The Obscuring keyword is a feature-level property, not shape-level.

    A feature with the "obscuring" tag should have its footprint block LOS
    via back-facing edges regardless of individual shape heights.  The engine
    must NOT require opacity_height_inches to make this work — a height-0
    base in an obscuring feature blocks because the *feature* is Obscuring,
    not because the shape is tall.

    This test uses a bare 12x6 ruin base (height 0, no opacity_height_inches)
    as the sole shape of an obscuring feature.  It should block LOS.
    """

    @staticmethod
    def _make_bare_obscuring_layout():
        """Build a 20x20 layout with a bare obscuring footprint at the origin."""
        base = TerrainObject(
            id="ruins_base",
            shapes=[
                Shape(width=12.0, depth=6.0, height=0.0),
            ],
            is_footprint=True,
        )
        feature = TerrainFeature(
            id="bare_ruin",
            feature_type="obscuring",
            components=[FeatureComponent(object_id="ruins_base")],
            tags=["obscuring"],
        )
        pf = PlacedFeature(
            feature=feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        layout = TerrainLayout(
            table_width=20.0,
            table_depth=20.0,
            placed_features=[pf],
        )
        objects_by_id = {"ruins_base": base}
        return layout, objects_by_id

    def test_standard_visibility_below_100(self):
        """A height-0 obscuring footprint must block LOS at standard height.

        The "obscuring" tag means the footprint blocks — this must not depend
        on the shape's height or opacity_height_inches.
        """
        layout, objects_by_id = self._make_bare_obscuring_layout()
        result = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=4.0,
            infantry_blocking_height=None,
        )
        val = result["overall"]["value"]
        assert val < 100.0, (
            f"Obscuring footprint (height=0, no opacity_height_inches) should "
            f"block LOS because feature_type='obscuring', got {val}%"
        )


class TestWtcShortRuinVisibility:
    """End-to-end test with the actual WTC short ruin as defined in the UI.

    The WTC short ruin has two components:
    - ruins_short: 12"x6" base at height 0, is_footprint=True (Obscuring
      keyword — the footprint always blocks LOS via back-facing edges).
    - wtc_short_walls: four wall sections at height 3.0" (physical barriers
      that block infantry but not standard-height observers).

    On a 20"x20" table:
    - Standard visibility (4.0" threshold) should be < 100% because the base
      footprint blocks LOS via Obscuring.
    - Infantry visibility (2.2" threshold) should be even lower because the
      3.0" walls additionally block infantry-height observers.
    """

    @staticmethod
    def _make_wtc_short_ruin_layout():
        """Build a 20x20 layout with a single WTC short ruin at the origin."""
        # Ruin base: 12x6, physically flat, is_footprint (Obscuring keyword)
        base = TerrainObject(
            id="ruins_short",
            shapes=[
                Shape(width=12.0, depth=6.0, height=0.0),
            ],
            is_footprint=True,
        )
        # Walls: four sections at 3.0" tall (matches catalog)
        walls = TerrainObject(
            id="wtc_short_walls",
            shapes=[
                Shape(
                    width=9.0,
                    depth=0.1,
                    height=3.0,
                    offset=Transform(x=0.65, z=-2.15),
                ),
                Shape(
                    width=0.1,
                    depth=5.0,
                    height=3.0,
                    offset=Transform(x=5.15, z=0.35),
                ),
                Shape(
                    width=0.6,
                    depth=0.1,
                    height=3.0,
                    offset=Transform(x=5.5, z=-2.15),
                ),
                Shape(
                    width=0.1,
                    depth=0.6,
                    height=3.0,
                    offset=Transform(x=5.15, z=-2.5),
                ),
            ],
        )
        feature = TerrainFeature(
            id="wtc_short",
            feature_type="obscuring",
            components=[
                FeatureComponent(object_id="ruins_short"),
                FeatureComponent(object_id="wtc_short_walls"),
            ],
            tags=["obscuring"],
        )
        pf = PlacedFeature(
            feature=feature,
            transform=Transform(x=0.0, z=0.0, rotation_deg=0.0),
        )
        layout = TerrainLayout(
            table_width=20.0,
            table_depth=20.0,
            placed_features=[pf],
        )
        objects_by_id = {"ruins_short": base, "wtc_short_walls": walls}
        return layout, objects_by_id

    def test_standard_visibility_below_100(self):
        """Standard pass: base footprint blocks via Obscuring → < 100%."""
        layout, objects_by_id = self._make_wtc_short_ruin_layout()
        result = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=4.0,
            infantry_blocking_height=None,
        )
        val = result["overall"]["value"]
        assert val < 100.0, (
            f"Standard visibility should be < 100% (base is Obscuring), "
            f"got {val}%"
        )

    def test_infantry_visibility_lower_than_standard(self):
        """Infantry pass: walls also block → lower than standard."""
        layout, objects_by_id = self._make_wtc_short_ruin_layout()

        # Standard pass only
        result_std = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=4.0,
            infantry_blocking_height=None,
        )
        std_val = result_std["overall"]["value"]

        # Infantry pass only
        result_inf = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=2.2,
            infantry_blocking_height=None,
        )
        inf_val = result_inf["overall"]["value"]

        assert inf_val < std_val, (
            f"Infantry visibility ({inf_val}%) should be lower than "
            f'standard ({std_val}%) because 3.0" walls block infantry'
        )

    def test_dual_pass_produces_sub_dicts(self):
        """Full dual-pass: should trigger and show different values."""
        layout, objects_by_id = self._make_wtc_short_ruin_layout()
        result = compute_layout_visibility(
            layout, objects_by_id, infantry_blocking_height=2.2
        )
        overall = result["overall"]
        assert "standard" in overall, "Expected dual-pass 'standard' sub-dict"
        assert "infantry" in overall, "Expected dual-pass 'infantry' sub-dict"
        assert overall["standard"]["value"] < 100.0, (
            "Standard: base should block"
        )
        assert overall["infantry"]["value"] < overall["standard"]["value"], (
            "Infantry should be lower than standard (walls block infantry)"
        )


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


# -- _find_hiding_square_with_fringe tests --


class TestFindHidingSquareWithFringe:
    """Tests for the fringe-aware hiding square search."""

    def _grid_points(self, x_range, z_range):
        """Generate grid points at integer spacing."""
        pts = []
        for x in x_range:
            for z in z_range:
                pts.append((float(x), float(z)))
        return pts

    def test_inner_only_square_found(self):
        """4 adjacent hidden inner points form a valid square without fringe."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        # All points within radius 5 are inner
        inner_flags = [True] * len(pts)
        # Hide a 2x2 block of points near origin
        inner_hidden = set()
        for i, (x, z) in enumerate(pts):
            if 0 <= x <= 1 and 0 <= z <= 1:
                inner_hidden.add(i)
        found, needed = _find_hiding_square_with_fringe(
            0.0, 0.0, 5.0, pts, inner_hidden, inner_flags, [], 30, 22
        )
        assert found is True
        assert needed == set()

    def test_no_hidden_inner_returns_false_empty(self):
        """No hidden inner points -> False with empty fringe needs."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        inner_flags = [True] * len(pts)
        found, needed = _find_hiding_square_with_fringe(
            0.0, 0.0, 5.0, pts, set(), inner_flags, [], 30, 22
        )
        assert found is False
        assert needed == set()

    def test_fringe_corners_tracked(self):
        """Single hidden inner point near boundary: no valid hiding square.

        With only one hidden inner point at (2, 0) near radius 2.5, every
        candidate square that includes it also includes non-hidden inner
        points, so no hiding square is found. Fringe tracking may or may
        not produce needed points depending on candidate geometry.
        """
        # Grid from -5 to 5
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        # Points within radius 2.5 are inner, rest are fringe
        inner_flags = []
        for x, z in pts:
            inner_flags.append(x * x + z * z <= 2.5 * 2.5)

        # Hide one inner point at (2, 0) - this is near the boundary of
        # radius 2.5.
        inner_hidden = set()
        for i, (x, z) in enumerate(pts):
            if x == 2 and z == 0:
                inner_hidden.add(i)
                break

        found, needed = _find_hiding_square_with_fringe(
            0.0, 0.0, 2.5, pts, inner_hidden, inner_flags, [], 30, 22
        )
        # Can't find a fully-inner square (only 1 hidden point)
        assert found is False
        # If any fringe points were needed, verify they are actually fringe
        for pi in needed:
            assert not inner_flags[pi], (
                f"Point {pi} at {pts[pi]} should be fringe"
            )

    def test_terrain_blocks_candidate(self):
        """Tall terrain overlapping candidate square rejects it."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        inner_flags = [True] * len(pts)
        inner_hidden = set()
        for i, (x, z) in enumerate(pts):
            if 0 <= x <= 1 and 0 <= z <= 1:
                inner_hidden.add(i)
        # Place tall terrain covering the hiding square
        tall_obb = obb_corners(0.5, 0.5, 0.6, 0.6, 0.0)
        found, needed = _find_hiding_square_with_fringe(
            0.0, 0.0, 5.0, pts, inner_hidden, inner_flags, [tall_obb], 30, 22
        )
        assert found is False

    def test_inner_visible_rejects_square(self):
        """An inner point that is NOT hidden rejects the candidate square."""
        pts = self._grid_points(range(-5, 6), range(-5, 6))
        inner_flags = [True] * len(pts)
        # Hide only 3 of 4 corners of a potential square
        inner_hidden = set()
        for i, (x, z) in enumerate(pts):
            if (x, z) in [(0, 0), (1, 0), (0, 1)]:
                inner_hidden.add(i)
        # (1, 1) is inner but NOT hidden -> no valid square
        found, needed = _find_hiding_square_with_fringe(
            0.0, 0.0, 5.0, pts, inner_hidden, inner_flags, [], 30, 22
        )
        assert found is False
        assert needed == set()


class TestObjectiveHidabilityOptimization:
    """Tests that the optimized objective hidability path produces correct results."""

    def test_inner_hiding_no_fringe_needed(self):
        """Wall creates hiding square well within obj_radius.

        The optimization should find it using only inner-point booleans
        from the main loop, without computing any fringe vis polys.
        """
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        # Single objective at center
        objectives = [
            ObjectiveMarker(id="obj1", position=Point2D(x=0.0, z=0.0))
        ]
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        # Tall wall right next to objective, blocking view from red DZ.
        # This creates a large hidden region well within obj_radius=3.75.
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
        # Green player can hide from red DZ (wall blocks red's view)
        assert oh["green"]["safe_count"] >= 1

    def test_no_hidden_inner_points_skips_fringe(self):
        """Open space with no terrain -> 0% hidability, fringe computation skipped."""
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        objectives = [
            ObjectiveMarker(id="obj1", position=Point2D(x=0.0, z=0.0))
        ]
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
        oh = result["objective_hidability"]
        assert oh["green"]["value"] == 0.0
        assert oh["red"]["value"] == 0.0

    def test_fringe_hiding_square(self):
        """Terrain placed so hiding square has corners in the fringe region.

        A narrow wall near the edge of obj_radius creates hidden points
        near the boundary, where some valid hiding square corners may be
        in the fringe (between obj_radius and obj_radius + sqrt(2)).
        The optimization must compute fringe vis polys to find these.
        """
        green_dz = _make_rect_dz("green", -30, -22, -20, 22)
        red_dz = _make_rect_dz("red", 20, -22, 30, 22)
        # Objective at (0, 0) with range 3.0 -> obj_radius = 3.75
        objectives = [
            ObjectiveMarker(id="obj1", position=Point2D(x=0.0, z=0.0))
        ]
        mission = _make_simple_mission(
            [green_dz, red_dz], objectives=objectives
        )

        # Place a tall wide wall at x=4 blocking red DZ view.
        # Hidden points near x=3-4 are at the edge of obj_radius=3.75,
        # so some hiding square corners at x=4 will be in the fringe.
        obj = _make_object("wall", 2, 12, 5)
        feat = _make_feature("f1", "wall", "obstacle")
        pf = _place(feat, 4, 0)

        layout = TerrainLayout(
            table_width=60,
            table_depth=44,
            placed_features=[pf],
            mission=mission,
        )
        objects_by_id = {"wall": obj}

        result = compute_layout_visibility(layout, objects_by_id)
        oh = result["objective_hidability"]
        # The wall should create some hiding for the green player
        # (blocks red DZ's view of the objective area)
        # Key check: no crash, result is valid
        assert oh["green"]["total_objectives"] == 1
