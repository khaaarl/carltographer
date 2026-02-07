"""Oriented bounding box (OBB) collision and bounds checking.

Uses the Separating Axis Theorem for overlap detection between
rotated rectangles on the 2D table surface.
"""

from __future__ import annotations

import math

from .types import PlacedFeature, TerrainObject, Transform

Corners = list[tuple[float, float]]


def _mirror_placed_feature(pf: PlacedFeature) -> PlacedFeature:
    """180 degree rotational mirror: (-x, -z, rot+180)."""
    t = pf.transform
    return PlacedFeature(
        feature=pf.feature,
        transform=Transform(
            x=-t.x, z=-t.z, rotation_deg=t.rotation_deg + 180.0
        ),
    )


def _is_at_origin(pf: PlacedFeature) -> bool:
    return pf.transform.x == 0.0 and pf.transform.z == 0.0


def compose_transform(inner: Transform, outer: Transform) -> Transform:
    """Apply inner transform, then outer transform."""
    cos_o = math.cos(math.radians(outer.rotation_deg))
    sin_o = math.sin(math.radians(outer.rotation_deg))
    return Transform(
        x=outer.x + inner.x * cos_o - inner.z * sin_o,
        z=outer.z + inner.x * sin_o + inner.z * cos_o,
        rotation_deg=inner.rotation_deg + outer.rotation_deg,
    )


def obb_corners(
    cx: float,
    cz: float,
    half_w: float,
    half_d: float,
    rot_rad: float,
) -> Corners:
    """Compute the 4 corners of a rotated rectangle."""
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    result: Corners = []
    for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx = sx * half_w
        lz = sz * half_d
        result.append(
            (
                cx + lx * cos_r - lz * sin_r,
                cz + lx * sin_r + lz * cos_r,
            )
        )
    return result


def _project(corners: Corners, ax: float, az: float) -> tuple[float, float]:
    """Project corners onto an axis, return (min, max)."""
    dots = [c[0] * ax + c[1] * az for c in corners]
    return min(dots), max(dots)


def obbs_overlap(a: Corners, b: Corners) -> bool:
    """True if the interiors of two OBBs overlap.

    Touching (shared edge or corner) is NOT counted as overlap.
    """
    for corners in (a, b):
        # Only need 2 edge normals per rectangle (opposite
        # edges are parallel and give the same axis).
        for i in range(2):
            j = (i + 1) % 4
            ex = corners[j][0] - corners[i][0]
            ez = corners[j][1] - corners[i][1]
            # Perpendicular to edge
            ax, az = -ez, ex
            min_a, max_a = _project(a, ax, az)
            min_b, max_b = _project(b, ax, az)
            if max_a <= min_b or max_b <= min_a:
                return False
    return True


def obb_in_bounds(
    corners: Corners,
    table_width: float,
    table_depth: float,
) -> bool:
    """True if all corners are within (or on) the table edges.

    Table coordinates are centered: x in [-w/2, w/2],
    z in [-d/2, d/2].
    """
    half_w = table_width / 2
    half_d = table_depth / 2
    return all(
        -half_w <= cx <= half_w and -half_d <= cz <= half_d
        for cx, cz in corners
    )


def get_world_obbs(
    placed: PlacedFeature,
    objects_by_id: dict[str, TerrainObject],
) -> list[Corners]:
    """Compute world-space OBB corners for every shape in a
    placed feature, composing shape offset, component
    transform, and feature table transform.
    """
    result: list[Corners] = []
    for comp in placed.feature.components:
        obj = objects_by_id[comp.object_id]
        comp_t = comp.transform or Transform()
        for shape in obj.shapes:
            shape_t = shape.offset or Transform()
            world = compose_transform(
                compose_transform(shape_t, comp_t),
                placed.transform,
            )
            corners = obb_corners(
                world.x,
                world.z,
                shape.width / 2,
                shape.depth / 2,
                math.radians(world.rotation_deg),
            )
            result.append(corners)
    return result


def point_to_segment_distance_squared(
    px: float,
    pz: float,
    x1: float,
    z1: float,
    x2: float,
    z2: float,
) -> float:
    """Compute squared distance from point to line segment.

    Uses vector projection. If projection falls within segment,
    returns perpendicular distance. Otherwise returns distance
    to nearest endpoint.
    """
    seg_len_sq = (x2 - x1) ** 2 + (z2 - z1) ** 2
    if seg_len_sq == 0:
        return (px - x1) ** 2 + (pz - z1) ** 2

    t = ((px - x1) * (x2 - x1) + (pz - z1) * (z2 - z1)) / seg_len_sq

    if 0 < t < 1:
        proj_x = x1 + t * (x2 - x1)
        proj_z = z1 + t * (z2 - z1)
        return (px - proj_x) ** 2 + (pz - proj_z) ** 2
    else:
        dist_to_start = (px - x1) ** 2 + (pz - z1) ** 2
        dist_to_end = (px - x2) ** 2 + (pz - z2) ** 2
        return min(dist_to_start, dist_to_end)


def segments_intersect(
    x1: float,
    z1: float,
    x2: float,
    z2: float,
    x3: float,
    z3: float,
    x4: float,
    z4: float,
) -> bool:
    """Test if two line segments intersect (excluding endpoints).

    Uses parametric line intersection algorithm.
    """
    denominator = (z4 - z3) * (x2 - x1) - (x4 - x3) * (z2 - z1)

    if denominator == 0:
        return False

    ua = ((x4 - x3) * (z1 - z3) - (z4 - z3) * (x1 - x3)) / denominator
    if ua <= 0 or ua >= 1:
        return False

    ub = ((x2 - x1) * (z1 - z3) - (z2 - z1) * (x1 - x3)) / denominator
    return 0 < ub < 1


def obb_distance(corners_a: Corners, corners_b: Corners) -> float:
    """Compute minimum distance between two oriented bounding boxes.

    Returns 0 if rectangles intersect or touch. Otherwise returns
    minimum distance between any corner and edge.

    Algorithm:
    1. Check edge pairs for intersection → distance 0
    2. Compute corner-to-edge distances for all pairs
    3. Return minimum
    """
    edges_a = [(corners_a[i], corners_a[(i + 1) % 4]) for i in range(4)]
    edges_b = [(corners_b[i], corners_b[(i + 1) % 4]) for i in range(4)]

    # Check for edge intersections
    for (ax1, az1), (ax2, az2) in edges_a:
        for (bx1, bz1), (bx2, bz2) in edges_b:
            if segments_intersect(ax1, az1, ax2, az2, bx1, bz1, bx2, bz2):
                return 0.0

    min_dist_sq = float("inf")

    # Check corners of A against edges of B
    for corner_x, corner_z in corners_a:
        for (bx1, bz1), (bx2, bz2) in edges_b:
            dist_sq = point_to_segment_distance_squared(
                corner_x, corner_z, bx1, bz1, bx2, bz2
            )
            min_dist_sq = min(min_dist_sq, dist_sq)

    # Check corners of B against edges of A
    for corner_x, corner_z in corners_b:
        for (ax1, az1), (ax2, az2) in edges_a:
            dist_sq = point_to_segment_distance_squared(
                corner_x, corner_z, ax1, az1, ax2, az2
            )
            min_dist_sq = min(min_dist_sq, dist_sq)

    return math.sqrt(min_dist_sq)


def obb_to_table_edge_distance(
    corners: Corners,
    table_width: float,
    table_depth: float,
) -> float:
    """Compute minimum distance from OBB to nearest table edge.

    Returns 0 if any corner is outside or on table boundary.
    """
    half_w = table_width / 2
    half_d = table_depth / 2

    min_dist = float("inf")

    for cx, cz in corners:
        dist_to_right = half_w - cx
        dist_to_left = half_w + cx
        dist_to_top = half_d - cz
        dist_to_bottom = half_d + cz

        if (
            dist_to_right <= 0
            or dist_to_left <= 0
            or dist_to_top <= 0
            or dist_to_bottom <= 0
        ):
            return 0.0

        min_dist = min(
            min_dist, dist_to_right, dist_to_left, dist_to_top, dist_to_bottom
        )

    return min_dist


def get_tall_world_obbs(
    placed: PlacedFeature,
    objects_by_id: dict[str, TerrainObject],
    min_height: float = 1.0,
) -> list[Corners]:
    """Extract world-space OBB corners for shapes with height >= min_height.

    Similar to get_world_obbs() but filters by shape height.
    """
    result: list[Corners] = []
    for comp in placed.feature.components:
        obj = objects_by_id[comp.object_id]
        comp_t = comp.transform or Transform()
        for shape in obj.shapes:
            if shape.height < min_height:
                continue

            shape_t = shape.offset or Transform()
            world = compose_transform(
                compose_transform(shape_t, comp_t),
                placed.transform,
            )
            corners = obb_corners(
                world.x,
                world.z,
                shape.width / 2,
                shape.depth / 2,
                math.radians(world.rotation_deg),
            )
            result.append(corners)
    return result


def is_valid_placement(
    placed_features: list[PlacedFeature],
    check_idx: int,
    table_width: float,
    table_depth: float,
    objects_by_id: dict[str, TerrainObject],
    min_feature_gap: float | None = None,
    min_edge_gap: float | None = None,
    rotationally_symmetric: bool = False,
) -> bool:
    """Check that the feature at check_idx is validly placed.

    Validates:
    1. All shapes within table bounds
    2. No overlap with other features (including mirrors if symmetric)
    3. Tall shapes (height >= 1") respect min_feature_gap
    4. Tall shapes (height >= 1") respect min_edge_gap
    """
    check_pf = placed_features[check_idx]

    # Get all shapes for bounds and overlap checking
    check_obbs = get_world_obbs(check_pf, objects_by_id)

    # 1. Check table bounds
    for corners in check_obbs:
        if not obb_in_bounds(corners, table_width, table_depth):
            return False

    # Build expanded "other features" list including mirrors when symmetric
    other_features: list[PlacedFeature] = []
    for i, pf in enumerate(placed_features):
        if i == check_idx:
            continue
        other_features.append(pf)
        if rotationally_symmetric and not _is_at_origin(pf):
            other_features.append(_mirror_placed_feature(pf))

    # Also check feature's own mirror (can't overlap itself)
    if rotationally_symmetric and not _is_at_origin(check_pf):
        other_features.append(_mirror_placed_feature(check_pf))

    # 2. Check overlap with other features
    for pf in other_features:
        other_obbs = get_world_obbs(pf, objects_by_id)
        for ca in check_obbs:
            for cb in other_obbs:
                if obbs_overlap(ca, cb):
                    return False

    # Gap checking only for tall geometries (height >= 1")
    check_tall = get_tall_world_obbs(
        check_pf,
        objects_by_id,
        min_height=1.0,
    )

    if not check_tall:
        return True

    # 3. Check edge gap
    if min_edge_gap is not None and min_edge_gap > 0:
        for corners in check_tall:
            dist = obb_to_table_edge_distance(
                corners, table_width, table_depth
            )
            if dist < min_edge_gap:
                return False

    # 4. Check feature gap
    if min_feature_gap is not None and min_feature_gap > 0:
        for pf in other_features:
            other_tall = get_tall_world_obbs(pf, objects_by_id, min_height=1.0)
            for ca in check_tall:
                for cb in other_tall:
                    dist = obb_distance(ca, cb)
                    if dist < min_feature_gap:
                        return False

    return True
