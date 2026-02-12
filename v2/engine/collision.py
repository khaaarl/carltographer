"""Collision detection, gap enforcement, and bounds checking for terrain shapes.

The central question this module answers: "is this feature placement legal?"
Every mutation in ``mutation.py`` calls ``is_valid_placement`` after
tentatively placing or moving a feature. The check enforces:

  * **No overlap** — shapes of the placed feature (and its rotational mirror,
    if symmetric) must not intersect any other feature's shapes. For
    rectangular shapes (4 corners), overlap uses SAT via ``obbs_overlap``.
    For polygon shapes (N corners), overlap uses ``polygons_overlap`` with
    edge-intersection + vertex-containment tests.
  * **Table bounds** — all shape corners must lie within the table.
  * **Tall-terrain gaps** — features with any shape height >= 1" must keep a
    minimum distance from other tall features (``min_feature_gap``) and from
    table edges (``min_edge_gap``). Distance is measured between edges,
    not centers.
  * **All-terrain gaps** — optional stricter gaps (``min_all_feature_gap``,
    ``min_all_edge_gap``) that apply to every feature regardless of height.

Shapes can be either rectangular prisms (4-corner OBBs) or arbitrary polygons
(N-corner footprints). The ``Shape.vertices`` field distinguishes them: when
present, vertices are transformed to world space instead of computing OBB
corners. Most functions (``obb_in_bounds``, ``obb_to_table_edge_distance``,
``obb_distance``) already work generically with any vertex count.

Also provides geometric utilities used elsewhere in the engine:

  * ``compose_transform`` / ``obb_corners`` / ``get_world_obbs`` — transform
    composition and shape vertex computation, used by ``mutation.py`` for
    tile-weight occupancy.
  * ``_shape_world_corners`` / ``_transform_polygon`` — polygon-aware helpers
    for computing world-space vertices.
  * ``_mirror_placed_feature`` / ``_is_at_origin`` — rotational symmetry
    helpers (180° mirroring), used by mutations and visibility.
  * ``obb_distance`` / ``obb_to_table_edge_distance`` — edge-to-edge distance
    for gap enforcement.
  * ``point_in_polygon`` / ``polygons_overlap`` — polygon containment and
    overlap tests, used for deployment zone analysis and polygon terrain
    overlap checks.

Subject to the Rust-parity constraint for rectangular shapes —
``engine_rs/src/collision.rs`` mirrors this module. Polygon support is
Python-only for now (Rust parity deferred).
"""

from __future__ import annotations

import math

import numpy as np

from .types import PlacedFeature, Shape, TerrainObject, Transform

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


def _transform_polygon(
    vertices: list[tuple[float, float]],
    cx: float,
    cz: float,
    rot_rad: float,
) -> Corners:
    """Rotate and translate local-space polygon vertices to world space."""
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    return [
        (cx + vx * cos_r - vz * sin_r, cz + vx * sin_r + vz * cos_r)
        for vx, vz in vertices
    ]


def _shape_world_corners(
    shape: Shape,
    world: Transform,
) -> Corners:
    """Compute world-space corners for a shape (rectangle or polygon)."""
    if shape.vertices is not None:
        return _transform_polygon(
            shape.vertices,
            world.x,
            world.z,
            math.radians(world.rotation_deg),
        )
    return obb_corners(
        world.x,
        world.z,
        shape.width / 2,
        shape.depth / 2,
        math.radians(world.rotation_deg),
    )


def get_world_obbs(
    placed: PlacedFeature,
    objects_by_id: dict[str, TerrainObject],
) -> list[Corners]:
    """Compute world-space corners for every shape in a placed feature.

    For rectangular shapes, returns 4-corner OBBs. For polygon shapes,
    returns the transformed polygon vertices (N corners).
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
            result.append(_shape_world_corners(shape, world))
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


def segments_intersect_inclusive(
    x1: float,
    z1: float,
    x2: float,
    z2: float,
    x3: float,
    z3: float,
    x4: float,
    z4: float,
) -> bool:
    """Test if two line segments intersect (including endpoints).

    Uses parametric line intersection algorithm with closed interval [0, 1].
    """
    denominator = (z4 - z3) * (x2 - x1) - (x4 - x3) * (z2 - z1)

    if denominator == 0:
        return False

    ua = ((x4 - x3) * (z1 - z3) - (z4 - z3) * (x1 - x3)) / denominator
    if ua < 0 or ua > 1:
        return False

    ub = ((x2 - x1) * (z1 - z3) - (z2 - z1) * (x1 - x3)) / denominator
    return 0 <= ub <= 1


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
    n_a = len(corners_a)
    n_b = len(corners_b)
    edges_a = [(corners_a[i], corners_a[(i + 1) % n_a]) for i in range(n_a)]
    edges_b = [(corners_b[i], corners_b[(i + 1) % n_b]) for i in range(n_b)]

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
    """Extract world-space corners for shapes with height >= min_height.

    Similar to get_world_obbs() but filters by shape height.
    Returns polygon vertices for polygon shapes, 4-corner OBBs for rectangles.
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
            result.append(_shape_world_corners(shape, world))
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
    min_all_feature_gap: float | None = None,
    min_all_edge_gap: float | None = None,
) -> bool:
    """Check that the feature at check_idx is validly placed.

    Validates:
    1. All shapes within table bounds
    2. No overlap with other features (including mirrors if symmetric)
    2b. All shapes respect min_all_edge_gap from table edges
    2c. All shapes respect min_all_feature_gap between features
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
    # Use obbs_overlap for rect-vs-rect (touching = OK, Rust-parity),
    # polygons_overlap for any polygon involvement.
    for pf in other_features:
        other_obbs = get_world_obbs(pf, objects_by_id)
        for ca in check_obbs:
            for cb in other_obbs:
                if len(ca) == 4 and len(cb) == 4:
                    if obbs_overlap(ca, cb):
                        return False
                elif polygons_overlap(ca, cb):
                    return False

    # 2b. All-feature edge gap (applies to all shapes, not just tall)
    if min_all_edge_gap is not None and min_all_edge_gap > 0:
        for corners in check_obbs:
            dist = obb_to_table_edge_distance(
                corners, table_width, table_depth
            )
            if dist < min_all_edge_gap:
                return False

    # 2c. All-feature gap (applies to all shapes, not just tall)
    if min_all_feature_gap is not None and min_all_feature_gap > 0:
        for pf in other_features:
            other_obbs = get_world_obbs(pf, objects_by_id)
            for ca in check_obbs:
                for cb in other_obbs:
                    dist = obb_distance(ca, cb)
                    if dist < min_all_feature_gap:
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


def point_in_polygon(
    px: float, pz: float, vertices: list[tuple[float, float]]
) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = vertices[i]
        xj, zj = vertices[j]
        if (zi > pz) != (zj > pz):
            intersect_x = (xj - xi) * (pz - zi) / (zj - zi) + xi
            if px < intersect_x:
                inside = not inside
        j = i
    return inside


def polygons_overlap(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """Test if two polygons overlap (share any interior area).

    Checks:
    1. Edge-edge intersections (inclusive of endpoints)
    2. Containment of any vertex of A inside B
    3. Containment of any vertex of B inside A
    """
    n_a = len(poly_a)
    n_b = len(poly_b)
    if n_a < 3 or n_b < 3:
        return False

    # AABB early-exit: skip expensive edge tests for distant polygons
    a_xs = [p[0] for p in poly_a]
    a_zs = [p[1] for p in poly_a]
    b_xs = [p[0] for p in poly_b]
    b_zs = [p[1] for p in poly_b]
    if (
        max(a_xs) < min(b_xs)
        or max(b_xs) < min(a_xs)
        or max(a_zs) < min(b_zs)
        or max(b_zs) < min(a_zs)
    ):
        return False

    # 1. Vectorized edge-edge intersection
    # Build edge arrays: each edge is (x1, z1, x2, z2)
    edges_a = np.empty((n_a, 4), dtype=np.float64)
    for i in range(n_a):
        j = (i + 1) % n_a
        edges_a[i] = (*poly_a[i], *poly_a[j])
    edges_b = np.empty((n_b, 4), dtype=np.float64)
    for i in range(n_b):
        j = (i + 1) % n_b
        edges_b[i] = (*poly_b[i], *poly_b[j])

    # Broadcast: (E_a, 1, 4) vs (1, E_b, 4) → cross products over all pairs
    ax1 = edges_a[:, 0:1]  # (E_a, 1)
    az1 = edges_a[:, 1:2]
    ax2 = edges_a[:, 2:3]
    az2 = edges_a[:, 3:4]
    bx1 = edges_b[:, 0:1].T  # (1, E_b)
    bz1 = edges_b[:, 1:2].T
    bx2 = edges_b[:, 2:3].T
    bz2 = edges_b[:, 3:4].T

    denom = (bz2 - bz1) * (ax2 - ax1) - (bx2 - bx1) * (az2 - az1)
    # Avoid division by zero for parallel segments
    non_parallel = denom != 0.0
    safe_denom = np.where(non_parallel, denom, 1.0)
    ua = ((bx2 - bx1) * (az1 - bz1) - (bz2 - bz1) * (ax1 - bx1)) / safe_denom
    ub = ((ax2 - ax1) * (az1 - bz1) - (az2 - az1) * (ax1 - bx1)) / safe_denom
    hits = non_parallel & (ua >= 0) & (ua <= 1) & (ub >= 0) & (ub <= 1)
    if np.any(hits):
        return True

    # 2. Any vertex of A inside B
    for px, pz in poly_a:
        if point_in_polygon(px, pz, poly_b):
            return True

    # 3. Any vertex of B inside A
    for px, pz in poly_b:
        if point_in_polygon(px, pz, poly_a):
            return True

    return False
