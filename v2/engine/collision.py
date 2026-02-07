"""Oriented bounding box (OBB) collision and bounds checking.

Uses the Separating Axis Theorem for overlap detection between
rotated rectangles on the 2D table surface.
"""

from __future__ import annotations

import math

from .types import PlacedFeature, TerrainObject, Transform

Corners = list[tuple[float, float]]


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


def is_valid_placement(
    placed_features: list[PlacedFeature],
    check_idx: int,
    table_width: float,
    table_depth: float,
    objects_by_id: dict[str, TerrainObject],
) -> bool:
    """Check that the feature at check_idx is within table
    bounds and does not overlap any other placed feature.
    """
    check_obbs = get_world_obbs(placed_features[check_idx], objects_by_id)
    for corners in check_obbs:
        if not obb_in_bounds(corners, table_width, table_depth):
            return False
    for i, pf in enumerate(placed_features):
        if i == check_idx:
            continue
        other_obbs = get_world_obbs(pf, objects_by_id)
        for ca in check_obbs:
            for cb in other_obbs:
                if obbs_overlap(ca, cb):
                    return False
    return True
