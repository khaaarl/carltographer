"""Visibility scoring for terrain layouts.

This module answers the question: how much of the table can a model standing
at position X see? It does this for hundreds of positions across the table,
then aggregates the results into several metrics that the generation engine
(generate.py) uses as its fitness signal during phase-2 optimization.

The core algorithm is an angular sweep visibility polygon. For each observer
position, we cast rays toward every vertex of every blocking segment (plus
epsilon-offset rays to see around corners). Each ray finds the nearest
segment intersection, and the resulting hit points form a visibility polygon.
The area of that polygon, divided by total table area, gives that observer's
visibility ratio.

Terrain blocks line of sight in two ways, depending on feature_type:
- Tall physical shapes (e.g. a stack of crates, a windowless wall section):
  all edges of shapes taller than min_blocking_height (default 4") always
  block regardless of observer position. These become "static segments"
  computed once per layout. This includes opaque walls inside ruins.
- Obscuring features (e.g. ruins in 40k): only the back-facing edges of the
  feature's outer footprint block, meaning an observer inside the ruin can
  see out. For each observer, we check which edges face away from them using
  precomputed outward normals and add only those to the segment list.

The module computes four metrics, all in the range 0-100%:

  overall         Average visibility ratio across all observer positions.
                  Lower means more terrain is blocking sightlines.

  dz_visibility   Per-deployment-zone: average fraction of a DZ's sample
                  points visible from observers OUTSIDE that DZ. Measures
                  how exposed each DZ is to the rest of the table.

  dz_to_dz        Cross-zone: fraction of target DZ sample points that at
                  least one observer in the source DZ's threat zone (DZ +
                  6" expansion) can see. Reported as hidden %, so higher
                  means more of the target DZ is concealed.

  objective_hidability
                  Per-DZ: fraction of objectives where at least one sample
                  point within the objective's range circle is hidden from
                  every observer in the opposing DZ's threat zone. Higher
                  means more objectives have safe approach angles.

Observer positions are laid out on an integer grid across the table, using
2" spacing (even coordinates) everywhere, with 1" spacing near objective
markers for finer resolution where it matters most. Points that fall inside
tall terrain footprints (height >= 1") are excluded — models will rarely be
on top of these features (and in many cases it's an illegal placement), so
we measure visibility from ground level only.

Performance notes:
- Segment data is precomputed once per layout (_precompute_segments), then
  per-observer work is just selecting obscuring back-faces + the ray sweep.
- VisibilityCache provides incremental updates: when a single feature is
  added/moved/removed during mutation, only that feature's blocked-point
  mask is recomputed rather than re-testing all ~600 observers against all
  footprints. This uses NumPy vectorized point-in-polygon for batch testing.
- The Rust engine (engine_rs) reimplements this module for ~10x throughput.
  Both produce identical scores for the same layout.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from .collision import (
    _is_at_origin,
    _mirror_placed_feature,
    compose_transform,
    get_tall_world_obbs,
    get_world_obbs,
    obb_corners,
    obbs_overlap,
    point_to_segment_distance_squared,
    polygons_overlap,
)
from .types import (
    DeploymentZone,
    PlacedFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)

Segment = tuple[float, float, float, float]  # (x1, z1, x2, z2)

DZ_EXPANSION_INCHES = 6.0


def _expand_dz_polygons(
    dz_polygons: list[list[tuple[float, float]]],
    expansion: float,
) -> list[list[tuple[float, float]]]:
    """Expand deployment zone polygons outward by the given distance.

    Uses shapely's buffer() to handle convex and concave shapes correctly.
    Returns expanded polygon(s) as lists of coordinate tuples.
    """
    if expansion <= 0:
        return dz_polygons

    result: list[list[tuple[float, float]]] = []
    for ring in dz_polygons:
        if len(ring) < 3:
            continue
        sp = ShapelyPolygon(ring)
        buffered = sp.buffer(expansion)
        if buffered.is_empty:
            continue
        if buffered.geom_type == "MultiPolygon":
            for geom in buffered.geoms:
                coords = list(geom.exterior.coords)
                if coords and coords[-1] == coords[0]:
                    coords = coords[:-1]
                result.append(coords)
        else:
            coords = list(buffered.exterior.coords)
            if coords and coords[-1] == coords[0]:
                coords = coords[:-1]
            result.append(coords)
    return result


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    """Compute polygon area using the shoelace formula.

    Returns positive area regardless of winding order.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def _point_in_polygon(
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


def _ray_segment_intersection(
    ox: float,
    oz: float,
    dx: float,
    dz: float,
    x1: float,
    z1: float,
    x2: float,
    z2: float,
) -> float | None:
    """Find parameter t where ray (ox+t*dx, oz+t*dz) hits segment (x1,z1)-(x2,z2).

    Returns t >= 0 if hit, None if miss or parallel.
    """
    sx = x2 - x1
    sz = z2 - z1
    denom = dx * sz - dz * sx
    if abs(denom) < 1e-12:
        return None

    t = ((x1 - ox) * sz - (z1 - oz) * sx) / denom
    u = ((x1 - ox) * dz - (z1 - oz) * dx) / denom

    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def _get_footprint_corners(
    placed: PlacedFeature,
    objects_by_id: dict[str, TerrainObject],
) -> list[list[tuple[float, float]]]:
    """Get world-space OBB corners for all shapes in a feature (ignoring height)."""
    return get_world_obbs(placed, objects_by_id)


# Edge with precomputed outward normal: (x1, z1, x2, z2, mx, mz, nx, nz)
_PrecomputedEdge = tuple[
    float, float, float, float, float, float, float, float
]


# Precomputed obscuring shape: (corners, edges_with_normals)
_ObscuringShape = tuple[list[tuple[float, float]], list[_PrecomputedEdge]]


# Full precomputed data: (static_segments, obscuring_shapes)
_PrecomputedSegments = tuple[list[Segment], list[_ObscuringShape]]


def _precompute_segments(
    layout: TerrainLayout,
    objects_by_id: dict[str, TerrainObject],
    min_blocking_height: float = 4.0,
) -> _PrecomputedSegments:
    """Precompute observer-independent segment data.

    Called once before the observer loop. Returns:
    - static_segments: segments from regular obstacles (always block, observer-independent)
    - obscuring_shapes: precomputed corners + edge normals for obscuring features
    """
    static_segments: list[Segment] = []
    obscuring_shapes: list[_ObscuringShape] = []

    # Build effective placed features list (include mirrors for symmetric)
    effective_features: list[PlacedFeature] = []
    for pf in layout.placed_features:
        effective_features.append(pf)
        if layout.rotationally_symmetric and not _is_at_origin(pf):
            effective_features.append(_mirror_placed_feature(pf))

    for pf in effective_features:
        is_obscuring = pf.feature.feature_type == "obscuring"

        if is_obscuring:
            # Get all shape footprints regardless of height
            all_corners = _get_footprint_corners(pf, objects_by_id)
            if not all_corners:
                continue

            # Precompute edges with outward normals for each shape
            for corners in all_corners:
                n = len(corners)
                cx = sum(c[0] for c in corners) / n
                cz = sum(c[1] for c in corners) / n
                edges: list[_PrecomputedEdge] = []
                for i in range(n):
                    j = (i + 1) % n
                    x1, z1 = corners[i]
                    x2, z2 = corners[j]
                    mx = (x1 + x2) / 2.0
                    mz = (z1 + z2) / 2.0
                    ex = x2 - x1
                    ez = z2 - z1
                    nx, nz = ez, -ex
                    dot_center = (cx - mx) * nx + (cz - mz) * nz
                    if dot_center > 0:
                        nx, nz = -nx, -nz
                    edges.append((x1, z1, x2, z2, mx, mz, nx, nz))
                obscuring_shapes.append((corners, edges))
        else:
            # Regular obstacle: precompute transforms + corners once
            for comp in pf.feature.components:
                obj = objects_by_id.get(comp.object_id)
                if obj is None:
                    continue
                comp_t = comp.transform or Transform()
                for shape in obj.shapes:
                    if shape.effective_opacity_height() < min_blocking_height:
                        continue
                    shape_t = shape.offset or Transform()
                    world = compose_transform(
                        compose_transform(shape_t, comp_t),
                        pf.transform,
                    )
                    corners = obb_corners(
                        world.x,
                        world.z,
                        shape.width / 2,
                        shape.depth / 2,
                        math.radians(world.rotation_deg),
                    )
                    for i in range(4):
                        j = (i + 1) % 4
                        static_segments.append(
                            (
                                corners[i][0],
                                corners[i][1],
                                corners[j][0],
                                corners[j][1],
                            )
                        )

    return static_segments, obscuring_shapes


def _get_observer_segments(
    precomputed: _PrecomputedSegments,
    observer_x: float,
    observer_z: float,
) -> list[Segment]:
    """Get blocking segments for a specific observer position.

    Starts from precomputed static segments, adds back-facing edges
    from obscuring features based on observer position.
    """
    static_segments, obscuring_shapes = precomputed

    # Start with a copy of static segments
    segments = list(static_segments)

    # Add back-facing edges from obscuring shapes
    for corners, edges in obscuring_shapes:
        # Check if observer is inside this shape
        if _point_in_polygon(observer_x, observer_z, corners):
            continue  # Can see out from inside

        # Observer outside: only back-facing edges block
        for x1, z1, x2, z2, mx, mz, nx, nz in edges:
            dot_observer = (observer_x - mx) * nx + (observer_z - mz) * nz
            if dot_observer < 0:
                segments.append((x1, z1, x2, z2))

    return segments


def _extract_blocking_segments(
    layout: TerrainLayout,
    objects_by_id: dict[str, TerrainObject],
    observer_x: float,
    observer_z: float,
    min_blocking_height: float = 4.0,
) -> list[Segment]:
    """Extract blocking segments for a single observer (convenience wrapper).

    Calls precompute + get_observer_segments. For multiple observers,
    use _precompute_segments() once + _get_observer_segments() per observer.
    """
    precomputed = _precompute_segments(
        layout, objects_by_id, min_blocking_height
    )
    return _get_observer_segments(precomputed, observer_x, observer_z)


def _compute_visibility_polygon(
    ox: float,
    oz: float,
    segments: list[Segment],
    table_half_w: float,
    table_half_d: float,
) -> list[tuple[float, float]]:
    """Compute visibility polygon from observer (ox, oz) via angular sweep.

    Returns polygon vertices sorted by angle.

    Uses numpy-vectorized intersection testing: all rays are tested against
    all segments in a single (R x S) matrix operation, replacing the O(R*S)
    Python inner loop with C-level numpy batch computation.
    """
    # Table boundary segments
    tw, td = table_half_w, table_half_d
    table_boundary: list[Segment] = [
        (-tw, -td, tw, -td),  # bottom
        (tw, -td, tw, td),  # right
        (tw, td, -tw, td),  # top
        (-tw, td, -tw, -td),  # left
    ]

    # Build segment array: (S, 4) for all blocking + boundary segments
    all_segs = np.array(
        list(itertools.chain(segments, table_boundary)), dtype=np.float64
    )
    # Precompute per-segment values (S,)
    seg_dx = all_segs[:, 2] - all_segs[:, 0]
    seg_dz = all_segs[:, 3] - all_segs[:, 1]
    d_x1 = all_segs[:, 0] - ox  # x1 - ox for each segment
    d_z1 = all_segs[:, 1] - oz  # z1 - oz for each segment
    # t numerator is observer-independent: same for all rays
    num_t = d_x1 * seg_dz - d_z1 * seg_dx  # (S,)

    # Collect unique endpoints from segments
    endpoints: set[tuple[float, float]] = set()
    for x1, z1, x2, z2 in itertools.chain(segments, table_boundary):
        endpoints.add((x1, z1))
        endpoints.add((x2, z2))

    eps_arr = np.array(list(endpoints), dtype=np.float64)  # (P, 2)
    dx_arr = eps_arr[:, 0] - ox
    dz_arr = eps_arr[:, 1] - oz
    angles = np.arctan2(dz_arr, dx_arr)  # (P,)
    dist = np.sqrt(dx_arr * dx_arr + dz_arr * dz_arr)  # (P,)
    # Normalized direction for direct ray
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    ndx = np.where(dist > 1e-12, dx_arr / safe_dist, np.cos(angles))
    ndz = np.where(dist > 1e-12, dz_arr / safe_dist, np.sin(angles))

    EPS = 1e-5
    a_minus = angles - EPS
    a_plus = angles + EPS

    # Build rays array: interleave (minus, direct, plus) per endpoint
    n_pts = len(endpoints)
    ray_angles = np.empty(n_pts * 3, dtype=np.float64)
    ray_dx = np.empty(n_pts * 3, dtype=np.float64)
    ray_dz = np.empty(n_pts * 3, dtype=np.float64)
    ray_angles[0::3] = a_minus
    ray_angles[1::3] = angles
    ray_angles[2::3] = a_plus
    ray_dx[0::3] = np.cos(a_minus)
    ray_dx[1::3] = ndx
    ray_dx[2::3] = np.cos(a_plus)
    ray_dz[0::3] = np.sin(a_minus)
    ray_dz[1::3] = ndz
    ray_dz[2::3] = np.sin(a_plus)

    # Sort rays by angle
    sort_idx = np.argsort(ray_angles)
    ray_dx = ray_dx[sort_idx]
    ray_dz = ray_dz[sort_idx]

    # --- Vectorized intersection: (R, S) matrix computation ---
    # denom[r, s] = ray_dx[r] * seg_dz[s] - ray_dz[r] * seg_dx[s]
    denom = (
        ray_dx[:, None] * seg_dz[None, :] - ray_dz[:, None] * seg_dx[None, :]
    )

    # Mask degenerate (parallel) cases
    valid_denom = np.abs(denom) >= 1e-12
    safe_denom = np.where(valid_denom, denom, 1.0)

    # t[r, s] = num_t[s] / denom[r, s]  (num_t is ray-independent)
    t = num_t[None, :] / safe_denom

    # u[r, s] = (d_x1[s] * ray_dz[r] - d_z1[s] * ray_dx[r]) / denom[r, s]
    num_u = d_x1[None, :] * ray_dz[:, None] - d_z1[None, :] * ray_dx[:, None]
    u = num_u / safe_denom

    # Valid intersections: non-degenerate, t >= 0, 0 <= u <= 1
    valid = valid_denom & (t >= 0) & (u >= 0) & (u <= 1)
    t_valid = np.where(valid, t, np.inf)

    # Find nearest intersection per ray
    min_t = np.min(t_valid, axis=1)  # (R,)

    # Build polygon from finite intersections
    finite = np.isfinite(min_t)
    px = ox + min_t[finite] * ray_dx[finite]
    pz = oz + min_t[finite] * ray_dz[finite]

    return list(zip(px.tolist(), pz.tolist()))


def _sample_points_in_polygon(
    polygon: list[tuple[float, float]],
    grid_spacing: float,
    grid_offset: float,
) -> list[tuple[float, float]]:
    """Generate grid sample points that fall inside a polygon.

    Uses AABB bounding box to limit search, then PIP-filters.
    """
    if len(polygon) < 3:
        return []
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    points: list[tuple[float, float]] = []
    # Align grid to global grid (same as table-wide grid)
    # Find first grid x >= min_x
    start_x = (
        math.floor((min_x - grid_offset) / grid_spacing) * grid_spacing
        + grid_offset
    )
    start_z = (
        math.floor((min_z - grid_offset) / grid_spacing) * grid_spacing
        + grid_offset
    )
    if start_x < min_x:
        start_x += grid_spacing
    if start_z < min_z:
        start_z += grid_spacing

    x = start_x
    while x < max_x:
        z = start_z
        while z < max_z:
            if _point_in_polygon(x, z, polygon):
                points.append((x, z))
            z += grid_spacing
        x += grid_spacing
    return points


def _sample_points_in_dz(
    dz: DeploymentZone,
    grid_spacing: float,
    grid_offset: float,
) -> list[tuple[float, float]]:
    """Generate grid sample points across all polygon rings of a deployment zone."""
    seen: set[tuple[float, float]] = set()
    points: list[tuple[float, float]] = []
    for poly in dz.polygons:
        poly_tuples = [(p.x, p.z) for p in poly]
        for pt in _sample_points_in_polygon(
            poly_tuples, grid_spacing, grid_offset
        ):
            if pt not in seen:
                seen.add(pt)
                points.append(pt)
    return points


def _sample_points_in_circle(
    cx: float,
    cz: float,
    radius: float,
    grid_spacing: float,
    grid_offset: float,
) -> list[tuple[float, float]]:
    """Generate grid sample points that fall inside a circle.

    Uses AABB bounding box to limit search, then distance-filters.
    """
    if radius <= 0:
        return []
    min_x = cx - radius
    max_x = cx + radius
    min_z = cz - radius
    max_z = cz + radius

    # Align to global grid
    start_x = (
        math.floor((min_x - grid_offset) / grid_spacing) * grid_spacing
        + grid_offset
    )
    start_z = (
        math.floor((min_z - grid_offset) / grid_spacing) * grid_spacing
        + grid_offset
    )
    if start_x < min_x:
        start_x += grid_spacing
    if start_z < min_z:
        start_z += grid_spacing

    r_sq = radius * radius
    points: list[tuple[float, float]] = []
    x = start_x
    while x < max_x:
        z = start_z
        while z < max_z:
            dx = x - cx
            dz = z - cz
            if dx * dx + dz * dz <= r_sq:
                points.append((x, z))
            z += grid_spacing
        x += grid_spacing
    return points


def _point_in_any_polygon(
    px: float, pz: float, polygons: list[list[tuple[float, float]]]
) -> bool:
    """Test if a point is inside any of the given polygon rings."""
    for poly in polygons:
        if _point_in_polygon(px, pz, poly):
            return True
    return False


def _point_near_any_polygon(
    px: float,
    pz: float,
    polygons: list[list[tuple[float, float]]],
    max_dist: float,
) -> bool:
    """True if point is inside or within max_dist of any polygon ring."""
    max_dist_sq = max_dist * max_dist
    for poly in polygons:
        if _point_in_polygon(px, pz, poly):
            return True
        n = len(poly)
        for i in range(n):
            x1, z1 = poly[i]
            x2, z2 = poly[(i + 1) % n]
            if (
                point_to_segment_distance_squared(px, pz, x1, z1, x2, z2)
                <= max_dist_sq
            ):
                return True
    return False


def _fraction_of_dz_visible(
    vis_poly: list[tuple[float, float]],
    dz_sample_points: list[tuple[float, float]],
) -> float:
    """Compute fraction of DZ sample points visible (inside vis polygon)."""
    if not dz_sample_points:
        return 0.0
    if len(vis_poly) < 3:
        return 0.0
    count = 0
    for px, pz in dz_sample_points:
        if _point_in_polygon(px, pz, vis_poly):
            count += 1
    return count / len(dz_sample_points)


def _vectorized_pip_mask(
    pts_x: np.ndarray,
    pts_z: np.ndarray,
    polygon: list[tuple[float, float]],
) -> np.ndarray:
    """Vectorized ray-casting PIP: test all points against one polygon.

    Returns boolean array — True where point is inside polygon.
    """
    n = len(polygon)
    inside = np.zeros(len(pts_x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        crosses = (zi > pts_z) != (zj > pts_z)
        if np.any(crosses):
            dz_edge = zj - zi
            intersect_x = np.where(
                crosses,
                (xj - xi) * (pts_z - zi) / np.where(crosses, dz_edge, 1.0)
                + xi,
                0.0,
            )
            inside ^= crosses & (pts_x < intersect_x)
        j = i
    return inside


# Key for identifying a placed feature's tall footprint contribution.
# (feature_type_id, x, z, rotation_deg) — unique because positions are quantized.
_FeatureKey = tuple[str, float, float, float]


class VisibilityCache:
    """Incrementally maintained cache of which observer grid points are
    blocked by tall terrain.

    The sample grid is constant for a given table size + mission.
    The blocked mask is updated incrementally: when one feature is
    added/moved/deleted, only that feature's contribution is recomputed
    instead of re-testing all ~600 points against all footprints.
    """

    def __init__(
        self,
        layout: TerrainLayout,
        objects_by_id: dict[str, TerrainObject],
    ) -> None:
        self._objects_by_id = objects_by_id
        self._rotationally_symmetric = layout.rotationally_symmetric

        # Precompute sample grid (constant for table size + mission)
        self._all_sample_points = self._build_sample_grid(layout)
        self._pts_x = np.array([p[0] for p in self._all_sample_points])
        self._pts_z = np.array([p[1] for p in self._all_sample_points])

        # Per-feature tracking: feature_key -> bool mask of blocked points
        self._feature_masks: dict[_FeatureKey, np.ndarray] = {}
        # Count of tall features covering each grid point
        self._blocked_count = np.zeros(
            len(self._all_sample_points), dtype=np.int32
        )

        # Initialize from current layout
        self._sync(layout)

    @staticmethod
    def _build_sample_grid(
        layout: TerrainLayout,
    ) -> list[tuple[float, float]]:
        """Generate the observer sample grid (same logic as compute_layout_visibility)."""
        half_w = layout.table_width / 2.0
        half_d = layout.table_depth / 2.0

        obj_ranges: list[tuple[float, float, float]] = []
        if layout.mission is not None:
            for obj_marker in layout.mission.objectives:
                obj_ranges.append(
                    (
                        obj_marker.position.x,
                        obj_marker.position.z,
                        0.75 + obj_marker.range_inches,
                    )
                )

        sample_points: list[tuple[float, float]] = []
        ix_start = int(-half_w) + 1
        ix_end = int(half_w) - 1
        iz_start = int(-half_d) + 1
        iz_end = int(half_d) - 1

        for ix in range(ix_start, ix_end + 1):
            for iz in range(iz_start, iz_end + 1):
                if ix % 2 == 0 and iz % 2 == 0:
                    sample_points.append((float(ix), float(iz)))
                else:
                    fx = float(ix)
                    fz = float(iz)
                    near_obj = False
                    for ox, oz, radius in obj_ranges:
                        dx = fx - ox
                        dz = fz - oz
                        if dx * dx + dz * dz <= radius * radius:
                            near_obj = True
                            break
                    if near_obj:
                        sample_points.append((fx, fz))

        return sample_points

    @staticmethod
    def _feature_key(
        pf: PlacedFeature, is_mirror: bool = False
    ) -> _FeatureKey:
        prefix = "m:" if is_mirror else ""
        return (
            prefix + pf.feature.id,
            pf.transform.x,
            pf.transform.z,
            pf.transform.rotation_deg,
        )

    def _compute_feature_mask(self, pf: PlacedFeature) -> np.ndarray:
        """Compute which sample points are inside this feature's tall OBBs."""
        mask = np.zeros(len(self._all_sample_points), dtype=bool)
        for corners in get_tall_world_obbs(
            pf, self._objects_by_id, min_height=1.0
        ):
            mask |= _vectorized_pip_mask(self._pts_x, self._pts_z, corners)
        return mask

    def _effective_features(
        self, layout: TerrainLayout
    ) -> list[tuple[PlacedFeature, bool]]:
        """Build effective features list with mirror flags."""
        result: list[tuple[PlacedFeature, bool]] = []
        for pf in layout.placed_features:
            result.append((pf, False))
            if self._rotationally_symmetric and not _is_at_origin(pf):
                result.append((_mirror_placed_feature(pf), True))
        return result

    def _sync(self, layout: TerrainLayout) -> None:
        """Sync cache to current layout state via incremental diff."""
        # Build current feature keys
        current_keys: dict[_FeatureKey, PlacedFeature] = {}
        for pf, is_mirror in self._effective_features(layout):
            key = self._feature_key(pf, is_mirror)
            current_keys[key] = pf

        new_key_set = set(current_keys.keys())
        old_key_set = set(self._feature_masks.keys())

        # Remove departed features
        for key in old_key_set - new_key_set:
            self._blocked_count -= self._feature_masks[key]
            del self._feature_masks[key]

        # Add new features
        for key in new_key_set - old_key_set:
            mask = self._compute_feature_mask(current_keys[key])
            if np.any(mask):
                self._feature_masks[key] = mask
                self._blocked_count += mask

    def get_filtered_sample_points(
        self, layout: TerrainLayout
    ) -> list[tuple[float, float]]:
        """Return sample points not inside any tall terrain (incrementally cached)."""
        self._sync(layout)
        if not self._feature_masks:
            return list(self._all_sample_points)
        unblocked = self._blocked_count == 0
        return [
            self._all_sample_points[i]
            for i in range(len(self._all_sample_points))
            if unblocked[i]
        ]


def _has_valid_hiding_square(
    obj_cx: float,
    obj_cz: float,
    obj_radius: float,
    sample_points: list[tuple[float, float]],
    hidden_indices: set[int],
    tall_obbs: list[list[tuple[float, float]]],
    half_w: float,
    half_d: float,
) -> bool:
    """Check if a 1x1" square of hidden grid points exists for model placement.

    A model needs physical space to stand, so a single hidden point is not
    enough. We look for a 1"x1" axis-aligned square (4 adjacent grid points
    at 1" spacing) where:
      1. All 4 corners are within table bounds
      2. At least 1 corner is within the objective's range circle
      3. All 4 corners are hidden from the opposing threat zone
      4. No tall terrain shape (height >= 1.0") intersects the square

    WARNING: The terrain-intersection check (step 4) uses OBB (oriented
    bounding box) geometry via obbs_overlap(). This assumes all terrain
    shapes are rectangular. If polygonal or cylindrical terrain shapes are
    added in the future, this check will need to be updated to use the
    appropriate intersection test.
    """
    # Build lookup from (x, z) -> sample index for O(1) access
    pt_index: dict[tuple[float, float], int] = {}
    for i, (px, pz) in enumerate(sample_points):
        pt_index[(px, pz)] = i

    r_sq = obj_radius * obj_radius

    # Collect candidate square origins (bottom-left corners).
    # For each hidden, in-range point, it could be any of the 4 corners
    # of a valid square. Deduplicate via set of (origin_x, origin_z).
    candidate_origins: set[tuple[float, float]] = set()
    for idx in hidden_indices:
        px, pz = sample_points[idx]
        dx = px - obj_cx
        dz = pz - obj_cz
        if dx * dx + dz * dz > r_sq:
            continue
        # This point is hidden AND in range — generate 4 candidate origins
        # where this point would be each corner of a 1x1 square
        for ox, oz in (
            (px, pz),  # point is bottom-left
            (px - 1.0, pz),  # point is bottom-right
            (px, pz - 1.0),  # point is top-left
            (px - 1.0, pz - 1.0),  # point is top-right
        ):
            candidate_origins.add((ox, oz))

    for ox, oz in candidate_origins:
        # The 4 corners of this candidate square
        corners_xz = [
            (ox, oz),
            (ox + 1.0, oz),
            (ox + 1.0, oz + 1.0),
            (ox, oz + 1.0),
        ]

        # Check 1: all 4 corners within table bounds
        all_in_bounds = True
        for cx, cz in corners_xz:
            if cx < -half_w or cx > half_w or cz < -half_d or cz > half_d:
                all_in_bounds = False
                break
        if not all_in_bounds:
            continue

        # Check 2: at least 1 corner in range of objective
        any_in_range = False
        for cx, cz in corners_xz:
            dx = cx - obj_cx
            dz = cz - obj_cz
            if dx * dx + dz * dz <= r_sq:
                any_in_range = True
                break
        if not any_in_range:
            continue

        # Check 3: all 4 corners are hidden (exist in sample and are hidden)
        all_hidden = True
        for cx, cz in corners_xz:
            idx = pt_index.get((cx, cz))
            if idx is None or idx not in hidden_indices:
                all_hidden = False
                break
        if not all_hidden:
            continue

        # Check 4: no tall terrain OBB overlaps the square
        square_obb = obb_corners(ox + 0.5, oz + 0.5, 0.5, 0.5, 0.0)
        terrain_blocks = False
        for tall_obb in tall_obbs:
            if obbs_overlap(square_obb, tall_obb):
                terrain_blocks = True
                break
        if terrain_blocks:
            continue

        # All checks passed — valid hiding square found
        return True

    return False


def _has_intermediate_shapes(
    layout: TerrainLayout,
    objects_by_id: dict[str, TerrainObject],
    infantry_height: float,
    standard_height: float,
) -> bool:
    """Check if any placed feature has shapes with effective opacity height
    in [infantry_height, standard_height). If not, the infantry pass would
    produce identical results to standard — skip it.
    """
    for pf in layout.placed_features:
        if pf.feature.feature_type == "obscuring":
            continue
        for comp in pf.feature.components:
            obj = objects_by_id.get(comp.object_id)
            if obj is None:
                continue
            for shape in obj.shapes:
                h = shape.effective_opacity_height()
                if h >= infantry_height and h < standard_height:
                    return True
    return False


def _merge_dual_pass_results(
    standard: dict,
    infantry: dict,
    standard_height: float,
    infantry_height: float,
) -> dict:
    """Merge standard and infantry visibility results into a dual-pass result.

    For each section, averages the "value" keys and includes both
    pass breakdowns as "standard" and "infantry" sub-dicts.
    """

    def _merge_entry(std_entry: dict, inf_entry: dict) -> dict:
        """Merge two metric entries: average values, include sub-dicts."""
        merged = {
            "value": round((std_entry["value"] + inf_entry["value"]) / 2.0, 2),
            "standard": std_entry,
            "infantry": inf_entry,
        }
        return merged

    result: dict = {}

    # Overall
    result["overall"] = _merge_entry(standard["overall"], infantry["overall"])

    # DZ hideability (per-DZ entries)
    if "dz_hideability" in standard and "dz_hideability" in infantry:
        merged_dz: dict = {}
        for key in standard["dz_hideability"]:
            if key in infantry["dz_hideability"]:
                merged_dz[key] = _merge_entry(
                    standard["dz_hideability"][key],
                    infantry["dz_hideability"][key],
                )
        result["dz_hideability"] = merged_dz

    # Objective hidability (per-DZ entries)
    if (
        "objective_hidability" in standard
        and "objective_hidability" in infantry
    ):
        merged_obj: dict = {}
        for key in standard["objective_hidability"]:
            if key in infantry["objective_hidability"]:
                merged_obj[key] = _merge_entry(
                    standard["objective_hidability"][key],
                    infantry["objective_hidability"][key],
                )
        result["objective_hidability"] = merged_obj

    return result


def compute_layout_visibility(
    layout: TerrainLayout,
    objects_by_id: dict[str, TerrainObject],
    min_blocking_height: float = 4.0,
    visibility_cache: VisibilityCache | None = None,
    infantry_blocking_height: float | None = None,
    overall_only: bool = False,
) -> dict:
    """Compute visibility score for a terrain layout.

    Samples observer positions on a grid across the table, computes
    visibility polygon for each, and returns the average visibility
    ratio (visible area / total area).

    Observer grid uses integer coordinates, skips table edges, uses 2" spacing
    (even coordinates only) except near objectives where 1" spacing is used.

    Returns dict with format:
    {
        "overall": {
            "value": 72.53,
            "min_blocking_height_inches": 4.0,
            "sample_count": 2640
        }
    }
    """
    half_w = layout.table_width / 2.0
    half_d = layout.table_depth / 2.0
    table_area = layout.table_width * layout.table_depth

    if visibility_cache is not None:
        # Use incremental cache: grid precomputed, blocked mask updated via diff
        sample_points = visibility_cache.get_filtered_sample_points(layout)
    else:
        # No cache: compute grid and filter from scratch
        obj_ranges: list[tuple[float, float, float]] = []
        if layout.mission is not None:
            for obj_marker in layout.mission.objectives:
                obj_ranges.append(
                    (
                        obj_marker.position.x,
                        obj_marker.position.z,
                        0.75 + obj_marker.range_inches,
                    )
                )

        sample_points: list[tuple[float, float]] = []
        ix_start = int(-half_w) + 1
        ix_end = int(half_w) - 1
        iz_start = int(-half_d) + 1
        iz_end = int(half_d) - 1

        for ix in range(ix_start, ix_end + 1):
            for iz in range(iz_start, iz_end + 1):
                if ix % 2 == 0 and iz % 2 == 0:
                    sample_points.append((float(ix), float(iz)))
                else:
                    fx = float(ix)
                    fz = float(iz)
                    near_obj = False
                    for ox, oz, radius in obj_ranges:
                        dx = fx - ox
                        dz = fz - oz
                        if dx * dx + dz * dz <= radius * radius:
                            near_obj = True
                            break
                    if near_obj:
                        sample_points.append((fx, fz))

        tall_footprints: list[list[tuple[float, float]]] = []
        effective_features: list[PlacedFeature] = []
        for pf in layout.placed_features:
            effective_features.append(pf)
            if layout.rotationally_symmetric and not _is_at_origin(pf):
                effective_features.append(_mirror_placed_feature(pf))
        for pf in effective_features:
            for corners in get_tall_world_obbs(
                pf, objects_by_id, min_height=1.0
            ):
                tall_footprints.append(corners)

        if tall_footprints:
            pts_x = np.array([p[0] for p in sample_points])
            pts_z = np.array([p[1] for p in sample_points])
            inside_any = np.zeros(len(sample_points), dtype=bool)
            for fp in tall_footprints:
                inside_any |= _vectorized_pip_mask(pts_x, pts_z, fp)
            mask = ~inside_any
            sample_points = list(
                zip(pts_x[mask].tolist(), pts_z[mask].tolist())
            )

    if not sample_points:
        return {
            "overall": {
                "value": 100.0,
                "min_blocking_height_inches": min_blocking_height,
                "sample_count": 0,
            }
        }

    # -- DZ pre-loop setup --
    # When overall_only=True (e.g. during scoring), skip DZ/objective work
    # to avoid the expensive per-observer polygon intersection tests.
    has_dzs = (
        not overall_only
        and layout.mission is not None
        and len(layout.mission.deployment_zones) > 0
    )
    # (dz_id, polygon_tuples, expanded_polygons)
    dz_data: list[
        tuple[
            str,
            list[list[tuple[float, float]]],
            list[list[tuple[float, float]]],
        ]
    ] = []

    # DZ hideability accumulators: {dz_id: [hidden_count, total_count]}
    dz_hide_accum: dict[str, list[int]] = {}

    if has_dzs:
        mission = layout.mission
        dzs = mission.deployment_zones

        for dz in dzs:
            polys_tuples = [[(p.x, p.z) for p in poly] for poly in dz.polygons]
            expanded = _expand_dz_polygons(polys_tuples, DZ_EXPANSION_INCHES)
            dz_data.append((dz.id, polys_tuples, expanded))
            dz_hide_accum[dz.id] = [0, 0]  # [hidden_count, total_count]

    # -- Objective hidability pre-loop setup --
    has_objectives = (
        has_dzs
        and layout.mission is not None
        and len(layout.mission.objectives) > 0
    )
    # Per-objective sample points
    obj_sample_points: list[list[tuple[float, float]]] = []
    # Original radii for each objective (used in model-fit check)
    obj_radii: list[float] = []
    # {dz_id: [bool mask per objective]} tracking seen sample indices
    obj_seen_from_dz: dict[str, list[np.ndarray]] = {}
    # Numpy arrays for vectorized PIP on objective sample points
    obj_np_x: list[np.ndarray] = []
    obj_np_z: list[np.ndarray] = []

    if has_objectives:
        mission = layout.mission
        for obj_marker in mission.objectives:
            obj_radius = 0.75 + obj_marker.range_inches
            obj_radii.append(obj_radius)
            # Expand sample radius by sqrt(2) so that all diagonal grid
            # neighbors of in-range points are included. This is needed
            # for the model-fit hiding square check which looks at 1x1"
            # squares of 4 adjacent grid points.
            expanded_radius = obj_radius + math.sqrt(2)
            pts = _sample_points_in_circle(
                obj_marker.position.x,
                obj_marker.position.z,
                expanded_radius,
                1.0,
                0.0,
            )
            obj_sample_points.append(pts)
            obj_np_x.append(np.array([p[0] for p in pts]))
            obj_np_z.append(np.array([p[1] for p in pts]))

        for dz in mission.deployment_zones:
            obj_seen_from_dz[dz.id] = [
                np.zeros(len(pts), dtype=bool) for pts in obj_sample_points
            ]

    # Precompute observer-independent segment data once
    precomputed = _precompute_segments(
        layout, objects_by_id, min_blocking_height
    )

    total_ratio = 0.0

    for sx, sz in sample_points:
        segments = _get_observer_segments(precomputed, sx, sz)

        if not segments:
            total_ratio += 1.0
            # No blocking segments = full visibility. For DZ hideability,
            # an observer with full visibility is NOT hidden (vis poly = table).
            if has_dzs:
                for dz_id, dz_polys, expanded_polys in dz_data:
                    if _point_in_any_polygon(sx, sz, dz_polys):
                        # Observer inside this DZ: check vs opponent expanded DZ
                        dz_hide_accum[dz_id][1] += 1
                        # Full visibility -> overlaps opponent -> not hidden
                    if has_objectives:
                        for other_id, _, other_exp in dz_data:
                            if other_id != dz_id:
                                if _point_in_any_polygon(sx, sz, dz_polys):
                                    for oi in range(len(obj_sample_points)):
                                        obj_seen_from_dz[dz_id][oi][:] = True
            continue

        vis_poly = _compute_visibility_polygon(
            sx, sz, segments, half_w, half_d
        )
        if len(vis_poly) < 3:
            total_ratio += 1.0
            if has_dzs:
                for dz_id, dz_polys, expanded_polys in dz_data:
                    if _point_in_any_polygon(sx, sz, dz_polys):
                        dz_hide_accum[dz_id][1] += 1
                    if has_objectives:
                        for other_id, _, other_exp in dz_data:
                            if other_id != dz_id:
                                if _point_in_any_polygon(sx, sz, dz_polys):
                                    for oi in range(len(obj_sample_points)):
                                        obj_seen_from_dz[dz_id][oi][:] = True
            continue

        vis_area = _polygon_area(vis_poly)
        ratio = min(vis_area / table_area, 1.0)
        total_ratio += ratio

        # DZ hideability + objective hidability (polygon intersection)
        if has_dzs:
            for dz_id, dz_polys, expanded_polys in dz_data:
                if not _point_in_any_polygon(sx, sz, dz_polys):
                    continue
                # This observer is inside DZ dz_id.
                # Find opponent expanded DZ polygons and test overlap.
                for other_id, _, other_exp in dz_data:
                    if other_id == dz_id:
                        continue
                    # DZ hideability: does this observer's vis poly
                    # overlap the opponent's expanded DZ?
                    dz_hide_accum[dz_id][1] += 1
                    overlaps = any(
                        polygons_overlap(vis_poly, ep) for ep in other_exp
                    )
                    if not overlaps:
                        dz_hide_accum[dz_id][0] += 1

                # Objective hidability: for each objective, test if vis poly
                # overlaps each opposing expanded DZ
                if has_objectives:
                    for oi in range(len(obj_sample_points)):
                        obj_x_arr = obj_np_x[oi]
                        obj_z_arr = obj_np_z[oi]
                        # Mark which objective sample points this observer can see
                        mask = _vectorized_pip_mask(
                            obj_x_arr, obj_z_arr, vis_poly
                        )
                        obj_seen_from_dz[dz_id][oi] |= mask

    avg_visibility = total_ratio / len(sample_points)
    value = round(avg_visibility * 100.0, 2)

    result: dict = {
        "overall": {
            "value": value,
            "min_blocking_height_inches": min_blocking_height,
            "sample_count": len(sample_points),
        }
    }

    # Build DZ hideability results
    if has_dzs:
        mission = layout.mission
        dzs = mission.deployment_zones

        dz_hideability: dict = {}

        for dz_id, accum in dz_hide_accum.items():
            hidden_count, total_count = accum
            pct = (
                round(hidden_count / total_count * 100.0, 2)
                if total_count > 0
                else 0.0
            )
            dz_hideability[dz_id] = {
                "value": pct,
                "hidden_count": hidden_count,
                "total_count": total_count,
            }

        result["dz_hideability"] = dz_hideability

        # Build objective hidability results
        if has_objectives:
            mission = layout.mission
            dzs = mission.deployment_zones
            objective_hidability: dict = {}

            # Collect tall OBBs for terrain-intersection check
            obj_tall_obbs: list[list[tuple[float, float]]] = []
            effective_features_obj: list[PlacedFeature] = []
            for pf in layout.placed_features:
                effective_features_obj.append(pf)
                if layout.rotationally_symmetric and not _is_at_origin(pf):
                    effective_features_obj.append(_mirror_placed_feature(pf))
            for pf in effective_features_obj:
                for corners in get_tall_world_obbs(
                    pf, objects_by_id, min_height=1.0
                ):
                    obj_tall_obbs.append(corners)

            for dz in dzs:
                # Objectives "safe" for the OPPOSING player means
                # objectives where a model can physically hide (1x1" square
                # of hidden grid points) from this DZ's threat zone
                total_objectives = len(mission.objectives)
                safe_count = 0
                for oi in range(total_objectives):
                    total_pts = len(obj_sample_points[oi])
                    if total_pts == 0:
                        continue
                    # Build set of hidden indices (complement of seen)
                    seen_mask = obj_seen_from_dz[dz.id][oi]
                    hidden = set(np.where(~seen_mask)[0].tolist())
                    if not hidden:
                        continue
                    obj_marker = mission.objectives[oi]
                    if _has_valid_hiding_square(
                        obj_marker.position.x,
                        obj_marker.position.z,
                        obj_radii[oi],
                        obj_sample_points[oi],
                        hidden,
                        obj_tall_obbs,
                        half_w,
                        half_d,
                    ):
                        safe_count += 1

                # This DZ's threat data: the opposing player can hide at
                # safe_count objectives. Label by the opposing player's color.
                # Convention: green.value = % objectives green can hide at
                # = objectives with hiding spots from red's perspective
                # So we store under the OTHER DZ's id.
                for other_dz in dzs:
                    if other_dz.id != dz.id:
                        pct = (
                            round(safe_count / total_objectives * 100.0, 2)
                            if total_objectives > 0
                            else 0.0
                        )
                        objective_hidability[other_dz.id] = {
                            "value": pct,
                            "safe_count": safe_count,
                            "total_objectives": total_objectives,
                        }

            result["objective_hidability"] = objective_hidability

    # Dual-pass infantry visibility: if enabled and intermediate shapes exist,
    # compute a second pass at the infantry blocking height and merge.
    if infantry_blocking_height is not None and _has_intermediate_shapes(
        layout, objects_by_id, infantry_blocking_height, min_blocking_height
    ):
        infantry_result = compute_layout_visibility(
            layout,
            objects_by_id,
            min_blocking_height=infantry_blocking_height,
            visibility_cache=visibility_cache,
            infantry_blocking_height=None,  # prevent recursion
            overall_only=overall_only,
        )
        return _merge_dual_pass_results(
            result,
            infantry_result,
            min_blocking_height,
            infantry_blocking_height,
        )

    return result
