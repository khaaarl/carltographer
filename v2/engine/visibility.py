"""Visibility score computation for terrain layouts.

Measures what percentage of the battlefield has clear line of sight
by sampling observer positions on a grid and computing visibility
polygons via angular sweep.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .collision import (
    _is_at_origin,
    _mirror_placed_feature,
    compose_transform,
    get_tall_world_obbs,
    get_world_obbs,
    obb_corners,
)
from .types import (
    DeploymentZone,
    PlacedFeature,
    TerrainLayout,
    TerrainObject,
    Transform,
)

Segment = tuple[float, float, float, float]  # (x1, z1, x2, z2)


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
                    if shape.height < min_blocking_height:
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
    """
    # Table boundary segments
    tw, td = table_half_w, table_half_d
    table_boundary: list[Segment] = [
        (-tw, -td, tw, -td),  # bottom
        (tw, -td, tw, td),  # right
        (tw, td, -tw, td),  # top
        (-tw, td, -tw, -td),  # left
    ]

    # Collect unique endpoints from both segment lists (no copy)
    endpoints: set[tuple[float, float]] = set()
    for x1, z1, x2, z2 in itertools.chain(segments, table_boundary):
        endpoints.add((x1, z1))
        endpoints.add((x2, z2))

    # For each endpoint, cast rays at angle and angle +/- epsilon
    EPS = 1e-5
    rays: list[tuple[float, float, float]] = []  # (angle, dx, dz)
    _atan2 = math.atan2
    _cos = math.cos
    _sin = math.sin

    for ex, ez in endpoints:
        dx = ex - ox
        dz = ez - oz
        angle = _atan2(dz, dx)
        # Epsilon-shifted rays need trig, but direct ray reuses normalized vector
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > 1e-12:
            ndx = dx / dist
            ndz = dz / dist
        else:
            ndx = _cos(angle)
            ndz = _sin(angle)
        a_minus = angle - EPS
        a_plus = angle + EPS
        rays.append((a_minus, _cos(a_minus), _sin(a_minus)))
        rays.append((angle, ndx, ndz))
        rays.append((a_plus, _cos(a_plus), _sin(a_plus)))

    # Sort rays by angle
    rays.sort(key=lambda r: r[0])

    # Cast each ray, find nearest intersection
    polygon: list[tuple[float, float]] = []

    for _angle, dx, dz in rays:
        min_t = float("inf")
        for x1, z1, x2, z2 in itertools.chain(segments, table_boundary):
            # Inlined _ray_segment_intersection to avoid function call overhead
            sx = x2 - x1
            sz = z2 - z1
            denom = dx * sz - dz * sx
            if abs(denom) < 1e-12:
                continue
            t = ((x1 - ox) * sz - (z1 - oz) * sx) / denom
            u = ((x1 - ox) * dz - (z1 - oz) * dx) / denom
            if t >= 0 and 0 <= u <= 1 and t < min_t:
                min_t = t

        if min_t < float("inf"):
            polygon.append((ox + min_t * dx, oz + min_t * dz))

    return polygon


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


def compute_layout_visibility(
    layout: TerrainLayout,
    objects_by_id: dict[str, TerrainObject],
    min_blocking_height: float = 4.0,
    visibility_cache: VisibilityCache | None = None,
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
    has_dzs = (
        layout.mission is not None and len(layout.mission.deployment_zones) > 0
    )
    dz_data: list[
        tuple[str, list[list[tuple[float, float]]], list[tuple[float, float]]]
    ] = []  # (dz_id, polygon_tuples, sample_points)

    # Accumulators for dz_visibility: {dz_id: [total_fraction, observer_count]}
    dz_vis_accum: dict[str, list[float]] = {}
    # Cross-DZ: track which target sample indices have been seen by ANY observer
    # {target_id_from_observer_id: set of seen target sample indices}
    dz_cross_seen: dict[str, set[int]] = {}
    dz_cross_obs_count: dict[str, int] = {}

    if has_dzs:
        mission = layout.mission
        dzs = mission.deployment_zones

        for dz in dzs:
            polys_tuples = [[(p.x, p.z) for p in poly] for poly in dz.polygons]
            dz_samples = _sample_points_in_dz(dz, 1.0, 0.0)
            dz_data.append((dz.id, polys_tuples, dz_samples))
            dz_vis_accum[dz.id] = [0.0, 0]  # [total_fraction, observer_count]
            for other_dz in dzs:
                if other_dz.id != dz.id:
                    key = f"{dz.id}_from_{other_dz.id}"
                    dz_cross_seen[key] = set()
                    dz_cross_obs_count[key] = 0

    # -- Objective hidability pre-loop setup --
    has_objectives = (
        has_dzs
        and layout.mission is not None
        and len(layout.mission.objectives) > 0
    )
    # Per-objective sample points
    obj_sample_points: list[list[tuple[float, float]]] = []
    # {dz_id: {obj_index: set of seen sample indices}}
    obj_seen_from_dz: dict[str, dict[int, set[int]]] = {}

    if has_objectives:
        mission = layout.mission
        for obj_marker in mission.objectives:
            obj_radius = 0.75 + obj_marker.range_inches
            pts = _sample_points_in_circle(
                obj_marker.position.x,
                obj_marker.position.z,
                obj_radius,
                1.0,
                0.0,
            )
            obj_sample_points.append(pts)

        for dz in mission.deployment_zones:
            obj_seen_from_dz[dz.id] = {
                i: set() for i in range(len(mission.objectives))
            }

    # Precompute observer-independent segment data once
    precomputed = _precompute_segments(
        layout, objects_by_id, min_blocking_height
    )

    total_ratio = 0.0

    for sx, sz in sample_points:
        segments = _get_observer_segments(precomputed, sx, sz)

        if not segments:
            total_ratio += 1.0
            if has_dzs:
                # Full visibility: all DZ samples visible
                for dz_id, dz_polys, dz_samples in dz_data:
                    observer_in_dz = _point_in_any_polygon(sx, sz, dz_polys)
                    if not observer_in_dz:
                        dz_vis_accum[dz_id][0] += 1.0
                        dz_vis_accum[dz_id][1] += 1
                    else:
                        # Observer inside this DZ: all target samples seen
                        for (
                            other_id,
                            _other_polys,
                            other_samples,
                        ) in dz_data:
                            if other_id != dz_id:
                                key = f"{other_id}_from_{dz_id}"
                                if key in dz_cross_seen:
                                    dz_cross_obs_count[key] += 1
                                    dz_cross_seen[key].update(
                                        range(len(other_samples))
                                    )
                        # Full vis: all objective samples seen from this DZ
                        if has_objectives:
                            for oi, obj_pts in enumerate(obj_sample_points):
                                obj_seen_from_dz[dz_id][oi].update(
                                    range(len(obj_pts))
                                )
            continue

        vis_poly = _compute_visibility_polygon(
            sx, sz, segments, half_w, half_d
        )
        if len(vis_poly) < 3:
            total_ratio += 1.0
            if has_dzs:
                for dz_id, dz_polys, dz_samples in dz_data:
                    observer_in_dz = _point_in_any_polygon(sx, sz, dz_polys)
                    if not observer_in_dz:
                        dz_vis_accum[dz_id][0] += 1.0
                        dz_vis_accum[dz_id][1] += 1
                    else:
                        for (
                            other_id,
                            _other_polys,
                            other_samples,
                        ) in dz_data:
                            if other_id != dz_id:
                                key = f"{other_id}_from_{dz_id}"
                                if key in dz_cross_seen:
                                    dz_cross_obs_count[key] += 1
                                    dz_cross_seen[key].update(
                                        range(len(other_samples))
                                    )
                        # Full vis: all objective samples seen from this DZ
                        if has_objectives:
                            for oi, obj_pts in enumerate(obj_sample_points):
                                obj_seen_from_dz[dz_id][oi].update(
                                    range(len(obj_pts))
                                )
            continue

        vis_area = _polygon_area(vis_poly)
        ratio = min(vis_area / table_area, 1.0)
        total_ratio += ratio

        # DZ visibility accumulation
        if has_dzs:
            for dz_id, dz_polys, dz_samples in dz_data:
                observer_in_dz = _point_in_any_polygon(sx, sz, dz_polys)
                if not observer_in_dz:
                    # Observer outside this DZ: contributes to dz_visibility
                    frac = _fraction_of_dz_visible(vis_poly, dz_samples)
                    dz_vis_accum[dz_id][0] += frac
                    dz_vis_accum[dz_id][1] += 1
                else:
                    # Observer inside this DZ: mark visible target samples
                    for (
                        other_id,
                        _other_polys,
                        other_samples,
                    ) in dz_data:
                        if other_id != dz_id:
                            key = f"{other_id}_from_{dz_id}"
                            if key in dz_cross_seen:
                                dz_cross_obs_count[key] += 1
                                seen = dz_cross_seen[key]
                                for i, (px, pz) in enumerate(other_samples):
                                    if i not in seen:
                                        if _point_in_polygon(px, pz, vis_poly):
                                            seen.add(i)
                    # Objective hidability: mark seen objective samples
                    if has_objectives:
                        for oi, obj_pts in enumerate(obj_sample_points):
                            seen = obj_seen_from_dz[dz_id][oi]
                            for i, (px, pz) in enumerate(obj_pts):
                                if i not in seen:
                                    if _point_in_polygon(px, pz, vis_poly):
                                        seen.add(i)

    avg_visibility = total_ratio / len(sample_points)
    value = round(avg_visibility * 100.0, 2)

    result: dict = {
        "overall": {
            "value": value,
            "min_blocking_height_inches": min_blocking_height,
            "sample_count": len(sample_points),
        }
    }

    # Build DZ visibility results
    if has_dzs:
        mission = layout.mission
        dzs = mission.deployment_zones

        dz_visibility: dict = {}
        dz_to_dz_visibility: dict = {}

        for dz_id, accum in dz_vis_accum.items():
            total_frac, obs_count = accum
            avg = total_frac / obs_count if obs_count > 0 else 0.0
            dz_samples_count = 0
            for did, _dp, ds in dz_data:
                if did == dz_id:
                    dz_samples_count = len(ds)
                    break
            dz_visibility[dz_id] = {
                "value": round(avg * 100.0, 2),
                "dz_sample_count": dz_samples_count,
                "observer_count": obs_count,
            }

        for key, seen_set in dz_cross_seen.items():
            # Extract target_id from key (format: "target_from_observer")
            parts = key.split("_from_")
            target_id = parts[0]
            target_samples_count = 0
            for did, _dp, ds in dz_data:
                if did == target_id:
                    target_samples_count = len(ds)
                    break
            hidden_count = target_samples_count - len(seen_set)
            hidden_pct = (
                round(hidden_count / target_samples_count * 100.0, 2)
                if target_samples_count > 0
                else 0.0
            )
            dz_to_dz_visibility[key] = {
                "value": hidden_pct,
                "hidden_count": hidden_count,
                "target_sample_count": target_samples_count,
                "observer_count": dz_cross_obs_count.get(key, 0),
            }

        result["dz_visibility"] = dz_visibility
        result["dz_to_dz_visibility"] = dz_to_dz_visibility

        # Build objective hidability results
        if has_objectives:
            mission = layout.mission
            dzs = mission.deployment_zones
            objective_hidability: dict = {}

            for dz in dzs:
                # Objectives "safe" for the OPPOSING player means
                # objectives where at least 1 sample was NOT seen from this DZ
                total_objectives = len(mission.objectives)
                safe_count = 0
                for oi in range(total_objectives):
                    total_pts = len(obj_sample_points[oi])
                    if total_pts == 0:
                        continue
                    seen_count = len(obj_seen_from_dz[dz.id][oi])
                    if seen_count < total_pts:
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

    return result
