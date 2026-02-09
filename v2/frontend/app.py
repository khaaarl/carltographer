"""Carltographer terrain layout viewer.

Displays a 2D top-down view of a Warhammer 40k terrain layout
with a control panel for engine parameters.
"""

import copy
import json
import math
import os
import random
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from ..engine import generate_json
from ..engine.collision import is_valid_placement
from ..engine.types import PlacedFeature as TypedPlacedFeature
from ..engine.types import TerrainObject as TypedTerrainObject
from ..engine_cmp.hash_manifest import verify_engine_unchanged
from .layout_io import load_layout, save_layout_png
from .missions import EDITIONS, find_mission_path

try:
    import engine_rs as _engine_rs

    _HAS_RUST_ENGINE = True
except ImportError:
    _engine_rs = None  # type: ignore
    _HAS_RUST_ENGINE = False


def _should_use_rust_engine() -> bool:
    """Determine which engine to use based on manifest verification.

    Returns True if:
    - Rust engine is available (built with maturin)
    - AND Python engine source matches the certified hashes

    Returns False if:
    - Rust engine is not available, OR
    - Python engine source has changed since last certification
    """
    if not _HAS_RUST_ENGINE:
        return False
    return verify_engine_unchanged()


# -- Visual constants --

TABLE_BG = "#2d5a27"  # dark green gaming mat
TABLE_GRID = "#264e22"  # subtle darker grid
TABLE_BORDER = "#111111"
CANVAS_BG = "#1e1e1e"
DEFAULT_FILL = "#888888"
HIGHLIGHT_COLOR = "#FFD700"  # gold highlight for selected features
MOVE_VALID_OUTLINE = "#00FF00"  # green outline during valid move
MOVE_INVALID_OUTLINE = "#FF4444"  # red outline during invalid move/copy

# Mission rendering colors
NO_MANS_LAND_BG = "#d4c5a0"  # cream/tan for no-man's land
NO_MANS_LAND_GRID = "#c4b590"  # slightly darker grid for NML

DZ_COLORS = {
    "red": {"bg": "#5c2a2a", "grid": "#4c2020"},
    "green": {"bg": "#2a4a2a", "grid": "#204020"},
}

OBJECTIVE_RADIUS_INCHES = 0.75  # 1.5" diameter marker
OBJECTIVE_FILL = "#111111"
OBJECTIVE_OUTLINE = "#000000"
RANGE_DASH_COLOR = "#000000"

# ---------------------------------------------------------------------------
# Sample data (structured per carltographer.schema.json)
# ---------------------------------------------------------------------------

CRATE_OBJECT = {
    "id": "crate",
    "name": "Crate (double-stack)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 5.0,
            "depth_inches": 2.5,
            "height_inches": 5.0,
        }
    ],
    "tags": ["container"],
    "fill_color": "#8b3a1a",
    "outline_color": "#000000",
}

CRATE_FEATURE = {
    "id": "crate",
    "feature_type": "obstacle",
    "components": [{"object_id": "crate"}],
}

RUINS_OBJECT = {
    "id": "ruins",
    "name": "Ruins",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 12.0,
            "depth_inches": 6.0,
            "height_inches": 0.0,
        }
    ],
    "tags": ["ruins"],
    "fill_color": "#888888",
    "outline_color": "#000000",
}

WTC_THREE_STOREY_WALLS = {
    "id": "wtc_three_storey_walls",
    "name": "WTC Three Storey Walls",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 9.0,
            "depth_inches": 0.1,
            "height_inches": 9.0,
            "offset": {"x_inches": -0.65, "z_inches": -2.15},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.1,
            "depth_inches": 5.0,
            "height_inches": 9.0,
            "offset": {"x_inches": -5.15, "z_inches": 0.35},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.6,
            "depth_inches": 0.1,
            "height_inches": 3.0,
            "offset": {"x_inches": -5.5, "z_inches": -2.15},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.1,
            "depth_inches": 0.6,
            "height_inches": 3.0,
            "offset": {"x_inches": -5.15, "z_inches": -2.5},
        },
    ],
    "tags": ["ruins", "wtc"],
    "fill_color": "#111111",
    "outline_color": "#000000",
}

WTC_THREE_STOREY_FEATURE = {
    "id": "wtc_three_storey",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "ruins"},
        {"object_id": "wtc_three_storey_walls"},
    ],
}

SAMPLE_CATALOG = {
    "name": "Sample terrain",
    "objects": [
        {"item": CRATE_OBJECT, "quantity": None},
        {"item": RUINS_OBJECT, "quantity": None},
        {"item": WTC_THREE_STOREY_WALLS, "quantity": None},
    ],
    "features": [
        {"item": CRATE_FEATURE, "quantity": None},
        {"item": WTC_THREE_STOREY_FEATURE, "quantity": None},
    ],
}


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _mirror_pf_dict(pf):
    """Create mirror dict for rendering."""
    tf = pf.get("transform", {})
    mirror_tf = {
        "x_inches": -tf.get("x_inches", 0.0),
        "y_inches": tf.get("y_inches", 0.0),
        "z_inches": -tf.get("z_inches", 0.0),
        "rotation_deg": tf.get("rotation_deg", 0.0) + 180.0,
    }
    return {**pf, "transform": mirror_tf}


def _get_tf(d):
    """Extract (x, z, rot_deg) from a transform dict, defaulting to 0."""
    if not d:
        return 0.0, 0.0, 0.0
    return (
        d.get("x_inches", 0.0),
        d.get("z_inches", 0.0),
        d.get("rotation_deg", 0.0),
    )


def _compose(outer, inner):
    """Compose two (x, z, rot_deg) transforms (inner applied first)."""
    ox, oz, orot = outer
    ix, iz, irot = inner
    rad = math.radians(orot)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    x = ox + ix * cos_r - iz * sin_r
    z = oz + ix * sin_r + iz * cos_r
    return (x, z, orot + irot)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class BattlefieldRenderer:
    """Renders a terrain layout to a Pillow image."""

    def __init__(
        self, table_width, table_depth, ppi, objects_by_id, mission=None
    ):
        self.table_width = table_width
        self.table_depth = table_depth
        self.ppi = ppi
        self.objects_by_id = objects_by_id
        self.mission = mission

    def _to_px(self, x_inches, z_inches):
        """Table coords (center origin) -> pixel coords (top-left origin)."""
        px = (x_inches + self.table_width / 2) * self.ppi
        py = (z_inches + self.table_depth / 2) * self.ppi
        return px, py

    def _build_dz_polys(self):
        """Build list of (polygon_points, color_id) for all DZ polygons.

        Each polygon_points is a list of (x, z) tuples in table inches.
        """
        if not self.mission:
            return []
        polys = []
        for dz in self.mission.get("deployment_zones", []):
            color_id = dz["id"]
            for poly in dz.get("polygons", []):
                pts = [(p["x_inches"], p["z_inches"]) for p in poly]
                polys.append((pts, color_id))
        return polys

    @staticmethod
    def _point_in_polygon(x, z, poly):
        """Ray-casting point-in-polygon test. poly is list of (x,z) tuples."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, zi = poly[i]
            xj, zj = poly[j]
            if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (
                zj - zi
            ) + xi:
                inside = not inside
            j = i
        return inside

    def _get_zone_color_at(self, x_inches, z_inches, dz_polys):
        """Return zone color_id at a table-space point, or None for NML."""
        for poly, color_id in dz_polys:
            if self._point_in_polygon(x_inches, z_inches, poly):
                return color_id
        return None

    @staticmethod
    def _line_polygon_intersections_z(x_val, poly):
        """Find all z values where the vertical line x=x_val intersects polygon edges."""
        zs = []
        n = len(poly)
        for i in range(n):
            x1, z1 = poly[i]
            x2, z2 = poly[(i + 1) % n]
            if x1 == x2:
                continue  # edge is vertical, parallel to our line
            if (x_val - x1) * (x_val - x2) > 0:
                continue  # x_val not between x1 and x2
            t = (x_val - x1) / (x2 - x1)
            if 0.0 <= t <= 1.0:
                zs.append(z1 + t * (z2 - z1))
        return zs

    @staticmethod
    def _line_polygon_intersections_x(z_val, poly):
        """Find all x values where the horizontal line z=z_val intersects polygon edges."""
        xs = []
        n = len(poly)
        for i in range(n):
            x1, z1 = poly[i]
            x2, z2 = poly[(i + 1) % n]
            if z1 == z2:
                continue  # edge is horizontal, parallel to our line
            if (z_val - z1) * (z_val - z2) > 0:
                continue  # z_val not between z1 and z2
            t = (z_val - z1) / (z2 - z1)
            if 0.0 <= t <= 1.0:
                xs.append(x1 + t * (x2 - x1))
        return xs

    def render(self, layout, highlight_index=None):
        w = int(self.table_width * self.ppi)
        h = int(self.table_depth * self.ppi)
        has_mission = self.mission is not None

        # 1. Background
        bg = NO_MANS_LAND_BG
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)

        # 2. DZ backgrounds
        dz_polys = []
        mission = self.mission
        if mission is not None:
            dz_polys = self._build_dz_polys()
            for dz in mission.get("deployment_zones", []):
                color_id = dz["id"]
                colors = DZ_COLORS.get(color_id, DZ_COLORS["red"])
                for poly in dz.get("polygons", []):
                    px_poly = [
                        self._to_px(p["x_inches"], p["z_inches"]) for p in poly
                    ]
                    draw.polygon(px_poly, fill=colors["bg"])

        # 3. Grid lines (zone-aware if mission)
        if has_mission:
            self._draw_zone_aware_grid(draw, w, h, dz_polys)
        else:
            for ix in range(1, int(self.table_width)):
                px = int(ix * self.ppi)
                draw.line([(px, 0), (px, h - 1)], fill=NO_MANS_LAND_GRID)
            for iz in range(1, int(self.table_depth)):
                py = int(iz * self.ppi)
                draw.line([(0, py), (w - 1, py)], fill=NO_MANS_LAND_GRID)

        # 4. Terrain features
        placed = layout["placed_features"]
        is_symmetric = layout.get("rotationally_symmetric", False)
        for pf in placed:
            self._draw_placed_feature(draw, pf)
            if is_symmetric:
                tf = pf.get("transform", {})
                if (
                    tf.get("x_inches", 0.0) != 0.0
                    or tf.get("z_inches", 0.0) != 0.0
                ):
                    self._draw_placed_feature(draw, _mirror_pf_dict(pf))

        # 4b. Highlight selected feature
        if highlight_index is not None and 0 <= highlight_index < len(placed):
            pf = placed[highlight_index]
            self._draw_placed_feature_highlight(draw, pf)
            if is_symmetric:
                tf = pf.get("transform", {})
                if (
                    tf.get("x_inches", 0.0) != 0.0
                    or tf.get("z_inches", 0.0) != 0.0
                ):
                    self._draw_placed_feature_highlight(
                        draw, _mirror_pf_dict(pf)
                    )

        # 5. Objective markers (on top of terrain)
        if mission is not None:
            for obj in mission.get("objectives", []):
                self._draw_objective(draw, obj)

        # 6. Table border
        draw.rectangle([0, 0, w - 1, h - 1], outline=TABLE_BORDER, width=3)
        return img

    def _draw_zone_aware_grid(self, draw, w, h, dz_polys):
        """Draw grid lines with colors matching the zone they pass through."""
        hw = self.table_width / 2
        hd = self.table_depth / 2

        # Vertical grid lines
        for ix in range(1, int(self.table_width)):
            px = int(ix * self.ppi)
            x_inches = ix - hw
            # Find all z-intersections with polygon edges
            z_breaks = {-hd, hd}
            for poly, _cid in dz_polys:
                for z in self._line_polygon_intersections_z(x_inches, poly):
                    z_breaks.add(z)
            z_breaks_sorted = sorted(z_breaks)
            for i in range(len(z_breaks_sorted) - 1):
                z_mid = (z_breaks_sorted[i] + z_breaks_sorted[i + 1]) / 2
                zone = self._get_zone_color_at(x_inches, z_mid, dz_polys)
                color = self._grid_color_for_zone(zone)
                py0 = int((z_breaks_sorted[i] + hd) * self.ppi)
                py1 = int((z_breaks_sorted[i + 1] + hd) * self.ppi)
                draw.line([(px, py0), (px, py1)], fill=color)

        # Horizontal grid lines
        for iz in range(1, int(self.table_depth)):
            py = int(iz * self.ppi)
            z_inches = iz - hd
            # Find all x-intersections with polygon edges
            x_breaks = {-hw, hw}
            for poly, _cid in dz_polys:
                for x in self._line_polygon_intersections_x(z_inches, poly):
                    x_breaks.add(x)
            x_breaks_sorted = sorted(x_breaks)
            for i in range(len(x_breaks_sorted) - 1):
                x_mid = (x_breaks_sorted[i] + x_breaks_sorted[i + 1]) / 2
                zone = self._get_zone_color_at(x_mid, z_inches, dz_polys)
                color = self._grid_color_for_zone(zone)
                px0 = int((x_breaks_sorted[i] + hw) * self.ppi)
                px1 = int((x_breaks_sorted[i + 1] + hw) * self.ppi)
                draw.line([(px0, py), (px1, py)], fill=color)

    def _grid_color_for_zone(self, zone_color_id):
        """Return the grid color for a given zone (or NML if None)."""
        if zone_color_id is None:
            return NO_MANS_LAND_GRID
        colors = DZ_COLORS.get(zone_color_id, DZ_COLORS["red"])
        return colors["grid"]

    def _draw_objective(self, draw, objective):
        """Draw an objective marker with skull icon and dashed range circle."""
        pos = objective["position"]
        cx, cy = self._to_px(pos["x_inches"], pos["z_inches"])
        r_px = OBJECTIVE_RADIUS_INCHES * self.ppi

        # Solid black circle (the marker)
        draw.ellipse(
            [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
            fill=OBJECTIVE_FILL,
            outline=OBJECTIVE_OUTLINE,
            width=2,
        )

        # White skull icon
        self._draw_skull(draw, cx, cy, r_px)

        # Dashed range circle (range is measured from marker edge, not center)
        range_inches = objective.get("range_inches", 3.0)
        total_radius_px = (OBJECTIVE_RADIUS_INCHES + range_inches) * self.ppi
        self._draw_dashed_circle(draw, cx, cy, total_radius_px)

    def _draw_skull(self, draw, cx, cy, marker_radius):
        """Draw a simple skull icon inside the objective marker."""
        s = marker_radius * 0.6  # scale factor
        white = "#ffffff"

        # Head (circle)
        head_r = s * 0.55
        draw.ellipse(
            [
                cx - head_r,
                cy - s * 0.35 - head_r,
                cx + head_r,
                cy - s * 0.35 + head_r,
            ],
            fill=white,
        )

        # Eyes (two dark dots)
        eye_r = s * 0.12
        eye_y = cy - s * 0.4
        for ex in [cx - s * 0.22, cx + s * 0.22]:
            draw.ellipse(
                [ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r],
                fill=OBJECTIVE_FILL,
            )

        # Jaw / teeth area
        jaw_w = s * 0.35
        jaw_h = s * 0.2
        jaw_y = cy + s * 0.1
        draw.rectangle(
            [cx - jaw_w, jaw_y, cx + jaw_w, jaw_y + jaw_h],
            fill=white,
        )
        # Tooth gaps
        gap = s * 0.12
        for gx in [cx - gap, cx + gap]:
            draw.line(
                [(gx, jaw_y), (gx, jaw_y + jaw_h)],
                fill=OBJECTIVE_FILL,
                width=max(1, int(s * 0.06)),
            )

    def _draw_dashed_circle(self, draw, cx, cy, radius, dash_count=36):
        """Draw a dashed circle as alternating arc segments."""
        arc_angle = 360.0 / dash_count
        for i in range(0, dash_count, 2):
            start = i * arc_angle
            end = start + arc_angle
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.arc(
                bbox, start=start, end=end, fill=RANGE_DASH_COLOR, width=2
            )

    def _draw_placed_feature(self, draw, placed_feature):
        feature = placed_feature["feature"]
        feat_tf = _get_tf(placed_feature.get("transform"))

        for comp in feature["components"]:
            obj = self.objects_by_id.get(comp["object_id"])
            if obj is None:
                continue
            comp_tf = _get_tf(comp.get("transform"))

            fill = obj.get("fill_color") or DEFAULT_FILL
            outline = obj.get("outline_color")

            for shape in obj["shapes"]:
                offset_tf = _get_tf(shape.get("offset"))
                combined = _compose(feat_tf, _compose(comp_tf, offset_tf))
                self._draw_rect(draw, shape, combined, fill, outline)

    def _draw_placed_feature_highlight(self, draw, placed_feature):
        """Draw gold outline highlight for a placed feature (no fill)."""
        feature = placed_feature["feature"]
        feat_tf = _get_tf(placed_feature.get("transform"))

        for comp in feature["components"]:
            obj = self.objects_by_id.get(comp["object_id"])
            if obj is None:
                continue
            comp_tf = _get_tf(comp.get("transform"))

            for shape in obj["shapes"]:
                offset_tf = _get_tf(shape.get("offset"))
                combined = _compose(feat_tf, _compose(comp_tf, offset_tf))
                self._draw_highlight_rect(draw, shape, combined)

    def _draw_highlight_rect(self, draw, shape, tf):
        """Draw a gold outline rectangle (no fill) for highlight."""
        cx, cz, rot_deg = tf
        hw = shape["width_inches"] / 2
        hd = shape["depth_inches"] / 2
        corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]

        rad = math.radians(rot_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)

        px_corners = []
        for lx, lz in corners:
            rx = cx + lx * cos_r - lz * sin_r
            rz = cz + lx * sin_r + lz * cos_r
            px_corners.append(self._to_px(rx, rz))

        draw.polygon(
            px_corners,
            fill=None,
            outline=HIGHLIGHT_COLOR,
            width=3,
        )

    def _draw_rect(self, draw, shape, tf, fill, outline):
        cx, cz, rot_deg = tf
        hw = shape["width_inches"] / 2
        hd = shape["depth_inches"] / 2
        corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]

        rad = math.radians(rot_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)

        px_corners = []
        for lx, lz in corners:
            rx = cx + lx * cos_r - lz * sin_r
            rz = cz + lx * sin_r + lz * cos_r
            px_corners.append(self._to_px(rx, rz))

        draw.polygon(
            px_corners,
            fill=fill,
            outline=outline,
            width=2 if outline else 0,
        )


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------


class ControlPanel(ttk.Frame):
    """Sidebar with engine parameter controls."""

    def __init__(
        self, parent, on_table_changed, on_generate, on_clear, on_save, on_load
    ):
        super().__init__(parent, padding=10)
        self.on_table_changed = on_table_changed
        self.on_generate = on_generate
        self.on_clear = on_clear
        self.on_save = on_save
        self.on_load = on_load

        # -- tk variables --
        self.table_width_var = tk.DoubleVar(value=60.0)
        self.table_depth_var = tk.DoubleVar(value=44.0)
        self.seed_var = tk.StringVar(value="")
        self.num_steps_var = tk.IntVar(value=1)
        self.symmetric_var = tk.BooleanVar(value=False)
        self.min_gap_var = tk.StringVar(value="")
        self.min_edge_gap_var = tk.StringVar(value="")
        self.min_crates_var = tk.StringVar(value="")
        self.max_crates_var = tk.StringVar(value="")
        self.min_ruins_var = tk.StringVar(value="")
        self.max_ruins_var = tk.StringVar(value="")

        # Scoring target variables
        self.overall_vis_target_var = tk.StringVar(value="30")
        self.overall_vis_weight_var = tk.StringVar(value="1.0")
        self.dz_vis_target_var = tk.StringVar(value="20")
        self.dz_vis_weight_var = tk.StringVar(value="1.0")
        self.dz_hidden_target_var = tk.StringVar(value="70")
        self.dz_hidden_weight_var = tk.StringVar(value="1.0")
        self.obj_hide_target_var = tk.StringVar(value="40")
        self.obj_hide_weight_var = tk.StringVar(value="1.0")

        # Mission selection variables
        self.edition_var = tk.StringVar(value="")
        self.pack_var = tk.StringVar(value="")
        self.deployment_var = tk.StringVar(value="None")
        self.selected_mission = None  # current mission dict or None
        self._initializing = True  # suppress callbacks during init

        # Combo widgets (set during _build via _combo helper)
        self.edition_combo: ttk.Combobox = None
        self.pack_combo: ttk.Combobox = None
        self.deployment_combo: ttk.Combobox = None

        self._build()
        self._initializing = False

        # Re-render battlefield when table dims change.
        self.table_width_var.trace_add("write", self._dims_changed)
        self.table_depth_var.trace_add("write", self._dims_changed)

    # -- layout --

    def _build(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.Y)

        # --- Left column: Mission, Table, Generation, Buttons ---
        row = 0
        row = self._section(left, row, "Mission")
        row = self._combo(
            left, row, "Edition:", self.edition_var, "edition_combo"
        )
        row = self._combo(left, row, "Pack:", self.pack_var, "pack_combo")
        row = self._combo(
            left, row, "Deployment:", self.deployment_var, "deployment_combo"
        )

        # Initialize mission dropdowns
        self._init_mission_combos()

        row = self._sep(left, row)
        row = self._section(left, row, "Table")
        row = self._field(left, row, "Width (in):", self.table_width_var)
        row = self._field(left, row, "Depth (in):", self.table_depth_var)

        row = self._sep(left, row)
        row = self._section(left, row, "Generation")
        row = self._field(left, row, "Seed:", self.seed_var)
        row = self._field(left, row, "Steps:", self.num_steps_var)
        ttk.Checkbutton(
            left, text="Rotationally symmetric", variable=self.symmetric_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        row = self._sep(left, row)
        ttk.Button(left, text="Generate", command=self.on_generate).grid(
            row=row, column=0, columnspan=2, pady=(10, 2), sticky="ew"
        )
        row += 1
        ttk.Button(left, text="Clear Layout", command=self.on_clear).grid(
            row=row, column=0, columnspan=2, pady=(2, 2), sticky="ew"
        )
        row += 1
        ttk.Button(left, text="Save Layout", command=self.on_save).grid(
            row=row, column=0, columnspan=2, pady=(2, 2), sticky="ew"
        )
        row += 1
        ttk.Button(left, text="Load Layout", command=self.on_load).grid(
            row=row, column=0, columnspan=2, pady=(2, 10), sticky="ew"
        )

        # --- Right column: Spacing, Feature Counts, Results ---
        row = 0
        row = self._section(right, row, "Spacing")
        row = self._field(right, row, "Feature gap (in):", self.min_gap_var)
        row = self._field(right, row, "Edge gap (in):", self.min_edge_gap_var)

        row = self._sep(right, row)
        row = self._section(right, row, "Feature Counts")
        row = self._field(right, row, "Min obstacles:", self.min_crates_var)
        row = self._field(right, row, "Max obstacles:", self.max_crates_var)
        row = self._field(right, row, "Min ruins:", self.min_ruins_var)
        row = self._field(right, row, "Max ruins:", self.max_ruins_var)

        row = self._sep(right, row)
        row = self._section(right, row, "Scoring Targets")
        row = self._field(
            right, row, "Overall vis %:", self.overall_vis_target_var
        )
        row = self._field(right, row, "  weight:", self.overall_vis_weight_var)
        row = self._field(right, row, "DZ vis %:", self.dz_vis_target_var)
        row = self._field(right, row, "  weight:", self.dz_vis_weight_var)
        row = self._field(
            right, row, "DZ hidden %:", self.dz_hidden_target_var
        )
        row = self._field(right, row, "  weight:", self.dz_hidden_weight_var)
        row = self._field(right, row, "Obj hide %:", self.obj_hide_target_var)
        row = self._field(right, row, "  weight:", self.obj_hide_weight_var)

        row = self._sep(right, row)
        row = self._section(right, row, "Results")
        self.visibility_label = ttk.Label(right, text="Visibility: --")
        self.visibility_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1
        self.dz_vis_label = ttk.Label(right, text="")
        self.dz_vis_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1
        self.dz_to_dz_vis_label = ttk.Label(right, text="")
        self.dz_to_dz_vis_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1
        self.obj_hide_label = ttk.Label(right, text="")
        self.obj_hide_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

    def _section(self, parent, row, title):
        ttk.Label(parent, text=title, font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(8, 4), sticky="w"
        )
        return row + 1

    def _field(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", pady=2
        )
        ttk.Entry(parent, textvariable=var, width=10).grid(
            row=row, column=1, sticky="w", pady=2, padx=(5, 0)
        )
        return row + 1

    def _combo(self, parent, row, label, var, attr_name):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", pady=2
        )
        combo = ttk.Combobox(
            parent, textvariable=var, width=28, state="readonly"
        )
        combo.grid(row=row, column=1, sticky="w", pady=2, padx=(5, 0))
        setattr(self, attr_name, combo)
        return row + 1

    def _sep(self, parent, row):
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8
        )
        return row + 1

    def _init_mission_combos(self):
        """Set up cascading mission dropdowns."""
        editions = list(EDITIONS.keys())
        self.edition_combo["values"] = editions
        if editions:
            self.edition_var.set(editions[0])
            self._on_edition_changed()

        self.edition_var.trace_add(
            "write", lambda *_: self._on_edition_changed()
        )
        self.pack_var.trace_add("write", lambda *_: self._on_pack_changed())
        self.deployment_var.trace_add(
            "write", lambda *_: self._on_deployment_changed()
        )

    def _on_edition_changed(self):
        edition = self.edition_var.get()
        packs = list(EDITIONS.get(edition, {}).keys())
        self.pack_combo["values"] = packs
        if packs:
            self.pack_var.set(packs[0])
        else:
            self.pack_var.set("")
        self._on_pack_changed()

    def _on_pack_changed(self):
        edition = self.edition_var.get()
        pack = self.pack_var.get()
        deployments = list(EDITIONS.get(edition, {}).get(pack, {}).keys())
        self.deployment_combo["values"] = ["None"] + deployments
        self.deployment_var.set("None")
        self._on_deployment_changed()

    def _on_deployment_changed(self):
        dep = self.deployment_var.get()
        if dep == "None" or dep == "":
            self.selected_mission = None
        else:
            edition = self.edition_var.get()
            pack = self.pack_var.get()
            self.selected_mission = (
                EDITIONS.get(edition, {}).get(pack, {}).get(dep)
            )
        if not self._initializing:
            self.on_table_changed()

    def _dims_changed(self, *_args):
        try:
            w = self.table_width_var.get()
            d = self.table_depth_var.get()
            if w > 0 and d > 0:
                self.on_table_changed()
        except (tk.TclError, ValueError):
            pass

    def get_params(self):
        """Return current params as a dict matching engine_params schema.

        Seed is None if the field is blank or not a valid integer.
        Optional fields (min/max crates, gaps) are only included if set.
        """
        try:
            seed_str = self.seed_var.get().strip()
            seed = int(seed_str) if seed_str else None
        except ValueError:
            seed = None

        try:
            # Parse optional integer fields
            def parse_int(s):
                s = s.strip()
                return int(s) if s else None

            # Parse optional float fields
            def parse_float(s):
                s = s.strip()
                return float(s) if s else None

            min_crates = parse_int(self.min_crates_var.get())
            max_crates = parse_int(self.max_crates_var.get())
            min_ruins = parse_int(self.min_ruins_var.get())
            max_ruins = parse_int(self.max_ruins_var.get())
            min_gap = parse_float(self.min_gap_var.get())
            min_edge_gap = parse_float(self.min_edge_gap_var.get())

            # Build feature count preferences only if values are set
            feature_count_prefs = []
            if min_crates is not None or max_crates is not None:
                feature_count_prefs.append(
                    {
                        "feature_type": "obstacle",
                        "min": min_crates if min_crates is not None else 0,
                        "max": max_crates,
                    }
                )
            if min_ruins is not None or max_ruins is not None:
                feature_count_prefs.append(
                    {
                        "feature_type": "obscuring",
                        "min": min_ruins if min_ruins is not None else 0,
                        "max": max_ruins,
                    }
                )

            # Auto-detect replica count from CPU cores
            cpu_count = os.cpu_count() or 2
            num_replicas = max(2, min(cpu_count, 8))

            # Build params dict
            params = {
                "seed": seed,
                "table_width_inches": self.table_width_var.get(),
                "table_depth_inches": self.table_depth_var.get(),
                "rotationally_symmetric": self.symmetric_var.get(),
                "num_steps": self.num_steps_var.get(),
                "catalog": SAMPLE_CATALOG,
                "feature_count_preferences": feature_count_prefs,
                "num_replicas": num_replicas,
            }

            # Only include gap parameters if they're set
            if min_gap is not None:
                params["min_feature_gap_inches"] = min_gap
            if min_edge_gap is not None:
                params["min_edge_gap_inches"] = min_edge_gap

            # Include mission if selected
            if self.selected_mission is not None:
                params["mission"] = self.selected_mission

            # Build scoring targets if any target is set
            overall_vis_t = parse_float(self.overall_vis_target_var.get())
            overall_vis_w = parse_float(self.overall_vis_weight_var.get())
            dz_vis_t = parse_float(self.dz_vis_target_var.get())
            dz_vis_w = parse_float(self.dz_vis_weight_var.get())
            dz_hidden_t = parse_float(self.dz_hidden_target_var.get())
            dz_hidden_w = parse_float(self.dz_hidden_weight_var.get())
            obj_hide_t = parse_float(self.obj_hide_target_var.get())
            obj_hide_w = parse_float(self.obj_hide_weight_var.get())

            has_any_target = any(
                t is not None
                for t in [overall_vis_t, dz_vis_t, dz_hidden_t, obj_hide_t]
            )
            if has_any_target:
                scoring_targets = {}
                if overall_vis_t is not None:
                    scoring_targets["overall_visibility_target"] = (
                        overall_vis_t
                    )
                    if overall_vis_w is not None:
                        scoring_targets["overall_visibility_weight"] = (
                            overall_vis_w
                        )
                if dz_vis_t is not None:
                    scoring_targets["dz_visibility_target"] = dz_vis_t
                    if dz_vis_w is not None:
                        scoring_targets["dz_visibility_weight"] = dz_vis_w
                if dz_hidden_t is not None:
                    scoring_targets["dz_hidden_target"] = dz_hidden_t
                    if dz_hidden_w is not None:
                        scoring_targets["dz_hidden_weight"] = dz_hidden_w
                if obj_hide_t is not None:
                    scoring_targets["objective_hidability_target"] = obj_hide_t
                    if obj_hide_w is not None:
                        scoring_targets["objective_hidability_weight"] = (
                            obj_hide_w
                        )
                params["scoring_targets"] = scoring_targets

            return params
        except (tk.TclError, ValueError):
            return None


# ---------------------------------------------------------------------------
# History panel
# ---------------------------------------------------------------------------


class HistoryPanel(ttk.Frame):
    """Panel displaying recent layout history."""

    def __init__(self, parent, on_load_layout):
        super().__init__(parent, padding=10)
        self.on_load_layout = on_load_layout
        self.layout_history = []  # list of (timestamp, layout_dict) tuples
        self.history_listbox = None
        self._build()

    def _build(self):
        ttk.Label(self, text="Recent Layouts", font=("", 11, "bold")).pack(
            anchor="w", pady=(0, 8)
        )

        # Listbox with scrollbar for history
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("", 9),
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)
        scrollbar.config(command=self.history_listbox.yview)

    def add_to_history(self, layout):
        """Add a layout to the history with current timestamp."""
        timestamp = int(time.time())
        self.layout_history.append((timestamp, copy.deepcopy(layout)))

        # Add to listbox display (limit to 50 recent)
        if len(self.layout_history) > 50:
            self.layout_history.pop(0)
            self.history_listbox.delete(0)

        self.history_listbox.insert(
            tk.END,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
        )

    def _on_history_select(self, event):
        """Load selected layout from history."""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx < len(self.layout_history):
            _timestamp, layout = self.layout_history[idx]
            self.on_load_layout(layout)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def _build_object_index(catalog):
    """Build {object_id: object_dict} from a catalog."""
    index = {}
    for entry in catalog.get("objects", []):
        obj = entry["item"]
        index[obj["id"]] = obj
    # Also index objects that appear inside catalog features.
    for entry in catalog.get("features", []):
        for comp in entry["item"].get("components", []):
            if "object" in comp:
                obj = comp["object"]
                index[obj["id"]] = obj
    return index


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Carltographer")
        self.root.geometry("1400x750")
        self.root.resizable(True, True)
        self.root.configure(bg=CANVAS_BG)

        style = ttk.Style()
        style.theme_use("clam")

        # Canvas on the left, controls and history on the right.
        self.canvas = tk.Canvas(self.root, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Right panel for controls and history
        self.right_panel = ttk.Frame(self.root)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, padx=(0, 5))

        self.controls = ControlPanel(
            self.right_panel,
            on_table_changed=self._render,
            on_generate=self._on_generate,
            on_clear=self._on_clear,
            on_save=self._on_save,
            on_load=self._on_load,
        )
        self.controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        self.history = HistoryPanel(
            self.right_panel,
            on_load_layout=self._load_layout,
        )
        self.history.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._photo = None  # prevent GC
        self.layout = {
            "table_width_inches": 60,
            "table_depth_inches": 44,
            "placed_features": [],
        }
        self.objects_by_id = _build_object_index(SAMPLE_CATALOG)

        # Rendering context for coordinate conversion
        self._render_ppi = None
        self._render_tw = None
        self._render_td = None
        self._img_offset_x = None
        self._img_offset_y = None

        # Selection state
        self._selected_feature_idx = None
        self._selected_is_mirror = False
        self._popup_window_id = None
        self._popup_frame = None

        # Move/copy mode state
        self._move_mode = False
        self._move_is_copy = False
        self._move_feature_idx = None
        self._move_original_pf_dict = None
        self._move_last_valid_pos = None
        self._move_current_rotation = 0.0
        self._move_overlay_ids: list[int] = []
        self._move_mirror_overlay_ids: list[int] = []
        self._move_base_photo = None
        self._move_typed_features = None
        self._move_typed_objects = None

        self.root.after(50, self._render)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Log which engine will be used
        self._log_engine_selection()

    def _log_engine_selection(self):
        """Print which engine is being used on startup."""
        if _should_use_rust_engine():
            print(
                "✓ Using Rust engine (Python source matches certified hashes)"
            )
        else:
            if not _HAS_RUST_ENGINE:
                print(
                    "⚠ Using Python engine "
                    "(Rust engine not available - run: cd v2/engine_rs && "
                    "maturin develop)"
                )
            else:
                print(
                    "⚠ Using Python engine "
                    "(Python source has changed since last certification - "
                    "run: python -m engine_cmp.compare)"
                )

    # -- rendering --

    def _render(self):
        if self._move_mode:
            return

        if hasattr(self, "_popup_window_id"):
            self._dismiss_popup()

        params = self.controls.get_params()
        if params is None:
            return

        tw = params["table_width_inches"]
        td = params["table_depth_inches"]

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return

        margin = 20
        ppi = min((cw - 2 * margin) / tw, (ch - 2 * margin) / td)
        if ppi <= 0:
            return

        # Save rendering context for coordinate conversion
        self._render_ppi = ppi
        self._render_tw = tw
        self._render_td = td
        img_w = int(tw * ppi)
        img_h = int(td * ppi)
        self._img_offset_x = cw / 2 - img_w / 2
        self._img_offset_y = ch / 2 - img_h / 2

        mission = self.controls.selected_mission
        renderer = BattlefieldRenderer(
            tw, td, ppi, self.objects_by_id, mission
        )
        img = renderer.render(
            self.layout, highlight_index=self._selected_feature_idx
        )

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            cw / 2, ch / 2, image=self._photo, anchor="center"
        )

    # -- coordinate conversion & hit testing --

    def _canvas_to_table(self, canvas_x, canvas_y):
        """Convert canvas pixel coords to table-space inches.

        Returns (x_inches, z_inches) or None if outside the table image.
        """
        ppi = self._render_ppi
        tw = self._render_tw
        td = self._render_td
        ox = self._img_offset_x
        oy = self._img_offset_y
        if ppi is None or tw is None or td is None or ox is None or oy is None:
            return None
        # Position relative to image top-left
        rel_x = canvas_x - ox
        rel_y = canvas_y - oy
        img_w = tw * ppi
        img_h = td * ppi
        if rel_x < 0 or rel_y < 0 or rel_x >= img_w or rel_y >= img_h:
            return None
        # Reverse of _to_px: px = (x + tw/2) * ppi => x = px/ppi - tw/2
        x_inches = rel_x / ppi - tw / 2
        z_inches = rel_y / ppi - td / 2
        return (x_inches, z_inches)

    def _get_feature_polygons(self, placed_feature):
        """Return list of table-space polygon corner lists for a placed feature."""
        feature = placed_feature["feature"]
        feat_tf = _get_tf(placed_feature.get("transform"))
        polygons = []

        for comp in feature["components"]:
            obj = self.objects_by_id.get(comp["object_id"])
            if obj is None:
                continue
            comp_tf = _get_tf(comp.get("transform"))

            for shape in obj["shapes"]:
                offset_tf = _get_tf(shape.get("offset"))
                combined = _compose(feat_tf, _compose(comp_tf, offset_tf))
                cx, cz, rot_deg = combined
                hw = shape["width_inches"] / 2
                hd = shape["depth_inches"] / 2
                corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]

                rad = math.radians(rot_deg)
                cos_r = math.cos(rad)
                sin_r = math.sin(rad)

                world_corners = []
                for lx, lz in corners:
                    rx = cx + lx * cos_r - lz * sin_r
                    rz = cz + lx * sin_r + lz * cos_r
                    world_corners.append((rx, rz))
                polygons.append(world_corners)
        return polygons

    def _hit_test(self, x_inches, z_inches):
        """Test if point hits a placed feature.

        Returns (feature_index, is_mirror) or None.
        Iterates in reverse order so topmost features are tested first.
        """
        placed = list(self.layout.get("placed_features", []))  # type: ignore[arg-type]
        is_symmetric = self.layout.get("rotationally_symmetric", False)

        for idx in range(len(placed) - 1, -1, -1):
            pf = placed[idx]

            # Test canonical placement
            for poly in self._get_feature_polygons(pf):
                if BattlefieldRenderer._point_in_polygon(
                    x_inches, z_inches, poly
                ):
                    return (idx, False)

            # Test mirror if symmetric
            if is_symmetric:
                tf = pf.get("transform", {})
                if (
                    tf.get("x_inches", 0.0) != 0.0
                    or tf.get("z_inches", 0.0) != 0.0
                ):
                    mirror = _mirror_pf_dict(pf)
                    for poly in self._get_feature_polygons(mirror):
                        if BattlefieldRenderer._point_in_polygon(
                            x_inches, z_inches, poly
                        ):
                            return (idx, True)

        return None

    # -- coordinate helpers --

    def _table_to_canvas(self, x_inches, z_inches):
        """Convert table-space inches to canvas pixel coords.

        Returns (canvas_x, canvas_y) or None if render context unavailable.
        """
        ppi = self._render_ppi
        tw = self._render_tw
        td = self._render_td
        ox = self._img_offset_x
        oy = self._img_offset_y
        if ppi is None or tw is None or td is None or ox is None or oy is None:
            return None
        px = (x_inches + tw / 2) * ppi + ox
        py = (z_inches + td / 2) * ppi + oy
        return (px, py)

    def _get_feature_canvas_polygons(self, placed_feature):
        """Return list of (flat_coords, fill_color, outline_color) for canvas polygons."""
        feature = placed_feature["feature"]
        feat_tf = _get_tf(placed_feature.get("transform"))
        results = []

        for comp in feature["components"]:
            obj = self.objects_by_id.get(comp["object_id"])
            if obj is None:
                continue
            comp_tf = _get_tf(comp.get("transform"))
            fill = obj.get("fill_color") or DEFAULT_FILL
            outline = obj.get("outline_color") or ""

            for shape in obj["shapes"]:
                offset_tf = _get_tf(shape.get("offset"))
                combined = _compose(feat_tf, _compose(comp_tf, offset_tf))
                cx, cz, rot_deg = combined
                hw = shape["width_inches"] / 2
                hd = shape["depth_inches"] / 2
                corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]

                rad = math.radians(rot_deg)
                cos_r = math.cos(rad)
                sin_r = math.sin(rad)

                canvas_coords = []
                for lx, lz in corners:
                    rx = cx + lx * cos_r - lz * sin_r
                    rz = cz + lx * sin_r + lz * cos_r
                    cp = self._table_to_canvas(rx, rz)
                    if cp is None:
                        return []
                    canvas_coords.extend(cp)
                results.append((canvas_coords, fill, outline))
        return results

    # -- move mode validation --

    def _build_typed_validation_context(self, exclude_idx):
        """Build typed PlacedFeature list and objects_by_id for engine validation.

        Excludes the feature at exclude_idx from placed_features.
        """
        placed = list(self.layout.get("placed_features", []))  # type: ignore[arg-type]
        typed_features = []
        for i, pf_dict in enumerate(placed):
            if i == exclude_idx:
                continue
            typed_features.append(TypedPlacedFeature.from_dict(pf_dict))

        typed_objects = {}
        for obj_id, obj_dict in self.objects_by_id.items():
            typed_objects[obj_id] = TypedTerrainObject.from_dict(obj_dict)

        return typed_features, typed_objects

    def _validate_move_position(self, x_inches, z_inches):
        """Check if placing the moving feature at (x, z) is valid."""
        if self._move_feature_idx is None:
            return False

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        pf_dict = placed[self._move_feature_idx]

        candidate = TypedPlacedFeature.from_dict(
            {
                "feature": pf_dict["feature"],
                "transform": {
                    "x_inches": x_inches,
                    "z_inches": z_inches,
                    "rotation_deg": self._move_current_rotation,
                },
            }
        )

        if (
            self._move_typed_features is None
            or self._move_typed_objects is None
        ):
            return False

        check_list = list(self._move_typed_features) + [candidate]
        check_idx = len(check_list) - 1

        params = self.controls.get_params()
        if params is None:
            return False

        tw = params["table_width_inches"]
        td = params["table_depth_inches"]
        min_gap = params.get("min_feature_gap_inches")
        min_edge_gap = params.get("min_edge_gap_inches")
        symmetric = bool(self.layout.get("rotationally_symmetric", False))

        return is_valid_placement(
            check_list,
            check_idx,
            tw,
            td,
            self._move_typed_objects,
            min_feature_gap=min_gap,
            min_edge_gap=min_edge_gap,
            rotationally_symmetric=symmetric,
        )

    # -- selection & popup --

    def _on_canvas_click(self, event):
        """Handle click on canvas: select/deselect features."""
        if self._move_mode:
            return
        self._dismiss_popup()

        table_coords = self._canvas_to_table(event.x, event.y)
        if table_coords is None:
            self._deselect()
            return

        hit = self._hit_test(*table_coords)
        if hit is None:
            self._deselect()
            return

        idx, is_mirror = hit
        self._select_feature(idx, is_mirror, event.x, event.y)

    def _select_feature(self, idx, is_mirror, canvas_x, canvas_y):
        """Select a feature and show popup."""
        self._selected_feature_idx = idx
        self._selected_is_mirror = is_mirror
        self._render()
        self._show_popup(canvas_x, canvas_y)

    def _deselect(self):
        """Clear selection and re-render if needed."""
        if self._selected_feature_idx is not None:
            self._selected_feature_idx = None
            self._selected_is_mirror = False
            self._render()

    def _show_popup(self, canvas_x, canvas_y):
        """Show Move/Copy/Delete/Cancel popup at the given canvas position."""
        self._dismiss_popup()

        frame = tk.Frame(
            self.canvas, bg="#333333", relief="raised", borderwidth=2
        )
        move_btn = tk.Button(
            frame,
            text="Move",
            command=self._on_move_selected,
            bg="#336699",
            fg="white",
            activebackground="#4488bb",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        move_btn.pack(side=tk.LEFT, padx=(4, 2), pady=4)
        copy_btn = tk.Button(
            frame,
            text="Copy",
            command=self._on_copy_selected,
            bg="#996633",
            fg="white",
            activebackground="#bb8844",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        copy_btn.pack(side=tk.LEFT, padx=(2, 2), pady=4)
        delete_btn = tk.Button(
            frame,
            text="Delete",
            command=self._on_delete_selected,
            bg="#cc3333",
            fg="white",
            activebackground="#ff4444",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        delete_btn.pack(side=tk.LEFT, padx=(2, 2), pady=4)
        cancel_btn = tk.Button(
            frame,
            text="Cancel",
            command=self._dismiss_popup_and_deselect,
            bg="#555555",
            fg="white",
            activebackground="#777777",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        cancel_btn.pack(side=tk.LEFT, padx=(2, 4), pady=4)

        # Clamp popup position to stay within canvas bounds
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        popup_w = 300  # approximate width
        popup_h = 40  # approximate height
        x = min(canvas_x, cw - popup_w)
        y = min(canvas_y, ch - popup_h)
        x = max(x, 0)
        y = max(y, 0)

        win_id = self.canvas.create_window(x, y, window=frame, anchor="nw")
        self._popup_window_id = win_id
        self._popup_frame = frame

    def _dismiss_popup(self):
        """Remove popup widget and canvas window item if present."""
        if self._popup_window_id is not None:
            self.canvas.delete(self._popup_window_id)
            self._popup_window_id = None
        if self._popup_frame is not None:
            self._popup_frame.destroy()
            self._popup_frame = None

    def _dismiss_popup_and_deselect(self):
        """Cancel button handler."""
        self._dismiss_popup()
        self._deselect()

    # -- shared engine helper --

    def _rerun_engine_zero_steps(self):
        """Re-run engine with zero steps to recompute visibility, then render."""
        params = self.controls.get_params()
        if params:
            params["num_steps"] = 0
            if params["seed"] is None:
                params["seed"] = random.randint(0, 2**32 - 1)
            params["initial_layout"] = self.layout

            if _should_use_rust_engine():
                result_json = _engine_rs.generate_json(json.dumps(params))  # type: ignore
                result = json.loads(result_json)
            else:
                result = generate_json(params)

            self.layout = result["layout"]
            self._update_visibility_display()
            self.history.add_to_history(self.layout)

        self._render()

    # -- delete action --

    def _on_delete_selected(self):
        """Delete the selected feature and re-run engine with zero steps."""
        idx = self._selected_feature_idx
        if idx is None:
            return

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        if idx < 0 or idx >= len(placed):
            return

        placed.pop(idx)
        self._selected_feature_idx = None
        self._dismiss_popup()
        self._rerun_engine_zero_steps()

    # -- move action --

    def _on_move_selected(self):
        """Enter move mode for the selected feature."""
        idx = self._selected_feature_idx
        if idx is None:
            return

        placed = list(self.layout.get("placed_features", []))  # type: ignore[arg-type]
        if idx < 0 or idx >= len(placed):
            return

        self._move_original_pf_dict = copy.deepcopy(placed[idx])
        self._move_feature_idx = idx
        self._move_mode = True

        pf_tf = placed[idx].get("transform", {})
        feat_x = pf_tf.get("x_inches", 0.0)
        feat_z = pf_tf.get("z_inches", 0.0)
        feat_rot = pf_tf.get("rotation_deg", 0.0)

        # If user clicked on the mirror, start from the mirror's transform
        if self._selected_is_mirror:
            feat_x = -feat_x
            feat_z = -feat_z
            feat_rot = feat_rot + 180.0

        self._move_last_valid_pos = (feat_x, feat_z)
        self._move_current_rotation = feat_rot

        self._dismiss_popup()
        self._selected_feature_idx = None
        self._selected_is_mirror = False

        self._move_typed_features, self._move_typed_objects = (
            self._build_typed_validation_context(idx)
        )

        self._render_move_base()
        self._create_move_overlay()
        self._bind_move_events()

    # -- copy action --

    def _on_copy_selected(self):
        """Enter copy mode: duplicate the selected feature and place the copy."""
        idx = self._selected_feature_idx
        if idx is None:
            return

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        if idx < 0 or idx >= len(placed):
            return

        # Append a deep copy to placed_features
        new_pf = copy.deepcopy(placed[idx])
        placed.append(new_pf)
        new_idx = len(placed) - 1

        self._move_original_pf_dict = None  # no original to revert to
        self._move_feature_idx = new_idx
        self._move_mode = True
        self._move_is_copy = True

        pf_tf = new_pf.get("transform", {})
        feat_rot = pf_tf.get("rotation_deg", 0.0)

        # If user clicked on the mirror, start with the mirror's rotation
        if self._selected_is_mirror:
            feat_rot = feat_rot + 180.0

        # Start with no valid position (copy starts on top of original)
        self._move_last_valid_pos = None
        self._move_current_rotation = feat_rot

        self._dismiss_popup()
        self._selected_is_mirror = False
        self._selected_feature_idx = None

        self._move_typed_features, self._move_typed_objects = (
            self._build_typed_validation_context(new_idx)
        )

        self._render_move_base()
        self._create_move_overlay()
        self._bind_move_events()

    def _bind_move_events(self):
        """Bind canvas/root events for move/copy mode."""
        # Steal focus from any text entry so keyboard bindings work
        self.canvas.focus_set()

        self.canvas.bind("<Motion>", self._on_move_motion)
        self.canvas.bind("<Button-1>", self._on_move_click)
        self.canvas.bind("<Button-3>", self._on_move_cancel)
        self.canvas.bind("<MouseWheel>", self._on_move_scroll)
        self.canvas.bind("<Button-4>", self._on_move_scroll_up)
        self.canvas.bind("<Button-5>", self._on_move_scroll_down)
        self.root.bind("<Escape>", self._on_move_cancel)
        self.root.bind("<q>", self._on_move_rotate_ccw)
        self.root.bind("<e>", self._on_move_rotate_cw)
        self.root.bind("<r>", self._on_move_rotate_cw)
        self.root.bind("<R>", self._on_move_rotate_ccw)
        self.root.bind("<Left>", self._on_move_rotate_ccw)
        self.root.bind("<Right>", self._on_move_rotate_cw)

    def _render_move_base(self):
        """Render layout without the moving feature as the base image."""
        if self._move_feature_idx is None:
            return
        params = self.controls.get_params()
        if params is None:
            return

        tw = params["table_width_inches"]
        td = params["table_depth_inches"]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return

        margin = 20
        ppi = min((cw - 2 * margin) / tw, (ch - 2 * margin) / td)
        if ppi <= 0:
            return

        self._render_ppi = ppi
        self._render_tw = tw
        self._render_td = td
        img_w = int(tw * ppi)
        img_h = int(td * ppi)
        self._img_offset_x = cw / 2 - img_w / 2
        self._img_offset_y = ch / 2 - img_h / 2

        temp_layout = copy.deepcopy(self.layout)
        temp_placed: list = temp_layout["placed_features"]  # type: ignore[assignment]
        temp_placed.pop(self._move_feature_idx)

        mission = self.controls.selected_mission
        renderer = BattlefieldRenderer(
            tw, td, ppi, self.objects_by_id, mission
        )
        img = renderer.render(temp_layout)

        self._move_base_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            cw / 2, ch / 2, image=self._move_base_photo, anchor="center"
        )

    def _create_move_overlay(self):
        """Create canvas polygon items for the feature being moved."""
        self._clear_move_overlay()

        idx = self._move_feature_idx
        if idx is None:
            return

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        pf_dict = placed[idx]
        polys = self._get_feature_canvas_polygons(pf_dict)

        for coords, fill, _outline in polys:
            item_id = self.canvas.create_polygon(
                coords,
                fill=fill,
                outline=MOVE_VALID_OUTLINE,
                width=2,
            )
            self._move_overlay_ids.append(item_id)

        is_symmetric = self.layout.get("rotationally_symmetric", False)
        if is_symmetric:
            tf = pf_dict.get("transform", {})
            if (
                tf.get("x_inches", 0.0) != 0.0
                or tf.get("z_inches", 0.0) != 0.0
            ):
                mirror_pf = _mirror_pf_dict(pf_dict)
                mirror_polys = self._get_feature_canvas_polygons(mirror_pf)
                for coords, fill, _outline in mirror_polys:
                    item_id = self.canvas.create_polygon(
                        coords,
                        fill=fill,
                        outline=MOVE_VALID_OUTLINE,
                        width=2,
                        stipple="gray50",
                    )
                    self._move_mirror_overlay_ids.append(item_id)

    def _update_move_overlay(self, new_x, new_z, is_valid):
        """Update the canvas polygon overlay to reflect new position."""
        idx = self._move_feature_idx
        if idx is None:
            return

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        pf_dict = placed[idx]
        old_tf = pf_dict.get("transform", {})

        temp_pf = {
            **pf_dict,
            "transform": {
                **old_tf,
                "x_inches": new_x,
                "z_inches": new_z,
                "rotation_deg": self._move_current_rotation,
            },
        }

        outline_color = (
            MOVE_VALID_OUTLINE if is_valid else MOVE_INVALID_OUTLINE
        )

        polys = self._get_feature_canvas_polygons(temp_pf)
        for i, (coords, _fill, _outline) in enumerate(polys):
            if i < len(self._move_overlay_ids):
                self.canvas.coords(self._move_overlay_ids[i], *coords)
                self.canvas.itemconfig(
                    self._move_overlay_ids[i],
                    outline=outline_color,
                )

        is_symmetric = self.layout.get("rotationally_symmetric", False)
        if is_symmetric and (new_x != 0.0 or new_z != 0.0):
            mirror_pf = _mirror_pf_dict(temp_pf)
            mirror_polys = self._get_feature_canvas_polygons(mirror_pf)
            for i, (coords, _fill, _outline) in enumerate(mirror_polys):
                if i < len(self._move_mirror_overlay_ids):
                    self.canvas.coords(
                        self._move_mirror_overlay_ids[i], *coords
                    )
                    self.canvas.itemconfig(
                        self._move_mirror_overlay_ids[i],
                        outline=outline_color,
                    )
            for item_id in self._move_mirror_overlay_ids:
                self.canvas.itemconfig(item_id, state="normal")
        elif is_symmetric:
            for item_id in self._move_mirror_overlay_ids:
                self.canvas.itemconfig(item_id, state="hidden")

    def _clear_move_overlay(self):
        """Remove all move overlay canvas items."""
        for item_id in self._move_overlay_ids:
            self.canvas.delete(item_id)
        self._move_overlay_ids = []
        for item_id in self._move_mirror_overlay_ids:
            self.canvas.delete(item_id)
        self._move_mirror_overlay_ids = []

    def _on_move_motion(self, event):
        """Track cursor during move mode, updating overlay position."""
        if not self._move_mode:
            return

        table_coords = self._canvas_to_table(event.x, event.y)
        if table_coords is None:
            return

        raw_x, raw_z = table_coords
        qx = round(raw_x / 0.1) * 0.1
        qz = round(raw_z / 0.1) * 0.1

        is_valid = self._validate_move_position(qx, qz)

        if is_valid:
            self._move_last_valid_pos = (qx, qz)
            self._update_move_overlay(qx, qz, True)
        elif self._move_last_valid_pos:
            # Have a prior valid position: held back there
            lx, lz = self._move_last_valid_pos
            self._update_move_overlay(lx, lz, False)
        else:
            # No valid position yet (copy mode start): follow cursor, red
            self._update_move_overlay(qx, qz, False)

    def _rotate_move_feature(self, delta_deg):
        """Rotate the moving feature by delta_deg and revalidate."""
        if not self._move_mode:
            return

        new_rot = (self._move_current_rotation + delta_deg) % 360.0
        old_rot = self._move_current_rotation
        self._move_current_rotation = new_rot

        if self._move_last_valid_pos is not None:
            lx, lz = self._move_last_valid_pos
            is_valid = self._validate_move_position(lx, lz)

            if is_valid:
                self._update_move_overlay(lx, lz, True)
            else:
                # Revert rotation if invalid at current held position
                self._move_current_rotation = old_rot
                self._update_move_overlay(lx, lz, False)

    def _on_move_rotate_cw(self, event=None):
        self._rotate_move_feature(15.0)

    def _on_move_rotate_ccw(self, event=None):
        self._rotate_move_feature(-15.0)

    def _on_move_scroll(self, event):
        """Handle mouse wheel rotation (Windows/macOS)."""
        if not self._move_mode:
            return
        if event.delta > 0:
            self._rotate_move_feature(15.0)
        elif event.delta < 0:
            self._rotate_move_feature(-15.0)

    def _on_move_scroll_up(self, event):
        """Handle scroll up (Linux Button-4)."""
        self._rotate_move_feature(15.0)

    def _on_move_scroll_down(self, event):
        """Handle scroll down (Linux Button-5)."""
        self._rotate_move_feature(-15.0)

    def _on_move_click(self, event):
        """Place feature at current position and show confirm/cancel popup."""
        if not self._move_mode:
            return

        if self._move_last_valid_pos is None:
            if self._move_is_copy:
                # Copy mode: clicking while invalid cancels the copy
                self._on_move_cancel()
            return

        # Stop tracking cursor
        self.canvas.bind("<Motion>", lambda e: None)

        # Show confirm/cancel popup near the feature
        new_x, new_z = self._move_last_valid_pos
        cp = self._table_to_canvas(new_x, new_z)
        if cp is None:
            return

        canvas_x, canvas_y = cp
        self._show_move_confirm_popup(canvas_x, canvas_y)

    def _show_move_confirm_popup(self, canvas_x, canvas_y):
        """Show Confirm/Cancel popup for move placement."""
        self._dismiss_popup()

        frame = tk.Frame(
            self.canvas, bg="#333333", relief="raised", borderwidth=2
        )
        confirm_btn = tk.Button(
            frame,
            text="Confirm",
            command=self._on_move_confirm,
            bg="#339933",
            fg="white",
            activebackground="#44bb44",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        confirm_btn.pack(side=tk.LEFT, padx=(4, 2), pady=4)
        cancel_btn = tk.Button(
            frame,
            text="Cancel",
            command=self._on_move_resume,
            bg="#555555",
            fg="white",
            activebackground="#777777",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        cancel_btn.pack(side=tk.LEFT, padx=(2, 4), pady=4)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        popup_w = 180
        popup_h = 40
        x = min(canvas_x, cw - popup_w)
        y = min(canvas_y - popup_h - 10, ch - popup_h)
        x = max(x, 0)
        y = max(y, 0)

        win_id = self.canvas.create_window(x, y, window=frame, anchor="nw")
        self._popup_window_id = win_id
        self._popup_frame = frame

    def _on_move_confirm(self):
        """Commit the move to the new position."""
        if self._move_last_valid_pos is None or self._move_feature_idx is None:
            return

        new_x, new_z = self._move_last_valid_pos
        idx = self._move_feature_idx

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        pf = placed[idx]
        old_tf = pf.get("transform", {})
        pf["transform"] = {
            **old_tf,
            "x_inches": new_x,
            "z_inches": new_z,
            "rotation_deg": self._move_current_rotation,
        }

        self._dismiss_popup()
        self._exit_move_mode()
        self._rerun_engine_zero_steps()

    def _on_move_resume(self):
        """Cancel placement and revert to original position."""
        self._on_move_cancel()

    def _on_move_cancel(self, event=None):
        """Cancel move/copy: revert to original state."""
        if not self._move_mode:
            return

        idx = self._move_feature_idx
        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        if self._move_is_copy:
            # Copy mode: remove the appended copy
            if idx is not None and idx < len(placed):
                placed.pop(idx)
        else:
            # Move mode: restore original feature
            if idx is not None and self._move_original_pf_dict is not None:
                placed[idx] = self._move_original_pf_dict

        self._dismiss_popup()
        self._exit_move_mode()
        self._render()

    def _exit_move_mode(self):
        """Clean up move/copy mode state and restore normal event bindings."""
        self._move_mode = False
        self._move_is_copy = False
        self._move_feature_idx = None
        self._move_original_pf_dict = None
        self._move_last_valid_pos = None
        self._move_current_rotation = 0.0
        self._move_typed_features = None
        self._move_typed_objects = None
        # NOTE: do NOT clear _move_base_photo here. It backs the canvas
        # image; clearing it before _render() creates a replacement causes
        # a black screen if _render() returns early (e.g. get_params()=None).
        # It will be naturally replaced by the next _render() call.

        self._clear_move_overlay()

        self.canvas.bind("<Motion>", lambda e: None)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", lambda e: None)
        self.canvas.bind("<MouseWheel>", lambda e: None)
        self.canvas.bind("<Button-4>", lambda e: None)
        self.canvas.bind("<Button-5>", lambda e: None)
        self.root.unbind("<Escape>")
        self.root.unbind("<q>")
        self.root.unbind("<e>")
        self.root.unbind("<r>")
        self.root.unbind("<R>")
        self.root.unbind("<Left>")
        self.root.unbind("<Right>")

    # -- canvas resize --

    def _on_canvas_configure(self, _event):
        """Handle canvas resize: dismiss popup and re-render."""
        if self._move_mode:
            self._on_move_cancel()
            return
        self._dismiss_popup()
        self._selected_feature_idx = None
        self._render()

    # -- actions --

    def _load_layout(self, layout):
        """Load a layout from history and display it."""
        self.layout = copy.deepcopy(layout)
        self._update_visibility_display()
        self._deselect()
        self._render()

    def _on_clear(self):
        self._dismiss_popup()
        self._selected_feature_idx = None
        self.layout = {
            "table_width_inches": self.controls.table_width_var.get(),
            "table_depth_inches": self.controls.table_depth_var.get(),
            "placed_features": [],
            "rotationally_symmetric": self.controls.symmetric_var.get(),
        }
        self._update_visibility_display()
        self._render()

    def _on_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            initialfile=f"layout_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png",
        )
        if not path:
            return

        params = self.controls.get_params()
        if params is None:
            return

        tw = params["table_width_inches"]
        td = params["table_depth_inches"]
        ppi = 20

        mission = self.controls.selected_mission
        renderer = BattlefieldRenderer(
            tw, td, ppi, self.objects_by_id, mission
        )
        img = renderer.render(self.layout)
        save_layout_png(img, self.layout, path)

    def _on_load(self):
        self._dismiss_popup()
        self._selected_feature_idx = None
        path = filedialog.askopenfilename(
            filetypes=[
                ("Layout files", "*.png *.json"),
                ("PNG files", "*.png"),
                ("JSON files", "*.json"),
            ],
        )
        if not path:
            return
        try:
            layout = load_layout(path)
        except (ValueError, Exception) as e:
            messagebox.showerror("Load Error", str(e))
            return
        self.layout = layout
        # Sync UI controls to match the loaded layout so Generate continues from it
        if "table_width_inches" in layout:
            self.controls.table_width_var.set(layout["table_width_inches"])
        if "table_depth_inches" in layout:
            self.controls.table_depth_var.set(layout["table_depth_inches"])
        self.controls.symmetric_var.set(
            layout.get("rotationally_symmetric", False)
        )
        # Sync mission dropdowns
        mission = layout.get("mission")
        if mission and "name" in mission:
            path = find_mission_path(mission["name"])
            if path:
                ed, pk, dep = path
                self.controls.edition_var.set(ed)
                self.controls._on_edition_changed()
                self.controls.pack_var.set(pk)
                self.controls._on_pack_changed()
                self.controls.deployment_var.set(dep)
            else:
                self.controls.deployment_var.set("None")
        else:
            self.controls.deployment_var.set("None")
        self._update_visibility_display()
        self._render()

    def _on_generate(self):
        self._dismiss_popup()
        self._selected_feature_idx = None
        params = self.controls.get_params()
        if not params:
            return
        if params["seed"] is None:
            params["seed"] = random.randint(0, 2**32 - 1)

        # Start fresh if symmetry setting changed
        symmetric = params.get("rotationally_symmetric", False)
        layout_symmetric = self.layout.get("rotationally_symmetric", False)
        if symmetric != layout_symmetric:
            params["initial_layout"] = None
        else:
            params["initial_layout"] = self.layout

        # Auto-select engine based on manifest verification
        if _should_use_rust_engine():
            result_json = _engine_rs.generate_json(json.dumps(params))  # type: ignore
            result = json.loads(result_json)
        else:
            result = generate_json(params)

        self.layout = result["layout"]
        self._update_visibility_display()
        self.history.add_to_history(self.layout)
        self._render()

    def _update_visibility_display(self):
        """Update visibility labels from current layout."""
        vis = self.layout.get("visibility")
        if isinstance(vis, dict) and "overall" in vis:
            overall = vis["overall"]
            if isinstance(overall, dict) and "value" in overall:
                val = overall["value"]
                self.controls.visibility_label.config(
                    text=f"Visibility: {val}%"
                )

                # DZ visibility
                dz_vis = vis.get("dz_visibility")
                if isinstance(dz_vis, dict) and dz_vis:
                    parts = [
                        f"{dz_id}: {data['value']}%"
                        for dz_id, data in dz_vis.items()
                    ]
                    self.controls.dz_vis_label.config(
                        text=f"DZ Vis: {', '.join(parts)}"
                    )
                else:
                    self.controls.dz_vis_label.config(text="")

                # DZ hidden fraction (% of DZ hidden from opposing DZ)
                dz_cross = vis.get("dz_to_dz_visibility")
                if isinstance(dz_cross, dict) and dz_cross:
                    parts = [
                        f"{key}: {data['value']}%"
                        for key, data in dz_cross.items()
                    ]
                    self.controls.dz_to_dz_vis_label.config(
                        text=f"DZ Hidden: {', '.join(parts)}"
                    )
                else:
                    self.controls.dz_to_dz_vis_label.config(text="")

                # Objective hidability
                obj_hide = vis.get("objective_hidability")
                if isinstance(obj_hide, dict) and obj_hide:
                    parts = [
                        f"{dz_id}: {data['value']}%"
                        for dz_id, data in obj_hide.items()
                    ]
                    self.controls.obj_hide_label.config(
                        text=f"Obj Hide: {', '.join(parts)}"
                    )
                else:
                    self.controls.obj_hide_label.config(text="")
                return

        self.controls.visibility_label.config(text="Visibility: --")
        self.controls.dz_vis_label.config(text="")
        self.controls.dz_to_dz_vis_label.config(text="")
        self.controls.obj_hide_label.config(text="")

    def run(self):
        self.root.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
