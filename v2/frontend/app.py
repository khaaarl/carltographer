"""Tkinter GUI for Carltographer.

This is the main application file — it wires together the engine, terrain
catalogs, and mission data into an interactive desktop tool. The major
classes are:

  * ``BattlefieldRenderer`` — draws the 2D top-down table view on a Tk
    Canvas. Handles rendering of terrain OBBs (with rotation), deployment
    zones, objective markers, grid lines, and the mirror-ghost overlay for
    rotationally symmetric layouts. Also supports click-to-select and
    drag-to-move for manual feature placement.
  * ``ControlPanel`` — the left sidebar with all engine parameters (table
    size, seed, steps, gap constraints, feature count preferences, scoring
    targets, mission selection). Reads widget state into an ``EngineParams``
    dict for the engine, and exposes callbacks for generate/clear/save/load.
  * ``HistoryPanel`` — the right sidebar showing a scrollable log of
    generation results with clickable thumbnail previews.
  * ``App`` — the top-level window that composes the above, manages the
    results bar (visibility/DZ/objective metrics), the "+ Add Terrain"
    button, and the generate-in-background threading.

The engine is called via ``generate_json()`` (either the Python
implementation or the Rust one if available via ``engine_rs``). The Rust
engine is preferred when present; a hash manifest check
(``hash_manifest.py``) warns if the Rust build is stale.

Terrain catalogs come from ``catalogs.py`` and mission/deployment-zone
data from ``missions.py``. Layout persistence (save as PNG with embedded
JSON, or load from PNG/JSON) is handled by ``layout_io.py``.
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
from ..engine.visibility import DZ_EXPANSION_INCHES, _expand_dz_polygons
from ..engine_cmp.hash_manifest import verify_engine_unchanged
from .catalogs import TERRAIN_CATALOGS
from .layout_io import load_layout, save_layout_png
from .missions import EDITIONS, find_mission_path, get_mission

try:
    import engine_rs as _engine_rs

    _HAS_RUST_ENGINE = True
except ImportError:
    _engine_rs = None  # type: ignore
    _HAS_RUST_ENGINE = False


def _enrich_params_for_rust(params: dict) -> dict:
    """Add expanded DZ polygons to params dict for the Rust engine.

    The Rust engine receives precomputed expanded DZ polygons via JSON
    (avoids reimplementing shapely buffer() in Rust).
    """
    mission = params.get("mission")
    if mission and "deployment_zones" in mission:
        for dz in mission["deployment_zones"]:
            if "expanded_polygons" not in dz:
                polys = dz.get("polygons", [])
                polys_tuples = [
                    [(p["x_inches"], p["z_inches"]) for p in poly]
                    for poly in polys
                ]
                expanded = _expand_dz_polygons(
                    polys_tuples, DZ_EXPANSION_INCHES
                )
                dz["expanded_polygons"] = [
                    [{"x_inches": x, "z_inches": z} for x, z in ring]
                    for ring in expanded
                ]
    return params


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
# Tooltip helper
# ---------------------------------------------------------------------------


class Tooltip:
    """Lightweight hover tooltip for any tkinter widget."""

    _DELAY_MS = 400

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self._DELAY_MS, self._show)

    def _cancel(self, _event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 4
        y = self.widget.winfo_rooty()
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            background="#ffffe0",
            foreground="#000000",
            relief="solid",
            borderwidth=1,
            padx=4,
            pady=2,
            wraplength=300,
        )
        label.pack()
        self._tip_window = tw

    def _hide(self):
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


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
        self,
        table_width,
        table_depth,
        ppi,
        objects_by_id,
        mission=None,
        line_scale=1,
    ):
        self.table_width = table_width
        self.table_depth = table_depth
        self.ppi = ppi
        self.objects_by_id = objects_by_id
        self.mission = mission
        self.line_scale = line_scale

    def _lw(self, base_width):
        """Scale a pixel width by the supersample factor."""
        return max(1, round(base_width * self.line_scale))

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
            glw = self._lw(1)
            for ix in range(1, int(self.table_width)):
                px = int(ix * self.ppi)
                draw.line(
                    [(px, 0), (px, h - 1)], fill=NO_MANS_LAND_GRID, width=glw
                )
            for iz in range(1, int(self.table_depth)):
                py = int(iz * self.ppi)
                draw.line(
                    [(0, py), (w - 1, py)], fill=NO_MANS_LAND_GRID, width=glw
                )

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

        # 4c. Padlock icons on locked features
        for pf in placed:
            if pf.get("locked", False):
                tf = pf.get("transform", {})
                cx = tf.get("x_inches", 0.0)
                cz = tf.get("z_inches", 0.0)
                self._draw_padlock(draw, cx, cz)
                if is_symmetric and (cx != 0.0 or cz != 0.0):
                    self._draw_padlock(draw, -cx, -cz)

        # 5. Objective markers (on top of terrain)
        if mission is not None:
            for obj in mission.get("objectives", []):
                self._draw_objective(draw, obj)

        # 6. Table border
        draw.rectangle(
            [0, 0, w - 1, h - 1], outline=TABLE_BORDER, width=self._lw(3)
        )
        return img

    def _draw_zone_aware_grid(self, draw, w, h, dz_polys):
        """Draw grid lines with colors matching the zone they pass through."""
        hw = self.table_width / 2
        hd = self.table_depth / 2
        glw = self._lw(1)

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
                draw.line([(px, py0), (px, py1)], fill=color, width=glw)

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
                draw.line([(px0, py), (px1, py)], fill=color, width=glw)

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
            width=self._lw(2),
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

    def _draw_padlock(self, draw, table_x, table_z):
        """Draw a padlock icon at the given table coordinates."""
        px, py = self._to_px(table_x, table_z)
        # Fixed size ~0.8" in table space
        s = self.ppi * 0.4  # half-size in pixels

        outline_color = "#333333"
        fill_color = "#ffffff"
        lw = max(1, int(s * 0.15))

        # Lock body (filled rectangle)
        bw = s * 0.8  # body width
        bh = s * 0.7  # body height
        body_top = py
        draw.rectangle(
            [px - bw, body_top, px + bw, body_top + bh],
            fill=fill_color,
            outline=outline_color,
            width=lw,
        )

        # Shackle (arc above the body)
        shackle_r = bw * 0.65
        shackle_cy = body_top
        bbox = [
            px - shackle_r,
            shackle_cy - shackle_r,
            px + shackle_r,
            shackle_cy + shackle_r,
        ]
        draw.arc(bbox, start=180, end=360, fill=outline_color, width=lw)

        # Keyhole (small circle in center of body)
        kr = s * 0.15
        kcy = body_top + bh * 0.45
        draw.ellipse(
            [px - kr, kcy - kr, px + kr, kcy + kr],
            fill=outline_color,
        )

    def _draw_dashed_circle(self, draw, cx, cy, radius, dash_count=36):
        """Draw a dashed circle as alternating arc segments."""
        arc_angle = 360.0 / dash_count
        for i in range(0, dash_count, 2):
            start = i * arc_angle
            end = start + arc_angle
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.arc(
                bbox,
                start=start,
                end=end,
                fill=RANGE_DASH_COLOR,
                width=self._lw(2),
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
            width=self._lw(3),
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
            width=self._lw(2) if outline else 0,
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
        self.num_steps_var = tk.IntVar(value=10)
        self.symmetric_var = tk.BooleanVar(value=False)
        self.rotation_granularity_var = tk.StringVar(value="15")
        self.min_gap_var = tk.DoubleVar(value=5.2)
        self.min_edge_gap_var = tk.DoubleVar(value=0.0)
        self.min_all_gap_var = tk.DoubleVar(value=0.0)
        self.min_all_edge_gap_var = tk.DoubleVar(value=0.0)
        self.min_crates_var = tk.IntVar(value=0)
        self.max_crates_var = tk.IntVar(value=99)
        self.min_ruins_var = tk.IntVar(value=0)
        self.max_ruins_var = tk.IntVar(value=99)

        # Scoring target variables
        self.overall_vis_target_var = tk.StringVar(value="30")
        self.overall_vis_weight_var = tk.StringVar(value="1.0")
        self.dz_hide_target_var = tk.StringVar(value="70")
        self.dz_hide_weight_var = tk.StringVar(value="5.0")
        self.obj_hide_target_var = tk.StringVar(value="50")
        self.obj_hide_weight_var = tk.StringVar(value="5.0")

        # Catalog selection variable
        self.catalog_var = tk.StringVar(value="Omnium Gatherum")

        # Mission selection variables
        self.edition_var = tk.StringVar(value="")
        self.pack_var = tk.StringVar(value="")
        self.deployment_var = tk.StringVar(value="None")
        self._initializing = True  # suppress callbacks during init

        # Combo widgets (set during _build via _combo helper)
        self.catalog_combo: ttk.Combobox = None
        self.edition_combo: ttk.Combobox = None
        self.pack_combo: ttk.Combobox = None
        self.deployment_combo: ttk.Combobox = None
        self.rotation_granularity_combo: ttk.Combobox = None

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

        # --- Left column: Terrain, Mission, Table, Generation, Buttons ---
        row = 0
        row = self._section(left, row, "Terrain")
        row = self._combo(
            left,
            row,
            "Catalog:",
            self.catalog_var,
            "catalog_combo",
            tooltip="Terrain collection to draw pieces from",
        )
        self.catalog_combo["values"] = list(TERRAIN_CATALOGS.keys())

        row = self._sep(left, row)
        row = self._section(left, row, "Mission")
        row = self._combo(
            left,
            row,
            "Edition:",
            self.edition_var,
            "edition_combo",
            tooltip="Rules edition for mission deployment zones",
        )
        row = self._combo(
            left,
            row,
            "Pack:",
            self.pack_var,
            "pack_combo",
            tooltip="Mission pack within the selected edition",
        )
        row = self._combo(
            left,
            row,
            "Deployment:",
            self.deployment_var,
            "deployment_combo",
            tooltip="Deployment zone layout; affects scoring targets",
        )

        # Initialize mission dropdowns
        self._init_mission_combos()

        row = self._sep(left, row)
        row = self._section(left, row, "Table")
        row = self._field(
            left,
            row,
            "Width (in):",
            self.table_width_var,
            tooltip="Table width in inches (standard: 60)",
        )
        row = self._field(
            left,
            row,
            "Depth (in):",
            self.table_depth_var,
            tooltip="Table depth in inches (standard: 44)",
        )

        row = self._sep(left, row)
        row = self._section(left, row, "Generation")
        row = self._field(
            left,
            row,
            "Seed:",
            self.seed_var,
            tooltip="Random seed for reproducibility; leave blank for random",
        )
        row = self._field(
            left,
            row,
            "Steps:",
            self.num_steps_var,
            tooltip="Number of optimization iterations (more = better but slower)",
        )
        cb = ttk.Checkbutton(
            left, text="Rotationally symmetric", variable=self.symmetric_var
        )
        cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        Tooltip(cb, "Mirror all off-center terrain 180° for balanced layouts")
        row += 1
        row = self._combo(
            left,
            row,
            "Rotation:",
            self.rotation_granularity_var,
            "rotation_granularity_combo",
            tooltip="Rotation snap angle in degrees for terrain placement",
        )
        self.rotation_granularity_combo["values"] = [
            "90",
            "45",
            "15",
        ]

        row = self._sep(left, row)
        btn_frame = ttk.Frame(left)
        btn_frame.grid(
            row=row, column=0, columnspan=2, pady=(10, 10), sticky="ew"
        )
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        btn = ttk.Button(btn_frame, text="Generate", command=self.on_generate)
        btn.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 2))
        Tooltip(btn, "Run the engine to produce or improve a layout")

        btn = ttk.Button(btn_frame, text="Clear", command=self.on_clear)
        btn.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 2))
        Tooltip(btn, "Remove all terrain and start with an empty table")

        btn = ttk.Button(btn_frame, text="Save", command=self.on_save)
        btn.grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=(2, 0))
        Tooltip(btn, "Export layout as a PNG with embedded data")

        btn = ttk.Button(btn_frame, text="Load", command=self.on_load)
        btn.grid(row=1, column=1, sticky="ew", padx=(2, 0), pady=(2, 0))
        Tooltip(btn, "Import a layout from a saved PNG or JSON file")

        # --- Right column: Spacing, Feature Counts, Results ---
        row = 0
        row = self._section(right, row, "Spacing (tall terrain)")
        row = self._field(
            right,
            row,
            "Feature gap (in):",
            self.min_gap_var,
            tooltip='Min gap between tall terrain (height >= 1")',
        )
        row = self._field(
            right,
            row,
            "Edge gap (in):",
            self.min_edge_gap_var,
            tooltip="Min gap from tall terrain to table edges",
        )

        row = self._sep(right, row)
        row = self._section(right, row, "Spacing (all features)")
        row = self._field(
            right,
            row,
            "Feature gap (in):",
            self.min_all_gap_var,
            tooltip="Min gap between any terrain pieces",
        )
        row = self._field(
            right,
            row,
            "Edge gap (in):",
            self.min_all_edge_gap_var,
            tooltip="Min gap from any terrain to table edges",
        )

        row = self._sep(right, row)
        row = self._section(right, row, "Feature Counts")
        row = self._field(
            right,
            row,
            "Min obstacles:",
            self.min_crates_var,
            tooltip="Target minimum obstacle count (soft constraint)",
        )
        row = self._field(
            right,
            row,
            "Max obstacles:",
            self.max_crates_var,
            tooltip="Target maximum obstacle count (soft constraint)",
        )
        row = self._field(
            right,
            row,
            "Min ruins:",
            self.min_ruins_var,
            tooltip="Target minimum ruin count (soft constraint)",
        )
        row = self._field(
            right,
            row,
            "Max ruins:",
            self.max_ruins_var,
            tooltip="Target maximum ruin count (soft constraint)",
        )

        row = self._sep(right, row)
        row = self._section(right, row, "Scoring Targets")
        row = self._field(
            right,
            row,
            "Overall vis %:",
            self.overall_vis_target_var,
            tooltip="Target line-of-sight visibility percentage",
        )
        row = self._field(
            right,
            row,
            "  weight:",
            self.overall_vis_weight_var,
            tooltip="How strongly to optimize toward the visibility target",
        )
        row = self._field(
            right,
            row,
            "DZ hide %:",
            self.dz_hide_target_var,
            tooltip="Target % of each deployment zone hidden from opposing DZ",
        )
        row = self._field(
            right,
            row,
            "  weight:",
            self.dz_hide_weight_var,
            tooltip="How strongly to optimize toward the DZ hide target",
        )
        row = self._field(
            right,
            row,
            "Obj hide %:",
            self.obj_hide_target_var,
            tooltip="Target % of objectives hidden from deployment zones",
        )
        row = self._field(
            right,
            row,
            "  weight:",
            self.obj_hide_weight_var,
            tooltip="How strongly to optimize toward the objective hide target",
        )

    def _section(self, parent, row, title):
        ttk.Label(parent, text=title, font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(8, 4), sticky="w"
        )
        return row + 1

    def _field(self, parent, row, label, var, tooltip=None):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky="w", pady=2, padx=(5, 0))
        if tooltip:
            Tooltip(lbl, tooltip)
            Tooltip(entry, tooltip)
        return row + 1

    def _combo(self, parent, row, label, var, attr_name, tooltip=None):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", pady=2)
        combo = ttk.Combobox(
            parent, textvariable=var, width=28, state="readonly"
        )
        combo.grid(row=row, column=1, sticky="w", pady=2, padx=(5, 0))
        setattr(self, attr_name, combo)
        if tooltip:
            Tooltip(lbl, tooltip)
            Tooltip(combo, tooltip)
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
        if not self._initializing:
            self.on_table_changed()

    def _resolve_mission(self):
        """Compute mission dict for current deployment + table dims."""
        dep = self.deployment_var.get()
        if dep in ("None", ""):
            return None
        edition = self.edition_var.get()
        pack = self.pack_var.get()
        try:
            tw = self.table_width_var.get()
            td = self.table_depth_var.get()
        except (tk.TclError, ValueError):
            tw, td = 60.0, 44.0
        return get_mission(edition, pack, dep, tw, td)

    @property
    def selected_mission(self):
        """Current mission dict resolved from deployment + table dimensions."""
        return self._resolve_mission()

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
            # Parse optional float fields
            def parse_float(s):
                s = s.strip()
                return float(s) if s else None

            min_crates = self.min_crates_var.get()
            max_crates = self.max_crates_var.get()
            min_ruins = self.min_ruins_var.get()
            max_ruins = self.max_ruins_var.get()
            min_gap = self.min_gap_var.get()
            min_edge_gap = self.min_edge_gap_var.get()
            min_all_gap = self.min_all_gap_var.get()
            min_all_edge_gap = self.min_all_edge_gap_var.get()
            rotation_granularity = float(self.rotation_granularity_var.get())

            # Build feature count preferences
            feature_count_prefs = [
                {
                    "feature_type": "obstacle",
                    "min": min_crates,
                    "max": max_crates,
                },
                {
                    "feature_type": "obscuring",
                    "min": min_ruins,
                    "max": max_ruins,
                },
            ]

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
                "catalog": TERRAIN_CATALOGS[self.catalog_var.get()],
                "feature_count_preferences": feature_count_prefs,
                "num_replicas": num_replicas,
                "rotation_granularity_deg": rotation_granularity,
            }

            # Only include gap parameters if > 0
            if min_gap > 0:
                params["min_feature_gap_inches"] = min_gap
            if min_edge_gap > 0:
                params["min_edge_gap_inches"] = min_edge_gap
            if min_all_gap > 0:
                params["min_all_feature_gap_inches"] = min_all_gap
            if min_all_edge_gap > 0:
                params["min_all_edge_gap_inches"] = min_all_edge_gap

            # Include mission if selected
            if self.selected_mission is not None:
                params["mission"] = self.selected_mission

            # Build scoring targets if any target is set
            overall_vis_t = parse_float(self.overall_vis_target_var.get())
            overall_vis_w = parse_float(self.overall_vis_weight_var.get())
            dz_hide_t = parse_float(self.dz_hide_target_var.get())
            dz_hide_w = parse_float(self.dz_hide_weight_var.get())
            obj_hide_t = parse_float(self.obj_hide_target_var.get())
            obj_hide_w = parse_float(self.obj_hide_weight_var.get())

            has_any_target = any(
                t is not None for t in [overall_vis_t, dz_hide_t, obj_hide_t]
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
                if dz_hide_t is not None:
                    scoring_targets["dz_hideability_target"] = dz_hide_t
                    if dz_hide_w is not None:
                        scoring_targets["dz_hideability_weight"] = dz_hide_w
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
        self.history_listbox: tk.Listbox | None = None
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
        assert self.history_listbox is not None
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
        assert self.history_listbox is not None
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

        # Left panel: canvas on top, results bar on bottom.
        self.left_panel = ttk.Frame(self.root)
        self.left_panel.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.canvas = tk.Canvas(
            self.left_panel, bg=CANVAS_BG, highlightthickness=0
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Results bar below canvas (vertical stack, fixed structure)
        self._results_frame = ttk.Frame(self.left_panel, padding=(5, 2))
        self._results_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.visibility_label = ttk.Label(
            self._results_frame, text="Visibility: --"
        )
        self.visibility_label.grid(row=0, column=0, sticky="w", pady=1)
        Tooltip(
            self.visibility_label,
            "Overall line-of-sight visibility across the table",
        )
        self.dz_hide_label = ttk.Label(self._results_frame, text="")
        self.dz_hide_label.grid(row=1, column=0, sticky="w", pady=1)
        Tooltip(
            self.dz_hide_label,
            "% of each deployment zone hidden from the opposing DZ",
        )
        self.obj_hide_label = ttk.Label(self._results_frame, text="")
        self.obj_hide_label.grid(row=2, column=0, sticky="w", pady=1)
        Tooltip(
            self.obj_hide_label,
            "% of objectives hidden from deployment zones",
        )

        # Right panel for controls and history
        self.right_panel = ttk.Frame(self.root)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, padx=(0, 5))

        self.controls = ControlPanel(
            self.right_panel,
            on_table_changed=self._on_table_changed,
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
        self.objects_by_id = _build_object_index(
            TERRAIN_CATALOGS["Omnium Gatherum"]
        )

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

        # Debounced visibility recompute timer
        self._vis_recompute_after_id = None

        # Feature grid state
        self._feature_grid_visible = False
        self._feature_grid_frame = None
        self._feature_grid_window_id = None
        self._feature_thumbnails = []  # prevent GC of PhotoImage refs

        # Add Terrain button (placed on canvas by _place_add_terrain_button)
        self._add_terrain_btn = tk.Button(
            self.canvas,
            text="+ Add Terrain",
            command=self._on_add_terrain,
            bg="#336633",
            fg="white",
            activebackground="#448844",
            activeforeground="white",
            padx=8,
            pady=4,
            relief="raised",
            borderwidth=2,
        )
        self._add_terrain_btn_window_id = None
        Tooltip(
            self._add_terrain_btn,
            "Manually place a terrain piece from the catalog",
        )

        self.root.after(50, self._render)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Snap features when rotation granularity changes
        self.controls.rotation_granularity_var.trace_add(
            "write", self._on_granularity_changed
        )

        # Dismiss feature grid on catalog change and rebuild object index
        self.controls.catalog_var.trace_add("write", self._on_catalog_changed)

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

    # -- table change handling --

    def _on_table_changed(self):
        """Called when table dims or deployment change. Re-render and schedule visibility recompute."""
        self._render()
        self._schedule_visibility_recompute()

    def _schedule_visibility_recompute(self):
        """Debounced: schedule a visibility recompute after 500ms of inactivity."""
        if self._vis_recompute_after_id is not None:
            self.root.after_cancel(self._vis_recompute_after_id)
            self._vis_recompute_after_id = None
        if not self.layout.get("placed_features"):
            return
        self._vis_recompute_after_id = self.root.after(
            500, self._recompute_visibility
        )

    def _recompute_visibility(self):
        """Re-run engine with num_steps=0 to get fresh visibility for current layout + mission."""
        self._vis_recompute_after_id = None
        params = self.controls.get_params()
        if params is None:
            return
        params["num_steps"] = 0
        params["seed"] = 1
        params["initial_layout"] = self.layout

        if _should_use_rust_engine():
            result_json = _engine_rs.generate_json(  # type: ignore[union-attr]
                json.dumps(_enrich_params_for_rust(params))
            )
            result = json.loads(result_json)
        else:
            result = generate_json(params)

        self.layout["visibility"] = result["layout"].get("visibility")
        self._update_visibility_display()
        self._render()

    # -- rendering --

    def _render(self):
        if self._move_mode:
            return
        if self._feature_grid_visible:
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

        # Render at 4x resolution for antialiasing, then downsample with
        # LANCZOS.  This gives 16x SSAA (4x per axis) which eliminates
        # the jagged polygon edges from PIL's non-antialiased drawing.
        supersample = 4
        render_ppi = ppi * supersample
        renderer = BattlefieldRenderer(
            tw,
            td,
            render_ppi,
            self.objects_by_id,
            mission,
            line_scale=supersample,
        )
        img = renderer.render(
            self.layout, highlight_index=self._selected_feature_idx
        )
        img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            cw / 2, ch / 2, image=self._photo, anchor="center"
        )
        self._place_add_terrain_button()

    def _place_add_terrain_button(self):
        """Place the '+ Add Terrain' button at bottom-left of the canvas."""
        if self._add_terrain_btn_window_id is not None:
            self.canvas.delete(self._add_terrain_btn_window_id)
            self._add_terrain_btn_window_id = None
        self._add_terrain_btn_window_id = self.canvas.create_window(
            10,
            self.canvas.winfo_height() - 10,
            window=self._add_terrain_btn,
            anchor="sw",
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
        min_all_gap = params.get("min_all_feature_gap_inches")
        min_all_edge_gap = params.get("min_all_edge_gap_inches")
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
            min_all_feature_gap=min_all_gap,
            min_all_edge_gap=min_all_edge_gap,
        )

    # -- selection & popup --

    def _on_canvas_click(self, event):
        """Handle click on canvas: select/deselect features."""
        if self._move_mode:
            return
        if self._feature_grid_visible:
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

        # Lock/Unlock button
        placed = list(self.layout.get("placed_features", []))  # type: ignore[arg-type]
        idx = self._selected_feature_idx
        is_locked = False
        if idx is not None and 0 <= idx < len(placed):
            is_locked = placed[idx].get("locked", False)
        lock_label = "Unlock" if is_locked else "Lock"
        lock_btn = tk.Button(
            frame,
            text=lock_label,
            command=self._on_lock_selected,
            bg="#cc9900",
            fg="white",
            activebackground="#ddaa22",
            activeforeground="white",
            padx=8,
            pady=2,
        )
        lock_btn.pack(side=tk.LEFT, padx=(2, 2), pady=4)

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
        popup_w = 370  # approximate width
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
                result_json = _engine_rs.generate_json(  # type: ignore[union-attr]
                    json.dumps(_enrich_params_for_rust(params))
                )
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

    # -- lock action --

    def _on_lock_selected(self):
        """Toggle locked state on the selected feature."""
        idx = self._selected_feature_idx
        if idx is None:
            return

        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        if idx < 0 or idx >= len(placed):
            return

        placed[idx]["locked"] = not placed[idx].get("locked", False)
        self._selected_feature_idx = None
        self._dismiss_popup()
        self._render()

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

        supersample = 4
        render_ppi = ppi * supersample
        renderer = BattlefieldRenderer(
            tw,
            td,
            render_ppi,
            self.objects_by_id,
            mission,
            line_scale=supersample,
        )
        img = renderer.render(temp_layout)
        img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)

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

    def _get_rotation_granularity(self):
        """Return the current rotation granularity from the UI dropdown."""
        try:
            return float(self.controls.rotation_granularity_var.get())
        except (ValueError, AttributeError):
            return 15.0

    def _on_granularity_changed(self, *args):
        """Snap all existing features to the new rotation granularity."""
        if self.layout is None:
            return
        granularity = self._get_rotation_granularity()
        placed = self.layout.get("placed_features")
        if not isinstance(placed, list):
            return
        changed = False
        for pf in placed:
            t = pf.get("transform", {})
            old_rot = t.get("rotation_deg", 0.0)
            new_rot = round(old_rot / granularity) * granularity
            if abs(new_rot - old_rot) > 1e-6:
                t["rotation_deg"] = new_rot
                changed = True
        if changed:
            self._render()

    def _on_move_rotate_cw(self, event=None):
        self._rotate_move_feature(self._get_rotation_granularity())

    def _on_move_rotate_ccw(self, event=None):
        self._rotate_move_feature(-self._get_rotation_granularity())

    def _on_move_scroll(self, event):
        """Handle mouse wheel rotation (Windows/macOS)."""
        if not self._move_mode:
            return
        g = self._get_rotation_granularity()
        if event.delta > 0:
            self._rotate_move_feature(g)
        elif event.delta < 0:
            self._rotate_move_feature(-g)

    def _on_move_scroll_up(self, event):
        """Handle scroll up (Linux Button-4)."""
        self._rotate_move_feature(self._get_rotation_granularity())

    def _on_move_scroll_down(self, event):
        """Handle scroll down (Linux Button-5)."""
        self._rotate_move_feature(-self._get_rotation_granularity())

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

    # -- add terrain / feature grid --

    def _on_add_terrain(self):
        """Open the feature selection grid overlay."""
        if self._move_mode or self._feature_grid_visible:
            return
        self._show_feature_grid()

    def _on_catalog_changed(self, *_args):
        """Handle catalog dropdown change: dismiss grid and rebuild object index."""
        if self._feature_grid_visible:
            self._dismiss_feature_grid()
        catalog_name = self.controls.catalog_var.get()
        if catalog_name in TERRAIN_CATALOGS:
            self.objects_by_id = _build_object_index(
                TERRAIN_CATALOGS[catalog_name]
            )

    def _show_feature_grid(self):
        """Show the feature selection grid overlay on the canvas."""
        catalog_name = self.controls.catalog_var.get()
        if catalog_name not in TERRAIN_CATALOGS:
            return
        catalog = TERRAIN_CATALOGS[catalog_name]
        features = catalog.get("features", [])
        if not features:
            return

        self._feature_grid_visible = True
        self._feature_thumbnails = []

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Outer frame with dark background
        outer = tk.Frame(
            self.canvas, bg="#2a2a2a", relief="raised", borderwidth=2
        )

        # Title bar
        title_bar = tk.Frame(outer, bg="#333333")
        title_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            title_bar,
            text=f"Add Terrain — {catalog_name}",
            bg="#333333",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT)
        tk.Button(
            title_bar,
            text="Cancel",
            command=self._dismiss_feature_grid,
            bg="#cc3333",
            fg="white",
            activebackground="#ff4444",
            activeforeground="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=8, pady=4)

        # Scrollable inner area
        scroll_frame = tk.Frame(outer, bg="#2a2a2a")
        scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scroll_canvas = tk.Canvas(
            scroll_frame, bg="#2a2a2a", highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            scroll_frame, orient="vertical", command=scroll_canvas.yview
        )
        inner = tk.Frame(scroll_canvas, bg="#2a2a2a")

        inner.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(
                scrollregion=scroll_canvas.bbox("all")
            ),
        )
        scroll_canvas.create_window((0, 0), window=inner, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind mousewheel scrolling
        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(-1 * (event.delta // 120), "units")

        def _on_scroll_up(_event):
            scroll_canvas.yview_scroll(-3, "units")

        def _on_scroll_down(_event):
            scroll_canvas.yview_scroll(3, "units")

        scroll_canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_canvas.bind("<Button-4>", _on_scroll_up)
        scroll_canvas.bind("<Button-5>", _on_scroll_down)

        # Build grid cells — 3 columns
        thumb_size = 120
        cols = 3
        for i, entry in enumerate(features):
            feat = entry["item"]
            row = i // cols
            col = i % cols

            cell = tk.Frame(
                inner, bg="#3a3a3a", relief="groove", borderwidth=1
            )
            cell.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

            # Thumbnail
            thumb = self._render_feature_thumbnail(feat, thumb_size)
            if thumb is not None:
                self._feature_thumbnails.append(thumb)
                thumb_label = tk.Label(cell, image=thumb, bg="#3a3a3a")
                thumb_label.pack(padx=4, pady=(4, 2))
                thumb_label.bind(
                    "<Button-1>",
                    lambda _e, f=feat: self._on_feature_selected(f),
                )
            else:
                tk.Label(
                    cell,
                    text="[no preview]",
                    bg="#3a3a3a",
                    fg="#888888",
                    width=thumb_size // 8,
                    height=thumb_size // 20,
                ).pack(padx=4, pady=(4, 2))

            # Name + type label
            feat_id = feat.get("id", "?")
            feat_type = feat.get("feature_type", "")
            tk.Label(
                cell,
                text=f"{feat_id}\n({feat_type})",
                bg="#3a3a3a",
                fg="white",
                font=("TkDefaultFont", 9),
                justify="center",
            ).pack(padx=4, pady=2)

            # Place button
            tk.Button(
                cell,
                text="Place",
                command=lambda f=feat: self._on_feature_selected(f),
                bg="#336699",
                fg="white",
                activebackground="#4488bb",
                activeforeground="white",
                padx=6,
                pady=1,
            ).pack(padx=4, pady=(2, 4))

        for c in range(cols):
            inner.columnconfigure(c, weight=1)

        # Size the overlay to ~80% of canvas
        overlay_w = int(cw * 0.8)
        overlay_h = int(ch * 0.8)
        outer.configure(width=overlay_w, height=overlay_h)

        x = (cw - overlay_w) // 2
        y = (ch - overlay_h) // 2

        win_id = self.canvas.create_window(
            x, y, window=outer, anchor="nw", width=overlay_w, height=overlay_h
        )
        self._feature_grid_frame = outer
        self._feature_grid_window_id = win_id

    def _render_feature_thumbnail(self, feature_dict, size):
        """Render a PIL thumbnail of a feature at its local origin.

        Returns an ImageTk.PhotoImage, or None if no shapes found.
        """
        # Collect all shape corners in local feature space
        all_corners = []
        draw_commands = []  # (polygon_corners, fill, outline)

        for comp in feature_dict.get("components", []):
            obj = self.objects_by_id.get(comp.get("object_id"))
            if obj is None:
                continue
            comp_tf = _get_tf(comp.get("transform"))
            fill = obj.get("fill_color") or DEFAULT_FILL
            outline = obj.get("outline_color")

            for shape in obj.get("shapes", []):
                offset_tf = _get_tf(shape.get("offset"))
                combined = _compose(comp_tf, offset_tf)
                cx, cz, rot_deg = combined
                hw = shape["width_inches"] / 2
                hd = shape["depth_inches"] / 2
                local_corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]

                rad = math.radians(rot_deg)
                cos_r = math.cos(rad)
                sin_r = math.sin(rad)

                world_corners = []
                for lx, lz in local_corners:
                    wx = cx + lx * cos_r - lz * sin_r
                    wz = cz + lx * sin_r + lz * cos_r
                    world_corners.append((wx, wz))
                    all_corners.append((wx, wz))

                draw_commands.append((world_corners, fill, outline))

        if not all_corners:
            return None

        # Compute bounding box
        xs = [c[0] for c in all_corners]
        zs = [c[1] for c in all_corners]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        span_x = max_x - min_x
        span_z = max_z - min_z
        if span_x < 0.01:
            span_x = 1.0
        if span_z < 0.01:
            span_z = 1.0

        # Auto-scale to fit thumbnail with padding
        padding = 8
        draw_size = size - 2 * padding
        scale = min(draw_size / span_x, draw_size / span_z)
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        img = Image.new("RGB", (size, size), "#3a3a3a")
        draw = ImageDraw.Draw(img)

        for corners, fill, outline in draw_commands:
            px_corners = []
            for wx, wz in corners:
                px = padding + (wx - center_x) * scale + draw_size / 2
                py = padding + (wz - center_z) * scale + draw_size / 2
                px_corners.append((px, py))
            draw.polygon(px_corners, fill=fill, outline=outline, width=1)

        return ImageTk.PhotoImage(img)

    def _dismiss_feature_grid(self):
        """Remove the feature grid overlay from canvas."""
        if self._feature_grid_window_id is not None:
            self.canvas.delete(self._feature_grid_window_id)
            self._feature_grid_window_id = None
        if self._feature_grid_frame is not None:
            self._feature_grid_frame.destroy()
            self._feature_grid_frame = None
        self._feature_grid_visible = False
        self._feature_thumbnails = []

    def _on_feature_selected(self, feature_dict):
        """Bridge from feature grid to the existing placement pipeline."""
        self._dismiss_feature_grid()

        # Construct a placed_feature dict at origin
        pf_dict = {
            "feature": copy.deepcopy(feature_dict),
            "transform": {
                "x_inches": 0.0,
                "z_inches": 0.0,
                "rotation_deg": 0.0,
            },
        }

        # Ensure referenced objects are in self.objects_by_id
        for comp in feature_dict.get("components", []):
            obj_id = comp.get("object_id")
            if obj_id and obj_id not in self.objects_by_id:
                # Rebuild from current catalog
                catalog_name = self.controls.catalog_var.get()
                if catalog_name in TERRAIN_CATALOGS:
                    self.objects_by_id = _build_object_index(
                        TERRAIN_CATALOGS[catalog_name]
                    )
                break

        # Append to placed_features
        placed: list = self.layout["placed_features"]  # type: ignore[assignment]
        placed.append(pf_dict)
        new_idx = len(placed) - 1

        # Set up move/copy mode identically to _on_copy_selected
        self._move_original_pf_dict = None  # no original to revert to
        self._move_feature_idx = new_idx
        self._move_mode = True
        self._move_is_copy = True  # so cancel removes the feature
        self._move_last_valid_pos = None  # no valid position yet
        self._move_current_rotation = 0.0

        self._dismiss_popup()
        self._selected_feature_idx = None
        self._selected_is_mirror = False

        self._move_typed_features, self._move_typed_objects = (
            self._build_typed_validation_context(new_idx)
        )

        self._render_move_base()
        self._create_move_overlay()
        self._bind_move_events()

    # -- canvas resize --

    def _on_canvas_configure(self, _event):
        """Handle canvas resize: dismiss popup and re-render."""
        if self._move_mode:
            self._on_move_cancel()
            return
        if self._feature_grid_visible:
            self._dismiss_feature_grid()
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
        supersample = 4
        render_ppi = ppi * supersample
        renderer = BattlefieldRenderer(
            tw,
            td,
            render_ppi,
            self.objects_by_id,
            mission,
            line_scale=supersample,
        )
        img = renderer.render(self.layout)
        img = img.resize(
            (int(tw * ppi), int(td * ppi)), Image.Resampling.LANCZOS
        )
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
            result_json = _engine_rs.generate_json(  # type: ignore[union-attr]
                json.dumps(_enrich_params_for_rust(params))
            )
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
                self.visibility_label.config(text=f"Visibility: {val}%")

                # DZ hideability
                dz_hide = vis.get("dz_hideability")
                if isinstance(dz_hide, dict) and dz_hide:
                    parts = [
                        f"{dz_id}: {data['value']}%"
                        for dz_id, data in dz_hide.items()
                    ]
                    self.dz_hide_label.config(
                        text=f"DZ Hide: {', '.join(parts)}"
                    )
                else:
                    self.dz_hide_label.config(text="")

                # Objective hidability
                obj_hide = vis.get("objective_hidability")
                if isinstance(obj_hide, dict) and obj_hide:
                    parts = [
                        f"{dz_id}: {data['value']}%"
                        for dz_id, data in obj_hide.items()
                    ]
                    self.obj_hide_label.config(
                        text=f"Obj Hide: {', '.join(parts)}"
                    )
                else:
                    self.obj_hide_label.config(text="")
                return

        self.visibility_label.config(text="Visibility: --")
        self.dz_hide_label.config(text="")
        self.obj_hide_label.config(text="")

    def run(self):
        self.root.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
