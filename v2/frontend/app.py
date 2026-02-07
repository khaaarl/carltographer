"""Carltographer terrain layout viewer.

Displays a 2D top-down view of a Warhammer 40k terrain layout
with a control panel for engine parameters.
"""

import math
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

# -- Visual constants --

TABLE_BG = "#2d5a27"  # dark green gaming mat
TABLE_BORDER = "#111111"
CRATE_FILL = "#8b3a1a"  # rusty red
CRATE_OUTLINE = "#000000"
CANVAS_BG = "#1e1e1e"

# -- Sample data --

# Each crate is (x_inches, z_inches, rotation_deg) relative to table center.
# Crate dimensions: 5" wide x 2.5" deep (obstacle terrain).
CRATE_WIDTH = 5.0
CRATE_DEPTH = 2.5

SAMPLE_CRATES = [
    (0, 0, 0),
    (-12, -18, 45),
    (14, 10, 0),
    (-8, 20, 90),
    (16, -14, 30),
    (-15, -4, 0),
    (5, -22, 15),
    (-10, 12, 60),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class BattlefieldRenderer:
    """Renders a terrain layout to a Pillow image."""

    def __init__(self, table_width, table_depth, ppi):
        self.table_width = table_width
        self.table_depth = table_depth
        self.ppi = ppi

    def _to_px(self, x_inches, z_inches):
        """Table coords (center origin) -> pixel coords (top-left origin)."""
        px = (x_inches + self.table_width / 2) * self.ppi
        py = (z_inches + self.table_depth / 2) * self.ppi
        return px, py

    def render(self, crates):
        w = int(self.table_width * self.ppi)
        h = int(self.table_depth * self.ppi)
        img = Image.new("RGB", (w, h), TABLE_BG)
        draw = ImageDraw.Draw(img)

        for cx, cz, rot in crates:
            self._draw_crate(draw, cx, cz, rot)

        # Table border
        draw.rectangle([0, 0, w - 1, h - 1], outline=TABLE_BORDER, width=3)
        return img

    def _draw_crate(self, draw, cx, cz, rot_deg):
        hw = CRATE_WIDTH / 2
        hd = CRATE_DEPTH / 2
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
            px_corners, fill=CRATE_FILL, outline=CRATE_OUTLINE, width=2
        )


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------


class ControlPanel(ttk.Frame):
    """Sidebar with engine parameter controls."""

    def __init__(self, parent, on_table_changed, on_generate):
        super().__init__(parent, padding=10)
        self.on_table_changed = on_table_changed
        self.on_generate = on_generate

        # -- tk variables --
        self.table_width_var = tk.DoubleVar(value=44.0)
        self.table_depth_var = tk.DoubleVar(value=60.0)
        self.seed_var = tk.IntVar(value=42)
        self.num_steps_var = tk.IntVar(value=1000)
        self.symmetric_var = tk.BooleanVar(value=False)
        self.min_gap_var = tk.DoubleVar(value=2.0)
        self.min_edge_gap_var = tk.DoubleVar(value=1.0)

        self._build()

        # Re-render battlefield when table dims change.
        self.table_width_var.trace_add("write", self._dims_changed)
        self.table_depth_var.trace_add("write", self._dims_changed)

    # -- layout --

    def _build(self):
        row = 0

        ttk.Label(self, text="Engine Parameters", font=("", 14, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(0, 12), sticky="w"
        )
        row += 1

        # Table
        row = self._section(row, "Table")
        row = self._field(row, "Width (in):", self.table_width_var)
        row = self._field(row, "Depth (in):", self.table_depth_var)

        # Generation
        row = self._sep(row)
        row = self._section(row, "Generation")
        row = self._field(row, "Seed:", self.seed_var)
        row = self._field(row, "Steps:", self.num_steps_var)
        ttk.Checkbutton(
            self, text="Rotationally symmetric", variable=self.symmetric_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # Spacing
        row = self._sep(row)
        row = self._section(row, "Spacing")
        row = self._field(row, "Feature gap (in):", self.min_gap_var)
        row = self._field(row, "Edge gap (in):", self.min_edge_gap_var)

        # Feature counts placeholder
        row = self._sep(row)
        row = self._section(row, "Feature Counts")
        ttk.Label(self, text="(not yet implemented)", foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        # Generate button
        row = self._sep(row)
        ttk.Button(self, text="Generate", command=self.on_generate).grid(
            row=row, column=0, columnspan=2, pady=10, sticky="ew"
        )

    def _section(self, row, title):
        ttk.Label(self, text=title, font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(8, 4), sticky="w"
        )
        return row + 1

    def _field(self, row, label, var):
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(self, textvariable=var, width=10).grid(
            row=row, column=1, sticky="w", pady=2, padx=(5, 0)
        )
        return row + 1

    def _sep(self, row):
        ttk.Separator(self, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8
        )
        return row + 1

    def _dims_changed(self, *_args):
        try:
            w = self.table_width_var.get()
            d = self.table_depth_var.get()
            if w > 0 and d > 0:
                self.on_table_changed()
        except (tk.TclError, ValueError):
            pass

    def get_params(self):
        """Return current params as a dict matching engine_params schema."""
        try:
            return {
                "seed": self.seed_var.get(),
                "table_width_inches": self.table_width_var.get(),
                "table_depth_inches": self.table_depth_var.get(),
                "rotationally_symmetric": self.symmetric_var.get(),
                "min_feature_gap_inches": self.min_gap_var.get(),
                "min_edge_gap_inches": self.min_edge_gap_var.get(),
                "num_steps": self.num_steps_var.get(),
                "catalog": {"name": "default", "objects": [], "features": []},
            }
        except (tk.TclError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Carltographer")
        self.root.geometry("1100x750")
        self.root.configure(bg=CANVAS_BG)

        style = ttk.Style()
        style.theme_use("clam")

        # Canvas on the left, controls on the right.
        self.canvas = tk.Canvas(self.root, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.controls = ControlPanel(
            self.root,
            on_table_changed=self._render,
            on_generate=self._on_generate,
        )
        self.controls.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)

        self._photo = None  # prevent GC
        self.crates = list(SAMPLE_CRATES)

        self.root.after(50, self._render)
        self.canvas.bind("<Configure>", lambda _e: self._render())

    # -- rendering --

    def _render(self):
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

        renderer = BattlefieldRenderer(tw, td, ppi)
        img = renderer.render(self.crates)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            cw / 2, ch / 2, image=self._photo, anchor="center"
        )

    # -- actions --

    def _on_generate(self):
        params = self.controls.get_params()
        if params:
            print(f"Generate requested: {params}")

    def run(self):
        self.root.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
