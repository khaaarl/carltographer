"""Terrain catalog definitions for Warhammer 40k terrain pieces.

Pure data module with no UI dependencies (no tkinter), so it can be imported
by headless scripts (e.g. hyperparameter tuning) as well as the GUI.

Defines:
  - Terrain objects: geometric shapes with dimensions, offsets, colors, and tags.
    Each object dict has an "id", "name", "shapes" list, "tags", and colors.
  - Terrain features: logical groupings of objects that form a playable terrain
    piece (e.g. a ruin = base + walls). Each feature has a "feature_type"
    (obstacle or obscuring) used for feature-count preferences.
  - TERRAIN_CATALOGS: dict mapping catalog name -> {objects, features} with
    per-item quantities (None = unlimited).

Catalog contents:
  - "WTC Set": Official WTC tournament terrain (crates, 3-storey ruins, short
    ruins). Fixed quantities matching the physical set.
  - "GW Misc": Games Workshop miscellaneous ruins (flat bases, L/U/J-shaped
    wall configurations on larger bases). Unlimited quantities.
  - "Omnium Gatherum": Union of WTC + GW Misc + polygon terrain, all unlimited.
"""

import math

# ---------------------------------------------------------------------------
# WTC terrain objects & features
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
    "tags": ["obstacle"],
}

WTC_RUINS_BASE_TALL = {
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
    "fill_color": "#B8D4E8",
    "outline_color": "#000000",
}

WTC_RUINS_BASE_SHORT = {
    "id": "ruins_short",
    "name": "Short Ruins Base",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 12.0,
            "depth_inches": 6.0,
            "height_inches": 0.0,
        }
    ],
    "tags": ["ruins"],
    "fill_color": "#E8B8B8",
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

WTC_SHORT_WALLS = {
    "id": "wtc_short_walls",
    "name": "WTC Short Walls",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 9.0,
            "depth_inches": 0.1,
            "height_inches": 3.0,
            "offset": {"x_inches": 0.65, "z_inches": -2.15},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.1,
            "depth_inches": 5.0,
            "height_inches": 3.0,
            "offset": {"x_inches": 5.15, "z_inches": 0.35},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.6,
            "depth_inches": 0.1,
            "height_inches": 3.0,
            "offset": {"x_inches": 5.5, "z_inches": -2.15},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.1,
            "depth_inches": 0.6,
            "height_inches": 3.0,
            "offset": {"x_inches": 5.15, "z_inches": -2.5},
        },
    ],
    "tags": ["ruins", "wtc"],
    "fill_color": "#888888",
    "outline_color": "#888888",
}

WTC_THREE_STOREY_FEATURE = {
    "id": "wtc_three_storey",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "ruins"},
        {"object_id": "wtc_three_storey_walls"},
    ],
    "tags": ["ruins", "obscuring", "wtc"],
}

WTC_SHORT_FEATURE = {
    "id": "wtc_short",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "ruins_short"},
        {"object_id": "wtc_short_walls"},
    ],
    "tags": ["ruins", "obscuring", "wtc"],
}

# ---------------------------------------------------------------------------
# GW Misc terrain objects & features
# ---------------------------------------------------------------------------

# Bases (flat footprints, height=0)

GW_FLAT_RUIN_BASE = {
    "id": "gw_flat_ruin_base",
    "name": "Flat Ruin Base (6x4)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 6.0,
            "depth_inches": 4.0,
            "height_inches": 0.0,
        }
    ],
    "tags": ["ruins"],
    "fill_color": "#C8D8C0",
    "outline_color": "#000000",
}

GW_RUIN_BASE_12X6 = {
    "id": "gw_ruin_base_12x6",
    "name": "Ruin Base (12x6)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 12.0,
            "depth_inches": 6.0,
            "height_inches": 0.0,
        }
    ],
    "tags": ["ruins"],
    "fill_color": "#D4C8A0",
    "outline_color": "#000000",
}

GW_RUIN_BASE_10X5 = {
    "id": "gw_ruin_base_10x5",
    "name": "Ruin Base (10x5)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 10.0,
            "depth_inches": 5.0,
            "height_inches": 0.0,
        }
    ],
    "tags": ["ruins"],
    "fill_color": "#C0C8D8",
    "outline_color": "#000000",
}

# Wall objects (tall shapes that block LOS)

# L-shaped walls (4"x8") for 12x6 base — L in bottom-left corner
# Base spans x:[-6,6], z:[-3,3]. Walls 0.2" thick, 0.1" from base edges.
GW_TALL_L_4X8_WALLS = {
    "id": "gw_tall_L_4x8",
    "name": "Tall L Walls (4x8)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 8.0,
            "depth_inches": 0.2,
            "height_inches": 5.0,
            "offset": {"x_inches": -1.9, "z_inches": -2.8},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.2,
            "depth_inches": 4.0,
            "height_inches": 5.0,
            "offset": {"x_inches": -5.8, "z_inches": -0.9},
        },
    ],
    "tags": ["ruins"],
    "fill_color": "#444444",
    "outline_color": "#000000",
}

# U-shaped walls (8" wide, 4" deep) for 12x6 base
# Bar 0.1" from -z long edge, centered (2" margin on each side).
# Base spans x:[-6,6], z:[-3,3]. Walls 0.2" thick.
GW_TALL_U_8X4_WALLS = {
    "id": "gw_tall_U_8x4",
    "name": "Tall U Walls (8x4)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 8.0,
            "depth_inches": 0.2,
            "height_inches": 5.0,
            "offset": {"x_inches": 0.0, "z_inches": -2.8},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.2,
            "depth_inches": 4.0,
            "height_inches": 5.0,
            "offset": {"x_inches": -3.9, "z_inches": -0.9},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.2,
            "depth_inches": 4.0,
            "height_inches": 5.0,
            "offset": {"x_inches": 3.9, "z_inches": -0.9},
        },
    ],
    "tags": ["ruins"],
    "fill_color": "#444444",
    "outline_color": "#000000",
}

# J-shaped walls (6"x~5") for 10x5 base
# Longer arm (6") along -z edge, tip in bottom-right corner.
# Shorter arm (~4.7") extends upward from the other end (J corner NOT in base corner).
# Base spans x:[-5,5], z:[-2.5,2.5]. Walls 0.2" thick, 0.1" from base edges.
GW_TALL_J_6X5_WALLS = {
    "id": "gw_tall_J_6x5",
    "name": "Tall J Walls (6x5)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 6.0,
            "depth_inches": 0.2,
            "height_inches": 5.0,
            "offset": {"x_inches": 1.9, "z_inches": -2.3},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.2,
            "depth_inches": 4.7,
            "height_inches": 5.0,
            "offset": {"x_inches": -1.1, "z_inches": -0.05},
        },
    ],
    "tags": ["ruins"],
    "fill_color": "#444444",
    "outline_color": "#000000",
}

# L-shaped walls (6"x~5") for 10x5 base
# Longer arm (6") along -z edge, L corner in bottom-right base corner.
# Shorter arm (~4.7") extends upward from the corner end.
# Base spans x:[-5,5], z:[-2.5,2.5]. Walls 0.2" thick, 0.1" from base edges.
GW_TALL_L_6X5_WALLS = {
    "id": "gw_tall_L_6x5",
    "name": "Tall L Walls (6x5)",
    "shapes": [
        {
            "shape_type": "rectangular_prism",
            "width_inches": 6.0,
            "depth_inches": 0.2,
            "height_inches": 5.0,
            "offset": {"x_inches": 1.9, "z_inches": -2.3},
        },
        {
            "shape_type": "rectangular_prism",
            "width_inches": 0.2,
            "depth_inches": 4.7,
            "height_inches": 5.0,
            "offset": {"x_inches": 4.8, "z_inches": -0.05},
        },
    ],
    "tags": ["ruins"],
    "fill_color": "#444444",
    "outline_color": "#000000",
}

# GW Misc features

GW_FLAT_RUIN_FEATURE = {
    "id": "gw_flat_ruin",
    "feature_type": "obscuring",
    "components": [{"object_id": "gw_flat_ruin_base"}],
    "tags": ["ruins", "obscuring"],
}

GW_RUIN_L_4X8_FEATURE = {
    "id": "gw_ruin_L_4x8",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "gw_ruin_base_12x6"},
        {"object_id": "gw_tall_L_4x8"},
    ],
    "tags": ["ruins", "obscuring"],
}

GW_RUIN_U_8X4_FEATURE = {
    "id": "gw_ruin_U_8x4",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "gw_ruin_base_12x6"},
        {"object_id": "gw_tall_U_8x4"},
    ],
    "tags": ["ruins", "obscuring"],
}

GW_RUIN_J_6X5_FEATURE = {
    "id": "gw_ruin_J_6x5",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "gw_ruin_base_10x5"},
        {"object_id": "gw_tall_J_6x5"},
    ],
    "tags": ["ruins", "obscuring"],
}

GW_RUIN_L_6X5_FEATURE = {
    "id": "gw_ruin_L_6x5",
    "feature_type": "obscuring",
    "components": [
        {"object_id": "gw_ruin_base_10x5"},
        {"object_id": "gw_tall_L_6x5"},
    ],
    "tags": ["ruins", "obscuring"],
}


# ---------------------------------------------------------------------------
# Polygon terrain objects & features
# ---------------------------------------------------------------------------

# Kidney-bean woods: ~8"x5" organic shape, flat (height=0)
# Wide oval with a concave notch on the north (top) long side.
# Traced clockwise from the right end.
_KIDNEY_BEAN_VERTICES = [
    # Right end
    (4.0, 0.0),
    (3.5, -1.2),
    # Bottom (smooth convex curve)
    (2.5, -2.0),
    (1.0, -2.4),
    (-1.0, -2.4),
    (-2.5, -2.0),
    (-3.5, -1.2),
    # Left end
    (-4.0, 0.0),
    (-3.5, 1.2),
    # Smooth top-left curve (gradual rise to broad peak)
    (-2.8, 1.8),
    (-2.2, 2.2),
    (-1.6, 2.4),
    # Wide notch descent
    (-0.8, 1.8),
    (-0.3, 1.2),
    # Flat notch floor
    (0.3, 1.2),
    # Wide notch ascent
    (0.8, 1.8),
    # Smooth top-right curve (broad peak, gradual descent)
    (1.6, 2.4),
    (2.2, 2.2),
    (2.8, 1.8),
    (3.5, 1.2),
]

KIDNEY_BEAN_WOODS_OBJECT = {
    "id": "kidney_bean_woods",
    "name": "Kidney-Bean Woods",
    "shapes": [
        {
            "shape_type": "polygon",
            "vertices": _KIDNEY_BEAN_VERTICES,
            "width_inches": 8.0,
            "depth_inches": 4.8,
            "height_inches": 0.0,
        }
    ],
    "tags": ["woods"],
    "fill_color": "#4a8c3f",
    "outline_color": "#000000",
}

KIDNEY_BEAN_WOODS_FEATURE = {
    "id": "kidney_bean_woods",
    "feature_type": "woods",
    "components": [{"object_id": "kidney_bean_woods"}],
    "tags": ["woods"],
}

# Circular industrial tank: 5" diameter, 5" tall, approximated as 24-gon
_TANK_RADIUS = 2.5
_TANK_VERTICES = [
    (
        round(_TANK_RADIUS * math.cos(2 * math.pi * i / 24), 4),
        round(_TANK_RADIUS * math.sin(2 * math.pi * i / 24), 4),
    )
    for i in range(24)
]

INDUSTRIAL_TANK_OBJECT = {
    "id": "industrial_tank",
    "name": 'Industrial Tank (5" dia)',
    "shapes": [
        {
            "shape_type": "polygon",
            "vertices": _TANK_VERTICES,
            "width_inches": 5.0,
            "depth_inches": 5.0,
            "height_inches": 5.0,
        }
    ],
    "tags": ["container"],
    "fill_color": "#505050",
    "outline_color": "#000000",
}

INDUSTRIAL_TANK_FEATURE = {
    "id": "industrial_tank",
    "feature_type": "obstacle",
    "components": [{"object_id": "industrial_tank"}],
    "tags": ["obstacle"],
}


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------


def _merge_catalogs(name, *catalogs):
    """Merge multiple catalogs, deduplicating by ID, all unlimited."""
    seen_obj_ids: set[str] = set()
    seen_feat_ids: set[str] = set()
    objects = []
    features = []
    for cat in catalogs:
        for entry in cat["objects"]:
            oid = entry["item"]["id"]
            if oid not in seen_obj_ids:
                seen_obj_ids.add(oid)
                objects.append({"item": entry["item"], "quantity": None})
        for entry in cat["features"]:
            fid = entry["item"]["id"]
            if fid not in seen_feat_ids:
                seen_feat_ids.add(fid)
                features.append({"item": entry["item"], "quantity": None})
    return {"name": name, "objects": objects, "features": features}


_WTC_SET = {
    "name": "WTC Set",
    "objects": [
        {"item": CRATE_OBJECT, "quantity": 2},
        {"item": WTC_RUINS_BASE_TALL, "quantity": 6},
        {"item": WTC_RUINS_BASE_SHORT, "quantity": 8},
        {"item": WTC_THREE_STOREY_WALLS, "quantity": 6},
        {"item": WTC_SHORT_WALLS, "quantity": 8},
    ],
    "features": [
        {"item": CRATE_FEATURE, "quantity": 2},
        {"item": WTC_THREE_STOREY_FEATURE, "quantity": 6},
        {"item": WTC_SHORT_FEATURE, "quantity": 8},
    ],
}

_GW_MISC = {
    "name": "GW Misc",
    "objects": [
        {"item": GW_FLAT_RUIN_BASE, "quantity": None},
        {"item": GW_RUIN_BASE_12X6, "quantity": None},
        {"item": GW_RUIN_BASE_10X5, "quantity": None},
        {"item": GW_TALL_L_4X8_WALLS, "quantity": None},
        {"item": GW_TALL_U_8X4_WALLS, "quantity": None},
        {"item": GW_TALL_J_6X5_WALLS, "quantity": None},
        {"item": GW_TALL_L_6X5_WALLS, "quantity": None},
    ],
    "features": [
        {"item": GW_FLAT_RUIN_FEATURE, "quantity": None},
        {"item": GW_RUIN_L_4X8_FEATURE, "quantity": None},
        {"item": GW_RUIN_U_8X4_FEATURE, "quantity": None},
        {"item": GW_RUIN_J_6X5_FEATURE, "quantity": None},
        {"item": GW_RUIN_L_6X5_FEATURE, "quantity": None},
    ],
}

_POLYGON_TERRAIN = {
    "name": "Polygon Terrain",
    "objects": [
        {"item": KIDNEY_BEAN_WOODS_OBJECT, "quantity": None},
        {"item": INDUSTRIAL_TANK_OBJECT, "quantity": None},
    ],
    "features": [
        {"item": KIDNEY_BEAN_WOODS_FEATURE, "quantity": None},
        {"item": INDUSTRIAL_TANK_FEATURE, "quantity": None},
    ],
}

TERRAIN_CATALOGS = {
    "Omnium Gatherum": _merge_catalogs(
        "Omnium Gatherum", _WTC_SET, _GW_MISC, _POLYGON_TERRAIN
    ),
    "WTC Set": _WTC_SET,
    "GW Misc": _GW_MISC,
}
