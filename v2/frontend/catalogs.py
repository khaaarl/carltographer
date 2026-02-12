"""Terrain catalog definitions for Warhammer 40k terrain pieces.

Pure data module with no UI dependencies (no tkinter), so it can be imported
by headless scripts (e.g. hyperparameter tuning) as well as the GUI.

Loads catalog data from JSON files under ``v2/catalogs/builtin/``, which
contain the full terrain definitions including UI fields (fill_color, etc.).
Each feature has a ``feature_type`` (obstacle, obscuring, or woods) used for
feature-count preferences.

Provides:
  - TERRAIN_CATALOGS: dict mapping catalog name -> {objects, features} with
    per-item quantities (None = unlimited).

Catalog contents:
  - "WTC Set": Official WTC tournament terrain (crates, 3-storey ruins, short
    ruins). Fixed quantities matching the physical set.
  - "GW Misc": Games Workshop miscellaneous ruins (flat bases, L/U/J-shaped
    wall configurations on larger bases). Unlimited quantities.
  - "Omnium Gatherum": Union of WTC + GW Misc + polygon terrain, all unlimited.
"""

from engine.catalog_io import builtin_catalog_path, load_catalog_dict

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


_WTC_SET = load_catalog_dict(builtin_catalog_path("wtc_set"))
_GW_MISC = load_catalog_dict(builtin_catalog_path("gw_misc"))
_POLYGON_TERRAIN = load_catalog_dict(builtin_catalog_path("polygon_terrain"))

TERRAIN_CATALOGS = {
    "Omnium Gatherum": _merge_catalogs(
        "Omnium Gatherum", _WTC_SET, _GW_MISC, _POLYGON_TERRAIN
    ),
    "WTC Set": _WTC_SET,
    "GW Misc": _GW_MISC,
}
