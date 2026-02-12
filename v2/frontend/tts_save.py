"""Tabletop Simulator save-file creation and object lookup.

Creates TTS-compatible ``.json`` save files that can be loaded directly by
Tabletop Simulator via its Saves folder.  The module provides:

* ``tts_default_save_location()`` — platform-specific path to the TTS Saves
  directory (Linux, macOS, Windows).
* ``reference_save_json()`` — loads the reference TTS save envelope from
  ``reference_info/tts/reference_save_file.json`` (cached).
* ``reference_objects()`` — builds a nickname-keyed dict of TTS object
  templates from the reference save (cached).
* ``reference_object(nickname)`` — returns a deep copy of a reference object
  with fresh GUIDs, ready to customize and place.
* ``create_empty_tts_save(name)`` — writes an empty TTS save (no objects)
  with the given name into the TTS Saves directory.
* ``tts_feature_info(catalog_dict)`` — extracts a ``feature_id -> tts info``
  mapping from a loaded catalog dict, for features that have a ``"tts"`` block.

The reference object lookup follows the same pattern as
``caverns-of-carl/lib/tts.py``: objects are identified by their TTS
``Nickname`` field, deep-copied on retrieval, and given fresh GUIDs to avoid
collisions when multiple instances are placed.
"""

import copy
import functools
import json
import math
import os
import pathlib
import random
import re
import sys


def tts_default_save_location() -> str:
    """Return the platform-specific TTS Saves directory path.

    On Linux, checks two common Steam install paths (native and snap) and
    returns the first that exists, falling back to the native path.
    """
    if sys.platform in ("linux", "linux2"):
        candidates = [
            os.path.join(
                str(pathlib.Path.home()),
                ".local",
                "share",
                "Tabletop Simulator",
                "Saves",
            ),
            os.path.join(
                str(pathlib.Path.home()),
                "snap",
                "steam",
                "common",
                ".local",
                "share",
                "Tabletop Simulator",
                "Saves",
            ),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return candidates[0]
    elif sys.platform == "darwin":
        return os.path.join(
            str(pathlib.Path.home()), "Library", "Tabletop Simulator", "Saves"
        )
    elif sys.platform == "win32":
        return os.path.join(
            os.environ["USERPROFILE"],
            "Documents",
            "My Games",
            "Tabletop Simulator",
            "Saves",
        )
    else:
        raise RuntimeError(
            f"Unknown platform {sys.platform!r}; cannot determine TTS save location"
        )


@functools.cache
def reference_save_json() -> dict:
    """Load and return the reference TTS save envelope (cached)."""
    ref_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "reference_info"
        / "tts"
        / "reference_save_file.json"
    )
    with open(ref_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GUID management
# ---------------------------------------------------------------------------

_seen_tts_guids: set[str] = set()


def _new_tts_guid() -> str:
    """Generate a unique 6-character hex GUID for TTS objects."""
    while True:
        guid = f"{random.randrange(0x1000000):06x}"
        if guid not in _seen_tts_guids:
            _seen_tts_guids.add(guid)
            return guid


def refresh_tts_guids(d):
    """Recursively replace all ``"GUID"`` values with fresh unique hex strings.

    Mutates *d* in place (works on dicts and lists). Tracks all seen GUIDs
    globally to avoid collisions across calls.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "GUID" and isinstance(v, str):
                _seen_tts_guids.add(v)
                d[k] = _new_tts_guid()
            elif isinstance(v, (dict, list)):
                refresh_tts_guids(v)
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                refresh_tts_guids(item)


# ---------------------------------------------------------------------------
# Reference object lookup
# ---------------------------------------------------------------------------


def _normalize_nickname(s: str) -> str:
    """Uppercase and collapse whitespace for case-insensitive matching."""
    return re.sub(r"\s+", " ", s.strip().upper())


@functools.cache
def reference_objects() -> dict:
    """Build a ``normalized_nickname -> object`` dict from the reference save.

    Scans ``ObjectStates`` for objects with a ``Nickname`` field and indexes
    them by normalized nickname.  Returns the cached dict (do not mutate).
    """
    d = {}
    for obj in reference_save_json().get("ObjectStates", []):
        nick = obj.get("Nickname", "")
        normalized = _normalize_nickname(nick)
        if normalized and normalized not in d:
            d[normalized] = obj
    return d


def reference_object(nickname: str) -> dict:
    """Return a deep copy of a reference TTS object, with fresh GUIDs.

    Raises ``KeyError`` if *nickname* is not found in the reference save.
    """
    normalized = _normalize_nickname(nickname)
    objs = reference_objects()
    if normalized not in objs:
        available = sorted(objs.keys())
        raise KeyError(
            f"TTS reference object {nickname!r} not found. "
            f"Available: {available}"
        )
    obj = copy.deepcopy(objs[normalized])
    refresh_tts_guids(obj)
    return obj


# ---------------------------------------------------------------------------
# Catalog TTS info extraction
# ---------------------------------------------------------------------------


def tts_feature_info(catalog_dict: dict) -> dict:
    """Extract ``feature_id -> tts_info`` from a loaded catalog dict.

    Returns a dict mapping feature IDs to their ``"tts"`` blocks, for
    features that have one.  Features without a ``"tts"`` key are omitted.
    """
    result = {}
    for entry in catalog_dict.get("features", []):
        item = entry.get("item", {})
        tts = item.get("tts")
        if tts is not None:
            result[item["id"]] = tts
    return result


# ---------------------------------------------------------------------------
# Object placement
# ---------------------------------------------------------------------------


def _place_tts_object(
    tts_obj: dict,
    feature_x: float,
    feature_z: float,
    feature_rot_deg: float,
    pivot_offset: dict,
) -> None:
    """Position a TTS object for a Carltographer placed feature.

    *feature_x*, *feature_z* are the Carltographer feature center in inches
    (origin at table center).  *feature_rot_deg* is the Carltographer rotation.
    *pivot_offset* is the ``pivot_offset`` dict from the catalog's tts block
    (x_inches, z_inches, rotation_deg — all in the feature's local frame).

    The TTS object's Transform is mutated in place.  posY is left unchanged
    (kept from the reference object).
    """
    dx = pivot_offset.get("x_inches", 0.0)
    dz = pivot_offset.get("z_inches", 0.0)
    rot_offset_deg = pivot_offset.get("rotation_deg", 0.0)

    # Rotate the local pivot offset by the feature's rotation
    theta = math.radians(feature_rot_deg)
    world_dx = dx * math.cos(theta) - dz * math.sin(theta)
    world_dz = dx * math.sin(theta) + dz * math.cos(theta)

    tts_obj["Transform"]["posX"] = feature_x + world_dx
    tts_obj["Transform"]["posZ"] = -(feature_z + world_dz)
    tts_obj["Transform"]["rotY"] = feature_rot_deg + rot_offset_deg


def _build_object_states(layout: dict, catalog_dict: dict) -> list[dict]:
    """Build TTS ObjectStates from a Carltographer layout + catalog.

    For each placed feature that has a ``"tts"`` block in the catalog,
    creates a positioned TTS object.  Features without TTS info are skipped.

    When the layout is rotationally symmetric, features not at the origin
    get a mirrored copy at ``(-x, -z, rotation + 180°)``.
    """
    tts_info = tts_feature_info(catalog_dict)
    symmetric = layout.get("rotationally_symmetric", False)
    objects = []

    for pf in layout.get("placed_features", []):
        feature = pf["feature"]
        feature_id = feature["id"]
        info = tts_info.get(feature_id)
        if info is None:
            continue

        nickname = info["nickname"]
        pivot_offset = info.get("pivot_offset", {})

        transform = pf.get("transform", {})
        fx = transform.get("x_inches", 0.0)
        fz = transform.get("z_inches", 0.0)
        frot = transform.get("rotation_deg", 0.0)

        obj = reference_object(nickname)
        _place_tts_object(obj, fx, fz, frot, pivot_offset)
        obj["Locked"] = True
        objects.append(obj)

        # Mirror for rotational symmetry (skip origin features)
        if symmetric and (fx != 0.0 or fz != 0.0):
            mirror_obj = reference_object(nickname)
            _place_tts_object(mirror_obj, -fx, -fz, frot + 180.0, pivot_offset)
            mirror_obj["Locked"] = True
            objects.append(mirror_obj)

    return objects


# ---------------------------------------------------------------------------
# Save file creation
# ---------------------------------------------------------------------------


def create_empty_tts_save(name: str) -> str:
    """Write an empty TTS save file and return the full path.

    Deep-copies the reference save, sets ``SaveName`` and ``GameMode`` to
    *name*, ensures ``ObjectStates`` is empty, and writes to
    ``<tts_save_dir>/<name>.json``.
    """
    save = copy.deepcopy(reference_save_json())
    save["SaveName"] = name
    save["GameMode"] = name
    save["ObjectStates"] = []

    save_dir = tts_default_save_location()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.json")
    with open(save_path, "w") as f:
        json.dump(save, f, indent=2)
    return save_path


def create_tts_save(name: str, layout: dict, catalog_dict: dict) -> str:
    """Write a TTS save file populated with terrain from *layout*.

    For each placed feature with TTS info in *catalog_dict*, a positioned
    TTS object is added to ``ObjectStates``.  Features without TTS info
    are silently skipped.

    Returns the full path to the written save file.
    """
    save = copy.deepcopy(reference_save_json())
    save["SaveName"] = name
    save["GameMode"] = name
    save["ObjectStates"] = _build_object_states(layout, catalog_dict)

    save_dir = tts_default_save_location()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.json")
    with open(save_path, "w") as f:
        json.dump(save, f, indent=2)
    return save_path
