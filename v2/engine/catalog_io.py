"""Load and save terrain catalogs from/to JSON files.

Provides helpers for reading catalog JSON files into typed ``TerrainCatalog``
objects (via ``types.py``) or raw dicts (preserving UI-only fields like
``fill_color``). Also provides path helpers for locating test and built-in
catalog files under ``v2/catalogs/``.

Used by:
  - ``engine_cmp/compare.py`` — loads test catalogs for parity scenarios.
  - ``frontend/catalogs.py`` — loads built-in production catalogs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import TerrainCatalog

# v2/catalogs/ is two levels up from v2/engine/catalog_io.py
_CATALOGS_DIR = Path(__file__).parent.parent / "catalogs"


def test_catalog_path(name: str) -> Path:
    """Return the path to a test catalog JSON file.

    Args:
        name: Catalog name without extension (e.g. "standard").

    Returns:
        Path to ``v2/catalogs/test/{name}.json``.
    """
    return _CATALOGS_DIR / "test" / f"{name}.json"


def builtin_catalog_path(name: str) -> Path:
    """Return the path to a built-in production catalog JSON file.

    Args:
        name: Catalog name without extension (e.g. "wtc_set").

    Returns:
        Path to ``v2/catalogs/builtin/{name}.json``.
    """
    return _CATALOGS_DIR / "builtin" / f"{name}.json"


def load_catalog(path: Path) -> TerrainCatalog:
    """Load a JSON catalog file and return a typed ``TerrainCatalog``.

    Only preserves engine-relevant fields (id, shapes, tags, quantities, etc.).
    UI-only fields like ``fill_color`` are dropped during ``from_dict()``.
    """
    with open(path) as f:
        data = json.load(f)
    return TerrainCatalog.from_dict(data)


def load_catalog_dict(path: Path) -> dict:
    """Load a JSON catalog file and return the raw dict.

    Preserves all fields including UI-only ones (``fill_color``,
    ``outline_color``). Use this when the caller needs non-engine fields.
    """
    with open(path) as f:
        return json.load(f)


def save_catalog_dict(data: dict, path: Path) -> None:
    """Write a catalog dict to a JSON file.

    Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
