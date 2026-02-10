"""Save and load terrain layouts as PNG (with embedded metadata) or JSON.

The primary format is PNG: the rendered layout image is saved with the full
layout JSON embedded in a PNG tEXt chunk (key: ``carltographer_layout``).
This means a saved file is both a shareable image and a complete,
machine-readable layout that can be loaded back into the app. JSON files
are also supported as a plain-text alternative.

Used by ``app.py`` for its Save/Load buttons.
"""

import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo

METADATA_KEY = "carltographer_layout"


def save_layout_png(img: Image.Image, layout: dict, path: str) -> None:
    """Save a rendered layout image with the layout JSON embedded as a PNG tEXt chunk."""
    info = PngInfo()
    info.add_text(METADATA_KEY, json.dumps(layout))
    img.save(path, pnginfo=info)


def load_layout_png(path: str) -> dict:
    """Load a layout dict from a PNG file's tEXt metadata.

    Raises ValueError if the PNG does not contain layout metadata.
    """
    img = Image.open(path)
    text_data = getattr(img, "text", None)
    if not text_data or METADATA_KEY not in text_data:
        raise ValueError(
            f"PNG file does not contain layout metadata (missing '{METADATA_KEY}' chunk)"
        )
    return json.loads(text_data[METADATA_KEY])


def load_layout_json(path: str) -> dict:
    """Load a layout dict from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_layout(path: str) -> dict:
    """Load a layout from a file, dispatching by extension.

    Supports .png (reads embedded metadata) and .json (reads raw JSON).
    Raises ValueError for unsupported extensions.
    """
    lower = path.lower()
    if lower.endswith(".png"):
        return load_layout_png(path)
    elif lower.endswith(".json"):
        return load_layout_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}")
