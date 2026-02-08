"""Tests for layout_io save/load helpers."""

import json

from PIL import Image

from .layout_io import (
    load_layout,
    load_layout_json,
    load_layout_png,
    save_layout_png,
)

SAMPLE_LAYOUT = {
    "table_width_inches": 60,
    "table_depth_inches": 44,
    "placed_features": [
        {
            "feature": {
                "id": "crate",
                "feature_type": "obstacle",
                "components": [{"object_id": "crate"}],
            },
            "transform": {
                "x_inches": 5.0,
                "y_inches": 0.0,
                "z_inches": -3.0,
                "rotation_deg": 45.0,
            },
        }
    ],
}


def test_save_and_load_png_roundtrip(tmp_path):
    """Save a layout in a PNG, load it back, and verify equality."""
    img = Image.new("RGB", (100, 100), "green")
    path = str(tmp_path / "layout.png")

    save_layout_png(img, SAMPLE_LAYOUT, path)
    loaded = load_layout_png(path)

    assert loaded == SAMPLE_LAYOUT


def test_load_png_missing_chunk(tmp_path):
    """A plain PNG without metadata raises ValueError."""
    img = Image.new("RGB", (100, 100), "red")
    path = str(tmp_path / "plain.png")
    img.save(path)

    try:
        load_layout_png(path)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "carltographer_layout" in str(e)


def test_load_json_roundtrip(tmp_path):
    """Write a JSON file and load it back."""
    path = str(tmp_path / "layout.json")
    with open(path, "w") as f:
        json.dump(SAMPLE_LAYOUT, f)

    loaded = load_layout_json(path)
    assert loaded == SAMPLE_LAYOUT


def test_load_layout_dispatches_by_extension(tmp_path):
    """load_layout dispatches to PNG or JSON loader based on extension."""
    # PNG path
    img = Image.new("RGB", (100, 100), "blue")
    png_path = str(tmp_path / "test.png")
    save_layout_png(img, SAMPLE_LAYOUT, png_path)

    assert load_layout(png_path) == SAMPLE_LAYOUT

    # JSON path
    json_path = str(tmp_path / "test.json")
    with open(json_path, "w") as f:
        json.dump(SAMPLE_LAYOUT, f)

    assert load_layout(json_path) == SAMPLE_LAYOUT


def test_load_layout_unsupported_extension(tmp_path):
    """load_layout raises ValueError for unsupported extensions."""
    path = str(tmp_path / "layout.txt")
    try:
        load_layout(path)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Unsupported" in str(e)
