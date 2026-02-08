"""Mission data definitions for Warhammer 40k deployment zones.

Pure data module — no UI or engine dependencies.
All coordinates are in inches relative to table center.
"""

from __future__ import annotations

from typing import Any

# Table dimensions for Strike Force (standard matched play)
_TABLE_W = 60.0  # width (X axis)
_TABLE_D = 44.0  # depth (Z axis)
_HW = _TABLE_W / 2  # 30
_HD = _TABLE_D / 2  # 22

# edition -> mission_pack -> deployment_name -> mission_dict
EDITIONS: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    "10th Edition": {
        "Matched Play: Chapter Approved 2025-26": {
            "Hammer and Anvil": {
                "name": "Hammer and Anvil",
                "rotationally_symmetric": True,
                "deployment_zones": [
                    {
                        "id": "green",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": -_HD},
                                {"x_inches": -_HW + 18, "z_inches": -_HD},
                                {"x_inches": -_HW + 18, "z_inches": _HD},
                                {"x_inches": -_HW, "z_inches": _HD},
                            ]
                        ],
                    },
                    {
                        "id": "red",
                        "polygons": [
                            [
                                {"x_inches": _HW - 18, "z_inches": -_HD},
                                {"x_inches": _HW, "z_inches": -_HD},
                                {"x_inches": _HW, "z_inches": _HD},
                                {"x_inches": _HW - 18, "z_inches": _HD},
                            ]
                        ],
                    },
                ],
                "objectives": [
                    {
                        "id": "1",
                        "position": {"x_inches": 0.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "2",
                        "position": {"x_inches": 0.0, "z_inches": -16.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "3",
                        "position": {"x_inches": 0.0, "z_inches": 16.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "4",
                        "position": {"x_inches": -20.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "5",
                        "position": {"x_inches": 20.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                ],
            },
            "Dawn of War": {
                "name": "Dawn of War",
                "rotationally_symmetric": True,
                "deployment_zones": [
                    {
                        "id": "red",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": -_HD},
                                {"x_inches": _HW, "z_inches": -_HD},
                                {"x_inches": _HW, "z_inches": -_HD + 10},
                                {"x_inches": -_HW, "z_inches": -_HD + 10},
                            ]
                        ],
                    },
                    {
                        "id": "green",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": _HD - 10},
                                {"x_inches": _HW, "z_inches": _HD - 10},
                                {"x_inches": _HW, "z_inches": _HD},
                                {"x_inches": -_HW, "z_inches": _HD},
                            ]
                        ],
                    },
                ],
                "objectives": [
                    {
                        "id": "1",
                        "position": {"x_inches": 0.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "2",
                        "position": {"x_inches": -20.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "3",
                        "position": {"x_inches": 20.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "4",
                        "position": {"x_inches": 0.0, "z_inches": -16.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "5",
                        "position": {"x_inches": 0.0, "z_inches": 16.0},
                        "range_inches": 3.0,
                    },
                ],
            },
            # L-shaped zones along short edges
            "Tipping Point": {
                "name": "Tipping Point",
                "rotationally_symmetric": True,
                "deployment_zones": [
                    {
                        "id": "green",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": _HD},
                                {"x_inches": -10.0, "z_inches": _HD},
                                {"x_inches": -10.0, "z_inches": 0.0},
                                {"x_inches": -18.0, "z_inches": 0.0},
                                {"x_inches": -18.0, "z_inches": -_HD},
                                {"x_inches": -_HW, "z_inches": -_HD},
                            ]
                        ],
                    },
                    {
                        "id": "red",
                        "polygons": [
                            [
                                {"x_inches": _HW, "z_inches": -_HD},
                                {"x_inches": 10.0, "z_inches": -_HD},
                                {"x_inches": 10.0, "z_inches": 0.0},
                                {"x_inches": 18.0, "z_inches": 0.0},
                                {"x_inches": 18.0, "z_inches": _HD},
                                {"x_inches": _HW, "z_inches": _HD},
                            ]
                        ],
                    },
                ],
                "objectives": [
                    {
                        "id": "1",
                        "position": {"x_inches": 0.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "2",
                        "position": {"x_inches": -16.0, "z_inches": 12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "3",
                        "position": {"x_inches": 16.0, "z_inches": -12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "4",
                        "position": {"x_inches": 8.0, "z_inches": 14.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "5",
                        "position": {"x_inches": -8.0, "z_inches": -14.0},
                        "range_inches": 3.0,
                    },
                ],
            },
            # L-shaped zones along long edges
            "Sweeping Engagement": {
                "name": "Sweeping Engagement",
                "rotationally_symmetric": True,
                "deployment_zones": [
                    {
                        "id": "green",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": _HD},
                                {"x_inches": _HW, "z_inches": _HD},
                                {"x_inches": _HW, "z_inches": 14.0},
                                {"x_inches": 0.0, "z_inches": 14.0},
                                {"x_inches": 0.0, "z_inches": 8.0},
                                {"x_inches": -_HW, "z_inches": 8.0},
                            ]
                        ],
                    },
                    {
                        "id": "red",
                        "polygons": [
                            [
                                {"x_inches": _HW, "z_inches": -_HD},
                                {"x_inches": -_HW, "z_inches": -_HD},
                                {"x_inches": -_HW, "z_inches": -14.0},
                                {"x_inches": 0.0, "z_inches": -14.0},
                                {"x_inches": 0.0, "z_inches": -8.0},
                                {"x_inches": _HW, "z_inches": -8.0},
                            ]
                        ],
                    },
                ],
                "objectives": [
                    {
                        "id": "1",
                        "position": {"x_inches": 0.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "2",
                        "position": {"x_inches": -12.0, "z_inches": 16.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "3",
                        "position": {"x_inches": 12.0, "z_inches": -16.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "4",
                        "position": {"x_inches": 20.0, "z_inches": 4.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "5",
                        "position": {"x_inches": -20.0, "z_inches": -4.0},
                        "range_inches": 3.0,
                    },
                ],
            },
            # Diagonal triangle zones
            "Crucible of Battle": {
                "name": "Crucible of Battle",
                "rotationally_symmetric": True,
                "deployment_zones": [
                    {
                        "id": "green",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": -_HD},
                                {"x_inches": -_HW, "z_inches": _HD},
                                {"x_inches": 0.0, "z_inches": _HD},
                            ]
                        ],
                    },
                    {
                        "id": "red",
                        "polygons": [
                            [
                                {"x_inches": _HW, "z_inches": _HD},
                                {"x_inches": _HW, "z_inches": -_HD},
                                {"x_inches": 0.0, "z_inches": -_HD},
                            ]
                        ],
                    },
                ],
                "objectives": [
                    {
                        "id": "1",
                        "position": {"x_inches": 0.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "2",
                        "position": {"x_inches": -16.0, "z_inches": 12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "3",
                        "position": {"x_inches": 16.0, "z_inches": -12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "4",
                        "position": {"x_inches": 10.0, "z_inches": 14.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "5",
                        "position": {"x_inches": -10.0, "z_inches": -14.0},
                        "range_inches": 3.0,
                    },
                ],
            },
            # Quarter-table zones with 9" arc cutout (approximated)
            "Search and Destroy": {
                "name": "Search and Destroy",
                "rotationally_symmetric": True,
                "deployment_zones": [
                    {
                        "id": "green",
                        "polygons": [
                            [
                                {"x_inches": -_HW, "z_inches": _HD},
                                {"x_inches": 0.0, "z_inches": _HD},
                                {"x_inches": 0.0, "z_inches": 9.0},
                                {"x_inches": -2.33, "z_inches": 8.69},
                                {"x_inches": -4.5, "z_inches": 7.79},
                                {"x_inches": -6.36, "z_inches": 6.36},
                                {"x_inches": -7.79, "z_inches": 4.5},
                                {"x_inches": -8.69, "z_inches": 2.33},
                                {"x_inches": -9.0, "z_inches": 0.0},
                                {"x_inches": -_HW, "z_inches": 0.0},
                            ]
                        ],
                    },
                    {
                        "id": "red",
                        "polygons": [
                            [
                                {"x_inches": _HW, "z_inches": -_HD},
                                {"x_inches": 0.0, "z_inches": -_HD},
                                {"x_inches": 0.0, "z_inches": -9.0},
                                {"x_inches": 2.33, "z_inches": -8.69},
                                {"x_inches": 4.5, "z_inches": -7.79},
                                {"x_inches": 6.36, "z_inches": -6.36},
                                {"x_inches": 7.79, "z_inches": -4.5},
                                {"x_inches": 8.69, "z_inches": -2.33},
                                {"x_inches": 9.0, "z_inches": 0.0},
                                {"x_inches": _HW, "z_inches": 0.0},
                            ]
                        ],
                    },
                ],
                "objectives": [
                    {
                        "id": "1",
                        "position": {"x_inches": 0.0, "z_inches": 0.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "2",
                        "position": {"x_inches": -16.0, "z_inches": 12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "3",
                        "position": {"x_inches": 16.0, "z_inches": -12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "4",
                        "position": {"x_inches": 16.0, "z_inches": 12.0},
                        "range_inches": 3.0,
                    },
                    {
                        "id": "5",
                        "position": {"x_inches": -16.0, "z_inches": -12.0},
                        "range_inches": 3.0,
                    },
                ],
            },
        },
    },
}


def get_mission(edition: str, pack: str, deployment: str) -> dict | None:
    """Look up a mission by edition/pack/deployment name. Returns None if not found."""
    return EDITIONS.get(edition, {}).get(pack, {}).get(deployment)


def find_mission_path(mission_name: str) -> tuple[str, str, str] | None:
    """Find (edition, pack, deployment) tuple for a mission name, or None."""
    for ed_name, packs in EDITIONS.items():
        for pack_name, deployments in packs.items():
            for dep_name, mission in deployments.items():
                if mission["name"] == mission_name:
                    return (ed_name, pack_name, dep_name)
    return None
