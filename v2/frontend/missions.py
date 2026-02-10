"""Mission data definitions for Warhammer 40k deployment zones.

Pure data module — no UI or engine dependencies.
All coordinates are in inches relative to table center.

Missions are defined as builder functions that take table dimensions and return
a mission dict. DZ outer edges scale with table boundaries; interior boundaries
maintain fixed distances from edges; objectives shift to preserve edge offsets.
"""

from __future__ import annotations

from typing import Any, Callable

MissionBuilder = Callable[[float, float], dict[str, Any]]


def _hammer_and_anvil(tw: float, td: float) -> dict[str, Any]:
    hw, hd = tw / 2, td / 2
    return {
        "name": "Hammer and Anvil",
        "rotationally_symmetric": True,
        "deployment_zones": [
            {
                "id": "green",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": -hd},
                        {"x_inches": -hw + 18, "z_inches": -hd},
                        {"x_inches": -hw + 18, "z_inches": hd},
                        {"x_inches": -hw, "z_inches": hd},
                    ]
                ],
            },
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": hw - 18, "z_inches": -hd},
                        {"x_inches": hw, "z_inches": -hd},
                        {"x_inches": hw, "z_inches": hd},
                        {"x_inches": hw - 18, "z_inches": hd},
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
                "position": {"x_inches": 0.0, "z_inches": -hd + 6},
                "range_inches": 3.0,
            },
            {
                "id": "3",
                "position": {"x_inches": 0.0, "z_inches": hd - 6},
                "range_inches": 3.0,
            },
            {
                "id": "4",
                "position": {"x_inches": -hw + 10, "z_inches": 0.0},
                "range_inches": 3.0,
            },
            {
                "id": "5",
                "position": {"x_inches": hw - 10, "z_inches": 0.0},
                "range_inches": 3.0,
            },
        ],
    }


def _dawn_of_war(tw: float, td: float) -> dict[str, Any]:
    hw, hd = tw / 2, td / 2
    return {
        "name": "Dawn of War",
        "rotationally_symmetric": True,
        "deployment_zones": [
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": -hd},
                        {"x_inches": hw, "z_inches": -hd},
                        {"x_inches": hw, "z_inches": -hd + 10},
                        {"x_inches": -hw, "z_inches": -hd + 10},
                    ]
                ],
            },
            {
                "id": "green",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": hd - 10},
                        {"x_inches": hw, "z_inches": hd - 10},
                        {"x_inches": hw, "z_inches": hd},
                        {"x_inches": -hw, "z_inches": hd},
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
                "position": {"x_inches": -hw + 10, "z_inches": 0.0},
                "range_inches": 3.0,
            },
            {
                "id": "3",
                "position": {"x_inches": hw - 10, "z_inches": 0.0},
                "range_inches": 3.0,
            },
            {
                "id": "4",
                "position": {"x_inches": 0.0, "z_inches": -hd + 6},
                "range_inches": 3.0,
            },
            {
                "id": "5",
                "position": {"x_inches": 0.0, "z_inches": hd - 6},
                "range_inches": 3.0,
            },
        ],
    }


def _tipping_point(tw: float, td: float) -> dict[str, Any]:
    hw, hd = tw / 2, td / 2
    return {
        "name": "Tipping Point",
        "rotationally_symmetric": True,
        "deployment_zones": [
            {
                "id": "green",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": hd},
                        {"x_inches": -hw + 20, "z_inches": hd},
                        {"x_inches": -hw + 20, "z_inches": 0.0},
                        {"x_inches": -hw + 12, "z_inches": 0.0},
                        {"x_inches": -hw + 12, "z_inches": -hd},
                        {"x_inches": -hw, "z_inches": -hd},
                    ]
                ],
            },
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": hw, "z_inches": -hd},
                        {"x_inches": hw - 20, "z_inches": -hd},
                        {"x_inches": hw - 20, "z_inches": 0.0},
                        {"x_inches": hw - 12, "z_inches": 0.0},
                        {"x_inches": hw - 12, "z_inches": hd},
                        {"x_inches": hw, "z_inches": hd},
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
                "position": {"x_inches": -hw + 14, "z_inches": hd - 10},
                "range_inches": 3.0,
            },
            {
                "id": "3",
                "position": {"x_inches": hw - 14, "z_inches": -hd + 10},
                "range_inches": 3.0,
            },
            {
                "id": "4",
                "position": {"x_inches": hw - 22, "z_inches": hd - 8},
                "range_inches": 3.0,
            },
            {
                "id": "5",
                "position": {"x_inches": -hw + 22, "z_inches": -hd + 8},
                "range_inches": 3.0,
            },
        ],
    }


def _sweeping_engagement(tw: float, td: float) -> dict[str, Any]:
    hw, hd = tw / 2, td / 2
    return {
        "name": "Sweeping Engagement",
        "rotationally_symmetric": True,
        "deployment_zones": [
            {
                "id": "green",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": hd},
                        {"x_inches": hw, "z_inches": hd},
                        {"x_inches": hw, "z_inches": hd - 8},
                        {"x_inches": 0.0, "z_inches": hd - 8},
                        {"x_inches": 0.0, "z_inches": hd - 14},
                        {"x_inches": -hw, "z_inches": hd - 14},
                    ]
                ],
            },
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": hw, "z_inches": -hd},
                        {"x_inches": -hw, "z_inches": -hd},
                        {"x_inches": -hw, "z_inches": -hd + 8},
                        {"x_inches": 0.0, "z_inches": -hd + 8},
                        {"x_inches": 0.0, "z_inches": -hd + 14},
                        {"x_inches": hw, "z_inches": -hd + 14},
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
                "position": {"x_inches": -hw + 18, "z_inches": hd - 6},
                "range_inches": 3.0,
            },
            {
                "id": "3",
                "position": {"x_inches": hw - 18, "z_inches": -hd + 6},
                "range_inches": 3.0,
            },
            {
                "id": "4",
                "position": {"x_inches": hw - 10, "z_inches": hd - 18},
                "range_inches": 3.0,
            },
            {
                "id": "5",
                "position": {"x_inches": -hw + 10, "z_inches": -hd + 18},
                "range_inches": 3.0,
            },
        ],
    }


def _crucible_of_battle(tw: float, td: float) -> dict[str, Any]:
    hw, hd = tw / 2, td / 2
    return {
        "name": "Crucible of Battle",
        "rotationally_symmetric": True,
        "deployment_zones": [
            {
                "id": "green",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": -hd},
                        {"x_inches": -hw, "z_inches": hd},
                        {"x_inches": 0.0, "z_inches": hd},
                    ]
                ],
            },
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": hw, "z_inches": hd},
                        {"x_inches": hw, "z_inches": -hd},
                        {"x_inches": 0.0, "z_inches": -hd},
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
                "position": {"x_inches": -hw + 14, "z_inches": hd - 10},
                "range_inches": 3.0,
            },
            {
                "id": "3",
                "position": {"x_inches": hw - 14, "z_inches": -hd + 10},
                "range_inches": 3.0,
            },
            {
                "id": "4",
                "position": {"x_inches": hw - 20, "z_inches": hd - 8},
                "range_inches": 3.0,
            },
            {
                "id": "5",
                "position": {"x_inches": -hw + 20, "z_inches": -hd + 8},
                "range_inches": 3.0,
            },
        ],
    }


def _search_and_destroy(tw: float, td: float) -> dict[str, Any]:
    hw, hd = tw / 2, td / 2
    return {
        "name": "Search and Destroy",
        "rotationally_symmetric": True,
        "deployment_zones": [
            {
                "id": "green",
                "polygons": [
                    [
                        {"x_inches": -hw, "z_inches": hd},
                        {"x_inches": 0.0, "z_inches": hd},
                        {"x_inches": 0.0, "z_inches": 9.0},
                        {"x_inches": -2.33, "z_inches": 8.69},
                        {"x_inches": -4.5, "z_inches": 7.79},
                        {"x_inches": -6.36, "z_inches": 6.36},
                        {"x_inches": -7.79, "z_inches": 4.5},
                        {"x_inches": -8.69, "z_inches": 2.33},
                        {"x_inches": -9.0, "z_inches": 0.0},
                        {"x_inches": -hw, "z_inches": 0.0},
                    ]
                ],
            },
            {
                "id": "red",
                "polygons": [
                    [
                        {"x_inches": hw, "z_inches": -hd},
                        {"x_inches": 0.0, "z_inches": -hd},
                        {"x_inches": 0.0, "z_inches": -9.0},
                        {"x_inches": 2.33, "z_inches": -8.69},
                        {"x_inches": 4.5, "z_inches": -7.79},
                        {"x_inches": 6.36, "z_inches": -6.36},
                        {"x_inches": 7.79, "z_inches": -4.5},
                        {"x_inches": 8.69, "z_inches": -2.33},
                        {"x_inches": 9.0, "z_inches": 0.0},
                        {"x_inches": hw, "z_inches": 0.0},
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
                "position": {"x_inches": -hw + 14, "z_inches": hd - 10},
                "range_inches": 3.0,
            },
            {
                "id": "3",
                "position": {"x_inches": hw - 14, "z_inches": -hd + 10},
                "range_inches": 3.0,
            },
            {
                "id": "4",
                "position": {"x_inches": hw - 14, "z_inches": hd - 10},
                "range_inches": 3.0,
            },
            {
                "id": "5",
                "position": {"x_inches": -hw + 14, "z_inches": -hd + 10},
                "range_inches": 3.0,
            },
        ],
    }


# edition -> mission_pack -> deployment_name -> builder function
EDITIONS: dict[str, dict[str, dict[str, MissionBuilder]]] = {
    "10th Edition": {
        "Matched Play: Chapter Approved 2025-26": {
            "Hammer and Anvil": _hammer_and_anvil,
            "Dawn of War": _dawn_of_war,
            "Tipping Point": _tipping_point,
            "Sweeping Engagement": _sweeping_engagement,
            "Crucible of Battle": _crucible_of_battle,
            "Search and Destroy": _search_and_destroy,
        },
    },
}


def get_mission(
    edition: str,
    pack: str,
    deployment: str,
    table_width: float = 60.0,
    table_depth: float = 44.0,
) -> dict | None:
    """Look up a mission by edition/pack/deployment name.

    Calls the builder with the given table dimensions.
    Returns None if not found.
    """
    builder = EDITIONS.get(edition, {}).get(pack, {}).get(deployment)
    if builder is None:
        return None
    return builder(table_width, table_depth)


def find_mission_path(mission_name: str) -> tuple[str, str, str] | None:
    """Find (edition, pack, deployment) tuple for a mission name, or None."""
    for ed_name, packs in EDITIONS.items():
        for pack_name, deployments in packs.items():
            if mission_name in deployments:
                return (ed_name, pack_name, mission_name)
    return None
