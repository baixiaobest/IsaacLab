# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connected rooms and corridor height-field terrain."""

from __future__ import annotations

import numpy as np

from isaaclab.utils import configclass

from .hf_terrains_cfg import HfTerrainBaseCfg
from .utils import height_field_to_mesh


def room_corridor_layout(level: int, cfg: RoomCorridorTerrainCfg) -> dict:
    """Return the geometry shared by mesh generation and route sampling."""
    rng = np.random.default_rng(cfg.layout_seed + int(level))
    evaluation = cfg.evaluation_layout
    progress = min(max(level, 0), cfg.num_layout_levels - 1) / (cfg.num_layout_levels - 1)
    width = (1.8, 1.2)[int(level) >= 4] if evaluation else np.interp(progress, (0.0, 1.0), cfg.corridor_width_range)
    width += float(rng.uniform(-0.04, 0.04))
    if evaluation:
        door_width = 1.5 - 0.6 * progress
        left_x = float(rng.uniform(-5.2, -3.5))
        right_x = float(rng.uniform(3.5, 5.2))
    else:
        door_width = float(np.interp(progress, (0.0, 1.0), cfg.door_width_range))
        gap = float(np.interp(progress, (0.0, 1.0), cfg.door_gap_range))
        gap = max(0.0, gap + float(rng.uniform(-0.4, 0.4)) * progress)
        offset = float(rng.uniform(-0.3, 0.3))
        left_x = offset - (door_width + gap) / 2
        right_x = offset + (door_width + gap) / 2
    return {"width": width, "door_width": door_width, "doors": (left_x, right_x), "wall": 0.3,
            "room_depth": float(rng.uniform(3.2, 4.4)), "room_half_length": float(rng.uniform(2.1, 2.8))}


@height_field_to_mesh
def room_corridor_terrain(difficulty: float, cfg: HfTerrainBaseCfg) -> np.ndarray:
    level = min(int(difficulty * cfg.num_layout_levels), cfg.num_layout_levels - 1)
    layout = room_corridor_layout(level, cfg)
    scale = cfg.horizontal_scale
    height = np.zeros((round(cfg.size[0] / scale), round(cfg.size[1] / scale)), dtype=np.int16)

    def box(x0, x1, y0, y1, *, left_open=False, right_open=False):
        ix0 = max(0, int((np.ceil if left_open else np.floor)((x0 + cfg.size[0] / 2) / scale)))
        ix1 = min(height.shape[0], int((np.floor if right_open else np.ceil)((x1 + cfg.size[0] / 2) / scale)))
        iy0 = max(0, int(np.floor((y0 + cfg.size[1] / 2) / scale)))
        iy1 = min(height.shape[1], int(np.ceil((y1 + cfg.size[1] / 2) / scale)))
        height[ix0:ix1, iy0:iy1] = round(1.5 / cfg.vertical_scale)

    half = layout["width"] / 2
    wall = layout["wall"]
    door_half = layout["door_width"] / 2
    for side, center in ((1, layout["doors"][0]), (-1, layout["doors"][1])):
        inner = half if side > 0 else -half - wall
        outer = inner + wall
        box(-10, center - door_half, inner, outer, right_open=True)
        box(center + door_half, 10, inner, outer, left_open=True)
        room_end = side * (half + wall + layout["room_depth"])
        near, far = sorted((side * (half + wall), room_end))
        room_half = layout["room_half_length"]
        box(center - room_half - wall, center - room_half, near, far)
        box(center + room_half, center + room_half + wall, near, far)
        if side > 0:
            box(center - room_half - wall, center + room_half + wall, far, far + wall)
        else:
            box(center - room_half - wall, center + room_half + wall, near - wall, near)
    return height


@configclass
class RoomCorridorTerrainCfg(HfTerrainBaseCfg):
    """Height-field config for two rooms opening onto a central corridor.

    Width and gap ranges give the easiest and hardest training-level values.
    The evaluation layout uses its fixed benchmark geometry and independent seed.
    """

    function = room_corridor_terrain
    proportion: float = 1.0
    layout_seed: int = 0
    num_layout_levels: int = 10
    evaluation_layout: bool = False
    corridor_width_range: tuple[float, float] = (3.0, 1.2)
    door_width_range: tuple[float, float] = (2.5, 0.9)
    door_gap_range: tuple[float, float] = (0.0, 7.0)
    route_distance_range: tuple[float, float] = (3.0, 7.0)
    early_route_probabilities: tuple[float, float, float, float] = (0.35, 0.35, 0.20, 0.10)
    early_route_levels: int = 3
    heading_spread_range: tuple[float, float] = (0.15, 0.35)

