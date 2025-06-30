# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg
from ..single_terrain_generator_cfg import SingleTerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

DIVERSE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=19,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "mesh_rail": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.2, rail_thickness_range=(1.0, 2.0), rail_height_range=(0.05, 0.4), platform_width=3.0
        ),
        "mesh_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.2, pit_depth_range=(0.05, 0.4), platform_width=3.0
        ),
        "mesh_box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.1, 0.5), platform_width=2.0
        ),
        "mesh_gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2, gap_width_range=(0.05, 0.3), platform_width=2.0,
        ),
        "mesh_repeat_object": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.1,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                size=(0.2, 0.2), num_objects=8, height=0.1
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                size=(0.5, 0.5), num_objects=20, height=0.5
            )
        ),
    },
)

COST_MAP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_height=20.0,
    num_rows=10,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "linear_stairs_ground": terrain_gen.MeshLinearStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            num_steps=10,
            step_width=0.2,
            stairs_width=2.0,
            stairs_length=6.0,
            origin_offset_y=4.5),
        "linear_stairs_top": terrain_gen.MeshLinearStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.15),
            num_steps=10,
            step_width=0.2,
            stairs_width=2.0,
            stairs_length=6.0,
            origin_offset_y=0.0),
        "twosided_rails": terrain_gen.MeshTwosidedRailsTerrainCfg(
            proportion=1.0,
            rail_thickness=0.5,
            rail_height_range=(0.05, 0.6),
            rail_width=4.0,
            platform_width=2.0
        ),
        "room": terrain_gen.MeshRoomTerrainCfg(
            proportion=1.0,
            room_size=6.0,
            wall_thickness=0.2,
            wall_height=3.0,
            door_width_range=(0.5, 1.5),
            door_height=2.0)
    }
)

MOUNTAIN_TERRAINS_CFG = SingleTerrainGeneratorCfg(
    goal_num_rows=2,
    goal_num_cols=2,
    goal_grid_area_size= (60.0, 60.0),
    total_terrain_levels=5,
    distance_increment_per_level=10.0,
    origins_per_level=8,
    obstacles_generator_config=SingleTerrainGeneratorCfg.ObstaclesGeneratorConfig(
        scale=20.0,
        amplitudes=[0.5, 0.3, 0.5, 1.0, 1.0],
        lacunarity=2.0,
        threshold=0.85,
        seed=1,
        size_range=(0.1, 1.0),
        obstacles_types=["cube", "cylinder", "sphere"]
    ),

    terrain_config=terrain_gen.HfMountainTerrainCfg(
        size=(150.0, 150.0),
        mountain_height_range=(-5.0, 5.0),
        scale=1000.0,
        amplitudes=[0.4, 1.0, 0.2, 0.1, 0.0, 0.0, 0.01, 0.0, 0.002, 0.0, 0.0005],
        lacunarity=2.0,
        seed=7,
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
)

"""Rough terrains configuration."""
