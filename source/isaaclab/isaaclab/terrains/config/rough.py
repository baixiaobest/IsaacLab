# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg, FlatPatchSamplingCfg
from ..single_terrain_generator_cfg import SingleTerrainGeneratorCfg
from ..trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

FLAT_PATCH_CFG = FlatPatchSamplingCfg(
    num_patches=1000,
    patch_radius=0.35,
    x_range=(-10, 10.0),
    y_range=(-10.0, 10.0),
    max_height_diff=0.2,
    min_distance=0.0
)

FLAT_PATCH_HEIGHT_LIMITTED_CFG = FlatPatchSamplingCfg(
    num_patches=1000,
    patch_radius=0.35,
    x_range=(-10, 10.0),
    y_range=(-10.0, 10.0),
    z_range=(-0.4, 0.4),  # Limit the height to a small range
    max_height_diff=0.2,
    min_distance=0.0
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
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
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "mesh_rail": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.2, rail_thickness_range=(1.0, 2.0), rail_height_range=(0.05, 0.4), platform_width=3.0,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "mesh_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.2, pit_depth_range=(0.05, 0.4), platform_width=3.0,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "mesh_box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.1, 0.5), platform_width=2.0,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "mesh_gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2, gap_width_range=(0.05, 0.3), platform_width=2.0,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "mesh_repeat_object": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.1,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                size=(0.2, 0.2), num_objects=8, height=0.1
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                size=(0.5, 0.5), num_objects=20, height=0.5
            ),
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
    },
)

NAVIGATION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
        "discrete_obstacles": terrain_gen.HfDiscretePositiveObstaclesTerrainCfg(
            proportion=0.1,
            min_num_low_obstacles=4,
            max_num_low_obstacles=14,
            min_num_high_obstacles=4,
            max_num_high_obstacles=14,
            low_obstacle_max_height=0.4,
            high_obstacle_height_range=(1.0, 2.0),
            obstacle_width_range=(0.5, 1.0),
            platform_width=1.1,
            flat_patch_sampling={"target": FLAT_PATCH_CFG})
    },
)

ROUGH_ONLY = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.06), noise_step=0.01, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_CFG}
        ),
    },
)

DISCRETE_OBSTACLES_ONLY = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "discrete_obstacles": terrain_gen.HfDiscretePositiveObstaclesTerrainCfg(
            proportion=0.1,
            min_num_low_obstacles=0,
            max_num_low_obstacles=4,
            min_num_high_obstacles=0,
            max_num_high_obstacles=10,
            low_obstacle_max_height=0.3,
            high_obstacle_height_range=(1.0, 2.0),
            obstacle_width_range=(0.5, 2.0),
            platform_width=1.1,
            flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG})
    },
)

DISCRETE_OBSTACLES_ROUGH_ONLY = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG}
        ),
        "discrete_obstacles": terrain_gen.HfDiscretePositiveObstaclesTerrainCfg(
            proportion=0.1,
            min_num_low_obstacles=0,
            max_num_low_obstacles=4,
            min_num_high_obstacles=2,
            max_num_high_obstacles=10,
            low_obstacle_max_height=0.3,
            high_obstacle_height_range=(1.0, 2.0),
            obstacle_width_range=(0.5, 2.0),
            platform_width=1.1,
            flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG})
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
    goal_num_rows=5,
    goal_num_cols=5,
    goal_grid_area_size= (60.0, 60.0),
    total_terrain_levels=12,
    distance_increment_per_level=2.0,
    origins_per_level=8,
    obstacles_generator_config=SingleTerrainGeneratorCfg.ObstaclesGeneratorConfig(
        scale=20.0,
        length_pixels=1000,
        width_pixels=1000,
        amplitudes=[0.5, 0.3, 0.5, 1.0, 1.0],
        lacunarity=2.0,
        threshold=0.85,
        seed=1,
        size_range=(0.1, 1.0),
        obstacles_types=["cube", "cylinder", "sphere"]
    ),

    terrain_config=terrain_gen.HfMountainTerrainCfg(
        size=(160.0, 160.0),
        mountain_height_range=(-5.0, 5.0),
        scale=1000.0,
        amplitudes=[0.4, 1.0, 0.2, 0.1, 0.0, 0.0, 0.01, 0.0, 0.002, 0.0, 0.0005],
        lacunarity=2.0,
        seed=7,
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
)

FLAT_TERRAINS_OBSTACLES_CFG = SingleTerrainGeneratorCfg(
    goal_num_rows=5,
    goal_num_cols=5,
    goal_grid_area_size= (110.0, 110.0),
    total_terrain_levels=12,
    distance_increment_per_level=2.0,
    origins_per_level=8,

    obstacles_generator_config=SingleTerrainGeneratorCfg.ObstaclesGeneratorConfig(
        scale=20.0,
        length_pixels=1000,
        width_pixels=1000,
        amplitudes=[0.5, 0.3, 0.5, 1.0, 1.0],
        lacunarity=2.0,
        threshold=0.85,
        seed=53,
        size_range=(0.1, 1.0),
        obstacles_types=["cube", "cylinder", "sphere"],
        goal_region_clearance=4.0
    ),

    terrain_config=terrain_gen.MeshPlaneTerrainCfg(
        size=(170.0, 170.0)
    )
)

FLAT_TERRAINS_CFG = SingleTerrainGeneratorCfg(
    goal_num_rows=5,
    goal_num_cols=5,
    goal_grid_area_size= (60.0, 60.0),
    total_terrain_levels=12,
    distance_increment_per_level=2.0,
    origins_per_level=8,

    obstacles_generator_config=None,

    terrain_config=terrain_gen.MeshPlaneTerrainCfg(
        size=(170.0, 170.0)
    )
)

"""Rough terrains configuration."""
