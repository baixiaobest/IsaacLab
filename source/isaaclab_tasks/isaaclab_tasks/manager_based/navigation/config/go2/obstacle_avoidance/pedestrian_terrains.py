"""Corridor terrain generators for pedestrian-flow obstacle-avoidance scenarios.

Both terrains use a single sub-terrain type repeated across ``num_rows`` difficulty
levels with identical geometry — the per-env :attr:`terrain.terrain_levels` value (driven
by the existing ``pose_2d_command_terrain_curriculum``) does not change the terrain shape,
only the pedestrian density/speed via :func:`isaaclab_tasks.manager_based.navigation.mdp
.curriculums.pedestrian_crowd_curriculum`.
"""

from __future__ import annotations

from isaaclab.terrains import FlatPatchSamplingCfg, HfDiscretePositiveObstaclesTerrainCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import FLAT_PATCH_HEIGHT_LIMITTED_CFG

# Number of curriculum difficulty levels (rows). Pedestrian count/speed ranges are
# linearly interpolated over terrain_levels in [0, PEDESTRIAN_CURRICULUM_MAX_LEVEL].
PEDESTRIAN_CURRICULUM_NUM_LEVELS = 10
PEDESTRIAN_CURRICULUM_MAX_LEVEL = PEDESTRIAN_CURRICULUM_NUM_LEVELS - 1

# Sparse static obstacles shared by both corridor terrains (lidar remains meaningfully
# exercised alongside the dynamic pedestrians).
_SPARSE_OBSTACLE_KWARGS = dict(
    min_num_low_obstacles=0,
    max_num_low_obstacles=2,
    min_num_high_obstacles=0,
    max_num_high_obstacles=2,
    low_obstacle_max_height=0.3,
    high_obstacle_height_range=(1.0, 1.5),
    obstacle_width_range=(0.3, 0.6),
    platform_width=1.1,
)


# ---------------------------------------------------------------------------
# Flow corridor (scenarios a/b: with/against pedestrian flow)
# ---------------------------------------------------------------------------
# Corridor local-x in roughly [-10, 10] (20 m long, flow axis), local-y in roughly
# [-3, 3] (6 m wide).

PEDESTRIAN_FLOW_CORRIDOR = TerrainGeneratorCfg(
    size=(20.0, 6.0),
    border_width=10.0,
    num_rows=PEDESTRIAN_CURRICULUM_NUM_LEVELS,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "ped_corridor": HfDiscretePositiveObstaclesTerrainCfg(
            proportion=1.0,
            **_SPARSE_OBSTACLE_KWARGS,
            flat_patch_sampling={
                "target": FLAT_PATCH_HEIGHT_LIMITTED_CFG,
                "goal_west": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-9.0, -7.0), y_range=(-2.0, 2.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                "goal_east": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(7.0, 9.0), y_range=(-2.0, 2.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
            },
        ),
    },
)


# ---------------------------------------------------------------------------
# Crossing corridor (scenario c: robot crosses a pedestrian flow corridor)
# ---------------------------------------------------------------------------
# Pedestrians flow along local-x in roughly [-5, 5] (10 m), robot crosses along local-y
# in roughly [-7, 7] (14 m).

PEDESTRIAN_CROSSING_CORRIDOR = TerrainGeneratorCfg(
    size=(10.0, 14.0),
    border_width=10.0,
    num_rows=PEDESTRIAN_CURRICULUM_NUM_LEVELS,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "ped_crossing": HfDiscretePositiveObstaclesTerrainCfg(
            proportion=1.0,
            **_SPARSE_OBSTACLE_KWARGS,
            flat_patch_sampling={
                "target": FLAT_PATCH_HEIGHT_LIMITTED_CFG,
                "spawn_south": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(-6.0, -4.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                "goal_north": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(4.0, 6.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
            },
        ),
    },
)
