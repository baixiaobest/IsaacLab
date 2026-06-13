"""Corridor terrain generator for pedestrian obstacle-avoidance scenarios.

A single corridor terrain hosts BOTH the flow (with/against) and crossing scenarios — the
pedestrian crowd always flows along the corridor's local-x axis, and only the robot's
spawn/goal placement differs between scenarios (along-x for flow, across in local-y for
crossing). The per-episode scenario is selected per-env at reset (see
``reset_pedestrian_scenario_robot`` / ``CorridorPedestrianPose2dCommand``), so both scenarios
co-train on this one terrain.

The terrain uses a single sub-terrain type repeated across ``num_rows`` difficulty levels with
identical geometry — the per-env :attr:`terrain.terrain_levels` value (driven by the existing
``pose_2d_command_terrain_curriculum``) does not change the terrain shape, only the pedestrian
density/speed via :func:`isaaclab_tasks.manager_based.navigation.mdp.curriculums
.pedestrian_crowd_curriculum`.
"""

from __future__ import annotations

from isaaclab.terrains import (
    FlatPatchSamplingCfg,
    HfConcentricMazeTerrainCfg,
    HfDiscretePositiveObstaclesTerrainCfg,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.config.rough import FLAT_PATCH_HEIGHT_LIMITTED_CFG

# Number of curriculum difficulty levels (rows). Pedestrian count/speed ranges are
# linearly interpolated over terrain_levels in [0, PEDESTRIAN_CURRICULUM_MAX_LEVEL].
PEDESTRIAN_CURRICULUM_NUM_LEVELS = 10
PEDESTRIAN_CURRICULUM_MAX_LEVEL = PEDESTRIAN_CURRICULUM_NUM_LEVELS - 1

# Sparse static obstacles shared by both corridor terrains (lidar remains meaningfully
# exercised alongside the dynamic pedestrians).
_SPARSE_OBSTACLE_KWARGS = dict(
    min_num_low_obstacles=0,
    max_num_low_obstacles=0,
    min_num_high_obstacles=0,
    max_num_high_obstacles=0,
    low_obstacle_max_height=0,
    high_obstacle_height_range=(1.0, 1.5),
    obstacle_width_range=(0.3, 0.6),
    platform_width=1.1,
)


# ---------------------------------------------------------------------------
# Unified corridor (hosts both the flow a/b and crossing c scenarios)
# ---------------------------------------------------------------------------
# Pedestrians always flow along local-x in roughly [-10, 10] (20 m long). The corridor is
# wide enough (local-y in roughly [-7, 7], 14 m) that the crossing scenario's robot can
# spawn near one edge (y ~ -5) and reach a goal near the other (y ~ +5) across the flow,
# while the flow scenario places its goal up/downstream along local-x. The named flat
# patches cover both scenarios' goal regions (goals are sampled analytically by
# ``CorridorPedestrianPose2dCommand``; the patches are kept for debug/inspection parity).

PEDESTRIAN_CORRIDOR = TerrainGeneratorCfg(
    size=(20.0, 14.0),
    border_width=10.0,
    num_rows=PEDESTRIAN_CURRICULUM_NUM_LEVELS,
    num_cols=1,
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
                # Flow scenario goal regions (up/downstream along local-x).
                "goal_west": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(-9.0, -7.0), y_range=(-2.0, 2.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                "goal_east": FlatPatchSamplingCfg(
                    num_patches=200, patch_radius=0.4, x_range=(7.0, 9.0), y_range=(-2.0, 2.0),
                    z_range=(-0.2, 0.2), max_height_diff=0.1,
                ),
                # Crossing scenario spawn/goal regions (across the flow along local-y).
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


# ---------------------------------------------------------------------------
# Mixed static + pedestrian corridor (co-trains static-obstacle columns with
# pedestrian-corridor columns on one terrain generator)
# ---------------------------------------------------------------------------
# The static sub-terrains reuse the "discrete_obstacles"/"concentric_maze" kwargs from
# DISCRETE_OBSTACLES_MAZE (12x12m) — the Hf* generators are size-agnostic, so they're reused
# here at the corridor's (20, 14) size.
_DISCRETE_OBSTACLES_KWARGS = dict(
    min_num_low_obstacles=0,
    max_num_low_obstacles=0,
    min_num_high_obstacles=0,
    max_num_high_obstacles=15,
    low_obstacle_max_height=0.3,
    high_obstacle_height_range=(1.0, 2.0),
    obstacle_width_range=(0.3, 1.5),
    platform_width=1.1,
)
_CONCENTRIC_MAZE_KWARGS = dict(
    fence_height_range=(0.5, 1.5),
    fence_spacing_range=(2.0, 3.0),
    opening_width_range=(1.0, 2.0),
    num_openings_range=(1, 3),
)

# Flat-patch sampling shared by the "ped_corridor" sub-terrain in PEDESTRIAN_CORRIDOR and in
# build_mixed_static_pedestrian_corridor (goal regions for the flow/crossing scenarios).
_PED_CORRIDOR_FLAT_PATCH_SAMPLING = {
    "target": FLAT_PATCH_HEIGHT_LIMITTED_CFG,
    "goal_west": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(-9.0, -7.0), y_range=(-2.0, 2.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
    "goal_east": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(7.0, 9.0), y_range=(-2.0, 2.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
    "spawn_south": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(-6.0, -4.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
    "goal_north": FlatPatchSamplingCfg(
        num_patches=200, patch_radius=0.4, x_range=(-1.5, 1.5), y_range=(4.0, 6.0),
        z_range=(-0.2, 0.2), max_height_diff=0.1,
    ),
}


def build_mixed_static_pedestrian_corridor(
    discrete_obstacles_proportion: float = 1.0,
    concentric_maze_proportion: float = 1.0,
    ped_corridor_proportion: float = 2.0,
    num_cols: int = 4,
) -> TerrainGeneratorCfg:
    """Build a terrain generator that splits columns between static obstacle/maze terrain
    (no pedestrians) and the pedestrian corridor (social-force crowd).

    With the defaults (proportions ``1:1:2`` over ``num_cols=4``), columns split as 1
    "discrete_obstacles" + 1 "concentric_maze" + 2 "ped_corridor" -- a 50/50 static/pedestrian
    split with an even split between the two static sub-terrain types. Pass different
    proportions/``num_cols`` for other splits (column assignment is a deterministic
    cumsum-threshold over the normalized proportions, in dict insertion order).
    """
    return TerrainGeneratorCfg(
        size=(20.0, 14.0),
        border_width=10.0,
        num_rows=PEDESTRIAN_CURRICULUM_NUM_LEVELS,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "discrete_obstacles": HfDiscretePositiveObstaclesTerrainCfg(
                proportion=discrete_obstacles_proportion,
                **_DISCRETE_OBSTACLES_KWARGS,
                flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG},
            ),
            "concentric_maze": HfConcentricMazeTerrainCfg(
                proportion=concentric_maze_proportion,
                **_CONCENTRIC_MAZE_KWARGS,
                flat_patch_sampling={"target": FLAT_PATCH_HEIGHT_LIMITTED_CFG},
            ),
            "ped_corridor": HfDiscretePositiveObstaclesTerrainCfg(
                proportion=ped_corridor_proportion,
                **_SPARSE_OBSTACLE_KWARGS,
                flat_patch_sampling=_PED_CORRIDOR_FLAT_PATCH_SAMPLING,
            ),
        },
    )
