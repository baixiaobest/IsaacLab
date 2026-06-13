"""Mixed static-obstacle + pedestrian-corridor co-training environment for Go2 obstacle avoidance.

Builds on :mod:`pedestrian_scenario_mixins` and :func:`pedestrian_terrains
.build_mixed_static_pedestrian_corridor`: each env is permanently pinned (at terrain-importer
init) to either a "ped_corridor" column (social-force crowd, flow + crossing scenarios) or a
static "discrete_obstacles"/"concentric_maze" column (no pedestrians). The per-env
``env.is_pedestrian_env`` mask (set in :class:`PedestrianCrowdNavigationEnv`) and the matching
:class:`MixedTerrainPose2dCommand` mask drive all the per-env branching (goal sampling, reset
pose, pedestrian-crowd curriculum/reset), so a single policy co-trains on both terrain families
at once.

The default 50/50 static/pedestrian split (1 "discrete_obstacles" col + 1 "concentric_maze" col +
2 "ped_corridor" cols) comes from
:func:`pedestrian_terrains.build_mixed_static_pedestrian_corridor`'s defaults; pass different
proportions/``num_cols`` to that function for other splits.
"""

from __future__ import annotations

import math

from isaaclab.assets.rigid_object_collection import RigidObjectCollectionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .obstacle_avoidance_env_cfg import (
    COMMAND_RESAMPLING_TIME_S,
    LIDAR_MAX_DISTANCE,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObstacleAvoidanceEnvCfg,
    ObstacleAvoidanceSceneCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .pedestrian_scene import (
    ENABLE_PEDESTRIAN_VISUAL_MESHES,
    PedestrianCollectionCfg,
    PedestrianVisualCollectionCfg,
)
from .pedestrian_scenario_mixins import (
    _CROSSING_NORTH_SPAWN_POSE_RANGE,
    _CROSSING_SOUTH_SPAWN_POSE_RANGE,
    _FLOW_SPAWN_POSE_RANGE,
    _ZERO_VELOCITY_RANGE,
)
from .pedestrian_terrains import (
    PEDESTRIAN_CURRICULUM_MAX_LEVEL,
    build_mixed_static_pedestrian_corridor,
)
from .temporal_lidar_env_cfg import TemporalLidarObservationsCfg, TemporalLidarPredictionObservationsCfg

# Static-env robot reset pose/velocity ranges, copied from EventCfg.reset_base.
_STATIC_SPAWN_POSE_RANGE = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)}

CROSSING_PROB = 0.5
PED_COUNT_RANGE_LOW = (2, 3)
PED_COUNT_RANGE_HIGH = (10, 12)
PED_SPEED_RANGE_LOW = (0.3, 0.7)
PED_SPEED_RANGE_HIGH = (0.9, 1.5)

# ---------------------------------------------------------------------------
# Scenario fragments
# ---------------------------------------------------------------------------

@configclass
class _MixedSceneCfg:
    terrain: TerrainImporterCfg = ObstacleAvoidanceSceneCfg().terrain.replace(
        terrain_generator=build_mixed_static_pedestrian_corridor(
            discrete_obstacles_proportion=2.0,
            concentric_maze_proportion=1.0,
            ped_corridor_proportion=2.0,
            num_cols=5,
        )
    )
    pedestrians: RigidObjectCollectionCfg = PedestrianCollectionCfg()
    pedestrian_visuals: RigidObjectCollectionCfg | None = (
        PedestrianVisualCollectionCfg() if ENABLE_PEDESTRIAN_VISUAL_MESHES else None
    )

    # Overrides the base RayCasterCfg obstacle_scanner: also ray-casts against each env's
    # pedestrian capsules (parked at z=-50, harmless, for static envs).
    obstacle_scanner: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=ObstacleAvoidanceSceneCfg().obstacle_scanner.offset,
        attach_yaw_only=True,
        max_distance=LIDAR_MAX_DISTANCE,
        pattern_cfg=ObstacleAvoidanceSceneCfg().obstacle_scanner.pattern_cfg,
        debug_vis=True,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Pedestrian_.*",
                track_mesh_transforms=True,
                merge_prim_meshes=False,
            ),
        ],
    )


@configclass
class _MixedCommandsCfg:
    pose_2d_command: nav_mdp.MixedTerrainPose2dCommandCfg = nav_mdp.MixedTerrainPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        stationary_prob=0.1,
        ranges=nav_mdp.MixedTerrainPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0), # Not used
            pos_y=(0.0, 0.0), # Not used
            heading=(-math.pi, math.pi),
            pos_z=(0.3, 0.4),
        ),
        resampling_time_range=(COMMAND_RESAMPLING_TIME_S, COMMAND_RESAMPLING_TIME_S),
        # flow-scenario goal (pedestrian-corridor envs)
        goal_distance_range=(4.0, 8.0),
        corridor_half_length=9.0,
        corridor_half_width=2.0,
        # crossing-scenario goal (pedestrian-corridor envs)
        goal_y=5.0,
        crossing_x_range=(-1.5, 1.5),
        debug_vis=True,
    )


@configclass
class _MixedEventCfg:
    # Overrides the parent reset_base: static envs reset uniformly, pedestrian-corridor envs
    # sample the per-env flow/crossing scenario mode.
    reset_base = EventTerm(
        func=nav_mdp.reset_robot_mixed,
        mode="reset",
        params={
            "static_pose_range": _STATIC_SPAWN_POSE_RANGE,
            "static_velocity_range": _ZERO_VELOCITY_RANGE,
            "flow_pose_range": _FLOW_SPAWN_POSE_RANGE,
            "crossing_south_pose_range": _CROSSING_SOUTH_SPAWN_POSE_RANGE,
            "crossing_north_pose_range": _CROSSING_NORTH_SPAWN_POSE_RANGE,
            "pedestrian_velocity_range": _ZERO_VELOCITY_RANGE,
            "crossing_prob": CROSSING_PROB,
        },
    )

    reset_pedestrians = EventTerm(
        func=nav_mdp.reset_pedestrian_crowd,
        mode="reset",
        params={"flow_dir": 1.0},
    )


@configclass
class _MixedCurriculumCfg:
    ped_corridor = CurrTerm(func=mdp.GetTerrainLevel, params={"terrain_name": "ped_corridor"})

    pedestrian_density = CurrTerm(
        func=nav_mdp.pedestrian_crowd_curriculum,
        params={
            "max_level": PEDESTRIAN_CURRICULUM_MAX_LEVEL,
            "count_range_low": PED_COUNT_RANGE_LOW,
            "count_range_high": PED_COUNT_RANGE_HIGH,
            "speed_range_low": PED_SPEED_RANGE_LOW,
            "speed_range_high": PED_SPEED_RANGE_HIGH,
        },
    )


@configclass
class _MixedRewardsCfg:
    pedestrian_collision_penalty = RewTerm(
        func=nav_mdp.pedestrian_capsule_collision_penalty,
        weight=-200.0,
    )


@configclass
class _MixedTerminationsCfg:
    pedestrian_collision = DoneTerm(func=nav_mdp.pedestrian_capsule_collision)


@configclass
class MixedSceneCfg(_MixedSceneCfg, ObstacleAvoidanceSceneCfg):
    pass


@configclass
class MixedCommandsCfg(_MixedCommandsCfg, CommandsCfg):
    pass


@configclass
class MixedEventCfg(_MixedEventCfg, EventCfg):
    pass


@configclass
class MixedCurriculumCfg(_MixedCurriculumCfg, CurriculumCfg):
    pass


@configclass
class MixedRewardsCfg(_MixedRewardsCfg, RewardsCfg):
    pass


@configclass
class MixedTerminationsCfg(_MixedTerminationsCfg, TerminationsCfg):
    pass


# ---------------------------------------------------------------------------
# Top-level environment configs
# ---------------------------------------------------------------------------

@configclass
class MixedObstacleAvoidanceEnvCfg(ObstacleAvoidanceEnvCfg):
    """Go2 co-trains on static obstacle/maze terrain and the pedestrian corridor at once."""

    scene: MixedSceneCfg = MixedSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: MixedCommandsCfg = MixedCommandsCfg()
    events: MixedEventCfg = MixedEventCfg()
    curriculum: MixedCurriculumCfg = MixedCurriculumCfg()
    rewards: MixedRewardsCfg = MixedRewardsCfg()
    terminations: MixedTerminationsCfg = MixedTerminationsCfg()

    social_force: nav_mdp.SocialForceCrowdCfg = nav_mdp.SocialForceCrowdCfg()
    pedestrian_flow_dir: float = 1.0
    pedestrian_init_count: int = PED_COUNT_RANGE_LOW[1]
    pedestrian_init_speed_range: tuple[float, float] = PED_SPEED_RANGE_LOW

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class MixedTemporalLidarObstacleAvoidanceEnvCfg(MixedObstacleAvoidanceEnvCfg):
    """Mixed static/pedestrian co-training with temporal-lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()


@configclass
class MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg(MixedObstacleAvoidanceEnvCfg):
    """Mixed static/pedestrian co-training with temporal-lidar + next-frame prediction observations."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()


@configclass
class MixedObstacleAvoidanceEnvCfg_PLAY(MixedObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True


@configclass
class MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY(MixedTemporalLidarObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True


@configclass
class MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg_PLAY(MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True
