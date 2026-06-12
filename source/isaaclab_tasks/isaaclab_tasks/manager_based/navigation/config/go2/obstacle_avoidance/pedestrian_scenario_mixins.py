"""Pedestrian-flow / pedestrian-crossing scenarios for Go2 obstacle avoidance.

Combines the corridor terrains (:mod:`pedestrian_terrains`), pedestrian rigid-object
collections (:mod:`pedestrian_scene`), corridor goal commands and pedestrian-crowd
events/curriculum (``mdp.pedestrian_commands`` / ``mdp.events`` / ``mdp.curriculums``) with
each of the three existing observation variants (base / temporal-lidar / temporal-lidar +
prediction) via multiple inheritance — mirroring how :mod:`temporal_lidar_env_cfg` overrides
only ``observations`` on top of :class:`ObstacleAvoidanceEnvCfg`.

Two scenario families:

- **Flow** (scenarios a/b: with/against pedestrian flow): goal is sampled up- or downstream of
  the robot's spawn along the corridor's flow axis — whichever direction is sampled relative to
  the crowd's flow direction determines whether the episode is "with-flow" or "against-flow".
- **Crossing** (scenario c): robot spawns on one side of a perpendicular pedestrian-flow
  corridor, goal is on the opposite side.
"""

from __future__ import annotations

import math

from isaaclab.assets.rigid_object_collection import RigidObjectCollectionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .obstacle_avoidance_env_cfg import (
    COMMAND_RESAMPLING_TIME_S,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObstacleAvoidanceEnvCfg,
    ObstacleAvoidanceSceneCfg,
)
from .pedestrian_scene import (
    ENABLE_PEDESTRIAN_VISUAL_MESHES,
    PedestrianCollectionCfg,
    PedestrianVisualCollectionCfg,
)
from .pedestrian_terrains import (
    PEDESTRIAN_CROSSING_CORRIDOR,
    PEDESTRIAN_CURRICULUM_MAX_LEVEL,
    PEDESTRIAN_FLOW_CORRIDOR,
)
from .temporal_lidar_env_cfg import TemporalLidarObservationsCfg, TemporalLidarPredictionObservationsCfg

# ---------------------------------------------------------------------------
# Pedestrian curriculum ranges
# ---------------------------------------------------------------------------

# Flow corridor (scenarios a/b): moderate density, ramps from a sparse/slow crowd to a
# dense/fast one as terrain_levels increases.
FLOW_COUNT_RANGE_LOW = (2, 4)
FLOW_COUNT_RANGE_HIGH = (8, 12)
FLOW_SPEED_RANGE_LOW = (0.3, 0.6)
FLOW_SPEED_RANGE_HIGH = (0.8, 1.4)

# Crossing corridor (scenario c): denser flow since the robot must thread across it.
CROSSING_COUNT_RANGE_LOW = (3, 6)
CROSSING_COUNT_RANGE_HIGH = (10, 12)
CROSSING_SPEED_RANGE_LOW = (0.4, 0.7)
CROSSING_SPEED_RANGE_HIGH = (1.0, 1.6)


# ---------------------------------------------------------------------------
# Flow scenario fragments (scenarios a/b: with/against pedestrian flow)
# ---------------------------------------------------------------------------

@configclass
class _PedestrianFlowSceneCfg:
    terrain: TerrainImporterCfg = ObstacleAvoidanceSceneCfg.terrain.replace(
        terrain_generator=PEDESTRIAN_FLOW_CORRIDOR
    )
    pedestrians: RigidObjectCollectionCfg = PedestrianCollectionCfg()
    pedestrian_visuals: RigidObjectCollectionCfg | None = (
        PedestrianVisualCollectionCfg() if ENABLE_PEDESTRIAN_VISUAL_MESHES else None
    )


@configclass
class _PedestrianFlowCommandsCfg:
    pose_2d_command: nav_mdp.CorridorFlowPose2dCommandCfg = nav_mdp.CorridorFlowPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        stationary_prob=0.1,
        ranges=nav_mdp.CorridorFlowPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            heading=(-math.pi, math.pi),
            pos_z=(0.3, 0.4),
        ),
        resampling_time_range=(COMMAND_RESAMPLING_TIME_S, COMMAND_RESAMPLING_TIME_S),
        goal_distance_range=(4.0, 8.0),
        corridor_half_length=9.0,
        corridor_half_width=2.0,
        debug_vis=True,
    )


@configclass
class _PedestrianFlowEventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_pedestrians = EventTerm(
        func=nav_mdp.reset_pedestrian_crowd,
        mode="reset",
        params={"flow_dir": 1.0},
    )


@configclass
class _PedestrianFlowCurriculumCfg:
    discrete_obstacles = CurrTerm(func=mdp.GetTerrainLevel, params={"terrain_name": "ped_corridor"})
    concentric_maze: CurrTerm | None = None

    pedestrian_density = CurrTerm(
        func=nav_mdp.pedestrian_crowd_curriculum,
        params={
            "max_level": PEDESTRIAN_CURRICULUM_MAX_LEVEL,
            "count_range_low": FLOW_COUNT_RANGE_LOW,
            "count_range_high": FLOW_COUNT_RANGE_HIGH,
            "speed_range_low": FLOW_SPEED_RANGE_LOW,
            "speed_range_high": FLOW_SPEED_RANGE_HIGH,
        },
    )


@configclass
class PedestrianFlowSceneCfg(_PedestrianFlowSceneCfg, ObstacleAvoidanceSceneCfg):
    pass


@configclass
class PedestrianFlowCommandsCfg(_PedestrianFlowCommandsCfg, CommandsCfg):
    pass


@configclass
class PedestrianFlowEventCfg(_PedestrianFlowEventCfg, EventCfg):
    pass


@configclass
class PedestrianFlowCurriculumCfg(_PedestrianFlowCurriculumCfg, CurriculumCfg):
    pass


# ---------------------------------------------------------------------------
# Crossing scenario fragments (scenario c: robot crosses the pedestrian flow)
# ---------------------------------------------------------------------------

@configclass
class _PedestrianCrossingSceneCfg:
    terrain: TerrainImporterCfg = ObstacleAvoidanceSceneCfg.terrain.replace(
        terrain_generator=PEDESTRIAN_CROSSING_CORRIDOR
    )
    pedestrians: RigidObjectCollectionCfg = PedestrianCollectionCfg()
    pedestrian_visuals: RigidObjectCollectionCfg | None = (
        PedestrianVisualCollectionCfg() if ENABLE_PEDESTRIAN_VISUAL_MESHES else None
    )


@configclass
class _PedestrianCrossingCommandsCfg:
    pose_2d_command: nav_mdp.CorridorCrossingPose2dCommandCfg = nav_mdp.CorridorCrossingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        stationary_prob=0.1,
        ranges=nav_mdp.CorridorCrossingPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            heading=(-math.pi, math.pi),
            pos_z=(0.3, 0.4),
        ),
        resampling_time_range=(COMMAND_RESAMPLING_TIME_S, COMMAND_RESAMPLING_TIME_S),
        spawn_y=-5.0,
        goal_y=5.0,
        x_range=(-1.5, 1.5),
        debug_vis=True,
    )


@configclass
class _PedestrianCrossingEventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.5, 1.5),
                "y": (-5.5, -4.5),
                "yaw": (math.pi / 2 - 0.5, math.pi / 2 + 0.5),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_pedestrians = EventTerm(
        func=nav_mdp.reset_pedestrian_crowd,
        mode="reset",
        params={"flow_dir": 1.0},
    )


@configclass
class _PedestrianCrossingCurriculumCfg:
    discrete_obstacles = CurrTerm(func=mdp.GetTerrainLevel, params={"terrain_name": "ped_crossing"})
    concentric_maze: CurrTerm | None = None

    pedestrian_density = CurrTerm(
        func=nav_mdp.pedestrian_crowd_curriculum,
        params={
            "max_level": PEDESTRIAN_CURRICULUM_MAX_LEVEL,
            "count_range_low": CROSSING_COUNT_RANGE_LOW,
            "count_range_high": CROSSING_COUNT_RANGE_HIGH,
            "speed_range_low": CROSSING_SPEED_RANGE_LOW,
            "speed_range_high": CROSSING_SPEED_RANGE_HIGH,
        },
    )


@configclass
class PedestrianCrossingSceneCfg(_PedestrianCrossingSceneCfg, ObstacleAvoidanceSceneCfg):
    pass


@configclass
class PedestrianCrossingCommandsCfg(_PedestrianCrossingCommandsCfg, CommandsCfg):
    pass


@configclass
class PedestrianCrossingEventCfg(_PedestrianCrossingEventCfg, EventCfg):
    pass


@configclass
class PedestrianCrossingCurriculumCfg(_PedestrianCrossingCurriculumCfg, CurriculumCfg):
    pass


# ---------------------------------------------------------------------------
# Top-level environment configs — Flow (scenarios a/b)
# ---------------------------------------------------------------------------

@configclass
class PedestrianFlowObstacleAvoidanceEnvCfg(ObstacleAvoidanceEnvCfg):
    """Go2 navigates a corridor with/against a social-force pedestrian flow."""

    scene: PedestrianFlowSceneCfg = PedestrianFlowSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: PedestrianFlowCommandsCfg = PedestrianFlowCommandsCfg()
    events: PedestrianFlowEventCfg = PedestrianFlowEventCfg()
    curriculum: PedestrianFlowCurriculumCfg = PedestrianFlowCurriculumCfg()

    social_force: nav_mdp.SocialForceCrowdCfg = nav_mdp.SocialForceCrowdCfg()
    pedestrian_flow_dir: float = 1.0
    pedestrian_init_count: int = FLOW_COUNT_RANGE_LOW[1]
    pedestrian_init_speed_range: tuple[float, float] = FLOW_SPEED_RANGE_LOW


@configclass
class PedestrianFlowTemporalLidarObstacleAvoidanceEnvCfg(PedestrianFlowObstacleAvoidanceEnvCfg):
    """Pedestrian-flow scenario with temporal-lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()


@configclass
class PedestrianFlowTemporalLidarPredictionObstacleAvoidanceEnvCfg(PedestrianFlowObstacleAvoidanceEnvCfg):
    """Pedestrian-flow scenario with temporal-lidar + next-frame prediction observations."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()


@configclass
class PedestrianFlowObstacleAvoidanceEnvCfg_PLAY(PedestrianFlowObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest pedestrian-density curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True


# ---------------------------------------------------------------------------
# Top-level environment configs — Crossing (scenario c)
# ---------------------------------------------------------------------------

@configclass
class PedestrianCrossingObstacleAvoidanceEnvCfg(ObstacleAvoidanceEnvCfg):
    """Go2 crosses a perpendicular social-force pedestrian flow corridor."""

    scene: PedestrianCrossingSceneCfg = PedestrianCrossingSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: PedestrianCrossingCommandsCfg = PedestrianCrossingCommandsCfg()
    events: PedestrianCrossingEventCfg = PedestrianCrossingEventCfg()
    curriculum: PedestrianCrossingCurriculumCfg = PedestrianCrossingCurriculumCfg()

    social_force: nav_mdp.SocialForceCrowdCfg = nav_mdp.SocialForceCrowdCfg()
    pedestrian_flow_dir: float = 1.0
    pedestrian_init_count: int = CROSSING_COUNT_RANGE_LOW[1]
    pedestrian_init_speed_range: tuple[float, float] = CROSSING_SPEED_RANGE_LOW


@configclass
class PedestrianCrossingTemporalLidarObstacleAvoidanceEnvCfg(PedestrianCrossingObstacleAvoidanceEnvCfg):
    """Pedestrian-crossing scenario with temporal-lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()


@configclass
class PedestrianCrossingTemporalLidarPredictionObstacleAvoidanceEnvCfg(PedestrianCrossingObstacleAvoidanceEnvCfg):
    """Pedestrian-crossing scenario with temporal-lidar + next-frame prediction observations."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()


@configclass
class PedestrianCrossingObstacleAvoidanceEnvCfg_PLAY(PedestrianCrossingObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest pedestrian-density curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True
