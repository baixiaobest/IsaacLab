"""Unified pedestrian scenario for Go2 obstacle avoidance (co-trains flow + crossing).

A single corridor terrain (:mod:`pedestrian_terrains`) hosts BOTH scenario families on the
same envs, with the per-episode scenario selected per-env at reset:

- **Flow** (scenarios a/b: with/against pedestrian flow): goal is sampled up- or downstream of
  the robot's spawn along the corridor's flow axis (local-x) — whichever direction is sampled
  relative to the crowd's flow direction makes the episode "with-flow" or "against-flow".
- **Crossing** (scenario c): robot spawns near one edge of the corridor and the goal is placed
  on the opposite side, so it must thread across the perpendicular pedestrian flow.

The pedestrian crowd is scenario-independent (it always flows along local-x), so both scenarios
share one crowd, one terrain, one curriculum, and one set of env configs — only the robot's
spawn pose (``reset_pedestrian_scenario_robot``) and goal sampling
(:class:`CorridorPedestrianPose2dCommand`) branch on the sampled scenario mode. This halves the
number of mixins/registrations versus separate flow and crossing families while keeping both
scenarios explicit. The three observation variants (base / temporal-lidar / temporal-lidar +
prediction) are layered on top via inheritance, mirroring :mod:`temporal_lidar_env_cfg`.
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
from .pedestrian_terrains import PEDESTRIAN_CORRIDOR, PEDESTRIAN_CURRICULUM_MAX_LEVEL
from .temporal_lidar_env_cfg import TemporalLidarObservationsCfg, TemporalLidarPredictionObservationsCfg

# ---------------------------------------------------------------------------
# Pedestrian curriculum ranges (merged across flow + crossing scenarios)
# ---------------------------------------------------------------------------
# Both scenarios share one crowd, so one count/speed ramp covers both. The high end is taken
# from the denser crossing scenario; the low end keeps early training sparse/slow.
PED_COUNT_RANGE_LOW = (9, 12)
PED_COUNT_RANGE_HIGH = (9, 12)
PED_SPEED_RANGE_LOW = (0.3, 0.7)
PED_SPEED_RANGE_HIGH = (0.9, 1.5)

# Probability that a given reset spawns a *crossing* episode (vs. a flow episode).
CROSSING_PROB = 0.5

# Robot spawn pose ranges (corridor-local, relative to the env/terrain origin).
_FLOW_SPAWN_POSE_RANGE = {"x": (-1.0, 1.0), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)}
# Crossing south start: spawn near the south edge facing +y (north). Crossing north start:
# spawn near the north edge facing -y (south). Randomizing the crossing direction lets the
# robot see the crowd (which always flows along +x) sweep across its path from both sides.
_CROSSING_SOUTH_SPAWN_POSE_RANGE = {
    "x": (-1.5, 1.5),
    "y": (-5.5, -4.5),
    "yaw": (math.pi / 2 - 0.5, math.pi / 2 + 0.5),
}
_CROSSING_NORTH_SPAWN_POSE_RANGE = {
    "x": (-1.5, 1.5),
    "y": (4.5, 5.5),
    "yaw": (-math.pi / 2 - 0.5, -math.pi / 2 + 0.5),
}
_ZERO_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Scenario fragments
# ---------------------------------------------------------------------------

@configclass
class _PedestrianSceneCfg:
    terrain: TerrainImporterCfg = ObstacleAvoidanceSceneCfg().terrain.replace(
        terrain_generator=PEDESTRIAN_CORRIDOR
    )
    pedestrians: RigidObjectCollectionCfg = PedestrianCollectionCfg()
    pedestrian_visuals: RigidObjectCollectionCfg | None = (
        PedestrianVisualCollectionCfg() if ENABLE_PEDESTRIAN_VISUAL_MESHES else None
    )

    # Overrides the base RayCasterCfg obstacle_scanner: also ray-casts against each env's
    # pedestrian capsules (per-env isolated — env A's rays never hit env B's pedestrians).
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
class _PedestrianCommandsCfg:
    pose_2d_command: nav_mdp.CorridorPedestrianPose2dCommandCfg = nav_mdp.CorridorPedestrianPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        stationary_prob=0.1,
        ranges=nav_mdp.CorridorPedestrianPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            heading=(-math.pi, math.pi),
            pos_z=(0.3, 0.4),
        ),
        resampling_time_range=(COMMAND_RESAMPLING_TIME_S, COMMAND_RESAMPLING_TIME_S),
        # flow-scenario goal
        goal_distance_range=(4.0, 8.0),
        corridor_half_length=9.0,
        corridor_half_width=2.0,
        # crossing-scenario goal
        goal_y=5.0,
        crossing_x_range=(-1.5, 1.5),
        debug_vis=True,
    )


@configclass
class _PedestrianEventCfg:
    # Overrides the parent reset_base: samples the per-env scenario mode and spawns the robot
    # from the flow or crossing pose range accordingly.
    reset_base = EventTerm(
        func=nav_mdp.reset_pedestrian_scenario_robot,
        mode="reset",
        params={
            "flow_pose_range": _FLOW_SPAWN_POSE_RANGE,
            "crossing_south_pose_range": _CROSSING_SOUTH_SPAWN_POSE_RANGE,
            "crossing_north_pose_range": _CROSSING_NORTH_SPAWN_POSE_RANGE,
            "velocity_range": _ZERO_VELOCITY_RANGE,
            "crossing_prob": CROSSING_PROB,
        },
    )

    reset_pedestrians = EventTerm(
        func=nav_mdp.reset_pedestrian_crowd,
        mode="reset",
        params={"flow_dir": 1.0},
    )


@configclass
class _PedestrianCurriculumCfg:
    discrete_obstacles = CurrTerm(func=mdp.GetTerrainLevel, params={"terrain_name": "ped_corridor"})
    concentric_maze: CurrTerm | None = None

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
class PedestrianSceneCfg(_PedestrianSceneCfg, ObstacleAvoidanceSceneCfg):
    pass


@configclass
class PedestrianCommandsCfg(_PedestrianCommandsCfg, CommandsCfg):
    pass


@configclass
class PedestrianEventCfg(_PedestrianEventCfg, EventCfg):
    pass


@configclass
class PedestrianCurriculumCfg(_PedestrianCurriculumCfg, CurriculumCfg):
    pass


@configclass
class _PedestrianRewardsCfg:
    pedestrian_collision_penalty = RewTerm(
        func=nav_mdp.pedestrian_capsule_collision_penalty,
        weight=-200.0,
    )


@configclass
class _PedestrianTerminationsCfg:
    pedestrian_collision = DoneTerm(func=nav_mdp.pedestrian_capsule_collision)


@configclass
class PedestrianRewardsCfg(_PedestrianRewardsCfg, RewardsCfg):
    pass


@configclass
class PedestrianTerminationsCfg(_PedestrianTerminationsCfg, TerminationsCfg):
    pass


# ---------------------------------------------------------------------------
# Top-level environment configs
# ---------------------------------------------------------------------------

@configclass
class PedestrianObstacleAvoidanceEnvCfg(ObstacleAvoidanceEnvCfg):
    """Go2 co-trains flow (with/against) and crossing social-force pedestrian scenarios."""

    scene: PedestrianSceneCfg = PedestrianSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: PedestrianCommandsCfg = PedestrianCommandsCfg()
    events: PedestrianEventCfg = PedestrianEventCfg()
    curriculum: PedestrianCurriculumCfg = PedestrianCurriculumCfg()
    rewards: PedestrianRewardsCfg = PedestrianRewardsCfg()
    terminations: PedestrianTerminationsCfg = PedestrianTerminationsCfg()

    social_force: nav_mdp.SocialForceCrowdCfg = nav_mdp.SocialForceCrowdCfg()
    pedestrian_flow_dir: float = 1.0
    pedestrian_init_count: int = PED_COUNT_RANGE_LOW[1]
    pedestrian_init_speed_range: tuple[float, float] = PED_SPEED_RANGE_LOW

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0

@configclass
class PedestrianTemporalLidarObstacleAvoidanceEnvCfg(PedestrianObstacleAvoidanceEnvCfg):
    """Unified pedestrian scenario with temporal-lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()


@configclass
class PedestrianTemporalLidarPredictionObstacleAvoidanceEnvCfg(PedestrianObstacleAvoidanceEnvCfg):
    """Unified pedestrian scenario with temporal-lidar + next-frame prediction observations."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()


@configclass
class PedestrianObstacleAvoidanceEnvCfg_PLAY(PedestrianObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest pedestrian-density curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True
