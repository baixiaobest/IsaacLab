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
    LIDAR_MAX_DISTANCE,
    LIDAR_FOV_DEG,
    NUM_LIDAR_RAYS,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObstacleAvoidanceEnvCfg,
    ObstacleAvoidanceSceneCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .two_cloud_lidar_env import TwoCloudLidarCfg
from .pedestrian_scene import (
    ENABLE_PEDESTRIAN_VISUAL_MESHES,
    PedestrianCollectionCfg,
    PedestrianVisualCollectionCfg,
    make_pedestrian_collection_cfg,
    make_pedestrian_visual_collection_cfg,
)
from .pedestrian_scenario_mixins import (
    _CROSSING_NORTH_SPAWN_POSE_RANGE,
    _CROSSING_SOUTH_SPAWN_POSE_RANGE,
    _FLOW_SPAWN_POSE_RANGE,
    _ZERO_VELOCITY_RANGE,
)
from .pedestrian_terrains import (
    PEDESTRIAN_CURRICULUM_MAX_LEVEL,
    PEDESTRIAN_CORRIDOR,
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
PED_LATERAL_HEADING_MAX_LOW = 0.0
PED_LATERAL_HEADING_MAX_HIGH = math.radians(12.0)

EPISODE_LENGTH = 15.0
RESAMPLING_TIME_RANGE = (15.1, 15.1)

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
        stationary_prob=0.0,
        ranges=nav_mdp.MixedTerrainPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0), # Not used
            pos_y=(0.0, 0.0), # Not used
            heading=(-math.pi, math.pi),
            pos_z=(0.3, 0.4),
        ),
        resampling_time_range=RESAMPLING_TIME_RANGE,
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
            "lateral_heading_max_low": PED_LATERAL_HEADING_MAX_LOW,
            "lateral_heading_max_high": PED_LATERAL_HEADING_MAX_HIGH,
        },
    )


@configclass
class _MixedRewardsCfg:
    # Override the inherited gated heading penalty with a bounded alignment reward.
    # This retains the 1 m approach region while removing the incentive to remain
    # just outside it to avoid a negative heading cost.
    orientation_tracking = RewTerm(
        func=nav_mdp.heading_command_error_tanh_within_range,
        weight=0.5,
        params={
            "command_name": "pose_2d_command",
            "std": 1.0,
            "range": 1.0,
        },
    )

    # One-off +5 completion bonus at the exact pose-and-stationary success state.
    # RewardManager scales term weights by the 80 ms high-level step duration.
    goal_reached_once = RewTerm(
        func=nav_mdp.pose_2d_command_goal_reached_once_with_velocity,
        weight=5.0/0.08, # 0.08s for update interval.
        params={
            "command_name": "pose_2d_command",
            "distance_threshold": 0.5,
            "angular_threshold": 0.2,
            "velocity_threshold": 0.1,
            "stay_for_seconds": 0.1,
        },
    )

    pedestrian_collision_penalty = RewTerm(
        func=nav_mdp.pedestrian_capsule_collision_penalty,
        weight=-200.0,
    )

    pedestrian_closest_approach = RewTerm(
        func=nav_mdp.pedestrian_closest_approach_penalty,
        weight=-0.0,
        params={
            # CPA assumes constant velocities only over this short look-ahead window.
            "horizon": 1.5,
            # Surface clearance (not centre distance) below which the risk starts.
            "safe_clearance": 0.5,
            "time_scale": 0.75,
            "min_relative_speed": 0.05,
        },
    )

    # Penalize traversing close to the forward half-plane of a moving pedestrian.
    # The relative-speed factor makes a quick pass accumulate approximately the same
    # cost as a slow pass through the same region.
    pedestrian_front_proximity_speed = RewTerm(
        func=nav_mdp.pedestrian_proximity_speed_penalty,
        weight=-0.0,
        params={
            "sigma": 2.0,
            "in_front_only": True,
            "min_agent_speed": 0.1,
        },
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
        self.episode_length_s = EPISODE_LENGTH


@configclass
class MixedTemporalLidarObstacleAvoidanceEnvCfg(MixedObstacleAvoidanceEnvCfg):
    """Mixed static/pedestrian co-training with temporal-lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()
    two_cloud_lidar_enabled: bool = True
    two_cloud_lidar: TwoCloudLidarCfg = TwoCloudLidarCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle_scanner.update_period = 0.0
        self.scene.obstacle_scanner.pattern_cfg.horizontal_res = LIDAR_FOV_DEG / (NUM_LIDAR_RAYS - 1)
        self.scene.obstacle_scanner.debug_vis = False


@configclass
class MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg(MixedObstacleAvoidanceEnvCfg):
    """Mixed static/pedestrian co-training with temporal-lidar + next-frame prediction observations."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()
    two_cloud_lidar_enabled: bool = True
    two_cloud_lidar: TwoCloudLidarCfg = TwoCloudLidarCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle_scanner.update_period = 0.0
        self.scene.obstacle_scanner.pattern_cfg.horizontal_res = LIDAR_FOV_DEG / (NUM_LIDAR_RAYS - 1)
        self.scene.obstacle_scanner.debug_vis = False


@configclass
class MixedObstacleAvoidanceEnvCfg_PLAY(MixedObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        # self.scene.num_envs = 16
        # self.scene.env_spacing = 2.5
        # self.scene.terrain.max_init_terrain_level = 0
        # self.observations.policy.enable_corruption = False
        # self.actions.pre_trained_policy_action.debug_vis = True


@configclass
class MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY(MixedTemporalLidarObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        # self.scene.num_envs = 16
        # self.scene.env_spacing = 2.5
        # self.scene.terrain.max_init_terrain_level = 0
        # self.observations.policy.enable_corruption = False
        # self.actions.pre_trained_policy_action.debug_vis = True


@configclass
class MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg_PLAY(MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg):
    """Play variant: fewer envs, starts at the lowest curriculum level."""

    def __post_init__(self):
        super().__post_init__()
        # self.scene.num_envs = 16
        # self.scene.env_spacing = 2.5
        # self.scene.terrain.max_init_terrain_level = 0
        # self.observations.policy.enable_corruption = False
        # self.actions.pre_trained_policy_action.debug_vis = True


# ---------------------------------------------------------------------------
# Dynamic-crowd evaluation overlay
# ---------------------------------------------------------------------------

EVALUATION_CROWD_SPEED_RANGE = (0.9, 1.5)
"""Pedestrian desired-speed range used by the standardized dynamic-crowd benchmark."""

EVALUATION_CROWD_LATERAL_HEADING_MAX = PED_LATERAL_HEADING_MAX_HIGH
"""Fixed maximum pedestrian heading offset used by the dynamic-crowd benchmark [rad]."""

EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD = 0.5
"""Maximum goal distance for an evaluation success [m]."""

EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD = math.radians(45.0)
"""Maximum goal-heading error for an evaluation success [rad]."""

EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD = 0.3
"""Maximum horizontal robot speed for an evaluation success [m/s]."""

EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS = 0.1
"""Required continuous time satisfying the evaluation goal condition [s]."""

EVALUATION_SCENARIO_CODES = {"crossing": 0, "with_flow": 1, "against_flow": 2}
"""Stable scenario names and codes used by the dynamic-crowd benchmark and its artifacts."""


def configure_dynamic_crowd_evaluation(env_cfg: MixedObstacleAvoidanceEnvCfg) -> MixedObstacleAvoidanceEnvCfg:
    """Overlay a mixed configuration with the deterministic dynamic-crowd benchmark setup.

    The supplied config is mutated before environment construction. Observation/action variants
    are inherited unchanged, so base, temporal-lidar, and prediction policies remain compatible.
    Per-environment crowd counts and scenario types are installed later with
    :func:`install_dynamic_crowd_evaluation_profiles`.
    """
    env_cfg.scene.terrain.terrain_generator = PEDESTRIAN_CORRIDOR
    # Spread vector environments across all generated corridor tiles at startup.  The
    # evaluation deliberately disables the terrain curriculum below, so these initial
    # levels stay fixed for the entire run.  Pinning this to zero placed every robot on
    # the first tile (shown as level 1 in the visualizer).
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.pedestrians = make_pedestrian_collection_cfg(16)
    if ENABLE_PEDESTRIAN_VISUAL_MESHES:
        env_cfg.scene.pedestrian_visuals = make_pedestrian_visual_collection_cfg(16)

    env_cfg.social_force.max_pedestrians = 16
    env_cfg.social_force.lateral_heading_max = EVALUATION_CROWD_LATERAL_HEADING_MAX
    env_cfg.pedestrian_init_count = 2
    env_cfg.pedestrian_init_speed_range = EVALUATION_CROWD_SPEED_RANGE

    # This benchmark accepts a controlled arrival at the goal instead of the stricter
    # training pose.  It is intentionally applied only by this evaluation overlay;
    # the training configuration remains unchanged.
    env_cfg.terminations.goal_reached.params.update(
        {
            "distance_threshold": EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD,
            "angular_threshold": EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD,
            "velocity_threshold": EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD,
            "stay_for_seconds": EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS,
        }
    )

    # Evaluation fixes terrain and crowd difficulty rather than advancing the training curricula.
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.discrete_obstacles = None
    env_cfg.curriculum.concentric_maze = None
    env_cfg.curriculum.ped_corridor = None
    env_cfg.curriculum.pedestrian_density = None

    env_cfg.events.reset_base = EventTerm(
        func=nav_mdp.reset_evaluation_pedestrian_scenario_robot,
        mode="reset",
        params={
            "flow_pose_range": _FLOW_SPAWN_POSE_RANGE,
            "crossing_south_pose_range": _CROSSING_SOUTH_SPAWN_POSE_RANGE,
            "crossing_north_pose_range": _CROSSING_NORTH_SPAWN_POSE_RANGE,
            "velocity_range": _ZERO_VELOCITY_RANGE,
            "speed_range": EVALUATION_CROWD_SPEED_RANGE,
        },
    )
    return env_cfg


def install_dynamic_crowd_evaluation_profiles(
    env,
    pedestrian_counts,
    scenario_codes,
) -> None:
    """Install fixed profile tensors consumed by dynamic-crowd evaluation reset hooks.

    Scenario codes are ``0=crossing``, ``1=with_flow``, and ``2=against_flow``. Inputs must
    contain one assignment per vector environment and are copied to the environment device.
    """
    import torch

    counts = torch.as_tensor(pedestrian_counts, device=env.device, dtype=torch.long)
    scenarios = torch.as_tensor(scenario_codes, device=env.device, dtype=torch.long)
    if counts.numel() != env.num_envs or scenarios.numel() != env.num_envs:
        raise ValueError("Evaluation profiles must provide exactly one count and scenario per environment.")
    if torch.any((counts < 1) | (counts > env.crowd_manager.max_pedestrians)):
        raise ValueError("Evaluation pedestrian counts must fit the configured crowd capacity.")
    if torch.any((scenarios < 0) | (scenarios > 2)):
        raise ValueError("Evaluation scenario codes must be 0 (crossing), 1 (with), or 2 (against).")

    env.evaluation_pedestrian_count = counts
    env.evaluation_scenario = scenarios
    env.evaluation_flow_goal_direction = torch.where(
        scenarios == 1,
        torch.ones_like(scenarios),
        torch.where(scenarios == 2, -torch.ones_like(scenarios), torch.zeros_like(scenarios)),
    )
