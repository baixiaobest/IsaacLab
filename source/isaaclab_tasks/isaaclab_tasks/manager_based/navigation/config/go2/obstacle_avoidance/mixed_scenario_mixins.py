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
from isaaclab.envs.mdp.observations import occupancy_grid_from_lidar
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .held_scan_lidar_env import HeldScanLidarCfg
from .observation_modifiers import policy_base_lin_vel_modifiers, policy_imu_ang_vel_modifiers
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

# The 10 m x 10 m local map is intentionally the final policy/critic term.  The
# encoder model splits its flat input at the tail, so changing this ordering
# would silently encode a proprioceptive term as part of the image instead.
MIXED_OCCUPANCY_GRID_SIZE = 50
MIXED_OCCUPANCY_GRID_RESOLUTION = 0.2

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
    # Smoothly activate the heading penalty near the goal instead of using a
    # discontinuous 1 m gate that can be exploited by circling just outside it.
    orientation_tracking = RewTerm(
        func=nav_mdp.heading_command_error_distance_weighted_abs,
        weight=-0.5,
        params={
            "command_name": "pose_2d_command",
            "distance_std": 1.0,
        },
    )

    pedestrian_collision_penalty = RewTerm(
        func=nav_mdp.pedestrian_capsule_collision_penalty,
        weight=-400.0,
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


@configclass
class MixedOccupancyObservationsCfg(ObservationsCfg):
    """Mixed-task observations with a binary local grid from ``obstacle_scanner``.

    ``obstacle_scanner`` is the mixed scene's multi-mesh ray caster, so the
    same rasterization includes static terrain geometry and pedestrian capsules.
    The row-major occupancy grid remains the final term in both groups for the
    tail-splitting CNN encoder.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        pose_2d_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_2d_command"})
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            modifiers=policy_base_lin_vel_modifiers(),
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            modifiers=policy_imu_ang_vel_modifiers(),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)
        occupancy_grid = ObsTerm(
            func=occupancy_grid_from_lidar,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "grid_size": MIXED_OCCUPANCY_GRID_SIZE,
                "grid_resolution": MIXED_OCCUPANCY_GRID_RESOLUTION,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        pose_2d_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_2d_command"})
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            modifiers=policy_base_lin_vel_modifiers(),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            modifiers=policy_imu_ang_vel_modifiers(),
        )
        actions = ObsTerm(func=mdp.last_action)
        occupancy_grid = ObsTerm(
            func=occupancy_grid_from_lidar,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "grid_size": MIXED_OCCUPANCY_GRID_SIZE,
                "grid_resolution": MIXED_OCCUPANCY_GRID_RESOLUTION,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
    held_scan_lidar_enabled: bool = True
    held_scan_lidar: HeldScanLidarCfg = HeldScanLidarCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle_scanner.update_period = 0.0
        self.scene.obstacle_scanner.pattern_cfg.horizontal_res = LIDAR_FOV_DEG / (NUM_LIDAR_RAYS - 1)
        self.scene.obstacle_scanner.debug_vis = False


@configclass
class MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg(MixedObstacleAvoidanceEnvCfg):
    """Mixed static/pedestrian co-training with temporal-lidar + next-frame prediction observations."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()
    held_scan_lidar_enabled: bool = True
    held_scan_lidar: HeldScanLidarCfg = HeldScanLidarCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle_scanner.update_period = 0.0
        self.scene.obstacle_scanner.pattern_cfg.horizontal_res = LIDAR_FOV_DEG / (NUM_LIDAR_RAYS - 1)
        self.scene.obstacle_scanner.debug_vis = False


@configclass
class MixedOccupancyObstacleAvoidanceEnvCfg(MixedObstacleAvoidanceEnvCfg):
    """Mixed static/pedestrian co-training with a 50x50 lidar occupancy grid."""

    observations: MixedOccupancyObservationsCfg = MixedOccupancyObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the experiment profile self-contained; callers can still override
        # this through the standard ``--num_envs`` configuration path.
        self.scene.num_envs = 2000


@configclass
class MixedOccupancyObstacleAvoidanceEnvCfg_PLAY(MixedOccupancyObstacleAvoidanceEnvCfg):
    """Playable mixed occupancy variant with clean high-level observations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True


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
        self.scene.num_envs = 16
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

EVALUATION_CROWD_SPEED_RANGE = (0.6, 1.0)
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

EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS = (0.25, 0.45)
"""Inclusive desired-speed sampling range [m/s] for the slow-leader benchmark."""

EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M = (1.5, 3.0)
"""Inclusive initial longitudinal-distance sampling range [m] from robot to leader."""

EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M = (-0.25, 0.25)
"""Initial lateral-offset sampling range [m] from the robot's lane to the slow leader."""

EVALUATION_SCENARIO_CODES = {
    "crossing": 0,
    "with_flow": 1,
    "against_flow": 2,
    "with_flow_slow_leader": 3,
}
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
    # Ordinary profiles keep the standard crowd reset.  The additional profile installs
    # a bounded-random slow leader in slot 0 after that same reset.
    env_cfg.events.reset_pedestrians = EventTerm(
        func=nav_mdp.reset_evaluation_pedestrian_crowd,
        mode="reset",
        params={
            "flow_dir": 1.0,
            "slow_leader_scenario_code": EVALUATION_SCENARIO_CODES["with_flow_slow_leader"],
            "slow_leader_speed_range_mps": EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
            "slow_leader_start_ahead_range_m": EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
            "slow_leader_lateral_offset_range_m": EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
        },
    )
    return env_cfg


def install_dynamic_crowd_evaluation_profiles(
    env,
    pedestrian_counts,
    scenario_codes,
) -> None:
    """Install fixed profile tensors consumed by dynamic-crowd evaluation reset hooks.

    Scenario codes are ``0=crossing``, ``1=with_flow``, ``2=against_flow``, and
    ``3=with_flow_slow_leader``. Inputs must contain one assignment per vector environment and
    are copied to the environment device.
    """
    import torch

    counts = torch.as_tensor(pedestrian_counts, device=env.device, dtype=torch.long)
    scenarios = torch.as_tensor(scenario_codes, device=env.device, dtype=torch.long)
    if counts.numel() != env.num_envs or scenarios.numel() != env.num_envs:
        raise ValueError("Evaluation profiles must provide exactly one count and scenario per environment.")
    if torch.any((counts < 1) | (counts > env.crowd_manager.max_pedestrians)):
        raise ValueError("Evaluation pedestrian counts must fit the configured crowd capacity.")
    max_scenario_code = max(EVALUATION_SCENARIO_CODES.values())
    if torch.any((scenarios < 0) | (scenarios > max_scenario_code)):
        raise ValueError(
            "Evaluation scenario codes must be 0 (crossing), 1 (with), 2 (against), "
            "or 3 (with slow leader)."
        )

    env.evaluation_pedestrian_count = counts
    env.evaluation_scenario = scenarios
    # Reset hooks fill these values for slow-leader episodes.  Keeping them on the
    # environment lets the evaluator record the actual sampled conditions per accepted
    # episode, rather than merely documenting the configured ranges.
    env.evaluation_slow_leader_speed_mps = torch.full(
        (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
    )
    env.evaluation_slow_leader_start_ahead_m = torch.full(
        (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
    )
    env.evaluation_slow_leader_lateral_offset_m = torch.full(
        (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
    )
    env.evaluation_flow_goal_direction = torch.where(
        (scenarios == EVALUATION_SCENARIO_CODES["with_flow"])
        | (scenarios == EVALUATION_SCENARIO_CODES["with_flow_slow_leader"]),
        torch.ones_like(scenarios),
        torch.where(scenarios == 2, -torch.ones_like(scenarios), torch.zeros_like(scenarios)),
    )
