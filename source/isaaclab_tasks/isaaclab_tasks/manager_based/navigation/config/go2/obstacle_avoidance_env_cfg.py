"""Obstacle-avoidance navigation environment for Unitree Go2.

The high-level navigation policy observes a front-facing lidar fan and a
relative goal pose, then emits velocity commands for a pre-trained low-level
locomotion controller.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import DISCRETE_OBSTACLES_ONLY
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .locomotion_env_cfg import LocomotionVelEnvCfg, MySceneCfg as LowLevelSceneCfg
from .observation_modifiers import policy_base_lin_vel_modifiers, policy_imu_ang_vel_modifiers

LOW_LEVEL_ENV_CFG = LocomotionVelEnvCfg()
LOW_LEVEL_POLICY_PATH = "logs/rsl_rl/ObstacleAvoidance/Locomotion/locomotion_policy_jit.pt"

NUM_LIDAR_RAYS = 128
LIDAR_FOV_DEG = 180.0
LIDAR_MAX_DISTANCE = 20.0
COMMAND_RESAMPLING_TIME_S = 12.0
EPISODE_LENGTH_S = 12.0
HIGH_LEVEL_DECIMATION_FACTOR = 4 # Run the navigation policy at 12.5hz, which is 1/4 of low-level policy.
GOAL_CONTACT_BODY_NAMES = ["base", "Head_upper", "Head_lower", ".*hip", ".*thigh"]
GOAL_REACHED_DISTANCE_THRESHOLD = 0.5
GOAL_REACHED_ANGULAR_THRESHOLD = 0.2

@configclass
class ObstacleAvoidanceSceneCfg(LowLevelSceneCfg):
    """Flat obstacle scene that preserves the low-level locomotion sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=DISCRETE_OBSTACLES_ONLY,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    obstacle_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-LIDAR_FOV_DEG / 2.0, LIDAR_FOV_DEG / 2.0),
            horizontal_res=LIDAR_FOV_DEG / NUM_LIDAR_RAYS,
        ),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """High-level navigation goals."""

    pose_2d_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        stationary_prob=0.1,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi),
            pos_z=(0.3, 0.4)
        ),
        resampling_time_range=(COMMAND_RESAMPLING_TIME_S, COMMAND_RESAMPLING_TIME_S),
        debug_vis=True
    )


@configclass
class ActionsCfg:
    """High-level velocity commands routed through a low-level locomotion policy."""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        action_scales=(1.0, 1.0, 1.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the high-level obstacle-avoidance policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Task-facing policy observations."""

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

        obstacle_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "max": LIDAR_MAX_DISTANCE,
                "scale_distance": True,
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset and startup events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for obstacle-aware goal reaching."""

    pose_2d_command_progress_reward = RewTerm(
        func=nav_mdp.pose_2d_command_progress_reward,
        weight=1.0,
        params={"command_name": "pose_2d_command"},
    )
    position_tracking_fine = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.5, "command_name": "pose_2d_command"},
    )
    orientation_tracking = RewTerm(
        func=nav_mdp.heading_command_error_within_range_abs,
        weight=-3.0,
        params={
            "command_name": "pose_2d_command", 
            "range": 1.0,
            },
    )
    obstacle_clearance_penalty = RewTerm(
        func=nav_mdp.obstacle_clearance_penalty,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
            "SOI": 0.6,
            "sensor_radius": 0.2,
        },
    )

    backward_movement_penalty = RewTerm(
        func=nav_mdp.velocity_heading_error_outside_goal_abs,
        weight=-0.5,
        params={
            "velocity_threshold": 0.1,
            "heading_deadband": 0.26,  # 15 degrees
            "command_name": "pose_2d_command",
            "goal_range": 1.0,
        }
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GOAL_CONTACT_BODY_NAMES),
            "threshold": 0.5,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)

    excessive_velocity = RewTerm(
        func=mdp.excessive_velocity, 
        weight=-0.1,
        params={
            "speed_threshold": 1.0,
        })

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=nav_mdp.pose_2d_command_terrain_curriculum,
                              params={
                                  "command_name": "pose_2d_command",
                                  "distance_threshold": GOAL_REACHED_DISTANCE_THRESHOLD,
                                  "angular_threshold": GOAL_REACHED_ANGULAR_THRESHOLD
                              })

@configclass
class TerminationsCfg:
    """Termination terms for the obstacle-avoidance task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GOAL_CONTACT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    base_vel_out_of_limit = DoneTerm(
        func=mdp.root_velocity_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "max_velocity": 5.0,
        },
    )


@configclass
class ObstacleAvoidanceEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 navigation env that emits velocity commands into a low-level policy."""

    scene: ObstacleAvoidanceSceneCfg = ObstacleAvoidanceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * HIGH_LEVEL_DECIMATION_FACTOR
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.imu is not None:
            self.scene.imu.update_period = self.sim.dt
        if self.scene.obstacle_scanner is not None:
            self.scene.obstacle_scanner.update_period = self.decimation * self.sim.dt


@configclass
class ObstacleAvoidanceEnvCfg_PLAY(ObstacleAvoidanceEnvCfg):
    """Play variant with fewer environments and clean high-level observations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 10
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True