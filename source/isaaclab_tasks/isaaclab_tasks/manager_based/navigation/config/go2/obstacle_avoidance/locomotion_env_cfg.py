"""Flat-terrain velocity-tracking locomotion environment for Unitree Go2."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains.config.rough import ROUGH_ONLY, ROUGH_AND_GRIDS

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from .observation_modifiers import (
    policy_base_lin_vel_modifiers,
    policy_imu_ang_vel_modifiers,
    policy_imu_lin_acc_modifiers,
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Flat-terrain scene with Go2.

    This is deliberately sensor-free: the regular locomotion policy and its
    estimator should not incur the LiDAR update cost or expose LiDAR data.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_AND_GRIDS,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0.0, 0.04232)),
        gravity_bias=(0.0, 0.0, 9.81),
        debug_vis=False,
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
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.15,
        rel_rotating_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy with inertial sensing."""

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
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
            modifiers=policy_imu_lin_acc_modifiers(),
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class GroundTruthCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")})
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")})

    policy: PolicyCfg = PolicyCfg()
    ground_truth: GroundTruthCfg = GroundTruthCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    joint_torque_offset_curriculum = EventTerm(
        func=mdp.apply_external_joint_torque_curriculum,
        mode="reset",
        params={
            "base_torque_range": (-0.0, 0.0),
            "max_torque_range": (-5.0, 5.0),
            "start_terrain_level": 5,
            "max_terrain_level": 10,
            "joint_names": [".*"],
        },
    )

    # Enabled only by ``LocomotionVelEnvCfg_ROBUST`` below.  The default,
    # play, and rollout tasks intentionally retain their current disturbance
    # distribution.
    base_wrench_impulse: EventTerm | None = None


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    zero_command_lin_vel_xy_l2 = RewTerm(
        func=mdp.zero_command_lin_vel_xy_l2,
        weight=-1.0,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    zero_command_ang_vel_xy_l2 = RewTerm(
        func=mdp.zero_command_ang_vel_xy_l2,
        weight=-0.1,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # Enabled only by ``LocomotionVelEnvCfg_ROBUST``.  The default, play, and
    # rollout tasks retain their existing reward definitions.
    stationary_base_height_l2: RewTerm | None = None
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0,
        },
    )
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    base_vel_out_of_limit = DoneTerm(
        func=mdp.root_velocity_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "max_velocity": 5.0,
        },
    )

@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_resampling_time = CurrTerm(
        func=mdp.command_resampling_time_level,
        params={
            "command_name": "base_velocity",
            "start_time_range": (10.0, 10.0),
            "end_time_range": (3.0, 3.0),
            "start_level": 0,
            "end_level": 5,
        },
    )


@configclass
class RobustCurriculumCfg(CurriculumCfg):
    """Robust-v1 progresses terrain from delivered-command tracking quality."""

    terrain_levels = CurrTerm(
        func=mdp.robust_velocity_tracking_terrain_curriculum,
        params={
            "planar_rms_threshold_mps": 0.25,
            "yaw_rms_threshold_radps": 0.35,
            "stop_planar_rms_threshold_mps": 0.15,
            "stop_yaw_rms_threshold_radps": 0.20,
        },
    )

@configclass
class LocomotionVelEnvCfg(ManagerBasedRLEnvCfg):
    """Flat-terrain locomotion env for Go2 (velocity command tracking)."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.imu is not None:
            self.scene.imu.update_period = self.sim.dt


@configclass
class LocomotionVelEnvCfg_PLAY(LocomotionVelEnvCfg):
    """Play variant: fewer envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.base_lin_vel.modifiers = None
        self.observations.policy.imu_ang_vel.modifiers = None
        self.observations.policy.imu_lin_acc.modifiers = None
        self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)


@configclass
class LocomotionVelEnvCfg_ROBUST(LocomotionVelEnvCfg):
    """Robust direct-twist training variant for the navigation locomotion policy."""

    def __post_init__(self):
        super().__post_init__()
        # All environments begin on the easiest terrain row.  The velocity
        # command profile is deliberately identical at every terrain level.
        self.scene.terrain.max_init_terrain_level = 0
        # Torque-offset randomization is deferred for this first robust-policy
        # revision.  Reintroduce it only as an explicit later curriculum.
        self.events.joint_torque_offset_curriculum = None
        self.rewards.stationary_base_height_l2 = RewTerm(
            func=mdp.stationary_base_height_l2,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "target_height": 0.37,
                "planar_deadzone_mps": 0.25,
                "planar_fade_end_mps": 0.40,
                "yaw_deadzone_radps": 0.10,
            },
        )
        self.rewards.feet_air_time = self.rewards.feet_air_time.replace(
            params={**self.rewards.feet_air_time.params, "command_speed_threshold": 0.25}
        )
        self.rewards.lower_head_contact = RewTerm(
            func=mdp.undesired_contacts,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_lower"),
                "threshold": 1.0,
            },
        )
        # The term executes each 20 ms RL control step and holds a selected
        # base-frame force across all four 5 ms physics substeps.  It uses a
        # per-environment terrain-level gate rather than a global curriculum
        # activation, so only robots that reached level 5 receive kick pulses.
        self.events.base_wrench_impulse = EventTerm(
            func=mdp.BaseWrenchImpulse,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_name": "base_velocity",
                "terrain_level_threshold": 5,
                "settle_time_s": 1.5,
                "command_speed_threshold": 0.2,
                "pulse_interval_range_s": (3.0, 6.0),
                "pulse_duration_range_s": (0.04, 0.08),
                # Terrain bands: levels 5-6, 7, 8, and 9+ respectively.
                "impulse_ranges_by_level": ((5.0, 10.0), (10.0, 16.0), (16.0, 22.0), (22.0, 25.0)),
                "application_point_range_m": (
                    (-0.15, 0.15),
                    (-0.15, 0.15),
                    (-0.10, 0.10),
                ),
            },
        )
        self.curriculum = RobustCurriculumCfg()
        self.commands.base_velocity = mdp.RobustVelocityCommandCfg(
            asset_name="robot",
            normal_yaw_full_cap_speed_mps=1.0,
            normal_yaw_cap_at_max_planar_speed_radps=1.0,
            slow_coupled_turn_probability=0.05,
            slow_straight_probability=0.05,
            rotate_in_place_probability=0.10,
            full_stop_probability=0.10,
            normal_coupled_motion_probability=0.40,
            sudden_change_probability=0.30,
            # Above terrain level 5, sudden targets repeat every 5 s -> 3 s
            # by level 9, while the 10% full-stop mode becomes a 4 s high-
            # speed cruise followed by a measured stop-and-repeat cycle.
            sudden_change_start_terrain_level=5,
            sudden_change_start_interval_s=5.0,
            sudden_change_full_interval_s=3.0,
            stop_cycle_start_terrain_level=5,
            stop_cycle_cruise_duration_s=4.0,
            stop_cycle_planar_speed_range_mps=(0.75, 1.5),
            stop_cycle_settle_planar_speed_mps=0.10,
            stop_cycle_settle_yaw_rate_radps=0.10,
            stop_cycle_settle_duration_s=0.5,
            stop_cycle_max_dwell_s=3.0,
            # Required by CommandTermCfg only; RobustVelocityCommand owns its
            # per-environment schedule and does not use this generic range.
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
        )
        # The generic curriculum mutates UniformVelocityCommandCfg timing globally;
        # robust-v1 instead derives timing from each environment's terrain level.
        self.curriculum.command_resampling_time = None

@configclass
class LocomotionVelEnvCfg_ROLLOUT(LocomotionVelEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        ROLLOUT_LENGTH = 10.0
        self.commands.base_velocity.resampling_time_range = (ROLLOUT_LENGTH/4.0, ROLLOUT_LENGTH)

        self.episode_length_s = ROLLOUT_LENGTH
        self.observations.policy.enable_corruption = True
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(ROLLOUT_LENGTH, ROLLOUT_LENGTH),
            rel_standing_envs=0.10,
            rel_rotating_standing_envs=0.10,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-2.0, 2.0),
                heading=(-math.pi, math.pi),
            ),
        )
