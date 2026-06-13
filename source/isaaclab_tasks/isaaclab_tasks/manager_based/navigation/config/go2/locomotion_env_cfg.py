"""Flat-terrain velocity-tracking locomotion environment for Unitree Go2."""

from __future__ import annotations

import math
import torch

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
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains.config.rough import ROUGH_ONLY, ROUGH_AND_GRIDS, DISCRETE_OBSTACLES_ONLY

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import occupancy_grid_from_lidar
from isaaclab_tasks.manager_based.navigation.mdp.vis_utils import acquire_debug_draw, draw_occupancy_grid_points

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from .observation_modifiers import (
    policy_base_lin_vel_modifiers,
    policy_imu_ang_vel_modifiers,
    policy_imu_lin_acc_modifiers,
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Flat-terrain scene with Go2."""

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

    # Unitree 4D LiDAR L2: 360°×96° FOV, 16-channel approximation, 30m range.
    # Position from official Go2 URDF (unitreerobotics/unitree_ros): xyz=(0.28945, 0, -0.046825)
    # relative to base. The URDF also has rpy=(0, 2.8782, 0) (≈165° pitch) which is a body-frame
    # mounting convention for the L2 housing — it does NOT tilt the scan plane and is not applied here.
    l2_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.28945, 0.0, -0.046825)),
        attach_yaw_only=True,
        max_distance=30.0,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=1.0,
        ),
        mesh_prim_paths=["/World/ground"],
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

    @configclass
    class LidarObsCfg(ObsGroup):
        """Occupancy grid from the Unitree L2 lidar (32×32 @ 0.4 m/cell, 12.8 m span)."""

        occupancy_grid = ObsTerm(
            func=occupancy_grid_from_lidar,
            params={"sensor_cfg": SceneEntityCfg("l2_lidar"), "grid_size": 32, "grid_resolution": 0.4},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    ground_truth: GroundTruthCfg = GroundTruthCfg()
    lidar: LidarObsCfg = LidarObsCfg()


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
        if self.scene.l2_lidar is not None:
            self.scene.l2_lidar.update_period = self.decimation * self.sim.dt


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
class LocomotionVelEnvCfg_LIDAR_TEST(LocomotionVelEnvCfg_PLAY):
    """Test variant: replaces terrain with tall discrete obstacles so the L2 lidar
    occupancy grid can be visually verified in the Isaac Sim viewport."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = DISCRETE_OBSTACLES_ONLY
        self.scene.terrain.max_init_terrain_level = 0  # start on easiest level (most obstacles)
        self.scene.l2_lidar.debug_vis = True            # show raw ray hits in viewport


class LocomotionLidarVizEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that draws the L2 lidar occupancy grid in the viewport each step."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._occ_draw = acquire_debug_draw()
        self._occ_sensor_cfg = SceneEntityCfg("l2_lidar")

    def step(self, action: torch.Tensor):
        result = super().step(action)
        if self._occ_draw is not None:
            grid_flat = occupancy_grid_from_lidar(self, self._occ_sensor_cfg, grid_size=32, grid_resolution=0.4)
            sensor_pos = self.scene["l2_lidar"].data.pos_w
            grid_2d = grid_flat[0].reshape(32, 32).cpu().numpy()
            sx, sy = float(sensor_pos[0, 0].item()), float(sensor_pos[0, 1].item())
            self._occ_draw.clear_points()
            draw_occupancy_grid_points(self._occ_draw, grid_2d, (sx, sy), grid_resolution=0.4)
        return result


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

