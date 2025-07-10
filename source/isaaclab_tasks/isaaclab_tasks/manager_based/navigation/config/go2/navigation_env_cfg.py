# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from ....locomotion.velocity.config.go2.rough_teacher_env_cfg import UnitreeGo2RoughTeacherEnvCfg_v2
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_teacher_env_cfg import UnitreeGo2RoughTeacherEnvCfg_v3
from isaaclab.sim.simulation_cfg import SimulationCfg

LOW_LEVEL_ENV_CFG = UnitreeGo2RoughTeacherEnvCfg_v2()

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import MOUNTAIN_TERRAINS_CFG, FLAT_TERRAINS_CFG, FLAT_TERRAINS_OBSTACLES_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

DISTANCE_THRESHOLD = 0.8
VELOCITY_THRESHOLD = 0.1
ACTION_THRESHOLD = 0.1
REWARD_MULTIPLIER = 1.4
UNDESIRED_CONTACTS_NAMES = ["base", ".*hip", "Head.*",".*thigh"]

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="single_terrain_generator",
            single_terrain_generator=MOUNTAIN_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    navigation_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(2.0, 0.0, 20.0)),
        drift_range=(0.05, 0.15),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[3, 3]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*foot", history_length=5, track_air_time=True, debug_vis=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 0.8),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
            "make_consistent": True
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

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
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
            }
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
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.single_terrain_level, 
                              params={
                                  'distance_threshold': DISTANCE_THRESHOLD,
                                  'velocity_threshold': VELOCITY_THRESHOLD})

@configclass
class CommandsCfg:
    navigation_command = mdp.NavigationPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(20, 30),
        debug_vis=False,
        command=mdp.NavigationPositionCommandCfg.PositionCommand(
                heading_type="target_heading", # If you are changing this, make sure to change the reward function accordingly
                command_scales=(0.1, 0.1, 0.1, 1.0) # Scale to stabilize the training
            )
    )

@configclass
class RewardsType1Cfg:
    progress_reward_mid_distance = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "command_name": "navigation_command",
            "std": 10.0
            }
    )
    progress_reward_short_distance = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "command_name": "navigation_command",
            "std": 2.0
            }
    )

    goal_reached_reward = RewTerm(
        func=nav_mdp.goal_reached_reward,
        weight=5.0,
        params={
            'distance_threshold': DISTANCE_THRESHOLD,
            'velocity_threshold': VELOCITY_THRESHOLD,
            'action_threshold': ACTION_THRESHOLD,
            'reward_multiplier': REWARD_MULTIPLIER
        })

    # velocity_heading_error = RewTerm(
    #     func=nav_mdp.velocity_heading_error_abs,
    #     params={"velocity_threshold": 0.2},
    #     weight=-0.05
    # )

    heading_command_error = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        params={"command_name": "navigation_command"},
        weight=-0.1
    )

    # action_penalty = RewTerm(func=mdp.action_l2, weight=-0.05)

    # # Extra penalty for angular velocity
    # ang_vel_penalty = RewTerm(
    #     func=mdp.ang_vel_z_l2,
    #     weight=-0.2
    # )

    action_rate_l2 = RewTerm(func=nav_mdp.action_rate_l2,  
                             weight=-0.05)
    

@configclass
class RewardsType2Cfg:
    progress_reward_long_distance = RewTerm(
        func=nav_mdp.navigation_progress,
        weight=1.0,
        params={
            "command_term_name": "navigation_command",
            "scale": 200.0 # Scale to compensate for small simulation time step
            }
    )

    goal_reached_reward = RewTerm(
        func=nav_mdp.goal_reached_reward,
        weight=5.0,
        params={
            'distance_threshold': DISTANCE_THRESHOLD,
            'velocity_threshold': VELOCITY_THRESHOLD,
            'action_threshold': ACTION_THRESHOLD,
            'reward_multiplier': REWARD_MULTIPLIER
        })
    
    lateral_movement_penalty = RewTerm(
        func=nav_mdp.lateral_movement_penalty,
        params={
            "command_term_name": "navigation_command",
            "std": 1.0
        },
        weight=-0.2
    )

    heading_command_error = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        params={"command_name": "navigation_command"},
        weight=-0.2
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACTS_NAMES),
            "threshold": 0.1
        },
        weight=-1.0/0.005 # It should be scaled by 1.0/step_dt, because the episode terminates after this reward is given.
    )

    action_rate_l2 = RewTerm(func=nav_mdp.navigation_command_w_rate_penalty_l2,  
                             weight=-0.05)
    
@configclass
class RewardsCNNCfg:
    progress_reward_long_distance = RewTerm(
        func=nav_mdp.navigation_progress,
        weight=1.0,
        params={
            "command_term_name": "navigation_command",
            "scale": 200.0 # Scale to compensate for small simulation time step
            }
    )

    goal_reached_reward = RewTerm(
        func=nav_mdp.goal_reached_reward,
        weight=5.0,
        params={
            'distance_threshold': DISTANCE_THRESHOLD,
            'velocity_threshold': VELOCITY_THRESHOLD,
            'action_threshold': ACTION_THRESHOLD,
            'reward_multiplier': REWARD_MULTIPLIER
        })
    
    lateral_movement_penalty = RewTerm(
        func=nav_mdp.lateral_movement_penalty,
        params={
            "command_term_name": "navigation_command",
            "std": 1.0
        },
        weight=-0.1
    )

    heading_command_error = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        params={"command_name": "navigation_command"},
        weight=-0.2
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACTS_NAMES),
            "threshold": 0.1
        },
        weight=-1.0/0.005 # It should be scaled by 1.0/step_dt, because the episode terminates after this reward is given.
    )

    action_rate_l2 = RewTerm(func=nav_mdp.navigation_command_w_rate_penalty_l2,  
                             weight=-0.05)

@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"logs/rsl_rl/EncoderActorCriticGO2/Teacher-v2/2025-07-03_06-06-34/model_jit.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        action_scales=(1.0, 1.0, 1.0),
        debug_vis=True
    )

@configclass
class NavigationObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class NavigationPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        heading_sin_cos = ObsTerm(
            func=mdp.root_yaw_sin_cos, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)

        navigation_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "navigation_command"}
        )

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("navigation_height_scanner"),
                "offset": 0.4},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: NavigationPolicyCfg = NavigationPolicyCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACTS_NAMES), "threshold": 0.05},
    )

    base_vel_out_of_limit = DoneTerm(
        func=mdp.root_velocity_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  
            "max_velocity": 5.0      
        }
    )

    goal_reached = DoneTerm(
        func=nav_mdp.navigation_goal_reached,
        params={
            "distance_threshold": DISTANCE_THRESHOLD,
            "velocity_threshold": VELOCITY_THRESHOLD,
            "action_threshold": ACTION_THRESHOLD,
        }
    )

##
# Environment configuration
##

@configclass
class NavigationMountainEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsType2Cfg = RewardsType2Cfg()
    actions: ActionsCfg = ActionsCfg()
    observations: NavigationObservationsCfg = NavigationObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if self.scene.num_envs > 500:
            self.sim.physx.gpu_collision_stack_size = 300_000_000
            self.sim.physx.gpu_max_rigid_patch_count = 1_000_000
        else:
            self.sim.physx.gpu_collision_stack_size = 600_000
            self.sim.physx.gpu_max_rigid_patch_count = 1_000_000

        self.scene.terrain.single_terrain_generator = FLAT_TERRAINS_CFG

@configclass
class NavigationMountainEnvCfg_PLAY(NavigationMountainEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.sim.physx.gpu_max_rigid_patch_count = 1_000_000
        self.sim.physx.gpu_collision_stack_size = 600_000
        self.scene.terrain.single_terrain_generator = FLAT_TERRAINS_CFG

@configclass
class NavigationMountainNoScandotsCfg(NavigationMountainEnvCfg):
    """Configuration for the locomotion velocity-tracking environment without scan dots."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Remove the height scan observation
        self.observations.policy.height_scan = None
        self.scene.navigation_height_scanner = None
        self.scene.terrain.single_terrain_generator = FLAT_TERRAINS_CFG

@configclass
class NavigationCNNCfg(NavigationMountainEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with CNN observations."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.rewards = RewardsCNNCfg()
        self.scene.terrain.single_terrain_generator = FLAT_TERRAINS_OBSTACLES_CFG


@configclass
class NavigationMountainNoScandotsCfg_PLAY(NavigationMountainNoScandotsCfg):
    """Configuration for the locomotion velocity-tracking environment without scan dots in play mode."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.sim.physx.gpu_max_rigid_patch_count = 1_000_000
        self.sim.physx.gpu_collision_stack_size = 600_000
        self.scene.terrain.single_terrain_generator = FLAT_TERRAINS_CFG
        self.scene.terrain.single_terrain_generator.goal_num_cols = 1
        self.scene.terrain.single_terrain_generator.goal_num_rows = 1

        self.terminations.goal_reached = DoneTerm(
            func=nav_mdp.navigation_goal_reached_timer,
            params={
                "distance_threshold": 0.8,
                "velocity_threshold": 0.1,
                "stay_for_seconds": 0.5
            }
        )

@configclass
class NavigationCNNCfg_PLAY(NavigationCNNCfg):
    """Configuration for the locomotion velocity-tracking environment with CNN observations in play mode."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.sim.physx.gpu_max_rigid_patch_count = 1_000_000
        self.sim.physx.gpu_collision_stack_size = 600_000
        self.scene.terrain.single_terrain_generator.goal_num_cols = 1
        self.scene.terrain.single_terrain_generator.goal_num_rows = 1
        self.scene.terrain.single_terrain_generator.origins_per_level = 16

        self.scene.terrain.max_init_terrain_level = self.scene.terrain.single_terrain_generator.total_terrain_levels

        self.scene.navigation_height_scanner.debug_vis = False

        self.terminations.goal_reached = DoneTerm(
            func=nav_mdp.navigation_goal_reached_timer,
            params={
                "distance_threshold": 0.8,
                "velocity_threshold": 0.1,
                "stay_for_seconds": 0.5
            }
        )