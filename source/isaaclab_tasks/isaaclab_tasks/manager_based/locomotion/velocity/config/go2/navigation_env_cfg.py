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
from .rough_teacher_env_cfg import UnitreeGo2RoughTeacherEnvCfg_v3
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_teacher_env_cfg import UnitreeGo2RoughTeacherEnvCfg_v3

LOW_LEVEL_ENV_CFG = UnitreeGo2RoughTeacherEnvCfg_v3()

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import MOUNTAIN_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

MOUNTAIN_TERRAINS_TRAIN_CFG = MOUNTAIN_TERRAINS_CFG.replace(
    goal_num_rows=5,
    goal_num_cols=5,
    goal_grid_area_size= (100.0, 100.0),
    total_terrain_levels=5,
    distance_increment_per_level=15.0,
    origins_per_level=8)

MOUNTAIN_TERRAINS_TRAIN_CFG.terrain_config = \
    MOUNTAIN_TERRAINS_TRAIN_CFG.terrain_config.replace(
        size=(260.0, 260.0)
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.single_terrain_level)

@configclass
class CommandsCfg:
    navigation_command = mdp.NavigationPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(20, 30),
        debug_vis=False,
        # command=mdp.NavigationPositionCommandCfg.VelocityCommand(
        #     max_velocity=1.0,
        #     P_heading=0.1
        # )
    )

@configclass
class RewardsCfg:
    position_error_long_distance = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={
            "command_name": "navigation_command",
            "std": 50.0
            }
    )
    position_error_mid_distance = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={
            "command_name": "navigation_command",
            "std": 10.0
            }
    )
    position_error_short_distance = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={
            "command_name": "navigation_command",
            "std": 0.5
            }
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"logs/rsl_rl/EncoderActorCriticGO2/Teacher-v3/2025-06-15_15-38-10/model_jit.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        enable_velocity_heading=True,
        velocity_heading_gain=0.1,
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
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
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
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

    base_vel_out_of_limit = DoneTerm(
        func=mdp.root_velocity_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  
            "max_velocity": 5.0      
        }
    )

    # goal_reached = DoneTerm(
    #     func=mdp.goal_reached,
    #     params={
    #         "command_name": "navigation_command",
    #         "threshold": 0.5,
    #     }
    # )

##
# Environment configuration
##

@configclass
class NavigationMountainEnvCfg(UnitreeGo2RoughTeacherEnvCfg_v3):
    """Configuration for the locomotion velocity-tracking environment."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.sim.physx.gpu_max_rigid_patch_count = 3_000_000
        self.sim.physx.gpu_collision_stack_size = 300_000_000

        self.curriculum = CurriculumCfg()
        self.commands = CommandsCfg()

        self.rewards = RewardsCfg()

        self.actions = ActionsCfg()

        self.observations = NavigationObservationsCfg()

        self.terminations = TerminationsCfg()

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="single_terrain_generator",
            single_terrain_generator=MOUNTAIN_TERRAINS_TRAIN_CFG,
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

        self.episode_length_s = 75.0

@configclass
class NavigationMountainEnvCfg_PLAY(NavigationMountainEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.single_terrain_generator = MOUNTAIN_TERRAINS_CFG

    