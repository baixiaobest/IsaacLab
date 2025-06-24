# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.terrains.config.rough import DIVERSE_TERRAINS_CFG, COST_MAP_TERRAINS_CFG, MOUNTAIN_TERRAINS_CFG


##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

@configclass
class RoughTeacherSceneCfg(MySceneCfg):
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2, 1.5]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*foot", history_length=5, track_air_time=True, debug_vis=True)

@configclass
class RoughTeacherObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RoughTeacherPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # Privileged information
        mass = ObsTerm(func=mdp.body_mass)
        com = ObsTerm(func=mdp.CachedBodyCenterOfMassTerm, 
                      params={"asset_cfg": SceneEntityCfg("robot", body_names="base")})
        
        foot_materials = ObsTerm(
            func=mdp.CachedBodyMaterialPropertiesTerm, 
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])})
        
        foot_contact_forces = ObsTerm(
            func=mdp.obs_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("foot_contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])})

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
    policy: RoughTeacherPolicyCfg = RoughTeacherPolicyCfg()

@configclass
class RoughTeacherScandotsOnlyObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RoughTeacherScandotsOnlyPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # Privileged information
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
    policy: RoughTeacherScandotsOnlyPolicyCfg = RoughTeacherScandotsOnlyPolicyCfg()

@configclass
class UnitreeGo2RoughTeacherCurriculum:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    pyramid_stairs_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "pyramid_stairs"})
    pyramid_stairs_inv_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "pyramid_stairs_inv"})
    
    cross_gap_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={"terrain_name": "mesh_gap"})
    climb_down_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "mesh_box"})
    climb_out_pit_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "mesh_pit"})
    climb_rail_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "mesh_rail"})

    pyramid_slope_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "hf_pyramid_slope"})
    pyramid_slope_inv_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "hf_pyramid_slope_inv"})

    repeat_objects = CurrTerm(func=mdp.GetMeanTerrainLevel, params={"terrain_name": "mesh_repeat_object"})
    
    random_rough_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "random_rough"})
    box_terrain_level = CurrTerm(func=mdp.GetMeanTerrainLevel, params={'terrain_name': "boxes"})
    


@configclass
class UnitreeGo2RoughTeacherEnvCfg(UnitreeGo2RoughEnvCfg):
    scene: RoughTeacherSceneCfg = RoughTeacherSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RoughTeacherObservationsCfg = RoughTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.base_com.params['com_range'] = {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (-0.01, 0.01)}
        self.events.physics_material.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 0.8),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
            "make_consistent": True
        }

        self.curriculum = UnitreeGo2RoughTeacherCurriculum()

@configclass
class UnitreeGo2RoughTeacherEnvCfg_v2(UnitreeGo2RoughTeacherEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        # Policy determines heading
        self.commands.base_velocity.velocity_heading = True
        self.commands.base_velocity.world_frame_command = True
        self.commands.base_velocity.resampling_time_range=(20.0, 50.0)
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.RangesAngleMag(
            lin_vel_mag = (0.3, 1.0), lin_vel_angle= (-math.pi, math.pi), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        )

        self.rewards.distance_traveled_reward.weight = 1.0

        # Add command velocity level to curriculum

        # self.curriculum.command_levels = CurrTerm(
        #     func=mdp.command_velocity_level, 
        #     params={
        #         'command_name': 'base_velocity',
        #         'terrain_level_to_velocity_range': {
        #             "0": mdp.UniformVelocityCommandCfg.RangesAngleMag(
        #                 lin_vel_mag=(0, 0.5), lin_vel_angle=(-math.pi, math.pi), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)),
        #             "3": mdp.UniformVelocityCommandCfg.RangesAngleMag(
        #                 lin_vel_mag=(0.0, 1.0), lin_vel_angle=(-math.pi, math.pi), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)),
        #             "4": mdp.UniformVelocityCommandCfg.RangesAngleMag(
        #                 lin_vel_mag=(0.0, 1.5), lin_vel_angle=(-math.pi, math.pi), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi))}})

@configclass
class UnitreeGo2RoughTeacherEnvCfg_v3(UnitreeGo2RoughTeacherEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        # Policy determines heading
        self.commands.base_velocity.velocity_heading = True
        self.commands.base_velocity.world_frame_command = True
        self.commands.base_velocity.resampling_time_range=(20.0, 50.0)
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.RangesAngleMag(
            lin_vel_mag = (0.3, 1.0), lin_vel_angle= (-math.pi, math.pi), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        )

        self.rewards.distance_traveled_reward.weight = 1.0
        # Due to high terrain which the robot needs to overcome,
        # it tends to stand higher before overcoming the terrain.
        # This causes the robot hip to turn more than usual to stand higher.
        # We need to penalize this behavior.
        self.rewards.joint_deviation.params = {
            'asset_cfg': SceneEntityCfg("robot", 
            joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"])}
        self.rewards.joint_deviation.weight = -0.2

        # Encourage larger step
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_range.weight = 0.4

        # Flater body during walking
        self.rewards.flat_orientation_range.weight = -0.1

        self.scene.terrain.terrain_generator = DIVERSE_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True

@configclass
class UnitreeGo2RoughTeacherScandotsOnlyEnvCfg(UnitreeGo2RoughTeacherEnvCfg):
    observations: RoughTeacherScandotsOnlyObservationsCfg = RoughTeacherScandotsOnlyObservationsCfg()

#### PLAY configs ####

@configclass
class UnitreeGo2RoughTeacherCfg_PLAY(UnitreeGo2RoughTeacherEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class UnitreeGo2RoughTeacherScandotsOnlyCfg_PLAY(UnitreeGo2RoughTeacherScandotsOnlyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class UnitreeGo2RoughTeacherCfg_PLAY_v2(UnitreeGo2RoughTeacherEnvCfg_v2):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 10

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class UnitreeGo2RoughTeacherCfg_PLAY_v3(UnitreeGo2RoughTeacherEnvCfg_v3):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 10

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
