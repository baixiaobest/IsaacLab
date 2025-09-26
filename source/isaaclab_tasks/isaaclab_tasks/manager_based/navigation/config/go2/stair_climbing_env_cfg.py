from dataclasses import MISSING

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.terrains.config.stairs import DIVERSE_STAIRS, TURN_90_STAIRS, TURN_180_STAIRS, PYRAMIDS_ONLY, PYRAMIDS_CLIMB_UP, PYRAMIDS_CLIMB_DOWN # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG, UNITREE_GO2_STIFF_CFG
from isaaclab.utils import configclass

EPISDOE_LENGTH = 10.0
GOAL_REACHED_ACTIVE_AFTER = 6.0
SIM_DT = 0.005
GOAL_REACHED_DISTANCE_THRESHOLD = 0.5
GOAL_REACHED_ANGULAR_THRESHOLD = 0.2
STRICT_GOAL_REACHED_DISTANCE_THRESHOLD = 0.15
STRICT_GOAL_REACHED_ANGULAR_THRESHOLD = 0.1
OBSTACLE_SCANNER_SPACING = 0.1
NUM_RAYS = 32
USE_TEST_ENV = False
REGULARIZATION_TERRAIN_LEVEL_THRESHOLD = 5
FOOT_SCANNER_RAIDUS = 0.10
FOOT_SCANNER_NUM_POINTS = 8
TERRAIN_LEVEL_NAMES = ['pyramid_stairs', 'pyramid_stairs_inv', 'linear_stairs_ground', 'linear_stairs_walled', 'turning_stairs_90_right','turning_stairs_90_left', 'turning_stairs_180_right', 'turning_stairs_180_left']
BASE_CONTACT_LIST = ["base", "Head_upper", "Head_lower", ".*hip", ".*thigh"]

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=DIVERSE_STAIRS,
            max_init_terrain_level=0,
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
    robot: ArticulationCfg = UNITREE_GO2_STIFF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(4.0, 4.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    fl_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.CirclePatternCfg(radius=FOOT_SCANNER_RAIDUS, num_points=FOOT_SCANNER_NUM_POINTS),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )

    fr_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.CirclePatternCfg(radius=FOOT_SCANNER_RAIDUS, num_points=FOOT_SCANNER_NUM_POINTS),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )

    rl_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.CirclePatternCfg(radius=FOOT_SCANNER_RAIDUS, num_points=FOOT_SCANNER_NUM_POINTS),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )

    rr_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.CirclePatternCfg(radius=FOOT_SCANNER_RAIDUS, num_points=FOOT_SCANNER_NUM_POINTS),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
            "static_friction_range": (0.8, 1.5),
            "dynamic_friction_range": (0.6, 1.3),
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
            "torque_range": (0.0, 0.0),
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

    push_robot = EventTerm(
        func=nav_mdp.activate_event_terrain_level_reached,
        interval_range_s=(5.0, 12.0),
        mode="interval",
        params={
            "func": mdp.push_by_setting_velocity,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
            "callback_params": {
                "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
            }
        }
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
    
    pyramids_stairs = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "pyramid_stairs"})
    pyramid_stairs_inv = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "pyramid_stairs_inv"})
    linear_stairs_ground = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "linear_stairs_ground"})
    linear_stairs_walled = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "linear_stairs_walled"})
    turning_stairs_90_right = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "turning_stairs_90_right"})
    turning_stairs_90_left = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "turning_stairs_90_left"})
    turning_stairs_180_right = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "turning_stairs_180_right"})
    turning_stairs_180_left = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "turning_stairs_180_left"})

@configclass
class CommandsCfg:
    pose_2d_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        stationary_prob=0.05,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi),
            pos_z=(0.2, 0.4)
        ),
        resampling_time_range=(EPISDOE_LENGTH+1.0, EPISDOE_LENGTH+1.0),
        debug_vis=True
    )

@configclass
class RewardsCfg:
    # Task reward
    goal_tracking_coarse = RewTerm(
        func=nav_mdp.active_after_time,
        weight=1.0,
        params={
            "func": nav_mdp.position_command_error_tanh,
            "active_after_time": 0.0,
            "callback_params": {
                "command_name":"pose_2d_command",
                "std": 5.0
            }
        })

    guidelines_reward = RewTerm(
        func=nav_mdp.guidelines_progress_reward,
        weight=0.0,
        params={
            "command_name": "pose_2d_command",
            "path_std": 8.0,
            "path_centering_std": 0.6,
            "centering_std": 0.4,
            "distance_scale": 0.95,
            "centering_scale": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
            'z_threshold': 0.5
        }
    )
    
    goal_tracking_fine = RewTerm(
        func=nav_mdp.active_after_time,
        weight=1.0,
        params={
            "func": nav_mdp.position_command_error_tanh,
            "active_after_time": GOAL_REACHED_ACTIVE_AFTER,
            "callback_params": {
                "command_name":"pose_2d_command",
                "std": 1.0,
            }
        })

    goal_heading_error = RewTerm(
        func=nav_mdp.active_after_time,
        weight=-0.3,
        params={
            "func": nav_mdp.heading_command_error_abs,
            "active_after_time": GOAL_REACHED_ACTIVE_AFTER,
            "callback_params": {
                "command_name":"pose_2d_command"
            }
        }
    )
    
    # Needed after adding countdown to the observation
    movement_reward = RewTerm(
        func=nav_mdp.inactivate_after_time,
        weight=0.2,
        params={
            'func': nav_mdp.movement_reward,
            'inactivate_after_time': 0.0,
            'callback_params': {
                'command_name': 'pose_2d_command',
                'std': 0.2
            }
        })
    
    speed_limit_penalty = RewTerm(
        func=nav_mdp.speed_limit_penalty,
        weight=-1.0,
        params={
            "speed_limit": 1.5,
            "std": 0.2
        }
    )

    backward_movement_penalty = RewTerm(
        func=nav_mdp.velocity_heading_error_abs,
        weight=-0.05,
        params={
            "velocity_threshold": 0.1,
            "heading_deadband": 0.26,  # 15 degrees
        }
    )
    # Undesired contacts for all terrain types
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_CONTACT_LIST), 
                "threshold": 0.6},
    )

    mild_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*calf"]), 
                "threshold": 0.6},
    )

    foot_wall_contacts = RewTerm(
        func=mdp.wall_contact_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "threshold": 0.8
        }
    )

    fall_penalty = RewTerm(
        func=mdp.fall_penalty,
        weight=-1.0,
        params={
            "velocity_threshold": 5.0,
        }
    )
    
    feet_air_time_range = RewTerm(
        func=mdp.feet_air_time_range,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "pose_2d_command",
            "zero_command_distance": 0.1,
            "range": (0.4, 1.0),
            "T": 0.5
        },
    )

    flying_penalty = RewTerm(
        func=mdp.flying_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
        },
    )

    # Energy minimization
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5) # Stationary power due to motor torque

class RegularizationRewardsCfg(RewardsCfg):
    ### The following regularization penalty can hinder exploration, 
    ### activate them after the robot can move reasonably well

    # Power transferred from motor to joints
    dof_power = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-1e-4,
        params={
            "func": mdp.joint_power,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD
        }
    )

    # Avoid jerky action
    action_rate_l2 = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.0001,
        params={
            "func": mdp.action_rate_l2,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD
        }
    )

    # Joint limit penalty
    joint_limit_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.1,
        params={
            "func": mdp.joint_pos_limits,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD
        }
    )

    # Penalize overly fast joint movement
    joint_vel_limit_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.1,
        params={
            "func": mdp.joint_vel_limits,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
            "callback_params": {
                'soft_ratio': 0.9,
            }
        }
    )

    joint_vel_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-5e-5,
        params={
            "func": mdp.joint_vel_l2,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
        }
    )

    joint_acc_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-2e-8,
        params={
            "func": mdp.joint_acc_l2,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
        }
    )
    
    #################################
    # Goal reached reward/penalty
    #################################
    goal_reached_action_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.01,
        params={
            "func": nav_mdp.pose_2d_goal_callback_reward,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
            "callback_params": {
                'func': mdp.action_rate_l2,
                'command_name': 'pose_2d_command',
                'distance_threshold': STRICT_GOAL_REACHED_DISTANCE_THRESHOLD,
                'angular_threshold': STRICT_GOAL_REACHED_ANGULAR_THRESHOLD,
            }
        }
    )

    goal_reached_flat_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.05,
        params={
            "func": mdp.flat_orientation_exp,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
            "callback_params": {
                "threshold_deg": 5.0
            }
        }
    )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_2d_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_2d_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # count_down = ObsTerm(
        #     func=mdp.count_down,
        #     params={"episode_length": EPISDOE_LENGTH}
        # )

        fl_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("fl_foot_scanner"), 
                'offset': 0.0},
            clip=(-1.0, 1.0),
        )
        fr_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("fr_foot_scanner"), 
                'offset': 0.0},
            clip=(-1.0, 1.0),
        )
        rl_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("rl_foot_scanner"), 
                'offset': 0.0},
            clip=(-1.0, 1.0),
        )
        rr_foot_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("rr_foot_scanner"), 
                'offset': 0.0},
            clip=(-1.0, 1.0),
        )
        
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), 'offset': 0.4},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_CONTACT_LIST), 
                "threshold": 0.6},
    )

    base_contact_discrete_obstacles = DoneTerm(
        func=nav_mdp.terrain_specific_callback,
        params={
            "terrain_names": ["discrete_obstacles"],
            "func": mdp.illegal_contact,
            "callback_params": {
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", 
                    body_names=["base", "Head_upper", ".*hip", "Head_lower", ".*thigh"]),
                "threshold": 0.2
            }
        }
    )

    base_vel_out_of_limit = DoneTerm(
        func=mdp.root_z_velocity_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  
            "max_z_velocity": 8.0
        }
    )

@configclass
class NavigationStairsEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RegularizationRewardsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.episode_length_s = EPISDOE_LENGTH
        # simulation settings
        self.sim.dt = SIM_DT
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

class NavigationPyramidStairsEnvCfg(NavigationStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = PYRAMIDS_CLIMB_UP
        self.rewards.guidelines_reward = None

class NavigationEnd2EndStairsOnlyEnvCfg(NavigationStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = TURN_90_STAIRS
        self.rewards.guidelines_reward.weight = 1.0
        self.rewards.goal_tracking_coarse.weight = 0.0
        self.rewards.undesired_contacts.weight = -20.0

@configclass
class NavigationEnd2EndStairsOnlyEnvCfg_PLAY(NavigationEnd2EndStairsOnlyEnvCfg):
    pass


class NavigationPyramidStairsEnvCfg_PLAY(NavigationPyramidStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 10
        