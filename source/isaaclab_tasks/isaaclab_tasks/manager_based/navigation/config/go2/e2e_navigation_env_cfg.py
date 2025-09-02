from dataclasses import MISSING

import math
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
from isaaclab.actuators import DCMotorCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, DIVERSE_TERRAINS_CFG, NAVIGATION_TERRAINS_CFG, \
    DISCRETE_OBSTACLES_ROUGH_ONLY, ROUGH_ONLY # isort: skip
from isaaclab.terrains.config.test_terrain import TEST_TERRAIN_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

EPISDOE_LENGTH = 10.0
GOAL_REACHED_ACTIVE_AFTER = 6.0
SIM_DT = 0.005
GOAL_REACHED_DISTANCE_THRESHOLD = 0.5
GOAL_REACHED_ANGULAR_THRESHOLD = 1.0
STRICT_GOAL_REACHED_DISTANCE_THRESHOLD = 0.15
STRICT_GOAL_REACHED_ANGULAR_THRESHOLD = 0.1
OBSTACLE_SCANNER_SPACING = 0.1
NUM_RAYS = 32
USE_TEST_ENV = False
REGULARIZATION_TERRAIN_LEVEL_THRESHOLD = 9
TERRAIN_LEVEL_NAMES = ["random_rough"]
BASE_CONTACT_LIST = ["base", "Head_upper", "Head_lower", ".*hip", ".*thigh", ".*calf"]

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_ONLY,
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
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(4.0, 4.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    obstacle_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360),
            horizontal_res=360/NUM_RAYS-1e-3),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )
    obstacle_scanner_dx = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(OBSTACLE_SCANNER_SPACING, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360),
            horizontal_res=360/NUM_RAYS-1e-3),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"]
    )
    obstacle_scanner_dy = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, OBSTACLE_SCANNER_SPACING, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360),
            horizontal_res=360/NUM_RAYS-1e-3),
        debug_vis=False,
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
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
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

    # joint_torque_offset_curriculum = EventTerm(
    #     func=mdp.apply_external_joint_torque_curriculum,
    #     mode="reset",
    #     params={
    #         "base_torque_range": (-0.0, 0.0),
    #         "max_torque_range": (-5.0, 5.0),
    #         "start_terrain_level": int(NAVIGATION_TERRAINS_CFG.num_rows/2),
    #         "max_terrain_level": NAVIGATION_TERRAINS_CFG.num_rows,
    #         "joint_names": [".*"],
    #     })
    
    # randomize_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         'asset_cfg': SceneEntityCfg("robot"),
    #         'stiffness_distribution_params': (20.0, 30.0),
    #         'damping_distribution_params': (0.5, 3.0),
    #         'operation': 'abs'
    #     }
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=nav_mdp.pose_2d_command_terrain_curriculum,
                              params={
                                  "command_name": "pose_2d_command",
                                  "distance_threshold": GOAL_REACHED_DISTANCE_THRESHOLD,
                                  "angular_threshold": GOAL_REACHED_ANGULAR_THRESHOLD
                              })
    pyramid_stairs_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "pyramid_stairs"})
    pyramid_stairs_inv_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "pyramid_stairs_inv"})
    
    cross_gap_level = CurrTerm(func=mdp.GetTerrainLevel, params={"terrain_name": "mesh_gap"})
    climb_down_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "mesh_box"})
    climb_out_pit_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "mesh_pit"})
    climb_rail_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "mesh_rail"})

    pyramid_slope_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "hf_pyramid_slope"})
    pyramid_slope_inv_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "hf_pyramid_slope_inv"})

    repeat_objects = CurrTerm(func=mdp.GetTerrainLevel, params={"terrain_name": "mesh_repeat_object"})
    
    random_rough_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "random_rough"})
    box_terrain_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "boxes"})

    discrete_obstacles_level = CurrTerm(func=mdp.GetTerrainLevel, params={'terrain_name': "discrete_obstacles"})

@configclass
class CommandsCfg:
    # pose_2d_command = mdp.TerrainBasedPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=False,
    #     ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
    #         heading=(-math.pi, math.pi)
    #     ),
    #     resampling_time_range=(1.5*EPISDOE_LENGTH, 1.5*EPISDOE_LENGTH),
    #     debug_vis=True
    # )
    pose_2d_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0),
            pos_y=(-5.0, 5.0),
            heading=(-math.pi, math.pi)
        ),
        resampling_time_range=(1.5*EPISDOE_LENGTH, 1.5*EPISDOE_LENGTH),
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
            "active_after_time": GOAL_REACHED_ACTIVE_AFTER,
            "callback_params": {
                "command_name":"pose_2d_command",
                "std": 5.0
            }
        })
    
    goal_tracking_fine = RewTerm(
        func=nav_mdp.active_after_time,
        weight=1.0,
        params={
            "func": nav_mdp.position_command_error_tanh,
            "active_after_time": GOAL_REACHED_ACTIVE_AFTER,
            "callback_params": {
                "command_name":"pose_2d_command",
                "std": 1.0
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
            'inactivate_after_time': GOAL_REACHED_ACTIVE_AFTER,
            'callback_params': {
                'command_name': 'pose_2d_command',
            }
        })
    
    speed_limit_penalty = RewTerm(
        func=nav_mdp.speed_limit_penalty,
        weight=-1.0,
        params={
            "speed_limit": 1.7,
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
    # Additional undesired contacts for discrete obstacle terrain types
    undesired_contacts_discrete_obstacles = RewTerm(
        func=nav_mdp.terrain_specific_callback,
        weight=-8.0,
        params={
            "terrain_names": ["discrete_obstacles"],
            "func": mdp.undesired_contacts,
            "callback_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_upper", "Head_lower", ".*hip", ".*thigh"]),
                "threshold": 0.2
            }
        })
    
    obstacle_gradient_penalty = RewTerm(
        func=nav_mdp.obstacle_gradient_penalty,
        weight=-0.5,
        params={
            'sensor_center_cfg': SceneEntityCfg("obstacle_scanner"),
            'sensor_dx_cfg': SceneEntityCfg("obstacle_scanner_dx"),
            'sensor_dy_cfg': SceneEntityCfg("obstacle_scanner_dy"),
            'sensor_spacing': OBSTACLE_SCANNER_SPACING,
            'robot_radius': 0.3,
            'SOI': 1.2 # Sphere of influence
        })
    
    feet_air_time_range = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "pose_2d_command",
            "threshold": 0.3
        },
    )
    
    #################################
    # Regularization terms
    #################################
    # Energy minimization
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5) # Stationary power due to motor torque

class RegularizationRewardsCfg(RewardsCfg):
    ### The following regularization penalty can hinder exploration, 
    ### activate them after the robot can move reasonably well

    # Power transferred from motor to joints
    dof_power = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-2e-5,
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

    # reduce x y angular velocity
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)

    # Reduce vertical movement
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)

    # Reduce pitch roll
    # pitch_roll_penalty = RewTerm(
    #     func=mdp.flat_orientation_exp,
    #     weight=-0.05,
    #     params=
    #     {
    #         "threshold_deg": 10.0
    #     }
    # )

    # Hip joint deviation penalty
    # hip_joint_deviation_penalty = RewTerm(
    #     func=mdp.joint_deviation_l2,
    #     weight=-0.1,
    #     params={
    #         'asset_cfg': SceneEntityCfg("robot", joint_names=[".*hip.*"])
    #     }
    # )
    
    #################################
    # Goal reached reward/penalty
    #################################
    goal_reached_action_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.05,
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

    # Better pose at goal
    goal_joint_deviation_penalty = RewTerm(
        func=nav_mdp.activate_reward_terrain_level_reached,
        weight=-0.1,
        params={
            "func": nav_mdp.pose_2d_goal_callback_reward,
            "terrain_names": TERRAIN_LEVEL_NAMES,
            "operator": "max",
            "terrain_level_threshold": REGULARIZATION_TERRAIN_LEVEL_THRESHOLD,
            "callback_params": {
                'func': mdp.joint_deviation_l2,
                'command_name': 'pose_2d_command',
                'distance_threshold': STRICT_GOAL_REACHED_DISTANCE_THRESHOLD,
                'angular_threshold': STRICT_GOAL_REACHED_ANGULAR_THRESHOLD,
            }
        }
    )
    
    # goal_reached_joint_movement_penalty = RewTerm(
    #     func=nav_mdp.pose_2d_goal_callback_reward,
    #     weight=-0.2,
    #     params={
    #         'func': mdp.joint_vel_l2,
    #         'command_name': 'pose_2d_command',
    #         'distance_threshold': STRICT_GOAL_REACHED_DISTANCE_THRESHOLD,
    #         'angular_threshold': STRICT_GOAL_REACHED_ANGULAR_THRESHOLD,
    #     }
    # )

    # goal_reached_movement_penalty = RewTerm(
    #     func=nav_mdp.pose_2d_goal_callback_reward,
    #     weight=-0.05,
    #     params={
    #         'func': mdp.lin_vel_l2,
    #         'command_name': 'pose_2d_command',
    #         'distance_threshold': STRICT_GOAL_REACHED_DISTANCE_THRESHOLD,
    #         'angular_threshold': STRICT_GOAL_REACHED_ANGULAR_THRESHOLD,
    #     }
    # )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # heading_sin_cos = ObsTerm(
        #     func=mdp.root_yaw_sin_cos, noise=Unoise(n_min=-0.05, n_max=0.05)
        # )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_2d_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_2d_command"})
        # scalar_velocity_command = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "scalar_velocity_command"}
        # )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        count_down = ObsTerm(
            func=mdp.count_down,
            params={"episode_length": EPISDOE_LENGTH}
        )
        osbtacles_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("obstacle_scanner"), 
                    "max": 10.0},
            noise=Unoise(n_min=-0.1, n_max=0.1))
        
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner"), 'offset': 0.4},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

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
        func=mdp.root_velocity_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  
            "max_velocity": 10.0
        }
    )

@configclass
class NavigationEnd2EndEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
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

@configclass
class NavigationEnd2EndNoEncoderEnvCfg(NavigationEnd2EndEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

@configclass
class NavigationEnd2EndEnvCfg_PLAY(NavigationEnd2EndEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 10

@configclass
class NavigationEnd2EndNoEncoderEnvCfg_PLAY(NavigationEnd2EndNoEncoderEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 10

        if USE_TEST_ENV:
            self.curriculum = None
            self.rewards.undesired_contacts_discrete_obstacles = None
            self.terminations.base_contact = DoneTerm(
                func=mdp.illegal_contact,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", 
                                                    body_names=["base", "Head_upper", ".*hip", "Head_lower", ".*thigh"]), 
                        "threshold": 0.2})
            
            self.terminations.base_contact_discrete_obstacles = None

            self.scene.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="test_generator",
                test_terrain_generator=TEST_TERRAIN_CFG,
                max_init_terrain_level=10,
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

            # goal_set_1 = [(-7, -5), (-7, -5)]
            # goal_set_2 = [(5, 7), (-7, -5)]
            # goal_set_3 = [(5, 7), (5, 7)]
            # goal_set_4 = [(-7, -5), (5, 7)]

            # goal_set = goal_set_2

            # self.commands.pose_2d_command = mdp.UniformPose2dCommandCfg(
            #     asset_name="robot",
            #     simple_heading=False,
            #     ranges=mdp.UniformPose2dCommandCfg.Ranges(
            #         heading=(-math.pi, math.pi),
            #         pos_x=goal_set[0],
            #         pos_y=goal_set[1]
            #     ),
            #     resampling_time_range=(1.5*EPISDOE_LENGTH, 1.5*EPISDOE_LENGTH),
            #     debug_vis=True
            # )

            goal_set_1 = [(2, 2), (8, 8)]

            goal_set = goal_set_1

            self.commands.pose_2d_command = mdp.UniformPose2dCommandCfg(
                asset_name="robot",
                simple_heading=False,
                ranges=mdp.UniformPose2dCommandCfg.Ranges(
                    heading=(-math.pi, math.pi),
                    pos_x=goal_set[0],
                    pos_y=goal_set[1]
                ),
                resampling_time_range=(1.5*EPISDOE_LENGTH, 1.5*EPISDOE_LENGTH),
                debug_vis=True
            )

