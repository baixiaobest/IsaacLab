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

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, DIVERSE_TERRAINS_CFG, NAVIGATION_TERRAINS_CFG # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

EPISDOE_LENGTH = 10.0
GOAL_REACHED_ACTIVE_AFTER = 6.0
SIM_DT = 0.005
GOAL_REACHED_DISTANCE_THRESHOLD = 0.5
GOAL_REACHED_ANGULAR_THRESHOLD = 0.1

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=NAVIGATION_TERRAINS_CFG,
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
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
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
    pose_2d_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi)
        ),
        resampling_time_range=(1.5*EPISDOE_LENGTH, 1.5*EPISDOE_LENGTH),
        debug_vis=True
    )
    # This controls the average velocity from origin to target.
    scalar_velocity_command = mdp.ScalarVelocityCommandCfg(
        asset_name="robot",
        velocity_range=(0.5, 2.0),
        resampling_time_range=(1.5*EPISDOE_LENGTH, 1.5*EPISDOE_LENGTH),
        debug_vis=True
    )

@configclass
class RewardsCfg:
    # Task reward
    goal_reached = RewTerm(
        func=nav_mdp.pose_2d_command_goal_reached_reward,
        weight=1.0,
        params={
            'command_name': 'pose_2d_command',
            'distance_threshold': GOAL_REACHED_DISTANCE_THRESHOLD,
            'angular_threshold': GOAL_REACHED_ANGULAR_THRESHOLD,
            'distance_reward_multiplier': 1.3,
            'angular_reward_multiplier': 1.3,
            'active_after_time': GOAL_REACHED_ACTIVE_AFTER,
        }
    )

    # goal_reached = RewTerm(
    #     func=nav_mdp.terrain_adaptive_pose_2d_command_goal_reached_reward,
    #     weight=1.0,
    #     params={
    #         'command_name': 'pose_2d_command',
    #         'max_distance_threshold': 0.8,
    #         'min_distance_threshold': GOAL_REACHED_DISTANCE_THRESHOLD,
    #         'max_angular_threshold': 0.5,
    #         'min_angular_threshold': GOAL_REACHED_ANGULAR_THRESHOLD,
    #         'distance_reward_multiplier': 1.3,
    #         'angular_reward_multiplier': 1.3,
    #         'active_after_time': GOAL_REACHED_ACTIVE_AFTER,
    #     }
    # )

    # Guide the task reward due to sparsity of task reward
    progress_reward = RewTerm(
        func=nav_mdp.pose_2d_command_progress_reward,
        weight=0.1,
        params={
            'command_name': 'pose_2d_command',
            'std': 1.0 * SIM_DT
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
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_upper"]), 
                "threshold": 0.2},
    )
    # Additional undesired contacts for discrete obstacle terrain types
    undesired_contacts_discrete_obstacles = RewTerm(
        func=nav_mdp.terrain_specific_callback,
        weight=-10.0,
        params={
            "terrain_names": ["discrete_obstacles"],
            "func": mdp.undesired_contacts,
            "callback_params": {
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_lower", ".*hip"]),
                "threshold": 0.2
            }
        })
    # Less serious contacts
    mild_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_lower"]),
            "threshold": 0.1,
        })
    
    # Energy minimization
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # Avoid jerky action
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Reduce motion at goal
    goal_reached_action_penalty = RewTerm(
        func=nav_mdp.pose_2d_goal_callback_reward,
        weight=-0.05,
        params={
            'func': mdp.action_rate_l2,
            'command_name': 'pose_2d_command',
            'distance_threshold': GOAL_REACHED_DISTANCE_THRESHOLD,
            'angular_threshold': GOAL_REACHED_ANGULAR_THRESHOLD,
        }
    )
    # Better pose at goal
    goal_joint_deviation_penalty = RewTerm(
        func=nav_mdp.pose_2d_goal_callback_reward,
        weight=-0.05,
        params={
            'func': mdp.joint_deviation_l1,
            'command_name': 'pose_2d_command',
            'distance_threshold': GOAL_REACHED_DISTANCE_THRESHOLD,
            'angular_threshold': GOAL_REACHED_ANGULAR_THRESHOLD,
        })

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
        # scalar_velocity_command = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "scalar_velocity_command"}
        # )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
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
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_upper"]), 
                "threshold": 1.0},
    )

    base_contact_discrete_obstacles = DoneTerm(
        func=nav_mdp.terrain_specific_callback,
        params={
            "terrain_names": ["discrete_obstacles"],
            "func": mdp.illegal_contact,
            "callback_params": {
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", 
                    body_names=[".*hip", "Head_lower"]),
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
        # self.commands.scalar_velocity_command.velocity_range = (0.3, 1.0)

