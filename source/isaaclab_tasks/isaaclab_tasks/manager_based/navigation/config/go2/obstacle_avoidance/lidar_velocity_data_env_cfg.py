"""Fixed-coverage configuration used only for LiDAR velocity dataset collection."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .held_scan_lidar_env import HeldScanLidarCfg
from .kp_mixed_scenario_env_cfg import MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY
from .lidar_velocity_data_env import FixedCoverageTerrainImporter, reset_fixed_level_pedestrian_crowd
from .mixed_scenario_mixins import build_obstacle_scanner_360
from .pedestrian_terrains import build_mixed_static_pedestrian_corridor
from .temporal_lidar_env_cfg import TemporalLidar360ObservationsCfg


@configclass
class MixedTemporalLidarKpPointVelocityDataEnvCfg(MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY):
    """Kp temporal-LiDAR task with fixed 10-level x 4-column data coverage."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 40
        self.scene.terrain.class_type = FixedCoverageTerrainImporter
        terrain_generator = build_mixed_static_pedestrian_corridor(
            discrete_obstacles_proportion=1.0,
            concentric_maze_proportion=1.0,
            ped_corridor_proportion=2.0,
            num_cols=4,
        )
        # The shared terrain defaults begin the discrete-obstacle curriculum
        # with zero high obstacles.  This data-only variant must retain useful
        # static LiDAR returns even at fixed level 0, while still growing more
        # cluttered at higher levels.
        discrete_cfg = terrain_generator.sub_terrains["discrete_obstacles"]
        discrete_cfg.min_num_high_obstacles = 4
        self.scene.terrain.terrain_generator = terrain_generator
        self.scene.terrain.max_init_terrain_level = None
        self.scene.obstacle_scanner.update_mesh_ids = True

        # No terrain/density curriculum may mutate the fixed coverage assignment.
        self.curriculum.terrain_levels = None
        self.curriculum.discrete_obstacles = None
        self.curriculum.concentric_maze = None
        self.curriculum.ped_corridor = None
        self.curriculum.pedestrian_density = None
        self.events.reset_pedestrians = EventTerm(
            func=reset_fixed_level_pedestrian_crowd,
            mode="reset",
            params={"flow_dir": 1.0},
        )


@configclass
class MixedTemporalLidarKp360PointVelocityDataEnvCfg(MixedTemporalLidarKpPointVelocityDataEnvCfg):
    """360-degree variant: adds a second, independent full-circle scanner + observation group.

    Everything else (terrain, fixed coverage, pedestrian crowd, Kp driving action,
    disabled curricula) is inherited unchanged from the 180-degree data-collection
    task above. The pretrained Kp policy still drives the robot using its original
    128-bin/180-degree ``policy`` observation group; the new ``policy_360``/``critic_360``
    groups exist solely so ``rollout.py`` can read genuinely full-circle scans for the
    360-degree lidar-velocity-predictor dataset.
    """

    observations: TemporalLidar360ObservationsCfg = TemporalLidar360ObservationsCfg()
    held_scan_lidar_360_enabled: bool = True
    held_scan_lidar_360: HeldScanLidarCfg = HeldScanLidarCfg(sensor_name="obstacle_scanner_360", full_fan_ray_count=512)

    def __post_init__(self):
        super().__post_init__()
        # Not a declared _MixedSceneCfg field on purpose (see build_obstacle_scanner_360's
        # docstring): InteractiveScene discovers sensors via self.cfg.__dict__, so assigning
        # it here scopes the extra 512-ray raycaster to only this task.
        self.scene.obstacle_scanner_360 = build_obstacle_scanner_360()
        self.scene.obstacle_scanner_360.update_mesh_ids = True
