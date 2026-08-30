"""Fixed-coverage configuration used only for LiDAR velocity dataset collection."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .kp_mixed_scenario_env_cfg import MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY
from .lidar_velocity_data_env import FixedCoverageTerrainImporter, reset_fixed_level_pedestrian_crowd
from .pedestrian_terrains import build_mixed_static_pedestrian_corridor


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
