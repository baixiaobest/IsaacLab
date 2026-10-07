"""Fixed-coverage configuration used only for LiDAR velocity dataset collection."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .kp_mixed_scenario_env_cfg import MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY
from .lidar_velocity_data_env import FixedCoverageTerrainImporter, reset_fixed_level_pedestrian_crowd
from .pedestrian_terrains import build_mixed_static_pedestrian_corridor
from .obstacle_avoidance_env_cfg import LIDAR_MAX_DISTANCE
from .temporal_lidar_env_cfg import (
    PREDICTOR_LIDAR_HORIZON, TEMPORAL_LIDAR_FOV_DEG, TEMPORAL_LIDAR_HISTORY_KEY,
    TEMPORAL_LIDAR_NUM_BINS, TEMPORAL_LIDAR_RAYS, PredictorLidarObservationsCfg,
)


@configclass
class PredictorDataObservationsCfg(PredictorLidarObservationsCfg):
    @configclass
    class CleanPredictorCfg(ObsGroup):
        obstacle_scan = ObsTerm(func=mdp.TemporalLidarScan, params={
            "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
            "horizon": PREDICTOR_LIDAR_HORIZON,
            "num_bins": TEMPORAL_LIDAR_NUM_BINS,
            "fov_degrees": TEMPORAL_LIDAR_FOV_DEG,
            "max_distance": LIDAR_MAX_DISTANCE,
            "pos_noise_std": 0.0,
            "yaw_drift_std_rad_per_scan": 0.0,
            "include_validity": True,
            "history_key": TEMPORAL_LIDAR_HISTORY_KEY,
            "history_num_rays": TEMPORAL_LIDAR_RAYS,
        })

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    predictor_clean: CleanPredictorCfg = CleanPredictorCfg()


@configclass
class MixedTemporalLidarKpPointVelocityDataEnvCfg(MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY):
    """Kp temporal-LiDAR task with fixed 10-level x 4-column data coverage."""

    observations: PredictorDataObservationsCfg = PredictorDataObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 40
        self.scene.terrain.class_type = FixedCoverageTerrainImporter
        terrain_generator = build_mixed_static_pedestrian_corridor(
            discrete_obstacles_proportion=1.0,
            concentric_maze_proportion=1.0,
            ped_corridor_proportion=1.0,
            indoor_ped_corridor_proportion=1.0,
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
        self.observations.predictor.obstacle_scan.params["record_reflection_classes"] = True

        # No terrain/density curriculum may mutate the fixed coverage assignment.
        self.curriculum.terrain_levels = None
        self.curriculum.discrete_obstacles = None
        self.curriculum.concentric_maze = None
        self.curriculum.open_dynamic_terrain_level = None
        self.curriculum.indoor_dynamic_terrain_level = None
        self.curriculum.pedestrian_density = None
        self.events.reset_pedestrians = EventTerm(
            func=reset_fixed_level_pedestrian_crowd,
            mode="reset",
            params={"flow_dir": 1.0},
        )
