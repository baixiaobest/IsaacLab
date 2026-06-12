import gymnasium as gym
from .. import agents

from . import locomotion_env_cfg
from . import obstacle_avoidance_env_cfg
from . import temporal_lidar_env_cfg
from . import observation_modifiers
from . import pedestrian_scenario_mixins

###############
# Locomotion Velocity
###############

gym.register(
    id="Isaac-Locomotion-Vel-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionVelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2LocomotionVelPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Locomotion-Vel-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionVelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2LocomotionVelPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Locomotion-Vel-Unitree-Go2-Rollout-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionVelEnvCfg_ROLLOUT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2LocomotionVelPPORunnerCfg_v0",
    },
)

######################
# Obstacle Avoidance Navigation
######################

gym.register(
    id="Isaac-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacle_avoidance_env_cfg:ObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacle_avoidance_env_cfg:ObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.temporal_lidar_env_cfg:TemporalLidarObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.temporal_lidar_env_cfg:TemporalLidarObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.temporal_lidar_env_cfg:TemporalLidarPredictionObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0",
    },
)

######################
# Pedestrian Flow (scenarios a/b: with/against pedestrian flow)
######################

_PED_CROWD_ENTRY_POINT = f"{__name__}.pedestrian_crowd_env:PedestrianCrowdNavigationEnv"

gym.register(
    id="Isaac-Pedestrian-Flow-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianFlowObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Flow-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianFlowObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Flow-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianFlowTemporalLidarObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Flow-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pedestrian_scenario_mixins:PedestrianFlowTemporalLidarPredictionObstacleAvoidanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0",
    },
)

######################
# Pedestrian Crossing (scenario c: robot crosses the pedestrian flow)
######################

gym.register(
    id="Isaac-Pedestrian-Crossing-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianCrossingObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Crossing-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianCrossingObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Crossing-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianCrossingTemporalLidarObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Crossing-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pedestrian_scenario_mixins:PedestrianCrossingTemporalLidarPredictionObstacleAvoidanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0",
    },
)

