import gymnasium as gym
from .. import agents

from . import locomotion_env_cfg
from . import obstacle_avoidance_env_cfg
from . import temporal_lidar_env_cfg
from . import observation_modifiers
from . import pedestrian_scenario_mixins
from . import mixed_scenario_mixins

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
# Pedestrian (unified: co-trains with/against flow + crossing scenarios)
######################

_PED_CROWD_ENTRY_POINT = f"{__name__}.pedestrian_crowd_env:PedestrianCrowdNavigationEnv"

gym.register(
    id="Isaac-Pedestrian-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pedestrian_scenario_mixins:PedestrianTemporalLidarObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Pedestrian-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pedestrian_scenario_mixins:PedestrianTemporalLidarPredictionObstacleAvoidanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0",
    },
)

######################
# Mixed static + pedestrian co-training (single policy on both terrain families)
######################

gym.register(
    id="Isaac-Mixed-Static-Pedestrian-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_scenario_mixins:MixedObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Mixed-Static-Pedestrian-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_scenario_mixins:MixedObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_scenario_mixins:MixedTemporalLidarObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_scenario_mixins:MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mixed_scenario_mixins:MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    entry_point=_PED_CROWD_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mixed_scenario_mixins:MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0",
    },
)

