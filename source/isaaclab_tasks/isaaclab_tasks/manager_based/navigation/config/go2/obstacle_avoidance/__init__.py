import gymnasium as gym
from .. import agents

from . import locomotion_env_cfg
from . import obstacle_avoidance_env_cfg
from . import temporal_lidar_env_cfg
from . import observation_modifiers

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

