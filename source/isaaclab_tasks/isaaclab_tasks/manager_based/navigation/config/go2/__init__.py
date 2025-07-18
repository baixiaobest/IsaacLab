import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Navigation-Mountain-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationMountainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Mountain-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationMountainEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Mountain-Unitree-Go2-No-Scandots-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationMountainNoScandotsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationNoScandotsPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Mountain-Unitree-Go2-No-Scandots-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationMountainNoScandotsCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationNoScandotsPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Mountain-Unitree-Go2-CNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationCNNCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationCNNPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Mountain-Unitree-Go2-CNN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationCNNCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationCNNPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.e2e_navigation_env_cfg:NavigationEnd2EndEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-No-Encoder-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.e2e_navigation_env_cfg:NavigationEnd2EndNoEncoderEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0"
    })