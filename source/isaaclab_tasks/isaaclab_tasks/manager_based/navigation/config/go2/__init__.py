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

gym.register(
    id="Isaac-End2End-Navigation-Unitree-Go2-2nd-Stage-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.e2e_navigation_env_cfg:NavigationEnd2End2ndStageEnvCfg",
        "rsl_rl_cfg_entry_point":f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.e2e_navigation_env_cfg:NavigationEnd2EndEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-No-Encoder-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.e2e_navigation_env_cfg:NavigationEnd2EndNoEncoderEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0"
    })

###############
# RVO2 Crowd Navigation
###############

gym.register(
    id="Isaac-Navigation-RVO2-Crowd-Unitree-Go2-v0",
    entry_point="isaaclab_tasks.manager_based.navigation.config.go2.rvo2_navigation_env_cfg:RVO2NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rvo2_navigation_env_cfg:RVO2NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RVO2CrowdPPORunnerCfg_v0",
    },
)

gym.register(
    id="Isaac-Navigation-RVO2-Crowd-Unitree-Go2-Play-v0",
    entry_point="isaaclab_tasks.manager_based.navigation.config.go2.rvo2_navigation_env_cfg:RVO2NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rvo2_navigation_env_cfg:RVO2NavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RVO2CrowdPPORunnerCfg_v0",
    },
)

###############
# Stair Climbing
###############

gym.register(
    id="Isaac-Navigation-Pyramids-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationPyramidStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderStairsOnlyEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-No-Encoder-Unitree-Go2-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndStairsOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderStairsOnlyEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-CNN-Unitree-Go2-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndStairsOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-CNN-RND-Unitree-Go2-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndStairsOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2End_CNN_RND_PPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Pyramids-CNN-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationPyramidStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-CNN-Unitree-Go2-Spiral-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndSpiralStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0",
    },
)

# Play versions

gym.register(
    id="Isaac-End2End-Navigation-No-Encoder-Unitree-Go2-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndStairsOnlyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderStairsOnlyEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-CNN-Unitree-Go2-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndStairsOnlyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Pyramid-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationPyramidStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndNoEncoderStairsOnlyEnvCfgPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-Navigation-Pyramid-CNN-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationPyramidStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0"
    })

gym.register(
    id="Isaac-End2End-Navigation-CNN-Unitree-Go2-Spiral-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_climbing_env_cfg:NavigationEnd2EndSpiralStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0"
    }
)