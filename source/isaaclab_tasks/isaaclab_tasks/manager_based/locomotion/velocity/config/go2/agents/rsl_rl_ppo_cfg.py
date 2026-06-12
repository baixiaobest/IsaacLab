# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlEncoderModelCfg,
    RslRlMLPModelCfg,
)


def _teacher_actor(encoder_dims, actor_hidden_dims, init_std=0.8):
    """Actor with an MLP encoder over the observation tail (migrated from EncoderActorCritic, no tanh, unshared)."""
    return RslRlEncoderModelCfg(
        hidden_dims=actor_hidden_dims,
        activation="elu",
        distribution_cfg=RslRlEncoderModelCfg.GaussianDistributionCfg(init_std=init_std, std_type="scalar"),
        encoder_dims=encoder_dims,
        encoder_type="mlp",
    )


def _teacher_critic(critic_hidden_dims):
    """Plain MLP critic over the full observation (EncoderActorCritic did not encode the critic when unshared)."""
    return RslRlMLPModelCfg(hidden_dims=critic_hidden_dims, activation="elu", distribution_cfg=None)


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="quadruped"

@configclass
class UnitreeGo2RoughTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_rough_teacher"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _teacher_actor([412, 256, 128, 64, 32], [512, 256, 128])
    critic = _teacher_critic([512, 256, 128])
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="quadruped"

@configclass
class UnitreeGo2RoughTeacherPPORunnerCfg_v2(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_rough_teacher_v2"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _teacher_actor([397, 256, 128, 64, 32], [512, 256, 128])
    critic = _teacher_critic([512, 256, 128])
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="quadruped"

@configclass
class UnitreeGo2RoughTeacherPPORunnerCfg_v3(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_rough_teacher_v2"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _teacher_actor([397, 256, 128, 64, 32], [512, 256, 128, 128])
    critic = _teacher_critic([512, 256, 128])
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="quadruped"

@configclass
class UnitreeGo2RoughTeacherScandotsOnlyPPORunnerCfg(UnitreeGo2RoughTeacherPPORunnerCfg):
    experiment_name = "unitree_go2_rough_teacher_scandots_only"
    actor = _teacher_actor([336, 256, 128, 64, 32], [512, 256, 128])
    critic = _teacher_critic([512, 256, 128])

@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]