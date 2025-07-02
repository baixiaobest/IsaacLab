# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoEncoderActorCriticCfg

PPOConfig = RslRlPpoAlgorithmCfg(
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
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=1.0,
        encoder_dims=[412, 256, 128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="quadruped"

@configclass
class UnitreeGo2RoughTeacherPPORunnerCfg_v2(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_rough_teacher_v2"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=1.0,
        encoder_dims=[397, 256, 128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
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
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=1.0,
        encoder_dims=[397, 256, 128, 64, 32],
        actor_hidden_dims=[512, 256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="quadruped"

@configclass
class UnitreeGo2RoughTeacherScandotsOnlyPPORunnerCfg(UnitreeGo2RoughTeacherPPORunnerCfg):
    experiment_name = "unitree_go2_rough_teacher_scandots_only"
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=1.0,
        encoder_dims=[336, 256, 128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

NavPPOConfig = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
class UnitreeGo2NavigationPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_navigation_v0"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=0.8,
        noise_clip=1.0,
        encoder_dims=[336, 128, 64, 32],
        actor_hidden_dims=[64, 64, 64, 32],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavPPOConfig
    logger="wandb"
    wandb_project="navigation"

@configclass
class UnitreeGo2NavigationNoScandotsPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_navigation_no_scandots_v0"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=0.5,
        noise_clip=0.6,
        encoder_dims=None,
        actor_hidden_dims=[64, 64, 128, 64],
        critic_hidden_dims=[64, 64, 128, 64],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavPPOConfig
    logger="wandb"
    wandb_project="navigation"

cnn_config = [
    # First layer: reshape flat input to image dimensions
    {
        'type': 'reshape',
        'input_size': 336,       # Input size of encoder
        'shape': [1, 16, 21]     # Reshape to 1-channel image of size 16x21, need to modify this if height scans changes
    },
    # Convolutional layer
    {
        'type': 'conv',
        'out_channels': 4,
        'kernel_size': 3,
        'dilation': 2,
        'stride': 1,
        'padding': 1
    },
    # Another convolutional layer
    {
        'type': 'conv',
        'out_channels': 16,
        'kernel_size': 3,
        'dilation': 3,
        'stride': 1,
        'padding': 1
    }
]

@configclass
class UnitreeGo2NavigationCNNPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_navigation_v0"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=0.6,
        noise_clip=0.8,
        encoder_dims=cnn_config,
        encoder_type="cnn",
        actor_hidden_dims=[64, 64, 64, 32],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavPPOConfig
    logger="wandb"
    wandb_project="navigation"
