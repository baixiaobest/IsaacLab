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
class UnitreeGo2RoughTeacherPPORunnerCfg_test(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_rough_teacher_test"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=1.0,
        encoder_dims=[397, 256, 128, 64, 32],
        actor_hidden_dims=[512, 256, 128, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = PPOConfig
    # resume = True
    # run_name = "2025-06-21_23-16-12"
    # load_checkpoint = "model_199.pt"
    logger="wandb"
    wandb_project="isaaclab"

@configclass
class UnitreeGo2RoughDepthCameraOnlyPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_rough_depth_camera_only"
    empirical_normalization = False
    cnn_config = [
        # Reshape flat input to image dimensions
        {
            'type': 'reshape',
            'input_size': 320 * 240,  # 76800
            'shape': [1, 320, 240]
        },
        # Initial pooling: reduce to 160x120
        {
            'type': 'pool',
            'kernel_size': 2
        },
        # Conv: output size 80x60
        {
            'type': 'conv',
            'out_channels': 8,
            'kernel_size': 5,
            'stride': 2,
            'padding': 2
        },
        # Conv: output size 40x30
        {
            'type': 'conv',
            'out_channels': 16,
            'kernel_size': 3,
            'stride': 2,
            'padding': 1
        },
        # Max pool: down to 20x15
        {
            'type': 'pool',
            'kernel_size': 2
        },
        # Conv: output size 20x15
        {
            'type': 'conv',
            'out_channels': 16,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
    ]
     
    
    policy = RslRlPpoEncoderActorCriticCfg(
        encoder_obs_normalize = True,
        share_encoder_with_critic=True,
        init_noise_std=1.0,
        encoder_dims=cnn_config,
        encoder_type="cnn",
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = PPOConfig
    logger="wandb"
    wandb_project="isaaclab"

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
