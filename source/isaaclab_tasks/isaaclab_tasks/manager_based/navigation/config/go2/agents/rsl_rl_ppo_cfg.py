from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoEncoderActorCriticCfg

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
        encoder_dims=[1066, 128, 64, 32],
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
        noise_clip=1.0,
        encoder_dims=None,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavPPOConfig
    logger="wandb"
    wandb_project="navigation"

cnn_config = [
    # 1) Unglue the flat vector into a 1×26×41 “image”
    { 'type':   'reshape',
      'input_size': 1066,
      'shape': [1, 26, 41]
    },

    # 2) One dilated conv to capture ~1 m radius features
    { 'type':        'conv',
      'out_channels': 16,
      'kernel_size':   5,
      'dilation':      2,
      'stride':        1,
      'padding':       4
    },

    # 3) 2×2 max‑pool to half H×W → 13×20
    { 'type':       'pool',
      'kernel_size': 2,
      'stride':      2
    },

    # 4) One plain conv to mix channels
    { 'type':        'conv',
      'out_channels': 32,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },
    # 5) 2×2 adaptive average‑pool
    { 'type':       'adaptive_pool',
      'output_size': (2, 2)  # Directly specify the output dimensions
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
        init_noise_std=0.5,
        noise_clip=1.0,
        encoder_dims=cnn_config,
        encoder_type="cnn",
        encoder_obs_normalize=False,
        share_encoder_with_critic=True,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavPPOConfig
    logger="wandb"
    wandb_project="navigation"
