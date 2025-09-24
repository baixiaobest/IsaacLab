from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, \
  RslRlPpoEncoderActorCriticCfg, RslRlSymmetryCfg
import torch

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
    # 1) Unglue the flat vector into a 1×16x16 “image”
    { 'type':   'reshape',
      'input_size': 416,
      'shape': [1, 16, 26]
    },

    # 2) One dilated conv to capture ~1 m radius features
    { 'type':        'conv',
      'out_channels': 16,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },

    # 3) 2×2 max‑pool to half H×W → 8x8
    { 'type':       'pool',
      'kernel_size': 2,
      'stride':      2
    },

    # 4) One plain conv to mix channels
    { 'type':        'conv',
      'out_channels': 16,
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
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavPPOConfig
    logger="wandb"
    wandb_project="navigation"

"""
END2END CONFIGURATION
"""

e2e_cnn_config = [
    # 1) Unglue the flat vector into a 1×21x21 “image”
    { 'type':   'reshape',
      'input_size': 441,
      'shape': [1, 21, 21]
    },

    # 2) convolution, -> 16x21x21
    { 'type':        'conv',
      'out_channels': 8,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },

    # 3) 2×2 max‑pool to half H×W → 8x10x10
    { 'type':       'pool',
      'kernel_size': 2,
      'stride':      2
    },

    # 4) Convolution -> 16x10x10
    { 'type':        'conv',
      'out_channels': 16,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },
    # 5) 2×2 max‑pool to half H×W → 16x5x5
    { 'type':       'pool',
      'kernel_size': 2,
      'stride':      2
    },

    # 6) Convolution -> 32x5x5
    { 'type':        'conv',
      'out_channels': 32,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },
    # 7) 2×2 adaptive average‑pool,  32x2x2
    { 'type':       'adaptive_pool',
      'output_size': (2, 2)  # Directly specify the output dimensions
    }
]

NavE2EPPOConfig = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.995,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

def no_encoder_augmentation(
        env, obs: torch.Tensor, actions: torch.Tensor, obs_type:str="policy") \
  -> tuple[torch.Tensor, torch.Tensor]:
    """Data augmentation function that flip robot states and obstacle scan horizontally"""
    augmented_obs = None
    augmented_action = None

    def swap_joints_horizontal(joints):
      joints_flipped = torch.zeros_like(joints, device=joints.device)
      for i in range(3):
        joints_flipped[:, 4*i + 0] = joints[:, 4*i + 1]
        joints_flipped[:, 4*i + 1] = joints[:, 4*i + 0]
        joints_flipped[:, 4*i + 2] = joints[:, 4*i + 3]
        joints_flipped[:, 4*i + 3] = joints[:, 4*i + 2]
      return joints_flipped

    if obs is not None:
      num_sample, obs_dim = obs.shape
      augmented_obs = torch.zeros((2*num_sample, obs_dim), device=obs.device, dtype=obs.dtype)
      augmented_obs[0:num_sample, :] = obs
      
      base_lin_vel = obs[:, 0:3]
      base_ang_vel = obs[:, 3:6]
      pose_2d_command = obs[:, 6:10]
      projected_gravity = obs[:, 10:13]
      joint_pos = obs[:, 13:25]
      joint_vel = obs[:, 25:37]
      last_action = obs[:, 37:49]
      count_down = obs[:, 49:50]
      obstacle_scan = obs[:, 50: 50+32]

      # Flip horizontally
      base_lin_vel_flipped = base_lin_vel * torch.tensor([1.0, -1.0, 1.0], device=obs.device)
      base_ang_vel_flipped = base_ang_vel * torch.tensor([-1.0, 1.0, -1.0], device=obs.device)
      pose_2d_command_flipped = pose_2d_command * torch.tensor([1.0, -1.0, 1.0, -1.0], device=obs.device)
      projected_gravity_flipped = projected_gravity * torch.tensor([1.0, -1.0, 1.0], device=obs.device)
      
      joint_pos_flipped = swap_joints_horizontal(joint_pos)

      joint_vel_flipped = swap_joints_horizontal(joint_vel)
      
      last_action_flipped = swap_joints_horizontal(last_action)

      count_down_flipped = count_down
      obstacle_scan_flipped = torch.flip(obstacle_scan, dims=[1])

      augmented_obs[num_sample:2*num_sample, :] = torch.cat([
          base_lin_vel_flipped,
          base_ang_vel_flipped,
          pose_2d_command_flipped,
          projected_gravity_flipped,
          joint_pos_flipped,
          joint_vel_flipped,
          last_action_flipped,
          count_down_flipped,
          obstacle_scan_flipped
      ], dim=1)

    if actions is not None:
      num_sample, action_dim = actions.shape
      augmented_action = torch.zeros((2*num_sample, action_dim), device=actions.device, dtype=actions.dtype)
      augmented_action[0:num_sample, :] = actions
      augmented_action[num_sample:, :] = swap_joints_horizontal(actions)

    return augmented_obs, augmented_action


ObstacleScanNoEncoderSymmetryConfig = RslRlSymmetryCfg(
    use_data_augmentation=True,
    data_augmentation_func=no_encoder_augmentation,
    use_mirror_loss=True,
    mirror_loss_coeff=0.1
)

NavE2EObstacleScanNoEncoderPPOConfig = RslRlPpoAlgorithmCfg(
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
        # symmetry_cfg=ObstacleScanNoEncoderSymmetryConfig
    )

@configclass
class UnitreeGo2NavigationEnd2EndEnvCfgPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_navigation_end2end_v0"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=0.8,
        noise_clip=1.0,
        encoder_dims=e2e_cnn_config,
        encoder_type="cnn",
        encoder_obs_normalize=False,
        share_encoder_with_critic=True,
        actor_hidden_dims=[128, 128, 64, 64],
        critic_hidden_dims=[128, 128, 64, 64],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavE2EPPOConfig
    logger="wandb"
    wandb_project="e2e_navigation"

@configclass
class UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_navigation_end2end_v0"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=0.8,
        noise_clip=1.0,
        encoder_dims=None,
        encoder_obs_normalize=False,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavE2EObstacleScanNoEncoderPPOConfig
    logger="wandb"
    wandb_project="e2e_navigation"

@configclass
class UnitreeGo2NavigationEnd2EndNoEncoderStairsOnlyEnvCfgPPORunnerCfg_v0(UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_navigation_stairs_v0"
    empirical_normalization = False
    policy = RslRlPpoEncoderActorCriticCfg(
        init_noise_std=0.8,
        noise_clip=1.0,
        encoder_dims=None,
        encoder_obs_normalize=False,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
        tanh_output=True,
    )
    algorithm = NavE2EObstacleScanNoEncoderPPOConfig
    wandb_project="stairs_climbing"
