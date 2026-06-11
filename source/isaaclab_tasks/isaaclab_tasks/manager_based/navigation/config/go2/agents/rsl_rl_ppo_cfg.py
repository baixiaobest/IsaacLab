from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlEncoderModelCfg,
    RslRlMLPModelCfg,
    RslRlLidarModelCfg,
    RslRlLidarPredictionCfg,
    RslRlSymmetryCfg,
    RslRlRndCfg,
)
import torch


# ---------------------------------------------------------------------------
# Helpers for the encoder-model navigation configs (migrated from EncoderActorCritic).
# The actor encodes the tail of the flat "policy" observation; the critic either shares that encoder
# (share_cnn_encoders=True on the algorithm) or, when not shared, is a plain MLP over the full observation.
# ---------------------------------------------------------------------------
def _enc_actor(hidden_dims, init_std, encoder_dims=None, encoder_type="mlp",
               encoder_obs_normalize=False, tanh_output=True):
    return RslRlEncoderModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        distribution_cfg=RslRlEncoderModelCfg.GaussianDistributionCfg(init_std=init_std, std_type="scalar"),
        encoder_dims=encoder_dims,
        encoder_type=encoder_type,
        encoder_obs_normalize=encoder_obs_normalize,
        tanh_output=tanh_output,
    )


def _enc_critic_shared(hidden_dims, encoder_dims, encoder_type="cnn", encoder_obs_normalize=False):
    return RslRlEncoderModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        distribution_cfg=None,
        encoder_dims=encoder_dims,
        encoder_type=encoder_type,
        encoder_obs_normalize=encoder_obs_normalize,
    )


def _mlp_critic(hidden_dims):
    return RslRlMLPModelCfg(hidden_dims=hidden_dims, activation="elu", distribution_cfg=None)

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
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([64, 64, 64, 32], init_std=0.8, encoder_dims=[1066, 128, 64, 32], encoder_type="mlp",
                       tanh_output=True)
    critic = _mlp_critic([256, 256, 128])
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
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128], init_std=0.5, encoder_dims=None, tanh_output=True)
    critic = _mlp_critic([128, 128])
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

# NavPPOConfig variant that shares the actor's CNN encoder with the critic (share_encoder_with_critic=True).
NavPPOShareCNNConfig = RslRlPpoAlgorithmCfg(
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
    share_cnn_encoders=True,
)

@configclass
class UnitreeGo2NavigationCNNPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_go2_navigation_v0"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128, 64], init_std=0.5, encoder_dims=cnn_config, encoder_type="cnn",
                       encoder_obs_normalize=False, tanh_output=True)
    critic = _enc_critic_shared([128, 128, 64], encoder_dims=cnn_config, encoder_type="cnn")
    algorithm = NavPPOShareCNNConfig
    logger="wandb"
    wandb_project="navigation"

"""
END2END CONFIGURATION
"""

new_e2e_cnn_config = [
    # 1) Unglue the flat vector into a 1×21x21 “image”
    { 'type':   'reshape',
      'input_size': 441,
      'shape': [1, 21, 21]
    },

    # 2) convolution, -> 8x21x21
    { 'type':        'conv',
      'out_channels': 8,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },

    # 3) Convolution -> 16x21x21
    { 'type':        'conv',
      'out_channels': 16,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },

    # 4) Convolution -> 32x21x21
    { 'type':        'conv',
      'out_channels': 32,
      'kernel_size':   3,
      'dilation':      1,
      'stride':        1,
      'padding':       1
    },
    # 5) 1×1 adaptive average‑pool,  32x1x1
    { 'type':       'adaptive_pool',
      'output_size': (1, 1)  # Directly specify the output dimensions
    }
]

e2e_cnn_config = [
    # 1) Unglue the flat vector into a 1×21x21 “image”
    { 'type':   'reshape',
      'input_size': 441,
      'shape': [1, 21, 21]
    },

    # 2) convolution, -> 8x21x21
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
        # All users of this config share the actor's CNN encoder with the critic.
        share_cnn_encoders=True,
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
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128, 64, 64], init_std=0.8, encoder_dims=e2e_cnn_config, encoder_type="cnn",
                       encoder_obs_normalize=False, tanh_output=True)
    critic = _enc_critic_shared([128, 128, 64, 64], encoder_dims=e2e_cnn_config, encoder_type="cnn")
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
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128, 64], init_std=0.8, encoder_dims=None, tanh_output=True)
    critic = _mlp_critic([128, 128])
    algorithm = NavE2EObstacleScanNoEncoderPPOConfig
    logger="wandb"
    wandb_project="e2e_navigation"

@configclass
class UnitreeGo2NavigationEnd2EndNoEncoderStairsOnlyEnvCfgPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_navigation_stairs_v0"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128, 64], init_std=0.8, encoder_dims=None, tanh_output=True)
    critic = _mlp_critic([128, 128, 64])
    algorithm = NavE2EObstacleScanNoEncoderPPOConfig
    wandb_project="stairs_climbing"

@configclass
class UnitreeGo2NavigationEnd2EndCNNPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_navigation_stairs_v0"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128, 64], init_std=0.8, encoder_dims=e2e_cnn_config, encoder_type="cnn",
                       encoder_obs_normalize=False, tanh_output=True)
    critic = _enc_critic_shared([128, 128, 64], encoder_dims=e2e_cnn_config, encoder_type="cnn")
    algorithm = NavE2EPPOConfig
    wandb_project="stairs_climbing"
    logger="wandb"


RNDConfig = RslRlRndCfg(
   weight=1.0,
    weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
        final_value=0.0,
        initial_step=200,
        final_step=1000,
    ),
    num_outputs=16,
    predictor_hidden_dims=[256, 256, 128, 64],
    target_hidden_dims=[256, 256, 128, 64],
)

NavRNDPPOConfig = RslRlPpoAlgorithmCfg(
        rnd_cfg=RNDConfig,
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
        # The only user of this config shares the actor's CNN encoder with the critic.
        share_cnn_encoders=True,
    )

@configclass
class UnitreeGo2NavigationEnd2End_CNN_RND_PPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    save_jit = True
    experiment_name = "unitree_go2_navigation_stairs_v0"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _enc_actor([128, 128, 64], init_std=0.8, encoder_dims=e2e_cnn_config, encoder_type="cnn",
                       encoder_obs_normalize=False, tanh_output=True)
    critic = _enc_critic_shared([128, 128, 64], encoder_dims=e2e_cnn_config, encoder_type="cnn")
    disable_rnd_load=True
    algorithm = NavRNDPPOConfig
    wandb_project="stairs_climbing"

@configclass
class UnitreeGo2RVO2CrowdPPORunnerCfg_v0(UnitreeGo2NavigationEnd2EndNoEncoderEnvCfgPPORunnerCfg_v0):
    experiment_name = "unitree_go2_rvo2_crowd_v0"
@configclass
class UnitreeGo2LocomotionVelPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "go2_locomotion_vel"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = NavPPOConfig
    wandb_project="locomotion"
    logger="wandb"


ObstacleAvoidancePPOConfig = RslRlPpoAlgorithmCfg(
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

@configclass
class UnitreeGo2ObstacleAvoidanceNavPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "go2_obstacle_avoidance_navigation"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = ObstacleAvoidancePPOConfig
    wandb_project="obstacle_avoidance_navigation"
    logger="wandb"


# ---------------------------------------------------------------------------
# Temporal-lidar obstacle avoidance (LidarModel)
# ---------------------------------------------------------------------------
# These dimensions MUST match the observation term. Import them from the env
# cfg (single source of truth) so the model and the observation can never
# drift out of sync — a mismatch produces a negative-dimension crash at model
# construction.
# ---------------------------------------------------------------------------
from ..obstacle_avoidance.temporal_lidar_env_cfg import (
    TEMPORAL_LIDAR_HORIZON as TEMPORAL_LIDAR_HORIZON,
    TEMPORAL_LIDAR_FOV_BINS as TEMPORAL_LIDAR_FOV_BINS,
    TEMPORAL_LIDAR_OBS_SIZE as TEMPORAL_LIDAR_OBS_SIZE,
)

# CNN over the (C, H, fov_bins) lidar tensor, where C is 1 or 2 channels depending
# on the observation term's include_validity flag. The first conv omits in_channels
# so LidarModel supplies the inferred channel count.
# Spatial progression (width): 128 → 128 → 64 → 32 → 16 → 2 (adaptive)
# The first layer uses stride 1 to extract local features before any downsampling.
TemporalLidarCNNConfig = [
    # 128
    {"type": "conv",
     "out_channels": 16,
     "kernel_size": (1, 5),
     "stride": (1, 1),
     "padding": (0, 2)},

    # 128 → 64
    {"type": "conv",
     "out_channels": 32,
     "kernel_size": (1, 5),
     "stride": (1, 2),
     "padding": (0, 2)},

    # 64 → 32
    {"type": "conv",
     "out_channels": 32,
     "kernel_size": (1, 3),
     "stride": (1, 2),
     "padding": (0, 1)},

    # 32 → 16
    {"type": "conv",
     "out_channels": 32,
     "kernel_size": (1, 3),
     "stride": (1, 2),
     "padding": (0, 1)},
]
# (32, H, 16) = 512 * H

TemporalLidarHorizonCNNConfig = [
    # Horizon: H Lidar: 128
    {"type": "conv",
     "out_channels": 16,
     "kernel_size": (1, 5),
     "stride": (1, 1),
     "padding": (0, 2)},

    # Horizon: H Lidar: 128 → 64
    {"type": "conv",
     "out_channels": 32,
     "kernel_size": (1, 5),
     "stride": (1, 2),
     "padding": (0, 2)},

    # Horizon: H → H/2 Lidar: 64 → 32
    {"type": "conv",
     "out_channels": 64,
     "kernel_size": (3, 3),
     "stride": (2, 2),
     "padding": (1, 1)},

    # Horizon: H/2 → H/4 Lidar: 32 → 16
    {"type": "conv",
     "out_channels": 64,
     "kernel_size": (3, 3),
     "stride": (2, 2),
     "padding": (1, 1)},

     # Horizon: H/4 Lidar: 16 → 8
    {"type": "conv",
     "out_channels": 64,
     "kernel_size": (1, 3),
     "stride": (1, 2),
     "padding": (0, 1)},
]
# (64, H/4, 8) = 512 * H/4

# Dedicated algorithm config (a copy of ObstacleAvoidancePPOConfig) that shares the lidar CNN encoder between the
# actor and critic LidarModels. Kept separate so the base task's shared config object is never mutated.
ObstacleAvoidanceLidarPPOConfig = RslRlPpoAlgorithmCfg(
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
    share_cnn_encoders=True,
)


def _temporal_lidar_model_cfg(**overrides) -> RslRlLidarModelCfg:
    """Build a LidarModel config for the temporal-lidar obstacle-avoidance task.

    Defaults match the actor; pass ``distribution_cfg=None`` for the (deterministic) critic and the prediction-head
    fields for the actor when the auxiliary head is enabled.
    """
    base = dict(
        hidden_dims=[256, 128, 64],
        activation="elu",
        lidar_obs_size=TEMPORAL_LIDAR_OBS_SIZE,
        lidar_horizon=TEMPORAL_LIDAR_HORIZON,
        lidar_fov_bins=TEMPORAL_LIDAR_FOV_BINS,
        lidar_cnn_dims=TemporalLidarHorizonCNNConfig,
        other_mlp_dims=[16, 16],
    )
    base.update(overrides)
    return RslRlLidarModelCfg(**base)


@configclass
class UnitreeGo2TemporalLidarPPORunnerCfg_v0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "go2_temporal_lidar_obstacle_avoidance"
    empirical_normalization = False
    # actor reads the (noisy) "policy" lidar group; critic reads the privileged noiseless "critic" group.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = _temporal_lidar_model_cfg(
        distribution_cfg=RslRlLidarModelCfg.GaussianDistributionCfg(init_std=0.6, std_type="scalar"),
    )
    critic = _temporal_lidar_model_cfg(distribution_cfg=None)
    algorithm = ObstacleAvoidanceLidarPPOConfig
    wandb_project="obstacle_avoidance_navigation"
    logger="wandb"


# ---------------------------------------------------------------------------
# Temporal-lidar with optional next-frame prediction head (world-model aux task)
# ---------------------------------------------------------------------------
# ConvTranspose1d stack upsampling the deconv input (C0=64, W0=8) back to the
# target arc width (fov_bins). Each {k=4, s=2, p=1} layer exactly doubles width:
# 8 -> 16 -> 32 -> 64 -> 128. Channels taper 64 -> 64 -> 32 -> 16 -> 1.
TemporalLidarPredictionDeconvConfig = [
    {"in_channels": 64, "out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 1},  # 8 -> 16
    {"out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1},                     # 16 -> 32
    {"out_channels": 16, "kernel_size": 4, "stride": 2, "padding": 1},                     # 32 -> 64
    {"out_channels": 1,  "kernel_size": 4, "stride": 2, "padding": 1},                     # 64 -> 128
]

# Dedicated algorithm config (a copy of ObstacleAvoidancePPOConfig) that also enables
# the auxiliary prediction phase. Kept separate so the base task's shared config object
# is never mutated.
ObstacleAvoidancePredictionPPOConfig = RslRlPpoAlgorithmCfg(
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
    # Share the lidar CNN encoder between actor and critic so the auxiliary prediction phase shapes the same
    # encoder the policy uses.
    share_cnn_encoders=True,
    lidar_prediction_cfg=RslRlLidarPredictionCfg(
        weight=0.2,
        learning_rate=1.0e-4,
        num_iterations=1,
        batch_size=4096,
        distance_weight_sigma=0.3,
    ),
)


@configclass
class UnitreeGo2TemporalLidarPredictionPPORunnerCfg_v0(UnitreeGo2TemporalLidarPPORunnerCfg_v0):
    """Temporal-lidar runner with the next-frame lidar prediction head enabled.

    Requires the matching prediction-enabled env cfg (which instantiates the
    ``prediction`` observation group).
    """

    experiment_name = "go2_temporal_lidar_obstacle_avoidance_prediction"
    # Only the actor carries the prediction head; the critic shares its CNN encoder via ``share_cnn_encoders``.
    actor = _temporal_lidar_model_cfg(
        distribution_cfg=RslRlLidarModelCfg.GaussianDistributionCfg(init_std=0.6, std_type="scalar"),
        enable_prediction_head=True,
        pred_cnn_dims=TemporalLidarPredictionDeconvConfig,
        pred_cnn_input_width=8,
        pred_target_channels=1,
    )
    critic = _temporal_lidar_model_cfg(distribution_cfg=None)
    algorithm = ObstacleAvoidancePredictionPPOConfig