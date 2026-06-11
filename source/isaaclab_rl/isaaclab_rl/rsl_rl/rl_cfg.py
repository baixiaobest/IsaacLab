# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Model configurations #
#########################

@configclass
class RslRlMLPModelCfg:
    """Configuration for the MLP model."""

    class_name: str = "MLPModel"
    """The model class name. Defaults to MLPModel."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the MLP network."""

    activation: str = MISSING
    """The activation function for the MLP network."""

    obs_normalization: bool = False
    """Whether to normalize the observation for the model. Defaults to False."""

    distribution_cfg: DistributionCfg | None = None
    """The configuration for the output distribution. Defaults to None, in which case no distribution is used."""

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING
        """The distribution class name."""

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"
        """The distribution class name. Default is GaussianDistribution."""

        init_std: float = MISSING
        """The initial standard deviation of the output distribution."""

        std_type: Literal["scalar", "log"] = "scalar"
        """The parameterization type of the output distribution's standard deviation. Default is scalar."""

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"
        """The distribution class name. Default is HeteroscedasticGaussianDistribution."""

    stochastic: bool = MISSING
    """Whether the model output is stochastic.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and set it to None
    for deterministic output or to a valid configuration class, e.g., `GaussianDistributionCfg` for stochastic output.
    """

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the model.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `init_std` field of the distribution configuration to specify the initial noise standard deviation.
    """

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the model. Defaults to scalar.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use the
    `std_type` field of the distribution configuration to specify the type of noise standard deviation.
    """

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Defaults to False.

    For rsl-rl >= 5.0.0, this configuration is is deprecated. Please use `distribution_cfg` instead and use
    the `HeteroscedasticGaussianDistributionCfg` if state-dependent standard deviation is desired.
    """

@configclass
class RslRlEncoderModelCfg(RslRlMLPModelCfg):
    """Configuration for the encoder model (``EncoderModel``).

    The active (1D) observation group is split into ``[main_obs | enc_obs]``, where the last ``encoder`` input
    entries (``enc_obs``) are processed by an MLP or CNN encoder and concatenated with ``main_obs`` before the MLP
    head. To share the encoder between actor and critic, set ``share_cnn_encoders=True`` on the algorithm config and
    use this config for both models. If ``encoder_dims`` is None, no encoder is built (plain MLP over the full
    observation, still honoring ``tanh_output``).
    """

    class_name: str = "EncoderModel"
    """The model class name. Defaults to EncoderModel."""

    encoder_dims: list[int] | list[dict] | None = None
    """Encoder spec: ``[input, *hidden, output]`` for an MLP encoder, or a list of CNN layer-config dicts. None
    disables the encoder."""

    encoder_type: Literal["mlp", "cnn"] = "mlp"
    """The type of encoder network. Default is "mlp"."""

    encoder_obs_normalize: bool = False
    """Whether to per-sample normalize the encoder input before passing it to the encoder. Default is False."""

    tanh_output: bool = False
    """Whether to apply a tanh activation to the model output (the action mean for scalar-std policies)."""


@configclass
class RslRlRNNModelCfg(RslRlMLPModelCfg):
    """Configuration for RNN model."""

    class_name: str = "RNNModel"
    """The model class name. Defaults to RNNModel."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlCNNModelCfg(RslRlMLPModelCfg):
    """Configuration for CNN model."""

    class_name: str = "CNNModel"
    """The model class name. Defaults to CNNModel."""

    @configclass
    class CNNCfg:
        output_channels: tuple[int] | list[int] = MISSING
        """The number of output channels for each convolutional layer for the CNN."""

        kernel_size: int | tuple[int] | list[int] = MISSING
        """The kernel size for the CNN."""

        stride: int | tuple[int] | list[int] = 1
        """The stride for the CNN. Defaults to 1."""

        dilation: int | tuple[int] | list[int] = 1
        """The dilation for the CNN. Defaults to 1."""

        padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
        """The padding for the CNN. Defaults to none."""

        norm: Literal["none", "batch", "layer"] | tuple[str] | list[str] = "none"
        """The normalization for the CNN. Defaults to none."""

        activation: str = MISSING
        """The activation function for the CNN."""

        max_pool: bool | tuple[bool] | list[bool] = False
        """Whether to use max pooling for the CNN. Defaults to False."""

        global_pool: Literal["none", "max", "avg"] = "none"
        """The global pooling for the CNN. Defaults to none."""

        flatten: bool = True
        """Whether to flatten the output of the CNN. Defaults to True."""

    cnn_cfg: CNNCfg = MISSING
    """The configuration for the CNN(s)."""


@configclass
class RslRlLidarModelCfg(RslRlMLPModelCfg):
    """Configuration for the lidar model (``LidarModel``).

    This model expects the active (1D) observation group to be a flat vector whose last ``lidar_obs_size`` entries
    are a flattened ``(C, H, fov_bins)`` temporal-lidar tensor (encoded by a shared 2D CNN) and whose remaining
    entries are proprioceptive observations (encoded by an optional MLP). The CNN encoder can be shared between the
    actor and critic via the algorithm's ``share_cnn_encoders`` option.
    """

    class_name: str = "LidarModel"
    """The model class name. Defaults to LidarModel."""

    lidar_obs_size: int = MISSING
    """Flattened size of the lidar observation: ``C * horizon * fov_bins`` (C inferred from this size)."""

    lidar_horizon: int = MISSING
    """Number of historical lidar timesteps H."""

    lidar_fov_bins: int = MISSING
    """Number of FOV bins per lidar frame."""

    lidar_cnn_dims: list[dict] = MISSING
    """CNN layer configs (list of dicts) applied to the ``(C, H, fov_bins)`` lidar input."""

    other_mlp_dims: list[int] | None = None
    """Hidden + output dims for the proprioceptive ("other") MLP encoder, e.g. ``[128, 64]``. If None, the other
    observations are passed through unchanged."""

    enable_prediction_head: bool = False
    """Whether to build the optional next-frame lidar prediction head. Default is False. Typically set on the actor
    only; the critic shares the encoder via ``share_cnn_encoders``."""

    pred_cnn_dims: list[dict] | None = None
    """ConvTranspose1d layer configs that upsample the deconv input back to ``fov_bins``. The first layer must
    specify ``in_channels``. Only used when the head is enabled."""

    pred_cnn_input_width: int = 8
    """Initial width fed into the deconv stack (reshaped from the prediction MLP output)."""

    pred_target_channels: int = 1
    """Number of channels in the prediction target (1 = distance only)."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlLidarPredictionCfg:
    """Configuration for the auxiliary next-frame lidar prediction training phase.

    When set on :class:`RslRlPpoAlgorithmCfg`, after each PPO update a separate auxiliary
    phase trains the shared lidar encoder + prediction head to predict the next lidar
    frame (see ``LidarModel.predict_next``). Requires the actor model to be a
    :class:`RslRlLidarModelCfg` with ``enable_prediction_head=True`` and the environment to
    emit a ``prediction`` observation group.
    """

    weight: float = 1.0
    """Scaling factor applied to the prediction MSE loss."""

    learning_rate: float = 1.0e-3
    """Learning rate of the dedicated prediction optimizer."""

    num_iterations: int = 4
    """Number of auxiliary minibatch updates performed per PPO iteration."""

    batch_size: int = 4096
    """Number of (obs, target) pairs per auxiliary minibatch."""

    distance_weight_sigma: float | None = None
    """Decay factor for near-field loss weighting.

    When set, each element of the prediction MSE is weighted by ``1 - tanh(d / sigma)``,
    where ``d`` is the ground-truth (normalized) target distance. This emphasizes accurate
    prediction of nearby obstacles (small ``d``) over distant/free-space ones (large ``d``).
    Smaller ``sigma`` sharpens the emphasis on the near field. If ``None``, the prediction
    loss is unweighted (uniform MSE).
    """


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Defaults to PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use. Defaults to adam."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Defaults to False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False."""

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Defaults to None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Defaults to None, in which case symmetry is not used."""

    lidar_prediction_cfg: RslRlLidarPredictionCfg | None = None
    """The configuration for the auxiliary next-frame lidar prediction head. Default is
    None, in which case the prediction head is not trained.
    """


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Defaults to 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Defaults to cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """This parameter is deprecated and will be removed in the future.

    For rsl-rl < 4.0.0, use `actor_obs_normalization` and `critic_obs_normalization` of the policy instead.
    For rsl-rl >= 4.0.0, use `obs_normalization` of the model instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "actor": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the actor will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    check_for_nan: bool = True
    """Whether to check for NaN values coming from the environment."""

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Defaults to empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Defaults to tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Defaults to "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Defaults to "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Defaults to False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Defaults to ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Defaults to ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    other_dir: str = ""
    """The extra directory to load the checkpoint from. Default is empty string."""

    disable_rnd_load: bool = False
    """Whether to disable loading the RND model from checkpoint. Default is False."""

@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    actor: RslRlMLPModelCfg = MISSING
    """The actor configuration."""

    critic: RslRlMLPModelCfg = MISSING
    """The critic configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration.

    For rsl-rl >= 4.0.0, this configuration is is deprecated. Please use `actor` and `critic` model configurations
    instead.
    """


#############################
# Deprecated configurations #
#############################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead.
    """

    class_name: str = "ActorCritic"
    """The policy class name. Defaults to ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Defaults to scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Defaults to False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers.

    For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlRNNModelCfg` instead.
    """

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Defaults to ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""
