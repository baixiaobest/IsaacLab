"""Temporal-lidar variant of the obstacle-avoidance navigation environment.

Inherits everything from ObstacleAvoidanceEnvCfg and replaces only the lidar
observation term with TemporalLidarScan, which stacks H historical scans into a
world-aligned 360° bin grid and returns a FOV-centred arc to the policy.
"""

from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .obstacle_avoidance_env_cfg import (
    LIDAR_MAX_DISTANCE,
    ObstacleAvoidanceEnvCfg,
    ObservationsCfg,
)

# ---------------------------------------------------------------------------
# Temporal lidar hyper-parameters
# ---------------------------------------------------------------------------

TEMPORAL_LIDAR_HORIZON = 4       # H – number of historical timesteps
TEMPORAL_LIDAR_NUM_BINS = 256    # B – total 360° world-aligned bins
TEMPORAL_LIDAR_FOV_DEG = 180.0   # arc returned to the policy
TEMPORAL_LIDAR_POS_NOISE_STD = 0.05   # odometry noise (metres)
TEMPORAL_LIDAR_INCLUDE_VALIDITY = True  # emit the per-bin validity channel alongside distance
TEMPORAL_LIDAR_ENABLE_PREDICTION = False  # emit the next-frame prediction target group (world-model head)

# Derived obs size: C channels × H × fov_bins, where C = 2 with validity else 1
TEMPORAL_LIDAR_FOV_BINS = int(round(TEMPORAL_LIDAR_NUM_BINS * TEMPORAL_LIDAR_FOV_DEG / 360.0))
# Keep even
if TEMPORAL_LIDAR_FOV_BINS % 2 != 0:
    TEMPORAL_LIDAR_FOV_BINS -= 1

TEMPORAL_LIDAR_CHANNELS = 2 if TEMPORAL_LIDAR_INCLUDE_VALIDITY else 1
TEMPORAL_LIDAR_OBS_SIZE = TEMPORAL_LIDAR_CHANNELS * TEMPORAL_LIDAR_HORIZON * TEMPORAL_LIDAR_FOV_BINS

# Prediction target is a single distance-only frame (1 channel × fov_bins).
TEMPORAL_LIDAR_PRED_TARGET_SIZE = TEMPORAL_LIDAR_FOV_BINS


# ---------------------------------------------------------------------------
# Observation overrides
# ---------------------------------------------------------------------------

@configclass
class TemporalLidarObservationsCfg(ObservationsCfg):
    """Replace the static lidar scan with the temporal version."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        obstacle_scan = ObsTerm(
            func=mdp.TemporalLidarScan,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "horizon": TEMPORAL_LIDAR_HORIZON,
                "num_bins": TEMPORAL_LIDAR_NUM_BINS,
                "fov_degrees": TEMPORAL_LIDAR_FOV_DEG,
                "max_distance": LIDAR_MAX_DISTANCE,
                "pos_noise_std": TEMPORAL_LIDAR_POS_NOISE_STD,
                "include_validity": TEMPORAL_LIDAR_INCLUDE_VALIDITY,
                # Owns the shared LidarHistoryStore for this sensor: the policy group
                # computes first, so it pushes the new scan before critic/prediction read it.
                "owns_history": True,
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """Critic receives noiseless temporal lidar projection (privileged obs)."""

        obstacle_scan = ObsTerm(
            func=mdp.TemporalLidarScan,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "horizon": TEMPORAL_LIDAR_HORIZON,
                "num_bins": TEMPORAL_LIDAR_NUM_BINS,
                "fov_degrees": TEMPORAL_LIDAR_FOV_DEG,
                "max_distance": LIDAR_MAX_DISTANCE,
                "pos_noise_std": 0.0,
                "include_validity": TEMPORAL_LIDAR_INCLUDE_VALIDITY,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PredictionCfg(ObsGroup):
        """Self-supervised next-frame prediction target (world-model head).

        Separate group (not concatenated into policy/critic). It is exposed to the
        algorithm via ``infos["observations"]["prediction"]`` and consumed only by the
        optional lidar prediction head during the auxiliary training phase.
        """

        target = ObsTerm(
            func=mdp.TemporalLidarPredictionTarget,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "num_bins": TEMPORAL_LIDAR_NUM_BINS,
                "fov_degrees": TEMPORAL_LIDAR_FOV_DEG,
                "max_distance": LIDAR_MAX_DISTANCE,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    # Only instantiate the prediction group when enabled — disabled runs pay zero extra
    # compute. The observation manager skips non-ObservationGroupCfg (None) members.
    prediction: PredictionCfg | None = PredictionCfg() if TEMPORAL_LIDAR_ENABLE_PREDICTION else None


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

@configclass
class TemporalLidarObstacleAvoidanceEnvCfg(ObstacleAvoidanceEnvCfg):
    """Obstacle-avoidance env with temporal lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()


@configclass
class TemporalLidarPredictionObservationsCfg(TemporalLidarObservationsCfg):
    """Temporal-lidar observations with the next-frame prediction group always active.

    Use this (instead of relying on the module-level ``TEMPORAL_LIDAR_ENABLE_PREDICTION``
    flag) to pair with the prediction-enabled runner cfg.
    """

    prediction: TemporalLidarObservationsCfg.PredictionCfg = TemporalLidarObservationsCfg.PredictionCfg()


@configclass
class TemporalLidarPredictionObstacleAvoidanceEnvCfg(TemporalLidarObstacleAvoidanceEnvCfg):
    """Temporal-lidar env that also emits the next-frame prediction target group."""

    observations: TemporalLidarPredictionObservationsCfg = TemporalLidarPredictionObservationsCfg()


@configclass
class TemporalLidarObstacleAvoidanceEnvCfg_PLAY(TemporalLidarObstacleAvoidanceEnvCfg):
    """Play variant with fewer envs and no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 10
        self.observations.policy.enable_corruption = False
        self.actions.pre_trained_policy_action.debug_vis = True
