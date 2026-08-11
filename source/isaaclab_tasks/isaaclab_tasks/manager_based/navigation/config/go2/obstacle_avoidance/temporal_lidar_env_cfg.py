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
    LIDAR_FOV_DEG,
    LIDAR_MAX_DISTANCE,
    NUM_LIDAR_RAYS,
    ObstacleAvoidanceEnvCfg,
    ObservationsCfg,
)
from .observation_modifiers import policy_base_lin_vel_modifiers, policy_imu_ang_vel_modifiers
from .two_cloud_lidar_env import TwoCloudLidarCfg

# ---------------------------------------------------------------------------
# Temporal lidar hyper-parameters
# ---------------------------------------------------------------------------

TEMPORAL_LIDAR_HORIZON = 4       # H – number of historical timesteps
TEMPORAL_LIDAR_NUM_BINS = 256    # B – total 360° world-aligned bins
TEMPORAL_LIDAR_FOV_DEG = 180.0   # arc returned to the policy
TEMPORAL_LIDAR_ACTOR_RAYS = 128  # completed actor cloud after 256→128 raw rebin
TEMPORAL_LIDAR_INCLUDE_VALIDITY = True  # emit the per-bin validity channel alongside distance
TEMPORAL_LIDAR_ACTOR_HISTORY_KEY = "actor_completed"
TEMPORAL_LIDAR_CRITIC_HISTORY_KEY = "critic_current"
TEMPORAL_LIDAR_COLLECTOR_NAME = "_two_cloud_lidar_collector"
TEMPORAL_LIDAR_SCAN_AGE_MAX_S = 0.25

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
    class PolicyCfg(ObsGroup):
        # Keep this immediately before obstacle_scan: LidarModel requires the lidar
        # tensor to occupy the tail of the concatenated observation vector.
        #
        # This is deliberately not a subclass of ObservationsCfg.PolicyCfg:
        # dataclass overrides retain their base field's original slot, which would
        # otherwise append scan_age after the lidar tensor.
        pose_2d_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_2d_command"})
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            modifiers=policy_base_lin_vel_modifiers(),
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            modifiers=policy_imu_ang_vel_modifiers(),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)

        scan_age = ObsTerm(
            func=mdp.temporal_lidar_scan_age,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "max_distance": LIDAR_MAX_DISTANCE,
                "history_key": TEMPORAL_LIDAR_ACTOR_HISTORY_KEY,
                "collector_name": TEMPORAL_LIDAR_COLLECTOR_NAME,
                "history_num_rays": TEMPORAL_LIDAR_ACTOR_RAYS,
                "history_horizon": TEMPORAL_LIDAR_HORIZON,
                "max_age_s": TEMPORAL_LIDAR_SCAN_AGE_MAX_S,
            },
        )

        obstacle_scan = ObsTerm(
            func=mdp.TemporalLidarScan,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "horizon": TEMPORAL_LIDAR_HORIZON,
                "num_bins": TEMPORAL_LIDAR_NUM_BINS,
                "fov_degrees": TEMPORAL_LIDAR_FOV_DEG,
                "max_distance": LIDAR_MAX_DISTANCE,
                # Actor errors are applied by TwoCloudLidarCollector.  Do not add
                # projection noise here: it would corrupt the carefully modelled,
                # correlated scan geometry and blur binary validity semantics.
                "pos_noise_std": 0.0,
                "include_validity": TEMPORAL_LIDAR_INCLUDE_VALIDITY,
                "history_key": TEMPORAL_LIDAR_ACTOR_HISTORY_KEY,
                "history_num_rays": TEMPORAL_LIDAR_ACTOR_RAYS,
                "collector_name": TEMPORAL_LIDAR_COLLECTOR_NAME,
                # The preceding scan_age term owns the same completed-scan update;
                # this stays idempotent and keeps the lidar tensor last.
                "owns_history": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """Critic owns an independent, current, ideal temporal-lidar history."""

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
                "history_key": TEMPORAL_LIDAR_CRITIC_HISTORY_KEY,
                "owns_history": True,
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
                "history_key": TEMPORAL_LIDAR_ACTOR_HISTORY_KEY,
                "history_num_rays": TEMPORAL_LIDAR_ACTOR_RAYS,
                "history_horizon": TEMPORAL_LIDAR_HORIZON,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

    @configclass
    class PredictionMaskCfg(ObsGroup):
        """Completion-event mask consumed by the auxiliary prediction minibatcher."""

        event = ObsTerm(
            func=mdp.temporal_lidar_scan_event,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "max_distance": LIDAR_MAX_DISTANCE,
                "history_key": TEMPORAL_LIDAR_ACTOR_HISTORY_KEY,
                "history_num_rays": TEMPORAL_LIDAR_ACTOR_RAYS,
                "history_horizon": TEMPORAL_LIDAR_HORIZON,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

@configclass
class TemporalLidarObstacleAvoidanceEnvCfg(ObstacleAvoidanceEnvCfg):
    """Obstacle-avoidance env with temporal lidar observations."""

    observations: TemporalLidarObservationsCfg = TemporalLidarObservationsCfg()
    two_cloud_lidar_enabled: bool = True
    two_cloud_lidar: TwoCloudLidarCfg = TwoCloudLidarCfg()

    def __post_init__(self):
        super().__post_init__()
        # The collector forces this sensor only on 65 ms raw-cloud boundaries.
        # A zero update period leaves it outdated between those boundaries, so the
        # critic's current-scan observation recomputes it at every policy step.
        self.scene.obstacle_scanner.update_period = 0.0
        # lidar_pattern includes both FOV endpoints.  Use 255 intervals to obtain
        # the planned 256 full-fan rays (rather than the base task's 257 rays).
        self.scene.obstacle_scanner.pattern_cfg.horizontal_res = LIDAR_FOV_DEG / (NUM_LIDAR_RAYS - 1)
        # With lazy sensor updates this prevents debug visualization from forcing a
        # RayCaster recompute every 5 ms.  The collector explicitly reads .data only
        # at a 65 ms raw-cloud boundary; the ideal critic reads it at policy rate.
        self.scene.obstacle_scanner.debug_vis = False


@configclass
class TemporalLidarPredictionObservationsCfg(TemporalLidarObservationsCfg):
    """Temporal-lidar observations with the next-frame prediction group always active.

    Use this (instead of relying on the module-level ``TEMPORAL_LIDAR_ENABLE_PREDICTION``
    flag) to pair with the prediction-enabled runner cfg.
    """

    prediction: TemporalLidarObservationsCfg.PredictionCfg = TemporalLidarObservationsCfg.PredictionCfg()
    prediction_mask: TemporalLidarObservationsCfg.PredictionMaskCfg = TemporalLidarObservationsCfg.PredictionMaskCfg()


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
