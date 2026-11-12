"""Kp-preprocessed mixed temporal-LiDAR navigation task configuration.

The non-prediction mixed static/pedestrian temporal-LiDAR baseline is inherited
unchanged.  This module replaces only its high-level action term.
"""

from copy import deepcopy

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .mixed_scenario_mixins import MixedSceneCfg, MixedTemporalLidarObstacleAvoidanceEnvCfg
from .obstacle_avoidance_env_cfg import ActionsCfg, LIDAR_MAX_DISTANCE, LOW_LEVEL_ENV_CFG, LOW_LEVEL_POLICY_PATH
from .temporal_lidar_env_cfg import PredictorLidarObservationsCfg


@configclass
class KpActionsCfg(ActionsCfg):
    """Baseline navigation action container with only its action term replaced."""

    pre_trained_policy_action: nav_mdp.KpPreTrainedPolicyActionCfg = nav_mdp.KpPreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        action_scales=(1.0, 1.0, 1.0),
        kp=(8.0, 8.0),
        acceleration_limits=((-5.0, 5.0), (-5.0, 5.0)),
        velocity_limits=((-1.5, 1.5), (-1.5, 1.5)),
        tracking_tau_s=0.30,
        debug_vis=True,
    )


@configclass
class MixedTemporalLidarKpObstacleAvoidanceEnvCfg(MixedTemporalLidarObstacleAvoidanceEnvCfg):
    """Mixed temporal-LiDAR task with bounded Kp velocity preprocessing only.

    Terrain/crowd setup, observations, rewards, curricula, terminations,
    timing, and the locomotion policy are inherited from the non-prediction
    mixed temporal-LiDAR baseline.
    """

    actions: KpActionsCfg = KpActionsCfg()


@configclass
class MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY(MixedTemporalLidarKpObstacleAvoidanceEnvCfg):
    """Sixteen-environment play variant of the Kp-preprocessed mixed task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.curriculum.lidar_density = None
        self.held_scan_lidar.density_curriculum_enabled = False
        self.held_scan_lidar.target_coverage = None
        # Evaluation uses a longer, matching command horizon than training.
        # The CBF PLAY variants inherit this configuration unchanged.
        self.episode_length_s = 20.0
        self.commands.pose_2d_command.resampling_time_range = (20.1, 20.1)


@configclass
class CbfKpActionsCfg(ActionsCfg):
    """PLAY-only Kp action container with the shared zero-velocity CBF-QP filter."""

    pre_trained_policy_action: nav_mdp.DynamicObstacleCbfPreTrainedPolicyActionCfg = (
        nav_mdp.DynamicObstacleCbfPreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=LOW_LEVEL_POLICY_PATH,
            low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
            low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
            low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
            action_scales=(1.0, 1.0, 1.0),
            kp=(8.0, 8.0),
            acceleration_limits=((-5.0, 5.0), (-5.0, 5.0)),
            velocity_limits=((-1.5, 1.5), (-1.5, 1.5)),
            d_margin=0.70,
            d_cbf_active=5.0,
            gamma1=2.0,
            gamma2=2.0,
            tracking_tau_s=0.30,
            slack_penalty=1000.0,
            max_lidar_points=64,
            debug_vis=True,
        )
    )


@configclass
class MixedTemporalLidarKpStaticObstacleCbfObstacleAvoidanceEnvCfg_PLAY(
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY
):
    """PLAY-only Kp navigation evaluation with a static-obstacle CBF-QP.

    The temporal-LiDAR observations and high-level action interface remain
    checkpoint-compatible with :class:`MixedTemporalLidarKpObstacleAvoidanceEnvCfg`.
    Only the private command supplied to the locomotion policy is filtered.
    """

    actions: CbfKpActionsCfg = CbfKpActionsCfg()


@configclass
class Cbf360SceneCfg(MixedSceneCfg):
    """Mixed scene with an additional current-time, all-around CBF scanner."""

    cbf_scanner: MultiMeshRayCasterCfg = deepcopy(MixedSceneCfg(num_envs=16, env_spacing=2.5).obstacle_scanner)

    def __post_init__(self):
        self.cbf_scanner.pattern_cfg = patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / 256.0,
        )
        self.cbf_scanner.max_distance = LIDAR_MAX_DISTANCE
        self.cbf_scanner.update_period = 0.0
        self.cbf_scanner.debug_vis = False


@configclass
class Cbf360KpActionsCfg(CbfKpActionsCfg):
    """Static CBF using only the dedicated 360-degree sensor for obstacles."""

    def __post_init__(self):
        self.pre_trained_policy_action.scan_source = "sensor"
        self.pre_trained_policy_action.cbf_sensor_name = "cbf_scanner"


@configclass
class MixedTemporalLidarKpStaticObstacleCbf360LidarObstacleAvoidanceEnvCfg_PLAY(
    MixedTemporalLidarKpStaticObstacleCbfObstacleAvoidanceEnvCfg_PLAY
):
    """Static CBF Play variant with a dedicated 256-ray, 360-degree scanner."""

    scene: Cbf360SceneCfg = Cbf360SceneCfg(num_envs=16, env_spacing=2.5)
    actions: Cbf360KpActionsCfg = Cbf360KpActionsCfg()


@configclass
class DynamicCbfKpActionsCfg(ActionsCfg):
    """PLAY-only Kp action container with learned body-frame point velocities."""

    pre_trained_policy_action: nav_mdp.DynamicObstacleCbfPreTrainedPolicyActionCfg = (
        nav_mdp.DynamicObstacleCbfPreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=LOW_LEVEL_POLICY_PATH,
            low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
            low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
            low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
            action_scales=(1.0, 1.0, 1.0),
            kp=(8.0, 8.0),
            acceleration_limits=((-5.0, 5.0), (-5.0, 5.0)),
            velocity_limits=((-1.5, 1.5), (-1.5, 1.5)),
            d_margin=0.70,
            d_cbf_active=5.0,
            gamma1=2.0,
            gamma2=2.0,
            tracking_tau_s=0.30,
            slack_penalty=1000.0,
            max_lidar_points=64,
            velocity_predictor_jit_path="logs/rsl_rl/ObstacleAvoidance/Navigation/CBF/lidar_velocity_predictor_4_frames_jit.pt",
            require_velocity_predictor=True,
            scan_type="policy",
            predictor_observation_group="predictor",
            debug_vis=True,
        )
    )


@configclass
class MixedTemporalLidarKpDynamicObstacleCbfObstacleAvoidanceEnvCfg_PLAY(
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY
):
    """PLAY task with body-frame JIT velocities rotated into the world-frame CBF."""

    actions: DynamicCbfKpActionsCfg = DynamicCbfKpActionsCfg()
    observations: PredictorLidarObservationsCfg = PredictorLidarObservationsCfg()
