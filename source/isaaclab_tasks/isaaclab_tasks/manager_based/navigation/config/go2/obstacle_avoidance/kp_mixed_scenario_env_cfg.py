"""Kp-preprocessed mixed temporal-LiDAR navigation task configuration.

The non-prediction mixed static/pedestrian temporal-LiDAR baseline is inherited
unchanged.  This module replaces only its high-level action term.
"""

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .mixed_scenario_mixins import MixedTemporalLidarObstacleAvoidanceEnvCfg
from .obstacle_avoidance_env_cfg import ActionsCfg, LOW_LEVEL_ENV_CFG, LOW_LEVEL_POLICY_PATH, NUM_LIDAR_RAYS


DYNAMIC_CBF_LIDAR_FOV_DEG = 360.0
DYNAMIC_CBF_LIDAR_RAYS = 2 * NUM_LIDAR_RAYS
DYNAMIC_CBF_VELOCITY_PREDICTOR_PATH = (
    "logs/rsl_rl/ObstacleAvoidance/Navigation/CBF/lidar_velocity_predictor_360_jit.pt"
)


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
            velocity_predictor_jit_path=DYNAMIC_CBF_VELOCITY_PREDICTOR_PATH,
            require_velocity_predictor=True,
            debug_vis=True,
        )
    )


@configclass
class MixedTemporalLidarKpDynamicObstacleCbfObstacleAvoidanceEnvCfg_PLAY(
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY
):
    """PLAY task with body-frame JIT velocities rotated into the world-frame CBF."""

    actions: DynamicCbfKpActionsCfg = DynamicCbfKpActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The 360-degree predictor needs returns behind the robot as well as in
        # the policy's checkpoint-compatible 180-degree observation arc.  Use
        # twice the baseline ray count to preserve its angular resolution; the
        # temporal projection still emits the same 128-bin policy arc.
        self.scene.obstacle_scanner.pattern_cfg.horizontal_fov_range = (
            -DYNAMIC_CBF_LIDAR_FOV_DEG / 2.0,
            DYNAMIC_CBF_LIDAR_FOV_DEG / 2.0,
        )
        self.scene.obstacle_scanner.pattern_cfg.horizontal_res = (
            DYNAMIC_CBF_LIDAR_FOV_DEG / DYNAMIC_CBF_LIDAR_RAYS
        )
        self.held_scan_lidar.full_fan_ray_count = DYNAMIC_CBF_LIDAR_RAYS
        for group_name in ("policy", "critic"):
            group = getattr(self.observations, group_name)
            group.scan_age.params["history_num_rays"] = DYNAMIC_CBF_LIDAR_RAYS
            group.obstacle_scan.params["history_num_rays"] = DYNAMIC_CBF_LIDAR_RAYS
