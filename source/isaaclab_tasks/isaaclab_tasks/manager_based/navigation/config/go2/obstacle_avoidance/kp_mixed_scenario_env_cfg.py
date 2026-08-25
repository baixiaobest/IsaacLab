"""Kp-preprocessed mixed temporal-LiDAR navigation task configuration.

The non-prediction mixed static/pedestrian temporal-LiDAR baseline is inherited
unchanged.  This module replaces only its high-level action term.
"""

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from .mixed_scenario_mixins import MixedTemporalLidarObstacleAvoidanceEnvCfg
from .obstacle_avoidance_env_cfg import LOW_LEVEL_ENV_CFG, LOW_LEVEL_POLICY_PATH


_BASE_ACTION_CFG = MixedTemporalLidarObstacleAvoidanceEnvCfg().actions.pre_trained_policy_action


@configclass
class MixedTemporalLidarKpObstacleAvoidanceEnvCfg(MixedTemporalLidarObstacleAvoidanceEnvCfg):
    """Mixed temporal-LiDAR task with bounded Kp velocity preprocessing only.

    Terrain/crowd setup, observations, rewards, curricula, terminations,
    timing, and the locomotion policy are inherited from the non-prediction
    mixed temporal-LiDAR baseline.
    """

    actions: nav_mdp.KpPreTrainedPolicyActionCfg = nav_mdp.KpPreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_actions=_BASE_ACTION_CFG.low_level_actions,
        low_level_observations=_BASE_ACTION_CFG.low_level_observations,
        action_scales=(1.0, 1.0, 1.0),
        kp=(5.0, 5.0),
        acceleration_limits=((-3.0, 3.0), (-3.0, 3.0)),
        velocity_limits=((-1.0, 1.0), (-1.0, 1.0)),
        debug_vis=True,
    )


@configclass
class MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY(MixedTemporalLidarKpObstacleAvoidanceEnvCfg):
    """Sixteen-environment play variant of the Kp-preprocessed mixed task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
