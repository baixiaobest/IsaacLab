"""Regression tests for the mixed-scene 128-ray collision-penalty treatment."""

import gymnasium as gym

from isaaclab_tasks.manager_based.navigation.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2TemporalLidarPPORunnerCfg_v0,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    MixedTemporalLidarObstacleAvoidanceEnvCfg,
    MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.temporal_lidar_env_cfg import (
    TEMPORAL_LIDAR_FOV_BINS,
    TEMPORAL_LIDAR_HORIZON,
    TEMPORAL_LIDAR_OBS_SIZE,
)


def _assert_treatment_env_cfg(cfg, expected_num_envs: int) -> None:
    assert cfg.scene.num_envs == expected_num_envs
    assert cfg.held_scan_lidar.full_fan_ray_count == 128
    assert round(180.0 / cfg.scene.obstacle_scanner.pattern_cfg.horizontal_res) + 1 == 128
    assert cfg.rewards.pedestrian_collision_penalty.weight == -600.0

    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name)
        assert group.scan_age.params["history_num_rays"] == 128
        assert group.obstacle_scan.params["history_num_rays"] == 128
        assert group.obstacle_scan.params["num_bins"] == 256


def test_train_and_play_use_the_same_treatment() -> None:
    _assert_treatment_env_cfg(MixedTemporalLidarObstacleAvoidanceEnvCfg(), expected_num_envs=2000)
    _assert_treatment_env_cfg(MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY(), expected_num_envs=16)


def test_train_and_play_registrations_use_corresponding_treatment_configs() -> None:
    task_prefix = "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2"
    train_spec = gym.spec(f"{task_prefix}-v0")
    play_spec = gym.spec(f"{task_prefix}-Play-v0")

    assert train_spec.kwargs["env_cfg_entry_point"].endswith(":MixedTemporalLidarObstacleAvoidanceEnvCfg")
    assert play_spec.kwargs["env_cfg_entry_point"].endswith(":MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY")
    expected_runner = ":UnitreeGo2TemporalLidarPPORunnerCfg_v0"
    assert train_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(expected_runner)
    assert play_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(expected_runner)


def test_cnn_contract_is_unchanged() -> None:
    runner = UnitreeGo2TemporalLidarPPORunnerCfg_v0()

    assert runner.actor.lidar_obs_size == TEMPORAL_LIDAR_OBS_SIZE
    assert runner.actor.lidar_horizon == TEMPORAL_LIDAR_HORIZON
    assert runner.actor.lidar_fov_bins == TEMPORAL_LIDAR_FOV_BINS
    assert runner.actor.lidar_cnn_dims == runner.critic.lidar_cnn_dims
