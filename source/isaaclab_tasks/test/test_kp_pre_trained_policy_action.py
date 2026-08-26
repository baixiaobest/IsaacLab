"""Unit tests for Kp navigation-velocity preprocessing."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.kp_mixed_scenario_env_cfg import (
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg,
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    MixedTemporalLidarObstacleAvoidanceEnvCfg,
    MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.navigation.mdp.kp_pre_trained_policy_action import (
    KpPreTrainedPolicyAction,
    compute_kp_velocity_command,
)
from isaaclab_tasks.utils import load_cfg_from_registry


KP_TASK_ID = "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Obstacle-Avoidance-Unitree-Go2-v0"
KP_PLAY_TASK_ID = "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Obstacle-Avoidance-Unitree-Go2-Play-v0"


def _limits(kp_value: float = 5.0):
    return (
        torch.tensor((kp_value, kp_value)),
        torch.tensor((-5.0, -5.0)),
        torch.tensor((5.0, 5.0)),
        torch.tensor((-1.3, -1.3)),
        torch.tensor((1.3, 1.3)),
    )


def test_kp_velocity_command_integrates_unclipped_acceleration() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[0.3, -0.2]]), torch.tensor([[0.1, -0.1]]), 0.08, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.allclose(acceleration, torch.tensor([[1.0, -0.5]]))
    assert torch.allclose(command, torch.tensor([[0.18, -0.14]]))


def test_kp_velocity_command_respects_acceleration_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[1.0, -1.0]]), torch.zeros(1, 2), 0.08, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.equal(acceleration, torch.tensor([[5.0, -5.0]]))
    assert torch.allclose(command, torch.tensor([[0.4, -0.4]]))


def test_kp_velocity_command_respects_velocity_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[2.0, -2.0]]), torch.tensor([[1.2, -1.2]]), 0.08, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.equal(acceleration, torch.tensor([[4.0, -4.0]]))
    assert torch.equal(command, torch.tensor([[1.3, -1.3]]))


def test_kp8_velocity_command_preserves_acceleration_and_velocity_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits(8.0)
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[2.0, -2.0]]), torch.tensor([[1.2, -1.2]]), 0.08, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )

    assert torch.equal(acceleration, torch.tensor([[5.0, -5.0]]))
    assert torch.equal(command, torch.tensor([[1.3, -1.3]]))


def test_action_term_forwards_integrated_velocity_and_unchanged_yaw() -> None:
    """The private low-level command differs only in planar Kp preprocessing."""
    term = object.__new__(KpPreTrainedPolicyAction)
    term.cfg = SimpleNamespace(action_scales=(1.0, 1.0, 1.0))
    term._action_scales = torch.ones(3)
    term._raw_actions = torch.zeros(1, 3)
    term._processed_actions = torch.zeros(1, 3)
    term._nominal_acceleration = torch.zeros(1, 2)
    term._kp, term._acceleration_lower, term._acceleration_upper, term._velocity_lower, term._velocity_upper = _limits()
    term._env = SimpleNamespace(step_dt=0.08)
    term.robot = SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0]])))

    term.process_actions(torch.tensor([[1.0, -1.0, 0.7]]))

    assert torch.equal(term.nominal_acceleration, torch.tensor([[5.0, -5.0]]))
    assert torch.allclose(term.processed_actions, torch.tensor([[0.4, -0.4, 0.7]]))


def test_kp_task_preserves_baseline_temporal_lidar_and_action_dimensions() -> None:
    baseline = MixedTemporalLidarObstacleAvoidanceEnvCfg()
    kp_task = MixedTemporalLidarKpObstacleAvoidanceEnvCfg()

    assert type(kp_task.observations) is type(baseline.observations)
    assert set(kp_task.actions.__dict__) == set(baseline.actions.__dict__) == {"pre_trained_policy_action"}
    assert len(kp_task.actions.pre_trained_policy_action.action_scales) == len(
        baseline.actions.pre_trained_policy_action.action_scales
    )
    assert not hasattr(kp_task.observations, "prediction")
    assert kp_task.actions.pre_trained_policy_action.kp == (8.0, 8.0)
    assert kp_task.actions.pre_trained_policy_action.acceleration_limits == ((-5.0, 5.0), (-5.0, 5.0))
    assert kp_task.actions.pre_trained_policy_action.velocity_limits == ((-1.5, 1.5), (-1.5, 1.5))
    assert kp_task.sim.dt * kp_task.decimation == 0.08


def test_kp_task_resolves_to_updated_config() -> None:
    baseline = MixedTemporalLidarObstacleAvoidanceEnvCfg()
    cfg = load_cfg_from_registry(KP_TASK_ID, "env_cfg_entry_point")

    assert isinstance(cfg, MixedTemporalLidarKpObstacleAvoidanceEnvCfg)
    assert cfg.actions.pre_trained_policy_action.kp == (8.0, 8.0)
    assert cfg.actions.pre_trained_policy_action.acceleration_limits == ((-5.0, 5.0), (-5.0, 5.0))
    assert cfg.actions.pre_trained_policy_action.velocity_limits == ((-1.5, 1.5), (-1.5, 1.5))
    assert type(cfg.observations) is type(baseline.observations)
    assert len(cfg.actions.pre_trained_policy_action.action_scales) == len(
        baseline.actions.pre_trained_policy_action.action_scales
    ) == 3


def test_kp_play_task_resolves_to_updated_config() -> None:
    cfg = load_cfg_from_registry(KP_PLAY_TASK_ID, "env_cfg_entry_point")

    assert isinstance(cfg, MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY)
    assert cfg.actions.pre_trained_policy_action.acceleration_limits == ((-5.0, 5.0), (-5.0, 5.0))
    assert cfg.actions.pre_trained_policy_action.velocity_limits == ((-1.5, 1.5), (-1.5, 1.5))


def test_kp_play_task_matches_baseline_temporal_lidar_play_setup() -> None:
    """The Kp play task changes only the high-level action term."""
    baseline = MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY()
    kp_task = MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY()

    assert kp_task.scene.num_envs == baseline.scene.num_envs == 16
    assert type(kp_task.observations) is type(baseline.observations)
    assert kp_task.held_scan_lidar_enabled == baseline.held_scan_lidar_enabled
    assert kp_task.actions.pre_trained_policy_action.action_scales == baseline.actions.pre_trained_policy_action.action_scales
    assert kp_task.actions.pre_trained_policy_action.acceleration_limits == ((-5.0, 5.0), (-5.0, 5.0))
    assert kp_task.actions.pre_trained_policy_action.velocity_limits == ((-1.5, 1.5), (-1.5, 1.5))
