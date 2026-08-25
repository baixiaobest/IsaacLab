"""Unit tests for Kp navigation-velocity preprocessing."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.kp_mixed_scenario_env_cfg import (
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    MixedTemporalLidarObstacleAvoidanceEnvCfg,
)
from isaaclab_tasks.manager_based.navigation.mdp.kp_pre_trained_policy_action import (
    KpPreTrainedPolicyAction,
    compute_kp_velocity_command,
)


def _limits():
    return (
        torch.tensor((5.0, 5.0)),
        torch.tensor((-3.0, -3.0)),
        torch.tensor((3.0, 3.0)),
        torch.tensor((-1.0, -1.0)),
        torch.tensor((1.0, 1.0)),
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
    assert torch.equal(acceleration, torch.tensor([[3.0, -3.0]]))
    assert torch.allclose(command, torch.tensor([[0.24, -0.24]]))


def test_kp_velocity_command_respects_velocity_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[1.0, -1.0]]), torch.tensor([[0.95, -0.95]]), 0.08, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.equal(acceleration, torch.tensor([[0.25, -0.25]]))
    assert torch.equal(command, torch.tensor([[1.0, -1.0]]))


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

    assert torch.equal(term.nominal_acceleration, torch.tensor([[3.0, -3.0]]))
    assert torch.allclose(term.processed_actions, torch.tensor([[0.24, -0.24, 0.7]]))


def test_kp_task_preserves_baseline_temporal_lidar_and_action_dimensions() -> None:
    baseline = MixedTemporalLidarObstacleAvoidanceEnvCfg()
    kp_task = MixedTemporalLidarKpObstacleAvoidanceEnvCfg()

    assert type(kp_task.observations) is type(baseline.observations)
    assert len(kp_task.actions.pre_trained_policy_action.action_scales) == len(
        baseline.actions.pre_trained_policy_action.action_scales
    )
    assert not hasattr(kp_task.observations, "prediction")
    assert kp_task.actions.pre_trained_policy_action.kp == (5.0, 5.0)
    assert kp_task.actions.pre_trained_policy_action.acceleration_limits == ((-3.0, 3.0), (-3.0, 3.0))
    assert kp_task.actions.pre_trained_policy_action.velocity_limits == ((-1.0, 1.0), (-1.0, 1.0))
    assert kp_task.sim.dt * kp_task.decimation == 0.08
