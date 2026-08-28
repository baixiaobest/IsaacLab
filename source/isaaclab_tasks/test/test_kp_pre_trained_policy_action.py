"""Unit tests for Kp navigation-velocity preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.kp_mixed_scenario_env_cfg import (
    MixedTemporalLidarKpStaticObstacleCbfObstacleAvoidanceEnvCfg_PLAY,
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg,
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    MixedTemporalLidarObstacleAvoidanceEnvCfg,
    MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.navigation.mdp.cbf_pre_trained_policy_action import (
    StaticObstacleCbfPreTrainedPolicyActionCfg,
    _OsqpSolveStats,
    StaticObstacleCbfPreTrainedPolicyAction,
    effective_zoh_acceleration_bounds,
    velocity_command_from_average_acceleration,
    zoh_average_acceleration_gain,
)
from isaaclab_tasks.manager_based.navigation.mdp.kp_pre_trained_policy_action import (
    KpPreTrainedPolicyAction,
    compute_kp_velocity_command,
    zoh_average_acceleration_gain,
)
from isaaclab_tasks.utils import load_cfg_from_registry


KP_TASK_ID = "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Obstacle-Avoidance-Unitree-Go2-v0"
KP_PLAY_TASK_ID = "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Obstacle-Avoidance-Unitree-Go2-Play-v0"
CBF_PLAY_TASK_ID = (
    "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Static-Obstacle-Cbf-Obstacle-Avoidance-Unitree-Go2-Play-v0"
)


def _limits(kp_value: float = 5.0):
    return (
        torch.tensor((kp_value, kp_value)),
        torch.tensor((-5.0, -5.0)),
        torch.tensor((5.0, 5.0)),
        torch.tensor((-1.3, -1.3)),
        torch.tensor((1.3, 1.3)),
    )


def test_kp_velocity_command_rebases_on_measured_velocity() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    gain = zoh_average_acceleration_gain(0.08, 0.30)
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[0.3, -0.2]]), torch.tensor([[0.1, -0.1]]), 0.08, 0.30, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.allclose(acceleration, torch.tensor([[1.0, -0.5]]))
    assert torch.allclose(command, torch.tensor([[0.1 + gain, -0.1 - 0.5 * gain]]))


def test_kp_velocity_command_respects_acceleration_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[1.0, -1.0]]), torch.zeros(1, 2), 0.08, 0.30, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.equal(acceleration, torch.tensor([[5.0, -5.0]]))
    assert torch.allclose(command, torch.tensor([[0.4, -0.4]]))


def test_kp_velocity_command_respects_velocity_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits()
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[2.0, -2.0]]), torch.tensor([[1.2, -1.2]]), 0.08, 0.30, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )
    assert torch.equal(acceleration, torch.tensor([[4.0, -4.0]]))
    assert torch.equal(command, torch.tensor([[1.3, -1.3]]))


def test_kp8_velocity_command_preserves_acceleration_and_velocity_limits() -> None:
    kp, accel_lo, accel_hi, vel_lo, vel_hi = _limits(8.0)
    acceleration, command = compute_kp_velocity_command(
        torch.tensor([[2.0, -2.0]]), torch.tensor([[1.2, -1.2]]), 0.08, 0.30, kp, accel_lo, accel_hi, vel_lo, vel_hi
    )

    assert torch.equal(acceleration, torch.tensor([[5.0, -5.0]]))
    assert torch.equal(command, torch.tensor([[1.3, -1.3]]))


def test_action_term_forwards_model_aware_velocity_and_unchanged_yaw() -> None:
    """The private low-level command differs only in planar Kp preprocessing."""
    term = object.__new__(KpPreTrainedPolicyAction)
    term.cfg = SimpleNamespace(action_scales=(1.0, 1.0, 1.0), tracking_tau_s=0.30)
    term._action_scales = torch.ones(3)
    term._raw_actions = torch.zeros(1, 3)
    term._processed_actions = torch.tensor([[1.2, -1.2, 0.0]])
    term._nominal_acceleration = torch.zeros(1, 2)
    term._kp, term._acceleration_lower, term._acceleration_upper, term._velocity_lower, term._velocity_upper = _limits()
    term._control_dt = 0.02
    term.robot = SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0]])))

    term.process_actions(torch.tensor([[1.0, -1.0, 0.7]]))
    term._update_model_aware_command()

    assert torch.equal(term.nominal_acceleration, torch.tensor([[5.0, -5.0]]))
    assert torch.allclose(term.processed_actions, torch.tensor([[1.3, -1.3, 0.7]]))


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
    assert kp_task.actions.pre_trained_policy_action.tracking_tau_s == 0.30
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


def test_cbf_play_task_preserves_the_trained_policy_interface() -> None:
    """The deployment filter changes only the private locomotion command path."""
    cfg = load_cfg_from_registry(CBF_PLAY_TASK_ID, "env_cfg_entry_point")

    assert isinstance(cfg, MixedTemporalLidarKpStaticObstacleCbfObstacleAvoidanceEnvCfg_PLAY)
    assert cfg.scene.num_envs == 16
    assert isinstance(cfg.actions.pre_trained_policy_action, StaticObstacleCbfPreTrainedPolicyActionCfg)
    assert cfg.actions.pre_trained_policy_action.action_scales == (1.0, 1.0, 1.0)
    assert cfg.actions.pre_trained_policy_action.kp == (8.0, 8.0)
    assert cfg.actions.pre_trained_policy_action.acceleration_limits == ((-5.0, 5.0), (-5.0, 5.0))
    assert cfg.actions.pre_trained_policy_action.velocity_limits == ((-1.5, 1.5), (-1.5, 1.5))
    assert cfg.actions.pre_trained_policy_action.d_margin == 0.70
    assert cfg.actions.pre_trained_policy_action.d_cbf_active == 5.0
    assert cfg.actions.pre_trained_policy_action.max_lidar_points == 64
    assert cfg.actions.pre_trained_policy_action.tracking_tau_s == 0.30
    assert cfg.sim.dt * cfg.actions.pre_trained_policy_action.low_level_decimation == 0.02


def test_cbf_zoh_mapping_rebases_on_measured_velocity() -> None:
    """The CBF command is a stateless inverse-model command, not an integrator."""
    gain = zoh_average_acceleration_gain(0.02, 0.30)
    assert gain == pytest.approx(0.310111, rel=1.0e-5)

    command = velocity_command_from_average_acceleration(
        torch.tensor([[1.0, -0.4]]), torch.tensor([[-5.0, 2.0]]), gain
    )
    assert torch.allclose(command, torch.tensor([[1.0 - 5.0 * gain, -0.4 + 2.0 * gain]]))


def test_cbf_zoh_effective_bounds_enforce_velocity_command_envelope() -> None:
    gain = zoh_average_acceleration_gain(0.02, 0.30)
    measured = torch.tensor([[1.4, -1.4]])
    lower, upper = effective_zoh_acceleration_bounds(
        measured,
        gain,
        torch.tensor((-5.0, -5.0)),
        torch.tensor((5.0, 5.0)),
        torch.tensor((-1.5, -1.5)),
        torch.tensor((1.5, 1.5)),
    )
    command_at_lower = velocity_command_from_average_acceleration(measured, lower, gain)
    command_at_upper = velocity_command_from_average_acceleration(measured, upper, gain)
    assert torch.all(command_at_lower >= torch.tensor((-1.5, -1.5)))
    assert torch.all(command_at_upper <= torch.tensor((1.5, 1.5)))


def test_cbf_solver_statistics_accumulate_osqp_results_without_cuda_tensors() -> None:
    """QP telemetry stays host-side so recording it does not add CUDA kernels."""
    term = object.__new__(StaticObstacleCbfPreTrainedPolicyAction)
    term._solver_solve_count = np.zeros(1, dtype=np.int64)
    term._solver_iteration_total = np.zeros(1, dtype=np.int64)
    term._solver_iteration_max = np.zeros(1, dtype=np.int64)
    term._solver_solve_time_total_s = np.zeros(1, dtype=np.float64)
    term._solver_solve_time_max_s = np.zeros(1, dtype=np.float64)
    term._solver_update_time_total_s = np.zeros(1, dtype=np.float64)
    term._solver_polish_time_total_s = np.zeros(1, dtype=np.float64)
    term._solver_primal_residual_max = np.zeros(1, dtype=np.float64)
    term._solver_dual_residual_max = np.zeros(1, dtype=np.float64)
    term._solver_inaccurate_count = np.zeros(1, dtype=np.int64)
    term._solver_max_iteration_count = np.zeros(1, dtype=np.int64)

    term._record_solver_stats(
        0,
        _OsqpSolveStats(
            iterations=25,
            solve_time_s=0.001,
            update_time_s=0.0002,
            polish_time_s=0.0001,
            primal_residual=2.0e-4,
            dual_residual=3.0e-4,
            status="solved inaccurate",
        ),
    )
    term._record_solver_stats(
        0,
        _OsqpSolveStats(
            iterations=40,
            solve_time_s=0.002,
            update_time_s=0.0003,
            polish_time_s=0.0,
            primal_residual=1.0e-4,
            dual_residual=5.0e-4,
            status="maximum iterations reached",
        ),
    )

    metrics = term.solver_metrics
    assert metrics["solve_count"].tolist() == [2]
    assert metrics["iteration_total"].tolist() == [65]
    assert metrics["iteration_max"].tolist() == [40]
    assert metrics["solve_time_total_s"].tolist() == pytest.approx([0.003])
    assert metrics["primal_residual_max"].tolist() == [2.0e-4]
    assert metrics["dual_residual_max"].tolist() == [5.0e-4]
    assert metrics["inaccurate_count"].tolist() == [1]
    assert metrics["max_iteration_count"].tolist() == [1]
