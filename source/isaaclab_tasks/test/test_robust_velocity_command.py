"""Unit tests for the robust direct-twist locomotion command curriculum."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.robust_velocity_command import (
    RobustVelocityCommand,
    ScriptedVelocityCommand,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import (
    robust_velocity_tracking_terrain_curriculum,
)


def _bare_command_term(
    delay_ticks: int = 1, command_type: type[RobustVelocityCommand] = RobustVelocityCommand
) -> RobustVelocityCommand:
    """Construct the update-state portion without requiring an Isaac Sim environment."""
    term = object.__new__(command_type)
    term.num_envs = 1
    term.device = "cpu"
    term.cfg = SimpleNamespace(
        curriculum_full_level=6,
        jitter_start_level=3,
        jitter_planar_std=0.08,
        jitter_yaw_std=0.20,
        jitter_correlation_time_s=0.20,
        max_planar_speed=1.0,
        max_yaw_rate=1.2,
        max_planar_accel=1.5,
        max_yaw_accel=3.0,
        sudden_transition_start_probability=0.015,
        sudden_transition_approach_hold_s=0.60,
        sudden_transition_response_hold_s=0.60,
    )
    term._last_dt = 0.02
    term._target_command = torch.tensor([[1.0, 0.0, 1.2]])
    term._emitted_command = torch.zeros(1, 3)
    term._command = torch.zeros(1, 3)
    term._jitter = torch.zeros(1, 3)
    term._delay_ticks = torch.tensor([delay_ticks], dtype=torch.long)
    term._mode = torch.zeros(1, dtype=torch.long)
    term._pending_sudden_transition = torch.zeros(1, dtype=torch.long)
    term.time_left = torch.zeros(1)
    term._history = torch.zeros(4, 1, 3)
    term._history_index = 0
    term._terrain_alpha = lambda env_ids: torch.ones(len(env_ids))
    return term


def test_sampled_direct_twist_mixture_and_limits() -> None:
    torch.manual_seed(3)
    command, mode = RobustVelocityCommand.sample_direct_targets(100_000, "cpu")
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)

    assert torch.all(speed <= 1.0 + 1.0e-6)
    assert torch.all(torch.abs(command[:, 2]) <= 1.2 + 1.0e-6)
    assert torch.all(speed[mode == RobustVelocityCommand.ROTATE_IN_PLACE] == 0.0)

    expected = {
        RobustVelocityCommand.SLOW_COUPLED_TURN: 0.25,
        RobustVelocityCommand.ROTATE_IN_PLACE: 0.20,
        RobustVelocityCommand.NORMAL_COUPLED_MOTION: 0.40,
        RobustVelocityCommand.STRAIGHT_LATERAL_STOP: 0.15,
    }
    for mode_id, probability in expected.items():
        observed = (mode == mode_id).float().mean().item()
        assert abs(observed - probability) < 0.01

    turning = torch.abs(command[:, 2]) > 0.0
    assert abs((command[turning, 2] > 0.0).float().mean().item() - 0.5) < 0.015


def test_direct_yaw_has_no_heading_dependency() -> None:
    """The sampler has no robot-heading input and produces direct yaw-rate targets."""
    torch.manual_seed(11)
    first, _ = RobustVelocityCommand.sample_direct_targets(128, "cpu")
    torch.manual_seed(11)
    second, _ = RobustVelocityCommand.sample_direct_targets(128, "cpu")
    torch.testing.assert_close(first, second)
    assert torch.any(torch.abs(first[:, 2]) > 0.0)


def test_rate_limit_and_jitter_behavior() -> None:
    torch.manual_seed(7)
    term = _bare_command_term(delay_ticks=0)
    term._update_command()

    assert torch.linalg.vector_norm(term.emitted_command[:, :2], dim=-1).item() <= 1.5 * 0.02 + 1.0e-6
    assert abs(term.emitted_command[0, 2].item()) <= 3.0 * 0.02 + 1.0e-6

    term._target_command.zero_()
    for _ in range(8):
        term._update_command()
    assert torch.all(term._jitter == 0.0)


def test_command_delay_uses_the_requested_history_tick() -> None:
    for delay_ticks in (1, 2, 3):
        term = _bare_command_term(delay_ticks=delay_ticks)
        emitted = []
        delayed = []
        for _ in range(delay_ticks + 3):
            term._update_command()
            emitted.append(term.emitted_command.clone())
            delayed.append(term.command.clone())

        for step in range(delay_ticks):
            torch.testing.assert_close(delayed[step], torch.zeros_like(delayed[step]))
        for step in range(delay_ticks, len(delayed)):
            torch.testing.assert_close(delayed[step], emitted[step - delay_ticks])


def test_scripted_delivery_reuses_limiter_and_has_no_jitter() -> None:
    term = _bare_command_term(delay_ticks=1, command_type=ScriptedVelocityCommand)
    term._target_command[:] = torch.tensor([[1.0, 0.0, 1.2]])
    term._jitter[:] = 99.0
    term._update_command()

    assert torch.all(term._jitter == 0.0)
    assert torch.linalg.vector_norm(term.emitted_command[:, :2], dim=-1).item() <= 1.5 * 0.02 + 1.0e-6
    assert abs(term.emitted_command[0, 2].item()) <= 3.0 * 0.02 + 1.0e-6


def test_sudden_interventions_build_speed_before_raw_stop_or_avoidance_step() -> None:
    torch.manual_seed(17)
    term = _bare_command_term()
    term.cfg.sudden_transition_start_probability = 1.0

    term._resample_command(torch.tensor([0]))
    pending = int(term._pending_sudden_transition[0])
    assert pending in {1, 2}
    torch.testing.assert_close(term.target_command[0, 0], torch.tensor(0.75))
    assert abs(term.time_left[0].item() - 0.60) < 1.0e-6

    term._resample_command(torch.tensor([0]))
    assert int(term._pending_sudden_transition[0]) == 0
    assert abs(term.time_left[0].item() - 0.60) < 1.0e-6
    if pending == 1:
        torch.testing.assert_close(term.target_command, torch.zeros_like(term.target_command))
        assert int(term.mode[0]) == RobustVelocityCommand.SUDDEN_STOP
    else:
        torch.testing.assert_close(term.target_command[0, 0], torch.tensor(0.25))
        assert abs(term.target_command[0, 1].item()) == 0.65
        assert abs(term.target_command[0, 2].item()) == 0.80
        assert int(term.mode[0]) == RobustVelocityCommand.SUDDEN_AVOIDANCE_SWITCH


def test_tracking_terrain_curriculum_promotes_good_and_demotes_bad_episodes() -> None:
    class Terrain:
        terrain_levels = torch.tensor([3, 3, 3, 3], dtype=torch.long)

        def update_env_origins(self, env_ids, move_up, move_down) -> None:
            self.terrain_levels[env_ids] += move_up.long() - move_down.long()
            self.terrain_levels.clamp_(min=0)

    metrics = {
        "planar_rms_mps": torch.tensor([0.20, 0.26, 0.20, 0.00]),
        "yaw_rms_radps": torch.tensor([0.30, 0.30, 0.30, 0.00]),
        "stop_planar_rms_mps": torch.tensor([0.00, 0.00, 0.00, 0.00]),
        "stop_yaw_rms_radps": torch.tensor([0.00, 0.00, 0.00, 0.00]),
        "planar_active_steps": torch.tensor([10.0, 10.0, 10.0, 0.0]),
        "yaw_active_steps": torch.tensor([10.0, 10.0, 10.0, 0.0]),
        "stop_steps": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    }
    env = SimpleNamespace(
        device="cpu",
        num_envs=4,
        scene=SimpleNamespace(terrain=Terrain()),
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(episode_tracking_metrics=lambda _: metrics)),
        # Third episode has a non-timeout termination. Fourth is the initial
        # reset, with no episode samples, and must retain its random level.
        reset_terminated=torch.tensor([False, False, True, False]),
    )

    result = robust_velocity_tracking_terrain_curriculum(env, torch.arange(4))

    assert env.scene.terrain.terrain_levels.tolist() == [4, 2, 2, 3]
    assert result["good_fraction"].item() == 0.25
    assert result["bad_fraction"].item() == 0.50
