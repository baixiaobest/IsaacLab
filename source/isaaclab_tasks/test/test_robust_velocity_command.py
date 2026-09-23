"""Unit tests for the robust direct-twist locomotion command curriculum."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandTerm
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.robust_velocity_command import (
    RobustVelocityCommand,
    RobustVelocityCommandCfg,
    ScriptedVelocityCommand,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import (
    robust_velocity_tracking_terrain_curriculum,
)


def _bare_command_term(
    level: int = 0, command_type: type[RobustVelocityCommand] = RobustVelocityCommand
) -> RobustVelocityCommand:
    """Construct command state without requiring an Isaac Sim environment."""
    term = object.__new__(command_type)
    term.num_envs = 1
    term.device = "cpu"
    term.cfg = SimpleNamespace(
        incremental_start_terrain_level=5,
        incremental_full_terrain_level=9,
        incremental_start_frequency_hz=0.5,
        incremental_full_frequency_hz=2.0,
        max_planar_delta_mps=0.2,
        max_yaw_delta_radps=0.3,
        max_planar_speed=1.5,
        max_yaw_rate=2.0,
        normal_yaw_full_cap_speed_mps=1.0,
        normal_yaw_cap_at_max_planar_speed_radps=1.0,
        sudden_change_time_fraction=0.5,
    )
    terrain = SimpleNamespace(terrain_levels=torch.tensor([level], dtype=torch.long))
    term._env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), max_episode_length_s=10.0)
    term._target_command = torch.tensor([[0.8, 0.2, 0.7]])
    term._emitted_command = torch.zeros(1, 3)
    term._command = torch.zeros(1, 3)
    term._mode = torch.tensor([RobustVelocityCommand.NORMAL_COUPLED_MOTION])
    term._sudden_change_fired = torch.zeros(1, dtype=torch.bool)
    term.time_left = torch.zeros(1)
    term.command_counter = torch.zeros(1, dtype=torch.long)
    return term


def test_sampled_direct_twist_mixture_and_limits() -> None:
    torch.manual_seed(3)
    command, mode = RobustVelocityCommand.sample_direct_targets(100_000, "cpu", 1.5, 2.0)
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)

    assert torch.all(speed <= 1.5 + 1.0e-6)
    assert torch.all(torch.abs(command[:, 2]) <= 2.0 + 1.0e-6)
    assert speed.max() > 1.49
    assert torch.abs(command[:, 2]).max() > 1.99
    assert torch.all(speed[mode == RobustVelocityCommand.ROTATE_IN_PLACE] == 0.0)
    assert torch.all(command[mode == RobustVelocityCommand.FULL_STOP] == 0.0)
    expected = {
        RobustVelocityCommand.SLOW_COUPLED_TURN: 0.15,
        RobustVelocityCommand.ROTATE_IN_PLACE: 0.15,
        RobustVelocityCommand.FULL_STOP: 0.10,
        RobustVelocityCommand.NORMAL_COUPLED_MOTION: 0.40,
        RobustVelocityCommand.SUDDEN_CHANGE: 0.20,
    }
    for mode_id, probability in expected.items():
        assert abs((mode == mode_id).float().mean().item() - probability) < 0.01
    assert set(mode.unique().tolist()) == set(expected)


def test_mode_mixture_is_configurable_and_must_sum_to_one() -> None:
    command, mode = RobustVelocityCommand.sample_direct_targets(
        1_000,
        "cpu",
        slow_coupled_turn_probability=0.0,
        rotate_in_place_probability=0.0,
        full_stop_probability=1.0,
        normal_coupled_motion_probability=0.0,
        sudden_change_probability=0.0,
    )
    assert torch.all(command == 0.0)
    assert torch.all(mode == RobustVelocityCommand.FULL_STOP)

    with pytest.raises(ValueError, match="sum to one"):
        RobustVelocityCommandCfg(
            resampling_time_range=(1.0, 1.0),
            full_stop_probability=0.2,
        )


def test_direct_yaw_has_no_heading_dependency() -> None:
    torch.manual_seed(11)
    first, _ = RobustVelocityCommand.sample_direct_targets(128, "cpu")
    torch.manual_seed(11)
    second, _ = RobustVelocityCommand.sample_direct_targets(128, "cpu")
    torch.testing.assert_close(first, second)
    assert torch.any(torch.abs(first[:, 2]) > 0.0)


def test_normal_and_sudden_yaw_cap_shrinks_above_one_meter_per_second() -> None:
    command, mode = RobustVelocityCommand.sample_direct_targets(100_000, "cpu")
    coupled = (mode == RobustVelocityCommand.NORMAL_COUPLED_MOTION) | (
        mode == RobustVelocityCommand.SUDDEN_CHANGE
    )
    speed = torch.linalg.vector_norm(command[coupled, :2], dim=-1)
    yaw_cap = 2.0 - 2.0 * (speed - 1.0).clamp(min=0.0, max=0.5)

    assert torch.all(torch.abs(command[coupled, 2]) <= yaw_cap + 1.0e-6)
    fast_coupled = coupled & (torch.linalg.vector_norm(command[:, :2], dim=-1) > 1.45)
    assert torch.any(fast_coupled)
    assert torch.all(torch.abs(command[fast_coupled, 2]) <= 1.1)


def test_robust_command_implements_debug_visualization() -> None:
    assert RobustVelocityCommand._set_debug_vis_impl is not CommandTerm._set_debug_vis_impl
    assert RobustVelocityCommand._debug_vis_callback is not CommandTerm._debug_vis_callback


def test_levels_zero_through_four_hold_normal_prior_for_episode() -> None:
    for level in range(5):
        term = _bare_command_term(level)
        term._set_next_resample_time(torch.tensor([0]))
        assert torch.isinf(term.time_left[0])
        before = term.target_command.clone()
        term.command_counter[:] = 1
        term._resample_existing_priors(torch.tensor([0]))
        torch.testing.assert_close(term.target_command, before)
        assert torch.isinf(term.time_left[0])


def test_sudden_change_occurs_once_at_episode_midpoint_and_is_independent() -> None:
    term = _bare_command_term(level=0)
    term._mode[:] = RobustVelocityCommand.SUDDEN_CHANGE
    prior = term.target_command.clone()
    term._set_next_resample_time(torch.tensor([0]))
    assert term.time_left.item() == 5.0

    torch.manual_seed(91)
    term.command_counter[:] = 1
    term._resample_existing_priors(torch.tensor([0]))
    assert term._sudden_change_fired.item()
    assert torch.isinf(term.time_left[0])
    assert not torch.equal(term.target_command, prior)
    assert torch.linalg.vector_norm(term.target_command[:, :2], dim=-1).item() <= 1.5
    assert abs(term.target_command[0, 2].item()) <= 2.0
    after = term.target_command.clone()
    term._resample_existing_priors(torch.tensor([0]))
    torch.testing.assert_close(term.target_command, after)


def test_normal_prior_updates_have_specified_schedule_and_delta_bounds() -> None:
    expected_periods = {5: 2.0, 6: 1.0 / 0.875, 7: 0.8, 8: 1.0 / 1.625, 9: 0.5, 12: 0.5}
    for level, expected_period in expected_periods.items():
        term = _bare_command_term(level)
        torch.testing.assert_close(term._normal_update_period_s(torch.tensor([0])), torch.tensor([expected_period]))
        prior = term.target_command.clone()
        term.command_counter[:] = 1
        torch.manual_seed(level)
        term._resample_existing_priors(torch.tensor([0]))
        planar_delta = torch.linalg.vector_norm(term.target_command[:, :2] - prior[:, :2], dim=-1)
        yaw_delta = torch.abs(term.target_command[:, 2] - prior[:, 2])
        assert planar_delta.item() <= 0.2 + 1.0e-6
        assert yaw_delta.item() <= 0.3 + 1.0e-6
        assert term.time_left.item() == pytest.approx(expected_period)


def test_incremental_updates_depend_on_previous_prior() -> None:
    torch.manual_seed(31)
    first = _bare_command_term(level=9)
    first.command_counter[:] = 1
    first._resample_existing_priors(torch.tensor([0]))
    torch.manual_seed(31)
    second = _bare_command_term(level=9)
    second._target_command[:] = torch.tensor([[-0.8, -0.2, -0.7]])
    second.command_counter[:] = 1
    second._resample_existing_priors(torch.tensor([0]))
    # Identical sampled deltas are applied to distinct priors.
    assert not torch.equal(first.target_command, second.target_command)


def test_command_is_immediate_and_has_no_delay_or_jitter_state() -> None:
    term = _bare_command_term()
    term._update_command()
    torch.testing.assert_close(term.command, term.target_command)
    torch.testing.assert_close(term.emitted_command, term.target_command)
    for obsolete_attribute in ("delay_ticks", "_delay_ticks", "_history", "_jitter", "set_delay_ticks"):
        assert not hasattr(term, obsolete_attribute)


def test_scripted_command_is_immediate_and_bounded() -> None:
    term = _bare_command_term(command_type=ScriptedVelocityCommand)
    term._target_command[:] = torch.tensor([[2.0, 0.0, 3.0]])
    term._update_command()
    torch.testing.assert_close(term.command, torch.tensor([[1.5, 0.0, 2.0]]))
    torch.testing.assert_close(term.command, term.emitted_command)
    assert not hasattr(term, "set_delay_ticks")


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
        reset_terminated=torch.tensor([False, False, True, False]),
    )
    result = robust_velocity_tracking_terrain_curriculum(env, torch.arange(4))
    assert env.scene.terrain.terrain_levels.tolist() == [4, 2, 2, 3]
    assert result["good_fraction"].item() == 0.25
    assert result["bad_fraction"].item() == 0.50
