"""Unit tests for the Robust-v1 bounded base-wrench pulse curriculum."""

from types import SimpleNamespace

import torch

from isaaclab.managers import EventTermCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.impulse_events import BaseWrenchImpulse
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.locomotion_env_cfg import (
    LocomotionVelEnvCfg,
    LocomotionVelEnvCfg_ROBUST,
)


class _FakeWrenchComposer:
    def __init__(self, num_envs: int):
        self.force = torch.zeros(num_envs, 1, 3)
        self.torque = torch.zeros(num_envs, 1, 3)

    def set_forces_and_torques(self, forces=None, torques=None, positions=None, body_ids=None, env_ids=None, **kwargs):
        del body_ids, kwargs
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        if forces is not None:
            self.force[ids] = forces
        if torques is not None:
            self.torque[ids] = torques
        elif positions is not None and forces is not None:
            self.torque[ids] = torch.linalg.cross(positions, forces)


class _FakeScene(dict):
    def __init__(self, robot, terrain_levels):
        super().__init__(robot=robot)
        self.terrain = SimpleNamespace(terrain_levels=terrain_levels)


class _FakeCommandManager:
    def __init__(self, commands):
        self.commands = commands

    def get_command(self, name):
        assert name == "base_velocity"
        return self.commands


def _make_term():
    num_envs = 3
    robot = SimpleNamespace(permanent_wrench_composer=_FakeWrenchComposer(num_envs))
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.02,
        scene=_FakeScene(robot, torch.tensor([4, 5, 9])),
        command_manager=_FakeCommandManager(torch.tensor([[0.5, 0.0, 0.0]] * num_envs)),
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0])
    params = {
        "asset_cfg": asset_cfg,
        "command_name": "base_velocity",
        "terrain_level_threshold": 5,
        "settle_time_s": 0.0,
        "command_speed_threshold": 0.2,
        "pulse_interval_range_s": (3.0, 3.0),
        "pulse_duration_range_s": (0.04, 0.04),
        "impulse_ranges_by_level": ((5.0, 10.0), (10.0, 16.0), (16.0, 22.0), (22.0, 25.0)),
        "application_point_range_m": ((-0.15, 0.15), (-0.15, 0.15), (-0.10, 0.10)),
    }
    return BaseWrenchImpulse(EventTermCfg(func=BaseWrenchImpulse, mode="interval", params=params), env), env, params


def test_base_wrench_impulse_gates_and_scales_per_environment():
    torch.manual_seed(7)
    term, env, params = _make_term()
    term._time_to_next_pulse_s.zero_()
    ids = torch.arange(env.num_envs)

    term(env, ids, **params)

    # Level 4 remains untouched; levels 5 and 9 independently receive a kick.
    assert torch.count_nonzero(env["robot"].permanent_wrench_composer.force[0]) == 0
    assert torch.linalg.vector_norm(term._last_force_b[1]) > 0.0
    assert torch.linalg.vector_norm(term._last_force_b[2]) > 0.0
    assert 5.0 <= term._last_impulse_ns[1] <= 10.0
    assert 22.0 <= term._last_impulse_ns[2] <= 25.0
    torch.testing.assert_close(
        term._last_torque_b[1], torch.linalg.cross(term._last_application_point_b[1], term._last_force_b[1])
    )
    torch.testing.assert_close(env["robot"].permanent_wrench_composer.torque[1, 0], term._last_torque_b[1])


def test_base_wrench_impulse_respects_settle_time_and_command_gate():
    torch.manual_seed(3)
    term, env, params = _make_term()
    params["settle_time_s"] = 1.5
    term.reset(torch.tensor([1, 2]))
    term._time_to_next_pulse_s[1:3] = 0.0

    # Eligible terrain alone is insufficient during the 1.5 s settling period.
    term(env, torch.tensor([1, 2]), **params)
    assert torch.count_nonzero(env["robot"].permanent_wrench_composer.force[1:3]) == 0

    # Once settled, the per-environment command gate still excludes a stopped
    # robot while allowing its moving peer to receive a pulse.
    term._elapsed_time_s[1:3] = 1.5
    env.command_manager.commands[2, :2] = 0.0
    term(env, torch.tensor([1, 2]), **params)
    assert torch.linalg.vector_norm(env["robot"].permanent_wrench_composer.force[1]) > 0.0
    assert torch.count_nonzero(env["robot"].permanent_wrench_composer.force[2]) == 0


def test_base_wrench_impulse_uses_independent_per_environment_timers():
    torch.manual_seed(5)
    term, env, params = _make_term()
    term._time_to_next_pulse_s[:] = torch.tensor([0.0, 0.0, 1.0])

    term(env, torch.arange(env.num_envs), **params)

    assert torch.linalg.vector_norm(env["robot"].permanent_wrench_composer.force[1]) > 0.0
    assert torch.count_nonzero(env["robot"].permanent_wrench_composer.force[2]) == 0


def test_base_wrench_impulse_duration_and_reset_cleanup():
    torch.manual_seed(11)
    term, env, params = _make_term()
    term._time_to_next_pulse_s.zero_()
    ids = torch.arange(env.num_envs)
    term(env, ids, **params)

    # 40 ms at 20 ms control steps must persist for exactly two callbacks.
    assert term._remaining_pulse_steps[1].item() == 2
    term(env, ids, **params)
    assert torch.linalg.vector_norm(env["robot"].permanent_wrench_composer.force[1]) > 0.0
    term(env, ids, **params)
    assert torch.count_nonzero(env["robot"].permanent_wrench_composer.force[1]) == 0

    term.reset(torch.tensor([2]))
    assert torch.count_nonzero(env["robot"].permanent_wrench_composer.force[2]) == 0
    assert term._remaining_pulse_steps[2].item() == 0


def test_impulse_curriculum_is_robust_task_only():
    assert LocomotionVelEnvCfg().events.base_wrench_impulse is None
    impulse_cfg = LocomotionVelEnvCfg_ROBUST().events.base_wrench_impulse
    assert impulse_cfg is not None
    assert impulse_cfg.func is BaseWrenchImpulse
    assert impulse_cfg.interval_range_s == (0.02, 0.02)
