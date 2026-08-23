"""Unit tests for six-frame temporal occupancy observations."""

from types import SimpleNamespace

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    MIXED_TEMPORAL_OCCUPANCY_COLLECTOR_CRITIC,
    MIXED_TEMPORAL_OCCUPANCY_COLLECTOR_POLICY,
    MIXED_TEMPORAL_OCCUPANCY_HISTORY_FRAMES,
    MixedOccupancyObservationsCfg,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.temporal_occupancy_env import (
    TemporalOccupancyCollector,
    temporal_occupancy_grid,
)


def _collector(num_envs: int = 3, frame_size: int = 1) -> TemporalOccupancyCollector:
    collector = object.__new__(TemporalOccupancyCollector)
    collector.num_envs = num_envs
    collector.device = "cpu"
    collector.frame_size = frame_size
    collector.cfg = SimpleNamespace(history_frames=6)
    collector._physics_steps = 0
    collector._last_capture_physics_step = torch.zeros(num_envs, dtype=torch.long)
    collector._capacity = 7
    collector._head = torch.zeros(num_envs, dtype=torch.long)
    collector._count = torch.zeros(num_envs, dtype=torch.long)
    collector._frames = torch.zeros(num_envs, collector._capacity, frame_size)
    return collector


def test_temporal_occupancy_history_is_six_chronological_non_current_frames() -> None:
    collector = _collector()
    env_ids = torch.arange(3)
    for value in range(7):
        collector._push(torch.full((3, 1), float(value)), env_ids)

    history = collector.history()

    assert history.shape == (3, 6)
    assert torch.equal(history[0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    # The seventh capture is current and must not enter the six-frame tail.
    assert history[0, -1] == 5.0


def test_temporal_occupancy_history_has_six_50x50_frames_per_environment() -> None:
    collector = _collector(num_envs=3, frame_size=50 * 50)

    assert collector.history_frames().shape == (3, 6, 50 * 50)
    assert collector.history().shape == (3, 6 * 50 * 50)


def test_temporal_occupancy_reset_clears_only_requested_environments() -> None:
    collector = _collector()
    env_ids = torch.arange(3)
    for value in range(7):
        collector._push(torch.full((3, 1), float(value)), env_ids)

    collector._physics_steps = 123
    collector.reset(torch.tensor([1]))

    history = collector.history()

    assert torch.equal(history[0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(history[1], torch.zeros(6))
    assert torch.equal(history[2], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))

    # Reset occurs before the simulator applies reset poses.  The collector
    # must not retain a reset-time sensor image and leak it into the first
    # emitted historical frame after the next two captures.
    reset_id = torch.tensor([1])
    collector._push(torch.tensor([[10.0]]), reset_id)
    collector._push(torch.tensor([[11.0]]), reset_id)
    assert torch.equal(collector.history()[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0]))


def test_temporal_occupancy_clock_uses_exact_half_second_physics_samples() -> None:
    collector = object.__new__(TemporalOccupancyCollector)
    collector._physics_steps = 0
    collector._sample_steps = 100
    collector._last_capture_physics_step = torch.zeros(2, dtype=torch.long)
    captures = []
    collector._capture = lambda env_ids: captures.append((collector._physics_steps, env_ids.clone()))

    for _ in range(300):
        collector.on_physics_step()

    assert [step for step, _ in captures] == [100, 200, 300]
    assert all(torch.equal(env_ids, torch.tensor([0, 1])) for _, env_ids in captures)


def test_temporal_occupancy_reset_subset_restarts_only_its_sampling_clock() -> None:
    collector = _collector()
    collector._sample_steps = 100
    captures = []
    collector._capture = lambda env_ids: captures.append((collector._physics_steps, env_ids.clone()))

    # Reset environment 1 one physics tick before the other two make their
    # first capture. Its next capture must be exactly 100 ticks after reset;
    # environments 0 and 2 must preserve their original cadence.
    for _ in range(99):
        collector.on_physics_step()
    collector.reset(torch.tensor([1]))

    collector.on_physics_step()
    for _ in range(99):
        collector.on_physics_step()
    collector.on_physics_step()

    assert [(step, env_ids.tolist()) for step, env_ids in captures] == [
        (100, [0, 2]),
        (199, [1]),
        (200, [0, 2]),
    ]


def test_mixed_policy_and_critic_use_separate_temporal_occupancy_histories() -> None:
    policy = MixedOccupancyObservationsCfg.PolicyCfg()
    critic = MixedOccupancyObservationsCfg.CriticCfg()
    assert policy.occupancy_grid.params["collector_name"] == MIXED_TEMPORAL_OCCUPANCY_COLLECTOR_POLICY
    assert critic.occupancy_grid.params["collector_name"] == MIXED_TEMPORAL_OCCUPANCY_COLLECTOR_CRITIC
    assert policy.occupancy_grid.params != critic.occupancy_grid.params


def test_registered_mixed_occupancy_configs_have_15000_value_tail_last() -> None:
    train_cfg = load_cfg_from_registry(
        "Isaac-Mixed-Static-Pedestrian-Occupancy-Obstacle-Avoidance-Unitree-Go2-v0", "env_cfg_entry_point"
    )
    play_cfg = load_cfg_from_registry(
        "Isaac-Mixed-Static-Pedestrian-Occupancy-Obstacle-Avoidance-Unitree-Go2-Play-v0", "env_cfg_entry_point"
    )
    agent_cfg = load_cfg_from_registry(
        "Isaac-Mixed-Static-Pedestrian-Occupancy-Obstacle-Avoidance-Unitree-Go2-v0", "rsl_rl_cfg_entry_point"
    )
    expected = MIXED_TEMPORAL_OCCUPANCY_HISTORY_FRAMES * 50 * 50

    for cfg in (train_cfg, play_cfg):
        assert cfg.temporal_occupancy_policy.history_frames == MIXED_TEMPORAL_OCCUPANCY_HISTORY_FRAMES
        assert cfg.temporal_occupancy_policy.sample_period_s == 0.5
        assert cfg.temporal_occupancy_critic.history_frames == MIXED_TEMPORAL_OCCUPANCY_HISTORY_FRAMES
        assert cfg.temporal_occupancy_critic.sample_period_s == 0.5
        # A history collector runs after each physics update, so the source
        # ray caster must also update on the physics grid rather than holding
        # a high-level-control-period scan.
        assert cfg.scene.obstacle_scanner.update_period == 0.0
        assert cfg.observations.policy.occupancy_grid.func is temporal_occupancy_grid
        assert cfg.observations.critic.occupancy_grid.func is temporal_occupancy_grid
        assert expected == 15000

    assert train_cfg.scene.num_envs == 2000
    assert play_cfg.scene.num_envs == 16
    assert agent_cfg.seed == 666
    assert agent_cfg.max_iterations == 2000
    assert agent_cfg.actor.class_name == "TemporalOccupancyModel"
    assert agent_cfg.critic.class_name == "TemporalOccupancyModel"
    assert agent_cfg.actor.temporal_obs_size == expected
    assert agent_cfg.critic.temporal_obs_size == expected
