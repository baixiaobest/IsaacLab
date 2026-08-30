"""Unit tests for the held full-scan lidar collector."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.navigation.lidar_geometry import (
    body_to_world_xy,
    forward_lidar_reflection_bins,
    world_to_body_xy,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.held_scan_lidar_env import (
    HeldScanLidarCollector,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.temporal_lidar_env_cfg import (
    TEMPORAL_LIDAR_COLLECTOR_NAME,
    TEMPORAL_LIDAR_HISTORY_KEY,
    TEMPORAL_LIDAR_POS_NOISE_STD,
    TEMPORAL_LIDAR_RAYS,
    TemporalLidarObservationsCfg,
)


def test_full_scan_clock_fires_every_twenty_six_physics_steps() -> None:
    """A 130 ms full scan is captured every 26 physics steps on the 5 ms grid."""
    collector = object.__new__(HeldScanLidarCollector)
    collector.env = SimpleNamespace(physics_dt=0.005)
    collector._physics_steps = 0
    collector._time_s = 0.0
    collector._scan_steps = 26
    captures = []
    collector._capture_full_scan = lambda: captures.append(collector._physics_steps)

    for _ in range(78):
        collector.on_physics_step()

    assert captures == [26, 52, 78]
    assert abs(collector._time_s - 0.390) < 1.0e-12


def test_consume_returns_full_scan_once_then_holds_it() -> None:
    """Collector output preserves all source rays and is consumed only once."""
    collector = object.__new__(HeldScanLidarCollector)
    collector.num_envs = 2
    collector.device = "cpu"
    collector.cfg = SimpleNamespace(scan_period_s=0.130)
    collector._time_s = 0.130
    collector._pending_valid = torch.tensor([True, False])
    collector._has_latest = torch.tensor([False, False])
    collector._latest_reference_time_s = torch.zeros(2)
    collector._pending_reference_time_s = torch.tensor([0.130, 0.0])
    collector._pending_hit_xy = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]],
        ]
    )
    collector._pending_state = torch.tensor([[2, 1, 2, 1], [1, 1, 1, 1]], dtype=torch.uint8)
    collector._pending_ego_xy = torch.zeros(2, 2)
    collector._pending_ego_yaw = torch.zeros(2)

    completed = collector.consume_completed()

    assert completed is not None
    assert completed["hit_xy"].shape == (1, 4, 2)
    assert torch.equal(completed["hit_xy"][0], collector._pending_hit_xy[0])
    assert torch.equal(completed["ray_state"], torch.tensor([[2, 1, 2, 1]], dtype=torch.uint8))
    assert collector.consume_completed() is None


def test_scan_age_grows_while_a_full_scan_is_held() -> None:
    collector = object.__new__(HeldScanLidarCollector)
    collector.num_envs = 2
    collector.device = "cpu"
    collector.cfg = SimpleNamespace(scan_period_s=0.130)
    collector._time_s = 0.210
    collector._has_latest = torch.tensor([True, False])
    collector._latest_reference_time_s = torch.tensor([0.130, 0.0])

    age = collector.scan_age_s()

    assert torch.allclose(age, torch.tensor([0.080, 0.130]))


def test_reset_queues_an_immediate_scan_for_only_reset_environments() -> None:
    """Reset environments get a valid scan before their next policy action."""
    collector = object.__new__(HeldScanLidarCollector)
    collector.num_envs = 3
    collector.device = "cpu"
    collector._time_s = 0.210
    collector._pending_valid = torch.tensor([True, True, True])
    collector._has_latest = torch.tensor([True, True, True])
    collector._latest_reference_time_s = torch.zeros(3)
    captured = []
    collector._capture_full_scan = lambda env_ids: captured.append(env_ids.clone())

    collector.reset(torch.tensor([1, 2]))

    assert torch.equal(collector._pending_valid, torch.tensor([True, False, False]))
    assert torch.equal(collector._has_latest, torch.tensor([True, False, False]))
    assert torch.equal(collector._latest_reference_time_s, torch.tensor([0.0, 0.210, 0.210]))
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([1, 2]))


def test_rebinning_and_collector_noise_are_absent() -> None:
    """The held-scan model intentionally contains timing only."""
    assert not hasattr(HeldScanLidarCollector, "_rebin_to_policy")
    assert not hasattr(HeldScanLidarCollector, "_apply_simple_scan_noise")


def test_body_world_velocity_rotation_is_invertible() -> None:
    velocity_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    yaw = torch.tensor([torch.pi / 2.0])

    velocity_b = world_to_body_xy(velocity_w, yaw)

    assert torch.allclose(velocity_b, torch.tensor([[[0.0, -1.0], [1.0, 0.0]]]), atol=1.0e-6)
    assert torch.allclose(body_to_world_xy(velocity_b, yaw), velocity_w, atol=1.0e-6)


def test_forward_bins_keep_only_nearest_valid_reflection() -> None:
    # Both first rays lie in the forward bin at yaw zero; the 1 m return owns it.
    capture = {
        "hit_xy": torch.tensor([[[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]]),
        "ray_state": torch.tensor([[2, 2, 1]], dtype=torch.uint8),
        "ego_xy": torch.zeros(1, 2),
        "ego_yaw": torch.zeros(1),
    }
    binned = forward_lidar_reflection_bins(capture, num_bins=256, fov_bins=128)
    forward = 64

    assert binned["reflection_mask"][0, forward]
    assert binned["range_m"][0, forward] == 1.0
    assert binned["winner_ray"][0, forward] == 1


def test_actor_and_critic_share_held_history_and_scan_age() -> None:
    """Only actor corruption may differ; lidar timing and layout must match."""
    policy = TemporalLidarObservationsCfg.PolicyCfg()
    critic = TemporalLidarObservationsCfg.CriticCfg()

    assert policy.scan_age.params["collector_name"] == TEMPORAL_LIDAR_COLLECTOR_NAME
    assert critic.scan_age.params == policy.scan_age.params
    assert policy.obstacle_scan.params["history_key"] == TEMPORAL_LIDAR_HISTORY_KEY
    assert critic.obstacle_scan.params["history_key"] == TEMPORAL_LIDAR_HISTORY_KEY
    assert policy.obstacle_scan.params["history_num_rays"] == TEMPORAL_LIDAR_RAYS
    assert critic.obstacle_scan.params["history_num_rays"] == TEMPORAL_LIDAR_RAYS
    assert policy.obstacle_scan.params["pos_noise_std"] == TEMPORAL_LIDAR_POS_NOISE_STD
    assert critic.obstacle_scan.params["pos_noise_std"] == 0.0
    assert policy.obstacle_scan.noise.n_min == -0.05
    assert policy.obstacle_scan.noise.n_max == 0.05
    assert critic.obstacle_scan.noise is None
