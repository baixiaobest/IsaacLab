"""Unit tests for the completed-scan lidar collector."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.two_cloud_lidar_env import (
    TwoCloudLidarCollector,
)


def test_simple_scan_clock_fires_every_twenty_six_physics_steps() -> None:
    """The collector captures one complete scan every 130 ms on the 5 ms grid."""
    collector = object.__new__(TwoCloudLidarCollector)
    collector.env = SimpleNamespace(physics_dt=0.005)
    collector._physics_steps = 0
    collector._time_s = 0.0
    collector._completed_steps = 26
    captures = []
    collector._capture_complete_scan = lambda: captures.append(collector._physics_steps)

    for _ in range(78):
        collector.on_physics_step()

    assert captures == [26, 52, 78]
    assert abs(collector._time_s - 0.390) < 1.0e-12


def test_rebin_retains_nearest_hit_or_observed_free_ray() -> None:
    """Two source rays reduce to one policy direction without fabricating validity."""
    collector = object.__new__(TwoCloudLidarCollector)
    collector.num_envs = 1
    collector.num_rays = 4
    collector.completed_num_rays = 2

    hit_xy = torch.tensor([[[5.0, 0.0], [2.0, 0.0], [8.0, 0.0], [9.0, 0.0]]])
    state = torch.tensor([[2, 2, 0, 1]], dtype=torch.uint8)
    xy, rebinned_state = collector._rebin_to_policy(hit_xy, state, torch.zeros(1, 2))

    # The first output takes the nearest of two hits. The second has no hit and
    # retains its one real free-space contribution rather than becoming invalid.
    assert torch.equal(rebinned_state, torch.tensor([[2, 1]], dtype=torch.uint8))
    assert torch.equal(xy[0, 0], torch.tensor([2.0, 0.0]))
    assert torch.equal(xy[0, 1], torch.tensor([9.0, 0.0]))


def test_simple_scan_noise_is_identity_when_both_stds_are_zero() -> None:
    """The baseline can be made exactly ideal for staged ablation training."""
    collector = object.__new__(TwoCloudLidarCollector)
    collector.num_envs = 1
    collector.device = "cpu"
    collector.cfg = SimpleNamespace(iid_hit_position_noise_std_m=0.0, iid_yaw_noise_std_deg=0.0)

    xy = torch.tensor([[[2.0, 0.0], [20.0, 0.0], [0.0, 0.0]]])
    state = torch.tensor([[2, 1, 0]], dtype=torch.uint8)
    result = collector._apply_simple_scan_noise(xy, state, torch.zeros(1, 2))

    assert torch.equal(result, xy)


def test_scan_age_grows_while_a_completed_scan_is_held() -> None:
    collector = object.__new__(TwoCloudLidarCollector)
    collector.num_envs = 2
    collector.device = "cpu"
    collector._time_s = 0.210
    collector._has_latest = torch.tensor([True, False])
    collector._latest_reference_time_s = torch.tensor([0.130, 0.0])

    age = collector.scan_age_s()

    assert torch.allclose(age, torch.tensor([0.080, 0.250]))
