"""Unit tests for the completed two-cloud lidar timing and merge rules."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.two_cloud_lidar_env import (
    TwoCloudLidarCollector,
)


def test_raw_cloud_clock_fires_every_thirteen_physics_steps() -> None:
    """A 65 ms raw cloud is captured on the 5 ms grid, not every physics tick."""
    collector = object.__new__(TwoCloudLidarCollector)
    collector.env = SimpleNamespace(physics_dt=0.005)
    collector._physics_steps = 0
    collector._time_s = 0.0
    collector._raw_steps = 13
    captures = []
    collector._capture_raw_cloud = lambda: captures.append(collector._physics_steps)

    for _ in range(39):
        collector.on_physics_step()

    assert captures == [13, 26, 39]
    assert abs(collector._time_s - 0.195) < 1.0e-12


def test_merge_prefers_younger_hit_without_world_origin_bias() -> None:
    """The second raw cloud wins an overlapping hit even far from world origin."""
    first_xy = torch.tensor([[[100.0, 100.0], [101.0, 100.0], [102.0, 100.0]]])
    second_xy = torch.tensor([[[150.0, 100.0], [151.0, 100.0], [152.0, 100.0]]])
    first_state = torch.tensor([[2, 2, 1]], dtype=torch.uint8)
    second_state = torch.tensor([[2, 1, 2]], dtype=torch.uint8)

    xy, state = TwoCloudLidarCollector._merge_raw(first_xy, first_state, second_xy, second_state)

    # A younger hit overrides an older hit/free contribution; a younger free ray
    # does not erase an older hit.
    assert torch.equal(state, torch.tensor([[2, 2, 2]], dtype=torch.uint8))
    assert torch.equal(xy[0, 0], second_xy[0, 0])
    assert torch.equal(xy[0, 1], first_xy[0, 1])
    assert torch.equal(xy[0, 2], second_xy[0, 2])


def test_raw_rebin_retains_nearest_hit_or_observed_free_ray() -> None:
    """Two source rays reduce to one policy direction without fabricating validity."""
    collector = object.__new__(TwoCloudLidarCollector)
    collector.num_envs = 1
    collector.num_rays = 4
    collector.completed_num_rays = 2

    hit_xy = torch.tensor([[[5.0, 0.0], [2.0, 0.0], [8.0, 0.0], [9.0, 0.0]]])
    state = torch.tensor([[2, 2, 0, 1]], dtype=torch.uint8)
    xy, rebinned_state = collector._rebin_raw_to_policy(hit_xy, state, torch.zeros(1, 2))

    # The first output takes the nearest of two hits. The second has no hit and
    # retains its one real free-space contribution rather than becoming invalid.
    assert torch.equal(rebinned_state, torch.tensor([[2, 1]], dtype=torch.uint8))
    assert torch.equal(xy[0, 0], torch.tensor([2.0, 0.0]))
    assert torch.equal(xy[0, 1], torch.tensor([9.0, 0.0]))


def test_scan_age_grows_while_a_completed_scan_is_held() -> None:
    collector = object.__new__(TwoCloudLidarCollector)
    collector.num_envs = 2
    collector.device = "cpu"
    collector._time_s = 0.210
    collector._has_latest = torch.tensor([True, False])
    collector._latest_reference_time_s = torch.tensor([0.130, 0.0])

    age = collector.scan_age_s()

    assert torch.allclose(age, torch.tensor([0.080, 0.250]))


def test_completed_scan_is_not_consumed_before_latency_expires() -> None:
    collector = object.__new__(TwoCloudLidarCollector)
    collector.num_envs = 1
    collector.device = "cpu"
    collector._pending_valid = torch.tensor([True])
    collector._pending_available_time_s = torch.tensor([0.155])
    collector._pending_reference_time_s = torch.tensor([0.130])
    collector._pending_hit_xy = torch.zeros(1, 2, 2)
    collector._pending_state = torch.ones(1, 2, dtype=torch.uint8)
    collector._pending_ego_xy = torch.zeros(1, 2)
    collector._pending_ego_yaw = torch.zeros(1)
    collector._latest_reference_time_s = torch.zeros(1)
    collector._has_latest = torch.zeros(1, dtype=torch.bool)

    collector._time_s = 0.150
    assert collector.consume_completed() is None
    assert collector._pending_valid.item()

    collector._time_s = 0.160
    completed = collector.consume_completed()
    assert completed is not None
    assert completed["env_ids"].tolist() == [0]
    assert not collector._pending_valid.item()
    assert collector._has_latest.item()
    assert torch.allclose(completed["scan_age_s"], torch.tensor([0.030]))
