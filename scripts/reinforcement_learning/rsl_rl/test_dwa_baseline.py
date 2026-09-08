"""Focused unit tests for the DWA baseline's pure-tensor rollout/scoring core.

These exercise ``DwaController._compute_actions_from_state`` directly with hand-built tensors, so
no Isaac Sim app launch is required (the module itself defers all ``isaaclab``/``isaacsim`` imports
into ``compute_actions``, which these tests never call).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None

TORCH_AVAILABLE = torch is not None and hasattr(torch, "zeros")

MODULE_PATH = Path(__file__).with_name("dwa_baseline.py")
SPEC = importlib.util.spec_from_file_location("rsl_rl_dwa_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
dwa_baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dwa_baseline
SPEC.loader.exec_module(dwa_baseline)


def _make_scan(num_envs: int, num_rays: int, fov_deg: float, max_distance: float, hit_deg=None, hit_dist=None):
    """Build a ``(num_envs, num_rays)`` scan that is all max-range except optional forced hits.

    ``hit_deg``/``hit_dist`` are optional lists of ``(bearing_deg, distance_m)`` pairs applied to
    every environment, snapped to the nearest ray.
    """
    ray_angles_rad = torch.linspace(math.radians(-fov_deg / 2), math.radians(fov_deg / 2), num_rays)
    distances = torch.full((num_envs, num_rays), max_distance)
    if hit_deg is not None:
        for bearing_deg, dist in zip(hit_deg, hit_dist):
            bearing_rad = math.radians(bearing_deg)
            ray_index = int(torch.argmin(torch.abs(ray_angles_rad - bearing_rad)))
            distances[:, ray_index] = dist
    return distances, ray_angles_rad


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for DwaController tests")
def test_favors_caution_only_once_an_obstacle_is_actually_reachable_this_horizon():
    """From rest, one control step's dynamic window only permits a small velocity change (that's
    correct DWA behavior, not a bug) — so a single decision can't demonstrate a sharp turn away
    from an obstacle. It also should *not* slow down for an obstacle far enough away that it's
    not actually reachable within the rollout horizon (w_velocity was raised from the classic
    DWA literature's 0.2 "tie-breaker" default specifically to stop that over-cautious behavior).
    What it robustly demonstrates is that the clearance term still makes the controller *more
    cautious* (lower chosen speed) once an obstacle is close enough to matter within the horizon,
    compared to an identical goal with a clear path.
    """
    num_envs = 1

    def _compute(hit_deg, hit_dist):
        distances, ray_angles_rad = _make_scan(
            num_envs, num_rays=181, fov_deg=180.0, max_distance=20.0, hit_deg=hit_deg, hit_dist=hit_dist
        )
        controller = dwa_baseline.DwaController(num_rollout_steps=10)
        return controller._compute_actions_from_state(
            distances=distances,
            ray_angles_rad=ray_angles_rad,
            max_distance=20.0,
            goal_xy=torch.tensor([[3.0, 0.0]]),
            v_meas=torch.zeros(num_envs),
            w_meas=torch.zeros(num_envs),
            dt=0.08,
        )

    clear_actions = _compute(hit_deg=None, hit_dist=None)
    far_actions = _compute(hit_deg=[0.0], hit_dist=[1.0])
    near_actions = _compute(hit_deg=[0.0], hit_dist=[0.5])

    clear_vx = clear_actions[0, 0].item()
    far_vx = far_actions[0, 0].item()
    near_vx = near_actions[0, 0].item()
    assert clear_actions[0, 1].item() == 0.0 and near_actions[0, 1].item() == 0.0
    assert clear_vx > 0.0, f"expected forward progress on a clear path, got vx={clear_vx}"
    assert far_vx == pytest.approx(clear_vx, abs=1e-6), (
        f"expected an obstacle too far to reach this horizon to have no effect, "
        f"got far_vx={far_vx} vs clear_vx={clear_vx}"
    )
    assert near_vx < clear_vx, (
        f"expected a slower chosen speed once an obstacle is actually reachable this horizon, "
        f"got near_vx={near_vx} >= clear_vx={clear_vx}"
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for DwaController tests")
def test_drives_at_the_reachable_speed_toward_goal_with_no_obstacles():
    num_envs = 1
    distances, ray_angles_rad = _make_scan(num_envs, num_rays=181, fov_deg=180.0, max_distance=20.0)
    accel_upper_mps2 = 3.0
    dt = 0.08
    controller = dwa_baseline.DwaController(vel_abs_upper_mps=1.0, accel_upper_mps2=accel_upper_mps2, num_rollout_steps=10)
    actions = controller._compute_actions_from_state(
        distances=distances,
        ray_angles_rad=ray_angles_rad,
        max_distance=20.0,
        goal_xy=torch.tensor([[5.0, 0.0]]),
        v_meas=torch.zeros(num_envs),
        w_meas=torch.zeros(num_envs),
        dt=dt,
    )
    vx, vy, wz = actions[0, 0].item(), actions[0, 1].item(), actions[0, 2].item()
    # from rest, the one-step dynamic window caps reachable speed at accel_upper_mps2 * dt; with
    # nothing to avoid and a straight-ahead goal, the controller should pick that window's ceiling.
    expected_vx = accel_upper_mps2 * dt
    assert vy == 0.0
    assert vx == pytest.approx(expected_vx, abs=1e-3), f"expected vx≈{expected_vx} (window ceiling), got vx={vx}"
    assert abs(wz) < 0.2, f"expected near-zero turn toward a straight-ahead goal, got wz={wz}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for DwaController tests")
def test_falls_back_to_stop_and_rotate_when_boxed_in():
    num_envs = 1
    num_rays = 181
    ray_angles_rad = torch.linspace(math.radians(-90.0), math.radians(90.0), num_rays)
    # Every ray reports a hit well inside robot_radius + safety_margin, from every direction.
    distances = torch.full((num_envs, num_rays), 0.2)
    controller = dwa_baseline.DwaController(robot_radius_m=0.4, safety_margin_m=0.1, num_rollout_steps=5)
    actions = controller._compute_actions_from_state(
        distances=distances,
        ray_angles_rad=ray_angles_rad,
        max_distance=20.0,
        goal_xy=torch.tensor([[2.0, 2.0]]),
        v_meas=torch.zeros(num_envs),
        w_meas=torch.zeros(num_envs),
        dt=0.08,
    )
    vx, vy, wz = actions[0, 0].item(), actions[0, 1].item(), actions[0, 2].item()
    assert vx == 0.0, f"expected the boxed-in fallback to stop translating, got vx={vx}"
    assert vy == 0.0
    expected_wz = controller.fallback_yaw_gain * math.atan2(2.0, 2.0)
    assert wz == pytest.approx(min(expected_wz, controller.max_yaw_rate_rad_s), abs=1e-4)
