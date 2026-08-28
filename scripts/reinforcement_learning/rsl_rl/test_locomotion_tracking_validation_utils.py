"""Tests for the tracking-validation controller equations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomotion_tracking_validation_utils import (  # noqa: E402
    first_order_prediction_step,
    select_tracking_tau,
    tracking_velocity_command,
)


def test_tau_switch_uses_deceleration_only_when_braking() -> None:
    assert select_tracking_tau(np.array([1.0, 0.0]), np.array([-0.5, 0.0]), 0.586, 0.353) == 0.353
    assert select_tracking_tau(np.array([0.0, 0.0]), np.array([0.5, 0.0]), 0.586, 0.353) == 0.586


def test_cbf_proportional_inverse_maps_nominal_acceleration() -> None:
    command, unclipped, acceleration, tau = tracking_velocity_command(
        np.array([1.2, 0.0]),
        np.array([0.5, 0.0]),  # The CBF-style controller deliberately ignores v_ref_dot.
        np.array([0.8, 0.0]),
        controller="cbf_proportional_inverse",
        tau_accel_s=0.586,
        tau_decel_s=0.353,
        navigation_kp_s_inv=2.0,
        acceleration_lower=np.array([-5.0, -5.0]),
        acceleration_upper=np.array([5.0, 5.0]),
        velocity_lower=np.array([-1.5, -1.5]),
        velocity_upper=np.array([1.5, 1.5]),
    )
    assert tau == 0.586
    assert np.allclose(acceleration, [0.8, 0.0])
    assert np.allclose(unclipped, [1.2688, 0.0])
    assert np.allclose(command, [1.2688, 0.0])


def test_cbf_proportional_inverse_respects_acceleration_and_command_limits() -> None:
    command, unclipped, acceleration, _ = tracking_velocity_command(
        np.array([4.0, 0.0]),
        np.zeros(2),
        np.array([0.8, 0.0]),
        controller="cbf_proportional_inverse",
        tau_accel_s=0.30,
        tau_decel_s=0.353,
        navigation_kp_s_inv=8.0,
        acceleration_lower=np.array([-5.0, -5.0]),
        acceleration_upper=np.array([5.0, 5.0]),
        velocity_lower=np.array([-1.5, -1.5]),
        velocity_upper=np.array([1.5, 1.5]),
    )
    assert np.allclose(acceleration, [5.0, 0.0])
    assert np.allclose(unclipped, [2.3, 0.0])
    assert np.allclose(command, [1.5, 0.0])


def test_first_order_prediction_uses_acceleration_tau() -> None:
    next_velocity, tau = first_order_prediction_step(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.02, 0.586, 0.353
    )
    assert tau == 0.586
    assert np.allclose(next_velocity, [0.02 / 0.586, 0.0])


def test_feedforward_uses_deceleration_tau_for_a_braking_ramp() -> None:
    command, unclipped, acceleration, tau = tracking_velocity_command(
        np.array([1.0, 0.0]),
        np.array([-0.5, 0.0]),
        np.array([1.4, 0.0]),
        controller="feedforward",
        tau_accel_s=0.30,
        tau_decel_s=0.35,
        navigation_kp_s_inv=8.0,
        acceleration_lower=np.array([-5.0, -5.0]),
        acceleration_upper=np.array([5.0, 5.0]),
        velocity_lower=np.array([-1.5, -1.5]),
        velocity_upper=np.array([1.5, 1.5]),
    )
    assert tau == 0.35
    assert np.allclose(acceleration, [-0.5, 0.0])
    assert np.allclose(unclipped, [0.825, 0.0])
    assert np.allclose(command, [0.825, 0.0])
