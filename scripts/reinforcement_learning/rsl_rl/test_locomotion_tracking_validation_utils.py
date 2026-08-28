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


def test_feedback_inverse_applies_model_inverse_and_clips() -> None:
    command, unclipped, acceleration, tau = tracking_velocity_command(
        np.array([1.2, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.8, 0.0]),
        controller="feedback_inverse",
        tau_accel_s=0.586,
        tau_decel_s=0.353,
        feedback_gain_s_inv=2.0,
        velocity_lower=np.array([-1.5, -1.5]),
        velocity_upper=np.array([1.5, 1.5]),
    )
    assert tau == 0.586
    assert np.allclose(acceleration, [1.3, 0.0])
    assert np.allclose(unclipped, [1.5618, 0.0])
    assert np.allclose(command, [1.5, 0.0])


def test_first_order_prediction_uses_acceleration_tau() -> None:
    next_velocity, tau = first_order_prediction_step(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.02, 0.586, 0.353
    )
    assert tau == 0.586
    assert np.allclose(next_velocity, [0.02 / 0.586, 0.0])
