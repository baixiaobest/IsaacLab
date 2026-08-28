"""Simulator-independent helpers for velocity-model tracking validation."""

from __future__ import annotations

import numpy as np


def select_tracking_tau(
    measured_velocity: np.ndarray,
    desired_acceleration: np.ndarray,
    tau_accel_s: float,
    tau_decel_s: float,
) -> float:
    """Select a global time constant from the intended speed change.

    A negative velocity--acceleration dot product means that the requested
    acceleration opposes the current direction of travel, i.e. braking.  At
    rest and during turns we deliberately use the conservative acceleration
    value.
    """
    if float(np.dot(measured_velocity, desired_acceleration)) < 0.0:
        return tau_decel_s
    return tau_accel_s


def tracking_velocity_command(
    reference_velocity: np.ndarray,
    reference_acceleration: np.ndarray,
    measured_velocity: np.ndarray,
    *,
    controller: str,
    tau_accel_s: float,
    tau_decel_s: float,
    feedback_gain_s_inv: float,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return clipped command, unclipped command, desired acceleration, and tau.

    ``feedback_inverse`` implements the inverse first-order model
    ``v_cmd = v + tau * (v_ref_dot + k_e * (v_ref - v))``.  The other two
    modes provide the no-compensation and feed-forward comparison baselines.
    """
    if controller == "baseline":
        desired_acceleration = reference_acceleration
        tau_s = select_tracking_tau(measured_velocity, desired_acceleration, tau_accel_s, tau_decel_s)
        unclipped = reference_velocity
    elif controller == "feedforward":
        desired_acceleration = reference_acceleration
        tau_s = select_tracking_tau(measured_velocity, desired_acceleration, tau_accel_s, tau_decel_s)
        unclipped = reference_velocity + tau_s * desired_acceleration
    elif controller == "feedback_inverse":
        desired_acceleration = reference_acceleration + feedback_gain_s_inv * (reference_velocity - measured_velocity)
        tau_s = select_tracking_tau(measured_velocity, desired_acceleration, tau_accel_s, tau_decel_s)
        unclipped = measured_velocity + tau_s * desired_acceleration
    else:
        raise ValueError(f"Unsupported tracking controller '{controller}'.")
    return np.clip(unclipped, velocity_lower, velocity_upper), unclipped, desired_acceleration, tau_s


def first_order_prediction_step(
    predicted_velocity: np.ndarray,
    command_velocity: np.ndarray,
    dt_s: float,
    tau_accel_s: float,
    tau_decel_s: float,
) -> tuple[np.ndarray, float]:
    """Advance the identified plant model by one Euler control step."""
    tau_s = select_tracking_tau(
        predicted_velocity,
        command_velocity - predicted_velocity,
        tau_accel_s,
        tau_decel_s,
    )
    return predicted_velocity + dt_s * (command_velocity - predicted_velocity) / tau_s, tau_s
