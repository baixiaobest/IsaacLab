"""Unit tests for the simulator-independent system-identification analysis helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomotion_system_identification_analysis import (  # noqa: E402
    conservative_tau,
    first_order_velocity,
    fit_first_order_response,
    fit_quality_warnings,
)


def test_first_order_fit_recovers_synthetic_tau() -> None:
    times = np.arange(0.02, 2.0, 0.02)
    expected_tau = 0.23
    measured = first_order_velocity(times, command_mps=1.0, initial_mps=0.0, tau_s=expected_tau)
    fit = fit_first_order_response(times, measured, command_mps=1.0, initial_mps=0.0)

    assert abs(fit.tau_s - expected_tau) < 1.0e-3
    assert fit.rmse_mps < 1.0e-5
    assert fit_quality_warnings(fit) == []


def test_fit_quality_flags_systematic_residuals() -> None:
    times = np.arange(0.02, 2.0, 0.02)
    # A delayed response has a systematic first-order residual rather than random noise.
    measured = first_order_velocity(np.maximum(times - 0.20, 0.0), command_mps=1.0, initial_mps=0.0, tau_s=0.18)
    fit = fit_first_order_response(times, measured, command_mps=1.0, initial_mps=0.0)

    assert fit_quality_warnings(fit)


def test_conservative_tau_uses_requested_percentile() -> None:
    assert conservative_tau([0.10, 0.20, 0.30, 0.40], percentile=95.0) == np.percentile([0.10, 0.20, 0.30, 0.40], 95.0)
