"""Pure fitting helpers for local locomotion velocity system identification.

This module intentionally has no Isaac Sim imports so saved traces can be
analysed and tested without launching the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FirstOrderFit:
    """A first-order response fit and diagnostics for one scalar velocity trace."""

    tau_s: float
    rmse_mps: float
    nrmse: float
    r_squared: float
    residual_lag1_correlation: float | None
    early_residual_mean_mps: float
    late_residual_mean_mps: float
    equation: str

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def first_order_velocity(times_s: np.ndarray, command_mps: float, initial_mps: float, tau_s: float) -> np.ndarray:
    """Evaluate ``v(t) = command + (initial - command) exp(-t / tau)``."""
    return command_mps + (initial_mps - command_mps) * np.exp(-np.asarray(times_s, dtype=float) / tau_s)


def parameterized_equation(command_mps: float, initial_mps: float, tau_s: float) -> str:
    """Return a compact human-readable form of the fitted response."""
    coefficient = initial_mps - command_mps
    sign = "+" if coefficient >= 0.0 else "-"
    return f"v(t) = {command_mps:.3f} {sign} {abs(coefficient):.3f} exp(-t / {tau_s:.3f})"


def fit_first_order_response(
    times_s: np.ndarray,
    measured_mps: np.ndarray,
    command_mps: float,
    initial_mps: float,
    *,
    min_tau_s: float = 0.005,
    max_tau_s: float = 5.0,
) -> FirstOrderFit:
    """Fit a positive time constant with deterministic bounded scalar search.

    The initial value is intentionally fixed to the measured pre-step velocity.
    That preserves the physical interpretation of the requested step model and
    avoids hiding an initial-condition error by fitting a second free parameter.
    """
    times = np.asarray(times_s, dtype=float)
    measured = np.asarray(measured_mps, dtype=float)
    if times.ndim != 1 or measured.ndim != 1 or len(times) != len(measured):
        raise ValueError("times_s and measured_mps must be one-dimensional arrays of equal length.")
    if len(times) < 5 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(measured)):
        raise ValueError("At least five finite response samples are required.")
    if min_tau_s <= 0.0 or max_tau_s <= min_tau_s:
        raise ValueError("Time-constant bounds must satisfy 0 < min_tau_s < max_tau_s.")

    if np.any(times < 0.0):
        raise ValueError("times_s must be elapsed non-negative response times.")

    def sse(log_tau: float) -> float:
        predicted = first_order_velocity(times, command_mps, initial_mps, math.exp(log_tau))
        return float(np.square(measured - predicted).sum())

    log_lower, log_upper = math.log(min_tau_s), math.log(max_tau_s)
    grid = np.linspace(log_lower, log_upper, 241)
    losses = np.asarray([sse(point) for point in grid])
    best_index = int(losses.argmin())
    lower = grid[max(0, best_index - 1)]
    upper = grid[min(len(grid) - 1, best_index + 1)]
    # Golden-section refinement in log(tau) space is stable across fast and slow responses.
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    left_loss, right_loss = sse(left), sse(right)
    for _ in range(80):
        if left_loss <= right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - golden * (upper - lower)
            left_loss = sse(left)
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + golden * (upper - lower)
            right_loss = sse(right)
    tau_s = math.exp((lower + upper) / 2.0)
    predicted = first_order_velocity(times, command_mps, initial_mps, tau_s)
    residuals = measured - predicted
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    step_size = max(abs(command_mps - initial_mps), 1.0e-6)
    total_variance = float(np.square(measured - measured.mean()).sum())
    # A constant trace cannot explain a non-zero command step.  Use a finite
    # sentinel so JSON output remains valid and quality checks reject it.
    r_squared = float(1.0 - np.square(residuals).sum() / total_variance) if total_variance > 1.0e-12 else -1.0
    if len(residuals) >= 3 and np.std(residuals[:-1]) > 1.0e-9 and np.std(residuals[1:]) > 1.0e-9:
        lag1: float | None = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        lag1 = None
    split = max(1, len(residuals) // 3)
    return FirstOrderFit(
        tau_s=tau_s,
        rmse_mps=rmse,
        nrmse=rmse / step_size,
        r_squared=r_squared,
        residual_lag1_correlation=lag1,
        early_residual_mean_mps=float(residuals[:split].mean()),
        late_residual_mean_mps=float(residuals[-split:].mean()),
        equation=parameterized_equation(command_mps, initial_mps, tau_s),
    )


def fit_quality_reasons(
    fit: FirstOrderFit,
    *,
    min_r_squared: float = 0.80,
    max_nrmse: float = 0.20,
    max_abs_residual_lag1: float = 0.90,
) -> list[str]:
    """Return explicit reasons a trace is unsuitable for conservative pooling."""
    reasons: list[str] = []
    if not math.isfinite(fit.r_squared) or fit.r_squared < min_r_squared:
        reasons.append(f"r_squared_below_{min_r_squared:g}")
    if fit.nrmse > max_nrmse:
        reasons.append(f"nrmse_above_{max_nrmse:g}")
    if fit.residual_lag1_correlation is not None and abs(fit.residual_lag1_correlation) > max_abs_residual_lag1:
        reasons.append(f"residual_lag1_above_{max_abs_residual_lag1:g}")
    return reasons


def conservative_tau(values_s: list[float] | np.ndarray, percentile: float = 95.0) -> float:
    """Return a conservative upper percentile after validating the input."""
    values = np.asarray(values_s, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Expected one or more finite positive tau values.")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100].")
    return float(np.percentile(values, percentile))
