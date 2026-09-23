"""Deterministic profiles and promotion gates for robust Go2 locomotion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


CONTROL_DT_S = 0.02
RESAMPLING_STRESS_UPDATE_S = 0.5
NOMINAL_CONDITION = "nominal_held"
RESAMPLING_STRESS_CONDITION = "resampling_stress_500ms"


@dataclass(frozen=True)
class LocomotionProfile:
    name: str
    family: str
    resampling_stress: bool


BASE_TRAJECTORIES = (
    "forward_tracking",
    "lateral_tracking",
    "diagonal_tracking",
    "rotate_in_place",
    "slow_forward_turn",
    "slow_lateral_turn",
    "normal_forward_turn",
    "normal_diagonal_turn",
    "stop_rotate_restart",
    "navigation_rate_sequence",
    "obstacle_stop",
    "obstacle_avoidance_switch",
)


def evaluation_profiles() -> tuple[LocomotionProfile, ...]:
    """Return the fixed 12-by-2 held-command and 2 Hz stress matrix."""
    conditions = (
        (NOMINAL_CONDITION, False),
        (RESAMPLING_STRESS_CONDITION, True),
    )
    return tuple(
        LocomotionProfile(f"{trajectory}:{condition}", trajectory, resampling_stress)
        for trajectory in BASE_TRAJECTORIES
        for condition, resampling_stress in conditions
    )


def _sign(episode_index: int) -> float:
    return -1.0 if episode_index % 2 else 1.0


def target_for_profile(profile: LocomotionProfile, elapsed_s: float, episode_index: int) -> np.ndarray:
    """Return an unfiltered latent target for one profile time and episode.

    A one-second settling phase starts every profile. All subsequent changes
    are intentional raw target steps delivered immediately by the scripted
    command term.
    """
    sign = _sign(episode_index)
    t = max(0.0, elapsed_s - 1.0)
    zero = np.zeros(3, dtype=np.float32)
    if elapsed_s < 1.0:
        return zero
    if profile.family == "forward_tracking":
        return np.array((0.70, 0.0, 0.0), dtype=np.float32)
    if profile.family == "lateral_tracking":
        return np.array((0.0, sign * 0.50, 0.0), dtype=np.float32)
    if profile.family == "diagonal_tracking":
        return np.array((0.70 / math.sqrt(2.0), sign * 0.70 / math.sqrt(2.0), 0.0), dtype=np.float32)
    if profile.family == "rotate_in_place":
        return np.array((0.0, 0.0, sign * 0.80), dtype=np.float32)
    if profile.family == "slow_forward_turn":
        return np.array((0.20, 0.0, sign * 0.60), dtype=np.float32)
    if profile.family == "slow_lateral_turn":
        return np.array((0.0, sign * 0.20, sign * 0.60), dtype=np.float32)
    if profile.family == "normal_forward_turn":
        return np.array((0.65, 0.0, sign * 0.80), dtype=np.float32)
    if profile.family == "normal_diagonal_turn":
        return np.array((0.65 / math.sqrt(2.0), sign * 0.65 / math.sqrt(2.0), sign * 0.80), dtype=np.float32)
    if profile.family == "stop_rotate_restart":
        if t < 2.0:
            return zero
        if t < 4.0:
            return np.array((0.0, 0.0, sign * 0.80), dtype=np.float32)
        if t < 7.0:
            return np.array((0.55, 0.0, sign * 0.55), dtype=np.float32)
        return zero
    if profile.family == "obstacle_stop":
        return np.array((0.75, 0.0, sign * 0.60), dtype=np.float32) if t < 3.0 else zero
    if profile.family == "obstacle_avoidance_switch":
        if t < 3.0 or t >= 6.0:
            return np.array((0.75, 0.0, 0.0), dtype=np.float32)
        return np.array((0.25, sign * 0.65, sign * 0.80), dtype=np.float32)
    if profile.family == "navigation_rate_sequence":
        # Fixed pseudo-random, 2 Hz target stream. The index and signs are
        # deterministic so candidate and baseline consume identical commands.
        # Nominal evaluation holds the initial sequence command. The stress
        # profile advances the deterministic sequence only on its 2 Hz update.
        sequence_index = int(t / RESAMPLING_STRESS_UPDATE_S) if profile.resampling_stress else 0
        generator = np.random.default_rng(10_000 * episode_index + sequence_index)
        mode = sequence_index % 4
        if mode == 0:
            target = np.array((0.65, 0.0, sign * 0.70), dtype=np.float32)
        elif mode == 1:
            target = np.array((0.20, sign * 0.55, -sign * 0.80), dtype=np.float32)
        elif mode == 2:
            target = np.array((0.0, 0.0, sign * 0.80), dtype=np.float32)
        else:
            angle = float(generator.uniform(-math.pi, math.pi))
            target = np.array((0.50 * math.cos(angle), 0.50 * math.sin(angle), sign * 0.50), dtype=np.float32)
        return target
    raise ValueError(f"Unknown locomotion evaluation trajectory {profile.family!r}.")


def percentile(values: Iterable[float], q: float) -> float | None:
    values = list(values)
    return float(np.percentile(values, q)) if values else None


def _rate(rows: list[dict[str, Any]], key: str = "fall") -> float:
    return sum(bool(row.get(key, False)) for row in rows) / len(rows) if rows else float("nan")


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def _bootstrap_rate_ratio(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]], rng_seed: int = 42
) -> tuple[float | None, float | None]:
    """Return a 95% bootstrap interval for candidate/baseline fall-rate ratio."""
    if not candidate or not baseline:
        return None, None
    candidate_values = np.asarray([bool(row.get("fall", False)) for row in candidate], dtype=float)
    baseline_values = np.asarray([bool(row.get("fall", False)) for row in baseline], dtype=float)
    baseline_rate = float(baseline_values.mean())
    if baseline_rate == 0.0:
        return None, None
    rng = np.random.default_rng(rng_seed)
    ratios = []
    for _ in range(2_000):
        c = float(rng.choice(candidate_values, size=len(candidate_values), replace=True).mean())
        b = float(rng.choice(baseline_values, size=len(baseline_values), replace=True).mean())
        if b > 0.0:
            ratios.append(c / b)
    return (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))) if ratios else (None, None)


def evaluate_gates(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]] | None, episodes_per_profile: int
) -> dict[str, Any]:
    """Compute absolute gates, with comparative gates only when a baseline exists."""
    baseline = baseline or []
    nominal = [row for row in candidate if row["condition"] == NOMINAL_CONDITION]
    stress = [row for row in candidate if row["condition"] != NOMINAL_CONDITION]
    coupled_families = {"slow_forward_turn", "slow_lateral_turn", "normal_forward_turn", "normal_diagonal_turn"}
    candidate_coupled = [row for row in stress if row["trajectory"] in coupled_families]
    baseline_coupled = [
        row for row in baseline if row["condition"] != NOMINAL_CONDITION and row["trajectory"] in coupled_families
    ]
    # A standalone evaluation has no baseline rows.  Preserve that absence as
    # JSON ``null`` in the artifact rather than emitting ``NaN``, which the
    # evaluator intentionally rejects when it serializes its summary.
    baseline_coupled_rate = _rate(baseline_coupled) if baseline_coupled else None
    candidate_coupled_rate = _rate(candidate_coupled)
    ratio_ci = _bootstrap_rate_ratio(candidate_coupled, baseline_coupled)

    tracking_keys = ("p95_vx_error", "p95_vy_error", "p95_wz_error")
    static_families = {"forward_tracking", "lateral_tracking", "diagonal_tracking", "rotate_in_place"}
    candidate_static = [row for row in candidate if row["trajectory"] in static_families]
    baseline_static = [row for row in baseline if row["trajectory"] in static_families]
    comparison_available = bool(baseline)
    tracking_regression = ({
        key: _mean_metric(candidate_static, key) <= 1.05 * _mean_metric(baseline_static, key)
        for key in tracking_keys
    } if comparison_available else {key: "not_assessed" for key in tracking_keys})
    quality_keys = ("p99_tilt_rad", "p95_action_rate", "torque_saturation_fraction")
    quality_regression = ({
        key: _mean_metric(candidate, key) <= 1.05 * _mean_metric(baseline, key)
        for key in quality_keys
    } if comparison_available else {key: "not_assessed" for key in quality_keys})
    candidate_stress_rate = _rate(stress)
    if comparison_available:
        zero_baseline = baseline_coupled_rate == 0.0
        coupled_gate = (
            candidate_coupled_rate == 0.0
            if zero_baseline
            else candidate_coupled_rate <= 0.70 * baseline_coupled_rate
            and ratio_ci[1] is not None and ratio_ci[1] <= 0.70
        )
    else:
        coupled_gate = "not_assessed"
    absolute_gates = {
        "sufficient_samples": episodes_per_profile >= 100,
        "nominal_zero_falls": _rate(nominal) == 0.0,
        "stress_fall_rate_at_most_5_percent": candidate_stress_rate <= 0.05,
    }
    comparative_gates = (
        {
            "coupled_turn_improvement": coupled_gate,
            "tracking_not_regressed": all(tracking_regression.values()),
            "quality_not_regressed": all(quality_regression.values()),
        }
        if comparison_available
        else {
            "coupled_turn_improvement": "not_assessed",
            "tracking_not_regressed": "not_assessed",
            "quality_not_regressed": "not_assessed",
        }
    )
    eligible = all(absolute_gates.values()) and (
        all(comparative_gates.values()) if comparison_available else True
    )
    return {
        "hardware_eligible": eligible,
        "eligible_for_hardware_validation": eligible,
        "comparison_available": comparison_available,
        "absolute_gates": absolute_gates,
        "comparative_gates": comparative_gates,
        # Retain the flattened shape for existing consumers while retaining an
        # explicit not_assessed state for standalone evaluations.
        "gates": {**absolute_gates, **comparative_gates},
        "fall_rates": {
            "nominal": _rate(nominal),
            "stress": candidate_stress_rate,
            "candidate_coupled_stress": candidate_coupled_rate,
            "baseline_coupled_stress": baseline_coupled_rate,
        },
        "coupled_turn_rate_ratio_95_ci": {"lower": ratio_ci[0], "upper": ratio_ci[1]},
        "tracking_regression": tracking_regression,
        "quality_regression": quality_regression,
        # Lower is better. Gates remain authoritative over this ranking value.
        "ranking_score": float(
            1000.0 * candidate_stress_rate
            + 100.0 * _mean_metric(candidate_coupled, "p95_tracking_error")
            + 10.0 * _mean_metric(candidate, "p99_tilt_rad")
            + _mean_metric(candidate, "p95_action_rate")
            + _mean_metric(candidate, "torque_saturation_fraction")
        ),
    }
