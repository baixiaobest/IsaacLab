# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation evaluation %s."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from .common import (
    INTERACTION_AGAINST_FLOW_SIDESTEP_LATERAL_M,
    INTERACTION_ASSERT_SPEED_RATIO,
    INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M,
    INTERACTION_LABELS,
    INTERACTION_MIN_BASELINE_SPEED_MPS,
    INTERACTION_MIN_DURATION_S,
    INTERACTION_YIELD_SPEED_RATIO,
)

def classify_speed_interaction(
    scenario: str,
    risk_seen: bool,
    duration_s: float,
    baseline_speed_mps: float,
    event_speeds_mps: Iterable[float],
    initial_longitudinal_m: float | None = None,
    final_longitudinal_m: float | None = None,
    front_crossed: bool = False,
) -> tuple[str, float | None, float | None]:
    """Return the canonical event label plus low speed and speed ratio.

    This is retained for diagnostic compatibility with the event collector. With-flow
    is intentionally absent: both with-flow variants use slot-zero episode outcomes.
    """
    if scenario not in INTERACTION_LABELS:
        raise ValueError(f"Unsupported interaction scenario: {scenario}")
    if not risk_seen:
        return "non_risky_close", None, None

    low_speed, ratio = interaction_speed_diagnostics(duration_s, baseline_speed_mps, event_speeds_mps)
    if low_speed is None or ratio is None:
        return "unclassified", None, None
    if ratio < INTERACTION_YIELD_SPEED_RATIO:
        return "yield", low_speed, ratio
    if scenario in {"against_flow", "against_flow_slow"} and ratio > INTERACTION_ASSERT_SPEED_RATIO:
        return "assert", low_speed, ratio
    return "ambiguous", low_speed, ratio


def interaction_speed_diagnostics(
    duration_s: float, baseline_speed_mps: float, event_speeds_mps: Iterable[float]
) -> tuple[float | None, float | None]:
    """Return legacy speed diagnostics without assigning a behavior label."""
    speeds = np.asarray(list(event_speeds_mps), dtype=float)
    if (
        duration_s < INTERACTION_MIN_DURATION_S
        or not np.isfinite(baseline_speed_mps)
        or baseline_speed_mps < INTERACTION_MIN_BASELINE_SPEED_MPS
        or speeds.size == 0
    ):
        return None, None
    # A low percentile is robust to small locomotion oscillations while preserving deliberate
    # short waits.  The event collector records one value per evaluation control step.
    low_speed = float(np.percentile(speeds, 10.0))
    return low_speed, low_speed / baseline_speed_mps


def classify_crossing_interaction(
    risk_seen: bool,
    front_crossed: bool,
    geometry_available: bool,
    core_resolved_sides: Iterable[int],
    rear_crossed: bool,
) -> str:
    """Classify crossing behavior from active-event pedestrian-frame geometry.

    A front crossing is deliberately checked first to preserve the established assert
    definition.  Yield is either a rear side-to-side pass or consistent travel on one
    lateral side during the active interaction.  Speed remains a diagnostic only.
    """
    if not risk_seen:
        return "non_risky_close"
    if front_crossed:
        return "assert"
    resolved_sides = [int(side) for side in core_resolved_sides if int(side) in {-1, 1}]
    if not geometry_available or len(resolved_sides) < 2:
        return "unclassified"
    if rear_crossed or len(set(resolved_sides)) == 1:
        return "yield"
    return "ambiguous"


def front_crossing_longitudinal_m(
    previous_longitudinal_m: float | None,
    previous_lateral_m: float | None,
    longitudinal_m: float | None,
    lateral_m: float | None,
    front_margin_m: float,
) -> float | None:
    """Return the front-axis position where a robust left/right crossing occurred.

    Coordinates are expressed in the pedestrian heading frame fixed at event entry.  The
    lateral hysteresis rejects side-to-side noise, while the interpolated zero crossing must
    lie beyond the pedestrian's front clearance margin to count as assertion.
    """
    values = (previous_longitudinal_m, previous_lateral_m, longitudinal_m, lateral_m)
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    lateral_hysteresis = INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M
    changed_sides = (
        previous_lateral_m <= -lateral_hysteresis and lateral_m >= lateral_hysteresis
    ) or (
        previous_lateral_m >= lateral_hysteresis and lateral_m <= -lateral_hysteresis
    )
    if not changed_sides:
        return None
    crossing_fraction = -previous_lateral_m / (lateral_m - previous_lateral_m)
    crossing_longitudinal = previous_longitudinal_m + crossing_fraction * (
        longitudinal_m - previous_longitudinal_m
    )
    return crossing_longitudinal if crossing_longitudinal >= front_margin_m else None


def rear_crossing_longitudinal_m(
    previous_longitudinal_m: float | None,
    previous_lateral_m: float | None,
    longitudinal_m: float | None,
    lateral_m: float | None,
    rear_margin_m: float,
) -> float | None:
    """Return the rear-axis position where a robust left/right crossing occurred."""
    values = (previous_longitudinal_m, previous_lateral_m, longitudinal_m, lateral_m)
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    lateral_hysteresis = INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M
    changed_sides = (
        previous_lateral_m <= -lateral_hysteresis and lateral_m >= lateral_hysteresis
    ) or (
        previous_lateral_m >= lateral_hysteresis and lateral_m <= -lateral_hysteresis
    )
    if not changed_sides:
        return None
    crossing_fraction = -previous_lateral_m / (lateral_m - previous_lateral_m)
    crossing_longitudinal = previous_longitudinal_m + crossing_fraction * (
        longitudinal_m - previous_longitudinal_m
    )
    return crossing_longitudinal if crossing_longitudinal <= -rear_margin_m else None


def front_lateral_side(lateral_m: float | None) -> int:
    """Return the stable pedestrian-frame side, excluding the hysteresis band."""
    if lateral_m is None or not math.isfinite(lateral_m):
        return 0
    if lateral_m <= -INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M:
        return -1
    if lateral_m >= INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M:
        return 1
    return 0


def classify_against_flow_interaction(
    front_crossed: bool, encounter: Mapping[str, Any] | None,
) -> str:
    """Classify an against-flow event from its pre-encounter geometry.

    Front-region lateral traversal intentionally reuses the crossing assertion detector,
    but is reported as ``front_crossing`` rather than assertion.  A sidestep needs both
    an observed forward-cone approach and robot-owned lateral motion; relative lateral
    separation alone could have been caused by the pedestrian.
    """
    if front_crossed:
        return "front_crossing"
    if encounter is None or not bool(encounter["front_cone_qualified"]):
        return "straight_pass"
    passing_side = int(encounter["passing_side"])
    robot_lateral = float(encounter["pass_robot_lateral_displacement_m"])
    if (
        bool(encounter["robust_pass_seen"])
        and passing_side in {-1, 1}
        and abs(robot_lateral) >= INTERACTION_AGAINST_FLOW_SIDESTEP_LATERAL_M
        and (robot_lateral > 0.0) == (passing_side > 0)
    ):
        return "sidestep"
    return "straight_pass"


