# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation benchmark artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    INTERACTION_AGAINST_FLOW_CONE_HALF_ANGLE_RAD,
    INTERACTION_AGAINST_FLOW_CONE_LATERAL_BUFFER_M,
    INTERACTION_AGAINST_FLOW_CONE_PERSISTENCE_S,
    INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M,
    INTERACTION_AGAINST_FLOW_REAR_PASS_MARGIN_M,
    INTERACTION_AGAINST_FLOW_SIDESTEP_LATERAL_M,
    INTERACTION_ASSERT_SPEED_RATIO,
    INTERACTION_ENTER_CLEARANCE_M,
    INTERACTION_EXIT_CLEARANCE_M,
    INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M,
    INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M,
    INTERACTION_LABELS,
    INTERACTION_MIN_BASELINE_SPEED_MPS,
    INTERACTION_MIN_DURATION_S,
    INTERACTION_OVERTAKE_LONGITUDINAL_MARGIN_M,
    INTERACTION_PRE_SPEED_WINDOW_S,
    INTERACTION_RISK_CLEARANCE_M,
    INTERACTION_RISK_HORIZON_S,
    INTERACTION_SCENARIO_ORDER,
    INTERACTION_YIELD_SPEED_RATIO,
    LEADER_OUTCOME_LABELS,
    LEADER_OUTCOME_SCENARIOS,
    LEADER_OVERTAKE_MARGIN_M,
    LEADER_SLOT,
    ROOM_CORRIDOR_SCENARIOS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    _json_safe,
)

def print_results(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    """Print a compact result table without introducing a tabular dependency."""
    header = (
        "scenario        crowd  episodes  success  nav coll  goal coll  all coll  success%  nav success%  "
        "nav coll%  goal coll%  all coll%  timeout  base contact  mean xy speed (m/s) +/- std"
    )
    print(header)
    print("-" * len(header))
    for row in [*rows, *aggregate_rows]:
        print(
            f"{row['scenario']:<15} {str(row['pedestrian_count']):>5} {row['episodes']:>9} "
            f"{row['successes']:>8} {row['collisions']:>9} {row['goal_region_collisions']:>10} "
            f"{row['all_collisions']:>9} {100 * row['success_rate']:>8.1f} "
            f"{100 * row['navigation_success_rate']:>12.1f} "
            f"{100 * row['collision_rate']:>10.1f} {100 * row['goal_region_collision_rate']:>11.1f} "
            f"{100 * row['all_collision_rate']:>10.1f} {row['timeouts']:>8} {row['base_contacts']:>13} "
            f"{row['mean_xy_speed_mps']:>8.3f} "
            f"+/- {row['std_xy_speed_mps']:<.3f}"
        )


def save_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write CSV, JSON, and the static-plus-dynamic summary plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_rows = [*rows, *aggregate_rows]
    fieldnames = [
        "scenario", "terrain_family", "nominal_corridor_width_m", "pedestrian_count", "obstacle_count", "terrain_column", "terrain_level", "terrain_replicas",
        "episodes", "successes", "collisions", "goal_region_collisions",
        "all_collisions", "timeouts", "base_contacts", "success_rate", "navigation_success_rate", "collision_rate",
        "goal_region_collision_rate", "all_collision_rate", "timeout_rate", "base_contact_rate",
        "mean_xy_speed_mps", "std_xy_speed_mps",
    ]
    with (output_path / "dynamic_crowd_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_path / "dynamic_crowd_results.json").open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe({"metadata": metadata, "results": rows, "aggregates": aggregate_rows}),
            file,
            indent=2,
            allow_nan=False,
        )
    _save_summary_plot(output_path / "dynamic_crowd_summary.png", rows)
    _save_failure_histogram(output_path / "dynamic_crowd_failure_histogram.png", aggregate_rows)
    return output_path


def save_interaction_event_artifacts(
    output_dir: str | Path,
    events: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> Path:
    """Write raw success-only event records, canonical totals, and scenario histograms."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    event_fields = [
        "scenario", "pedestrian_count", "environment_id", "pedestrian_id", "canonical_label",
        "start_time_s", "end_time_s", "duration_s", "risk_seen", "minimum_clearance_m",
        "baseline_speed_mps", "low_event_speed_mps", "speed_ratio", "initial_longitudinal_m",
        "final_longitudinal_m", "front_crossed", "front_cross_time_s", "front_cross_longitudinal_m",
        "front_cross_margin_m", "core_sample_count", "core_resolved_side_count", "core_resolved_sides",
        "rear_crossed", "rear_cross_time_s", "rear_cross_longitudinal_m", "rear_cross_margin_m",
        "yield_geometry_available", "yield_speed_ratio", "assert_speed_ratio",
        "encounter_acquisition_time_s", "encounter_pedestrian_direction_xy", "forward_cone_qualified",
        "forward_cone_qualified_time_s", "robot_lateral_displacement_m", "robust_pass_seen",
        "robust_pass_time_s", "passing_side", "pass_lateral_m", "pass_robot_lateral_displacement_m",
    ]
    with (output_path / "interaction_events.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=event_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    with (output_path / "interaction_event_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["scenario", "label", "events"])
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "schema_version": 3,
        "detector": {
            "enter_clearance_m": INTERACTION_ENTER_CLEARANCE_M,
            "exit_clearance_m": INTERACTION_EXIT_CLEARANCE_M,
            "exit_requires_pair_not_closing": True,
            "risk_clearance_m": INTERACTION_RISK_CLEARANCE_M,
            "risk_horizon_s": INTERACTION_RISK_HORIZON_S,
            "pre_speed_window_s": INTERACTION_PRE_SPEED_WINDOW_S,
            "minimum_duration_s": INTERACTION_MIN_DURATION_S,
            "minimum_baseline_speed_mps": INTERACTION_MIN_BASELINE_SPEED_MPS,
            "speed_measurement": "robot total planar speed",
            "yield_speed_ratio": INTERACTION_YIELD_SPEED_RATIO,
            "assert_speed_ratio": INTERACTION_ASSERT_SPEED_RATIO,
            "crossing_assert_definition": "left/right traversal through the pedestrian forward region",
            "front_cross_clearance_margin_m": INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M,
            "front_cross_lateral_hysteresis_m": INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M,
            "crossing_yield_core": "all samples in the active event, including exit hysteresis",
            "crossing_yield_definition": (
                "rear left/right traversal behind the pedestrian or consistent resolved lateral side; "
                "robot speed is diagnostic only"
            ),
            "crossing_unclassified_definition": "missing entry heading or fewer than two resolved core-side samples",
            "against_flow_outcomes": ["sidestep", "straight_pass", "front_crossing"],
            "against_flow_encounter_range_m": INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M,
            "against_flow_cone_full_angle_degrees": math.degrees(2.0 * INTERACTION_AGAINST_FLOW_CONE_HALF_ANGLE_RAD),
            "against_flow_cone_lateral_buffer_m": INTERACTION_AGAINST_FLOW_CONE_LATERAL_BUFFER_M,
            "against_flow_cone_persistence_s": INTERACTION_AGAINST_FLOW_CONE_PERSISTENCE_S,
            "against_flow_sidestep_lateral_m": INTERACTION_AGAINST_FLOW_SIDESTEP_LATERAL_M,
            "against_flow_rear_pass_margin_m": INTERACTION_AGAINST_FLOW_REAR_PASS_MARGIN_M,
            "against_flow_sidestep_definition": (
                "persistent pedestrian-forward-cone approach, then same-side robot lateral motion "
                "of at least the sidestep threshold at a robust rear pass"
            ),
            "against_flow_front_crossing_definition": "existing active-event front-region lateral crossing",
            "against_flow_straight_pass_definition": "all remaining accepted against-flow interactions",
            "overtake_longitudinal_margin_m": INTERACTION_OVERTAKE_LONGITUDINAL_MARGIN_M,
        },
        "labels": {scenario: list(labels) for scenario, labels in INTERACTION_LABELS.items()},
        "events": events,
        "summary": summary_rows,
    }
    with (output_path / "interaction_events.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
    _save_interaction_histogram(output_path / "interaction_event_histogram.png", summary_rows)
    return output_path


def save_leader_outcome_artifacts(
    output_dir: str | Path,
    outcomes: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> Path:
    """Write success-only slot-zero Follow/Overtake outcomes for both leader variants."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outcome_fields = [
        "scenario", "pedestrian_count", "environment_id", "seed", "pedestrian_id", "outcome",
        "start_time_s", "end_time_s", "duration_s", "pass_time_s", "minimum_clearance_m",
        "initial_longitudinal_m", "final_longitudinal_m", "leader_recycled", "speed_mps",
        "start_ahead_m", "lateral_offset_m",
    ]
    with (output_path / "leader_outcomes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=outcome_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outcomes)
    summary_fields = [
        "scenario", "pedestrian_count", "successful_episodes", "follow", "overtake", "follow_rate", "overtake_rate",
    ]
    with (output_path / "leader_outcome_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_path / "leader_outcomes.json").open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(
                {
                    "schema_version": 2,
                    "scenarios": list(LEADER_OUTCOME_SCENARIOS),
                    "pedestrian_slot": LEADER_SLOT,
                    "overtake_margin_m": LEADER_OVERTAKE_MARGIN_M,
                    "labels": list(LEADER_OUTCOME_LABELS),
                    "outcomes": outcomes,
                    "summary": summary_rows,
                }
            ),
            file,
            indent=2,
            allow_nan=False,
        )
    _save_leader_outcome_plot(output_path / "leader_outcomes.png", summary_rows)
    return output_path


def print_leader_outcomes(summary_rows: list[dict[str, Any]]) -> None:
    """Print the episode-level designated-leader outcome table when profiles are enabled."""
    if not summary_rows:
        return
    print("leader outcomes       crowd  successful  follow  overtake  follow%  overtake%")
    print("-" * 76)
    for row in summary_rows:
        follow_rate = row["follow_rate"]
        overtake_rate = row["overtake_rate"]
        follow_percent = "n/a" if follow_rate is None else f"{100.0 * follow_rate:.1f}"
        overtake_percent = "n/a" if overtake_rate is None else f"{100.0 * overtake_rate:.1f}"
        print(
            f"{row['scenario']:<21} {row['pedestrian_count']:>5} "
            f"{row['successful_episodes']:>11} {row['follow']:>7} {row['overtake']:>9} "
            f"{follow_percent:>7} {overtake_percent:>10}"
        )


def _save_interaction_histogram(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {(row["scenario"], row["label"]): int(row["events"]) for row in summary_rows}
    figure, axes = plt.subplots(
        1, len(INTERACTION_SCENARIO_ORDER), figsize=(4.5 * len(INTERACTION_SCENARIO_ORDER), 4.5)
    )
    axes = np.atleast_1d(axes)
    for axis, scenario in zip(axes, INTERACTION_SCENARIO_ORDER):
        labels = INTERACTION_LABELS[scenario]
        values = [summary.get((scenario, label), 0) for label in labels]
        bars = axis.bar(labels, values, color=["#60a5fa", "#f87171", "#a78bfa", "#94a3b8", "#64748b"][:len(labels)])
        axis.bar_label(bars, padding=3)
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Successful-episode events")
    figure.suptitle("Interaction-event categories (canonical thresholds)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_leader_outcome_plot(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    """Save compact Follow/Overtake count charts by leader scenario and crowd count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = [scenario for scenario in LEADER_OUTCOME_SCENARIOS if any(row["scenario"] == scenario for row in summary_rows)]
    figure, axes = plt.subplots(1, max(1, len(scenarios)), figsize=(8 * max(1, len(scenarios)), 4.5), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        rows = [row for row in summary_rows if row["scenario"] == scenario]
        counts = [int(row["pedestrian_count"]) for row in rows]
        follows = [int(row["follow"]) for row in rows]
        overtakes = [int(row["overtake"]) for row in rows]
        x = np.arange(len(counts))
        width = 0.38
        follow_bars = axis.bar(x - width / 2.0, follows, width, label="Follow", color="#60a5fa")
        overtake_bars = axis.bar(x + width / 2.0, overtakes, width, label="Overtake", color="#f59e0b")
        axis.bar_label(follow_bars, padding=3)
        axis.bar_label(overtake_bars, padding=3)
        axis.set_xticks(x, [str(count) for count in counts])
        axis.set_xlabel("Pedestrians")
        axis.set_ylabel("Successful episodes")
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


# Compatibility exports for out-of-tree users. New evaluations write only the
# generic leader artifacts; ResearchAgent reads both old and new forms.
save_slow_leader_outcome_artifacts = save_leader_outcome_artifacts
print_slow_leader_outcomes = print_leader_outcomes


def _save_summary_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save metric-by-scenario facets with crowd count on every x-axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = (
        ("success_rate", None, "Success rate (%)", 100.0, (0.0, 100.0)),
        ("navigation_success_rate", None, "Navigation success rate (%)", 100.0, (0.0, 100.0)),
        ("collision_rate", None, "Navigation collision rate (%)", 100.0, (0.0, 100.0)),
        ("mean_xy_speed_mps", "std_xy_speed_mps", "Mean XY speed (m/s)", 1.0, None),
    )
    figure, axes = plt.subplots(
        len(metric_specs), len(SCENARIO_ORDER), figsize=(4.5 * len(SCENARIO_ORDER), 13), sharex="col"
    )
    for col, scenario in enumerate(SCENARIO_ORDER):
        is_room = scenario in ROOM_CORRIDOR_SCENARIOS
        scenario_rows = sorted(
            (row for row in rows if row["scenario"] == scenario),
            key=lambda row: row["nominal_corridor_width_m"] if is_room else row["pedestrian_count"],
        )
        crowd_counts = [row["nominal_corridor_width_m"] if is_room else row["pedestrian_count"] for row in scenario_rows]
        for row_index, (metric, std_metric, ylabel, scale, ylim) in enumerate(metric_specs):
            axis = axes[row_index, col]
            values = [row[metric] * scale for row in scenario_rows]
            axis.plot(crowd_counts, values, marker="o", linewidth=2)
            if std_metric is not None:
                standard_deviations = [row[std_metric] * scale for row in scenario_rows]
                lower = [value - standard_deviation for value, standard_deviation in zip(values, standard_deviations)]
                upper = [value + standard_deviation for value, standard_deviation in zip(values, standard_deviations)]
                if ylim is not None:
                    lower = [max(ylim[0], value) for value in lower]
                    upper = [min(ylim[1], value) for value in upper]
                axis.fill_between(crowd_counts, lower, upper, alpha=0.2)
            axis.grid(True, alpha=0.3)
            if ylim is not None:
                axis.set_ylim(*ylim)
            if row_index == 0:
                axis.set_title(SCENARIO_LABELS[scenario])
            if col == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(metric_specs) - 1:
                axis.set_xlabel("Pedestrians")
    figure.suptitle("Dynamic crowd evaluation (speed shaded: ±1 sample SD)", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_failure_histogram(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    """Save one failure-type histogram for each scenario.

    Pedestrian collisions in the terminal-goal buffer are intentionally omitted because the
    ``collisions`` count already represents navigation collisions only.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    failure_specs = (
        ("timeouts", "Timeout", "#4C78A8"),
        ("collisions", "Agent collision", "#E45756"),
        ("base_contacts", "Base contact", "#F2A541"),
    )
    aggregate_by_scenario = {row["scenario"]: row for row in aggregate_rows}
    figure, axes = plt.subplots(
        1, len(SCENARIO_ORDER), figsize=(4.5 * len(SCENARIO_ORDER), 4.5), sharey=True
    )
    for axis, scenario in zip(axes, SCENARIO_ORDER):
        row = aggregate_by_scenario.get(scenario, {})
        labels = [label for _, label, _ in failure_specs]
        values = [int(row.get(metric, 0)) for metric, _, _ in failure_specs]
        bars = axis.bar(labels, values, color=[color for _, _, color in failure_specs])
        axis.bar_label(bars, padding=3)
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Completed episodes")
    figure.suptitle("Dynamic crowd failures (goal-region agent collisions excluded)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=180)
    plt.close(figure)
