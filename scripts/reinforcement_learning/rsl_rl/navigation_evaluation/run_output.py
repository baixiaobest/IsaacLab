# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation evaluation metadata and run artifact writing."""

from __future__ import annotations

import json
import math

from evaluation import (
    GOAL_REGION_COLLISION_RADIUS_M,
    _json_safe,
    print_leader_outcomes,
    print_results,
    save_artifacts,
    save_interaction_event_artifacts,
    save_leader_outcome_artifacts,
)
from evaluation_coverage import COVERAGE_TARGETS, coverage_rows
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    EVALUATION_CROWD_LATERAL_HEADING_MAX,
    EVALUATION_CROWD_SPEED_RANGE,
    EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD,
    EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD,
    EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS,
    EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD,
)

STATIC_DYNAMIC_GRID_CELLS = 64
INDOOR_DYNAMIC_GRID_CELLS = 12

# Older task branches may omit slow-agent evaluation ranges.
try:
    from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
        EVALUATION_CROWD_SLOW_SPEED_RANGE,
        EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
        EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
        EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
        EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M,
        EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS,
        EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M,
    )
except ImportError:
    EVALUATION_CROWD_SLOW_SPEED_RANGE = None
    EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M = None
    EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS = None
    EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M = None
    EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M = None
    EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS = None
    EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M = None


def write_run_output(
    *, args_cli, agent_cfg, output_dir, output_root, checkpoint, profiles, seeds, seed_count,
    required_cells, coverage_sweep, sweep_quotas, coverage_action_metadata, step_dt_s,
    episode_length_s, collector, leader_records, leader_outcome_collector, leader_summary,
    cbf_solver_summary, stage_counts, accepted_coverage, failure_output_dir, replay_recorder,
    interaction_collector, interaction_replay_recorder, progress_reporter, rvo2_crowd_eval,
    leader_outcomes_available, slow_crowd_available,
) -> None:
    """Write the established benchmark files and complete live progress reporting."""
    if collector.velocity_metric_source == "direct_world_xy_speed":
        print("[INFO] Mean XY speed was measured directly from the robot world-frame velocity.")
    elif collector.velocity_metric_source != "linear_velocity_xy":
        print(
            "[WARN] The task did not export linear_velocity_xy; using the legacy "
            f"{collector.velocity_metric_source} metric on the flat pedestrian corridor."
        )
    rows = collector.rows()
    aggregates = collector.aggregate_rows()
    if not rvo2_crowd_eval:
        replicas = args_cli.num_envs // required_cells
        for row in [*rows, *aggregates]:
            row["terrain_replicas"] = replicas
    leader_outcome_summary = leader_outcome_collector.summary_rows()
    artifact_dir = save_artifacts(
        output_dir,
        rows,
        aggregates,
        {
            "task": args_cli.task,
            "benchmark_suite": args_cli.benchmark_suite,
            "checkpoint": str(checkpoint),
            "seed": agent_cfg.seed,
            "seeds": seeds,
            "seed_count": seed_count,
            "run_id": output_dir.name,
            "output_root": str(output_root),
            "episodes_per_profile": args_cli.episodes_per_profile,
            "coverage_sweep": {
                "targets": list(COVERAGE_TARGETS),
                "quota_by_seed": sweep_quotas,
                "result_file": "coverage_sweep_results.json",
                **coverage_action_metadata,
            } if coverage_sweep else None,
            "success_cases_per_scenario": args_cli.success_cases_per_scenario,
            "interesting_interaction_distance_m": args_cli.interesting_interaction_distance_m,
            "interaction_events": {
                "success_only": True,
                "cases_per_label": args_cli.interaction_event_cases_per_label,
                "padding_seconds": args_cli.interaction_event_padding_seconds,
                "canonical_yield_speed_ratio": 0.70,
                "canonical_assert_speed_ratio": 0.85,
                "crossing_assert": "pedestrian-frame front crossing",
                "crossing_yield": "total planar robot speed reduction without a front crossing",
                "speed_measurement": "robot total planar speed",
            },
            "pedestrian_counts": sorted({profile.pedestrian_count for profile in profiles if profile.pedestrian_count}),
            "static_obstacle_counts": sorted({profile.obstacle_count for profile in profiles if profile.obstacle_count is not None}),
            "scenarios": list(dict.fromkeys(profile.scenario for profile in profiles)),
            "terrain_grid": None if rvo2_crowd_eval else (
                {
                    "columns": 3, "rows": 4, "cells": INDOOR_DYNAMIC_GRID_CELLS,
                    "selected_cells": required_cells, "replicas": args_cli.num_envs // required_cells,
                    "terrain_family": "indoor", "count_levels": [2, 4, 6, 8],
                } if args_cli.benchmark_suite == "indoor_dynamic" else {
                    "columns": 8, "rows": 8, "cells": STATIC_DYNAMIC_GRID_CELLS,
                    "selected_cells": required_cells, "replicas": args_cli.num_envs // required_cells,
                    "static_columns": sorted({
                        profile.terrain_column for profile in profiles if profile.pedestrian_count == 0
                    }),
                    "dynamic_columns": len({
                        profile.terrain_column for profile in profiles if profile.pedestrian_count > 0
                    }),
                    "count_levels": [2, 4, 6, 8, 10, 12, 14, 16],
                }
            ),
            "crowd_speed_range_mps": EVALUATION_CROWD_SPEED_RANGE,
            "leaders": {
                "available": leader_outcomes_available and args_cli.benchmark_suite == "static_dynamic",
                "scenarios": ["with_flow", "with_flow_slow_leader"],
                "pedestrian_counts": sorted({
                    profile.pedestrian_count for profile in profiles
                    if profile.scenario in ("with_flow", "with_flow_slow_leader")
                }),
                "pedestrian_slot": 0,
                "with_flow": {
                    "speed_range_mps": EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS,
                    "start_ahead_range_m": EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M,
                    "lateral_offset_range_m": EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M,
                },
                "with_flow_slow_leader": {
                    "speed_range_mps": EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
                    "start_ahead_range_m": EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
                    "lateral_offset_range_m": EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
                },
                "sampled_conditions_file": "leader_conditions.json",
                "outcomes_file": "leader_outcomes.json",
                "outcome_summary_file": "leader_outcome_summary.csv",
                "outcome_protocol": {
                    "success_only": True,
                    "labels": ["Follow", "Overtake"],
                    "pedestrian_slot": 0,
                    "overtake_margin_m": 0.5,
                    "recycle_handling": "A slot-zero recycle freezes the encounter; only a prior pass is Overtake.",
                },
                "sampled_conditions": leader_summary,
            },
            "slow_crowd": {
                "available": slow_crowd_available and args_cli.benchmark_suite == "static_dynamic",
                "scenarios": ["crossing_slow", "against_flow_slow"],
                "pedestrian_counts": sorted({
                    profile.pedestrian_count for profile in profiles
                    if profile.scenario in ("crossing_slow", "against_flow_slow")
                }),
                "speed_range_mps": EVALUATION_CROWD_SLOW_SPEED_RANGE,
            },
            "crowd_lateral_heading_max_deg": math.degrees(EVALUATION_CROWD_LATERAL_HEADING_MAX),
            "goal_reach_condition": {
                "distance_threshold_m": EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD,
                "heading_error_threshold_deg": math.degrees(EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD),
                "xy_speed_threshold_mps": EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD,
                "stay_for_seconds": EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS,
            },
            "metrics": {
                "success_rate": "goal_reached term; collisions take precedence when simultaneous",
                "navigation_success_rate": "successes divided by episodes outside the terminal-goal buffer",
                "collision_rate": (
                    "pedestrian collisions outside the terminal-goal buffer for dynamic scenarios; "
                    "base-contact static-obstacle collisions for static_obstacles"
                ),
                "goal_region_collision_rate": (
                    f"benchmark collisions within {GOAL_REGION_COLLISION_RADIUS_M:.2f} m of the goal"
                ),
                "all_collision_rate": "all benchmark collisions before goal-region classification",
                "timeout_rate": "episodes terminated by the time_out term",
                "base_contact_rate": "episodes terminated by the base_contact term",
                "mean_xy_speed_mps": "episode-average world-frame horizontal robot speed over all episodes",
            },
            "velocity_metric_source": collector.velocity_metric_source,
            "step_dt_s": step_dt_s,
            "episode_length_s": episode_length_s,
            "cbf_qp_solver": cbf_solver_summary,
            "failure_replays": {
                "enabled": replay_recorder is not None and not args_cli.disable_failure_recording,
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "history_seconds": args_cli.failure_history_seconds if replay_recorder is not None else None,
                "goal_region_radius_m": GOAL_REGION_COLLISION_RADIUS_M,
                "collision_cases": replay_recorder.collision_case_count if replay_recorder is not None else 0,
            },
            "success_replays": {
                "enabled": args_cli.success_cases_per_scenario > 0,
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "cases_per_scenario": args_cli.success_cases_per_scenario,
                "interesting_interaction_distance_m": args_cli.interesting_interaction_distance_m,
                "success_cases": replay_recorder.success_case_count if replay_recorder is not None else 0,
            },
        },
    )
    if coverage_sweep:
        breakdown = coverage_rows(profiles, stage_counts)
        breakdown["measured_coverage"] = [
            {
                "coverage_target": target,
                "accepted_episodes": int(accepted_coverage[target]["episodes"]),
                "mean_terminal_scan_ray_coverage": accepted_coverage[target]["ray_sum"] / max(accepted_coverage[target]["episodes"], 1),
                "mean_terminal_scan_reflected_bin_coverage": accepted_coverage[target]["reflection_sum"] / max(accepted_coverage[target]["episodes"], 1),
            }
            for target in COVERAGE_TARGETS
        ]
        with (artifact_dir / "coverage_sweep_results.json").open("w", encoding="utf-8") as file:
            json.dump(_json_safe(breakdown), file, indent=2, allow_nan=False)
    with (artifact_dir / "leader_conditions.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": 2,
                "scenarios": ["with_flow", "with_flow_slow_leader"],
                "configured_ranges": {
                    "with_flow": {
                        "speed_mps": EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS,
                        "start_ahead_m": EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M,
                        "lateral_offset_m": EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M,
                    },
                    "with_flow_slow_leader": {
                        "speed_mps": EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
                        "start_ahead_m": EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
                        "lateral_offset_m": EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
                    },
                },
                "summary": leader_summary,
                "episodes": leader_records,
            },
            file,
            indent=2,
            allow_nan=False,
        )
    save_interaction_event_artifacts(artifact_dir, interaction_collector.events, interaction_collector.summary_rows())
    save_leader_outcome_artifacts(
        artifact_dir, leader_outcome_collector.outcomes, leader_outcome_summary
    )
    if seed_count > 1:
        per_seed_breakdown = {
            "seeds": seeds,
            "episodes_per_profile": args_cli.episodes_per_profile,
            "per_profile": collector.per_seed_rows(seeds),
            "per_scenario": collector.per_seed_aggregate_rows(seeds),
        }
        with (artifact_dir / "per_seed_aggregates.json").open("w", encoding="utf-8") as file:
            json.dump(_json_safe(per_seed_breakdown), file, indent=2, allow_nan=False)
        print("[INFO] Wrote per-seed breakdown to per_seed_aggregates.json")
    print_results(rows, aggregates)
    print_leader_outcomes(leader_outcome_summary)
    print(f"[INFO] Wrote static-plus-dynamic evaluation artifacts to: {artifact_dir}")
    if replay_recorder is not None:
        print(
            f"[INFO] Wrote {replay_recorder.collision_case_count} collision replay(s) and "
            f"{replay_recorder.success_case_count} complete success replay(s) to: {failure_output_dir}"
        )
    if interaction_replay_recorder is not None:
        print(
            f"[INFO] Wrote {interaction_replay_recorder.case_count} interaction-event replay(s) to: "
            f"{interaction_replay_recorder.output_dir}"
        )
    progress_reporter.report(
        collector.total_episodes,
        seed=seeds[-1],
        seed_index=seed_count,
        status="complete",
        force=True,
    )
    progress_reporter.close()


