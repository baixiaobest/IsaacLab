# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a mixed Go2 policy on the fixed static-plus-dynamic benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

from evaluation import (  # isort: skip
    CollisionReplayRecorder,
    EpisodeMetricsCollector,
    EpisodeVelocityAccumulator,
    GOAL_REGION_COLLISION_RADIUS_M,
    InteractionEventCollector,
    InteractionEventReplayRecorder,
    LeaderOutcomeCollector,
    _json_safe,
    dynamic_crowd_profiles,
    indoor_dynamic_profiles,
    room_corridor_profiles,
    select_scenario_profiles,
    fixed_grid_profile_indices,
    print_leader_outcomes,
    print_results,
    save_artifacts,
    save_interaction_event_artifacts,
    save_leader_outcome_artifacts,
    terminal_collision_ids,
    terminal_goal_region_collision_ids,
)
from evaluation_telemetry import ParquetTelemetryRecorder  # isort: skip
from evaluation_coverage import COVERAGE_TARGETS, coverage_quotas, coverage_rows, subtract_counts  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy on the fixed static-plus-dynamic benchmark.")
parser.add_argument("--task", type=str, required=True, help="Existing mixed obstacle-avoidance task ID.")
parser.add_argument(
    "--benchmark_suite", choices=("static_dynamic", "indoor_dynamic"), default="static_dynamic",
    help="Benchmark terrain/profile suite; enabled scenarios are set in ENABLED_SCENARIOS.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL-agent config entry point.")
parser.add_argument(
    "--num_envs", type=int, default=None,
    help=(
        "Vector environments. Must be a positive multiple of the selected suite's cell count; "
        "defaults to one replica per cell."
    ),
)
parser.add_argument("--seed", type=int, default=42, help="Benchmark random seed.")
parser.add_argument(
    "--seeds", type=int, default=1,
    help=(
        "Number of consecutive benchmark seeds starting at --seed (each contributes an "
        "equal share of --episodes_per_profile, so total runtime is unchanged)."
    ),
)
parser.add_argument(
    "--episodes_per_profile", type=int, default=100,
    help="Total completed episodes per scenario/count cell; sparse-LiDAR tasks split this quota across 40/60/80/100%% coverage.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Evaluation artifact root; each run creates a timestamped subdirectory (defaults under the checkpoint).",
)
parser.add_argument(
    "--failure_history_seconds", type=float, default=3.0,
    help="Seconds of context before a pedestrian collision to retain in each replay.",
)
parser.add_argument(
    "--success_cases_per_scenario",
    type=int,
    default=0,
    help=(
        "Interesting complete successful episodes to save for each scenario; "
        "0 disables success recording (default)."
    ),
)
parser.add_argument(
    "--interesting_interaction_distance_m",
    type=float,
    default=1.5,
    help="A success replay is sampled only if the robot comes within this distance of an active pedestrian.",
)
parser.add_argument(
    "--interaction_event_cases_per_label",
    type=int,
    default=20,
    help="Maximum saved interaction-event clips per scenario and canonical label (0 disables event clips).",
)
parser.add_argument(
    "--interaction_event_padding_seconds",
    type=float,
    default=1.0,
    help="Required context before and after a saved interaction event clip.",
)
parser.add_argument(
    "--failure_output_dir",
    "--replay_output_dir",
    dest="replay_output_dir",
    type=str,
    default=None,
    help="Episode-replay root; each run creates a timestamped subdirectory (defaults to the evaluation run).",
)
parser.add_argument(
    "--telemetry_dir", type=str, default=None,
    help="Structured telemetry output directory. Omit to preserve the legacy file-only evaluator.",
)
parser.add_argument(
    "--disable_failure_recording", action="store_true",
    help="Do not save pedestrian-collision replay artifacts during evaluation.",
)
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the published checkpoint when available."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# RVO2 tasks register their evaluation variant before Hydra resolves the task.
if "RVO2-Crowd" in args_cli.task:
    from isaaclab_tasks.manager_based.navigation.config.go2.rvo2_navigation_eval_mixins import register_rvo2_eval_task
    register_rvo2_eval_task()

# Set a scenario to False to skip all its terrain rows/counts during evaluation.
# room_corridor controls all four room/corridor route types together.
ENABLED_SCENARIOS = {
    "static_dynamic": {
        "static_obstacles": True,
        "crossing": True,
        "with_flow": True,
        "against_flow": True,
        "with_flow_slow_leader": True,
        "crossing_slow": True,
        "against_flow_slow": True,
        "room_corridor": False,
    },
    "indoor_dynamic": {
        "crossing": True,
        "with_flow": True,
        "against_flow": True,
    },
}


from navigation_evaluation.run import run_evaluation  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Run the navigation benchmark through the stable Research Agent entrypoint."""
    return run_evaluation(
        env_cfg, agent_cfg, args_cli=args_cli, simulation_app=simulation_app,
        enabled_scenarios=ENABLED_SCENARIOS,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
