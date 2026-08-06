"""Focused unit tests for the reusable dynamic-crowd evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluation.py")
SPEC = importlib.util.spec_from_file_location("rsl_rl_evaluation", MODULE_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def _extras(completed, success=(), collision=(), velocity=()):
    return {
        "log": {
            "Episode_Termination/Envs/Ids/time_out": list(completed),
            "Episode_Termination/Envs/Ids/goal_reached": list(success),
            "Episode_Termination/Envs/Ids/pedestrian_collision": list(collision),
            "Metrics/pose_2d_command/linear_velocity_xy/Ids": list(completed),
            "Metrics/pose_2d_command/linear_velocity_xy/Envs": list(velocity),
        }
    }


def test_dynamic_profiles_cover_all_scenarios_and_counts():
    profiles = evaluation.dynamic_crowd_profiles()
    assert len(profiles) == 24
    assert [profile.pedestrian_count for profile in profiles[:8]] == list(range(2, 17, 2))
    assert {profile.scenario for profile in profiles} == set(evaluation.SCENARIO_ORDER)


def test_collector_applies_collision_precedence_and_profile_quota():
    profiles = [
        evaluation.BenchmarkProfile("crossing", 2),
        evaluation.BenchmarkProfile("with_flow", 2),
    ]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0, 1], episodes_per_profile=1)

    assert collector.consume(_extras([0, 1], success=[0, 1], collision=[0], velocity=[0.4, 0.8])) == 2
    # Both profiles are full, so later completions are deliberately discarded.
    assert collector.consume(_extras([0, 1], success=[0, 1], velocity=[1.0, 1.0])) == 0

    rows = collector.rows()
    assert rows[0]["successes"] == 0
    assert rows[0]["collisions"] == 1
    assert rows[0]["collision_rate"] == 1.0
    assert rows[0]["mean_xy_speed_mps"] == 0.4
    assert rows[1]["successes"] == 1
    assert rows[1]["collisions"] == 0
    assert collector.complete


def test_collector_accepts_scalar_environment_logs():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=1)
    extras = _extras([0], success=[0], velocity=[0.75])
    extras["log"]["Episode_Termination/Envs/Ids/time_out"] = 0
    extras["log"]["Metrics/pose_2d_command/linear_velocity_xy/Ids"] = 0
    extras["log"]["Metrics/pose_2d_command/linear_velocity_xy/Envs"] = 0.75

    assert collector.consume(extras) == 1
    assert collector.rows()[0]["successes"] == 1
    assert collector.rows()[0]["mean_xy_speed_mps"] == 0.75


def test_collector_reports_sample_standard_deviations():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=2)

    assert collector.consume(_extras([0], success=[0], velocity=[0.4])) == 1
    assert collector.consume(_extras([0], collision=[0], velocity=[0.8])) == 1

    row = collector.rows()[0]
    assert row["success_rate"] == 0.5
    assert row["collision_rate"] == 0.5
    assert math.isclose(row["success_rate_std"], math.sqrt(0.5))
    assert math.isclose(row["collision_rate_std"], math.sqrt(0.5))
    assert math.isclose(row["mean_xy_speed_mps"], 0.6)
    assert math.isclose(row["std_xy_speed_mps"], math.sqrt(0.08))

    aggregate = collector.aggregate_rows()[0]
    assert aggregate["episodes"] == 2
    assert math.isclose(aggregate["std_xy_speed_mps"], math.sqrt(0.08))


def test_collector_falls_back_to_legacy_linear_velocity_metric():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=1)
    extras = _extras([0], velocity=[0.6])
    del extras["log"]["Metrics/pose_2d_command/linear_velocity_xy/Ids"]
    del extras["log"]["Metrics/pose_2d_command/linear_velocity_xy/Envs"]
    extras["log"]["Metrics/pose_2d_command/linear_velocity/Ids"] = [0]
    extras["log"]["Metrics/pose_2d_command/linear_velocity/Envs"] = [0.6]

    assert collector.consume(extras) == 1
    assert collector.velocity_metric_source == "linear_velocity"
    assert collector.rows()[0]["mean_xy_speed_mps"] == 0.6


def test_direct_world_xy_velocity_accumulator_works_without_command_metrics():
    accumulator = evaluation.EpisodeVelocityAccumulator(2)
    accumulator.record_step([1.0, 2.0])
    accumulator.record_step([3.0, 4.0])
    accumulator.record_terminal([5.0, 6.0], [0])

    assert accumulator.completed_means([0]) == {0: 3.0}

    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=1)
    extras = {"log": {"Episode_Termination/Envs/Ids/time_out": [0]}}
    assert collector.consume(extras, accumulator.completed_means([0])) == 1
    assert collector.velocity_metric_source == "direct_world_xy_speed"
    assert collector.rows()[0]["mean_xy_speed_mps"] == 3.0
    accumulator.reset([0])
    accumulator.record_step([7.0, 8.0])
    assert accumulator.completed_means([0]) == {0: 7.0}


def test_completed_ids_match_all_logged_termination_reasons():
    extras = {
        "log": {
            "Episode_Termination/Envs/Ids/time_out": [1],
            "Episode_Termination/Envs/Ids/goal_reached": 2,
            "Episode_Termination/Envs/Ids/pedestrian_collision": [3],
        }
    }
    assert evaluation.completed_environment_ids(extras) == {1, 2, 3}


def test_explicit_done_ids_ignore_idle_scalar_termination_placeholders():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=1)
    idle_extras = {
        "log": {
            "Episode_Termination/Envs/Ids/time_out": 0,
            "Episode_Termination/Envs/Ids/goal_reached": 0,
            "Episode_Termination/Envs/Ids/pedestrian_collision": 0,
        }
    }

    assert collector.consume(idle_extras, completed_env_ids=[]) == 0
    assert collector.total_episodes == 0


def test_artifacts_include_csv_json_and_summary_plot(tmp_path):
    profiles = evaluation.dynamic_crowd_profiles([2])
    collector = evaluation.EpisodeMetricsCollector(profiles, [0, 1, 2], episodes_per_profile=1)
    collector.consume(_extras([0, 1, 2], success=[0, 1], collision=[2], velocity=[0.4, 0.5, 0.6]))
    output = evaluation.save_artifacts(tmp_path, collector.rows(), collector.aggregate_rows(), {"seed": 42})

    assert (output / "dynamic_crowd_results.csv").is_file()
    assert (output / "dynamic_crowd_results.json").is_file()
    assert (output / "dynamic_crowd_summary.png").is_file()
    with (output / "dynamic_crowd_results.json").open(encoding="utf-8") as file:
        results = json.load(file)["results"]
    assert {"success_rate_std", "collision_rate_std", "std_xy_speed_mps"} <= results[0].keys()
