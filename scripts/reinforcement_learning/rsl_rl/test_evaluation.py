"""Focused unit tests for the reusable dynamic-crowd evaluation helpers."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = Path(__file__).with_name("evaluation.py")
EVALUATE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("rsl_rl_evaluation", MODULE_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)
sys.modules["evaluation"] = evaluation

VIEWER_PATH = Path(__file__).with_name("view_failure_cases.py")
VIEWER_SPEC = importlib.util.spec_from_file_location("rsl_rl_failure_viewer", VIEWER_PATH)
assert VIEWER_SPEC and VIEWER_SPEC.loader
failure_viewer = importlib.util.module_from_spec(VIEWER_SPEC)
sys.modules[VIEWER_SPEC.name] = failure_viewer
VIEWER_SPEC.loader.exec_module(failure_viewer)

TORCH_AVAILABLE = torch is not None and hasattr(torch, "zeros")


class _FakeRobotData:
    def __init__(self, num_envs):
        self.root_pos_w = torch.zeros(num_envs, 3)
        self.root_lin_vel_w = torch.zeros(num_envs, 3)
        self.heading_w = torch.zeros(num_envs)


class _FakeCrowd:
    class _Cfg:
        robot_radius = 0.4

    def __init__(self, num_envs, max_pedestrians):
        self.max_pedestrians = max_pedestrians
        self.cfg = self._Cfg()
        self.pos = torch.zeros(num_envs, max_pedestrians, 2)
        self.vel = torch.zeros(num_envs, max_pedestrians, 2)
        self.active = torch.zeros(num_envs, max_pedestrians, dtype=torch.bool)
        self.radius = torch.full((num_envs, max_pedestrians), 0.25)

    def get_world_positions(self):
        return self.pos

    def get_velocities(self):
        return self.vel

    def get_active_mask(self):
        return self.active

    def get_robot_collision(self, robot_positions):
        distance = torch.linalg.vector_norm(self.pos - robot_positions.unsqueeze(1), dim=-1)
        return torch.any((distance < self.radius + self.cfg.robot_radius) & self.active, dim=1)


class _FakeEnv:
    def __init__(self, num_envs=2, max_pedestrians=3):
        self.num_envs = num_envs
        self.device = "cpu"
        self.scene = {"robot": type("Robot", (), {"data": _FakeRobotData(num_envs)})()}
        self.crowd_manager = _FakeCrowd(num_envs, max_pedestrians)
        self.command = type("Command", (), {"pos_command_w": torch.zeros(num_envs, 3)})()
        self.command_manager = type(
            "CommandManager",
            (), {"get_term": lambda _, __: self.command},
        )()


def _extras(completed, success=(), collision=(), base_contact=(), velocity=()):
    return {
        "log": {
            "Episode_Termination/Envs/Ids/time_out": list(completed),
            "Episode_Termination/Envs/Ids/goal_reached": list(success),
            "Episode_Termination/Envs/Ids/pedestrian_collision": list(collision),
            "Episode_Termination/Envs/Ids/base_contact": list(base_contact),
            "Metrics/pose_2d_command/linear_velocity_xy/Ids": list(completed),
            "Metrics/pose_2d_command/linear_velocity_xy/Envs": list(velocity),
        }
    }


def test_dynamic_profiles_cover_all_scenarios_and_counts():
    profiles = evaluation.dynamic_crowd_profiles()
    assert len(profiles) == 24
    assert [profile.pedestrian_count for profile in profiles[:8]] == list(range(2, 17, 2))
    assert {profile.scenario for profile in profiles} == set(evaluation.SCENARIO_ORDER)


def test_speed_interaction_labels_cover_yield_assert_ambiguous_and_non_risky():
    label, low_speed, ratio = evaluation.classify_speed_interaction(
        "crossing", True, 0.5, 1.0, [0.2, 0.3, 0.4]
    )
    assert (label, low_speed, ratio) == ("yield", pytest.approx(0.22), pytest.approx(0.22))
    assert evaluation.classify_speed_interaction("against_flow", True, 0.5, 1.0, [0.95, 1.0])[0] == "assert"
    assert evaluation.classify_speed_interaction("crossing", True, 0.5, 1.0, [0.75, 0.8])[0] == "ambiguous"
    assert evaluation.classify_speed_interaction("crossing", False, 0.5, 1.0, [0.1])[0] == "non_risky_close"
    assert evaluation.classify_speed_interaction("crossing", True, 0.1, 1.0, [0.1])[0] == "unclassified"


def test_crossing_assert_requires_pedestrian_frame_front_crossing():
    assert evaluation.classify_speed_interaction(
        "crossing", True, 0.5, 1.0, [0.1, 0.2], front_crossed=True
    ) == ("assert", None, None)
    # Maintaining speed alone is no longer assertion in a crossing scenario.
    assert evaluation.classify_speed_interaction("crossing", True, 0.5, 1.0, [0.95, 1.0])[0] == "ambiguous"
    assert evaluation.front_crossing_longitudinal_m(0.8, -0.4, 1.0, 0.4, 0.6) == pytest.approx(0.9)
    assert evaluation.front_crossing_longitudinal_m(-0.8, -0.4, 0.2, 0.4, 0.6) is None
    assert evaluation.front_crossing_longitudinal_m(1.0, -0.1, 1.0, 0.1, 0.6) is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_crossing_event_collector_detects_front_region_side_to_side_assertion():
    collector = evaluation.InteractionEventCollector([evaluation.BenchmarkProfile("crossing", 2)], [0], step_dt_s=0.1)
    env = _FakeEnv(num_envs=1)
    env.crowd_manager.active[0, 0] = True
    env.crowd_manager.vel[0, 0] = torch.tensor([0.5, 0.0])
    env.scene["robot"].data.root_lin_vel_w[0, :2] = torch.tensor([-1.0, 0.0])

    # The pedestrian moves in +x; the robot switches from its right to left side while
    # remaining one metre in front, then exits the interaction.
    env.scene["robot"].data.root_pos_w[0, :2] = torch.tensor([1.0, -0.4])
    collector.record_pre_step(env)
    # A physical crossing takes multiple steps through the hysteresis band.
    env.scene["robot"].data.root_pos_w[0, :2] = torch.tensor([1.0, -0.1])
    collector.record_pre_step(env)
    env.scene["robot"].data.root_pos_w[0, :2] = torch.tensor([1.0, 0.1])
    collector.record_pre_step(env)
    env.scene["robot"].data.root_pos_w[0, :2] = torch.tensor([1.0, 0.4])
    collector.record_pre_step(env)
    env.scene["robot"].data.root_pos_w[0, :2] = torch.tensor([3.0, 0.4])
    collector.record_pre_step(env)

    event = collector._completed[0][0]
    assert event["front_crossed"]
    assert event["canonical_label"] == "assert"
    assert event["front_cross_longitudinal_m"] == pytest.approx(1.0)


def test_with_flow_interaction_overtake_and_ordering_labels():
    assert evaluation.classify_speed_interaction("with_flow", True, 1.0, 1.0, [], -0.6, 0.6)[0] == "overtake"
    assert evaluation.classify_speed_interaction("with_flow", True, 1.0, 1.0, [], -0.6, -0.1)[0] == "non_overtake"
    assert evaluation.classify_speed_interaction("with_flow", True, 1.0, 1.0, [], 0.1, 0.6)[0] == "unclassified"


def test_interaction_artifacts_include_zero_categories_and_raw_events(tmp_path):
    event = {
        "scenario": "crossing", "pedestrian_count": 2, "environment_id": 0, "pedestrian_id": 1,
        "canonical_label": "yield", "start_time_s": 1.0, "end_time_s": 1.5, "duration_s": 0.5,
        "risk_seen": True, "minimum_clearance_m": 0.2, "baseline_speed_mps": 1.0,
        "low_event_speed_mps": 0.2, "speed_ratio": 0.2, "initial_longitudinal_m": None,
        "final_longitudinal_m": None, "yield_speed_ratio": 0.7, "assert_speed_ratio": 0.85,
    }
    summary = [
        {"scenario": scenario, "label": label, "events": int(scenario == "crossing" and label == "yield")}
        for scenario, labels in evaluation.INTERACTION_LABELS.items() for label in labels
    ]
    evaluation.save_interaction_event_artifacts(tmp_path, [event], summary)
    payload = json.loads((tmp_path / "interaction_events.json").read_text(encoding="utf-8"))
    assert payload["events"] == [event]
    assert any(row["label"] == "unclassified" and row["events"] == 0 for row in payload["summary"])
    assert (tmp_path / "interaction_event_histogram.png").is_file()


def test_interaction_artifacts_encode_missing_speed_baseline_as_json_null(tmp_path):
    event = {
        "scenario": "crossing", "pedestrian_count": 2, "environment_id": 0, "pedestrian_id": 1,
        "canonical_label": "unclassified", "start_time_s": 0.0, "end_time_s": 0.2, "duration_s": 0.2,
        "risk_seen": True, "minimum_clearance_m": 0.2, "baseline_speed_mps": float("nan"),
        "low_event_speed_mps": None, "speed_ratio": None, "initial_longitudinal_m": None,
        "final_longitudinal_m": None, "yield_speed_ratio": 0.7, "assert_speed_ratio": 0.85,
    }
    summary = [{"scenario": scenario, "label": label, "events": 0}
               for scenario, labels in evaluation.INTERACTION_LABELS.items() for label in labels]
    evaluation.save_interaction_event_artifacts(tmp_path, [event], summary)

    payload = json.loads(
        (tmp_path / "interaction_events.json").read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert payload["events"][0]["baseline_speed_mps"] is None


def test_interaction_preset_round_trip(tmp_path):
    failure_viewer.save_interaction_presets(
        tmp_path, {"conservative": {"yield_speed_ratio": 0.75, "assert_speed_ratio": 0.9}}
    )
    assert failure_viewer.load_interaction_presets(tmp_path)["conservative"]["yield_speed_ratio"] == 0.75


def test_interaction_collector_only_admits_counted_successes():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.InteractionEventCollector(profiles, [0], step_dt_s=0.1)
    event = {
        "scenario": "crossing", "pedestrian_id": 1, "start_time_s": 0.0, "end_time_s": 0.4,
        "duration_s": 0.4, "risk_seen": True, "minimum_clearance_m": 0.2,
        "baseline_speed_mps": 1.0, "low_event_speed_mps": 0.2, "speed_ratio": 0.2,
        "initial_longitudinal_m": None, "final_longitudinal_m": None, "canonical_label": "yield",
        "yield_speed_ratio": 0.7, "assert_speed_ratio": 0.85,
    }
    collector._completed[0] = [event]
    collector.finalize_terminal([0])
    assert collector.resolve_terminal([0], []) == []
    assert not collector.events
    collector._completed[0] = [event]
    collector.finalize_terminal([0])
    admitted = collector.resolve_terminal([0], [0])
    assert len(admitted) == 1
    assert collector.summary_rows()[0]["events"] == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_interaction_replay_adds_profile_metadata_before_event_admission(tmp_path):
    """Terminal-staged events do not yet have the collector's admission metadata."""
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    source = evaluation.CollisionReplayRecorder(profiles, [0], tmp_path, step_dt_s=0.1, history_seconds=1.0)
    recorder = evaluation.InteractionEventReplayRecorder(tmp_path / "interaction_events", source, 1, 0.1)
    env = _FakeEnv(num_envs=1)
    for step in range(4):
        env.scene["robot"].data.root_pos_w[0, 0] = float(step)
        source.record_pre_step(env, torch.zeros(1, 3))
    event = {
        "scenario": "crossing", "pedestrian_id": 0, "canonical_label": "yield",
        "start_time_s": 0.0, "end_time_s": 0.2, "duration_s": 0.2,
        "minimum_clearance_m": 0.2, "baseline_speed_mps": 1.0,
        "low_event_speed_mps": 0.2, "speed_ratio": 0.2,
        "yield_speed_ratio": 0.7, "assert_speed_ratio": 0.85,
    }

    recorder.stage_terminal_success(env, 0, [event])
    recorder.resolve_terminal([0], [0])

    index = json.loads((tmp_path / "interaction_events" / "interaction_event_cases.json").read_text())
    assert index["cases"][0]["pedestrian_count"] == 2
    assert index["cases"][0]["environment_id"] == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_interaction_replay_samples_non_primary_canonical_labels(tmp_path):
    """Every canonical label is eligible for a separately quota-limited replay."""
    profiles = [evaluation.BenchmarkProfile("with_flow", 2)]
    source = evaluation.CollisionReplayRecorder(profiles, [0], tmp_path, step_dt_s=0.1, history_seconds=1.0)
    recorder = evaluation.InteractionEventReplayRecorder(tmp_path / "interaction_events", source, 1, 0.1)
    env = _FakeEnv(num_envs=1)
    for step in range(4):
        env.scene["robot"].data.root_pos_w[0, 0] = float(step)
        source.record_pre_step(env, torch.zeros(1, 3))
    event = {
        "scenario": "with_flow", "pedestrian_id": 0, "canonical_label": "non_overtake",
        "start_time_s": 0.0, "end_time_s": 0.2, "duration_s": 0.2,
        "minimum_clearance_m": 0.2, "baseline_speed_mps": 1.0,
        "low_event_speed_mps": 1.0, "speed_ratio": 1.0,
        "yield_speed_ratio": 0.7, "assert_speed_ratio": 0.85,
    }

    recorder.stage_terminal_success(env, 0, [event])
    recorder.resolve_terminal([0], [0])

    index = json.loads((tmp_path / "interaction_events" / "interaction_event_cases.json").read_text())
    assert index["cases"][0]["canonical_label"] == "non_overtake"


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


def test_collector_reports_speed_sample_standard_deviation_only():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=2)

    assert collector.consume(_extras([0], success=[0], velocity=[0.4])) == 1
    assert collector.consume(_extras([0], collision=[0], velocity=[0.8])) == 1

    row = collector.rows()[0]
    assert row["success_rate"] == 0.5
    assert row["collision_rate"] == 0.5
    assert "success_rate_std" not in row
    assert "collision_rate_std" not in row
    assert math.isclose(row["mean_xy_speed_mps"], 0.6)
    assert math.isclose(row["std_xy_speed_mps"], math.sqrt(0.08))

    aggregate = collector.aggregate_rows()[0]
    assert aggregate["episodes"] == 2
    assert math.isclose(aggregate["std_xy_speed_mps"], math.sqrt(0.08))


def test_collector_separates_goal_region_collisions_from_navigation_collisions():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=2)

    assert collector.consume(
        _extras([0], collision=[0], velocity=[0.4]), goal_region_collision_env_ids=[0]
    ) == 1
    assert collector.consume(_extras([0], collision=[0], velocity=[0.4])) == 1

    row = collector.rows()[0]
    assert row["collisions"] == 1
    assert row["goal_region_collisions"] == 1
    assert row["all_collisions"] == 2
    assert row["collision_rate"] == 0.5
    assert row["goal_region_collision_rate"] == 0.5
    assert row["all_collision_rate"] == 1.0


def test_navigation_success_rate_excludes_goal_region_collision_episodes():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=2)

    collector.consume(_extras([0], success=[0], velocity=[0.4]))
    collector.consume(_extras([0], collision=[0], velocity=[0.4]), goal_region_collision_env_ids=[0])

    row = collector.rows()[0]
    assert row["success_rate"] == 0.5
    assert row["navigation_success_rate"] == 1.0


def test_failure_counts_include_timeouts_and_base_contacts_but_exclude_goal_region_agent_collisions():
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    collector = evaluation.EpisodeMetricsCollector(profiles, [0], episodes_per_profile=3)

    collector.consume(_extras([0], collision=[0], velocity=[0.4]), goal_region_collision_env_ids=[0])
    collector.consume(_extras([0], collision=[0], velocity=[0.4]))
    collector.consume(_extras([0], base_contact=[0], velocity=[0.4]))

    row = collector.rows()[0]
    assert row["timeouts"] == 3
    assert row["collisions"] == 1
    assert row["goal_region_collisions"] == 1
    assert row["base_contacts"] == 1


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
    assert (output / "dynamic_crowd_failure_histogram.png").is_file()
    with (output / "dynamic_crowd_results.json").open(encoding="utf-8") as file:
        results = json.load(file)["results"]
    assert "std_xy_speed_mps" in results[0]
    assert "success_rate_std" not in results[0]
    assert "collision_rate_std" not in results[0]
    assert "timeouts" in results[0]
    assert "base_contacts" in results[0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_collision_replay_keeps_ordered_history_terminal_state_and_active_slots(tmp_path):
    profiles = [evaluation.BenchmarkProfile("crossing", 2), evaluation.BenchmarkProfile("with_flow", 4)]
    recorder = evaluation.CollisionReplayRecorder(profiles, [0, 1], tmp_path, step_dt_s=0.1, history_seconds=0.3)
    env = _FakeEnv()
    env.crowd_manager.active[0, 0] = True
    env.crowd_manager.active[1, 0] = True

    for step in range(4):
        env.scene["robot"].data.root_pos_w[:, 0] = float(step)
        env.scene["robot"].data.root_lin_vel_w[:, 0] = float(step)
        env.scene["robot"].data.heading_w[:] = 0.1 * step
        env.crowd_manager.pos[:, 0, 0] = 10.0
        recorder.record_pre_step(env, torch.tensor([[float(step), 0.0, 0.1], [0.0, 0.0, 0.0]]))

    # Only env 0 collides on its terminal state; env 1 resets without a replay.
    env.scene["robot"].data.root_pos_w[0, 0] = 4.0
    env.crowd_manager.pos[0, 0] = torch.tensor([4.0, 0.0])
    entries = recorder.capture_terminal_collisions(env, torch.tensor([0, 1]))

    assert len(entries) == 1
    entry = entries[0]
    assert entry["scenario"] == "crossing"
    assert entry["pedestrian_count"] == 2
    assert entry["colliding_agent_ids"] == [0]
    assert entry["frame_count"] == 4
    with np.load(tmp_path / entry["replay_file"], allow_pickle=False) as replay:
        assert np.allclose(replay["time_s"], [0.1, 0.2, 0.3, 0.4])
        assert np.allclose(replay["robot_position_xy"][:, 0], [1.0, 2.0, 3.0, 4.0])
        assert replay["pedestrian_active_mask"].shape == (4, 3)
        assert np.all(replay["pedestrian_active_mask"][:, 0])
        assert not np.any(replay["pedestrian_active_mask"][:, 1:])
    with (tmp_path / "failure_cases.json").open(encoding="utf-8") as file:
        assert json.load(file)["cases"] == [entry]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_goal_region_collision_ids_and_replay_automatic_tag(tmp_path):
    env = _FakeEnv(num_envs=1)
    env.crowd_manager.active[0, 0] = True
    env.crowd_manager.pos[0, 0] = torch.tensor([0.0, 0.0])
    env.command.pos_command_w[0, :2] = torch.tensor([0.5, 0.0])

    assert evaluation.terminal_goal_region_collision_ids(env, torch.tensor([0]), radius_m=0.75) == {0}
    assert evaluation.terminal_goal_region_collision_ids(env, torch.tensor([0]), radius_m=0.25) == set()

    recorder = evaluation.CollisionReplayRecorder(
        [evaluation.BenchmarkProfile("crossing", 2)], [0], tmp_path, step_dt_s=0.1
    )
    recorder.record_pre_step(env, torch.zeros(1, 3))
    entry = recorder.capture_terminal_collisions(env, torch.tensor([0]))[0]
    assert entry["goal_region_collision"]
    assert entry["automatic_tags"] == [evaluation.GOAL_REGION_TAG]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_collision_replay_writes_empty_index_and_skips_non_colliding_resets(tmp_path):
    recorder = evaluation.CollisionReplayRecorder(
        [evaluation.BenchmarkProfile("crossing", 2)], [0], tmp_path, step_dt_s=0.1
    )
    env = _FakeEnv(num_envs=1)
    env.crowd_manager.active[0, 0] = True
    env.crowd_manager.pos[0, 0] = torch.tensor([10.0, 0.0])
    recorder.record_pre_step(env, torch.zeros(1, 3))

    assert recorder.capture_terminal_collisions(env, torch.tensor([0])) == []
    assert recorder.case_count == 0
    with (tmp_path / "failure_cases.json").open(encoding="utf-8") as file:
        assert json.load(file)["cases"] == []


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_success_replays_keep_the_complete_episode_and_obey_scenario_quotas(tmp_path):
    profiles = [evaluation.BenchmarkProfile("crossing", 2)]
    recorder = evaluation.CollisionReplayRecorder(
        profiles,
        [0],
        tmp_path,
        step_dt_s=0.1,
        history_seconds=0.2,
        successes_per_scenario=1,
        episode_length_s=1.0,
    )
    env = _FakeEnv(num_envs=1)
    env.crowd_manager.active[0, 0] = True
    for step in range(4):
        env.scene["robot"].data.root_pos_w[0, 0] = float(step)
        env.crowd_manager.pos[0, 0, 0] = float(step) + 1.4
        recorder.record_pre_step(env, torch.zeros(1, 3))

    env.scene["robot"].data.root_pos_w[0, 0] = 4.0
    env.crowd_manager.pos[0, 0, 0] = 5.4
    entries = recorder.capture_terminal_episodes(env, torch.tensor([0]), success_env_ids=torch.tensor([0]))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["case_id"] == "success_000001"
    assert entry["outcome"] == "success"
    assert entry["full_episode"]
    assert entry["automatic_tags"] == [evaluation.INTERESTING_INTERACTION_TAG]
    assert math.isclose(entry["minimum_agent_distance_m"], 1.4, abs_tol=1e-6)
    assert entry["frame_count"] == 5
    assert recorder.success_case_count == 1
    assert recorder.success_recording_complete
    with np.load(tmp_path / entry["replay_file"], allow_pickle=False) as replay:
        assert np.allclose(replay["time_s"], [0.0, 0.1, 0.2, 0.3, 0.4])
        assert np.allclose(replay["robot_position_xy"][:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])

    recorder.record_pre_step(env, torch.zeros(1, 3))
    assert recorder.capture_terminal_episodes(env, torch.tensor([0]), success_env_ids=torch.tensor([0])) == []
    assert recorder.success_case_count == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="The active Isaac Sim Python environment has no PyTorch installation.")
def test_success_replays_require_interaction_and_share_one_quota_per_scenario(tmp_path):
    recorder = evaluation.CollisionReplayRecorder(
        [evaluation.BenchmarkProfile("crossing", 2), evaluation.BenchmarkProfile("crossing", 4)],
        [0, 1],
        tmp_path,
        step_dt_s=0.1,
        successes_per_scenario=1,
        episode_length_s=1.0,
        interesting_interaction_distance_m=1.5,
    )
    env = _FakeEnv(num_envs=2)
    env.crowd_manager.active[:, 0] = True
    env.crowd_manager.pos[0, 0, 0] = 1.4
    env.crowd_manager.pos[1, 0, 0] = 1.4
    recorder.record_pre_step(env, torch.zeros(2, 3))

    entries = recorder.capture_terminal_episodes(env, torch.tensor([0, 1]), success_env_ids=torch.tensor([0, 1]))
    assert [entry["case_id"] for entry in entries] == ["success_000001"]
    assert entries[0]["pedestrian_count"] == 2
    assert recorder.success_recording_complete

    uninteresting = evaluation.CollisionReplayRecorder(
        [evaluation.BenchmarkProfile("against_flow", 2)],
        [0],
        tmp_path / "uninteresting",
        step_dt_s=0.1,
        successes_per_scenario=1,
        episode_length_s=1.0,
    )
    uninteresting_env = _FakeEnv(num_envs=1)
    uninteresting_env.crowd_manager.active[0, 0] = True
    uninteresting_env.crowd_manager.pos[0, 0, 0] = 1.6
    uninteresting.record_pre_step(uninteresting_env, torch.zeros(1, 3))
    assert uninteresting.capture_terminal_episodes(
        uninteresting_env, torch.tensor([0]), success_env_ids=torch.tensor([0])
    ) == []
    assert not uninteresting.success_recording_complete


def test_failure_viewer_filters_tags_and_rotates_body_commands(tmp_path):
    tags = {"collision_000001": ["late brake", "crossing"]}
    failure_viewer.save_case_tags(tmp_path, tags)
    assert failure_viewer.load_case_tags(tmp_path) == tags
    cases = [
        {
            "case_id": "collision_000001",
            "scenario": "crossing",
            "pedestrian_count": 2,
            "automatic_tags": ["goal-region"],
        },
        {"case_id": "collision_000002", "scenario": "with_flow", "pedestrian_count": 4},
    ]
    assert failure_viewer.filter_cases(cases, tags, scenario="crossing", tag_filter="late brake") == [cases[0]]
    assert failure_viewer.filter_cases(cases, tags, tag_filter="goal-region") == [cases[0]]
    assert failure_viewer.filter_cases(cases, tags, exclude_tag="goal-region") == [cases[1]]
    assert failure_viewer.available_tags(cases, tags) == ["crossing", "goal-region", "late brake"]
    assert failure_viewer.filter_cases(cases, tags, tag_filter="missing") == []
    assert np.allclose(failure_viewer.body_velocity_to_world(np.array([1.0, 0.0]), np.pi / 2), [0.0, 1.0])


def test_failure_viewer_discovers_and_selects_timestamped_evaluation_runs(tmp_path):
    evaluation_root = tmp_path / "dynamic_crowd"
    for run_id in ("2026-08-09_10-30-00", "2026-08-09_11-45-00"):
        run_dir = evaluation_root / run_id
        (run_dir / "episode_cases").mkdir(parents=True)
        with (run_dir / "dynamic_crowd_results.json").open("w", encoding="utf-8") as file:
            json.dump({"results": [], "aggregates": []}, file)
        with (run_dir / "episode_cases" / "failure_cases.json").open("w", encoding="utf-8") as file:
            json.dump({"schema_version": 1, "cases": []}, file)

    runs = failure_viewer.discover_evaluation_runs(evaluation_root)
    assert [run.run_id for run in runs] == ["2026-08-09_11-45-00", "2026-08-09_10-30-00"]
    (evaluation_root / "failure_cases").mkdir()
    legacy_runs = failure_viewer.discover_evaluation_runs(evaluation_root / "failure_cases")
    assert [run.run_id for run in legacy_runs] == ["2026-08-09_11-45-00", "2026-08-09_10-30-00"]
    server = failure_viewer.FailureCaseWebServer(("127.0.0.1", 0), evaluation_root, view_radius=5.0)
    try:
        payload = server.index_payload("2026-08-09_10-30-00")
        assert payload["selected_run_id"] == "2026-08-09_10-30-00"
        assert [run["id"] for run in payload["runs"]] == ["2026-08-09_11-45-00", "2026-08-09_10-30-00"]
    finally:
        server.server_close()


def test_failure_viewer_web_api_serves_replay_and_persists_tags(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    np.savez_compressed(
        cases_dir / "collision_000001.npz",
        time_s=np.array([0.0]),
        robot_position_xy=np.zeros((1, 2)),
        robot_yaw=np.zeros(1),
        robot_velocity_xy_world=np.zeros((1, 2)),
        robot_command_velocity_body=np.zeros((1, 3)),
        goal_position_xy=np.zeros((1, 2)),
        pedestrian_position_xy=np.zeros((1, 1, 2)),
        pedestrian_velocity_xy_world=np.zeros((1, 1, 2)),
        pedestrian_active_mask=np.ones((1, 1), dtype=bool),
    )
    with (tmp_path / "failure_cases.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "collision_000001",
                        "scenario": "crossing",
                        "pedestrian_count": 2,
                        "collision_time_s": 1.0,
                        "colliding_agent_ids": [0],
                        "step_dt_s": 0.1,
                        "replay_file": "cases/collision_000001.npz",
                    }
                ],
            },
            file,
        )
    with (tmp_path / "dynamic_crowd_results.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "results": [
                    {
                        "scenario": "crossing",
                        "pedestrian_count": 2,
                        "episodes": 10,
                        "successes": 6,
                        "collisions": 2,
                        "goal_region_collisions": 1,
                        "all_collisions": 3,
                    }
                ],
                "aggregates": [],
            },
            file,
        )

    server = failure_viewer.FailureCaseWebServer(
        ("127.0.0.1", 0), tmp_path, view_radius=5.0, evaluation_dir=tmp_path
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base_url}/api/index") as response:
            payload = json.load(response)
            assert payload["index"]["cases"][0]["case_id"] == "collision_000001"
            assert payload["evaluation"]["results"][0]["successes"] == 6
        with urlopen(f"{base_url}/api/case/collision_000001") as response:
            assert json.load(response)["pedestrian_active_mask"] == [[True]]
        request = Request(
            f"{base_url}/api/tags/collision_000001",
            data=json.dumps({"tags": "late brake, crossing"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert json.load(response)["tags_by_case"] == {"collision_000001": ["late brake", "crossing"]}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    assert failure_viewer.load_case_tags(tmp_path) == {"collision_000001": ["late brake", "crossing"]}


def test_multi_seed_evaluator_imports_json_for_per_seed_aggregates():
    """The multi-seed completion path serializes ``per_seed_aggregates.json``."""
    source = EVALUATE_PATH.read_text(encoding="utf-8")
    assert "import json" in source
    assert "per_seed_aggregates.json" in source


def test_evaluator_publishes_throttled_live_progress_to_wandb():
    """A long remote benchmark exposes progress without depending on Pod logs."""
    tree = ast.parse(EVALUATE_PATH.read_text(encoding="utf-8"))
    reporter_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EvaluationProgressReporter"
    )
    namespace = {"os": os, "time": time, "datetime": datetime, "timezone": timezone}
    exec(compile(ast.Module(body=[reporter_node], type_ignores=[]), str(EVALUATE_PATH), "exec"), namespace)
    reporter_class = namespace["EvaluationProgressReporter"]

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.logged = []
            self.finished = False

        def log(self, values, *, commit):
            self.logged.append((values, commit))

        def finish(self):
            self.finished = True

    run = FakeRun()
    fake_wandb = types.SimpleNamespace(init=lambda **kwargs: run)
    original_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    previous_experiment_id = os.environ.get("RESEARCH_EXPERIMENT_ID")
    previous_project = os.environ.get("WANDB_PROJECT")
    os.environ["RESEARCH_EXPERIMENT_ID"] = "progress-test"
    os.environ["WANDB_PROJECT"] = "agent_obstacle_avoidance"
    try:
        reporter = reporter_class(profile_count=24, episodes_per_profile=100, seed_count=5)
        reporter.report(480, seed=104, seed_index=5, status="running", force=True)
        assert run.summary["research_agent_evaluation_accepted_episodes"] == 480
        assert run.summary["research_agent_evaluation_total_episodes"] == 2400
        assert run.summary["research_agent_evaluation_percent"] == 20.0
        assert run.summary["research_agent_evaluation_estimated_remaining_seconds"] is not None
        assert len(run.logged) == 1
        reporter.report(481, seed=104, seed_index=5, status="running")
        assert len(run.logged) == 1  # The normal path is rate limited to 30 seconds.
        reporter.report(2400, seed=104, seed_index=5, status="complete", force=True)
        assert run.summary["research_agent_evaluation_status"] == "complete"
        assert run.summary["research_agent_evaluation_percent"] == 100.0
        reporter.close()
        assert run.finished
    finally:
        if original_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original_wandb
        if previous_experiment_id is None:
            os.environ.pop("RESEARCH_EXPERIMENT_ID", None)
        else:
            os.environ["RESEARCH_EXPERIMENT_ID"] = previous_experiment_id
        if previous_project is None:
            os.environ.pop("WANDB_PROJECT", None)
        else:
            os.environ["WANDB_PROJECT"] = previous_project
