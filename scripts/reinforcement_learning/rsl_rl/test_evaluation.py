"""Focused unit tests for the reusable dynamic-crowd evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = Path(__file__).with_name("evaluation.py")
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
    assert "std_xy_speed_mps" in results[0]
    assert "success_rate_std" not in results[0]
    assert "collision_rate_std" not in results[0]


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

    server = failure_viewer.FailureCaseWebServer(("127.0.0.1", 0), tmp_path, view_radius=5.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base_url}/api/index") as response:
            assert json.load(response)["index"]["cases"][0]["case_id"] == "collision_000001"
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
