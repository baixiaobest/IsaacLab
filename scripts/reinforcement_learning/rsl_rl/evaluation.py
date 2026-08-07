"""Reusable helpers for vectorized policy-evaluation benchmarks.

The collector intentionally consumes Isaac Lab's reset-log payload instead of private
environment buffers. This makes it usable by any ManagerBased task that reports per-episode
termination IDs and metrics through ``extras["log"]``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCENARIO_ORDER = ("crossing", "with_flow", "against_flow")
SCENARIO_LABELS = {
    "crossing": "Crossing",
    "with_flow": "With flow",
    "against_flow": "Against flow",
}


@dataclass(frozen=True)
class BenchmarkProfile:
    """One benchmark cell assigned to one or more vector environments."""

    scenario: str
    pedestrian_count: int


def _write_json_atomically(path: Path, payload: Any) -> None:
    """Replace a JSON file atomically so an interrupted evaluation leaves the old index usable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


class CollisionReplayRecorder:
    """Capture bounded vector-environment history and export pedestrian-collision replays.

    State stays in a per-environment GPU ring buffer during evaluation.  CPU transfers and disk
    writes occur only at a collision, immediately before Isaac Lab clears the terminal state.
    """

    schema_version = 1

    def __init__(
        self,
        profiles: list[BenchmarkProfile],
        env_profile_indices: Iterable[int],
        output_dir: str | Path,
        step_dt_s: float,
        history_seconds: float = 3.0,
    ):
        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive.")
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive.")

        self.profiles = profiles
        self.env_profile_indices = [int(index) for index in env_profile_indices]
        if not self.env_profile_indices or any(
            index < 0 or index >= len(profiles) for index in self.env_profile_indices
        ):
            raise ValueError("Every vector environment must be assigned a valid profile index.")
        self.output_dir = Path(output_dir)
        self.cases_dir = self.output_dir / "cases"
        self.index_path = self.output_dir / "failure_cases.json"
        self.step_dt_s = float(step_dt_s)
        self.history_seconds = float(history_seconds)
        # The terminal collision frame is added only when exporting, so the ring itself contains
        # exactly the requested leading history.
        self.history_frames = math.ceil(self.history_seconds / self.step_dt_s - 1e-9)
        self.capacity = self.history_frames

        self._buffers: dict[str, Any] | None = None
        self._write_indices = None
        self._counts = None
        self._elapsed_steps = None
        self._last_command = None
        self._env_ids = None
        self._next_case_number = 1
        self._cases: list[dict[str, Any]] = []
        self._load_existing_index()
        if not self.index_path.is_file():
            self._write_index()

    @property
    def case_count(self) -> int:
        return len(self._cases)

    def _load_existing_index(self) -> None:
        if not self.index_path.is_file():
            return
        with self.index_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("schema_version") != self.schema_version or not isinstance(payload.get("cases"), list):
            raise ValueError(f"Unsupported failure-case index: {self.index_path}")
        self._cases = payload["cases"]
        numbers = []
        for case in self._cases:
            case_id = str(case.get("case_id", ""))
            if case_id.startswith("collision_") and case_id[10:].isdigit():
                numbers.append(int(case_id[10:]))
        self._next_case_number = max(numbers, default=0) + 1

    def _write_index(self) -> None:
        _write_json_atomically(
            self.index_path,
            {
                "schema_version": self.schema_version,
                "step_dt_s": self.step_dt_s,
                "history_seconds": self.history_seconds,
                "cases": self._cases,
            },
        )

    def _initialize_buffers(self, env: Any) -> None:
        import torch

        robot = env.scene["robot"]
        crowd = env.crowd_manager
        num_envs = len(self.env_profile_indices)
        if env.num_envs != num_envs:
            raise ValueError("Replay recorder profile assignment does not match env.num_envs.")
        max_pedestrians = crowd.max_pedestrians
        device = robot.data.root_pos_w.device
        self._buffers = {
            "time_s": torch.zeros(num_envs, self.capacity, device=device),
            "robot_position_xy": torch.zeros(num_envs, self.capacity, 2, device=device),
            "robot_yaw": torch.zeros(num_envs, self.capacity, device=device),
            "robot_velocity_xy_world": torch.zeros(num_envs, self.capacity, 2, device=device),
            "robot_command_velocity_body": torch.zeros(num_envs, self.capacity, 3, device=device),
            "goal_position_xy": torch.zeros(num_envs, self.capacity, 2, device=device),
            "pedestrian_position_xy": torch.zeros(num_envs, self.capacity, max_pedestrians, 2, device=device),
            "pedestrian_velocity_xy_world": torch.zeros(num_envs, self.capacity, max_pedestrians, 2, device=device),
            "pedestrian_active_mask": torch.zeros(
                num_envs, self.capacity, max_pedestrians, dtype=torch.bool, device=device
            ),
        }
        self._write_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._last_command = torch.zeros(num_envs, 3, device=device)
        self._env_ids = torch.arange(num_envs, device=device)

    @staticmethod
    def _three_component_command(command_velocity_body: Any) -> Any:
        """Pad a command tensor to ``(vx, vy, yaw_rate)`` without accepting ambiguous ranks."""
        import torch

        if command_velocity_body.ndim != 2 or command_velocity_body.shape[1] < 2:
            raise ValueError("Command velocity must have shape (num_envs, at least 2).")
        command = torch.zeros(command_velocity_body.shape[0], 3, device=command_velocity_body.device)
        command[:, : min(3, command_velocity_body.shape[1])] = command_velocity_body[:, :3]
        return command

    def _snapshot(self, env: Any, command_velocity_body: Any) -> None:
        import torch

        if self._buffers is None:
            self._initialize_buffers(env)
        assert self._buffers is not None
        assert self._env_ids is not None
        assert self._write_indices is not None
        assert self._counts is not None
        assert self._elapsed_steps is not None

        command = self._three_component_command(command_velocity_body)
        if command.shape[0] != len(self.env_profile_indices):
            raise ValueError("Command velocity must contain one row per vector environment.")

        robot = env.scene["robot"]
        crowd = env.crowd_manager
        command_term = env.command_manager.get_term("pose_2d_command")
        goal = command_term.pos_command_w[:, :2]
        indices = self._write_indices
        env_ids = self._env_ids

        self._buffers["time_s"][env_ids, indices] = self._elapsed_steps.to(dtype=torch.float32) * self.step_dt_s
        self._buffers["robot_position_xy"][env_ids, indices] = robot.data.root_pos_w[:, :2]
        self._buffers["robot_yaw"][env_ids, indices] = robot.data.heading_w
        self._buffers["robot_velocity_xy_world"][env_ids, indices] = robot.data.root_lin_vel_w[:, :2]
        self._buffers["robot_command_velocity_body"][env_ids, indices] = command
        self._buffers["goal_position_xy"][env_ids, indices] = goal
        self._buffers["pedestrian_position_xy"][env_ids, indices] = crowd.get_world_positions()
        self._buffers["pedestrian_velocity_xy_world"][env_ids, indices] = crowd.get_velocities()
        self._buffers["pedestrian_active_mask"][env_ids, indices] = crowd.get_active_mask()
        self._last_command[:] = command
        self._write_indices = (indices + 1) % self.capacity
        self._counts = (self._counts + 1).clamp(max=self.capacity)
        self._elapsed_steps += 1

    def record_pre_step(self, env: Any, command_velocity_body: Any) -> None:
        """Store the state and body-frame command immediately before an environment step."""
        self._snapshot(env, command_velocity_body)

    def _ordered_frames(self, env_id: int) -> dict[str, np.ndarray]:
        import torch

        if self._buffers is None or self._counts is None or self._write_indices is None:
            raise RuntimeError("No replay state has been recorded.")
        count = int(self._counts[env_id].item())
        if count == 0:
            raise RuntimeError(f"Environment {env_id} has no replay frames.")
        if count < self.capacity:
            order = torch.arange(count, device=self._write_indices.device)
        else:
            start = self._write_indices[env_id]
            order = (torch.arange(self.capacity, device=start.device) + start) % self.capacity
        return {
            name: values[env_id].index_select(0, order).detach().cpu().numpy()
            for name, values in self._buffers.items()
        }

    def _terminal_frame(self, env: Any, env_id: int) -> dict[str, np.ndarray]:
        """Read one terminal-state frame without mutating other live environments' rings."""
        assert self._elapsed_steps is not None and self._last_command is not None
        robot = env.scene["robot"]
        crowd = env.crowd_manager
        command_term = env.command_manager.get_term("pose_2d_command")
        return {
            "time_s": np.asarray([float(self._elapsed_steps[env_id].item()) * self.step_dt_s], dtype=np.float32),
            "robot_position_xy": robot.data.root_pos_w[env_id : env_id + 1, :2].detach().cpu().numpy(),
            "robot_yaw": robot.data.heading_w[env_id : env_id + 1].detach().cpu().numpy(),
            "robot_velocity_xy_world": robot.data.root_lin_vel_w[env_id : env_id + 1, :2].detach().cpu().numpy(),
            "robot_command_velocity_body": self._last_command[env_id : env_id + 1].detach().cpu().numpy(),
            "goal_position_xy": command_term.pos_command_w[env_id : env_id + 1, :2].detach().cpu().numpy(),
            "pedestrian_position_xy": crowd.get_world_positions()[env_id : env_id + 1].detach().cpu().numpy(),
            "pedestrian_velocity_xy_world": crowd.get_velocities()[env_id : env_id + 1].detach().cpu().numpy(),
            "pedestrian_active_mask": crowd.get_active_mask()[env_id : env_id + 1].detach().cpu().numpy(),
        }

    def _collision_indices(self, env: Any, env_id: int) -> list[int]:
        import torch

        robot_position = env.scene["robot"].data.root_pos_w[env_id, :2]
        crowd = env.crowd_manager
        distance = torch.linalg.vector_norm(crowd.get_world_positions()[env_id] - robot_position, dim=-1)
        threshold = crowd.radius[env_id] + crowd.cfg.robot_radius
        colliding = (distance < threshold) & crowd.get_active_mask()[env_id]
        return torch.nonzero(colliding, as_tuple=False).reshape(-1).detach().cpu().tolist()

    def capture_terminal_collisions(self, env: Any, reset_env_ids: Any) -> list[dict[str, Any]]:
        """Export collisions among environments about to reset, then clear their histories."""
        import torch

        if self._buffers is None:
            return []
        env_ids = torch.as_tensor(reset_env_ids, device=self._env_ids.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return []
        robot_positions = env.scene["robot"].data.root_pos_w[:, :2]
        collision_mask = env.crowd_manager.get_robot_collision(robot_positions)
        collision_env_ids = env_ids[collision_mask[env_ids]]

        exported = []
        for env_id in collision_env_ids.detach().cpu().tolist():
            exported.append(self._export_case(env, int(env_id)))
        self.reset(env_ids)
        return exported

    def _export_case(self, env: Any, env_id: int) -> dict[str, Any]:
        frames = self._ordered_frames(env_id)
        terminal_frame = self._terminal_frame(env, env_id)
        frames = {name: np.concatenate([values, terminal_frame[name]], axis=0) for name, values in frames.items()}
        profile = self.profiles[self.env_profile_indices[env_id]]
        colliding_agent_ids = self._collision_indices(env, env_id)
        case_id = f"collision_{self._next_case_number:06d}"
        self._next_case_number += 1
        filename = f"{case_id}.npz"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cases_dir / filename, **frames)
        entry = {
            "case_id": case_id,
            "scenario": profile.scenario,
            "pedestrian_count": profile.pedestrian_count,
            "environment_id": env_id,
            "collision_time_s": float(frames["time_s"][-1]),
            "colliding_agent_ids": colliding_agent_ids,
            "step_dt_s": self.step_dt_s,
            "history_seconds": self.history_seconds,
            "frame_count": int(frames["time_s"].shape[0]),
            "replay_file": str(Path("cases") / filename),
        }
        self._cases.append(entry)
        self._write_index()
        return entry

    def reset(self, env_ids: Any) -> None:
        """Discard history for environments that have just completed an episode."""
        if self._buffers is None:
            return
        import torch

        ids = torch.as_tensor(env_ids, device=self._env_ids.device, dtype=torch.long).reshape(-1)
        self._write_indices[ids] = 0
        self._counts[ids] = 0
        self._elapsed_steps[ids] = 0
        self._last_command[ids] = 0.0


def _sample_standard_deviation(values: Iterable[float]) -> float:
    """Return the sample standard deviation, or zero for fewer than two samples."""
    samples = list(values)
    if len(samples) < 2:
        return 0.0
    mean = sum(samples) / len(samples)
    return math.sqrt(sum((value - mean) ** 2 for value in samples) / (len(samples) - 1))


def dynamic_crowd_profiles(counts: Iterable[int] = range(2, 17, 2)) -> list[BenchmarkProfile]:
    """Return crossing, with-flow, and against-flow profiles for every crowd count."""
    return [
        BenchmarkProfile(scenario, count)
        for scenario in SCENARIO_ORDER
        for count in counts
    ]


def _flat_list(value: Any) -> list[Any]:
    """Normalize scalar, tensor, and sequence log values to a flat Python list."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "reshape") and hasattr(value, "tolist"):
        value = value.reshape(-1).tolist()
    elif isinstance(value, Number):
        return [value]

    # ``tolist()`` for some scalar array types returns a scalar rather than a list.
    if isinstance(value, Number):
        return [value]
    return list(value)


def _ids(value: Any) -> set[int]:
    """Normalize scalar/tensor/sequence environment IDs, including ``torch.nonzero`` output."""
    return {int(item) for item in _flat_list(value)}


def completed_environment_ids(extras: Mapping[str, Any]) -> set[int]:
    """Return the environment IDs that completed an episode in an Isaac Lab reset log."""
    log = extras.get("log", {})
    completed_ids: set[int] = set()
    for key, value in log.items():
        if key.startswith("Episode_Termination/Envs/Ids/"):
            completed_ids |= _ids(value)
    return completed_ids


class EpisodeVelocityAccumulator:
    """Accumulate world-XY speed directly from a vector environment's robot state.

    Command-manager metrics are optional in Isaac Lab tasks. This tracker provides a reusable
    source for the same episode-level metric when a task does not export such metrics.
    """

    def __init__(self, num_envs: int):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        self._sums = [0.0] * num_envs
        self._samples = [0] * num_envs

    def record_step(self, world_xy_speed: Any) -> None:
        """Add one pre-action speed sample for every vector environment."""
        values = _flat_list(world_xy_speed)
        if len(values) != len(self._sums):
            raise ValueError("Speed samples must contain one value for every vector environment.")
        for env_id, value in enumerate(values):
            self._sums[env_id] += float(value)
            self._samples[env_id] += 1

    def record_terminal(self, world_xy_speed: Any, env_ids: Any) -> None:
        """Add the final, pre-reset sample for environments ending an episode."""
        values = _flat_list(world_xy_speed)
        if len(values) != len(self._sums):
            raise ValueError("Speed samples must contain one value for every vector environment.")
        for env_id in _ids(env_ids):
            if env_id < 0 or env_id >= len(self._sums):
                raise IndexError(f"Invalid environment ID {env_id}.")
            self._sums[env_id] += float(values[env_id])
            self._samples[env_id] += 1

    def completed_means(self, env_ids: Any) -> dict[int, float]:
        """Return the current episode mean for each completed environment."""
        means = {}
        for env_id in _ids(env_ids):
            if self._samples[env_id] == 0:
                raise RuntimeError(f"No velocity samples recorded for completed environment {env_id}.")
            means[env_id] = self._sums[env_id] / self._samples[env_id]
        return means

    def reset(self, env_ids: Any) -> None:
        """Clear accumulators after their episodes have been consumed."""
        for env_id in _ids(env_ids):
            self._sums[env_id] = 0.0
            self._samples[env_id] = 0


class EpisodeMetricsCollector:
    """Collect bounded per-profile episode outcomes from vector-environment reset logs."""

    def __init__(
        self,
        profiles: list[BenchmarkProfile],
        env_profile_indices: Iterable[int],
        episodes_per_profile: int,
        command_name: str = "pose_2d_command",
        velocity_metric: str = "linear_velocity_xy",
        fallback_velocity_metric: str | None = "linear_velocity",
        success_term: str = "goal_reached",
        collision_term: str = "pedestrian_collision",
    ):
        if episodes_per_profile <= 0:
            raise ValueError("episodes_per_profile must be positive.")
        self.profiles = profiles
        self.env_profile_indices = [int(index) for index in env_profile_indices]
        if not self.env_profile_indices or any(
            index < 0 or index >= len(profiles) for index in self.env_profile_indices
        ):
            raise ValueError("Every vector environment must be assigned a valid profile index.")
        self.episodes_per_profile = episodes_per_profile
        self.success_ids_key = f"Episode_Termination/Envs/Ids/{success_term}"
        self.collision_ids_key = f"Episode_Termination/Envs/Ids/{collision_term}"
        self.metric_ids_key = f"Metrics/{command_name}/{velocity_metric}/Ids"
        self.metric_values_key = f"Metrics/{command_name}/{velocity_metric}/Envs"
        self.fallback_metric_ids_key = (
            f"Metrics/{command_name}/{fallback_velocity_metric}/Ids" if fallback_velocity_metric else None
        )
        self.fallback_metric_values_key = (
            f"Metrics/{command_name}/{fallback_velocity_metric}/Envs" if fallback_velocity_metric else None
        )
        self.velocity_metric_source = velocity_metric
        self._episodes = [0] * len(profiles)
        self._successes = [0] * len(profiles)
        self._collisions = [0] * len(profiles)
        self._velocity_sums = [0.0] * len(profiles)
        # Retain episode-level speed means to report their variation across episodes.
        self._velocity_values: list[list[float]] = [[] for _ in profiles]

    @property
    def complete(self) -> bool:
        return all(episodes >= self.episodes_per_profile for episodes in self._episodes)

    @property
    def total_episodes(self) -> int:
        return sum(self._episodes)

    def consume(
        self,
        extras: dict[str, Any],
        velocity_by_env: Mapping[int, float] | None = None,
        completed_env_ids: Any | None = None,
    ) -> int:
        """Consume completed episodes from one environment step and return accepted count.

        ``completed_env_ids`` should be supplied from the vector-environment done mask when it
        is available. Isaac Lab clears idle ``Episode_Termination/...`` log fields to the scalar
        ``0``; that value is a metric placeholder, not a completion of environment zero.
        """
        log = extras.get("log", {})
        completed_ids = _ids(completed_env_ids) if completed_env_ids is not None else completed_environment_ids(extras)
        if not completed_ids:
            return 0

        metric_by_env: dict[int, float] | None = (
            {int(env_id): float(value) for env_id, value in velocity_by_env.items()}
            if velocity_by_env is not None
            else None
        )
        metric_ids_key = self.metric_ids_key
        metric_values_key = self.metric_values_key
        metric_values_raw = log.get(metric_values_key)
        # Older Isaac Lab command terms expose only ``linear_velocity``. The pedestrian
        # corridor is flat, so this legacy world-speed metric is a safe compatibility fallback.
        if metric_values_raw is None and self.fallback_metric_values_key is not None:
            fallback_values = log.get(self.fallback_metric_values_key)
            if fallback_values is not None:
                metric_ids_key = self.fallback_metric_ids_key
                metric_values_key = self.fallback_metric_values_key
                metric_values_raw = fallback_values
                self.velocity_metric_source = metric_values_key.rsplit("/", 1)[0].split("/")[-1]
        if metric_values_raw is None and metric_by_env is None:
            available_metrics = sorted(key for key in log if key.startswith("Metrics/"))
            raise KeyError(
                f"Missing required per-episode metric: {self.metric_values_key}. "
                f"Available metrics: {available_metrics}"
            )
        if metric_by_env is None:
            metric_values = _flat_list(metric_values_raw)
            raw_ids = log.get(metric_ids_key)
            if raw_ids is None:
                raise KeyError(f"Missing required per-episode metric IDs: {metric_ids_key}")
            raw_ids = _flat_list(raw_ids)
            if len(raw_ids) != len(metric_values):
                raise ValueError("Metric IDs and values must have equal lengths.")
            metric_by_env = {int(env_id): float(value) for env_id, value in zip(raw_ids, metric_values)}
        else:
            self.velocity_metric_source = "direct_world_xy_speed"

        success_ids = _ids(log.get(self.success_ids_key))
        collision_ids = _ids(log.get(self.collision_ids_key))
        accepted = 0
        for env_id in sorted(completed_ids):
            if env_id < 0 or env_id >= len(self.env_profile_indices):
                raise IndexError(f"Termination reported invalid environment ID {env_id}.")
            profile_index = self.env_profile_indices[env_id]
            if self._episodes[profile_index] >= self.episodes_per_profile:
                continue
            if env_id not in metric_by_env:
                raise KeyError(f"Missing velocity metric for completed environment {env_id}.")

            self._episodes[profile_index] += 1
            self._velocity_sums[profile_index] += metric_by_env[env_id]
            # Collision takes precedence when both terms trigger on the same final step.
            if env_id in collision_ids:
                self._collisions[profile_index] += 1
            elif env_id in success_ids:
                self._successes[profile_index] += 1
            self._velocity_values[profile_index].append(metric_by_env[env_id])
            accepted += 1
        return accepted

    def rows(self) -> list[dict[str, Any]]:
        """Return one normalized result row for every profile."""
        rows = []
        for index, profile in enumerate(self.profiles):
            episodes = self._episodes[index]
            rows.append(
                {
                    **asdict(profile),
                    "episodes": episodes,
                    "successes": self._successes[index],
                    "collisions": self._collisions[index],
                    "success_rate": self._successes[index] / episodes if episodes else 0.0,
                    "collision_rate": self._collisions[index] / episodes if episodes else 0.0,
                    "mean_xy_speed_mps": self._velocity_sums[index] / episodes if episodes else 0.0,
                    "std_xy_speed_mps": _sample_standard_deviation(self._velocity_values[index]),
                }
            )
        return rows

    def aggregate_rows(self) -> list[dict[str, Any]]:
        """Return pooled per-episode aggregates for every scenario."""
        aggregates = []
        for scenario in SCENARIO_ORDER:
            profile_indices = [index for index, profile in enumerate(self.profiles) if profile.scenario == scenario]
            if not profile_indices:
                continue
            episodes = sum(self._episodes[index] for index in profile_indices)
            successes = sum(self._successes[index] for index in profile_indices)
            collisions = sum(self._collisions[index] for index in profile_indices)
            velocity_values = [value for index in profile_indices for value in self._velocity_values[index]]
            aggregates.append(
                {
                    "scenario": scenario,
                    "pedestrian_count": "all",
                    "episodes": episodes,
                    "successes": successes,
                    "collisions": collisions,
                    "success_rate": successes / episodes if episodes else 0.0,
                    "collision_rate": collisions / episodes if episodes else 0.0,
                    "mean_xy_speed_mps": sum(velocity_values) / episodes if episodes else 0.0,
                    "std_xy_speed_mps": _sample_standard_deviation(velocity_values),
                }
            )
        return aggregates


def print_results(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    """Print a compact result table without introducing a tabular dependency."""
    header = (
        "scenario        crowd  episodes  success  collision  success%  "
        "collision%  mean xy speed (m/s) +/- std"
    )
    print(header)
    print("-" * len(header))
    for row in [*rows, *aggregate_rows]:
        print(
            f"{row['scenario']:<15} {str(row['pedestrian_count']):>5} {row['episodes']:>9} "
            f"{row['successes']:>8} {row['collisions']:>10} {100 * row['success_rate']:>8.1f} "
            f"{100 * row['collision_rate']:>10.1f} {row['mean_xy_speed_mps']:>8.3f} "
            f"+/- {row['std_xy_speed_mps']:<.3f}"
        )


def save_artifacts(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write CSV, JSON, and the standard 3x3 dynamic-crowd summary plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_rows = [*rows, *aggregate_rows]
    fieldnames = [
        "scenario", "pedestrian_count", "episodes", "successes", "collisions",
        "success_rate", "collision_rate", "mean_xy_speed_mps", "std_xy_speed_mps",
    ]
    with (output_path / "dynamic_crowd_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_path / "dynamic_crowd_results.json").open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "results": rows, "aggregates": aggregate_rows}, file, indent=2)
    _save_summary_plot(output_path / "dynamic_crowd_summary.png", rows)
    return output_path


def _save_summary_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save metric-by-scenario facets with crowd count on every x-axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = (
        ("success_rate", None, "Success rate (%)", 100.0, (0.0, 100.0)),
        ("collision_rate", None, "Collision rate (%)", 100.0, (0.0, 100.0)),
        ("mean_xy_speed_mps", "std_xy_speed_mps", "Mean XY speed (m/s)", 1.0, None),
    )
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), sharex="col")
    for col, scenario in enumerate(SCENARIO_ORDER):
        scenario_rows = sorted(
            (row for row in rows if row["scenario"] == scenario), key=lambda row: row["pedestrian_count"]
        )
        crowd_counts = [row["pedestrian_count"] for row in scenario_rows]
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
            if row_index == 2:
                axis.set_xlabel("Pedestrians")
    figure.suptitle("Dynamic crowd evaluation (speed shaded: ±1 sample SD)", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180)
    plt.close(figure)
