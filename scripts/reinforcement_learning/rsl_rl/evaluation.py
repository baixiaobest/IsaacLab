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


SCENARIO_ORDER = (
    "crossing",
    "with_flow",
    "against_flow",
    "with_flow_slow_leader",
    "crossing_slow",
    "against_flow_slow",
)
SCENARIO_LABELS = {
    "crossing": "Crossing",
    "with_flow": "With flow",
    "against_flow": "Against flow",
    "with_flow_slow_leader": "With flow — slow leader",
    "crossing_slow": "Crossing — slow crowd",
    "against_flow_slow": "Against flow — slow crowd",
}

GOAL_REGION_COLLISION_RADIUS_M = 0.75
"""Terminal-goal buffer used to report goal-region collisions separately."""

GOAL_REGION_TAG = "goal-region"
"""Immutable replay tag assigned to collisions inside the terminal-goal buffer."""

INTERESTING_INTERACTION_TAG = "interesting-interaction"
"""Immutable replay tag assigned to sampled successes with a close pedestrian interaction."""

SUCCESS_INTERACTION_DISTANCE_M = 1.5
"""Default robot-to-pedestrian distance required to sample a successful replay."""

# Interaction-event evaluation defaults.  These are deliberately kept separate from the
# training reward settings: they define a reproducible analysis protocol, not policy reward.
INTERACTION_ENTER_CLEARANCE_M = 1.5
INTERACTION_EXIT_CLEARANCE_M = 1.75
INTERACTION_RISK_CLEARANCE_M = 0.5
INTERACTION_RISK_HORIZON_S = 1.5
INTERACTION_PRE_SPEED_WINDOW_S = 0.5
INTERACTION_MIN_DURATION_S = 0.2
INTERACTION_MIN_BASELINE_SPEED_MPS = 0.2
INTERACTION_YIELD_SPEED_RATIO = 0.70
INTERACTION_ASSERT_SPEED_RATIO = 0.85
INTERACTION_OVERTAKE_LONGITUDINAL_MARGIN_M = 0.5
INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M = 0.15
INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M = 0.25

INTERACTION_LABELS = {
    "crossing": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
    "against_flow": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
    "with_flow": ("overtake", "non_overtake", "non_risky_close", "unclassified"),
    "with_flow_slow_leader": ("overtake", "non_overtake", "non_risky_close", "unclassified"),
    "crossing_slow": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
    "against_flow_slow": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
}


@dataclass(frozen=True)
class BenchmarkProfile:
    """One benchmark cell assigned to one or more vector environments."""

    scenario: str
    pedestrian_count: int


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON-compatible values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_atomically(path: Path, payload: Any) -> None:
    """Replace a JSON file atomically so an interrupted evaluation leaves the old index usable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
        file.write("\n")
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def terminal_goal_region_collision_ids(
    env: Any,
    reset_env_ids: Any,
    radius_m: float = GOAL_REGION_COLLISION_RADIUS_M,
    command_name: str = "pose_2d_command",
) -> set[int]:
    """Return resetting environments that collide within ``radius_m`` of their world-frame goal."""
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    import torch

    env_ids = torch.as_tensor(reset_env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if env_ids.numel() == 0:
        return set()
    robot_positions = env.scene["robot"].data.root_pos_w[:, :2]
    goal_positions = env.command_manager.get_term(command_name).pos_command_w[:, :2]
    collision_mask = env.crowd_manager.get_robot_collision(robot_positions)
    within_goal_region = torch.linalg.vector_norm(robot_positions - goal_positions, dim=1) <= radius_m
    return set(env_ids[collision_mask[env_ids] & within_goal_region[env_ids]].detach().cpu().tolist())


class CollisionReplayRecorder:
    """Capture collision context and a quota of interesting complete successful episodes.

    State stays in a per-environment GPU ring buffer during evaluation.  Collision exports retain
    only the requested leading history, while optional success exports retain every frame from
    reset through the terminal goal-reached state. CPU transfers and disk writes occur only when
    an episode ends, immediately before Isaac Lab clears its terminal state.
    """

    schema_version = 2

    def __init__(
        self,
        profiles: list[BenchmarkProfile],
        env_profile_indices: Iterable[int],
        output_dir: str | Path,
        step_dt_s: float,
        history_seconds: float = 3.0,
        goal_region_radius_m: float = GOAL_REGION_COLLISION_RADIUS_M,
        successes_per_scenario: int = 0,
        episode_length_s: float | None = None,
        record_collisions: bool = True,
        interesting_interaction_distance_m: float = SUCCESS_INTERACTION_DISTANCE_M,
    ):
        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive.")
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive.")
        if goal_region_radius_m <= 0.0:
            raise ValueError("goal_region_radius_m must be positive.")
        if successes_per_scenario < 0:
            raise ValueError("successes_per_scenario must be non-negative.")
        if successes_per_scenario and (episode_length_s is None or episode_length_s <= 0.0):
            raise ValueError("episode_length_s must be positive when recording successful episodes.")
        if interesting_interaction_distance_m <= 0.0:
            raise ValueError("interesting_interaction_distance_m must be positive.")

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
        self.goal_region_radius_m = float(goal_region_radius_m)
        self.successes_per_scenario = int(successes_per_scenario)
        self.episode_length_s = float(episode_length_s) if episode_length_s is not None else None
        self.record_collisions = bool(record_collisions)
        self.interesting_interaction_distance_m = float(interesting_interaction_distance_m)
        # The terminal frame is added only when exporting, so the ring itself contains exactly
        # the requested leading history or the complete pre-terminal successful episode.
        self.history_frames = math.ceil(self.history_seconds / self.step_dt_s - 1e-9)
        self.full_episode_frames = (
            math.ceil(self.episode_length_s / self.step_dt_s - 1e-9) if self.episode_length_s is not None else 0
        )
        self.capacity = max(self.history_frames, self.full_episode_frames)

        self._buffers: dict[str, Any] | None = None
        self._write_indices = None
        self._counts = None
        self._elapsed_steps = None
        self._last_command = None
        self._last_cbf_filtered_command = None
        self._env_ids = None
        self._next_case_numbers = {"collision": 1, "success": 1}
        self._cases: list[dict[str, Any]] = []
        self._successes_by_scenario = {profile.scenario: 0 for profile in profiles}
        self._minimum_agent_distances = None
        self._load_existing_index()
        if not self.index_path.is_file():
            self._write_index()

    @property
    def case_count(self) -> int:
        return len(self._cases)

    @property
    def collision_case_count(self) -> int:
        """Return the number of pedestrian-collision replay artifacts."""
        return sum(case.get("outcome", "collision") == "collision" for case in self._cases)

    @property
    def success_case_count(self) -> int:
        """Return the number of complete successful-episode replay artifacts."""
        return sum(case.get("outcome") == "success" for case in self._cases)

    @property
    def success_recording_complete(self) -> bool:
        """Whether every scenario has reached its interesting-success replay quota."""
        return self.successes_per_scenario == 0 or all(
            count >= self.successes_per_scenario for count in self._successes_by_scenario.values()
        )

    def _load_existing_index(self) -> None:
        if not self.index_path.is_file():
            return
        with self.index_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("schema_version") != self.schema_version or not isinstance(payload.get("cases"), list):
            raise ValueError(f"Unsupported failure-case index: {self.index_path}")
        self._cases = payload["cases"]
        numbers = {"collision": [], "success": []}
        for case in self._cases:
            case_id = str(case.get("case_id", ""))
            for outcome, prefix in (("collision", "collision_"), ("success", "success_")):
                if case_id.startswith(prefix) and case_id[len(prefix) :].isdigit():
                    numbers[outcome].append(int(case_id[len(prefix) :]))
            if case.get("outcome") == "success":
                scenario = case.get("scenario")
                if scenario in self._successes_by_scenario:
                    self._successes_by_scenario[scenario] += 1
        self._next_case_numbers = {outcome: max(values, default=0) + 1 for outcome, values in numbers.items()}

    def _write_index(self) -> None:
        _write_json_atomically(
            self.index_path,
            {
                "schema_version": self.schema_version,
                "step_dt_s": self.step_dt_s,
                "history_seconds": self.history_seconds,
                "goal_region_radius_m": self.goal_region_radius_m,
                "successes_per_scenario": self.successes_per_scenario,
                "episode_length_s": self.episode_length_s,
                "record_collisions": self.record_collisions,
                "interesting_interaction_distance_m": self.interesting_interaction_distance_m,
                "cases": self._cases,
            },
        )

    def _initialize_buffers(self, env: Any, cbf_filtered_command: Any | None = None) -> None:
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
            # Kept as a legacy alias for existing replay consumers.
            "robot_command_velocity_body": torch.zeros(num_envs, self.capacity, 3, device=device),
            "navigation_policy_velocity_body": torch.zeros(num_envs, self.capacity, 3, device=device),
            "goal_position_xy": torch.zeros(num_envs, self.capacity, 2, device=device),
            "pedestrian_position_xy": torch.zeros(num_envs, self.capacity, max_pedestrians, 2, device=device),
            "pedestrian_velocity_xy_world": torch.zeros(num_envs, self.capacity, max_pedestrians, 2, device=device),
            "pedestrian_active_mask": torch.zeros(
                num_envs, self.capacity, max_pedestrians, dtype=torch.bool, device=device
            ),
        }
        if cbf_filtered_command is not None:
            filtered = self._three_component_command(cbf_filtered_command)
            if filtered.shape[0] != num_envs:
                raise ValueError("CBF command must contain one row per vector environment.")
            self._buffers["cbf_filtered_command_velocity_body"] = torch.zeros(
                num_envs, self.capacity, 3, device=device
            )
            self._last_cbf_filtered_command = torch.zeros(num_envs, 3, device=device)
        self._write_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._last_command = torch.zeros(num_envs, 3, device=device)
        self._env_ids = torch.arange(num_envs, device=device)
        self._minimum_agent_distances = torch.full((num_envs,), float("inf"), device=device)

    @staticmethod
    def _three_component_command(command_velocity_body: Any) -> Any:
        """Pad a command tensor to ``(vx, vy, yaw_rate)`` without accepting ambiguous ranks."""
        import torch

        if command_velocity_body.ndim != 2 or command_velocity_body.shape[1] < 2:
            raise ValueError("Command velocity must have shape (num_envs, at least 2).")
        command = torch.zeros(command_velocity_body.shape[0], 3, device=command_velocity_body.device)
        command[:, : min(3, command_velocity_body.shape[1])] = command_velocity_body[:, :3]
        return command

    def _snapshot(self, env: Any, command_velocity_body: Any, cbf_filtered_command: Any | None = None) -> None:
        import torch

        if self._buffers is None:
            self._initialize_buffers(env, cbf_filtered_command)
        assert self._buffers is not None
        assert self._env_ids is not None
        assert self._write_indices is not None
        assert self._counts is not None
        assert self._elapsed_steps is not None
        assert self._minimum_agent_distances is not None

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
        self._buffers["navigation_policy_velocity_body"][env_ids, indices] = command
        self._buffers["goal_position_xy"][env_ids, indices] = goal
        self._buffers["pedestrian_position_xy"][env_ids, indices] = crowd.get_world_positions()
        self._buffers["pedestrian_velocity_xy_world"][env_ids, indices] = crowd.get_velocities()
        self._buffers["pedestrian_active_mask"][env_ids, indices] = crowd.get_active_mask()
        self._minimum_agent_distances = torch.minimum(
            self._minimum_agent_distances, self._minimum_active_pedestrian_distances(env)
        )
        self._last_command[:] = command
        self._write_indices = (indices + 1) % self.capacity
        self._counts = (self._counts + 1).clamp(max=self.capacity)
        self._elapsed_steps += 1

    def record_pre_step(
        self, env: Any, command_velocity_body: Any, cbf_filtered_command: Any | None = None
    ) -> None:
        """Store the state and raw policy command immediately before an environment step.

        ``cbf_filtered_command`` enables CBF replay recording. Call
        :meth:`record_cbf_filtered_command` after the step to write the final
        CBF command produced for this policy action.
        """
        self._snapshot(env, command_velocity_body, cbf_filtered_command)

    def record_cbf_filtered_command(self, cbf_filtered_command: Any) -> None:
        """Attach the final CBF command to the latest navigation-rate replay frame."""
        import torch

        if self._buffers is None or "cbf_filtered_command_velocity_body" not in self._buffers:
            return
        assert self._write_indices is not None and self._counts is not None
        assert self._last_cbf_filtered_command is not None
        command = self._three_component_command(cbf_filtered_command)
        if command.shape[0] != len(self.env_profile_indices):
            raise ValueError("CBF command must contain one row per vector environment.")
        active = self._counts > 0
        if not torch.any(active):
            return
        env_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
        indices = (self._write_indices[env_ids] - 1) % self.capacity
        self._buffers["cbf_filtered_command_velocity_body"][env_ids, indices] = command[env_ids]
        self._last_cbf_filtered_command[env_ids] = command[env_ids]

    def _ordered_frames(self, env_id: int, max_frames: int | None = None) -> dict[str, np.ndarray]:
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
        frames = {
            name: values[env_id].index_select(0, order).detach().cpu().numpy()
            for name, values in self._buffers.items()
        }
        if max_frames is not None:
            frames = {name: values[-max_frames:] for name, values in frames.items()}
        return frames

    def _terminal_frame(self, env: Any, env_id: int) -> dict[str, np.ndarray]:
        """Read one terminal-state frame without mutating other live environments' rings."""
        assert self._elapsed_steps is not None and self._last_command is not None
        robot = env.scene["robot"]
        crowd = env.crowd_manager
        command_term = env.command_manager.get_term("pose_2d_command")
        terminal = {
            "time_s": np.asarray([float(self._elapsed_steps[env_id].item()) * self.step_dt_s], dtype=np.float32),
            "robot_position_xy": robot.data.root_pos_w[env_id : env_id + 1, :2].detach().cpu().numpy(),
            "robot_yaw": robot.data.heading_w[env_id : env_id + 1].detach().cpu().numpy(),
            "robot_velocity_xy_world": robot.data.root_lin_vel_w[env_id : env_id + 1, :2].detach().cpu().numpy(),
            "robot_command_velocity_body": self._last_command[env_id : env_id + 1].detach().cpu().numpy(),
            "navigation_policy_velocity_body": self._last_command[env_id : env_id + 1].detach().cpu().numpy(),
            "goal_position_xy": command_term.pos_command_w[env_id : env_id + 1, :2].detach().cpu().numpy(),
            "pedestrian_position_xy": crowd.get_world_positions()[env_id : env_id + 1].detach().cpu().numpy(),
            "pedestrian_velocity_xy_world": crowd.get_velocities()[env_id : env_id + 1].detach().cpu().numpy(),
            "pedestrian_active_mask": crowd.get_active_mask()[env_id : env_id + 1].detach().cpu().numpy(),
        }
        if self._last_cbf_filtered_command is not None:
            terminal["cbf_filtered_command_velocity_body"] = (
                self._last_cbf_filtered_command[env_id : env_id + 1].detach().cpu().numpy()
            )
        return terminal

    def _collision_indices(self, env: Any, env_id: int) -> list[int]:
        import torch

        robot_position = env.scene["robot"].data.root_pos_w[env_id, :2]
        crowd = env.crowd_manager
        distance = torch.linalg.vector_norm(crowd.get_world_positions()[env_id] - robot_position, dim=-1)
        threshold = crowd.radius[env_id] + crowd.cfg.robot_radius
        colliding = (distance < threshold) & crowd.get_active_mask()[env_id]
        return torch.nonzero(colliding, as_tuple=False).reshape(-1).detach().cpu().tolist()

    @staticmethod
    def _minimum_active_pedestrian_distances(env: Any) -> Any:
        """Return each environment's minimum robot distance to an active pedestrian, or infinity."""
        import torch

        robot_positions = env.scene["robot"].data.root_pos_w[:, :2]
        crowd = env.crowd_manager
        distances = torch.linalg.vector_norm(crowd.get_world_positions() - robot_positions.unsqueeze(1), dim=-1)
        return distances.masked_fill(~crowd.get_active_mask(), float("inf")).amin(dim=1)

    def capture_terminal_collisions(self, env: Any, reset_env_ids: Any) -> list[dict[str, Any]]:
        """Export collisions among environments about to reset, then clear their histories.

        This compatibility method does not inspect success terminal terms. Evaluator code should
        use :meth:`capture_terminal_episodes` so successful episodes can also be sampled.
        """
        return self.capture_terminal_episodes(env, reset_env_ids, success_env_ids=[])

    def capture_terminal_episodes(
        self, env: Any, reset_env_ids: Any, success_env_ids: Any
    ) -> list[dict[str, Any]]:
        """Export collision context and quota-limited complete successful episodes before reset."""
        import torch

        if self._buffers is None:
            return []
        env_ids = torch.as_tensor(reset_env_ids, device=self._env_ids.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return []
        robot_positions = env.scene["robot"].data.root_pos_w[:, :2]
        collision_mask = env.crowd_manager.get_robot_collision(robot_positions)
        collision_env_ids = env_ids[collision_mask[env_ids]]
        collision_ids = set(collision_env_ids.detach().cpu().tolist())
        assert self._minimum_agent_distances is not None
        episode_minimum_distances = torch.minimum(
            self._minimum_agent_distances, self._minimum_active_pedestrian_distances(env)
        )
        requested_success_ids = set(
            torch.as_tensor(success_env_ids, device=self._env_ids.device, dtype=torch.long).reshape(-1).cpu().tolist()
        )

        exported = []
        if self.record_collisions:
            for env_id in collision_env_ids.detach().cpu().tolist():
                exported.append(
                    self._export_case(
                        env,
                        int(env_id),
                        outcome="collision",
                        minimum_agent_distance_m=episode_minimum_distances[env_id],
                    )
                )
        if self.successes_per_scenario:
            for env_id in env_ids.detach().cpu().tolist():
                if env_id in collision_ids or env_id not in requested_success_ids:
                    continue
                profile = self.profiles[self.env_profile_indices[env_id]]
                if self._successes_by_scenario[profile.scenario] >= self.successes_per_scenario:
                    continue
                if float(episode_minimum_distances[env_id].item()) >= self.interesting_interaction_distance_m:
                    continue
                exported.append(
                    self._export_case(
                        env, env_id, outcome="success", minimum_agent_distance_m=episode_minimum_distances[env_id]
                    )
                )
                self._successes_by_scenario[profile.scenario] += 1
        self.reset(env_ids)
        return exported

    def _export_case(self, env: Any, env_id: int, outcome: str, minimum_agent_distance_m: Any) -> dict[str, Any]:
        if outcome not in ("collision", "success"):
            raise ValueError(f"Unsupported replay outcome: {outcome}")
        frames = self._ordered_frames(env_id, None if outcome == "success" else self.history_frames)
        terminal_frame = self._terminal_frame(env, env_id)
        frames = {name: np.concatenate([values, terminal_frame[name]], axis=0) for name, values in frames.items()}
        profile = self.profiles[self.env_profile_indices[env_id]]
        colliding_agent_ids = self._collision_indices(env, env_id) if outcome == "collision" else []
        case_id = f"{outcome}_{self._next_case_numbers[outcome]:06d}"
        self._next_case_numbers[outcome] += 1
        filename = f"{case_id}.npz"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cases_dir / filename, **frames)
        goal_region_collision = outcome == "collision" and bool(
            np.linalg.norm(frames["robot_position_xy"][-1] - frames["goal_position_xy"][-1])
            <= self.goal_region_radius_m
        )
        automatic_tags = [GOAL_REGION_TAG] if goal_region_collision else []
        if outcome == "success":
            automatic_tags.append(INTERESTING_INTERACTION_TAG)
        entry = {
            "case_id": case_id,
            "scenario": profile.scenario,
            "pedestrian_count": profile.pedestrian_count,
            "environment_id": env_id,
            "outcome": outcome,
            "terminal_time_s": float(frames["time_s"][-1]),
            "collision_time_s": float(frames["time_s"][-1]) if outcome == "collision" else None,
            "colliding_agent_ids": colliding_agent_ids,
            "minimum_agent_distance_m": float(minimum_agent_distance_m.item()),
            "interesting_interaction": outcome == "success",
            "interesting_interaction_distance_m": (
                self.interesting_interaction_distance_m if outcome == "success" else None
            ),
            "goal_region_collision": goal_region_collision,
            "goal_region_radius_m": self.goal_region_radius_m,
            "automatic_tags": automatic_tags,
            "step_dt_s": self.step_dt_s,
            "history_seconds": self.history_seconds,
            "full_episode": outcome == "success",
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
        if self._last_cbf_filtered_command is not None:
            self._last_cbf_filtered_command[ids] = 0.0
        self._minimum_agent_distances[ids] = float("inf")


class InteractionEventReplayRecorder:
    """Quota-limited event clips staged until their parent success is benchmark-counted."""

    schema_version = 1

    def __init__(self, output_dir: str | Path, source: CollisionReplayRecorder, cases_per_label: int, padding_s: float):
        if cases_per_label < 0:
            raise ValueError("cases_per_label must be non-negative.")
        if padding_s <= 0.0:
            raise ValueError("padding_s must be positive.")
        self.output_dir = Path(output_dir)
        self.cases_dir = self.output_dir / "cases"
        self.index_path = self.output_dir / "interaction_event_cases.json"
        self.source = source
        self.cases_per_label = int(cases_per_label)
        self.padding_s = float(padding_s)
        self._pending: dict[int, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._cases: list[dict[str, Any]] = []
        self._next_case_number = 1
        self._write_index()

    @property
    def case_count(self) -> int:
        return len(self._cases)

    def _write_index(self) -> None:
        _write_json_atomically(self.index_path, {
            "schema_version": self.schema_version,
            "padding_s": self.padding_s,
            "cases_per_label": self.cases_per_label,
            "cases": self._cases,
        })

    def stage_terminal_success(self, env: Any, env_id: int, events: Iterable[dict[str, Any]]) -> None:
        """Copy eligible clips before reset; commit only after profile-quota acceptance."""
        if self.cases_per_label == 0:
            return
        # ``events`` are still in the interaction collector's terminal staging area.
        # It adds episode-level metadata only when the episode is accepted, but clips
        # must be copied before the environment reset.  Attach the same stable profile
        # metadata here so the pending clip record is self-contained at commit time.
        profile = self.source.profiles[self.source.env_profile_indices[env_id]]
        terminal = self.source._terminal_frame(env, env_id)
        frames = self.source._ordered_frames(env_id, None)
        frames = {name: np.concatenate([values, terminal[name]], axis=0) for name, values in frames.items()}
        candidates: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
        last_time = float(frames["time_s"][-1])
        for raw_event in events:
            event = {
                **raw_event,
                "environment_id": int(env_id),
                "pedestrian_count": profile.pedestrian_count,
            }
            end_time = float(event["end_time_s"])
            if end_time + self.padding_s > last_time + 1e-8:
                continue
            start_time = float(event["start_time_s"]) - self.padding_s
            mask = (frames["time_s"] >= start_time) & (frames["time_s"] <= end_time + self.padding_s)
            if not bool(np.any(mask)):
                continue
            candidates.append((event, {name: values[mask] for name, values in frames.items()}))
        if candidates:
            self._pending[env_id] = candidates

    def resolve_terminal(self, completed_env_ids: Any, accepted_success_ids: Iterable[int]) -> None:
        successful = {int(env_id) for env_id in accepted_success_ids}
        wrote = False
        for env_id in _ids(completed_env_ids):
            candidates = self._pending.pop(env_id, [])
            if env_id not in successful:
                continue
            for event, frames in candidates:
                key = (str(event["scenario"]), str(event["canonical_label"]))
                if self._counts.get(key, 0) >= self.cases_per_label:
                    continue
                case_id = f"event_{self._next_case_number:06d}"
                self._next_case_number += 1
                self._counts[key] = self._counts.get(key, 0) + 1
                filename = f"{case_id}.npz"
                self.cases_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.cases_dir / filename, **frames)
                self._cases.append({
                    "case_id": case_id,
                    "scenario": event["scenario"],
                    "pedestrian_count": event["pedestrian_count"],
                    "environment_id": event["environment_id"],
                    "pedestrian_id": event["pedestrian_id"],
                    "canonical_label": event["canonical_label"],
                    "start_time_s": event["start_time_s"],
                    "end_time_s": event["end_time_s"],
                    "duration_s": event["duration_s"],
                    "minimum_clearance_m": event["minimum_clearance_m"],
                    "baseline_speed_mps": event["baseline_speed_mps"],
                    "low_event_speed_mps": event["low_event_speed_mps"],
                    "speed_ratio": event["speed_ratio"],
                    "front_crossed": bool(event.get("front_crossed", False)),
                    "front_cross_time_s": event.get("front_cross_time_s"),
                    "front_cross_longitudinal_m": event.get("front_cross_longitudinal_m"),
                    "front_cross_margin_m": event.get("front_cross_margin_m"),
                    "yield_speed_ratio": event["yield_speed_ratio"],
                    "assert_speed_ratio": event["assert_speed_ratio"],
                    "padding_s": self.padding_s,
                    "step_dt_s": self.source.step_dt_s,
                    "replay_file": str(Path("cases") / filename),
                })
                wrote = True
        if wrote:
            self._write_index()


def _sample_standard_deviation(values: Iterable[float]) -> float:
    """Return the sample standard deviation, or zero for fewer than two samples."""
    samples = list(values)
    if len(samples) < 2:
        return 0.0
    mean = sum(samples) / len(samples)
    return math.sqrt(sum((value - mean) ** 2 for value in samples) / (len(samples) - 1))


def dynamic_crowd_profiles(
    counts: Iterable[int] = range(2, 17, 2), *, include_slow_leader: bool = True,
    include_slow_crowd: bool = True,
) -> list[BenchmarkProfile]:
    """Return the normal crowd grid plus the slow-leader and slow-crowd grids.

    The normal grid covers ``crossing``/``with_flow``/``against_flow`` for every count.
    Every slow-leader cell uses the same total pedestrian count as its regular
    with-flow counterpart: slot zero is the deterministic leader and the remaining
    slots retain the normal randomized crowd, so overtaking is measured both in
    isolation and under increasing surrounding density.

    The slow-crowd cells (``crossing_slow``/``against_flow_slow``) repeat the base
    crossing/against-flow layout but drive the ENTIRE crowd at the slow speed band —
    a whole-crowd speed perturbation with no leader.
    """
    counts = tuple(counts)
    ordinary_scenarios = ("crossing", "with_flow", "against_flow")
    profiles = [
        BenchmarkProfile(scenario, count)
        for scenario in ordinary_scenarios
        for count in counts
    ]
    if include_slow_leader:
        profiles.extend(BenchmarkProfile("with_flow_slow_leader", count) for count in counts)
    if include_slow_crowd:
        profiles.extend(
            BenchmarkProfile(scenario, count)
            for scenario in ("crossing_slow", "against_flow_slow")
            for count in counts
        )
    return profiles


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

    This pure helper is also the canonical implementation mirrored by the viewer's live
    reclassification.  Crossing asserts are geometric: the robot crossed through the
    pedestrian's forward region.  Yield uses loss of total planar robot speed, independent
    of whether the robot temporarily moves away from its goal during the maneuver.
    """
    if scenario not in INTERACTION_LABELS:
        raise ValueError(f"Unsupported interaction scenario: {scenario}")
    if not risk_seen:
        return "non_risky_close", None, None
    if scenario in {"with_flow", "with_flow_slow_leader"}:
        if initial_longitudinal_m is None or final_longitudinal_m is None:
            return "unclassified", None, None
        margin = INTERACTION_OVERTAKE_LONGITUDINAL_MARGIN_M
        if initial_longitudinal_m <= -margin and final_longitudinal_m >= margin:
            return "overtake", None, None
        if initial_longitudinal_m <= -margin:
            return "non_overtake", None, None
        return "unclassified", None, None

    if scenario in {"crossing", "crossing_slow"} and front_crossed:
        return "assert", None, None

    speeds = np.asarray(list(event_speeds_mps), dtype=float)
    if (
        duration_s < INTERACTION_MIN_DURATION_S
        or not np.isfinite(baseline_speed_mps)
        or baseline_speed_mps < INTERACTION_MIN_BASELINE_SPEED_MPS
        or speeds.size == 0
    ):
        return "unclassified", None, None
    # A low percentile is robust to small locomotion oscillations while preserving deliberate
    # short waits.  The event collector records one value per evaluation control step.
    low_speed = float(np.percentile(speeds, 10.0))
    ratio = low_speed / baseline_speed_mps
    if ratio < INTERACTION_YIELD_SPEED_RATIO:
        return "yield", low_speed, ratio
    if scenario in {"against_flow", "against_flow_slow"} and ratio > INTERACTION_ASSERT_SPEED_RATIO:
        return "assert", low_speed, ratio
    return "ambiguous", low_speed, ratio


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


def front_lateral_side(lateral_m: float | None) -> int:
    """Return the stable pedestrian-frame side, excluding the hysteresis band."""
    if lateral_m is None or not math.isfinite(lateral_m):
        return 0
    if lateral_m <= -INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M:
        return -1
    if lateral_m >= INTERACTION_FRONT_CROSS_LATERAL_HYSTERESIS_M:
        return 1
    return 0


class InteractionEventCollector:
    """Collect pairwise close/risky interactions and admit only successful episodes.

    Evaluation uses a modest number of vector environments, so compact Python metadata is a
    better fit than adding large permanent GPU buffers.  Live state is sampled before every
    policy action, while completed event records are held until the evaluator confirms the
    episode was both successful and counted by its profile quota.
    """

    def __init__(self, profiles: list[BenchmarkProfile], env_profile_indices: Iterable[int], step_dt_s: float):
        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive.")
        self.profiles = profiles
        self.env_profile_indices = [int(index) for index in env_profile_indices]
        self.step_dt_s = float(step_dt_s)
        self._times = [0.0] * len(self.env_profile_indices)
        self._speed_history: list[list[tuple[float, float]]] = [[] for _ in self.env_profile_indices]
        self._active: list[dict[int, dict[str, Any]]] = [{} for _ in self.env_profile_indices]
        self._completed: list[list[dict[str, Any]]] = [[] for _ in self.env_profile_indices]
        self._pending_terminal: dict[int, list[dict[str, Any]]] = {}
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _risk(robot_pos: np.ndarray, robot_vel: np.ndarray, pedestrian_pos: np.ndarray,
              pedestrian_vel: np.ndarray, surface_radius: float) -> tuple[float, bool]:
        relative_position = pedestrian_pos - robot_pos
        relative_velocity = robot_vel - pedestrian_vel
        speed_sq = float(np.dot(relative_velocity, relative_velocity))
        clearance = float(np.linalg.norm(relative_position) - surface_radius)
        if speed_sq <= 1e-8:
            return clearance, False
        closing = float(np.dot(relative_position, relative_velocity))
        time_to_cpa = closing / speed_sq
        if closing <= 0.0 or time_to_cpa < 0.0 or time_to_cpa > INTERACTION_RISK_HORIZON_S:
            return clearance, False
        cpa_position = relative_position - relative_velocity * time_to_cpa
        cpa_clearance = float(np.linalg.norm(cpa_position) - surface_radius)
        return clearance, cpa_clearance <= INTERACTION_RISK_CLEARANCE_M

    def record_pre_step(self, env: Any) -> None:
        """Sample all active robot-pedestrian pairs before the next physics step."""
        robot = env.scene["robot"]
        crowd = env.crowd_manager
        robot_pos = robot.data.root_pos_w[:, :2].detach().cpu().numpy()
        robot_vel = robot.data.root_lin_vel_w[:, :2].detach().cpu().numpy()
        pedestrian_pos = crowd.get_world_positions().detach().cpu().numpy()
        pedestrian_vel = crowd.get_velocities().detach().cpu().numpy()
        active_mask = crowd.get_active_mask().detach().cpu().numpy()
        radii = crowd.radius.detach().cpu().numpy()
        robot_radius = float(crowd.cfg.robot_radius)

        for env_id, profile_index in enumerate(self.env_profile_indices):
            time_s = self._times[env_id]
            scenario = self.profiles[profile_index].scenario
            # Yield measures translational accommodation, not progress toward the goal: a
            # robot can legitimately move sideways or briefly away from its goal while
            # passing through a crossing interaction.
            speed = float(np.linalg.norm(robot_vel[env_id]))
            history = self._speed_history[env_id]
            history[:] = [(time, value) for time, value in history if time >= time_s - INTERACTION_PRE_SPEED_WINDOW_S]
            baseline_values = [value for _, value in history]
            baseline = float(np.mean(baseline_values)) if baseline_values else float("nan")
            history.append((time_s, speed))

            active_pairs = self._active[env_id]
            active_slots = set(np.flatnonzero(active_mask[env_id]).tolist())
            # A slot becoming inactive cannot form a complete event; drop it rather than
            # inventing an exit classification at recycle/reset.
            for slot in list(active_pairs):
                if slot not in active_slots:
                    del active_pairs[slot]

            for pedestrian_id in active_slots:
                clearance, risky = self._risk(
                    robot_pos[env_id], robot_vel[env_id], pedestrian_pos[env_id, pedestrian_id],
                    pedestrian_vel[env_id, pedestrian_id], robot_radius + radii[env_id, pedestrian_id],
                )
                close = clearance <= INTERACTION_ENTER_CLEARANCE_M
                state = active_pairs.get(pedestrian_id)
                if state is None and (close or risky):
                    pedestrian_speed = float(np.linalg.norm(pedestrian_vel[env_id, pedestrian_id]))
                    initial_longitudinal = None
                    initial_lateral = None
                    pedestrian_direction_xy = None
                    if pedestrian_speed >= INTERACTION_MIN_BASELINE_SPEED_MPS:
                        pedestrian_direction = pedestrian_vel[env_id, pedestrian_id] / pedestrian_speed
                        pedestrian_direction_xy = (float(pedestrian_direction[0]), float(pedestrian_direction[1]))
                        relative_position = robot_pos[env_id] - pedestrian_pos[env_id, pedestrian_id]
                        initial_longitudinal = float(np.dot(relative_position, pedestrian_direction))
                        initial_lateral = float(np.dot(
                            relative_position, np.array([-pedestrian_direction[1], pedestrian_direction[0]])
                        ))
                    active_pairs[pedestrian_id] = {
                        "scenario": scenario,
                        "pedestrian_id": int(pedestrian_id),
                        "start_time_s": time_s,
                        "baseline_speed_mps": baseline,
                        "risk_seen": bool(risky),
                        "minimum_clearance_m": clearance,
                        "event_speeds_mps": [speed],
                        "initial_longitudinal_m": initial_longitudinal,
                        "final_longitudinal_m": initial_longitudinal,
                        "pedestrian_direction_xy": pedestrian_direction_xy,
                        # Keep the last *stable* side rather than the immediately preceding
                        # sample.  A physical crossing traverses the deadband over several
                        # control steps, so adjacent samples cannot be required to straddle it.
                        "previous_front_longitudinal_m": (
                            initial_longitudinal if front_lateral_side(initial_lateral) else None
                        ),
                        "previous_front_lateral_m": initial_lateral if front_lateral_side(initial_lateral) else None,
                        "front_crossed": False,
                        "front_cross_time_s": None,
                        "front_cross_longitudinal_m": None,
                        "front_cross_margin_m": float(
                            robot_radius + radii[env_id, pedestrian_id] + INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M
                        ),
                    }
                    continue
                if state is None:
                    continue

                state["risk_seen"] = bool(state["risk_seen"] or risky)
                state["minimum_clearance_m"] = min(float(state["minimum_clearance_m"]), clearance)
                state["event_speeds_mps"].append(speed)
                direction_xy = state["pedestrian_direction_xy"]
                if direction_xy is not None:
                    pedestrian_direction = np.asarray(direction_xy, dtype=float)
                    relative_position = robot_pos[env_id] - pedestrian_pos[env_id, pedestrian_id]
                    longitudinal = float(np.dot(relative_position, pedestrian_direction))
                    lateral = float(np.dot(relative_position, np.array([-pedestrian_direction[1], pedestrian_direction[0]])))
                    state["final_longitudinal_m"] = longitudinal
                    stable_side = front_lateral_side(lateral)
                    if not state["front_crossed"] and stable_side:
                        front_crossing = front_crossing_longitudinal_m(
                            state["previous_front_longitudinal_m"], state["previous_front_lateral_m"], longitudinal, lateral,
                            float(state["front_cross_margin_m"]),
                        )
                        if front_crossing is not None:
                            state["front_crossed"] = True
                            state["front_cross_time_s"] = time_s
                            state["front_cross_longitudinal_m"] = front_crossing
                    if stable_side:
                        state["previous_front_longitudinal_m"] = longitudinal
                        state["previous_front_lateral_m"] = lateral
                if clearance > INTERACTION_EXIT_CLEARANCE_M and not risky:
                    self._finish_event(env_id, pedestrian_id, time_s)
            self._times[env_id] += self.step_dt_s

    def _finish_event(self, env_id: int, pedestrian_id: int, end_time_s: float) -> None:
        state = self._active[env_id].pop(pedestrian_id)
        duration_s = end_time_s - float(state["start_time_s"])
        label, low_speed, speed_ratio = classify_speed_interaction(
            state["scenario"], bool(state["risk_seen"]), duration_s, float(state["baseline_speed_mps"]),
            state["event_speeds_mps"], state["initial_longitudinal_m"], state["final_longitudinal_m"], state["front_crossed"],
        )
        state.update({
            "end_time_s": end_time_s,
            "duration_s": duration_s,
            "canonical_label": label,
            "low_event_speed_mps": low_speed,
            "speed_ratio": speed_ratio,
            "yield_speed_ratio": INTERACTION_YIELD_SPEED_RATIO,
            "assert_speed_ratio": INTERACTION_ASSERT_SPEED_RATIO,
        })
        del state["event_speeds_mps"]
        del state["pedestrian_direction_xy"]
        del state["previous_front_longitudinal_m"]
        del state["previous_front_lateral_m"]
        self._completed[env_id].append(state)

    def finalize_terminal(self, env_ids: Any) -> None:
        """Stage completed events until the outer evaluator decides which episodes counted."""
        for env_id in _ids(env_ids):
            # Open events are deliberately censored: they have no full post-event context.
            self._active[env_id].clear()
            self._pending_terminal[env_id] = self._completed[env_id]
            self._completed[env_id] = []

    def pending_events(self, env_id: int) -> list[dict[str, Any]]:
        """Return terminal-staged events while the replay ring is still available."""
        return self._pending_terminal.get(int(env_id), [])

    def resolve_terminal(self, completed_env_ids: Any, accepted_success_ids: Iterable[int]) -> list[dict[str, Any]]:
        """Admit staged events only from successful, quota-counted episodes."""
        successful = {int(env_id) for env_id in accepted_success_ids}
        admitted: list[dict[str, Any]] = []
        for env_id in _ids(completed_env_ids):
            candidates = self._pending_terminal.pop(env_id, [])
            if env_id in successful:
                profile = self.profiles[self.env_profile_indices[env_id]]
                for event in candidates:
                    admitted_event = {
                        **event,
                        "environment_id": env_id,
                        "pedestrian_count": profile.pedestrian_count,
                    }
                    self.events.append(admitted_event)
                    admitted.append(admitted_event)
            self._times[env_id] = 0.0
            self._speed_history[env_id].clear()
        return admitted

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for scenario in SCENARIO_ORDER:
            labels = INTERACTION_LABELS[scenario]
            for label in labels:
                rows.append({
                    "scenario": scenario,
                    "label": label,
                    "events": sum(1 for event in self.events if event["scenario"] == scenario and event["canonical_label"] == label),
                })
        return rows


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
        timeout_term: str = "time_out",
        base_contact_term: str = "base_contact",
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
        self.timeout_ids_key = f"Episode_Termination/Envs/Ids/{timeout_term}"
        self.base_contact_ids_key = f"Episode_Termination/Envs/Ids/{base_contact_term}"
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
        self._goal_region_collisions = [0] * len(profiles)
        self._timeouts = [0] * len(profiles)
        self._base_contacts = [0] * len(profiles)
        self._velocity_sums = [0.0] * len(profiles)
        # Retain episode-level speed means to report their variation across episodes.
        self._velocity_values: list[list[float]] = [[] for _ in profiles]
        # Populated by ``consume`` so side collectors can use exactly the same capped episode
        # set as the primary benchmark metrics.
        self.last_accepted_ids: set[int] = set()
        self.last_accepted_success_ids: set[int] = set()
        # Multi-seed stage bookkeeping: per-profile acceptance caps for the current seed
        # chunk, plus cumulative-count boundaries recorded at the start of every stage so
        # per-seed results can be reconstructed as deltas between consecutive boundaries.
        self._stage_limit = [episodes_per_profile] * len(profiles)
        self._stage_boundaries: list[list[dict[str, int | float]]] = []

    @property
    def complete(self) -> bool:
        return all(episodes >= self.episodes_per_profile for episodes in self._episodes)

    @property
    def stage_complete(self) -> bool:
        """Whether every profile reached the current seed stage's acceptance cap."""
        return all(episodes >= limit for episodes, limit in zip(self._episodes, self._stage_limit))

    @property
    def total_episodes(self) -> int:
        return sum(self._episodes)

    def consume(
        self,
        extras: dict[str, Any],
        velocity_by_env: Mapping[int, float] | None = None,
        completed_env_ids: Any | None = None,
        goal_region_collision_env_ids: Any | None = None,
    ) -> int:
        """Consume completed episodes from one environment step and return accepted count.

        ``completed_env_ids`` should be supplied from the vector-environment done mask when it
        is available. Isaac Lab clears idle ``Episode_Termination/...`` log fields to the scalar
        ``0``; that value is a metric placeholder, not a completion of environment zero.
        """
        log = extras.get("log", {})
        completed_ids = _ids(completed_env_ids) if completed_env_ids is not None else completed_environment_ids(extras)
        self.last_accepted_ids = set()
        self.last_accepted_success_ids = set()
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
        timeout_ids = _ids(log.get(self.timeout_ids_key))
        base_contact_ids = _ids(log.get(self.base_contact_ids_key))
        goal_region_collision_ids = _ids(goal_region_collision_env_ids)
        accepted = 0
        for env_id in sorted(completed_ids):
            if env_id < 0 or env_id >= len(self.env_profile_indices):
                raise IndexError(f"Termination reported invalid environment ID {env_id}.")
            profile_index = self.env_profile_indices[env_id]
            if self._episodes[profile_index] >= self._stage_limit[profile_index]:
                continue
            if env_id not in metric_by_env:
                raise KeyError(f"Missing velocity metric for completed environment {env_id}.")

            self._episodes[profile_index] += 1
            self._velocity_sums[profile_index] += metric_by_env[env_id]
            # Collision takes precedence when both terms trigger on the same final step.
            if env_id in collision_ids:
                if env_id in goal_region_collision_ids:
                    self._goal_region_collisions[profile_index] += 1
                else:
                    self._collisions[profile_index] += 1
            elif env_id in success_ids:
                self._successes[profile_index] += 1
                self.last_accepted_success_ids.add(env_id)
            if env_id in timeout_ids:
                self._timeouts[profile_index] += 1
            if env_id in base_contact_ids:
                self._base_contacts[profile_index] += 1
            self._velocity_values[profile_index].append(metric_by_env[env_id])
            self.last_accepted_ids.add(env_id)
            accepted += 1
        return accepted

    def rows(self) -> list[dict[str, Any]]:
        """Return one normalized result row for every profile."""
        return [
            _result_row(profile, counts)
            for profile, counts in zip(self.profiles, self.snapshot_counts())
        ]

    def aggregate_rows(self) -> list[dict[str, Any]]:
        """Return pooled per-episode aggregates for every scenario."""
        return _aggregate_rows_from_counts(self.profiles, self.snapshot_counts())

    def snapshot_counts(self) -> list[dict[str, int | float]]:
        """Return cumulative per-profile outcome counts (the authoritative metric source)."""
        return [
            {
                "episodes": self._episodes[index],
                "successes": self._successes[index],
                "collisions": self._collisions[index],
                "goal_region_collisions": self._goal_region_collisions[index],
                "timeouts": self._timeouts[index],
                "base_contacts": self._base_contacts[index],
                "velocity_sum": self._velocity_sums[index],
                "velocity_values": list(self._velocity_values[index]),
            }
            for index in range(len(self.profiles))
        ]

    def set_stage_limit(self, limit: int) -> None:
        """Cap per-profile acceptance at ``limit`` for the current seed stage.

        Called at the start of every stage; the recorded boundary captures the
        cumulative counts reached at the end of the previous stage (all-zero for
        the first stage).  Consumed by :meth:`per_seed_counts`.
        """
        self._stage_boundaries.append(self.snapshot_counts())
        self._stage_limit = [min(int(limit), self.episodes_per_profile)] * len(self.profiles)

    def per_seed_counts(self, seeds: list[int]) -> list[list[dict[str, int | float]]]:
        """Return per-seed per-profile outcome counts (deltas between stage boundaries).

        Requires :meth:`set_stage_limit` at the start of every stage and a final
        :meth:`snapshot_counts` boundary recorded by the caller after the last stage.
        Episodes already in flight when a stage boundary is crossed are attributed to
        the stage in which they complete (at most one episode per profile).
        """
        boundaries = [*self._stage_boundaries, self.snapshot_counts()]
        per_seed: list[list[dict[str, int | float]]] = []
        for seed_index in range(len(seeds)):
            previous = boundaries[seed_index]
            current = boundaries[seed_index + 1]
            seed_counts = []
            for index in range(len(self.profiles)):
                counts = {
                    key: current[index][key] - previous[index][key]
                    for key in current[index] if key != "velocity_values"
                }
                counts["velocity_values"] = current[index]["velocity_values"][int(previous[index]["episodes"]):]
                seed_counts.append(counts)
            per_seed.append(seed_counts)
        return per_seed

    def per_seed_rows(self, seeds: list[int]) -> list[list[dict[str, Any]]]:
        """One normalized result row per profile per seed."""
        return [
            [_result_row(profile, counts[index], seed=seed) for index, profile in enumerate(self.profiles)]
            for seed, counts in zip(seeds, self.per_seed_counts(seeds))
        ]

    def per_seed_aggregate_rows(self, seeds: list[int]) -> list[list[dict[str, Any]]]:
        """Pooled per-scenario aggregate rows per seed."""
        return [
            _aggregate_rows_from_counts(self.profiles, counts, seed=seed)
            for seed, counts in zip(seeds, self.per_seed_counts(seeds))
        ]


def _result_row(
    profile: BenchmarkProfile,
    counts: Mapping[str, int | float],
    seed: int | None = None,
) -> dict[str, Any]:
    """Build one normalized result row from per-profile outcome counts."""
    episodes = int(counts["episodes"])
    successes = int(counts["successes"])
    collisions = int(counts["collisions"])
    goal_region_collisions = int(counts["goal_region_collisions"])
    timeouts = int(counts["timeouts"])
    base_contacts = int(counts["base_contacts"])
    navigation_episodes = episodes - goal_region_collisions
    velocity_values = list(counts.get("velocity_values", []))
    row = {
        **asdict(profile),
        "episodes": episodes,
        "successes": successes,
        "collisions": collisions,
        "goal_region_collisions": goal_region_collisions,
        "all_collisions": collisions + goal_region_collisions,
        "timeouts": timeouts,
        "base_contacts": base_contacts,
        "success_rate": successes / episodes if episodes else 0.0,
        "navigation_success_rate": successes / navigation_episodes if navigation_episodes else 0.0,
        "collision_rate": collisions / episodes if episodes else 0.0,
        "goal_region_collision_rate": goal_region_collisions / episodes if episodes else 0.0,
        "all_collision_rate": (collisions + goal_region_collisions) / episodes if episodes else 0.0,
        "timeout_rate": timeouts / episodes if episodes else 0.0,
        "base_contact_rate": base_contacts / episodes if episodes else 0.0,
        "mean_xy_speed_mps": float(counts["velocity_sum"]) / episodes if episodes else 0.0,
        "std_xy_speed_mps": _sample_standard_deviation(velocity_values),
    }
    if seed is not None:
        row["seed"] = seed
    return row


def _aggregate_rows_from_counts(
    profiles: list[BenchmarkProfile],
    counts: list[Mapping[str, int | float]],
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build pooled per-scenario aggregate rows from per-profile outcome counts."""
    aggregates = []
    for scenario in SCENARIO_ORDER:
        profile_indices = [index for index, profile in enumerate(profiles) if profile.scenario == scenario]
        if not profile_indices:
            continue
        episodes = sum(int(counts[index]["episodes"]) for index in profile_indices)
        successes = sum(int(counts[index]["successes"]) for index in profile_indices)
        collisions = sum(int(counts[index]["collisions"]) for index in profile_indices)
        goal_region_collisions = sum(int(counts[index]["goal_region_collisions"]) for index in profile_indices)
        timeouts = sum(int(counts[index]["timeouts"]) for index in profile_indices)
        base_contacts = sum(int(counts[index]["base_contacts"]) for index in profile_indices)
        navigation_episodes = episodes - goal_region_collisions
        velocity_values = [
            value for index in profile_indices for value in counts[index].get("velocity_values", [])
        ]
        aggregates.append(
            {
                "scenario": scenario,
                "pedestrian_count": "all",
                "episodes": episodes,
                "successes": successes,
                "collisions": collisions,
                "goal_region_collisions": goal_region_collisions,
                "all_collisions": collisions + goal_region_collisions,
                "timeouts": timeouts,
                "base_contacts": base_contacts,
                "success_rate": successes / episodes if episodes else 0.0,
                "navigation_success_rate": successes / navigation_episodes if navigation_episodes else 0.0,
                "collision_rate": collisions / episodes if episodes else 0.0,
                "goal_region_collision_rate": goal_region_collisions / episodes if episodes else 0.0,
                "all_collision_rate": (collisions + goal_region_collisions) / episodes if episodes else 0.0,
                "timeout_rate": timeouts / episodes if episodes else 0.0,
                "base_contact_rate": base_contacts / episodes if episodes else 0.0,
                "mean_xy_speed_mps": (
                    sum(float(counts[index]["velocity_sum"]) for index in profile_indices) / episodes if episodes else 0.0
                ),
                "std_xy_speed_mps": _sample_standard_deviation(velocity_values),
            }
        )
        if seed is not None:
            aggregates[-1]["seed"] = seed
    return aggregates


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
    """Write CSV, JSON, and the dynamic-crowd summary plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_rows = [*rows, *aggregate_rows]
    fieldnames = [
        "scenario", "pedestrian_count", "episodes", "successes", "collisions", "goal_region_collisions",
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
        "front_cross_margin_m", "yield_speed_ratio", "assert_speed_ratio",
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
        "schema_version": 1,
        "detector": {
            "enter_clearance_m": INTERACTION_ENTER_CLEARANCE_M,
            "exit_clearance_m": INTERACTION_EXIT_CLEARANCE_M,
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


def _save_interaction_histogram(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {(row["scenario"], row["label"]): int(row["events"]) for row in summary_rows}
    figure, axes = plt.subplots(1, len(SCENARIO_ORDER), figsize=(4.5 * len(SCENARIO_ORDER), 4.5))
    for axis, scenario in zip(axes, SCENARIO_ORDER):
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
