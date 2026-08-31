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
INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M = 4.0
INTERACTION_AGAINST_FLOW_CONE_HALF_ANGLE_RAD = math.radians(30.0)
INTERACTION_AGAINST_FLOW_CONE_LATERAL_BUFFER_M = 0.3
INTERACTION_AGAINST_FLOW_CONE_PERSISTENCE_S = 0.2
INTERACTION_AGAINST_FLOW_SIDESTEP_LATERAL_M = 0.4
INTERACTION_AGAINST_FLOW_REAR_PASS_MARGIN_M = 0.5

INTERACTION_LABELS = {
    "crossing": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
    "against_flow": ("sidestep", "straight_pass", "front_crossing"),
    "with_flow": ("overtake", "non_overtake", "non_risky_close", "unclassified"),
    "crossing_slow": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
    "against_flow_slow": ("sidestep", "straight_pass", "front_crossing"),
}
"""Pairwise interaction labels. Slow-leader evaluation uses episode outcomes instead."""

INTERACTION_SCENARIO_ORDER = tuple(scenario for scenario in SCENARIO_ORDER if scenario in INTERACTION_LABELS)

SLOW_LEADER_SCENARIO = "with_flow_slow_leader"
SLOW_LEADER_SLOT = 0
SLOW_LEADER_OVERTAKE_MARGIN_M = 0.5
SLOW_LEADER_OUTCOME_LABELS = ("Follow", "Overtake")


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
        retain_full_episode: bool = False,
    ):
        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive.")
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive.")
        if goal_region_radius_m <= 0.0:
            raise ValueError("goal_region_radius_m must be positive.")
        if successes_per_scenario < 0:
            raise ValueError("successes_per_scenario must be non-negative.")
        if (successes_per_scenario or retain_full_episode) and (episode_length_s is None or episode_length_s <= 0.0):
            raise ValueError("episode_length_s must be positive when retaining complete episodes.")
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
        self.retain_full_episode = bool(retain_full_episode)
        # The terminal frame is added only when exporting, so the ring itself contains exactly
        # the requested leading history or the complete pre-terminal successful episode.
        self.history_frames = math.ceil(self.history_seconds / self.step_dt_s - 1e-9)
        self.full_episode_frames = (
            math.ceil(self.episode_length_s / self.step_dt_s - 1e-9)
            if self.episode_length_s is not None and (self.successes_per_scenario or self.retain_full_episode)
            else 0
        )
        self.capacity = max(self.history_frames, self.full_episode_frames)

        self._buffers: dict[str, Any] | None = None
        self._write_indices = None
        self._counts = None
        self._elapsed_steps = None
        self._last_command = None
        self._last_cbf_filtered_command = None
        self._last_cbf_nominal_acceleration = None
        self._last_cbf_filtered_acceleration = None
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
                "retain_full_episode": self.retain_full_episode,
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

    def record_cbf_accelerations(self, nominal_acceleration_body: Any, filtered_acceleration_xy_world: Any) -> None:
        """Attach the CBF nominal and QP-filtered planar accelerations to the latest replay frame.

        The nominal acceleration is the bounded body-frame Kp value
        ``Kp * (v_nav - v_robot)``.  The filtered value is the world-frame
        acceleration returned by the CBF-QP.  They are recorded after stepping,
        alongside the filtered velocity command, so they describe the same
        controller update.
        """
        import torch

        if self._buffers is None:
            return
        assert self._write_indices is not None and self._counts is not None
        if nominal_acceleration_body.ndim != 2 or nominal_acceleration_body.shape[1] < 2:
            raise ValueError("CBF nominal acceleration must have shape (num_envs, at least 2).")
        if filtered_acceleration_xy_world.ndim != 2 or filtered_acceleration_xy_world.shape[1] < 2:
            raise ValueError("CBF filtered acceleration must have shape (num_envs, at least 2).")
        num_envs = len(self.env_profile_indices)
        if nominal_acceleration_body.shape[0] != num_envs or filtered_acceleration_xy_world.shape[0] != num_envs:
            raise ValueError("CBF accelerations must contain one row per vector environment.")
        if "cbf_nominal_acceleration_body" not in self._buffers:
            device = nominal_acceleration_body.device
            self._buffers["cbf_nominal_acceleration_body"] = torch.zeros(num_envs, self.capacity, 2, device=device)
            self._buffers["cbf_filtered_acceleration_xy_world"] = torch.zeros(
                num_envs, self.capacity, 2, device=device
            )
            self._last_cbf_nominal_acceleration = torch.zeros(num_envs, 2, device=device)
            self._last_cbf_filtered_acceleration = torch.zeros(num_envs, 2, device=device)
        assert self._last_cbf_nominal_acceleration is not None
        assert self._last_cbf_filtered_acceleration is not None
        active = self._counts > 0
        if not torch.any(active):
            return
        env_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
        indices = (self._write_indices[env_ids] - 1) % self.capacity
        nominal = nominal_acceleration_body[:, :2]
        filtered = filtered_acceleration_xy_world[:, :2]
        self._buffers["cbf_nominal_acceleration_body"][env_ids, indices] = nominal[env_ids]
        self._buffers["cbf_filtered_acceleration_xy_world"][env_ids, indices] = filtered[env_ids]
        self._last_cbf_nominal_acceleration[env_ids] = nominal[env_ids]
        self._last_cbf_filtered_acceleration[env_ids] = filtered[env_ids]

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
        if self._last_cbf_nominal_acceleration is not None:
            terminal["cbf_nominal_acceleration_body"] = (
                self._last_cbf_nominal_acceleration[env_id : env_id + 1].detach().cpu().numpy()
            )
            terminal["cbf_filtered_acceleration_xy_world"] = (
                self._last_cbf_filtered_acceleration[env_id : env_id + 1].detach().cpu().numpy()
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
        if self._last_cbf_nominal_acceleration is not None:
            self._last_cbf_nominal_acceleration[ids] = 0.0
            self._last_cbf_filtered_acceleration[ids] = 0.0
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
            # Slow-leader evaluation exports only robust slot-zero passes. Follow is an
            # outcome report, not an interaction replay candidate.
            if (
                raw_event.get("scenario") == SLOW_LEADER_SCENARIO
                and raw_event.get("canonical_label") != "overtake"
            ):
                continue
            event = {
                **raw_event,
                "environment_id": int(env_id),
                "pedestrian_count": profile.pedestrian_count,
            }
            if event.get("start_time_s") is None or event.get("end_time_s") is None:
                continue
            end_time = float(event["end_time_s"])
            if end_time + self.padding_s > last_time + 1e-8:
                continue
            start_time = float(event["start_time_s"]) - self.padding_s
            mask = (frames["time_s"] >= start_time) & (frames["time_s"] <= end_time + self.padding_s)
            if not bool(np.any(mask)):
                continue
            candidates.append((event, {name: values[mask] for name, values in frames.items()}))
        if candidates:
            self._pending.setdefault(env_id, []).extend(candidates)

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
                    "baseline_speed_mps": event.get("baseline_speed_mps"),
                    "low_event_speed_mps": event.get("low_event_speed_mps"),
                    "speed_ratio": event.get("speed_ratio"),
                    "outcome": event.get("outcome"),
                    "front_crossed": bool(event.get("front_crossed", False)),
                    "front_cross_time_s": event.get("front_cross_time_s"),
                    "front_cross_longitudinal_m": event.get("front_cross_longitudinal_m"),
                    "front_cross_margin_m": event.get("front_cross_margin_m"),
                    "yield_speed_ratio": event.get("yield_speed_ratio"),
                    "assert_speed_ratio": event.get("assert_speed_ratio"),
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
    reclassification.  This is the speed-based protocol retained for against-flow and
    with-flow interactions.  Crossing scenarios use :func:`classify_crossing_interaction`
    so their yield label is determined by pedestrian-frame geometry instead.
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
        # Against-flow behavior begins before the 1.5 m interaction event.  This map
        # preserves that lead-in without changing the close/risky event lifecycle.
        self._encounters: list[dict[int, dict[str, Any]]] = [{} for _ in self.env_profile_indices]
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

    @staticmethod
    def _pair_is_closing(
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        pedestrian_pos: np.ndarray,
        pedestrian_vel: np.ndarray,
    ) -> bool:
        """Whether this specific robot--pedestrian pair is still approaching.

        The CPA-risk threshold is deliberately stricter than the interaction exit
        clearance.  Therefore, loss of CPA risk alone cannot establish that an
        interaction is over: a pair can still be closing and become close again a
        few control steps later.  This predicate is pairwise and uses the same
        relative-position/velocity convention as :meth:`_risk`.
        """
        relative_position = pedestrian_pos - robot_pos
        relative_velocity = robot_vel - pedestrian_vel
        return float(np.dot(relative_position, relative_velocity)) > 0.0

    @staticmethod
    def _encounter_coordinates(
        encounter: Mapping[str, Any], robot_position: np.ndarray, pedestrian_position: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return frozen-heading longitudinal, relative lateral, and robot lateral motion."""
        direction = np.asarray(encounter["pedestrian_direction_xy"], dtype=float)
        lateral_axis = np.array([-direction[1], direction[0]])
        relative_position = robot_position - pedestrian_position
        robot_displacement = robot_position - np.asarray(encounter["robot_position_at_acquisition_xy"], dtype=float)
        return (
            float(np.dot(relative_position, direction)),
            float(np.dot(relative_position, lateral_axis)),
            float(np.dot(robot_displacement, lateral_axis)),
        )

    def _start_or_update_against_flow_encounter(
        self,
        env_id: int,
        pedestrian_id: int,
        time_s: float,
        robot_position: np.ndarray,
        robot_velocity: np.ndarray,
        pedestrian_position: np.ndarray,
        pedestrian_velocity: np.ndarray,
        has_active_event: bool,
    ) -> dict[str, Any] | None:
        """Maintain one early-approach tracker for an against-flow pedestrian slot."""
        encounters = self._encounters[env_id]
        encounter = encounters.get(pedestrian_id)
        center_distance = float(np.linalg.norm(pedestrian_position - robot_position))
        relative_position = pedestrian_position - robot_position
        relative_velocity = robot_velocity - pedestrian_velocity
        approaching = float(np.dot(relative_position, relative_velocity)) > 0.0
        pedestrian_speed = float(np.linalg.norm(pedestrian_velocity))
        if encounter is None:
            if (
                center_distance > INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M
                or not approaching
                or pedestrian_speed < INTERACTION_MIN_BASELINE_SPEED_MPS
            ):
                return None
            direction = pedestrian_velocity / pedestrian_speed
            encounter = {
                "acquisition_time_s": time_s,
                "pedestrian_direction_xy": (float(direction[0]), float(direction[1])),
                "robot_position_at_acquisition_xy": (float(robot_position[0]), float(robot_position[1])),
                "front_cone_duration_s": 0.0,
                "front_cone_qualified": False,
                "front_cone_qualified_time_s": None,
                "robot_lateral_displacement_m": 0.0,
                "robust_pass_seen": False,
                "robust_pass_time_s": None,
                "passing_side": 0,
                "pass_lateral_m": None,
                "pass_robot_lateral_displacement_m": 0.0,
            }
            encounters[pedestrian_id] = encounter

        # A live close/risky event owns the tracker until it finishes.  Before event
        # entry, a pedestrian that has cleanly separated can begin a fresh encounter.
        if center_distance > INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M and not has_active_event:
            del encounters[pedestrian_id]
            return None

        longitudinal, lateral, robot_lateral = self._encounter_coordinates(
            encounter, robot_position, pedestrian_position
        )
        encounter["robot_lateral_displacement_m"] = robot_lateral
        cone_half_width = (
            longitudinal * math.tan(INTERACTION_AGAINST_FLOW_CONE_HALF_ANGLE_RAD)
            + INTERACTION_AGAINST_FLOW_CONE_LATERAL_BUFFER_M
        )
        in_front_cone = 0.0 < longitudinal <= INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M and abs(lateral) <= cone_half_width
        if in_front_cone:
            encounter["front_cone_duration_s"] = float(encounter["front_cone_duration_s"]) + self.step_dt_s
            if (
                not encounter["front_cone_qualified"]
                and float(encounter["front_cone_duration_s"]) >= INTERACTION_AGAINST_FLOW_CONE_PERSISTENCE_S
            ):
                encounter["front_cone_qualified"] = True
                encounter["front_cone_qualified_time_s"] = time_s
        else:
            encounter["front_cone_duration_s"] = 0.0

        if not encounter["robust_pass_seen"] and longitudinal <= -INTERACTION_AGAINST_FLOW_REAR_PASS_MARGIN_M:
            passing_side = front_lateral_side(lateral)
            if passing_side:
                encounter["robust_pass_seen"] = True
                encounter["robust_pass_time_s"] = time_s
                encounter["passing_side"] = int(passing_side)
                encounter["pass_lateral_m"] = lateral
                encounter["pass_robot_lateral_displacement_m"] = robot_lateral
        return encounter

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
            # Slow-leader is deliberately evaluated as one slot-zero episode outcome,
            # not as a collection of pairwise crowd interactions.
            if scenario == SLOW_LEADER_SCENARIO:
                self._times[env_id] += self.step_dt_s
                continue
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
            encounters = self._encounters[env_id]
            active_slots = set(np.flatnonzero(active_mask[env_id]).tolist())
            # A slot becoming inactive cannot form a complete event; drop it rather than
            # inventing an exit classification at recycle/reset.
            for slot in list(active_pairs):
                if slot not in active_slots:
                    del active_pairs[slot]
            for slot in list(encounters):
                if slot not in active_slots:
                    del encounters[slot]

            for pedestrian_id in active_slots:
                state = active_pairs.get(pedestrian_id)
                encounter = None
                if scenario in {"against_flow", "against_flow_slow"}:
                    encounter = self._start_or_update_against_flow_encounter(
                        env_id,
                        pedestrian_id,
                        time_s,
                        robot_pos[env_id],
                        robot_vel[env_id],
                        pedestrian_pos[env_id, pedestrian_id],
                        pedestrian_vel[env_id, pedestrian_id],
                        state is not None,
                    )
                clearance, risky = self._risk(
                    robot_pos[env_id], robot_vel[env_id], pedestrian_pos[env_id, pedestrian_id],
                    pedestrian_vel[env_id, pedestrian_id], robot_radius + radii[env_id, pedestrian_id],
                )
                close = clearance <= INTERACTION_ENTER_CLEARANCE_M
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
                    initial_stable_side = front_lateral_side(initial_lateral)
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
                            initial_longitudinal if initial_stable_side else None
                        ),
                        "previous_front_lateral_m": initial_lateral if initial_stable_side else None,
                        "front_crossed": False,
                        "front_cross_time_s": None,
                        "front_cross_longitudinal_m": None,
                        "front_cross_margin_m": float(
                            robot_radius + radii[env_id, pedestrian_id] + INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M
                        ),
                        # Yield geometry spans the complete active event, including its exit
                        # hysteresis.  A rear pass before the event formally ends remains an
                        # interaction outcome, just like the existing front assertion.
                        "core_sample_count": 1,
                        "core_resolved_sides": [int(initial_stable_side)] if initial_stable_side else [],
                        "previous_rear_longitudinal_m": initial_longitudinal if initial_stable_side else None,
                        "previous_rear_lateral_m": initial_lateral if initial_stable_side else None,
                        "rear_crossed": False,
                        "rear_cross_time_s": None,
                        "rear_cross_longitudinal_m": None,
                        "rear_cross_margin_m": float(
                            robot_radius + radii[env_id, pedestrian_id] + INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M
                        ),
                        "yield_geometry_available": pedestrian_direction_xy is not None,
                        # Store the mutable tracker rather than a snapshot: early movement
                        # before event entry and the pass itself both inform one outcome.
                        "against_flow_encounter": encounter,
                    }
                    continue
                if state is None:
                    continue

                state["risk_seen"] = bool(state["risk_seen"] or risky)
                state["minimum_clearance_m"] = min(float(state["minimum_clearance_m"]), clearance)
                state["event_speeds_mps"].append(speed)
                state["core_sample_count"] += 1
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
                    if stable_side:
                        state["core_resolved_sides"].append(int(stable_side))
                        if not state["rear_crossed"]:
                            rear_crossing = rear_crossing_longitudinal_m(
                                state["previous_rear_longitudinal_m"], state["previous_rear_lateral_m"],
                                longitudinal, lateral, float(state["rear_cross_margin_m"]),
                            )
                            if rear_crossing is not None:
                                state["rear_crossed"] = True
                                state["rear_cross_time_s"] = time_s
                                state["rear_cross_longitudinal_m"] = rear_crossing
                        state["previous_rear_longitudinal_m"] = longitudinal
                        state["previous_rear_lateral_m"] = lateral
                # A temporary loss of CPA risk must not split one physical maneuver into
                # two events while this same pair is still approaching.  In particular,
                # the early CPA-warning segment and the later front-side traversal must
                # share one crossing label.
                pair_is_closing = self._pair_is_closing(
                    robot_pos[env_id], robot_vel[env_id],
                    pedestrian_pos[env_id, pedestrian_id], pedestrian_vel[env_id, pedestrian_id],
                )
                if clearance > INTERACTION_EXIT_CLEARANCE_M and not risky and not pair_is_closing:
                    self._finish_event(env_id, pedestrian_id, time_s)
            self._times[env_id] += self.step_dt_s

    def _finish_event(self, env_id: int, pedestrian_id: int, end_time_s: float) -> None:
        state = self._active[env_id].pop(pedestrian_id)
        duration_s = end_time_s - float(state["start_time_s"])
        low_speed, speed_ratio = interaction_speed_diagnostics(
            duration_s, float(state["baseline_speed_mps"]), state["event_speeds_mps"]
        )
        if state["scenario"] in {"crossing", "crossing_slow"}:
            label = classify_crossing_interaction(
                bool(state["risk_seen"]), bool(state["front_crossed"]), bool(state["yield_geometry_available"]),
                state["core_resolved_sides"], bool(state["rear_crossed"]),
            )
        elif state["scenario"] in {"against_flow", "against_flow_slow"}:
            encounter = state["against_flow_encounter"]
            label = classify_against_flow_interaction(bool(state["front_crossed"]), encounter)
            if encounter is None:
                state.update({
                    "encounter_acquisition_time_s": None,
                    "encounter_pedestrian_direction_xy": None,
                    "forward_cone_qualified": False,
                    "forward_cone_qualified_time_s": None,
                    "robot_lateral_displacement_m": None,
                    "robust_pass_seen": False,
                    "robust_pass_time_s": None,
                    "passing_side": 0,
                    "pass_lateral_m": None,
                    "pass_robot_lateral_displacement_m": None,
                })
            else:
                state.update({
                    "encounter_acquisition_time_s": encounter["acquisition_time_s"],
                    "encounter_pedestrian_direction_xy": encounter["pedestrian_direction_xy"],
                    "forward_cone_qualified": encounter["front_cone_qualified"],
                    "forward_cone_qualified_time_s": encounter["front_cone_qualified_time_s"],
                    "robot_lateral_displacement_m": encounter["robot_lateral_displacement_m"],
                    "robust_pass_seen": encounter["robust_pass_seen"],
                    "robust_pass_time_s": encounter["robust_pass_time_s"],
                    "passing_side": encounter["passing_side"],
                    "pass_lateral_m": encounter["pass_lateral_m"],
                    "pass_robot_lateral_displacement_m": encounter["pass_robot_lateral_displacement_m"],
                })
        else:
            label, _, _ = classify_speed_interaction(
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
        del state["previous_rear_longitudinal_m"]
        del state["previous_rear_lateral_m"]
        del state["against_flow_encounter"]
        state["core_resolved_side_count"] = len(state["core_resolved_sides"])
        self._completed[env_id].append(state)

    def finalize_terminal(self, env_ids: Any) -> None:
        """Stage completed events until the outer evaluator decides which episodes counted."""
        for env_id in _ids(env_ids):
            # Open events are deliberately censored: they have no full post-event context.
            self._active[env_id].clear()
            self._encounters[env_id].clear()
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
        for scenario in INTERACTION_SCENARIO_ORDER:
            labels = INTERACTION_LABELS[scenario]
            for label in labels:
                rows.append({
                    "scenario": scenario,
                    "label": label,
                    "events": sum(1 for event in self.events if event["scenario"] == scenario and event["canonical_label"] == label),
                })
        return rows


class SlowLeaderOutcomeCollector:
    """Classify accepted successful slow-leader episodes as Follow or Overtake.

    The fixed slow leader is pedestrian slot zero.  Unlike the ordinary interaction
    protocol, this tracker intentionally ignores all other pedestrians and produces
    exactly one outcome for a successful slow-leader episode.
    """

    def __init__(self, profiles: list[BenchmarkProfile], env_profile_indices: Iterable[int], step_dt_s: float):
        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive.")
        self.profiles = profiles
        self.env_profile_indices = [int(index) for index in env_profile_indices]
        if not self.env_profile_indices or any(
            index < 0 or index >= len(profiles) for index in self.env_profile_indices
        ):
            raise ValueError("Every vector environment must be assigned a valid profile index.")
        self.step_dt_s = float(step_dt_s)
        self._times = [0.0] * len(self.env_profile_indices)
        self._active: list[dict[str, Any] | None] = [None] * len(self.env_profile_indices)
        self._pending_terminal: dict[int, dict[str, Any]] = {}
        self.outcomes: list[dict[str, Any]] = []

    def _is_slow_leader_env(self, env_id: int) -> bool:
        return self.profiles[self.env_profile_indices[env_id]].scenario == SLOW_LEADER_SCENARIO

    @staticmethod
    def _flow_direction(crowd: Any, env_id: int) -> float:
        """Return the corridor's longitudinal world-X direction for one environment."""
        flow_dir = getattr(crowd, "flow_dir", None)
        if flow_dir is None:
            return 1.0
        value = float(flow_dir[env_id].detach().cpu().item())
        return 1.0 if value >= 0.0 else -1.0

    @staticmethod
    def _corridor_progress(crowd: Any, env_id: int, position_xy: np.ndarray, flow_direction: float) -> float | None:
        """Project a leader location along flow for recycle detection when geometry is available."""
        origin = getattr(crowd, "corridor_origin", None)
        if origin is None:
            return None
        origin_xy = origin[env_id].detach().cpu().numpy()
        return float((position_xy[0] - origin_xy[0]) * flow_direction)

    @staticmethod
    def _recycled(
        previous_progress: float | None,
        progress: float | None,
        crowd: Any,
        env_id: int,
    ) -> bool:
        """Identify the discontinuous upstream jump applied by the crowd recycler."""
        corridor_length = getattr(crowd, "corridor_length", None)
        if previous_progress is None or progress is None or corridor_length is None:
            return False
        length = float(corridor_length[env_id].detach().cpu().item())
        return length > 0.0 and progress < previous_progress - 0.5 * length

    def _observe(self, env: Any, env_id: int, conditions: Mapping[str, float] | None = None) -> None:
        if not self._is_slow_leader_env(env_id):
            return
        crowd = env.crowd_manager
        robot_position = env.scene["robot"].data.root_pos_w[env_id, :2].detach().cpu().numpy()
        leader_position = crowd.get_world_positions()[env_id, SLOW_LEADER_SLOT].detach().cpu().numpy()
        leader_active = bool(crowd.get_active_mask()[env_id, SLOW_LEADER_SLOT].detach().cpu().item())
        flow_direction = self._flow_direction(crowd, env_id)
        time_s = self._times[env_id]

        state = self._active[env_id]
        if state is None:
            state = {
                "scenario": SLOW_LEADER_SCENARIO,
                "pedestrian_id": SLOW_LEADER_SLOT,
                "start_time_s": time_s,
                "saw_behind_margin": False,
                "pass_start_time_s": None,
                "pass_time_s": None,
                "leader_recycled": False,
                "previous_progress_m": None,
                "minimum_clearance_m": float("inf"),
                "initial_longitudinal_m": None,
                "final_longitudinal_m": None,
                **dict(conditions or {}),
            }
            self._active[env_id] = state

        # The slot may be temporarily unavailable only for an incompatible caller.  Preserve
        # the existing Follow state rather than fabricating an ordering transition.
        if not leader_active:
            state["leader_recycled"] = True
            return

        progress = self._corridor_progress(crowd, env_id, leader_position, flow_direction)
        if self._recycled(state["previous_progress_m"], progress, crowd, env_id):
            state["leader_recycled"] = True
        state["previous_progress_m"] = progress
        if state["leader_recycled"]:
            return

        longitudinal = float((robot_position[0] - leader_position[0]) * flow_direction)
        if state["initial_longitudinal_m"] is None:
            state["initial_longitudinal_m"] = longitudinal
        state["final_longitudinal_m"] = longitudinal

        radii = crowd.radius[env_id]
        robot_radius = float(crowd.cfg.robot_radius)
        leader_radius = float(radii[SLOW_LEADER_SLOT].detach().cpu().item())
        clearance = float(np.linalg.norm(robot_position - leader_position) - robot_radius - leader_radius)
        state["minimum_clearance_m"] = min(float(state["minimum_clearance_m"]), clearance)

        margin = SLOW_LEADER_OVERTAKE_MARGIN_M
        if longitudinal <= -margin:
            state["saw_behind_margin"] = True
        if state["saw_behind_margin"] and state["pass_start_time_s"] is None and longitudinal > -margin:
            state["pass_start_time_s"] = time_s
        if state["saw_behind_margin"] and state["pass_time_s"] is None and longitudinal >= margin:
            state["pass_time_s"] = time_s

    def record_pre_step(self, env: Any, conditions_by_env: Mapping[int, Mapping[str, float]] | None = None) -> None:
        """Sample leader ordering once per evaluation control step before the policy action."""
        conditions_by_env = conditions_by_env or {}
        for env_id in range(len(self.env_profile_indices)):
            self._observe(env, env_id, conditions_by_env.get(env_id))
            self._times[env_id] += self.step_dt_s

    def finalize_terminal(self, env: Any, env_ids: Any) -> None:
        """Capture final pre-reset state and stage one potential outcome per terminating episode."""
        for env_id in _ids(env_ids):
            if not self._is_slow_leader_env(env_id):
                continue
            self._observe(env, env_id)
            state = self._active[env_id]
            if state is None:
                continue
            pass_time = state["pass_time_s"]
            outcome = "Overtake" if pass_time is not None else "Follow"
            pass_start = state["pass_start_time_s"]
            record = {
                **state,
                "outcome": outcome,
                "canonical_label": "overtake" if outcome == "Overtake" else "follow",
                "end_time_s": pass_time if pass_time is not None else self._times[env_id],
                "duration_s": (
                    float(pass_time - pass_start)
                    if pass_time is not None and pass_start is not None
                    else None
                ),
            }
            # Overtake replays cover the passing maneuver, not the whole episode.
            if pass_start is not None:
                record["start_time_s"] = pass_start
            self._pending_terminal[env_id] = record
            self._active[env_id] = None

    def pending_outcome(self, env_id: int) -> dict[str, Any] | None:
        """Return a terminal-staged slow-leader outcome before quota admission."""
        return self._pending_terminal.get(int(env_id))

    def resolve_terminal(
        self, completed_env_ids: Any, accepted_success_ids: Iterable[int], *, seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Admit only successful, profile-quota-counted slow-leader outcomes."""
        successful = {int(env_id) for env_id in accepted_success_ids}
        admitted: list[dict[str, Any]] = []
        for env_id in _ids(completed_env_ids):
            record = self._pending_terminal.pop(env_id, None)
            if record is not None and env_id in successful:
                profile = self.profiles[self.env_profile_indices[env_id]]
                admitted_record = {
                    **record,
                    "environment_id": env_id,
                    "pedestrian_count": profile.pedestrian_count,
                }
                if seed is not None:
                    admitted_record["seed"] = int(seed)
                self.outcomes.append(admitted_record)
                admitted.append(admitted_record)
            self._times[env_id] = 0.0
            self._active[env_id] = None
        return admitted

    def summary_rows(self) -> list[dict[str, Any]]:
        """Return Follow/Overtake totals and rates for each slow-leader crowd count."""
        counts = sorted({profile.pedestrian_count for profile in self.profiles if profile.scenario == SLOW_LEADER_SCENARIO})
        rows = []
        for count in counts:
            outcomes = [row for row in self.outcomes if row["pedestrian_count"] == count]
            follows = sum(row["outcome"] == "Follow" for row in outcomes)
            overtakes = sum(row["outcome"] == "Overtake" for row in outcomes)
            total = len(outcomes)
            rows.append({
                "scenario": SLOW_LEADER_SCENARIO,
                "pedestrian_count": count,
                "successful_episodes": total,
                "follow": follows,
                "overtake": overtakes,
                "follow_rate": follows / total if total else None,
                "overtake_rate": overtakes / total if total else None,
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


def save_slow_leader_outcome_artifacts(
    output_dir: str | Path,
    outcomes: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> Path:
    """Write success-only slot-zero Follow/Overtake outcomes and their aggregates."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outcome_fields = [
        "scenario", "pedestrian_count", "environment_id", "seed", "pedestrian_id", "outcome",
        "start_time_s", "end_time_s", "duration_s", "pass_time_s", "minimum_clearance_m",
        "initial_longitudinal_m", "final_longitudinal_m", "leader_recycled", "speed_mps",
        "start_ahead_m", "lateral_offset_m",
    ]
    with (output_path / "slow_leader_outcomes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=outcome_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outcomes)
    summary_fields = [
        "scenario", "pedestrian_count", "successful_episodes", "follow", "overtake", "follow_rate", "overtake_rate",
    ]
    with (output_path / "slow_leader_outcome_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_path / "slow_leader_outcomes.json").open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(
                {
                    "schema_version": 1,
                    "scenario": SLOW_LEADER_SCENARIO,
                    "pedestrian_slot": SLOW_LEADER_SLOT,
                    "overtake_margin_m": SLOW_LEADER_OVERTAKE_MARGIN_M,
                    "labels": list(SLOW_LEADER_OUTCOME_LABELS),
                    "outcomes": outcomes,
                    "summary": summary_rows,
                }
            ),
            file,
            indent=2,
            allow_nan=False,
        )
    _save_slow_leader_outcome_plot(output_path / "slow_leader_outcomes.png", summary_rows)
    return output_path


def print_slow_leader_outcomes(summary_rows: list[dict[str, Any]]) -> None:
    """Print the episode-level slow-leader outcome table when the profile is enabled."""
    if not summary_rows:
        return
    print("slow-leader outcomes  crowd  successful  follow  overtake  follow%  overtake%")
    print("-" * 76)
    for row in summary_rows:
        follow_rate = row["follow_rate"]
        overtake_rate = row["overtake_rate"]
        follow_percent = "n/a" if follow_rate is None else f"{100.0 * follow_rate:.1f}"
        overtake_percent = "n/a" if overtake_rate is None else f"{100.0 * overtake_rate:.1f}"
        print(
            f"{SLOW_LEADER_SCENARIO:<21} {row['pedestrian_count']:>5} "
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


def _save_slow_leader_outcome_plot(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    """Save a compact Follow/Overtake count chart by crowd count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    counts = [int(row["pedestrian_count"]) for row in summary_rows]
    follows = [int(row["follow"]) for row in summary_rows]
    overtakes = [int(row["overtake"]) for row in summary_rows]
    x = np.arange(len(counts))
    width = 0.38
    follow_bars = axis.bar(x - width / 2.0, follows, width, label="Follow", color="#60a5fa")
    overtake_bars = axis.bar(x + width / 2.0, overtakes, width, label="Overtake", color="#f59e0b")
    axis.bar_label(follow_bars, padding=3)
    axis.bar_label(overtake_bars, padding=3)
    axis.set_xticks(x, [str(count) for count in counts])
    axis.set_xlabel("Pedestrians")
    axis.set_ylabel("Successful episodes")
    axis.set_title("Slow-leader episode outcomes")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
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
