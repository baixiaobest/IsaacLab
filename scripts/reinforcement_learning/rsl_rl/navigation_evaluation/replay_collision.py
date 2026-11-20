# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation evaluation %s."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    BenchmarkProfile,
    GOAL_REGION_COLLISION_RADIUS_M,
    GOAL_REGION_TAG,
    INTERESTING_INTERACTION_TAG,
    ROOM_CORRIDOR_SCENARIOS,
    STATIC_OBSTACLE_SCENARIO,
    SUCCESS_INTERACTION_DISTANCE_M,
    _write_json_atomically,
)

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
        """Return the number of pedestrian or static-obstacle collision replays."""
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
        self.robot_radius_m = float(crowd.cfg.robot_radius)
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
            "crowd_flow_direction": torch.ones(num_envs, self.capacity, device=device),
            "corridor_origin_xy": torch.zeros(num_envs, self.capacity, 2, device=device),
            "corridor_length": torch.zeros(num_envs, self.capacity, device=device),
            "reward": torch.full((num_envs, self.capacity), float("nan"), device=device),
            "pedestrian_position_xy": torch.zeros(num_envs, self.capacity, max_pedestrians, 2, device=device),
            "pedestrian_velocity_xy_world": torch.zeros(num_envs, self.capacity, max_pedestrians, 2, device=device),
            "pedestrian_radius": torch.zeros(num_envs, self.capacity, max_pedestrians, device=device),
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
        if getattr(crowd, "flow_dir", None) is not None:
            self._buffers["crowd_flow_direction"][env_ids, indices] = crowd.flow_dir
        if getattr(crowd, "corridor_origin", None) is not None:
            self._buffers["corridor_origin_xy"][env_ids, indices] = crowd.corridor_origin[:, :2]
        if getattr(crowd, "corridor_length", None) is not None:
            self._buffers["corridor_length"][env_ids, indices] = crowd.corridor_length
        self._buffers["pedestrian_position_xy"][env_ids, indices] = crowd.get_world_positions()
        self._buffers["pedestrian_velocity_xy_world"][env_ids, indices] = crowd.get_velocities()
        self._buffers["pedestrian_radius"][env_ids, indices] = crowd.radius
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

    def record_reward(self, rewards: Any) -> None:
        """Attach post-step scalar rewards to the pre-step frames that produced them."""
        import torch

        if self._buffers is None or self._counts is None or self._write_indices is None:
            return
        values = torch.as_tensor(rewards, device=self._write_indices.device).reshape(-1)
        if values.shape[0] != len(self.env_profile_indices):
            raise ValueError("Rewards must contain one scalar per vector environment.")
        active = self._counts > 0
        env_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
        indices = (self._write_indices[env_ids] - 1) % self.capacity
        self._buffers["reward"][env_ids, indices] = values[env_ids]

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
            "crowd_flow_direction": (
                crowd.flow_dir[env_id : env_id + 1].detach().cpu().numpy()
                if getattr(crowd, "flow_dir", None) is not None else np.asarray([1.0], dtype=np.float32)
            ),
            "corridor_origin_xy": (
                crowd.corridor_origin[env_id : env_id + 1, :2].detach().cpu().numpy()
                if getattr(crowd, "corridor_origin", None) is not None else np.zeros((1, 2), dtype=np.float32)
            ),
            "corridor_length": (
                crowd.corridor_length[env_id : env_id + 1].detach().cpu().numpy()
                if getattr(crowd, "corridor_length", None) is not None else np.asarray([0.0], dtype=np.float32)
            ),
            "reward": np.asarray([np.nan], dtype=np.float32),
            "pedestrian_position_xy": crowd.get_world_positions()[env_id : env_id + 1].detach().cpu().numpy(),
            "pedestrian_velocity_xy_world": crowd.get_velocities()[env_id : env_id + 1].detach().cpu().numpy(),
            "pedestrian_radius": crowd.radius[env_id : env_id + 1].detach().cpu().numpy(),
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
        self, env: Any, reset_env_ids: Any, success_env_ids: Any, collision_env_ids: Iterable[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Export collision context and quota-limited complete successful episodes before reset."""
        import torch

        if self._buffers is None:
            return []
        env_ids = torch.as_tensor(reset_env_ids, device=self._env_ids.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return []
        if collision_env_ids is None:
            robot_positions = env.scene["robot"].data.root_pos_w[:, :2]
            collision_mask = env.crowd_manager.get_robot_collision(robot_positions)
            terminal_collision_env_ids = env_ids[collision_mask[env_ids]]
        else:
            requested_collision_ids = torch.as_tensor(
                list(collision_env_ids), device=self._env_ids.device, dtype=torch.long
            ).reshape(-1)
            reset_mask = torch.zeros(env.num_envs, device=self._env_ids.device, dtype=torch.bool)
            reset_mask[env_ids] = True
            terminal_collision_env_ids = requested_collision_ids[reset_mask[requested_collision_ids]]
        collision_ids = set(terminal_collision_env_ids.detach().cpu().tolist())
        assert self._minimum_agent_distances is not None
        episode_minimum_distances = torch.minimum(
            self._minimum_agent_distances, self._minimum_active_pedestrian_distances(env)
        )
        requested_success_ids = set(
            torch.as_tensor(success_env_ids, device=self._env_ids.device, dtype=torch.long).reshape(-1).cpu().tolist()
        )

        exported = []
        if self.record_collisions:
            for env_id in terminal_collision_env_ids.detach().cpu().tolist():
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
        collision_source = (
            "static_obstacle_contact" if profile.scenario == STATIC_OBSTACLE_SCENARIO or profile.scenario in ROOM_CORRIDOR_SCENARIOS else "pedestrian_capsule"
        )
        colliding_agent_ids = (
            self._collision_indices(env, env_id)
            if outcome == "collision" and collision_source == "pedestrian_capsule"
            else []
        )
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
            "obstacle_count": profile.obstacle_count,
            "terrain_column": profile.terrain_column,
            "terrain_level": profile.terrain_level,
            "terrain_family": profile.terrain_family,
            "nominal_corridor_width_m": profile.nominal_corridor_width_m,
            "environment_id": env_id,
            "coverage_target": getattr(self, "coverage_target", None),
            "outcome": outcome,
            "terminal_time_s": float(frames["time_s"][-1]),
            "collision_time_s": float(frames["time_s"][-1]) if outcome == "collision" else None,
            "collision_source": collision_source if outcome == "collision" else None,
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


