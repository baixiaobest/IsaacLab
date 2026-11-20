# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation evaluation %s."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .common import (
    BenchmarkProfile,
    LEADER_OUTCOME_SCENARIOS,
    LEADER_OVERTAKE_MARGIN_M,
    LEADER_SLOT,
)
from .metrics import (
    _ids,
)

class LeaderOutcomeCollector:
    """Classify accepted successful designated-leader episodes as Follow or Overtake.

    The fixed leader is pedestrian slot zero. Unlike the ordinary interaction
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

    def _is_leader_env(self, env_id: int) -> bool:
        return self.profiles[self.env_profile_indices[env_id]].scenario in LEADER_OUTCOME_SCENARIOS

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
        if not self._is_leader_env(env_id):
            return
        crowd = env.crowd_manager
        robot_position = env.scene["robot"].data.root_pos_w[env_id, :2].detach().cpu().numpy()
        leader_position = crowd.get_world_positions()[env_id, LEADER_SLOT].detach().cpu().numpy()
        leader_active = bool(crowd.get_active_mask()[env_id, LEADER_SLOT].detach().cpu().item())
        flow_direction = self._flow_direction(crowd, env_id)
        time_s = self._times[env_id]

        state = self._active[env_id]
        if state is None:
            state = {
                "scenario": self.profiles[self.env_profile_indices[env_id]].scenario,
                "pedestrian_id": LEADER_SLOT,
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
        leader_radius = float(radii[LEADER_SLOT].detach().cpu().item())
        clearance = float(np.linalg.norm(robot_position - leader_position) - robot_radius - leader_radius)
        state["minimum_clearance_m"] = min(float(state["minimum_clearance_m"]), clearance)

        margin = LEADER_OVERTAKE_MARGIN_M
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
            if not self._is_leader_env(env_id):
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
        """Return a terminal-staged leader outcome before quota admission."""
        return self._pending_terminal.get(int(env_id))

    def resolve_terminal(
        self, completed_env_ids: Any, accepted_success_ids: Iterable[int], *, seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Admit only successful, profile-quota-counted leader outcomes."""
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
        """Return Follow/Overtake totals and rates for each leader scenario and crowd count."""
        rows = []
        for scenario in LEADER_OUTCOME_SCENARIOS:
            counts = sorted({profile.pedestrian_count for profile in self.profiles if profile.scenario == scenario})
            for count in counts:
                outcomes = [
                    row for row in self.outcomes
                    if row["scenario"] == scenario and row["pedestrian_count"] == count
                ]
                follows = sum(row["outcome"] == "Follow" for row in outcomes)
                overtakes = sum(row["outcome"] == "Overtake" for row in outcomes)
                total = len(outcomes)
                rows.append({
                    "scenario": scenario,
                    "pedestrian_count": count,
                    "successful_episodes": total,
                    "follow": follows,
                    "overtake": overtakes,
                    "follow_rate": follows / total if total else None,
                    "overtake_rate": overtakes / total if total else None,
                })
        return rows


# Existing callers retain the old import while the tracker supports both variants.
SlowLeaderOutcomeCollector = LeaderOutcomeCollector


