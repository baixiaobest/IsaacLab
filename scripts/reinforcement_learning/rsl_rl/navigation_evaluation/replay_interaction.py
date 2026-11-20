# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation evaluation %s."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    LEADER_OUTCOME_SCENARIOS,
    _write_json_atomically,
)
from .metrics import (
    _ids,
)
from .replay_collision import (
    CollisionReplayRecorder,
)

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
            # Designated-leader evaluation exports only robust slot-zero passes. Follow
            # is an outcome report, not an interaction replay candidate.
            if (
                raw_event.get("scenario") in LEADER_OUTCOME_SCENARIOS
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


