# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation evaluation %s."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from .common import (
    BenchmarkProfile,
    INTERACTION_AGAINST_FLOW_CONE_HALF_ANGLE_RAD,
    INTERACTION_AGAINST_FLOW_CONE_LATERAL_BUFFER_M,
    INTERACTION_AGAINST_FLOW_CONE_PERSISTENCE_S,
    INTERACTION_AGAINST_FLOW_ENCOUNTER_RANGE_M,
    INTERACTION_AGAINST_FLOW_REAR_PASS_MARGIN_M,
    INTERACTION_ASSERT_SPEED_RATIO,
    INTERACTION_ENTER_CLEARANCE_M,
    INTERACTION_EXIT_CLEARANCE_M,
    INTERACTION_FRONT_CROSS_CLEARANCE_MARGIN_M,
    INTERACTION_LABELS,
    INTERACTION_MIN_BASELINE_SPEED_MPS,
    INTERACTION_PRE_SPEED_WINDOW_S,
    INTERACTION_RISK_CLEARANCE_M,
    INTERACTION_RISK_HORIZON_S,
    INTERACTION_SCENARIO_ORDER,
    INTERACTION_YIELD_SPEED_RATIO,
)
from .metrics import (
    _ids,
)
from .interaction_classification import (
    classify_against_flow_interaction,
    classify_crossing_interaction,
    classify_speed_interaction,
    front_crossing_longitudinal_m,
    front_lateral_side,
    interaction_speed_diagnostics,
    rear_crossing_longitudinal_m,
)

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
            # Static terrain and with-flow variants have no pairwise interaction protocol.
            if scenario not in INTERACTION_LABELS:
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


