# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation benchmark common."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCENARIO_ORDER = (
    "static_obstacles",
    "room_to_corridor", "corridor_to_room", "corridor_to_corridor", "room_to_room",
    "crossing",
    "with_flow",
    "against_flow",
    "with_flow_slow_leader",
    "crossing_slow",
    "against_flow_slow",
)
SCENARIO_LABELS = {
    "static_obstacles": "Static obstacles",
    "room_to_corridor": "Room to corridor",
    "corridor_to_room": "Corridor to room",
    "corridor_to_corridor": "Along corridor",
    "room_to_room": "Room to room",
    "crossing": "Crossing",
    "with_flow": "With flow",
    "against_flow": "Against flow",
    "with_flow_slow_leader": "With flow — slow leader",
    "crossing_slow": "Crossing — slow crowd",
    "against_flow_slow": "Against flow — slow crowd",
}

STATIC_OBSTACLE_SCENARIO = "static_obstacles"
ROOM_CORRIDOR_SCENARIOS = frozenset(("room_to_corridor", "corridor_to_room", "corridor_to_corridor", "room_to_room"))
EVALUATION_LEVEL_COUNTS = (2, 4, 6, 8, 10, 12, 14, 16)

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
    "crossing_slow": ("yield", "assert", "ambiguous", "non_risky_close", "unclassified"),
    "against_flow_slow": ("sidestep", "straight_pass", "front_crossing"),
}
"""Pairwise interaction labels. With-flow variants use slot-zero episode outcomes instead."""

INTERACTION_SCENARIO_ORDER = tuple(scenario for scenario in SCENARIO_ORDER if scenario in INTERACTION_LABELS)

LEADER_OUTCOME_SCENARIOS = ("with_flow", "with_flow_slow_leader")
LEADER_SLOT = 0
LEADER_OVERTAKE_MARGIN_M = 0.5
LEADER_OUTCOME_LABELS = ("Follow", "Overtake")

# Compatibility aliases for callers and historical artifact documentation.
SLOW_LEADER_SCENARIO = "with_flow_slow_leader"
SLOW_LEADER_SLOT = LEADER_SLOT
SLOW_LEADER_OVERTAKE_MARGIN_M = LEADER_OVERTAKE_MARGIN_M
SLOW_LEADER_OUTCOME_LABELS = LEADER_OUTCOME_LABELS


@dataclass(frozen=True)
class BenchmarkProfile:
    """One benchmark cell assigned to one or more vector environments."""

    scenario: str
    pedestrian_count: int
    obstacle_count: int | None = None
    terrain_column: int | None = None
    terrain_level: int | None = None
    terrain_family: str = "open"
    nominal_corridor_width_m: float | None = None


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
    collision_env_ids: Iterable[int] | None = None,
) -> set[int]:
    """Return resetting collision environments within ``radius_m`` of their world-frame goal.

    When ``collision_env_ids`` is omitted, preserve the historical pedestrian-capsule
    query.  Supplying the terminal collision IDs lets the static-obstacle benchmark use
    its ``base_contact`` termination without pretending static terrain is a pedestrian.
    """
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    import torch

    env_ids = torch.as_tensor(reset_env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if env_ids.numel() == 0:
        return set()
    robot_positions = env.scene["robot"].data.root_pos_w[:, :2]
    goal_positions = env.command_manager.get_term(command_name).pos_command_w[:, :2]
    within_goal_region = torch.linalg.vector_norm(robot_positions - goal_positions, dim=1) <= radius_m
    if collision_env_ids is None:
        collision_mask = env.crowd_manager.get_robot_collision(robot_positions)
        return set(env_ids[collision_mask[env_ids] & within_goal_region[env_ids]].detach().cpu().tolist())
    collisions = torch.as_tensor(list(collision_env_ids), device=env.device, dtype=torch.long).reshape(-1)
    if collisions.numel() == 0:
        return set()
    resetting = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    resetting[env_ids] = True
    collision_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    collision_mask[collisions] = True
    return set(torch.nonzero(collision_mask & resetting & within_goal_region, as_tuple=False).reshape(-1).cpu().tolist())


def terminal_collision_ids(
    env: Any,
    reset_env_ids: Any,
    profiles: list[BenchmarkProfile],
    env_profile_indices: Iterable[int],
    pedestrian_collision_term: str = "pedestrian_collision",
    static_collision_term: str = "base_contact",
) -> set[int]:
    """Return the benchmark collision IDs for terminal environments.

    Dynamic profiles use the pedestrian-capsule termination.  Static profiles have no
    active pedestrians, so their corresponding physical collision signal is the
    task's existing ``base_contact`` termination.  Keeping this selection here makes
    metrics, goal-region classification, and collision replays share one definition.
    """
    import torch

    indices = [int(index) for index in env_profile_indices]
    if len(indices) != env.num_envs:
        raise ValueError("Profile assignment does not match env.num_envs.")
    env_ids = torch.as_tensor(reset_env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if env_ids.numel() == 0:
        return set()
    pedestrian_mask = env.termination_manager.get_term(pedestrian_collision_term)
    static_contact_mask = env.termination_manager.get_term(static_collision_term)
    result = set()
    for env_id in env_ids.detach().cpu().tolist():
        profile = profiles[indices[env_id]]
        is_static = profile.scenario == STATIC_OBSTACLE_SCENARIO or profile.scenario in ROOM_CORRIDOR_SCENARIOS
        if (is_static and bool(static_contact_mask[env_id])) or (
            not is_static and bool(pedestrian_mask[env_id])
        ):
            result.add(int(env_id))
    return result




def _sample_standard_deviation(values: Iterable[float]) -> float:
    """Return the sample standard deviation, or zero for fewer than two samples."""
    samples = list(values)
    if len(samples) < 2:
        return 0.0
    mean = sum(samples) / len(samples)
    return math.sqrt(sum((value - mean) ** 2 for value in samples) / (len(samples) - 1))


