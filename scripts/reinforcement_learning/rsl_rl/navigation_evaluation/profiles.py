# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation benchmark profiles."""

from __future__ import annotations

from typing import Iterable


from .common import (
    BenchmarkProfile,
    EVALUATION_LEVEL_COUNTS,
    ROOM_CORRIDOR_SCENARIOS,
    STATIC_OBSTACLE_SCENARIO,
)

def dynamic_crowd_profiles(
    counts: Iterable[int] = EVALUATION_LEVEL_COUNTS, *, include_static: bool = True,
    include_slow_leader: bool = True, include_slow_crowd: bool = True,
) -> list[BenchmarkProfile]:
    """Return the fixed static-plus-dynamic benchmark grid.

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
        BenchmarkProfile(STATIC_OBSTACLE_SCENARIO, 0, obstacle_count=count, terrain_column=0, terrain_level=row)
        for row, count in enumerate(counts)
    ] if include_static else []
    column = 1 if include_static else 0
    profiles.extend(
        BenchmarkProfile(scenario, count, terrain_column=column + scenario_index, terrain_level=row)
        for scenario_index, scenario in enumerate(ordinary_scenarios)
        for row, count in enumerate(counts)
    )
    if include_slow_leader:
        profiles.extend(
            BenchmarkProfile("with_flow_slow_leader", count, terrain_column=column + 3, terrain_level=row)
            for row, count in enumerate(counts)
        )
    if include_slow_crowd:
        profiles.extend(
            BenchmarkProfile(scenario, count, terrain_column=column + 4 + scenario_index, terrain_level=row)
            for scenario_index, scenario in enumerate(("crossing_slow", "against_flow_slow"))
            for row, count in enumerate(counts)
        )
    return profiles


def room_corridor_profiles() -> list[BenchmarkProfile]:
    """Eight fixed room-route profiles, two widths for each route."""
    routes = ("room_to_corridor", "corridor_to_room", "corridor_to_corridor", "room_to_room")
    return [BenchmarkProfile(routes[row % 4], 0, terrain_column=7, terrain_level=row,
                             terrain_family="room_corridor", nominal_corridor_width_m=1.8 if row < 4 else 1.2)
            for row in range(8)]


def select_scenario_profiles(
    profiles: list[BenchmarkProfile], enabled_scenarios: dict[str, bool]
) -> list[BenchmarkProfile]:
    """Select complete scenario columns, treating all room routes as one group."""
    selected = []
    for profile in profiles:
        group = "room_corridor" if profile.scenario in ROOM_CORRIDOR_SCENARIOS else profile.scenario
        if group not in enabled_scenarios:
            raise ValueError(f"Missing evaluation scenario setting: {group}")
        if enabled_scenarios[group]:
            selected.append(profile)
    if not selected:
        raise ValueError("At least one benchmark scenario must remain enabled.")
    return selected


def indoor_dynamic_profiles() -> list[BenchmarkProfile]:
    """Return the 3 core indoor scenarios x 4 lower-density benchmark cells."""
    counts = (2, 4, 6, 8)
    scenarios = ("crossing", "with_flow", "against_flow")
    return [
        BenchmarkProfile(scenario, count, terrain_column=column, terrain_level=row, terrain_family="indoor")
        for column, scenario in enumerate(scenarios)
        for row, count in enumerate(counts)
    ]


def fixed_grid_profile_indices(
    profiles: list[BenchmarkProfile], terrain_levels: Iterable[int], terrain_columns: Iterable[int],
    *, num_rows: int = 8, num_cols: int = 8,
) -> list[int]:
    """Return the selected profile index for each terrain cell."""
    levels = [int(value) for value in terrain_levels]
    columns = [int(value) for value in terrain_columns]
    if len(levels) != len(columns):
        raise ValueError("Fixed evaluation requires matched terrain coordinates.")
    profile_by_cell = {(profile.terrain_column, profile.terrain_level): index for index, profile in enumerate(profiles)}
    if len(profile_by_cell) != len(profiles):
        raise ValueError("Fixed evaluation profiles must have distinct terrain coordinates.")
    indices = []
    for level, column in zip(levels, columns):
        if not 0 <= level < num_rows or not 0 <= column < num_cols:
            raise ValueError("Fixed evaluation terrain coordinates do not fit the configured grid.")
        if (column, level) not in profile_by_cell:
            raise ValueError("Fixed evaluation terrain coordinates have no selected profile.")
        indices.append(profile_by_cell[(column, level)])
    return indices


