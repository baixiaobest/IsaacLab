# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation benchmark metrics."""

from __future__ import annotations

from dataclasses import asdict
from numbers import Number
from typing import Any, Iterable, Mapping


from .common import (
    BenchmarkProfile,
    ROOM_CORRIDOR_SCENARIOS,
    SCENARIO_ORDER,
    STATIC_OBSTACLE_SCENARIO,
    _sample_standard_deviation,
)

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
        static_collision_term: str = "base_contact",
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
        self.static_collision_ids_key = f"Episode_Termination/Envs/Ids/{static_collision_term}"
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
        static_collision_ids = _ids(log.get(self.static_collision_ids_key))
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
            is_static_collision = (
                (self.profiles[profile_index].scenario == STATIC_OBSTACLE_SCENARIO
                 or self.profiles[profile_index].scenario in ROOM_CORRIDOR_SCENARIOS)
                and env_id in static_collision_ids
            )
            if env_id in collision_ids or is_static_collision:
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

    def set_stage_limit(self, limit: int, *, record_seed_boundary: bool = True) -> None:
        """Cap per-profile acceptance at ``limit`` for the current evaluation stage.

        Record a boundary at each seed's first stage; coverage substages pass
        ``record_seed_boundary=False`` so :meth:`per_seed_counts` remains correct.
        """
        if record_seed_boundary:
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
                "terrain_family": next(
                    (profiles[index].terrain_family for index in profile_indices), "open"
                ),
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


