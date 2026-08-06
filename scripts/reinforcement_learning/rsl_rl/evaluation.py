"""Reusable helpers for vectorized policy-evaluation benchmarks.

The collector intentionally consumes Isaac Lab's reset-log payload instead of private
environment buffers. This makes it usable by any ManagerBased task that reports per-episode
termination IDs and metrics through ``extras["log"]``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Mapping


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
        # Retain episode-level values to report uncertainty across evaluation episodes.
        self._success_outcomes: list[list[float]] = [[] for _ in profiles]
        self._collision_outcomes: list[list[float]] = [[] for _ in profiles]
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
                success, collision = 0.0, 1.0
            elif env_id in success_ids:
                self._successes[profile_index] += 1
                success, collision = 1.0, 0.0
            else:
                success, collision = 0.0, 0.0
            self._success_outcomes[profile_index].append(success)
            self._collision_outcomes[profile_index].append(collision)
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
                    "success_rate_std": _sample_standard_deviation(self._success_outcomes[index]),
                    "collision_rate_std": _sample_standard_deviation(self._collision_outcomes[index]),
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
            success_outcomes = [outcome for index in profile_indices for outcome in self._success_outcomes[index]]
            collision_outcomes = [outcome for index in profile_indices for outcome in self._collision_outcomes[index]]
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
                    "success_rate_std": _sample_standard_deviation(success_outcomes),
                    "collision_rate_std": _sample_standard_deviation(collision_outcomes),
                    "std_xy_speed_mps": _sample_standard_deviation(velocity_values),
                }
            )
        return aggregates


def print_results(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    """Print a compact result table without introducing a tabular dependency."""
    header = (
        "scenario        crowd  episodes  success  collision  success% +/- std  "
        "collision% +/- std  xy speed (m/s) +/- std"
    )
    print(header)
    print("-" * len(header))
    for row in [*rows, *aggregate_rows]:
        print(
            f"{row['scenario']:<15} {str(row['pedestrian_count']):>5} {row['episodes']:>9} "
            f"{row['successes']:>8} {row['collisions']:>10} {100 * row['success_rate']:>8.1f} "
            f"+/- {100 * row['success_rate_std']:<5.1f} {100 * row['collision_rate']:>8.1f} "
            f"+/- {100 * row['collision_rate_std']:<5.1f} {row['mean_xy_speed_mps']:>8.3f} "
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
        "success_rate", "success_rate_std", "collision_rate", "collision_rate_std",
        "mean_xy_speed_mps", "std_xy_speed_mps",
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
        ("success_rate", "success_rate_std", "Success rate (%)", 100.0, (0.0, 100.0)),
        ("collision_rate", "collision_rate_std", "Collision rate (%)", 100.0, (0.0, 100.0)),
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
            standard_deviations = [row[std_metric] * scale for row in scenario_rows]
            axis.plot(crowd_counts, values, marker="o", linewidth=2)
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
    figure.suptitle("Dynamic crowd evaluation (shaded: ±1 sample SD)", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180)
    plt.close(figure)
