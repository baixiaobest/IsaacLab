"""Episode-budget and result helpers for sparse-LiDAR coverage evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from evaluation import BenchmarkProfile, _aggregate_rows_from_counts, _result_row


COVERAGE_TARGETS = (0.4, 0.6, 0.8, 1.0)


def coverage_quotas(episodes_per_profile: int, seed_count: int) -> list[list[int]]:
    """Allocate an exact, nearly equal profile quota to every seed and coverage."""
    if episodes_per_profile < len(COVERAGE_TARGETS):
        raise ValueError("Sparse-LiDAR coverage sweep requires at least four episodes per profile.")
    if seed_count < 1:
        raise ValueError("Seed count must be positive.")
    cells = [(seed, coverage) for seed in range(seed_count) for coverage in range(len(COVERAGE_TARGETS))]
    quotas = [[0] * len(COVERAGE_TARGETS) for _ in range(seed_count)]
    for index in range(episodes_per_profile):
        seed, coverage = cells[index % len(cells)]
        quotas[seed][coverage] += 1
    return quotas


def subtract_counts(
    after: list[Mapping[str, Any]], before: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Extract one stage's per-profile counts from cumulative snapshots."""
    result = []
    for current, previous in zip(after, before, strict=True):
        row = {key: current[key] - previous[key] for key in current if key != "velocity_values"}
        row["velocity_values"] = current["velocity_values"][int(previous["episodes"]):]
        result.append(row)
    return result


def coverage_rows(
    profiles: list[BenchmarkProfile], stage_counts: list[tuple[float, list[dict[str, Any]]]]
) -> dict[str, list[dict[str, Any]]]:
    """Return profile and scenario tables keyed by the assigned LiDAR coverage."""
    by_coverage: dict[float, list[list[dict[str, Any]]]] = {}
    for coverage, counts in stage_counts:
        by_coverage.setdefault(coverage, []).append(counts)
    profile_rows = []
    scenario_rows = []
    for coverage in COVERAGE_TARGETS:
        snapshots = by_coverage.get(coverage, [])
        if not snapshots:
            continue
        combined = []
        for profile_index in range(len(profiles)):
            rows = [stage[profile_index] for stage in snapshots]
            combined.append({
                **{key: sum(row[key] for row in rows) for key in rows[0] if key != "velocity_values"},
                "velocity_values": [value for row in rows for value in row["velocity_values"]],
            })
        profile_rows.extend({**_result_row(profile, counts), "coverage_target": coverage}
                            for profile, counts in zip(profiles, combined, strict=True))
        scenario_rows.extend({**row, "coverage_target": coverage}
                             for row in _aggregate_rows_from_counts(profiles, combined))
    return {"per_profile": profile_rows, "per_scenario": scenario_rows}
