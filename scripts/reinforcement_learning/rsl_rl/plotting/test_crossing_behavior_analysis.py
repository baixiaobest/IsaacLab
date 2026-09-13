"""Tests for the simulator-independent crossing behavior analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossing_behavior_analysis import (  # noqa: E402
    aggregate_cases,
    load_captured_cases,
    pedestrian_in_robot_bev,
)


def test_pedestrian_in_robot_bev_places_forward_on_positive_y() -> None:
    x_relative, y_relative = pedestrian_in_robot_bev(
        robot_position_xy=np.asarray([[10.0, 20.0], [10.0, 20.0]]),
        robot_yaw=np.asarray([0.0, np.pi / 2.0]),
        pedestrian_position_xy=np.asarray([[12.0, 19.0], [11.0, 22.0]]),
    )

    np.testing.assert_allclose(x_relative, [1.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(y_relative, [2.0, 2.0], atol=1.0e-12)


def _write_replay(path: Path, *, yaw: float, pedestrian_positions: np.ndarray) -> None:
    frame_count = len(pedestrian_positions)
    all_pedestrians = np.zeros((frame_count, 2, 2), dtype=np.float32)
    all_pedestrians[:, 1] = pedestrian_positions
    active = np.ones((frame_count, 2), dtype=bool)
    active[3, 1] = False
    np.savez_compressed(
        path,
        time_s=np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        robot_position_xy=np.zeros((frame_count, 2), dtype=np.float32),
        robot_yaw=np.full(frame_count, yaw, dtype=np.float32),
        pedestrian_position_xy=all_pedestrians,
        pedestrian_active_mask=active,
    )


def test_aggregate_uses_manifest_labels_event_interval_activity_and_radius(tmp_path: Path) -> None:
    replay_dir = tmp_path / "episode_cases" / "interaction_events"
    cases_dir = replay_dir / "cases"
    cases_dir.mkdir(parents=True)
    # For yaw=0: world +X is forward and world -Y is right.
    _write_replay(
        cases_dir / "assert.npz",
        yaw=0.0,
        pedestrian_positions=np.asarray([[3.0, 0.0], [2.2, -1.2], [0.5, 0.0], [0.0, -2.0], [-3.0, 0.0]]),
    )
    # For yaw=pi/2: world +Y is forward and world +X is right.
    _write_replay(
        cases_dir / "yield.npz",
        yaw=np.pi / 2.0,
        pedestrian_positions=np.asarray([[0.0, 3.0], [1.2, 2.2], [0.0, 0.5], [2.0, 0.0], [0.0, -3.0]]),
    )
    manifest_path = replay_dir / "interaction_event_cases.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "assert-case",
                        "canonical_label": "assert",
                        "pedestrian_id": 1,
                        "start_time_s": 0.0,
                        "end_time_s": 2.0,
                        "step_dt_s": 1.0,
                        "replay_file": "cases/assert.npz",
                    },
                    {
                        "case_id": "yield-case",
                        "canonical_label": "yield",
                        "pedestrian_id": 1,
                        "start_time_s": 0.0,
                        "end_time_s": 2.0,
                        "step_dt_s": 1.0,
                        "replay_file": "cases/yield.npz",
                    },
                    {"case_id": "ignored", "canonical_label": "ambiguous"},
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_captured_cases(manifest_path)
    result = aggregate_cases(
        manifest_path,
        cases,
        r_min=1.0,
        r_max=4.0,
        resolution=1.0,
        min_samples=2,
    )

    assert [case["canonical_label"] for case in cases] == ["assert", "yield"]
    assert result.event_count_assert == 1
    assert result.event_count_yield == 1
    assert result.sample_count_before_radius_filter == 4
    assert result.sample_count_after_radius_filter == 2
    assert result.assert_count.sum() == 1
    assert result.yield_count.sum() == 1
    assert result.total_count.sum() == 2
    assert np.nanmax(result.p_assert_masked) == 0.5
    assert np.count_nonzero(np.isfinite(result.p_assert_masked)) == 1
