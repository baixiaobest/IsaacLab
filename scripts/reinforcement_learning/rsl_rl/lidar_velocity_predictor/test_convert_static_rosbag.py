"""Schema and projection checks for static ROS bag conversion, without ROS."""

import json
from types import SimpleNamespace

import h5py
import numpy as np

from convert_static_rosbag import SNAPSHOT_TOPIC, WORLD_TOPIC, bag_paths, convert, output_paths


def _stamp(seconds):
    return SimpleNamespace(sec=seconds, nanosec=0)


def _scan(sequence, session="session"):
    points = np.zeros((256, 2), dtype=np.float64)
    points[128] = (2.0, 0.0)
    states = np.zeros(256, dtype=np.uint8)
    states[128] = 2
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(sequence + 1)), sequence=sequence, session_id=session,
        capture_points_xy_m=points.ravel(), ray_states=states,
    )


def _snapshot(scan, evaluation_time):
    return SimpleNamespace(
        valid=True, session_id=scan.session_id, scan=scan,
        header=SimpleNamespace(stamp=_stamp(evaluation_time)),
        robot_xy_m=(0.0, 0.0), robot_yaw=0.0,
    )


def test_conversion_writes_static_schema_and_uses_embedded_capture(tmp_path):
    scans = [_scan(index) for index in range(10)]
    # The world topic is delivered in reverse order; the snapshot carries its
    # authoritative newest capture even if that capture's world message is absent.
    messages = [(WORLD_TOPIC, scan) for scan in reversed(scans[:-1])]
    messages += [(SNAPSHOT_TOPIC, _snapshot(scans[7], 9)),
                 (SNAPSHOT_TOPIC, _snapshot(scans[8], 10)),
                 (SNAPSHOT_TOPIC, _snapshot(scans[9], 11)),
                 (SNAPSHOT_TOPIC, _snapshot(scans[9], 12))]
    output = tmp_path / "static.hdf5"
    result = convert(messages, output, frames=8)
    assert result["samples"] == 4
    assert result["skipped_incomplete_history"] == 0
    assert not (tmp_path / "static.audit_splits.json").exists()
    with h5py.File(output) as handle:
        assert json.loads(handle["data"].attrs["metadata"])["data_source"] == "real_sensor"
        group = handle["data"]["episode_0000000"]
        assert group["lidar_measured"].shape == (4, 2, 8, 128)
        assert "lidar_clean" not in group and "lidar_noisy" not in group
        assert group["reflection_mask"][0].sum() == 1
        assert group["dynamic_mask"][:].sum() == 0
        assert np.abs(group["point_velocity_b"][:]).sum() == 0
        assert group["first_after_capture"][:].tolist() == [True, True, True, False]
        assert group["history_reflection_mask"].shape == (4, 8, 128)
        assert group["history_reflection_mask"][0, :, 64].all()


def test_missing_capture_is_skipped_without_padding(tmp_path):
    scans = [_scan(index) for index in range(8) if index != 3]
    output = tmp_path / "gap.hdf5"
    messages = [(WORLD_TOPIC, scan) for scan in scans]
    messages.append((SNAPSHOT_TOPIC, _snapshot(_scan(7), 9)))
    try:
        convert(messages, output, frames=8)
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("Expected incomplete eight-frame history to be rejected")
    assert not output.exists()


def test_parent_directory_finds_bag_runs_and_names_outputs(tmp_path):
    for name in ("run_b", "run_a"):
        bag = tmp_path / name
        bag.mkdir()
        (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}")
    (tmp_path / "converted").mkdir()
    bags = bag_paths(tmp_path)
    assert [path.name for path in bags] == ["run_a", "run_b"]
    assert output_paths(bags, tmp_path / "converted") == [
        tmp_path / "converted" / "run_a.hdf5", tmp_path / "converted" / "run_b.hdf5"
    ]
    assert bag_paths(bags[0]) == [bags[0]]
