"""Schema and episode split checks without Isaac Sim."""

import json

import h5py
import numpy as np
import pytest

from src.dataset import PointVelocityDataset, validate_collection_metadata


def _write_episode(group, capture_index):
    group.create_dataset("lidar_noisy", data=np.zeros((2, 2, 4, 128), dtype=np.float32))
    group.create_dataset("lidar_clean", data=np.zeros((2, 2, 4, 128), dtype=np.float32))
    group.create_dataset("point_velocity_b", data=np.zeros((2, 128, 2), dtype=np.float32))
    for name in ("reflection_mask", "dynamic_mask"):
        group.create_dataset(name, data=np.zeros((2, 128), dtype=bool))
    group.create_dataset("range_m", data=np.zeros((2, 128), dtype=np.float32))
    group.create_dataset("capture_index", data=np.full(2, capture_index, dtype=np.int64))
    for name in ("capture_time_s", "evaluation_time_s", "evaluation_yaw", "ray_coverage", "reflection_coverage"):
        group.create_dataset(name, data=np.zeros(2, dtype=np.float32))
    group.create_dataset("evaluation_xy", data=np.zeros((2, 2), dtype=np.float32))
    group.create_dataset("first_after_capture", data=np.array([True, False]))


def test_schema_v3_keeps_held_samples_in_same_episode_partition(tmp_path):
    path = tmp_path / "samples.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["metadata"] = json.dumps({"schema_version": 3, "velocity_frame": "evaluation_yaw_xy"})
        for episode in range(10):
            _write_episode(data.create_group(f"episode_{episode}"), episode)
    dataset = PointVelocityDataset(str(path))
    train, validation, test = dataset.split()
    partitions = [set(subset.indices) for subset in (train, validation, test)]
    for episode in range(10):
        assert sum({2 * episode, 2 * episode + 1} <= part for part in partitions) == 1
    assert not dataset[0]["first_after_capture"].logical_not()
    assert dataset[1]["first_after_capture"].logical_not()


def test_schema_v2_is_rejected(tmp_path):
    path = tmp_path / "old.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["metadata"] = json.dumps({"schema_version": 2, "velocity_frame": "body_xy"})
    with pytest.raises(RuntimeError, match="schema-v3"):
        PointVelocityDataset(str(path))


def test_rollout_rejects_different_yaw_drift_when_appending(tmp_path):
    metadata = {"schema_version": 3, "velocity_frame": "evaluation_yaw_xy", "yaw_drift_std_rad_per_scan": 0.0}
    path = tmp_path / "samples.hdf5"
    validate_collection_metadata(metadata, metadata, path)
    with pytest.raises(RuntimeError, match="different LiDAR yaw drift"):
        validate_collection_metadata(metadata, {**metadata, "yaw_drift_std_rad_per_scan": np.deg2rad(0.5)}, path)
    with pytest.raises(RuntimeError, match="historical scan audit format"):
        validate_collection_metadata(metadata, {**metadata, "audit_history_format": 1}, path)
