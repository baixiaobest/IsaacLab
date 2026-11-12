"""A distribution audit remains valid when every reflection is static."""

import json
import sys

import h5py
import matplotlib
import numpy as np

from audit import main


matplotlib.use("Agg")


def test_audit_static_only_dataset(tmp_path, monkeypatch):
    path = tmp_path / "real.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["metadata"] = json.dumps({"schema_version": 3, "velocity_frame": "evaluation_yaw_xy"})
        group = data.create_group("episode_0")
        reflection = np.zeros((1, 128), dtype=bool)
        reflection[0, 64] = True
        group.create_dataset("reflection_mask", data=reflection)
        group.create_dataset("dynamic_mask", data=np.zeros_like(reflection))
        group.create_dataset("range_m", data=np.full((1, 128), 2.0, dtype=np.float32))
        group.create_dataset("point_velocity_b", data=np.zeros((1, 128, 2), dtype=np.float32))
        group.create_dataset("ray_coverage", data=np.array([0.5], dtype=np.float32))
        group.create_dataset("reflection_coverage", data=np.array([1 / 128], dtype=np.float32))
        group.create_dataset("first_after_capture", data=np.array([True]))
    output = tmp_path / "audit"
    output.mkdir()
    (output / "dynamic_velocity_distribution.png").write_bytes(b"stale")
    monkeypatch.setattr(sys, "argv", ["audit.py", "--dataset_path", str(path), "--output_dir", str(output),
                                  "--num_scan_samples", "0"])
    main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"]["dynamic"] == 0
    assert summary["dynamic_distance_quantiles_m"] == []
    assert summary["effective_distance_weight"]["dynamic"] == 0.0
    assert (output / "distance_distribution.png").exists()
    assert not (output / "dynamic_velocity_distribution.png").exists()
