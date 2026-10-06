"""Audit plots show all four labeled policy scan frames."""

import h5py
import matplotlib
import numpy as np
from matplotlib.axes import Axes

from audit import ScanSample, _plot_scan_sample


matplotlib.use("Agg")


def test_audit_plot_uses_four_frame_static_and_pedestrian_colors(tmp_path, monkeypatch):
    path = tmp_path / "rollout.hdf5"
    ranges = np.full((1, 4, 128), 20.0, dtype=np.float32)
    reflected = np.zeros((1, 4, 128), dtype=bool)
    dynamic = np.zeros_like(reflected)
    for age in range(4):
        ranges[0, age, 65] = 2.0 + age
        ranges[0, age, 67] = 3.0 + age
        reflected[0, age, [65, 67]] = True
        dynamic[0, age, 67] = True
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data/episode_0")
        group.create_dataset("history_range_m", data=ranges)
        group.create_dataset("history_reflection_mask", data=reflected)
        group.create_dataset("history_dynamic_mask", data=dynamic)
        group.create_dataset("point_velocity_b", data=np.zeros((1, 128, 2), dtype=np.float32))
        group.create_dataset("capture_index", data=np.array([3]))

    calls = {}
    original_scatter = Axes.scatter

    def record_scatter(self, *args, **kwargs):
        label = kwargs.get("label")
        if label and (label.startswith("static") or label.startswith("pedestrian")):
            calls[label] = (kwargs["c"], len(args[0]))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    manifest = _plot_scan_sample(tmp_path, ScanSample("dynamic", str(path), "episode_0", 0), 10.0, 1.0)
    assert (tmp_path / "scan_samples" / manifest["plot"].split("/")[-1]).exists()
    assert manifest["valid_returns_by_frame"] == [2, 2, 2, 2]
    assert manifest["dynamic_returns_by_frame"] == [1, 1, 1, 1]
    assert [calls[name][0] for name in ("pedestrian current", "pedestrian current −1", "pedestrian current −2", "pedestrian current −3")] == [
        "#d7191c", "#f46d43", "#fdae61", "#ffe34d"
    ]
    assert [calls[name][0] for name in ("static current", "static current −1", "static current −2", "static current −3")] == [
        "#303030", "#686868", "#a6a6a6", "#e3e3e3"
    ]
