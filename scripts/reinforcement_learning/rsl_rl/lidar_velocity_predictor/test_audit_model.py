"""Focused checks for the model-versus-label visual audit."""

from types import SimpleNamespace

import h5py
import matplotlib
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes

from audit_model import choose_samples, load_model, plot_sample, split_indices, validate_dataset_contract
from src.model import TemporalLidarVelocityCNN


matplotlib.use("Agg")


@pytest.mark.parametrize("frames", [4, 8])
def test_load_checkpoint_and_jit_frame_metadata(tmp_path, frames):
    model = TemporalLidarVelocityCNN(frames).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"projection_contract": "temporal_lidar_v4", "num_frames": frames,
                "model_state_dict": model.state_dict()}, checkpoint)
    loaded, count, contract = load_model(checkpoint, torch.device("cpu"))
    assert (count, contract) == (frames, "temporal_lidar_v4")
    assert loaded(torch.zeros(1, 2, frames, 128)).shape == (1, 128, 2)
    jit = tmp_path / "model_jit.pt"
    torch.jit.save(torch.jit.script(model), str(jit),
                   _extra_files={"projection_contract.txt": b"temporal_lidar_v4",
                                 "num_frames.txt": str(frames).encode()})
    _, count, contract = load_model(jit, torch.device("cpu"))
    assert (count, contract) == (frames, "temporal_lidar_v4")


def test_plot_uses_model_input_positions_and_only_supervised_arrows(tmp_path, monkeypatch):
    lidar = np.zeros((2, 4, 128), dtype=np.float32)
    lidar[0] = 1.0
    lidar[0, 0, 64] = 0.1  # 2 m forward
    lidar[1, 0, 64] = 1.0
    lidar[0, 0, 65] = 0.2
    lidar[1, 0, 65] = 1.0
    target = np.zeros((128, 2), dtype=np.float32)
    target[64] = (0, 1)
    prediction = np.zeros_like(target)
    prediction[64] = (1, 0)
    prediction[65] = (9, 0)  # valid scan return but no velocity label
    reflection = np.zeros(128, dtype=bool)
    reflection[64] = True
    dynamic = reflection.copy()
    arrows = []
    original = Axes.annotate

    def record(self, *args, **kwargs):
        if args and args[0] == "" and "arrowprops" in kwargs:
            arrows.append((np.asarray(kwargs["xytext"]), np.asarray(kwargs["xy"]), kwargs["arrowprops"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", record)
    result = plot_sample(tmp_path / "plot.png", lidar, target, prediction, reflection, dynamic, 5.0, 1.0, "fixture")
    assert (tmp_path / "plot.png").is_file()
    assert result["dynamic_bins"] == 1
    assert len(arrows) == 2
    assert np.allclose(arrows[0][0], [2, 0])
    assert np.allclose(arrows[0][1], [2, 1])
    assert np.allclose(arrows[1][1], [3, 0])
    assert arrows[1][2]["linestyle"] == "dashed"


def test_sampling_uses_only_requested_split_and_separates_held_scans(tmp_path):
    path = tmp_path / "rollout.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data/episode_0")
        reflected = np.zeros((4, 128), dtype=bool)
        reflected[:, 64] = True
        dynamic = np.zeros_like(reflected)
        dynamic[2:, 64] = True
        group.create_dataset("reflection_mask", data=reflected)
        group.create_dataset("dynamic_mask", data=dynamic)
        group.create_dataset("first_after_capture", data=np.array([True, False, True, False]))
    dataset = type("FakeDataset", (), {"__len__": lambda self: 4})()
    dataset.offsets = [0, 4]
    dataset.entries = [SimpleNamespace(file_path=str(path), episode_name="episode_0", length=4)]
    splits = tmp_path / "splits.json"
    splits.write_text('{"test_indices": [1, 2]}', encoding="utf-8")
    indices, source = split_indices(dataset, tmp_path / "model.pt", splits)
    assert source == str(splits)
    selected, counts = choose_samples(dataset, indices, 2, 42)
    assert selected == {"static_held": [1], "dynamic_first": [2]}
    assert counts == {"static_held": 1, "dynamic_first": 1}


def test_cross_version_contract_requires_four_frames(tmp_path):
    path = tmp_path / "rollout.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("data").attrs["metadata"] = '{"projection_contract": "temporal_lidar_v4"}'
    dataset = SimpleNamespace(entries=[SimpleNamespace(file_path=str(path))])
    validate_dataset_contract(dataset, 4, "temporal_lidar_v3")
    with pytest.raises(ValueError, match="four frames"):
        validate_dataset_contract(dataset, 8, "temporal_lidar_v3")
