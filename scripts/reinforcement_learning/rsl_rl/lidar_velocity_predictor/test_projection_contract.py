"""Portable projection and fixture checks; runs without Isaac Sim."""

import json
import importlib.util
from pathlib import Path

import numpy as np

from src.projection import project_history


def _isaac_bins():
    root = Path(__file__).resolve().parents[4]
    source = root / "source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/lidar_geometry.py"
    spec = importlib.util.spec_from_file_location("lidar_geometry_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward_lidar_reflection_bins


def test_projection_fixture_and_held_pose_shift() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/projection_fixture_v3.json").read_text())
    hits = np.asarray(fixture["world_hit_xy_newest_first"])
    states = np.asarray(fixture["ray_state_newest_first"])
    xy = np.asarray(fixture["evaluation_xy"])
    tensor, winner = project_history(hits, states, xy, fixture["evaluation_yaw"])
    np.testing.assert_array_equal(tensor, fixture["expected_input_2x4x128"])
    np.testing.assert_array_equal(winner, fixture["expected_newest_winner_ray"])
    np.testing.assert_array_equal(winner >= 0, fixture["expected_newest_reflection_mask"])
    assert winner[64] == 0
    assert np.isclose(tensor[0, 0, 64], 0.05)

    # A later evaluation pose changes bin identity without changing the capture.
    shifted, shifted_winner = project_history(hits, states, xy + np.array([0.0, 0.5]), 0.0)
    assert shifted_winner[64] == -1
    assert np.any(shifted_winner == 0)
    assert np.any(shifted[1, 0] == 1.0)


def test_missing_and_no_return_rays_are_not_reflections() -> None:
    hits = np.zeros((4, 3, 2), dtype=np.float64)
    hits[:, 0] = [2, 0]
    hits[:, 1] = [1, 1]
    hits[:, 2] = [3, 0]
    states = np.tile(np.array([2, 1, 0], dtype=np.uint8), (4, 1))
    tensor, winner = project_history(hits, states, np.zeros(2), 0.0)
    assert winner[64] == 0
    assert (winner >= 0).sum() == 1
    assert tensor[1, 0].sum() == 2  # No-return is observed, but has no obstacle-velocity label.
    assert tensor[0, 0, 96] == 1.0


def test_portable_winners_match_isaac_reprojection() -> None:
    import torch

    angles = np.linspace(-np.pi / 2, np.pi / 2, 256)
    radius = 3.0 + 0.01 * np.arange(256)
    points = np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=-1)
    hits = np.tile(points[None], (4, 1, 1))
    states = np.tile(np.where(np.arange(256) % 3 == 0, 0, 2).astype(np.uint8)[None], (4, 1))
    bins = _isaac_bins()
    for xy, yaw in (([0.0, 0.0], 0.0), ([0.3, -0.2], 0.2)):
        _, winner = project_history(hits, states, np.asarray(xy), yaw)
        expected = bins({
            "hit_xy": torch.tensor(hits[:1], dtype=torch.float32),
            "ray_state": torch.tensor(states[:1]),
            "ego_xy": torch.tensor([xy], dtype=torch.float32),
            "ego_yaw": torch.tensor([yaw], dtype=torch.float32),
        })
        np.testing.assert_array_equal(winner >= 0, expected["reflection_mask"][0].numpy())
        np.testing.assert_array_equal(winner[winner >= 0], expected["winner_ray"][0].numpy()[winner >= 0])


def test_export_embeds_projection_contract(tmp_path) -> None:
    import torch
    from src.model import TemporalLidarVelocityCNN
    from train import _save_torchscript

    path = tmp_path / "predictor.pt"
    _save_torchscript(TemporalLidarVelocityCNN(8), path)
    extra = {"projection_contract.txt": b"", "num_frames.txt": b""}
    model = torch.jit.load(str(path), _extra_files=extra)
    assert extra["projection_contract.txt"] == b"temporal_lidar_v4"
    assert extra["num_frames.txt"] == b"8"
    assert model(torch.zeros(1, 2, 8, 128)).shape == (1, 128, 2)


def test_projection_accepts_eight_newest_first_frames() -> None:
    hits = np.zeros((8, 1, 2), dtype=np.float64)
    hits[:, 0, 0] = np.arange(1, 9)
    states = np.full((8, 1), 2, dtype=np.uint8)
    tensor, winner = project_history(hits, states, np.zeros(2), 0.0)
    assert tensor.shape == (2, 8, 128)
    np.testing.assert_allclose(tensor[0, :, 64], np.arange(1, 9) / 20)
    assert winner[64] == 0
