"""HDF5 dataset and split/sampling helpers for LiDAR velocity prediction."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset


@dataclass(frozen=True)
class EpisodeEntry:
    file_path: str
    episode_name: str
    length: int
    stratum: tuple[str, int, int, int]
    dynamic_indices: tuple[int, ...]


def _files(dataset_path: str) -> list[Path]:
    path = Path(dataset_path).expanduser().resolve()
    candidates = [path] if path.is_file() else sorted(path.glob("*.hdf5"))
    if not candidates:
        raise FileNotFoundError(f"No HDF5 files found under {path}.")
    return candidates


class PointVelocityDataset(Dataset):
    """Lazy individual scan-event samples stored in point-velocity HDF5 episodes."""

    def __init__(self, dataset_path: str, input_name: str = "lidar_noisy") -> None:
        if input_name not in {"lidar_noisy", "lidar_clean"}:
            raise ValueError("input_name must be lidar_noisy or lidar_clean.")
        self.input_name = input_name
        self.entries: list[EpisodeEntry] = []
        self.offsets = [0]
        for file_path in _files(dataset_path):
            with h5py.File(file_path, "r") as handle:
                data = handle.get("data")
                if data is None:
                    continue
                for episode_name in sorted(data.keys()):
                    group = data[episode_name]
                    required = {"lidar_noisy", "lidar_clean", "point_velocity_w", "reflection_mask", "dynamic_mask", "range_m"}
                    if not required.issubset(group.keys()):
                        continue
                    length = int(group["lidar_noisy"].shape[0])
                    dynamic_indices = tuple(np.flatnonzero(np.asarray(group["dynamic_mask"]).any(axis=1)).tolist())
                    attrs = group.attrs
                    stratum = (
                        str(attrs.get("terrain_name", "unknown")),
                        int(attrs.get("terrain_level", -1)),
                        int(attrs.get("replica_index", -1)),
                        int(attrs.get("scenario_mode", -1)),
                    )
                    self.entries.append(EpisodeEntry(str(file_path), episode_name, length, stratum, dynamic_indices))
                    self.offsets.append(self.offsets[-1] + length)
        if not self.entries:
            raise RuntimeError("No valid point-velocity episodes were found.")

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        entry = self.entries[entry_index]
        sample_index = index - self.offsets[entry_index]
        with h5py.File(entry.file_path, "r") as handle:
            group = handle["data"][entry.episode_name]
            return {
                "lidar": torch.from_numpy(np.asarray(group[self.input_name][sample_index], dtype=np.float32)),
                "target": torch.from_numpy(np.asarray(group["point_velocity_w"][sample_index], dtype=np.float32)),
                "reflection_mask": torch.from_numpy(np.asarray(group["reflection_mask"][sample_index], dtype=np.bool_)),
                "dynamic_mask": torch.from_numpy(np.asarray(group["dynamic_mask"][sample_index], dtype=np.bool_)),
                "range_m": torch.from_numpy(np.asarray(group["range_m"][sample_index], dtype=np.float32)),
            }

    def split(self, seed: int = 42) -> tuple[Subset, Subset, Subset]:
        """Episode-level 80/10/10 split stratified by static metadata."""
        groups: dict[tuple[str, int, int, int], list[int]] = {}
        for index, entry in enumerate(self.entries):
            groups.setdefault(entry.stratum, []).append(index)
        rng = random.Random(seed)
        partitions = [[], [], []]
        for episode_ids in groups.values():
            rng.shuffle(episode_ids)
            n = len(episode_ids)
            n_test = max(1, round(n * 0.1)) if n >= 3 else 0
            n_val = max(1, round(n * 0.1)) if n - n_test >= 2 else 0
            chunks = (episode_ids[n_test + n_val :], episode_ids[n_test : n_test + n_val], episode_ids[:n_test])
            for partition, chunk in zip(partitions, chunks, strict=True):
                for episode_id in chunk:
                    partition.extend(range(self.offsets[episode_id], self.offsets[episode_id + 1]))
        return tuple(Subset(self, partition) for partition in partitions)  # type: ignore[return-value]


class DynamicAwareBatchSampler(Sampler[list[int]]):
    """Half of each batch comes from scan events containing dynamic reflections."""

    def __init__(self, subset: Subset, batch_size: int, seed: int = 42) -> None:
        self.indices = list(subset.indices)
        dataset = subset.dataset
        if not isinstance(dataset, PointVelocityDataset):
            raise TypeError("DynamicAwareBatchSampler requires a PointVelocityDataset subset.")
        dynamic_global = set()
        for entry_index, entry in enumerate(dataset.entries):
            dynamic_global.update(dataset.offsets[entry_index] + local for local in entry.dynamic_indices)
        self.dynamic = [index for index in self.indices if index in dynamic_global]
        self.batch_size = batch_size
        self.seed = seed
        if not self.dynamic:
            raise RuntimeError("Training split has no dynamic LiDAR reflection samples.")

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(len(self)):
            dynamic_count = self.batch_size // 2
            yield [rng.choice(self.dynamic) for _ in range(dynamic_count)] + [
                rng.choice(self.indices) for _ in range(self.batch_size - dynamic_count)
            ]

    def __len__(self) -> int:
        return max(1, len(self.indices) // self.batch_size)


def save_split_metadata(output_path: str, train: Subset, validation: Subset, test: Subset) -> None:
    Path(output_path).write_text(
        json.dumps({"train_indices": list(train.indices), "validation_indices": list(validation.indices), "test_indices": list(test.indices)}, indent=2),
        encoding="utf-8",
    )
