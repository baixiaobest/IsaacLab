# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import bisect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class EpisodeIndexEntry:
    """Metadata for one episode inside an HDF5 rollout file."""

    file_path: str
    episode_name: str
    num_steps: int


class Hdf5ValidationResult(NamedTuple):
    """Validation status for one candidate HDF5 file."""

    file_path: str
    is_valid: bool
    error: str | None = None


def _require_group(node: h5py.Group | h5py.Dataset | h5py.Datatype, context: str) -> h5py.Group:
    """Assert that an HDF5 node is a group and return it."""
    if not isinstance(node, h5py.Group):
        raise RuntimeError(f"Expected HDF5 group for {context}, got {type(node).__name__}")
    return node


def _require_dataset(node: h5py.Group | h5py.Dataset | h5py.Datatype, context: str) -> h5py.Dataset:
    """Assert that an HDF5 node is a dataset and return it."""
    if not isinstance(node, h5py.Dataset):
        raise RuntimeError(f"Expected HDF5 dataset for {context}, got {type(node).__name__}")
    return node


def _validate_hdf5_file(file_path: str) -> Hdf5ValidationResult:
    """Check that a file can be opened and has the expected rollout layout."""
    try:
        with h5py.File(file_path, "r") as handle:
            if "data" not in handle:
                return Hdf5ValidationResult(file_path=file_path, is_valid=False, error="missing root group 'data'")
            _ = handle["data"].attrs.get("env_args", "{}")
        return Hdf5ValidationResult(file_path=file_path, is_valid=True)
    except Exception as error:  # noqa: BLE001 - surface the exact HDF5 failure to the user.
        return Hdf5ValidationResult(file_path=file_path, is_valid=False, error=str(error))


def _collect_dataset_files(dataset_path: str) -> list[str]:
    """Resolve a dataset path into a sorted list of HDF5 files."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".hdf5":
            raise ValueError(f"Expected an HDF5 file, got: {path}")
        validation = _validate_hdf5_file(str(path))
        if not validation.is_valid:
            raise RuntimeError(
                "The provided dataset file cannot be opened as a rollout HDF5 dataset: "
                f"{path}\nUnderlying error: {validation.error}"
            )
        return [str(path)]

    candidate_files = sorted(str(file_path) for file_path in path.glob("*.hdf5"))
    if not candidate_files:
        raise FileNotFoundError(f"No .hdf5 files found under: {path}")

    valid_files: list[str] = []
    invalid_files: list[Hdf5ValidationResult] = []
    for file_path in candidate_files:
        validation = _validate_hdf5_file(file_path)
        if validation.is_valid:
            valid_files.append(file_path)
        else:
            invalid_files.append(validation)

    if not valid_files:
        error_lines = [f"- {Path(result.file_path).name}: {result.error}" for result in invalid_files]
        raise RuntimeError(
            f"No readable rollout HDF5 files were found under: {path}\n"
            "All candidate files failed validation:\n"
            + "\n".join(error_lines)
            + "\nRegenerate the rollout dataset or remove the corrupted files before training."
        )

    if invalid_files:
        skipped = ", ".join(Path(result.file_path).name for result in invalid_files)
        print(f"[WARN] Skipping unreadable rollout dataset files: {skipped}")

    return valid_files


def _load_env_args(file_path: str) -> dict[str, object]:
    """Read dataset metadata stored in the HDF5 root group."""
    with h5py.File(file_path, "r") as handle:
        data_group = _require_group(handle["data"], f"{file_path}:data")
        env_args_raw = data_group.attrs.get("env_args", "{}")
    return json.loads(env_args_raw)


def _list_leaf_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    """Return sorted leaf dataset paths relative to the provided group."""
    leaf_paths: list[str] = []
    for key in sorted(group.keys()):
        value = group[key]
        key_path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            leaf_paths.extend(_list_leaf_paths(value, key_path))
        else:
            leaf_paths.append(key_path)
    return leaf_paths


def _first_leaf_length(group: h5py.Group) -> int:
    """Infer episode length from the first leaf dataset in the group."""
    for key in sorted(group.keys()):
        value = group[key]
        if isinstance(value, h5py.Group):
            length = _first_leaf_length(value)
            if length > 0:
                return length
        else:
            leaf = _require_dataset(value, f"{group.name}/{key}")
            return int(leaf.shape[0])
    return 0


def _read_leaf_window(dataset: h5py.Dataset, end_index: int, horizon: int) -> np.ndarray:
    """Read a fixed-length history window, left-padding with the first frame when needed."""
    start_index = max(0, end_index - horizon + 1)
    window = np.asarray(dataset[start_index : end_index + 1], dtype=np.float32)
    if window.shape[0] < horizon:
        pad_count = horizon - window.shape[0]
        padding = np.repeat(window[0:1], pad_count, axis=0)
        window = np.concatenate([padding, window], axis=0)
    return window


def _flatten_window(window: np.ndarray) -> np.ndarray:
    """Flatten a horizon window to (H, D)."""
    return window.reshape(window.shape[0], -1)


def _read_target(dataset: h5py.Dataset, step_index: int) -> np.ndarray:
    """Read the target value for a single step and flatten it to 1D."""
    target = np.asarray(dataset[step_index], dtype=np.float32)
    return target.reshape(-1)


class VelocityEstimatorDataset(Dataset):
    """Sliding-window dataset for velocity estimation from rollout episodes."""

    def __init__(
        self,
        dataset_path: str,
        horizon: int,
        observation_group: str = "observations",
        target_group: str = "ground_truth",
        excluded_input_terms: set[str] | None = None,
        target_term_names: tuple[str, ...] | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero.")

        self.dataset_files = _collect_dataset_files(dataset_path)
        self.horizon = horizon
        self.observation_group = observation_group
        self.target_group = target_group
        self.env_args = _load_env_args(self.dataset_files[0])
        self.target_term_names = tuple(target_term_names) if target_term_names is not None else None

        self.episode_entries: list[EpisodeIndexEntry] = []
        self.episode_offsets: list[int] = []
        self.input_paths: list[str] = []
        self.target_paths: list[str] = []

        self._build_index(excluded_input_terms)
        self._num_samples = self.episode_offsets[-1] if self.episode_offsets else 0
        if self._num_samples == 0:
            raise RuntimeError("No training samples were found in the dataset.")

        self.input_dim = self._infer_input_dim()
        self.target_dim = self._infer_target_dim()

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._num_samples:
            raise IndexError(f"Sample index out of range: {index}")

        episode_idx = bisect.bisect_right(self.episode_offsets, index) - 1
        episode_entry = self.episode_entries[episode_idx]
        step_index = index - self.episode_offsets[episode_idx]

        with h5py.File(episode_entry.file_path, "r") as handle:
            data_group = _require_group(handle["data"], f"{episode_entry.file_path}:data")
            episode_group = _require_group(data_group[episode_entry.episode_name], episode_entry.episode_name)
            observation_group = _require_group(episode_group[self.observation_group], self.observation_group)
            target_group = _require_group(episode_group[self.target_group], self.target_group)

            input_chunks = []
            for path in self.input_paths:
                leaf = _require_dataset(observation_group[path], f"{observation_group.name}/{path}")
                window = _read_leaf_window(leaf, step_index, self.horizon)
                input_chunks.append(_flatten_window(window))

            target_chunks = []
            for path in self.target_paths:
                leaf = _require_dataset(target_group[path], f"{target_group.name}/{path}")
                target_chunks.append(_read_target(leaf, step_index))

        inputs = np.concatenate(input_chunks, axis=-1)
        targets = np.concatenate(target_chunks, axis=0)
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
        }

    def get_episode_splits(self, validation_fraction: float, seed: int) -> tuple[Subset, Subset]:
        """Split the dataset by episode so windows from the same episode stay together."""
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must satisfy 0.0 <= value < 1.0.")

        episode_ids = list(range(len(self.episode_entries)))
        if not episode_ids:
            raise RuntimeError("No episodes are available for splitting.")

        rng = random.Random(seed)
        rng.shuffle(episode_ids)

        if validation_fraction == 0.0 or len(episode_ids) == 1:
            train_indices = self._episode_ids_to_sample_indices(episode_ids)
            return Subset(self, train_indices), Subset(self, [])

        num_validation_episodes = max(1, int(round(len(episode_ids) * validation_fraction)))
        num_validation_episodes = min(num_validation_episodes, len(episode_ids) - 1)
        validation_episode_ids = set(episode_ids[:num_validation_episodes])
        train_episode_ids = [episode_id for episode_id in episode_ids if episode_id not in validation_episode_ids]

        train_indices = self._episode_ids_to_sample_indices(train_episode_ids)
        validation_indices = self._episode_ids_to_sample_indices(sorted(validation_episode_ids))
        return Subset(self, train_indices), Subset(self, validation_indices)

    def _build_index(self, excluded_input_terms: set[str] | None) -> None:
        """Collect episode metadata and infer input/target schemas from the first episode."""
        running_offset = 0
        for file_path in self.dataset_files:
            with h5py.File(file_path, "r") as handle:
                data_group = _require_group(handle["data"], f"{file_path}:data")
                for episode_name in sorted(data_group.keys()):
                    episode_group = _require_group(data_group[episode_name], episode_name)
                    if self.observation_group not in episode_group or self.target_group not in episode_group:
                        continue

                    observation_group = _require_group(episode_group[self.observation_group], self.observation_group)
                    target_group = _require_group(episode_group[self.target_group], self.target_group)

                    if not self.target_paths:
                        all_target_paths = _list_leaf_paths(target_group)
                        if self.target_term_names is None:
                            self.target_paths = all_target_paths
                        else:
                            self.target_paths = [
                                path for path in all_target_paths if Path(path).name in set(self.target_term_names)
                            ]
                            missing_target_terms = sorted(
                                set(self.target_term_names).difference({Path(path).name for path in self.target_paths})
                            )
                            if missing_target_terms:
                                missing_target_terms_str = ", ".join(missing_target_terms)
                                raise RuntimeError(
                                    "The rollout dataset is missing the requested estimator target term(s): "
                                    f"{missing_target_terms_str}"
                                )
                        target_term_names = {Path(path).name for path in self.target_paths}
                        blocked_terms = set(excluded_input_terms or set()) | target_term_names
                        self.input_paths = [
                            path for path in _list_leaf_paths(observation_group) if Path(path).name not in blocked_terms
                        ]
                        if not self.input_paths:
                            raise RuntimeError("No estimator input terms remain after excluding velocity terms.")
                        if not self.target_paths:
                            raise RuntimeError("No ground truth target terms were found in the dataset.")

                    num_steps = _first_leaf_length(target_group)
                    if num_steps <= 0:
                        continue

                    self.episode_offsets.append(running_offset)
                    self.episode_entries.append(EpisodeIndexEntry(file_path=file_path, episode_name=episode_name, num_steps=num_steps))
                    running_offset += num_steps

        if self.episode_entries:
            self.episode_offsets.append(running_offset)

    def _episode_ids_to_sample_indices(self, episode_ids: list[int]) -> list[int]:
        """Convert episode IDs into flattened sample indices."""
        sample_indices: list[int] = []
        for episode_id in episode_ids:
            start = self.episode_offsets[episode_id]
            stop = start + self.episode_entries[episode_id].num_steps
            sample_indices.extend(range(start, stop))
        return sample_indices

    def _infer_input_dim(self) -> int:
        """Compute flattened feature dimension for one time step."""
        with h5py.File(self.episode_entries[0].file_path, "r") as handle:
            data_group = _require_group(handle["data"], f"{self.episode_entries[0].file_path}:data")
            episode_group = _require_group(data_group[self.episode_entries[0].episode_name], self.episode_entries[0].episode_name)
            observation_group = _require_group(episode_group[self.observation_group], self.observation_group)
            total_dim = 0
            for path in self.input_paths:
                leaf = _require_dataset(observation_group[path], f"{observation_group.name}/{path}")
                total_dim += int(np.prod(leaf.shape[1:]))
        return total_dim

    def _infer_target_dim(self) -> int:
        """Compute flattened target dimension."""
        with h5py.File(self.episode_entries[0].file_path, "r") as handle:
            data_group = _require_group(handle["data"], f"{self.episode_entries[0].file_path}:data")
            episode_group = _require_group(data_group[self.episode_entries[0].episode_name], self.episode_entries[0].episode_name)
            target_group = _require_group(episode_group[self.target_group], self.target_group)
            total_dim = 0
            for path in self.target_paths:
                leaf = _require_dataset(target_group[path], f"{target_group.name}/{path}")
                total_dim += int(np.prod(leaf.shape[1:]))
        return total_dim


def create_estimator_datasets(
    dataset_path: str,
    horizon: int,
    validation_fraction: float,
    seed: int,
    excluded_input_terms: set[str] | None = None,
    target_term_names: tuple[str, ...] | None = None,
) -> tuple[VelocityEstimatorDataset, Subset, Subset]:
    """Build the full dataset and episode-level train/validation splits."""
    dataset = VelocityEstimatorDataset(
        dataset_path=dataset_path,
        horizon=horizon,
        excluded_input_terms=excluded_input_terms,
        target_term_names=target_term_names,
    )
    train_subset, validation_subset = dataset.get_episode_splits(validation_fraction=validation_fraction, seed=seed)
    return dataset, train_subset, validation_subset