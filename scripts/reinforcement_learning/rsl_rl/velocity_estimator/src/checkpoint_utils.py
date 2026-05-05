# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared checkpoint helpers for velocity estimator tooling."""

from __future__ import annotations

import os

import torch

from isaaclab.utils.assets import retrieve_file_path

from scripts.reinforcement_learning.rsl_rl.velocity_estimator.src.model import VelocityEstimator


def resolve_policy_checkpoint(experiment_name: str, checkpoint: str | None, purpose: str) -> str:
    """Resolve a required policy checkpoint path."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if not checkpoint:
        raise ValueError(f"--checkpoint is required to {purpose}.")
    return retrieve_file_path(checkpoint)


def get_checkpoint_string_list(checkpoint: dict[str, object], key: str) -> list[str]:
    """Read a list of strings from a checkpoint entry."""
    value = checkpoint.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"Estimator checkpoint entry '{key}' must be a list of strings.")
    return value


def get_checkpoint_dict(checkpoint: dict[str, object], key: str) -> dict[str, object]:
    """Read a dictionary from a checkpoint entry."""
    value = checkpoint.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Estimator checkpoint entry '{key}' must be a dictionary.")
    return value


def get_checkpoint_int(checkpoint: dict[str, object], key: str) -> int:
    """Read an integer-valued checkpoint entry."""
    value = checkpoint.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"Estimator checkpoint entry '{key}' must be an integer.")
    return value


def get_checkpoint_float(mapping: dict[str, object], key: str, default: float) -> float:
    """Read a float-valued entry from a generic checkpoint mapping."""
    value = mapping.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"Checkpoint entry '{key}' must be numeric.")


def get_checkpoint_int_list(mapping: dict[str, object], key: str, default: list[int]) -> list[int]:
    """Read a list of integers from a generic checkpoint mapping."""
    value = mapping.get(key, default)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise RuntimeError(f"Checkpoint entry '{key}' must be a list of integers.")
    return value


def load_estimator_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[VelocityEstimator, dict[str, object]]:
    """Load a trained estimator checkpoint and rebuild the model."""
    estimator_checkpoint = torch.load(checkpoint_path, map_location=device)
    estimator_args = get_checkpoint_dict(estimator_checkpoint, "args")
    estimator = VelocityEstimator(
        input_dim=get_checkpoint_int(estimator_checkpoint, "input_dim"),
        horizon=get_checkpoint_int(estimator_checkpoint, "horizon"),
        output_dim=get_checkpoint_int(estimator_checkpoint, "target_dim"),
        hidden_dims=get_checkpoint_int_list(estimator_args, "hidden_dims", [256, 256, 128]),
        activation=str(estimator_args.get("activation", "elu")),
        dropout=get_checkpoint_float(estimator_args, "dropout", 0.0),
    ).to(device)
    estimator.load_state_dict(estimator_checkpoint["model_state_dict"])
    estimator.eval()
    return estimator, estimator_checkpoint