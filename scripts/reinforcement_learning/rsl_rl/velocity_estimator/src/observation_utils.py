# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for working with flattened observation groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


ESTIMATOR_TARGET_GROUP = "ground_truth"
ESTIMATOR_TARGET_TERM_NAMES = ("base_lin_vel",)


@dataclass(frozen=True)
class ObservationTermSpec:
    """Description of a single flattened observation term."""

    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


def build_observation_term_specs(
    env: ManagerBasedRLEnv,
    consumer_name: str,
) -> dict[str, list[ObservationTermSpec]]:
    """Infer flattened tensor slices for each observation term in every observation group."""
    observation_manager = env.observation_manager
    observation_specs: dict[str, list[ObservationTermSpec]] = {}

    for group_name, term_names in observation_manager.active_terms.items():
        if not observation_manager.group_obs_concatenate[group_name]:
            raise RuntimeError(
                f"Observation group '{group_name}' is not concatenated. This {consumer_name} expects concatenated "
                "groups so term slices remain well defined."
            )

        specs: list[ObservationTermSpec] = []
        offset = 0
        for term_name, term_shape in zip(term_names, observation_manager.group_obs_term_dim[group_name], strict=True):
            width = math.prod(term_shape)
            specs.append(ObservationTermSpec(term_name, tuple(term_shape), offset, offset + width))
            offset += width
        observation_specs[group_name] = specs

    return observation_specs


def serialize_observation_specs(specs: dict[str, list[ObservationTermSpec]]) -> dict[str, dict[str, object]]:
    """Convert observation specs into JSON-serializable metadata."""
    output: dict[str, dict[str, object]] = {}
    for group_name, group_specs in specs.items():
        output[group_name] = {
            "terms": [
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "slice": [spec.start, spec.stop],
                }
                for spec in group_specs
            ]
        }
    return output


def get_estimator_target_term_names() -> tuple[str, ...]:
    """Return the observation term names predicted by the velocity estimator."""
    return ESTIMATOR_TARGET_TERM_NAMES


def get_estimator_target_paths(
    specs: dict[str, list[ObservationTermSpec]],
    target_group: str = ESTIMATOR_TARGET_GROUP,
) -> list[str]:
    """Return fully qualified target paths for estimator supervision/inference."""
    if target_group not in specs:
        raise RuntimeError(f"The environment must expose the '{target_group}' observation group.")

    target_spec_map = {spec.name: spec for spec in specs[target_group]}
    missing_terms = [term_name for term_name in ESTIMATOR_TARGET_TERM_NAMES if term_name not in target_spec_map]
    if missing_terms:
        missing_terms_str = ", ".join(missing_terms)
        raise RuntimeError(
            f"The environment is missing the estimator target term(s) in '{target_group}': {missing_terms_str}"
        )

    return [f"{target_group}/{term_name}" for term_name in ESTIMATOR_TARGET_TERM_NAMES]


def split_observation_groups(
    obs_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    specs: dict[str, list[ObservationTermSpec]],
    env_id: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Split concatenated observation groups into named tensors.

    When ``env_id`` is provided, returns tensors for a single environment. Otherwise,
    returns batched tensors for every environment.
    """
    extracted: dict[str, dict[str, torch.Tensor]] = {}

    for group_name, group_obs in obs_dict.items():
        if isinstance(group_obs, dict):
            if env_id is None:
                extracted[group_name] = {name: value.detach().clone() for name, value in group_obs.items()}
            else:
                extracted[group_name] = {name: value[env_id].detach().clone() for name, value in group_obs.items()}
            continue

        extracted[group_name] = {}
        if env_id is None:
            for term_spec in specs[group_name]:
                term_value = group_obs[:, term_spec.start : term_spec.stop]
                if term_spec.shape:
                    term_value = term_value.reshape(group_obs.shape[0], *term_spec.shape)
                extracted[group_name][term_spec.name] = term_value.detach().clone()
            continue

        env_group_obs = group_obs[env_id]
        for term_spec in specs[group_name]:
            term_value = env_group_obs[term_spec.start : term_spec.stop]
            if term_spec.shape:
                term_value = term_value.reshape(term_spec.shape)
            extracted[group_name][term_spec.name] = term_value.detach().clone()

    return extracted