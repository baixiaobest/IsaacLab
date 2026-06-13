"""Shared observation modifier presets for Go2 navigation/locomotion configs."""

from __future__ import annotations

import isaaclab.utils.modifiers as modifiers

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp


def _episode_scale(scale_min: tuple[float, ...], scale_max: tuple[float, ...]) -> modifiers.ModifierCfg:
    return modifiers.ModifierCfg(
        func=nav_mdp.UniformEpisodeScale,
        params={"scale_min": scale_min, "scale_max": scale_max},
    )


def _episode_bias(bias_min: tuple[float, ...], bias_max: tuple[float, ...]) -> modifiers.ModifierCfg:
    return modifiers.ModifierCfg(
        func=nav_mdp.UniformEpisodeBias,
        params={"bias_min": bias_min, "bias_max": bias_max},
    )


def _random_walk_bias(
    step_min: tuple[float, ...],
    step_max: tuple[float, ...],
    drift_min: tuple[float, ...],
    drift_max: tuple[float, ...],
) -> modifiers.ModifierCfg:
    return modifiers.ModifierCfg(
        func=nav_mdp.UniformRandomWalkBias,
        params={
            "step_min": step_min,
            "step_max": step_max,
            "drift_min": drift_min,
            "drift_max": drift_max,
        },
    )


def _dropout(drop_probability: tuple[float, ...], fill_value: float = 0.0) -> modifiers.ModifierCfg:
    return modifiers.ModifierCfg(
        func=nav_mdp.ElementwiseDropout,
        params={"drop_probability": drop_probability, "fill_value": fill_value},
    )


def policy_base_lin_vel_modifiers() -> list[modifiers.ModifierCfg]:
    return [
        _episode_scale((0.92, 0.92, 0.98), (1.08, 1.08, 1.02)),
        _episode_bias((-0.08, -0.08, -0.02), (0.08, 0.08, 0.02)),
        _random_walk_bias(
            step_min=(-0.002, -0.002, -0.0005),
            step_max=(0.002, 0.002, 0.0005),
            drift_min=(-0.15, -0.15, -0.03),
            drift_max=(0.15, 0.15, 0.03),
        ),
        _dropout((0.02, 0.02, 0.01)),
    ]


def policy_imu_ang_vel_modifiers() -> list[modifiers.ModifierCfg]:
    return [
        _episode_scale((0.97, 0.97, 0.97), (1.03, 1.03, 1.03)),
        _episode_bias((-0.08, -0.08, -0.08), (0.08, 0.08, 0.08)),
        _random_walk_bias(
            step_min=(-0.003, -0.003, -0.003),
            step_max=(0.003, 0.003, 0.003),
            drift_min=(-0.12, -0.12, -0.12),
            drift_max=(0.12, 0.12, 0.12),
        ),
        _dropout((0.03, 0.03, 0.03)),
    ]


def policy_imu_lin_acc_modifiers() -> list[modifiers.ModifierCfg]:
    return [
        _episode_scale((0.92, 0.92, 0.92), (1.08, 1.08, 1.08)),
        _episode_bias((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        _random_walk_bias(
            step_min=(-0.02, -0.02, -0.02),
            step_max=(0.02, 0.02, 0.02),
            drift_min=(-1.0, -1.0, -1.0),
            drift_max=(1.0, 1.0, 1.0),
        ),
        _dropout((0.04, 0.04, 0.04)),
    ]
