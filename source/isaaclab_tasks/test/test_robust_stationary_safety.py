"""Unit tests for Robust-v1 stationary posture and lower-head safety shaping."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.locomotion_env_cfg import (
    LocomotionVelEnvCfg,
    LocomotionVelEnvCfg_ROBUST,
)


def _height_reward_env(command: torch.Tensor, height: float):
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, height]])))
    return SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )


def test_stationary_base_height_reward_is_gated_by_planar_and_yaw_commands():
    stationary = _height_reward_env(torch.tensor([[0.0, 0.0, 0.0]]), 0.50)
    assert mdp.stationary_base_height_l2(stationary, "base_velocity").item() == pytest.approx(0.01)

    at_target = _height_reward_env(torch.tensor([[0.0, 0.0, 0.0]]), 0.40)
    assert mdp.stationary_base_height_l2(at_target, "base_velocity").item() == 0.0

    moving = _height_reward_env(torch.tensor([[0.10, 0.0, 0.0]]), 0.50)
    assert mdp.stationary_base_height_l2(moving, "base_velocity").item() == 0.0

    rotating = _height_reward_env(torch.tensor([[0.0, 0.0, 0.10]]), 0.50)
    assert mdp.stationary_base_height_l2(rotating, "base_velocity").item() == 0.0


def test_stationary_and_lower_head_rewards_are_robust_task_only():
    default_cfg = LocomotionVelEnvCfg()
    robust_cfg = LocomotionVelEnvCfg_ROBUST()

    assert default_cfg.rewards.stationary_base_height_l2 is None
    assert not hasattr(default_cfg.rewards, "lower_head_contact")
    assert robust_cfg.rewards.stationary_base_height_l2 is not None
    assert robust_cfg.rewards.stationary_base_height_l2.func is mdp.stationary_base_height_l2
    assert robust_cfg.rewards.stationary_base_height_l2.weight == -5.0
    assert robust_cfg.rewards.lower_head_contact.func is mdp.undesired_contacts
    assert robust_cfg.rewards.lower_head_contact.weight == -2.0
    assert robust_cfg.rewards.lower_head_contact.params["sensor_cfg"].body_names == "Head_lower"
    assert robust_cfg.rewards.lower_head_contact.params["threshold"] == 1.0
    assert default_cfg.terminations.base_contact.params["sensor_cfg"].body_names == "base"
    assert robust_cfg.terminations.base_contact.params["sensor_cfg"].body_names == "base"
