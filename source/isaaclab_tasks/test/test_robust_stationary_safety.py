"""Unit tests for Robust-v1 stationary posture and lower-head safety shaping."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.locomotion_env_cfg import (
    LocomotionVelEnvCfg,
    LocomotionVelEnvCfg_PLAY,
    LocomotionVelEnvCfg_ROBUST,
    LocomotionVelEnvCfg_ROLLOUT,
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


def test_robust_height_cost_covers_slow_straight_motion_and_tapers():
    params = {
        "target_height": 0.37,
        "planar_deadzone_mps": 0.25,
        "planar_fade_end_mps": 0.40,
        "yaw_deadzone_radps": 0.10,
    }
    for speed, expected in ((0.0, 0.01), (0.25, 0.01), (0.325, 0.005), (0.40, 0.0)):
        env = _height_reward_env(torch.tensor([[speed, 0.0, 0.0]]), 0.47)
        assert mdp.stationary_base_height_l2(env, "base_velocity", **params).item() == pytest.approx(expected)
    at_target = _height_reward_env(torch.tensor([[0.20, 0.0, 0.0]]), 0.37)
    assert mdp.stationary_base_height_l2(at_target, "base_velocity", **params).item() == 0.0
    turning = _height_reward_env(torch.tensor([[0.20, 0.0, 0.10]]), 0.47)
    assert mdp.stationary_base_height_l2(turning, "base_velocity", **params).item() == 0.0


def test_robust_feet_air_time_ignores_short_steps_at_low_speed():
    sensor = SimpleNamespace(
        compute_first_contact=lambda dt: torch.tensor([[True]]),
        data=SimpleNamespace(last_air_time=torch.tensor([[0.20]])),
    )
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0])

    def reward_at(speed: float, threshold: float) -> float:
        env = SimpleNamespace(
            step_dt=0.02,
            scene=SimpleNamespace(sensors={"contact_forces": sensor}),
            command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[speed, 0.0, 0.0]])),
        )
        return mdp.feet_air_time(
            env, "base_velocity", sensor_cfg, threshold=0.5, command_speed_threshold=threshold
        ).item()

    assert reward_at(0.20, 0.25) == 0.0
    assert reward_at(0.25, 0.25) == 0.0
    assert reward_at(0.30, 0.25) == pytest.approx(-0.30)
    assert reward_at(0.20, 0.10) == pytest.approx(-0.30)
    sensor.data.last_air_time[:] = 0.80
    assert reward_at(0.20, 0.25) == 0.0
    assert reward_at(0.30, 0.25) == pytest.approx(0.30)


def test_stationary_and_lower_head_rewards_are_robust_task_only():
    default_cfg = LocomotionVelEnvCfg()
    play_cfg = LocomotionVelEnvCfg_PLAY()
    robust_cfg = LocomotionVelEnvCfg_ROBUST()
    rollout_cfg = LocomotionVelEnvCfg_ROLLOUT()

    assert default_cfg.rewards.stationary_base_height_l2 is None
    assert not hasattr(default_cfg.rewards, "lower_head_contact")
    assert robust_cfg.rewards.stationary_base_height_l2 is not None
    assert robust_cfg.rewards.stationary_base_height_l2.func is mdp.stationary_base_height_l2
    assert robust_cfg.rewards.stationary_base_height_l2.weight == -5.0
    assert robust_cfg.rewards.stationary_base_height_l2.params["target_height"] == 0.37
    assert robust_cfg.rewards.stationary_base_height_l2.params["planar_deadzone_mps"] == 0.25
    assert robust_cfg.rewards.stationary_base_height_l2.params["planar_fade_end_mps"] == 0.40
    assert robust_cfg.commands.base_velocity.slow_coupled_turn_probability == 0.05
    assert robust_cfg.commands.base_velocity.slow_straight_probability == 0.05
    assert robust_cfg.rewards.feet_air_time.params["command_speed_threshold"] == 0.25
    assert "command_speed_threshold" not in default_cfg.rewards.feet_air_time.params
    assert "command_speed_threshold" not in play_cfg.rewards.feet_air_time.params
    assert "command_speed_threshold" not in rollout_cfg.rewards.feet_air_time.params
    assert play_cfg.rewards.stationary_base_height_l2 is None
    assert rollout_cfg.rewards.stationary_base_height_l2 is None
    assert robust_cfg.rewards.lower_head_contact.func is mdp.undesired_contacts
    assert robust_cfg.rewards.lower_head_contact.weight == -2.0
    assert robust_cfg.rewards.lower_head_contact.params["sensor_cfg"].body_names == "Head_lower"
    assert robust_cfg.rewards.lower_head_contact.params["threshold"] == 1.0
    assert default_cfg.terminations.base_contact.params["sensor_cfg"].body_names == "base"
    assert robust_cfg.terminations.base_contact.params["sensor_cfg"].body_names == "base"
