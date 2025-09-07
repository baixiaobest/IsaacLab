# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_range(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    zero_command_distance: float = 0.1,
    range: tuple[float, float] = (0.5, 1.0),
    T: float = 0.3
) -> torch.Tensor:
    """ Reward long steps taken by the feet within a specified range.

    This function rewards the agent for taking steps that are within a specified range. The reward is computed
    using double sigmoid function, which decreases to zero outside the range and increases to 1 within the range.

    First, define a preliminary function as f1(x) = 1 / (1 + exp(-c * (x-r1))) - 1 / (1 + exp(-c * (x-r2))),
    where r1 and r2 are the lower and upper bounds of the range, respectively, and c is a constant that controls
    the steepness of the sigmoid function.

    The steepness c is defined by: c = 4.39 / T. This means that the function increase from 0.1 to 0.9 in range of T at
    the boundary r1 and r2.

    The problem with f1 is that the reward is only 0.5 when x is at the boundary r1 or r2. We can make the reward to 0.9 
    by defining:
        f2(x) = 1 / (1 + exp(-c * (x - r1 + T/2))) - 1 / (1 + exp(-c * (x - r2 -T/2))), 
    which shifts the sigmoid function outward so that f2(r1) = 0.9 and f2(r2) = 0.9.

    f2(x) is what we compute for the reward.

    """

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    c = 4.39 / T  # steepness of the sigmoid function
    r1, r2 = range  # lower and upper bounds of the range
    # compute the reward using the double sigmoid function
    reward = (
        1.0 / (1.0 + torch.exp(-c * (last_air_time - r1 + T / 2))) -
        1.0 / (1.0 + torch.exp(-c * (last_air_time - r2 - T / 2)))
    ) * first_contact
    reward_dim1 = reward.size(dim=1)
    reward = torch.sum(reward, dim=1) / reward_dim1
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > zero_command_distance
    # return the reward
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def flying_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the robot for being in the air (not in contact with the ground).

    This function penalizes the agent for being in the air. 
    This encourages the agent to keep at least one foot on the ground.
    """
    # Penalize flying
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0
    all_in_air = torch.all(in_air, dim=1).float()

    return all_in_air


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_drag_penalty(
    env: ManagerBasedRLEnv, 
    contact_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 1.0,
    use_horizontal_vel_only: bool = False
) -> torch.Tensor:
    """Penalize foot dragging when feet are in contact with the ground.
    
    This function detects dragging by checking when:
    1. Foot is in contact (normal force above threshold)
    2. Foot has tangential velocity relative to the contact surface
    
    Args:
        env: The environment.
        contact_sensor_cfg: Configuration for the contact sensor with body_ids indicating feet.
        asset_cfg: Configuration for the robot asset.
        force_threshold: Minimum contact force to consider the foot in contact.
        use_horizontal_vel_only: If True, only penalize horizontal (XY) movement of feet.
    
    Returns:
        Tensor with drag penalty for each environment instance.
    """
    # Get the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    # Get the articulation
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the normal contact forces
    contact_forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    
    # Check which feet are in contact with the ground
    contact_norms = torch.norm(contact_forces, dim=-1)
    is_in_contact = contact_norms > force_threshold
    
    # Get the normal direction of contact (normalized)
    normal_dirs = torch.zeros_like(contact_forces)
    # Avoid division by zero for non-contact points
    mask = contact_norms > 1e-6
    normal_dirs[mask] = contact_forces[mask] / contact_norms[mask].unsqueeze(-1)
    
    # Get the velocities of the contact bodies (feet)
    body_velocities = asset.data.body_lin_vel_w[:, contact_sensor_cfg.body_ids]
    
    # Project velocity onto the tangential plane (perpendicular to normal)
    vel_dot_normal = torch.sum(body_velocities * normal_dirs, dim=-1, keepdim=True)
    tangential_vel = body_velocities - vel_dot_normal * normal_dirs
    
    # Option to only consider horizontal movement
    if use_horizontal_vel_only:
        tangential_vel = tangential_vel.clone()
        tangential_vel[..., 2] = 0.0
    
    # Compute the magnitude of tangential velocity
    tangential_vel_norm = torch.norm(tangential_vel, dim=-1)
    
    # Only penalize when in contact
    drag_penalty = is_in_contact * tangential_vel_norm
    
    # Sum over all bodies (feet) for each environment
    return torch.sum(drag_penalty, dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
