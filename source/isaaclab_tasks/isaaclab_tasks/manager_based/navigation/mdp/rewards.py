# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def velocity_heading_error_abs(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the difference between the direction of velocity and the robot's heading."""
    asset = env.scene[asset_cfg.name]
    robot_heading = asset.data.heading_w
    robot_vel = asset.data.root_lin_vel_w
    has_vel = torch.norm(robot_vel, dim=1) > 0.01

    rewards = torch.zeros(env.num_envs, device=env.device)

    if has_vel.any():
        vel_heading = torch.atan2(robot_vel[has_vel, 1], robot_vel[has_vel, 0])
        # Calculate the absolute difference between the robot's heading and the velocity heading
        heading_diff = torch.abs(math_utils.wrap_to_pi(robot_heading[has_vel] - vel_heading))
        rewards[has_vel] = heading_diff / torch.pi  # Normalize to [0, 1]
    
    return rewards


def goal_position_error_tanh(env: ManagerBasedRLEnv, std: float, command_term_name: str) -> torch.Tensor:
    """Reward for moving towards the goal position with tanh kernel."""
    command_term = env.command_manager.get_term(command_term_name)
    goal_positions = command_term.goal_positions
    robot_pos = env.scene["robot"].data.root_pos_w
    # Calculate the distance to the goal position
    distance_to_goal = torch.norm(goal_positions - robot_pos, dim=1)

    return 1 - torch.tanh(distance_to_goal / std)

class navigation_progress(ManagerTermBase):
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # initialize to -1.0
        self.prev_distance = torch.full(
            (env.num_envs,), -1.0, device=env.device
        )

    def __call__(self, env: ManagerBasedRLEnv, command_term_name: str, scale: float=1.0) -> torch.Tensor:
        """Reward for moving towards the goal position."""
        goal_positions = env.command_manager.get_term(command_term_name).goal_positions
        robot_pos = env.scene["robot"].data.root_pos_w
        # Calculate the distance to the goal position
        distance_to_goal = torch.norm(goal_positions - robot_pos, dim=1)

        valid_prev_dist_mask = self.prev_distance > 0.0
        progress = torch.zeros(env.num_envs, device=env.device)

        if torch.any(valid_prev_dist_mask):
            # Calculate progress only for valid previous distances
            progress[valid_prev_dist_mask] = (
                self.prev_distance[valid_prev_dist_mask] - distance_to_goal[valid_prev_dist_mask]
            )
        
        self.prev_distance = distance_to_goal.clone()

        return torch.tanh(progress * scale)
    
    def reset(self, env_ids=None):
        if env_ids is None:
            # Reset all environments
            self.prev_distance = torch.full(
                (self._env.num_envs,), -1.0, device=self._env.device
            )
        elif self.prev_distance is not None:
            # For partial resets, set distances to invalid values
            # We use -1.0 which will be detected as invalid in the next call
            self.prev_distance[env_ids] = -1.0
    

class navigation_command_w_rate_penalty_l2(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.prev_quat = None

    def __call__(
        self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
        """Penalize the difference between the current and previous actions in the world frame."""
        asset = env.scene[asset_cfg.name]
        current_quat = asset.data.root_quat_w

        # Initialize reward tensor
        penalty = torch.zeros(env.num_envs, device=env.device)
        
        # First call - initialize quaternions for all environments
        if self.prev_quat is None:
            self.prev_quat = current_quat.clone()
            return penalty
            
        # Create mask for valid quaternions (norm should be close to 1)
        quat_norm = torch.norm(self.prev_quat, dim=1)
        valid_quat_mask = torch.logical_and(quat_norm > 0.9, quat_norm < 1.1)
        
        # Only compute penalties for environments with valid quaternions
        if torch.any(valid_quat_mask):
            # action b is vx, vy, angular velocity z, we want to convert vx, vy into global frame
            action_b = env.action_manager.action
            action_ang_vel = action_b[:, 2]
            action_vel_xy_b = torch.zeros((env.num_envs, 3), device=env.device)
            action_vel_xy_b[:, :2] = action_b[:, :2]

            prev_action_b = env.action_manager.prev_action
            prev_action_ang_vel = prev_action_b[:, 2]
            prev_action_vel_xy_b = torch.zeros((env.num_envs, 3), device=env.device)
            prev_action_vel_xy_b[:, :2] = prev_action_b[:, :2]

            action_w = torch.zeros((env.num_envs, 4), device=env.device)
            prev_action_w = torch.zeros((env.num_envs, 4), device=env.device)
            
            # Convert all to world frame
            action_w[:, :3] = math_utils.quat_rotate(current_quat, action_vel_xy_b)
            
            # Only compute for valid quaternions to avoid errors in quat_rotate_inverse
            # For environments with valid quaternions:
            prev_action_w[valid_quat_mask, :3] = math_utils.quat_rotate(
                self.prev_quat[valid_quat_mask], 
                prev_action_vel_xy_b[valid_quat_mask]
            )
            
            action_w[:, 3] = action_ang_vel
            prev_action_w[:, 3] = prev_action_ang_vel

            # Compute penalty only for environments with valid quaternions
            penalty[valid_quat_mask] = torch.sum(
                torch.square(
                    action_w[valid_quat_mask] - prev_action_w[valid_quat_mask]
                ), 
                dim=1
            )

        # Update previous quaternion for next iteration - use current for all envs
        self.prev_quat = current_quat.clone()

        return penalty

    def reset(self, env_ids=None):
        """Reset the previous quaternion state for specified environments.
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all environments.
        """
        if env_ids is None:
            # Reset all environments
            self.prev_quat = None
        elif self.prev_quat is not None:
            # For partial resets, set quaternions to invalid values
            # We use zeros which will be detected as invalid in the next call
            self.prev_quat[env_ids] = 0.0

