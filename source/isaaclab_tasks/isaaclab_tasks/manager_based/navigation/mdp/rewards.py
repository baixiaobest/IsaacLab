# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import inspect
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors.ray_caster import RayCaster

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
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        velocity_threshold: float = 0.1,
        heading_deadband: float = 0.0 # in rad
        ) -> torch.Tensor:
    """Penalize the difference between the direction of velocity and the robot's heading."""
    asset = env.scene[asset_cfg.name]
    robot_heading = asset.data.heading_w
    robot_vel = asset.data.root_lin_vel_w
    has_vel = torch.norm(robot_vel, dim=1) > velocity_threshold

    rewards = torch.zeros(env.num_envs, device=env.device)

    if has_vel.any():
        vel_heading = torch.atan2(robot_vel[has_vel, 1], robot_vel[has_vel, 0])
        # Calculate the absolute difference between the robot's heading and the velocity heading
        heading_diff = torch.abs(math_utils.wrap_to_pi(robot_heading[has_vel] - vel_heading))
        heading_diff = torch.clamp(heading_diff - heading_deadband, min=0.0)  # Apply deadband
        rewards[has_vel] = heading_diff / torch.pi  # Normalize to [0, 1]
    
    return rewards

class terrain_specific_reward_callback(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.matched_env_ids = None
        self.rewards_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            func: callable,
            terrain_names: list[str],
            callback_params: dict = {}
    ) -> torch.Tensor:
        """Callback reward for terrain-specific conditions."""
        terrain: TerrainImporter = env.scene.terrain

        if self.matched_env_ids is None:
            env_terrain_names = terrain.get_env_terrain_names()
            self.matched_env_ids = torch.tensor(
                [i for i, name in enumerate(env_terrain_names) if name in terrain_names], 
                dtype=torch.int, 
                device=env.device) 
            self.rewards_mask[self.matched_env_ids] = 1.0
            
        if self.matched_env_ids.size() == 0:
            return torch.zeros(env.num_envs, device=env.device)

        rewards = func(env, **callback_params)

        return rewards * self.rewards_mask
    

"""
Pose 2d command related rewards
"""

def pose_2d_command_goal_reached_reward(
        env: ManagerBasedRLEnv, 
        command_name: str, 
        distance_threshold: float = 0.5, 
        angular_threshold: float = 0.1,
        distance_reward_multiplier: float = 1.5,
        angular_reward_multiplier: float = 1.2,
        active_after_time: float = 0.0) -> torch.Tensor:
    """ When pose 2d command is within threshold, goal is considered reached. """
    command = env.command_manager.get_command(command_name)
    robot_to_goal_distance = torch.norm(command[:, :3], dim=1)
    within_distance = robot_to_goal_distance <= distance_threshold
    within_angular_distance = torch.abs(command[:, 3]) <= angular_threshold

    goal_reached = torch.logical_and(within_distance, within_angular_distance)

    distance_multiplier = torch.ones_like(robot_to_goal_distance)
    distance_multiplier[goal_reached] = distance_reward_multiplier - (distance_reward_multiplier - 1.0) * (
        robot_to_goal_distance[goal_reached] / distance_threshold
    )

    angular_multiplier = torch.ones_like(distance_multiplier)
    angular_multiplier[goal_reached] = angular_reward_multiplier - (angular_reward_multiplier - 1.0) * (
        torch.abs(command[goal_reached, 3]) / angular_threshold
    )

    reward_active = (env.episode_length_buf * env.step_dt) >= active_after_time

    return goal_reached * reward_active * distance_multiplier * angular_multiplier

class pose_2d_command_goal_reached_once_reward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reward_awarded = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids = None):
        if env_ids is None:
            # Reset all environments
            self.reward_awarded = torch.zeros(self._env.num_envs, device=self._env.device)
        elif self.reward_awarded is not None:
            # For partial resets, set rewards to zero
            self.reward_awarded[env_ids] = 0.0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            command_name: str,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            distance_threshold: float = 0.5,
            angular_threshold: float = 0.1
    ) -> torch.Tensor:
        """Reward based on the average velocity of the robot."""
        robot: Articulation = env.scene[asset_cfg.name]

        pose_command = env.command_manager.get_command(command_name)
        robot_to_goal_distance = torch.norm(pose_command[:, :3], dim=1)

        within_distance = robot_to_goal_distance <= distance_threshold
        within_angular_distance = torch.abs(pose_command[:, 3]) <= angular_threshold

        goal_reached = torch.logical_and(within_distance, within_angular_distance)

        new_goal_reached = torch.logical_and(goal_reached, self.reward_awarded == 0.0)
        self.reward_awarded = torch.logical_or(self.reward_awarded, new_goal_reached)

        return new_goal_reached

def pose_2d_command_progress_reward(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        std: float = 1.0
) -> torch.Tensor:
    """Reward for velocity toward the goal position"""

    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_dir_b = goal_pos_b / (torch.norm(goal_pos_b, dim=1, keepdim=True) + 1e-6)
    robot: Articulation = env.scene[asset_cfg.name]
    robot_vel_b = robot.data.root_lin_vel_b

    return torch.tanh(torch.sum(robot_vel_b * goal_dir_b, dim=1) / std)

def pose_2d_goal_callback_reward(
        env: ManagerBasedRLEnv,
        func: callable,
        command_name: str,
        distance_threshold: float = 0.5,
        angular_threshold: float = 0.1,
        callback_params: dict = {} ) -> torch.Tensor:
    """Callback reward for reaching the goal position."""
    command = env.command_manager.get_command(command_name)
    robot_to_goal_distance = torch.norm(command[:, :3], dim=1)
    within_distance = robot_to_goal_distance <= distance_threshold
    within_angular_distance = torch.abs(command[:, 3]) <= angular_threshold

    goal_reached = torch.logical_and(within_distance, within_angular_distance)
    
    params = callback_params.copy()

    sig = inspect.signature(func)
        
    if 'goal_reached' in sig.parameters:
        params['goal_reached'] = goal_reached

    # Call the instance with the parameters
    return goal_reached * func(env, **params)

def pose_2d_command_norm_penalty(
        env: ManagerBasedRLEnv,
        command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    # Calculate the norm of the command vector
    command_sq = torch.square(command).sum(dim=1)
    
    return command_sq

"""
Scalar velocity command related rewards
"""

class average_velocity_reward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reward_awarded = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids = None):
        if env_ids is None:
            # Reset all environments
            self.reward_awarded = torch.zeros(self._env.num_envs, device=self._env.device)
        elif self.reward_awarded is not None:
            # For partial resets, set rewards to zero
            self.reward_awarded[env_ids] = 0.0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            pose_command_name: str,
            scalar_vel_command_name: str,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            std: float = 1.0,
            distance_threshold: float = 0.5,
            angular_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Reward based on the average velocity of the robot."""
        pose_command = env.command_manager.get_command(pose_command_name)
        robot_to_goal_distance = torch.norm(pose_command[:, :3], dim=1)
        within_distance = robot_to_goal_distance <= distance_threshold
        within_angular_distance = torch.abs(pose_command[:, 3]) <= angular_threshold

        goal_reached = torch.logical_and(within_distance, within_angular_distance)

        new_goal_reached = torch.logical_and(goal_reached, self.reward_awarded == 0.0)
        self.reward_awarded = torch.logical_or(self.reward_awarded, new_goal_reached)

        origins = env.scene.terrain.env_origins
        robot: Articulation = env.scene[asset_cfg.name]
        scalar_velocity_command = env.command_manager.get_command(scalar_vel_command_name)[:, 0]

        distance_travelled = torch.norm(robot.data.root_pos_w - origins, dim=1)
        average_velocity = distance_travelled / (env.step_dt * env.episode_length_buf + 1e-6)
        velocity_diff = torch.abs(average_velocity - scalar_velocity_command)

        reward = 1.0 - torch.tanh(velocity_diff / std)

        reward[~new_goal_reached] = 0.0  # Set reward to 0 for environments that have not reached the goal

        return reward
"""
Navigation command related rewards
"""

class goal_reached_reward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.reward_awarded = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        """Reset the reward state for specified environments.
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all environments.
        """
        if env_ids is None:
            # Reset all environments
            self.reward_awarded = torch.zeros(self._env.num_envs, device=self._env.device)
        elif self.reward_awarded is not None:
            # For partial resets, set rewards to zero
            self.reward_awarded[env_ids] = 0.0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            distance_threshold: float = 0.5,
            velocity_threshold: float = 0.1,
            action_threshold: float = 0.05,
            reward_multiplier: float = 2.0) -> torch.Tensor:
        """Reward for reaching the goal position."""
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        terrain: TerrainImporter = env.scene.terrain
        
        # Each goal has origins_per_level number of origins/terrain types.
        # We need to devide terrain_types by origins_per_level to get the goal type.
        terrain_types = terrain.terrain_types
        goal_types = terrain_types // terrain.cfg.single_terrain_generator.origins_per_level

        goals = terrain.single_terrain_generator.goal_locations[goal_types]
        
        robot_pos = asset.data.root_pos_w
        robot_to_goal_distances = torch.norm(robot_pos - goals, dim=1)

        robot_vel = torch.norm(asset.data.root_lin_vel_w, dim=1)

        goal_reached = torch.logical_and(
            robot_to_goal_distances < distance_threshold, 
            robot_vel < velocity_threshold)
        
        action = env.action_manager.action
        no_action = torch.norm(action, dim=1) < action_threshold

        # We only award the reward if the goal is reached and not already awarded
        # This prevents multiple rewards in the same episode
        should_award = torch.logical_and(torch.logical_and(
            goal_reached, 
            self.reward_awarded == 0.0),
            no_action)

        # set corresponding reward_awarded to 1.0 if reward should be awarded
        self.reward_awarded[should_award] = 1.0

        # Calculate distance-based reward multiplier
        # Linear interpolation from reward_multiplier at distance 0 to 1.0 at distance_threshold
        distance_multiplier = torch.ones_like(robot_to_goal_distances)
        distance_multiplier[should_award] = reward_multiplier - (reward_multiplier - 1.0) * (
            robot_to_goal_distances[should_award] / distance_threshold
        )

        # The reward is scaled by 1/sted_dt, so the average reward per second is 1.0 if the robot reaches the goal
        return should_award.float() \
                * distance_multiplier \
                * (1.0 / env.step_dt) 

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

def goal_position_error_tanh(
        env: ManagerBasedRLEnv, 
        std: float, command_term_name: str, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward for moving towards the goal position with tanh kernel."""
    command_term = env.command_manager.get_term(command_term_name)
    goal_positions = command_term.goal_positions
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w
    # Calculate the distance to the goal position
    distance_to_goal = torch.norm(goal_positions - robot_pos, dim=1)

    return 1 - torch.tanh(distance_to_goal / std)

def lateral_movement_penalty(
        env: ManagerBasedRLEnv,
        std: float,
        command_term_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes a penalty based on the robot's lateral (perpendicular) velocity 
    relative to the direction towards the goal.

    The penalty is higher when the robot moves more sideways with respect to the 
    goal direction, encouraging the robot to move directly towards the goal.

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) with the lateral movement penalty for each environment.
    """
    command_term = env.command_manager.get_term(command_term_name)
    goal_positions = command_term.goal_positions
    
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w
    robot_vel = robot.data.root_lin_vel_w

    robot_to_goal_vec = goal_positions - robot_pos
    robot_to_goal_norm = torch.norm(robot_to_goal_vec, dim=1, keepdim=True)
    robot_to_goal_dir = robot_to_goal_vec / robot_to_goal_norm

    lateral_vel = robot_vel - torch.sum(robot_vel * robot_to_goal_dir, dim=1, keepdim=True) * robot_to_goal_dir
    lateral_vel_norm = torch.norm(lateral_vel, dim=1)

    return torch.tanh(lateral_vel_norm / std)

def lateral_movement_penalty_obstacle_dependent(
        env: ManagerBasedRLEnv,
        command_term_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("obstacle_scanner"),
        std_lateral: float = 1.0,
        std_obstacle: float = 1.0) -> torch.Tensor:
    """Computes a penalty based on the robot's lateral (perpendicular) velocity 
    relative to the direction towards the goal, adjusted by the distance to obstacles.
    """
    command_term = env.command_manager.get_term(command_term_name)
    goal_positions = command_term.goal_positions
    
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w
    robot_vel = robot.data.root_lin_vel_w

    robot_to_goal_vec = goal_positions - robot_pos
    robot_to_goal_norm = torch.norm(robot_to_goal_vec, dim=1, keepdim=True)
    robot_to_goal_dir = robot_to_goal_vec / robot_to_goal_norm

    lateral_vel = robot_vel - torch.sum(robot_vel * robot_to_goal_dir, dim=1, keepdim=True) * robot_to_goal_dir
    lateral_vel_norm = torch.norm(lateral_vel, dim=1)

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]   
    # Get the distances to the closest obstacles
    distances = torch.norm((sensor.data.pos_w.unsqueeze(1) - sensor.data.ray_hits_w), dim=2)
    min_distances, _ = torch.min(distances, dim=1)

    obstacle_weight = torch.tanh(min_distances / std_obstacle)

    return torch.tanh(lateral_vel_norm / std_lateral) * obstacle_weight


def obstacle_clearance_penalty(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        std: float = 1.0,
        sensor_radius = 0.2) -> torch.Tensor:
    """Penalty for being too close to obstacles."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]   
    # Get the distances to the closest obstacles
    distances = torch.norm((sensor.data.pos_w.unsqueeze(1) - sensor.data.ray_hits_w), dim=2)
    min_distances, _ = torch.min(distances, dim=1)
    # Calculate the penalty based on the distances
    return 1.0 - torch.tanh((min_distances - sensor_radius) / std)
