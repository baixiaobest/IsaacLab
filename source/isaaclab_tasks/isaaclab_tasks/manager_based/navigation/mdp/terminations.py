from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from collections.abc import Sequence
from isaaclab.terrains import TerrainImporter
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def navigation_goal_reached(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        distance_threshold: float=0.5, 
        velocity_threshold: float=0.1,
        action_threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    This is useful for tasks where the goal is to reach a specific position or orientation.
    """
    # extract the used quantities (to enable typehinting)
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

    action = env.action_manager.action
    no_action = torch.norm(action, dim=1) < action_threshold

    return torch.logical_and(torch.logical_and(
        robot_to_goal_distances < distance_threshold, 
        robot_vel < velocity_threshold),
        no_action)


class navigation_goal_reached_timer(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Initialize with negative values to indicate "not at goal"
        self.last_time_goal_reached = torch.full((env.num_envs,), -1.0, device=env.device)

    def __call__(
            self,
            env: ManagerBasedRLEnv, 
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            distance_threshold: float=0.5, 
            velocity_threshold: float=0.1,
            stay_for_seconds: float = 0.1
    ) -> torch.Tensor:
        """Terminate the episode when the goal is reached and maintained for a duration.

        This is useful for tasks where the goal is to reach a specific position or orientation.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        terrain: TerrainImporter = env.scene.terrain
        
        # Each goal has origins_per_level number of origins/terrain types.
        # We need to divide terrain_types by origins_per_level to get the goal type.
        terrain_types = terrain.terrain_types
        goal_types = terrain_types // terrain.cfg.single_terrain_generator.origins_per_level

        goals = terrain.single_terrain_generator.goal_locations[goal_types]
        
        robot_pos = asset.data.root_pos_w
        robot_to_goal_distances = torch.norm(robot_pos - goals, dim=1)
        robot_vel = torch.norm(asset.data.root_lin_vel_w, dim=1)

        # Check if robot is at goal with low velocity
        goal_reached = torch.logical_and(
            robot_to_goal_distances < distance_threshold, 
            robot_vel < velocity_threshold
        )
        
        current_time = env.episode_length_buf * env.step_dt
        
        # For robots that just reached the goal, record the time
        newly_reached = torch.logical_and(goal_reached, self.last_time_goal_reached < 0)
        self.last_time_goal_reached[newly_reached] = current_time[newly_reached]
        
        # For robots that are no longer at the goal, reset the timer
        not_at_goal = ~goal_reached
        self.last_time_goal_reached[not_at_goal] = -1.0
        
        # Calculate how long each robot has been at the goal
        time_at_goal = torch.zeros_like(self.last_time_goal_reached)
        valid_times = self.last_time_goal_reached >= 0
        time_at_goal[valid_times] = current_time[valid_times] - self.last_time_goal_reached[valid_times]
        
        # Terminate if robot has been at goal for the specified duration
        terminate = torch.logical_and(goal_reached, time_at_goal >= stay_for_seconds)
        
        return terminate

    def reset(self, env_ids=None):
        """Reset the goal tracking for specified environments.
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all environments.
        """
        if env_ids is None:
            # Reset all environments
            self.last_time_goal_reached = torch.full(
                (self._env.num_envs,), -1.0, device=self._env.device
            )
        elif self.last_time_goal_reached is not None:
            # For partial resets, set times to negative
            self.last_time_goal_reached[env_ids] = -1.0
