from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from collections.abc import Sequence
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def navigation_goal_reached(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        distance_threshold: float=0.5, 
        velocity_threshold: float=0.1
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    This is useful for tasks where the goal is to reach a specific position or orientation.
    """
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

    return torch.logical_and(robot_to_goal_distances < distance_threshold, robot_vel < velocity_threshold)
