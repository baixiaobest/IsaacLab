import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs import ManagerBasedRLEnv


def pose_2d_command_terrain_curriculum(
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        command_name: str,
        distance_threshold: float = 0.5, 
        angular_threshold: float = 0.1):
    """ When pose 2d command is within threshold, goal is considered reached. Then the terrain level is increased."""
    command = env.command_manager.get_command(command_name)[env_ids]
    within_distance = torch.norm(command[:, :3], dim=1) <= distance_threshold
    within_angular_distance = torch.abs(command[:, 3]) <= angular_threshold

    move_up = torch.logical_and(within_distance, within_angular_distance)
    move_down = ~move_up

    # update terrain levels
    terrain: TerrainImporter = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean terrain level
    return {"mean": torch.mean(terrain.terrain_levels.float()), "max": torch.max(terrain.terrain_levels.float())}

def pose_2d_command_terrain_curriculum_with_threshold(
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        command_name: str,
        min_level_thresholds: int,
        max_level_thresholds: int
):
    """When the level is beyond a threshold, the level cannot be decreased."""
    command = env.command_manager.get_command(command_name)[env_ids]
    within_distance = torch.norm(command[:, :3], dim=1) <= 0.5
    within_angular_distance = torch.abs(command[:, 3]) <= 0.1

    # update terrain levels
    terrain: TerrainImporter = env.scene.terrain

    can_move_up = terrain.terrain_levels < max_level_thresholds

    move_up = torch.logical_and(within_distance, within_angular_distance)
    move_up = torch.logical_and(move_up, can_move_up)
    move_down = ~move_up

    # only allow decrease if current level is below threshold
    can_move_down = terrain.terrain_levels < min_level_thresholds
    move_down = torch.logical_and(move_down, can_move_down)

    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean terrain level
    return {"mean": torch.mean(terrain.terrain_levels.float()), "max": torch.max(terrain.terrain_levels.float())}