# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # if robots are required to walk at least 1 meter
    # and the robots that walked less than half of their required distance go to simpler terrains
    commanded_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    ones = torch.full(commanded_distance.size(), 1.0, device=env.device)
    move_down = torch.logical_and(commanded_distance > ones, distance <  commanded_distance * 0.5)
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def single_terrain_level(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 1.0
) -> torch.Tensor:
    """Curriculum based on the terrain level.

    This term is used to increase the difficulty of the terrain when the robot reaches the target location
    and decrease the difficulty when the robot does not reach the target location.

    .. note::
        It is only possible to use this term with the terrain type ``single_terrain_generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # Each goal has origins_per_level number of origins/terrain types.
    # We need to devide terrain_types by origins_per_level to get the goal type.
    terrain_types = terrain.terrain_types[env_ids]
    goal_types = terrain_types // terrain.cfg.single_terrain_generator.origins_per_level
    levels = terrain.terrain_levels[env_ids]

    origins = terrain.single_terrain_generator.terrain_origins[levels, terrain_types]
    goals = terrain.single_terrain_generator.goal_locations[goal_types]
    # compute the distance from the origin to the goal
    origin_to_goal_distances = torch.norm(origins - goals, dim=1)
    
    robot_pos = asset.data.root_pos_w[env_ids]
    robot_to_goal_distances = torch.norm(robot_pos - goals, dim=1)

    move_up = robot_to_goal_distances < distance_threshold

    move_down = robot_to_goal_distances > origin_to_goal_distances * 0.5

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_velocity_level(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int],
    command_name: str = "base_velocity", 
    terrain_level_to_velocity_range: dict[str, UniformVelocityCommandCfg.Ranges | UniformVelocityCommandCfg.RangesAngleMag] = None
) -> torch.Tensor:
    """Curriculum based on the terrain level.

    Define what command velocity to use when at different terrain levels.

    Returns:
        The mean terrain level for the given environment ids.
    """
    if terrain_level_to_velocity_range is None:
        raise ValueError("terrain_level_to_velocity_range must be provided to command_velocity_level function.")
    
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    command_term: UniformVelocityCommand = env.command_manager.get_term(command_name)

    # get the terrain levels
    terrain_level = torch.mean(terrain.terrain_levels[:].float())
    # get the velocity ranges for the terrain levels
    max_level = -1
    for min_level_str, _ in terrain_level_to_velocity_range.items():
        min_level = int(min_level_str)
        if min_level > max_level and min_level <= terrain_level:
            max_level = min_level

    if max_level >= 0:
        command_term.cfg.ranges = terrain_level_to_velocity_range[str(max_level)]
    
    return terrain_level

class GetMeanTerrainLevel(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.matched_env_ids = None

    def __call__(
        self,
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int],
        terrain_name: str) -> torch.Tensor:
        """Get the mean terrain level for the given environment ids.
        Args:
            env: The environment to get the terrain level from.
            env_ids: The environment ids to get the terrain level for.
            terrain_name: The name of the terrain to get the level for.
        Returns:
            The mean terrain level for the given environment ids.
        """
        # if not previously computed, compute the env ids that corresponds with terrain name.
        terrain: TerrainImporter = env.scene.terrain
        if self.matched_env_ids is None:
            env_terrain_names = terrain.get_env_terrain_names()
            self.matched_env_ids = torch.tensor([i for i, name in enumerate(env_terrain_names) if name == terrain_name], dtype=torch.int, device=env.device) 

        if self.matched_env_ids.size() == 0:
            return torch.zeros(1)
        
        matched_env_terrain_levels = terrain.terrain_levels[self.matched_env_ids]
        return torch.mean(matched_env_terrain_levels, dtype=torch.float32)