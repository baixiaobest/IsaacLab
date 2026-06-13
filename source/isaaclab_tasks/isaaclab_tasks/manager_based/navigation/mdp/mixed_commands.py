"""Goal-position command for the mixed static/pedestrian obstacle-avoidance environment.

Each env is permanently pinned (at terrain-importer init) to either a "ped_corridor" column or
a static obstacle/maze column (``terrain.terrain_types``, fixed for the whole run).
:class:`MixedTerrainPose2dCommand` resamples each env's goal with the matching strategy:

- pedestrian-corridor envs: corridor-relative flow/crossing goals
  (:meth:`_CorridorPose2dCommandBase._resample_corridor_pose`, shared with
  :class:`CorridorPedestrianPose2dCommand`).
- static envs: flat-patch terrain goals, replicating
  :class:`isaaclab.envs.mdp.commands.pose_2d_command.TerrainBasedPose2dCommand`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from .pedestrian_commands import CorridorPedestrianPose2dCommandCfg, _CorridorPose2dCommandBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MixedTerrainPose2dCommand(_CorridorPose2dCommandBase):
    """Per-env goal command that branches between corridor (pedestrian) and flat-patch
    (static) goal sampling based on each env's fixed terrain-column assignment."""

    cfg: MixedTerrainPose2dCommandCfg

    def __init__(self, cfg: MixedTerrainPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "MixedTerrainPose2dCommand requires a valid flat patch under 'target' in the"
                f" terrain. Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

        # Self-contained (the command manager is built before PedestrianCrowdNavigationEnv's
        # __init__ body sets env.is_pedestrian_env), derived from the same terrain data.
        env_terrain_names = self.terrain.get_env_terrain_names()
        self._is_pedestrian_env = torch.tensor(
            [name == "ped_corridor" for name in env_terrain_names], dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ped_mask = self._is_pedestrian_env[env_ids_t]
        ped_env_ids = env_ids_t[ped_mask]
        static_env_ids = env_ids_t[~ped_mask]

        if len(ped_env_ids) > 0:
            self._resample_corridor_pose(ped_env_ids)
        if len(static_env_ids) > 0:
            self._resample_terrain_pose(static_env_ids)

    def _resample_terrain_pose(self, env_ids: torch.Tensor):
        """Flat-patch terrain goal sampling, replicating
        :class:`isaaclab.envs.mdp.commands.pose_2d_command.TerrainBasedPose2dCommand`
        (isaaclab/envs/mdp/commands/pose_2d_command.py)."""
        stationary_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.stationary_prob

        non_stationary_env_ids = env_ids[~stationary_mask]
        if len(non_stationary_env_ids) > 0:
            ids = torch.randint(0, self.valid_targets.shape[2], size=(len(non_stationary_env_ids),), device=self.device)
            levels = self.terrain.terrain_levels[non_stationary_env_ids]
            types = self.terrain.terrain_types[non_stationary_env_ids]
            self.pos_command_w[non_stationary_env_ids] = self.valid_targets[levels, types, ids]
            self._set_pos_z(non_stationary_env_ids)

        stationary_env_ids = env_ids[stationary_mask]
        if len(stationary_env_ids) > 0:
            self.pos_command_w[stationary_env_ids] = self.robot.data.root_pos_w[stationary_env_ids]

        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)


@configclass
class MixedTerrainPose2dCommandCfg(CorridorPedestrianPose2dCommandCfg):
    """Configuration for :class:`MixedTerrainPose2dCommand`."""

    class_type: type = MixedTerrainPose2dCommand
