"""Goal-position commands for pedestrian-flow obstacle-avoidance scenarios.

Both commands sample the goal relative to the per-env corridor origin
(``terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]``) rather than
``scene.env_origins`` directly, so the goal stays inside the active corridor sub-terrain
regardless of which terrain row/column an env currently occupies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class _CorridorPose2dCommandBase(UniformPose2dCommand):
    """Shared corridor-origin lookup, heading sampling, and z-offset for corridor goals."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.terrain: TerrainImporter = env.scene["terrain"]

    def _corridor_origins(self, env_ids: Sequence[int]) -> torch.Tensor:
        levels = self.terrain.terrain_levels[env_ids]
        types = self.terrain.terrain_types[env_ids]
        return self.terrain.terrain_origins[levels, types]

    def _set_pos_z(self, env_ids: Sequence[int]):
        if self.cfg.ranges.pos_z is not None:
            r = torch.empty(len(env_ids), device=self.device)
            self.pos_command_w[env_ids, 2] += r.uniform_(*self.cfg.ranges.pos_z)
        else:
            self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

    def _sample_heading(self, env_ids: Sequence[int]):
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


class CorridorFlowPose2dCommand(_CorridorPose2dCommandBase):
    """Goal sampled up- or downstream of the corridor origin (scenarios a/b: with/against flow).

    The goal is placed at a random distance along the corridor axis (local x), in a random
    direction (50/50), clamped to stay within the corridor. Whether the resulting episode is a
    "with-flow" or "against-flow" instance is purely a consequence of this random direction
    relative to the crowd's flow direction — no separate code paths are needed.
    """

    cfg: CorridorFlowPose2dCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        corridor_origin = self._corridor_origins(env_ids)
        self.pos_command_w[env_ids] = corridor_origin

        n = len(env_ids)
        direction = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.tensor(-1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        distance = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_distance_range)
        local_x = (direction * distance).clamp(-self.cfg.corridor_half_length, self.cfg.corridor_half_length)
        local_y = torch.empty(n, device=self.device).uniform_(-self.cfg.corridor_half_width, self.cfg.corridor_half_width)

        self.pos_command_w[env_ids, 0] += local_x
        self.pos_command_w[env_ids, 1] += local_y
        self._set_pos_z(env_ids)
        self._sample_heading(env_ids)


class CorridorCrossingPose2dCommand(_CorridorPose2dCommandBase):
    """Goal sampled on the far side of the crossing corridor (scenario c: crossing the flow).

    The robot spawn (near corridor-local y = ``spawn_y``) is handled entirely by
    ``EventCfg.reset_base``'s ``pose_range`` — this command only places the goal on the
    opposite side, at corridor-local y = ``goal_y``.
    """

    cfg: CorridorCrossingPose2dCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        corridor_origin = self._corridor_origins(env_ids)
        self.pos_command_w[env_ids] = corridor_origin

        n = len(env_ids)
        local_x = torch.empty(n, device=self.device).uniform_(*self.cfg.x_range)

        self.pos_command_w[env_ids, 0] += local_x
        self.pos_command_w[env_ids, 1] += self.cfg.goal_y
        self._set_pos_z(env_ids)
        self._sample_heading(env_ids)


@configclass
class CorridorFlowPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for :class:`CorridorFlowPose2dCommand`."""

    class_type: type = CorridorFlowPose2dCommand

    goal_distance_range: tuple[float, float] = (4.0, 8.0)
    """Range of distances (m) along the corridor axis at which the goal is sampled."""

    corridor_half_length: float = 9.0
    """Half-length of the corridor along its flow axis (m); goal local-x is clamped to this."""

    corridor_half_width: float = 2.0
    """Half-width of the corridor (m); goal local-y is sampled within ``±corridor_half_width``."""


@configclass
class CorridorCrossingPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for :class:`CorridorCrossingPose2dCommand`."""

    class_type: type = CorridorCrossingPose2dCommand

    spawn_y: float = -5.0
    """Corridor-local y at which the robot spawns (informational; spawn is set via reset_base)."""

    goal_y: float = 5.0
    """Corridor-local y at which the goal is placed (opposite side of the flow corridor)."""

    x_range: tuple[float, float] = (-1.5, 1.5)
    """Range of corridor-local x at which the goal is sampled."""
