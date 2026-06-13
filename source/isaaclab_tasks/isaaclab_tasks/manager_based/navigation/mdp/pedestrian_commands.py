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

    def _resample_corridor_pose(self, env_ids: Sequence[int]):
        """Sample a corridor-relative flow/crossing goal for ``env_ids``.

        Shared by :class:`CorridorPedestrianPose2dCommand` (all envs) and
        :class:`MixedTerrainPose2dCommand` (pedestrian-corridor envs only). Reads
        ``env.pedestrian_scenario_mode`` (0 = flow, 1 = cross S->N, 2 = cross N->S), set by
        ``reset_pedestrian_scenario_robot``.
        """
        corridor_origin = self._corridor_origins(env_ids)
        self.pos_command_w[env_ids] = corridor_origin

        n = len(env_ids)
        mode = self._env.pedestrian_scenario_mode[env_ids]
        is_crossing = mode >= 1
        is_north_start = mode == 2  # spawned north → goal on the south side

        # --- flow goal (local-x up/downstream, small local-y offset) ---
        direction = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.tensor(-1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        distance = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_distance_range)
        flow_x = (direction * distance).clamp(-self.cfg.corridor_half_length, self.cfg.corridor_half_length)
        flow_y = torch.empty(n, device=self.device).uniform_(
            -self.cfg.corridor_half_width, self.cfg.corridor_half_width
        )

        # --- crossing goal (far side across the flow, random local-x) ---
        cross_x = torch.empty(n, device=self.device).uniform_(*self.cfg.crossing_x_range)
        # North-start crosses to the south (-goal_y); south-start crosses to the north (+goal_y).
        cross_y = torch.where(is_north_start, -self.cfg.goal_y, self.cfg.goal_y)

        local_x = torch.where(is_crossing, cross_x, flow_x)
        local_y = torch.where(is_crossing, cross_y, flow_y)

        self.pos_command_w[env_ids, 0] += local_x
        self.pos_command_w[env_ids, 1] += local_y
        self._set_pos_z(env_ids)
        self._sample_heading(env_ids)

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


class CorridorPedestrianPose2dCommand(_CorridorPose2dCommandBase):
    """Unified corridor goal command that hosts both the flow and crossing scenarios.

    The per-env episode scenario is read from ``env.pedestrian_scenario_mode`` (0 = flow,
    1 = crossing south→north, 2 = crossing north→south), which is sampled at reset by
    ``reset_pedestrian_scenario_robot``. Both goal formulas are computed vectorized and selected
    per-env with ``torch.where``, so a single command term co-trains both scenarios on one
    corridor terrain:

    - **flow** (mode 0): goal placed up/downstream along the corridor's flow axis (local-x)
      at a random distance/direction, with a small lateral (local-y) offset.
    - **crossing** (modes 1/2): goal placed on the far side across the flow at corridor-local
      ``y = ±goal_y`` (sign opposite the robot's spawn side), with a random local-x. The two
      crossing directions let the robot see the crowd sweep across from both relative sides.
    """

    cfg: CorridorPedestrianPose2dCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        self._resample_corridor_pose(env_ids)


@configclass
class CorridorPedestrianPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for :class:`CorridorPedestrianPose2dCommand` (unified flow + crossing)."""

    class_type: type = CorridorPedestrianPose2dCommand

    # -- flow-scenario fields (mode 0) --
    goal_distance_range: tuple[float, float] = (4.0, 8.0)
    """Range of distances (m) along the corridor axis at which a flow goal is sampled."""

    corridor_half_length: float = 9.0
    """Half-length of the corridor along its flow axis (m); flow goal local-x is clamped to this."""

    corridor_half_width: float = 2.0
    """Half-width of the corridor (m); flow goal local-y is sampled within ``±corridor_half_width``."""

    # -- crossing-scenario fields (modes 1/2) --
    goal_y: float = 5.0
    """Corridor-local |y| at which a crossing goal is placed; the sign is chosen opposite the
    robot's spawn side (south-start → +goal_y, north-start → -goal_y)."""

    crossing_x_range: tuple[float, float] = (-1.5, 1.5)
    """Range of corridor-local x at which a crossing goal is sampled."""
