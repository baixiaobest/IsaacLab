# Author: Baixiao Huang
# Date: 2025-06-26

"""Sub-module containing command generators for the goal position navigation task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from .commands_cfg import NavigationPositionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class NavigationPositionCommand(CommandTerm):
    """Command generator for the goal position navigation task."""

    cfg: NavigationPositionCommandCfg

    def __init__(self, cfg: NavigationPositionCommandCfg, env: ManagerBasedEnv):
        from .commands_cfg import NavigationPositionCommandCfg

        """Initialize the command generator."""
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Position of the goal in world frame.
        self.goal_positions = self._get_goal_positions(torch.arange(env.num_envs, device=env.device))

        # Relative position of the goal in the robot base frame.
        self.navigation_commands = torch.zeros(
            (env.num_envs, 3), dtype=torch.float32, device=env.device
        )

        if not env.scene.terrain.cfg.terrain_type == "single_terrain_generator":
            raise ValueError(
                "The NavigationPositionCommand can only be used with the single_terrain_generator terrain type."
            )
        
        self.output_velocity = isinstance(self.cfg.command, NavigationPositionCommandCfg.VelocityCommand)

        
    @property
    def command(self) -> torch.Tensor:
        """The desired position in robot base frame. Shape is (num_envs, 3)."""
        return self.navigation_commands
    
    def _resample_command(self, env_ids):
        self.goal_positions[env_ids] = self._get_goal_positions(env_ids)
    
    def _update_metrics(self):
        """Update metrics for the navigation command."""
        # Get robot position and orientation
        robot_pos = self.robot.data.root_pos_w
        robot_rot_quat_w = self.robot.data.root_quat_w
        
        # Calculate position error (distance to goal)
        relative_position = self.goal_positions - robot_pos
        distance_to_goal = torch.norm(relative_position, dim=1)
        
        # Convert the relative position to the robot's base frame
        relative_position_b = math_utils.quat_rotate_inverse(
            robot_rot_quat_w, relative_position
        )
        
        # Calculate desired heading in the robot's base frame
        # This is the angle in the XY plane to the goal
        heading_error = torch.atan2(relative_position_b[:, 1], relative_position_b[:, 0])

        self.metrics = {
            "distance_to_goal": distance_to_goal,
            "heading_error": heading_error,
            "relative_position": relative_position,
            "relative_position_b": relative_position_b
        }

    
    def _update_command(self):
        robot_pos = self.robot.data.root_pos_w
        robot_rot_quat_w = self.robot.data.root_quat_w
        relative_position = self.goal_positions - robot_pos

        if self.output_velocity:
            distance_to_goal = torch.norm(relative_position, dim=1, keepdim=True)
            # Velocity command vector in world frame, pointing to the goal and clamped to the maximum velocity.
            vel_cmd_w = relative_position / distance_to_goal * torch.clamp(distance_to_goal, 0.0, self.cfg.command.max_velocity)
            
            # Trasform velocity command to the robot base frame.
            vel_cmd_b = math_utils.quat_rotate_inverse(robot_rot_quat_w, vel_cmd_w) * torch.norm(vel_cmd_w, dim=1, keepdim=True) 
            
            # Z component of the velocity command is later used for the heading command.
            # So we need to normalize the XY component to the length of world command velocity.
            vel_cmd_b = vel_cmd_b / torch.norm(vel_cmd_b[:, :2], dim=1, keepdim=True) * torch.norm(vel_cmd_w, dim=1, keepdim=True)
            goal_heading_b = torch.atan2(vel_cmd_b[:, 1], vel_cmd_b[:, 0])
            vel_cmd_b[:, 2] = goal_heading_b * self.cfg.command.P_heading
            self.navigation_commands = vel_cmd_b
        else:    
            # Get the goal positions in the robot base frame.
            goal_position_local = math_utils.quat_rotate_inverse(
                robot_rot_quat_w, relative_position
            )
            self.navigation_commands = goal_position_local

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/navigation_velocity_goal"
                ))
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/navigation_velocity_current"
                ))
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.navigation_commands[:, :2], body_frame=True)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """ Helper functions"""

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, body_frame: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = torch.tensor([1.0, 1.0, 0.1], device=self.device)
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # convert everything back from base to world frame
        if body_frame:
            base_quat_w = self.robot.data.root_quat_w
            arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    
    def _get_goal_positions(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Get the goal positions for the given environment ids."""
        terrain: TerrainImporter = self._env.scene.terrain

        # Each goal has origins_per_level number of origins/terrain types.
        # We need to devide terrain_types by origins_per_level to get the goal type.
        terrain_types = terrain.terrain_types[env_ids]
        goal_types = terrain_types // terrain.cfg.single_terrain_generator.origins_per_level

        goals = terrain.single_terrain_generator.goal_locations[goal_types]

        return goals

