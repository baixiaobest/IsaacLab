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
from isaaclab.sensors import ContactSensor

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

def _extract_goals_from_command(cmd: object) -> torch.Tensor:
    """
    Return goals tensor of shape (num_envs, 2 or 3) from a command object.
    Tries known attributes used by TerrainBasedPose2dCommandCfg implementations.
    """
    # direct tensors on the command
    for name in ("goal_locations", "goals", "targets", "current", "target", "goal"):
        if hasattr(cmd, name):
            g = getattr(cmd, name)
            if isinstance(g, torch.Tensor):
                return g
    # nested .data container (common pattern)
    if hasattr(cmd, "data"):
        data = getattr(cmd, "data")
        for name in ("goal_locations", "goals", "targets", "current", "target", "goal"):
            if hasattr(data, name):
                g = getattr(data, name)
                if isinstance(g, torch.Tensor):
                    return g
    # generic buffer fields often used by command terms
    for name in ("values", "buf", "command", "command_buf"):
        if hasattr(cmd, name):
            g = getattr(cmd, name)
            if isinstance(g, torch.Tensor):
                # Expect [x, y, yaw] or [x, y]; slice to XY if longer
                return g
    # nothing found
    raise RuntimeError(f"Command object of type {type(cmd).__name__} does not expose goal tensor")

def _resolve_command(env: ManagerBasedRLEnv, command_name: str):
    """
    Return the command object by name, supporting different CommandManager implementations.
    Tries: .get(), .commands[name], ._terms[name], dict-like access.
    """
    cm = env.command_manager
    # method
    if hasattr(cm, "get") and callable(getattr(cm, "get")):
        try:
            return cm.get(command_name)
        except Exception:
            pass
    # dict of commands
    if hasattr(cm, "commands"):
        cmds = getattr(cm, "commands")
        if isinstance(cmds, dict) and command_name in cmds:
            return cmds[command_name]
    # internal terms dict
    if hasattr(cm, "_terms"):
        terms = getattr(cm, "_terms")
        if isinstance(terms, dict) and command_name in terms:
            return terms[command_name]
    # dict-like manager
    if isinstance(cm, dict) and command_name in cm:
        return cm[command_name]
    # last resort: attribute on env (some setups register commands at env level)
    if hasattr(env, command_name):
        return getattr(env, command_name)
    raise AttributeError(f"Cannot find command '{command_name}' in CommandManager (no commands/_terms).")

def navigation_goal_reached_by_command(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "pose_2d_command",
        distance_threshold: float = 0.5,
        velocity_threshold: float = 0.1,
        action_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Terminate when the goal from a command (e.g. pose_2d_command) is reached.

    Reads the goal from the command manager instead of the TerrainImporter.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = _resolve_command(env, command_name)

    # Extract goals robustly; on failure, print available attrs for debugging
    try:
        goals = _extract_goals_from_command(cmd)
    except RuntimeError as e:
        attrs = [a for a in dir(cmd) if not a.startswith("_")]
        raise RuntimeError(f"{e}. Available attributes: {attrs}")

    # Ensure shape (num_envs, 3) by padding Z if needed
    if goals.shape[1] == 2:
        z = torch.zeros((goals.shape[0], 1), device=goals.device, dtype=goals.dtype)
        goals = torch.cat([goals, z], dim=1)

    # Fix: pose_2d_command contains relative goal position. 
    # The distance is simply the norm of the command vector.
        robot_to_goal_distances = torch.norm(goals[:, :2], dim=1)
        
        robot_vel = torch.norm(asset.data.root_lin_vel_w, dim=1)
    action = env.action_manager.action
    no_action = torch.norm(action, dim=1) < action_threshold

    return torch.logical_and(
        torch.logical_and(robot_to_goal_distances < distance_threshold, robot_vel < velocity_threshold),
        no_action
    )


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


class navigation_goal_reached_timer_by_command(ManagerTermBase):
    """
    Terminate when the command goal is reached and maintained for a duration.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.last_time_goal_reached = torch.full((env.num_envs,), -1.0, device=env.device)
        # Counter to track total number of goal reaches
        self.goal_reached_count = 0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            command_name: str = "pose_2d_command",
            distance_threshold: float = 0.5,
            velocity_threshold: float = 0.1,
            stay_for_seconds: float = 0.1
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        cmd = _resolve_command(env, command_name)
        try:
            goals = _extract_goals_from_command(cmd)
        except RuntimeError as e:
            attrs = [a for a in dir(cmd) if not a.startswith("_")]
            raise RuntimeError(f"{e}. Available attributes: {attrs}")

        if goals.shape[1] == 2:
            z = torch.zeros((goals.shape[0], 1), device=goals.device, dtype=goals.dtype)
            goals = torch.cat([goals, z], dim=1)

        # Fix: pose_2d_command contains relative goal position. 
        # The distance is simply the norm of the command vector.
        robot_to_goal_distances = torch.norm(goals[:, :2], dim=1)
        
        # print(f"Distance to goal: {robot_to_goal_distances}")

        robot_vel = torch.norm(asset.data.root_lin_vel_w, dim=1)

        # print(f"Robot velocities: {robot_vel}")

        goal_reached = torch.logical_and(
            robot_to_goal_distances < distance_threshold,
            robot_vel < velocity_threshold
        )

        current_time = env.episode_length_buf * env.step_dt

        newly_reached = torch.logical_and(goal_reached, self.last_time_goal_reached < 0)
        self.last_time_goal_reached[newly_reached] = current_time[newly_reached]

        not_at_goal = ~goal_reached
        self.last_time_goal_reached[not_at_goal] = -1.0

        time_at_goal = torch.zeros_like(self.last_time_goal_reached)
        valid_times = self.last_time_goal_reached >= 0
        time_at_goal[valid_times] = current_time[valid_times] - self.last_time_goal_reached[valid_times]

        terminate = torch.logical_and(goal_reached, time_at_goal >= stay_for_seconds)
        
        # Count how many environments reached the goal (terminated)
        num_goals_reached = terminate.sum().item()
        if num_goals_reached > 0:
            self.goal_reached_count += num_goals_reached
        
        return terminate

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_time_goal_reached = torch.full((self._env.num_envs,), -1.0, device=self._env.device)
        elif self.last_time_goal_reached is not None:
            self.last_time_goal_reached[env_ids] = -1.0


class navigation_time_out(ManagerTermBase):
    """
    Terminate when the episode length exceeds the maximum episode length.
    Similar structure to navigation_goal_reached_timer_by_command but for timeout.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Counter to track total number of timeouts
        self.timeout_count = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Terminate the episode when the episode length exceeds the maximum episode length."""
        timeout = env.episode_length_buf >= env.max_episode_length
        
        # Count how many environments timed out
        num_timeouts = timeout.sum().item()
        if num_timeouts > 0:
            self.timeout_count += num_timeouts
        
        return timeout

    def reset(self, env_ids=None):
        """Reset the timeout tracking for specified environments.
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all environments.
                    Note: We don't reset the counter here as it's cumulative across all episodes.
        """
        # The timeout_count is cumulative, so we don't reset it here
        # This allows tracking total timeouts across the entire session
        pass


class navigation_illegal_contact(ManagerTermBase):
    """
    Terminate when the contact force on the sensor exceeds the force threshold.
    Similar structure to navigation_goal_reached_timer_by_command but for illegal contacts.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Counter to track total number of illegal contacts
        self.illegal_contact_count = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        """Terminate when the contact force on the sensor exceeds the force threshold."""
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        # check if any contact force exceeds the threshold
        illegal_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
        )
        
        # Count how many environments had illegal contacts
        num_illegal_contacts = illegal_contact.sum().item()
        if num_illegal_contacts > 0:
            self.illegal_contact_count += num_illegal_contacts
        
        return illegal_contact

    def reset(self, env_ids=None):
        """Reset the illegal contact tracking for specified environments.
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all environments.
                    Note: We don't reset the counter here as it's cumulative across all episodes.
        """
        # The illegal_contact_count is cumulative, so we don't reset it here
        # This allows tracking total illegal contacts across the entire session
        pass


class navigation_root_z_velocity_out_of_limit(ManagerTermBase):
    """
    Terminate when the asset's root z-velocity is outside the provided limits.
    Similar structure to navigation_goal_reached_timer_by_command but for z-velocity violations.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Counter to track total number of z-velocity violations
        self.z_velocity_violation_count = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_z_velocity: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """Terminate when the asset's root z-velocity is outside the provided limits."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        # compute any violations
        z_velocity_violation = torch.abs(asset.data.root_vel_w[:, 2]) > max_z_velocity
        
        # Count how many environments had z-velocity violations
        num_violations = z_velocity_violation.sum().item()
        if num_violations > 0:
            self.z_velocity_violation_count += num_violations
        
        return z_velocity_violation

    def reset(self, env_ids=None):
        """Reset the z-velocity violation tracking for specified environments.
        
        Args:
            env_ids: Indices of environments to reset. If None, reset all environments.
                    Note: We don't reset the counter here as it's cumulative across all episodes.
        """
        # The z_velocity_violation_count is cumulative, so we don't reset it here
        # This allows tracking total z-velocity violations across the entire session
        pass
