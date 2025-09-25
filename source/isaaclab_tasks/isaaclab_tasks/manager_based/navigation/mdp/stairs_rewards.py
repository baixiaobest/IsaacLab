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

def _transform_command_to_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Transform a pose2d command from body frame to world frame.
    
    Args:
        env: Environment
        command_name: Name of the command
        asset_cfg: Asset configuration
        
    Returns:
        World position of command target. Shape: (num_envs, 3)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get robot position and orientation in world frame
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    
    # Get command in robot's local frame
    # Commands are typically stored in env.commands[command_name]
    # Assuming command is (x, y, theta) in local frame
    command = env.command_manager.get_command(command_name)  # Shape: (num_envs, 3)
    
    # Extract command positions (x, y) and convert to 3D
    command_local_pos = torch.zeros((env.num_envs, 3), device=env.device)
    command_local_pos[:, :] = command[:, :3]
    
    
    # Transform local position to world frame
    command_world_pos = math_utils.quat_apply(robot_quat, command_local_pos) + robot_pos
    
    return command_world_pos

def _find_closest_points_on_segments(
    positions: torch.Tensor,  # Shape: (batch_size, 3)
    guide_lines: torch.Tensor,  # Shape: (batch_size, max_points, 3)
    valid_guide_mask: torch.Tensor,  # Shape: (batch_size, max_points)
    z_threshold: float = 1.0  # Maximum allowed z-distance from position
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find closest points on line segments defined by guide lines.
    
    Args:
        positions: Positions to find closest points for. Shape: (batch_size, 3)
        guide_lines: Guide line points. Shape: (batch_size, max_points, 3)
        valid_guide_mask: Mask for valid guide points. Shape: (batch_size, max_points)
        z_threshold: Maximum allowed z-distance from position
        
    Returns:
        tuple containing:
            - closest_points: Points on segments closest to positions. Shape: (batch_size, 3)
            - min_segment_distances: Distances to closest points. Shape: (batch_size)
            - segment_indices: Indices of closest segments. Shape: (batch_size)
            - t_values: Parametric values along segments. Shape: (batch_size)
    """
    # Create segments by pairing consecutive points
    p1 = guide_lines[:, :-1, :]  # Shape: (batch_size, max_points-1, 3)
    p2 = guide_lines[:, 1:, :]   # Shape: (batch_size, max_points-1, 3)
    
    # Create mask for valid segments (both points must be valid)
    valid_segment_mask = valid_guide_mask[:, :-1] & valid_guide_mask[:, 1:]  # Shape: (batch_size, max_points-1)
    
    # Calculate segment vectors and position-to-p1 vectors
    segment_vec = p2 - p1  # Shape: (batch_size, max_points-1, 3)
    pos_to_p1 = positions.unsqueeze(1) - p1  # Shape: (batch_size, max_points-1, 3)
    
    # Calculate segment lengths squared
    segment_length_sq = torch.sum(segment_vec ** 2, dim=-1)  # Shape: (batch_size, max_points-1)
    
    # Handle degenerate segments (length near zero)
    degenerate_mask = segment_length_sq < 1e-8
    
    # Calculate projection parameter t
    dot_product = torch.sum(pos_to_p1 * segment_vec, dim=-1)  # Shape: (batch_size, max_points-1)
    t = torch.where(
        degenerate_mask,
        torch.zeros_like(dot_product),
        dot_product / segment_length_sq
    )
    
    # Clamp t to [0, 1] to ensure projection is on segment
    t_clamped = torch.clamp(t, 0.0, 1.0)
    
    # Calculate closest points on segments
    closest_points_per_segment = p1 + t_clamped.unsqueeze(-1) * segment_vec  # Shape: (batch_size, max_points-1, 3)
    
    # NOW apply z-threshold filtering based on the computed closest points
    # This ensures the projected point is within z_threshold
    closest_point_z_valid = torch.abs(closest_points_per_segment[:, :, 2] - positions[:, 2].unsqueeze(1)) <= z_threshold
    
    # Combine with existing valid segment mask
    valid_segment_mask = valid_segment_mask & closest_point_z_valid
    
    # Calculate distances from positions to closest points on segments
    segment_distances = torch.norm(positions.unsqueeze(1) - closest_points_per_segment, dim=-1)  # Shape: (batch_size, max_points-1)
    
    # Handle degenerate cases: use distance to p1, but also apply z-threshold check
    degenerate_distances = torch.norm(pos_to_p1, dim=-1)
    # For degenerate segments, check z-threshold against p1
    degenerate_z_valid = torch.abs(p1[:, :, 2] - positions[:, 2].unsqueeze(1)) <= z_threshold
    
    segment_distances = torch.where(
        degenerate_mask,
        degenerate_distances,
        segment_distances
    )
    
    # Update valid mask for degenerate segments
    valid_segment_mask = torch.where(
        degenerate_mask,
        valid_segment_mask & degenerate_z_valid,
        valid_segment_mask
    )
    
    # Mask invalid segments with infinity
    masked_segment_distances = torch.where(
        valid_segment_mask,
        segment_distances,
        torch.tensor(float('inf'), device=positions.device)
    )
    
    # Find minimum distance and index for each position
    min_segment_distances, segment_indices = torch.min(masked_segment_distances, dim=1)
    
    # Handle case where no valid segments exist
    no_valid_segments = torch.all(~valid_segment_mask, dim=1)
    segment_indices = torch.where(no_valid_segments, torch.tensor(-1, device=positions.device), segment_indices)
    min_segment_distances = torch.where(no_valid_segments, torch.tensor(float('inf'), device=positions.device), min_segment_distances)
    
    # Gather the closest points and t values
    batch_indices = torch.arange(positions.shape[0], device=positions.device)
    
    # Only gather for valid indices
    valid_batch_mask = segment_indices >= 0
    closest_points = torch.zeros_like(positions)
    t_values = torch.zeros(positions.shape[0], device=positions.device)
    
    if valid_batch_mask.any():
        valid_batch_indices = batch_indices[valid_batch_mask]
        valid_segment_indices = segment_indices[valid_batch_mask]
        
        closest_points[valid_batch_mask] = closest_points_per_segment[valid_batch_indices, valid_segment_indices]
        t_values[valid_batch_mask] = t_clamped[valid_batch_indices, valid_segment_indices]
    
    return closest_points, min_segment_distances, segment_indices, t_values


def _compute_path_distance(
    guide_lines: torch.Tensor,
    valid_guide_mask: torch.Tensor,
    start_indices: torch.Tensor,
    start_t: torch.Tensor,
    end_indices: torch.Tensor,
    end_t: torch.Tensor,
    robot_pos: torch.Tensor,
    command_pos: torch.Tensor,
) -> torch.Tensor:
    """GPU-optimized path distance calculation along guide lines."""
    batch_size = guide_lines.shape[0]
    max_points = guide_lines.shape[1]
    device = guide_lines.device
    dtype = guide_lines.dtype  # Get the data type from input tensor
    
    # Calculate segment lengths (distance between consecutive guide points)
    segment_lengths = torch.norm(guide_lines[:, 1:] - guide_lines[:, :-1], dim=-1)  # (batch_size, max_points-1)
    
    # Mask out invalid segments with zeros
    valid_segment_mask = valid_guide_mask[:, :-1] & valid_guide_mask[:, 1:]
    segment_lengths = torch.where(valid_segment_mask, segment_lengths, torch.zeros_like(segment_lengths))
    
    # Create case masks
    same_segment_mask = (start_indices == end_indices)
    forward_mask = (start_indices < end_indices) & ~same_segment_mask
    backward_mask = (start_indices > end_indices) & ~same_segment_mask
    
    # Initialize path distances
    path_distances = torch.full((batch_size,), float('inf'), device=device, dtype=dtype)
    
    # CASE 1: Same segment - direct distance between points
    if same_segment_mask.any():
        direct_distances = torch.norm(command_pos - robot_pos, dim=1)
        path_distances = torch.where(same_segment_mask, direct_distances, path_distances)
    
    # CASE 2: Forward direction (start_idx < end_idx)
    if forward_mask.any():
        # Create a tensor to store forward distances for all environments
        all_forward_distances = torch.zeros_like(path_distances)
        
        # Create a mask to identify which segments are between start and end for each env
        segment_indices = torch.arange(max_points-1, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # For each environment, mark segments that are between start and end indices
        between_segments_mask = (segment_indices > start_indices.unsqueeze(1)) & \
                               (segment_indices < end_indices.unsqueeze(1))
        between_segments_mask = between_segments_mask & forward_mask.unsqueeze(1)
        
        # Get the indices of environments with forward paths
        forward_indices = forward_mask.nonzero(as_tuple=True)[0]
        
        if len(forward_indices) > 0:
            # Compute partial length of start segment (from projection to end)
            start_segment_contrib = torch.zeros_like(path_distances)
            
            # Ensure consistent data types with explicit casting
            start_segment_values = (segment_lengths[forward_indices, start_indices[forward_indices]] * \
                                  (1.0 - start_t[forward_indices])).to(dtype=dtype)
            start_segment_contrib.index_put_((forward_indices,), start_segment_values)
            
            # Compute partial length of end segment (from start to projection)
            end_segment_contrib = torch.zeros_like(path_distances)
            
            # Ensure consistent data types with explicit casting
            end_segment_values = (segment_lengths[forward_indices, end_indices[forward_indices]] * \
                                end_t[forward_indices]).to(dtype=dtype)
            end_segment_contrib.index_put_((forward_indices,), end_segment_values)
            
            # Sum all intermediate segment lengths
            intermediate_segments_sum = torch.sum(segment_lengths * between_segments_mask.float(), dim=1)
            
            # Compute total path distance for forward case
            all_forward_distances = start_segment_contrib + intermediate_segments_sum + end_segment_contrib
            
            # Update path distances for forward case
            path_distances = torch.where(forward_mask, all_forward_distances, path_distances)
    
    # CASE 3: Backward direction (start_idx > end_idx)
    if backward_mask.any():
        # Create a tensor to store backward distances for all environments
        all_backward_distances = torch.zeros_like(path_distances)
        
        # Create a mask for segments between end and start (going backward)
        segment_indices = torch.arange(max_points-1, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # For each environment, mark segments that are between end and start indices (reversed)
        between_segments_mask = (segment_indices < start_indices.unsqueeze(1)) & \
                               (segment_indices > end_indices.unsqueeze(1))
        between_segments_mask = between_segments_mask & backward_mask.unsqueeze(1)
        
        # Get the indices of environments with backward paths
        backward_indices = backward_mask.nonzero(as_tuple=True)[0]
        
        if len(backward_indices) > 0:
            # Compute partial length of start segment (from projection to start)
            start_segment_contrib = torch.zeros_like(path_distances)
            
            # Ensure consistent data types with explicit casting
            start_segment_values = (segment_lengths[backward_indices, start_indices[backward_indices]] * \
                                  start_t[backward_indices]).to(dtype=dtype)
            start_segment_contrib.index_put_((backward_indices,), start_segment_values)
            
            # Compute partial length of end segment (from end to projection)
            end_segment_contrib = torch.zeros_like(path_distances)
            
            # Ensure consistent data types with explicit casting
            end_segment_values = (segment_lengths[backward_indices, end_indices[backward_indices]] * \
                                (1.0 - end_t[backward_indices])).to(dtype=dtype)
            end_segment_contrib.index_put_((backward_indices,), end_segment_values)
            
            # Sum all intermediate segment lengths
            intermediate_segments_sum = torch.sum(segment_lengths * between_segments_mask.float(), dim=1)
            
            # Compute total path distance for backward case
            all_backward_distances = start_segment_contrib + intermediate_segments_sum + end_segment_contrib
            
            # Update path distances for backward case
            path_distances = torch.where(backward_mask, all_backward_distances, path_distances)
    
    # Handle invalid cases
    invalid_mask = (start_indices < 0) | (end_indices < 0)
    path_distances = torch.where(invalid_mask, torch.tensor(float('inf'), device=device, dtype=dtype), path_distances)
    
    return path_distances


def guidelines_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_std: float = 8.0,
    path_centering_std: float = 0.6,
    centering_std: float = 0.3,
    distance_scale: float = 1.0,
    centering_scale: float = 0.05,
    z_threshold: float = 1.0,  # Maximum z-distance for guide line association
) -> torch.Tensor:
    """Reward for progress along guide lines.

    Calculates reward based on:
    1. Distance of the robot to the nearest point on the guide line
    2. Progress of the robot along the guide line towards the commanded target
    3. Centering of the robot on the guide line

    Args:
        env: The environment instance
        command_name: Name of the command to follow
        asset_cfg: Asset configuration for the robot
        path_std: Standard deviation for the distance-based reward
        centering_std: Standard deviation for the centering-based reward
        distance_scale: Scaling factor for the distance-based reward
        centering_scale: Scaling factor for the centering-based reward
        z_threshold: Maximum allowed z-distance for guide line association

    Returns:
        Reward tensor. Shape: (num_envs)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # Get the rows and cols of the robots
    env_ids = torch.arange(env.num_envs, device=env.device)
    cols = terrain.terrain_types[env_ids]
    rows = terrain.terrain_levels[env_ids]
    
    # Get guide lines for current terrains
    guide_lines_tensor = terrain.terrain_generator.guide_lines
    env_guide_lines = guide_lines_tensor[rows, cols]  # Shape: (num_envs, max_points, 3)
    
    # Create a mask to identify valid guide points (not infinity)
    valid_guide_mask = ~torch.isinf(env_guide_lines).any(dim=-1)  # Shape: (num_envs, max_points)
    
    # Get robot positions
    robot_pos = robot.data.root_pos_w[:, :3]  # Shape: (num_envs, 3)
    
    # Transform command from local to world frame
    command_world_pos = _transform_command_to_world(env, command_name, asset_cfg)
    
    # Find closest points for both robot and command, including segment indices and t values
    robot_closest_points, robot_distances, robot_segment_indices, robot_t = _find_closest_points_on_segments(
        robot_pos, env_guide_lines, valid_guide_mask, z_threshold
    )
    
    command_closest_points, command_distances, command_segment_indices, command_t = _find_closest_points_on_segments(
        command_world_pos, env_guide_lines, valid_guide_mask, z_threshold
    )
    
    # Calculate path distance along guide lines from robot to command
    path_distances = _compute_path_distance(
        env_guide_lines, 
        valid_guide_mask,
        robot_segment_indices, 
        robot_t,
        command_segment_indices, 
        command_t,
        robot_pos,
        command_world_pos
    )
    
    # Check which environments have valid guide segments
    has_valid_segments = (robot_distances < float('inf')) & (command_distances < float('inf'))
    
    total_distances = robot_distances + command_distances + path_distances
    
    centering_reward = 1.0 - torch.tanh(robot_distances / centering_std)

    # Total distance reward is conditioned on centering
    total_distance_reward = (1.0 - torch.tanh(total_distances / path_std)) \
        * (1.0 - torch.tanh(robot_distances / path_centering_std))

    # Combine distance and progress rewards
    total_reward = distance_scale * total_distance_reward + centering_scale * centering_reward
    
    # Zero out reward for environments without valid guide lines
    total_reward = torch.where(
        has_valid_segments,
        total_reward,
        torch.zeros_like(total_reward)
    )
    
    return total_reward


