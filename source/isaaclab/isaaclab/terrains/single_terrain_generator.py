from .single_terrain_generator_cfg import SingleTerrainGeneratorCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG
from .height_field.hf_terrains import mountain_terrain
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import trimesh
import numpy as np
import torch

class SingleTerrainGenerator:

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: torch.Tensor
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    goal_locations: torch.Tensor
    """The locations of the goals in the terrain. Shape is (num_goals, 3)."""

    def __init__(self, cfg: SingleTerrainGeneratorCfg, device="cpu"):
        self.cfg = cfg
        self.device = device

        self.terrain_meshes = list()

        mountain_meshes, origin, heights = mountain_terrain(0, self.cfg.terrain_config)
        
        self.terrain_meshes += mountain_meshes

        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        goal_locations_np = \
            self._compute_goal_locations(heights)
        goal_locations_np = goal_locations_np.reshape(-1, 3)

        terrain_origins_np = self._compute_terrain_origins(heights, goal_locations_np)

        # Centering the terrain
        shift = np.array([self.cfg.terrain_config.size[0] * 0.5, 
                  self.cfg.terrain_config.size[1] * 0.5, 
                  0.0])
        
        terrain_origins_np -= shift
        goal_locations_np -= shift

        self.terrain_origins = torch.tensor(terrain_origins_np, dtype=torch.float32, device=self.device)
        self.goal_locations = torch.tensor(goal_locations_np, dtype=torch.float32, device=self.device)

        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.terrain_config.size[0] * 0.5, -self.cfg.terrain_config.size[1] * 0.5
        self.terrain_mesh.apply_transform(transform)

        self._visualize_goals()

    def _compute_goal_locations(self, heights):
        """Compute the goal locations based on the terrain size."""
        terrain_length_pixels = heights.shape[0]
        terrain_width_pixels = heights.shape[1]

        goal_region_length_pixels = int(self.cfg.goal_grid_area_size[1] / self.cfg.terrain_config.size[1] * terrain_length_pixels)
        goal_region_width_pixels = int(self.cfg.goal_grid_area_size[0] / self.cfg.terrain_config.size[0] * terrain_width_pixels)
        
        row_step = int(goal_region_length_pixels / self.cfg.goal_num_rows)
        col_step = int(goal_region_width_pixels / self.cfg.goal_num_cols)

        goal_region_length_border = int((terrain_length_pixels - goal_region_length_pixels) / 2)
        goal_region_width_border = int((terrain_width_pixels - goal_region_width_pixels) / 2)

        row = np.arange(goal_region_length_border + int(row_step / 2), 
                        terrain_length_pixels - goal_region_length_border, 
                        row_step)
        
        col = np.arange(goal_region_width_border + int(col_step / 2), 
                        terrain_width_pixels - goal_region_width_border, 
                        col_step)
        
        cc, rr = np.meshgrid(col, row)
        goal_heights = heights[rr, cc] * self.cfg.terrain_config.vertical_scale

        rr_metrics = rr * self.cfg.terrain_config.horizontal_scale
        cc_metrics = cc * self.cfg.terrain_config.horizontal_scale
        goal_locations = np.stack((rr_metrics, cc_metrics, goal_heights), axis=-1)

        return goal_locations
    
    def _compute_terrain_origins(self, heights, goal_locations):
        """
        Compute terrain origins for each goal location across multiple terrain levels.
        For each level and goal, creates origins_per_level origins at evenly spaced angles.
        
        Args:
            heights: Heightmap array
        
        Returns:
            Terrain origins with shape (total_terrain_levels, num_goals * origins_per_level, 3)
        """
        num_goals = len(goal_locations)
        total_origins_per_level = num_goals * self.cfg.origins_per_level
        terrain_origins = np.zeros((self.cfg.total_terrain_levels, total_origins_per_level, 3))
        
        # Calculate pixel resolution for height mapping
        terrain_length_pixels = heights.shape[0]
        terrain_width_pixels = heights.shape[1]
        pixels_per_meter_col = terrain_width_pixels / self.cfg.terrain_config.size[0]
        pixels_per_meter_row = terrain_length_pixels / self.cfg.terrain_config.size[1]
        
        # Calculate angle step for evenly distributing origins
        angle_step = 2 * np.pi / self.cfg.origins_per_level
        
        for level in range(self.cfg.total_terrain_levels):
            # Calculate distance for this level
            distance = (level + 1) * self.cfg.distance_increment_per_level
            
            for goal_idx, goal in enumerate(goal_locations):
                for origin_idx in range(self.cfg.origins_per_level):
                    # Get evenly distributed angle
                    angle = origin_idx * angle_step
                    
                    # Calculate new x,y position
                    dx = distance * np.cos(angle)
                    dy = distance * np.sin(angle)
                    
                    new_x = goal[1] + dx
                    new_y = goal[0] + dy
                    
                    # Ensure we're within terrain boundaries
                    new_x = np.clip(new_x, 0, self.cfg.terrain_config.size[0] - 1e-6)
                    new_y = np.clip(new_y, 0, self.cfg.terrain_config.size[1] - 1e-6)
                    
                    # Convert to pixel coordinates to get height
                    pixel_col = int(new_x * pixels_per_meter_col)
                    pixel_row = int(new_y * pixels_per_meter_row)
                    
                    # Ensure pixel coordinates are within bounds
                    pixel_col = np.clip(pixel_col, 0, terrain_width_pixels - 1)
                    pixel_row = np.clip(pixel_row, 0, terrain_length_pixels - 1)
                    
                    # Get height at this location
                    height = heights[pixel_row, pixel_col] * self.cfg.terrain_config.vertical_scale
                    
                    # Calculate the index in the flattened array
                    flat_idx = goal_idx * self.cfg.origins_per_level + origin_idx
                    
                    # Store the origin
                    terrain_origins[level, flat_idx] = [new_y, new_x, height]
        
        return terrain_origins
    
    def _visualize_goals(self):
        self.goal_markers = []

        for idx, goal in enumerate(self.goal_locations):
            goal_marker_cfg = VisualizationMarkersCfg(
                prim_path=f"/Visuals/Goal/goal_{idx}",
                markers={
                    "target": sim_utils.SphereCfg(
                                radius=0.5,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(np.random.rand(3))))
                }
            )
            goal_marker= VisualizationMarkers(goal_marker_cfg)
            goal_marker.set_visibility(True)
            goal_marker.visualize(translations=goal.unsqueeze(dim=0), scales=np.array([[1, 1, 1]]))
            self.goal_markers.append(goal_marker)