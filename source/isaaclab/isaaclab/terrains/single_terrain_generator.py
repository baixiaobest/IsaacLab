from .single_terrain_generator_cfg import SingleTerrainGeneratorCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG
from .height_field.hf_terrains import mountain_terrain
from .height_field.utils import custom_perlin_noise
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from .height_field.hf_terrains_cfg import HfMountainTerrainCfg
from .terrain_generator_cfg import SubTerrainBaseCfg
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

        heights = None
        if isinstance(cfg.terrain_config, HfMountainTerrainCfg):
            terrain_meshes, origin, heights = mountain_terrain(0, self.cfg.terrain_config)
        elif isinstance(cfg.terrain_config, SubTerrainBaseCfg):
            terrain_meshes, origins = cfg.terrain_config.function(0, self.cfg.terrain_config)
        else:
            raise ValueError(f"Unsupported terrain configuration type: {type(cfg.terrain_config)}")
        
        self.terrain_meshes += terrain_meshes

        if cfg.obstacles_generator_config is not None:
            obstacle_meshes = self._generate_obstacles(heights, cfg.obstacles_generator_config)
            self.terrain_meshes += obstacle_meshes

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

    def _compute_goal_locations(self, heights: np.ndarray | None = None) -> np.ndarray:
        """Compute goal locations in a grid pattern across the terrain.
        
        Args:
            heights: Optional height field. If provided, goal z-values will use terrain heights.
                    If None, goal z-values will be zero.
        
        Returns:
            np.ndarray: Goal locations with shape (num_goals, 3)
        """
        # Always work with metric coordinates first
        terrain_length = self.cfg.terrain_config.size[0]  # x dimension
        terrain_width = self.cfg.terrain_config.size[1]   # y dimension
        
        goal_region_length = self.cfg.goal_grid_area_size[0]
        goal_region_width = self.cfg.goal_grid_area_size[1]
        
        # Calculate borders (how much space to leave around the edges)
        goal_region_length_border = (terrain_length - goal_region_length) / 2
        goal_region_width_border = (terrain_width - goal_region_width) / 2
        
        # Calculate step size between goals
        row_step = goal_region_length / self.cfg.goal_num_rows
        col_step = goal_region_width / self.cfg.goal_num_cols
        
        # Create goal coordinates in metric space
        row = np.arange(goal_region_length_border + row_step / 2,
                        terrain_length - goal_region_length_border,
                        row_step)
        col = np.arange(goal_region_width_border + col_step / 2,
                        terrain_width - goal_region_width_border,
                        col_step)
        
        # Create grid of coordinates
        cc, rr = np.meshgrid(col, row)
        
        # Initialize goal heights to zero
        goal_heights = np.zeros_like(rr)
        
        # If heights are provided, sample from heightmap
        if heights is not None:
            # Get pixel dimensions
            terrain_length_pixels = heights.shape[0]
            terrain_width_pixels = heights.shape[1]
            
            # Calculate pixels per meter
            pixels_per_meter_row = terrain_length_pixels / terrain_length
            pixels_per_meter_col = terrain_width_pixels / terrain_width
            
            # Convert metric coordinates to pixel indices
            row_pixels = np.round(rr * pixels_per_meter_row).astype(int)
            col_pixels = np.round(cc * pixels_per_meter_col).astype(int)
            
            # Clip to valid pixel range
            row_pixels = np.clip(row_pixels, 0, terrain_length_pixels - 1)
            col_pixels = np.clip(col_pixels, 0, terrain_width_pixels - 1)
            
            # Sample heights at these pixel locations
            goal_heights = heights[row_pixels, col_pixels] * self.cfg.terrain_config.vertical_scale
        
        # Combine coordinates with heights
        goal_locations = np.stack((rr, cc, goal_heights), axis=-1)
        
        return goal_locations
    
    def _compute_terrain_origins(self, heights: np.ndarray, goal_locations: np.ndarray) -> np.ndarray:
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
        if heights is not None:
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
                    if heights is not None:
                        pixel_col = int(new_x * pixels_per_meter_col)
                        pixel_row = int(new_y * pixels_per_meter_row)
                        
                        # Ensure pixel coordinates are within bounds
                        pixel_col = np.clip(pixel_col, 0, terrain_width_pixels - 1)
                        pixel_row = np.clip(pixel_row, 0, terrain_length_pixels - 1)
                        
                        # Get height at this location
                        height = heights[pixel_row, pixel_col] * self.cfg.terrain_config.vertical_scale
                    else:
                        height = 0.0
                    
                    # Calculate the index in the flattened array
                    flat_idx = goal_idx * self.cfg.origins_per_level + origin_idx
                    
                    # Store the origin
                    terrain_origins[level, flat_idx] = [new_y, new_x, height]
        
        return terrain_origins
    
    def _generate_obstacles(self, heights: np.ndarray, obstacle_cfg: SingleTerrainGeneratorCfg.ObstaclesGeneratorConfig) -> list[trimesh.Trimesh]:
        """Generate obstacle meshes based on perlin noise map and obstacle configuration."""
        num_rows, num_cols = heights.shape
        noise = custom_perlin_noise(
            num_cols, num_rows, obstacle_cfg.scale, obstacle_cfg.amplitudes, obstacle_cfg.lacunarity, 
            obstacle_cfg.seed)
        noise = np.where(noise < obstacle_cfg.threshold, 0, noise)
        indices = np.argwhere(noise > 0)
        
        # Convert pixel resolution for height mapping
        pixels_per_meter_col = num_cols / self.cfg.terrain_config.size[0]
        pixels_per_meter_row = num_rows / self.cfg.terrain_config.size[1]
        
        obstacle_meshes = []
        
        # Randomly select obstacle types for each position
        num_obstacles = len(indices)
        if num_obstacles > 0:
            # Choose random obstacle types from the available types
            obstacle_types = np.random.choice(
                obstacle_cfg.obstacles_types,
                size=num_obstacles
            )
            
            # Pre-draw random sizes for all obstacles
            size_low, size_high = obstacle_cfg.size_range
            random_sizes = {
                'cube_size_x': np.random.uniform(size_low, size_high, num_obstacles),
                'cube_size_y': np.random.uniform(size_low, size_high, num_obstacles),
                'cube_size_z': np.random.uniform(size_low, size_high, num_obstacles),
                'cylinder_radius': np.random.uniform(size_low/2, size_high/2, num_obstacles),
                'cylinder_height': np.random.uniform(size_low, size_high, num_obstacles),
                'sphere_radius': np.random.uniform(size_low/2, size_high/2, num_obstacles)
            }
            
            # Pre-draw random orientations (rotations around each axis)
            random_rotations = {
                'roll': np.random.uniform(-np.pi, np.pi, num_obstacles),  # rotation around x-axis
                'pitch': np.random.uniform(-np.pi, np.pi, num_obstacles), # rotation around y-axis
                'yaw': np.random.uniform(-np.pi, np.pi, num_obstacles)    # rotation around z-axis
            }
            
            for i, (row, col) in enumerate(indices):
                # Get height at this position
                height = heights[row, col] * self.cfg.terrain_config.vertical_scale
                
                # Convert to world coordinates
                col = col / pixels_per_meter_col
                row = row / pixels_per_meter_row
                
                # Get selected obstacle type
                obstacle_type = obstacle_types[i]
                
                # Create obstacle-specific size dictionary
                obstacle_sizes = {
                    'cube_size_x': random_sizes['cube_size_x'][i],
                    'cube_size_y': random_sizes['cube_size_y'][i],
                    'cube_size_z': random_sizes['cube_size_z'][i],
                    'cylinder_radius': random_sizes['cylinder_radius'][i],
                    'cylinder_height': random_sizes['cylinder_height'][i],
                    'sphere_radius': random_sizes['sphere_radius'][i]
                }
                
                # Create obstacle-specific rotation dictionary
                obstacle_rotations = {
                    'roll': random_rotations['roll'][i],
                    'pitch': random_rotations['pitch'][i],
                    'yaw': random_rotations['yaw'][i]
                }
                
                # Create trimesh for this obstacle
                obstacle_mesh = self._create_obstacle_mesh(
                    obstacle_type, row, col, height, obstacle_cfg, 
                    obstacle_sizes, obstacle_rotations
                )
                if obstacle_mesh is not None:
                    obstacle_meshes.append(obstacle_mesh)
    
        return obstacle_meshes

    def _create_obstacle_mesh(self, obstacle_type: str, row: int, col: int, height: int, 
            obstacle_cfg: SingleTerrainGeneratorCfg.ObstaclesGeneratorConfig, obstacle_sizes: dict=None, 
            obstacle_rotations: dict=None) ->trimesh.Trimesh:
        """Create a trimesh for an obstacle of the given type at the specified position with orientation."""
        # Import utility functions from math module
        from isaaclab.utils.math import matrix_from_euler
    
        # Base position (bottom center of the obstacle)
        position = [row, col, height]
        
        # Use provided sizes or get from config
        if obstacle_sizes is None:
            obstacle_sizes = {}
        
        # Use provided rotations or use defaults (no rotation)
        if obstacle_rotations is None:
            obstacle_rotations = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        
        # Get Euler angles for rotation
        roll = obstacle_rotations.get('roll', 0.0)
        pitch = obstacle_rotations.get('pitch', 0.0)
        yaw = obstacle_rotations.get('yaw', 0.0)
        
        # Create rotation matrix from Euler angles using utility function
        euler_angles = torch.tensor([roll, pitch, yaw], dtype=torch.float)
        R = matrix_from_euler(euler_angles, convention="XYZ").numpy()
        
        if obstacle_type == "cube":
            # Get cube dimensions from sizes, config or use defaults
            size_x = obstacle_sizes.get('cube_size_x', getattr(obstacle_cfg, 'cube_size_x', 1.0))
            size_y = obstacle_sizes.get('cube_size_y', getattr(obstacle_cfg, 'cube_size_y', 1.0))
            size_z = obstacle_sizes.get('cube_size_z', getattr(obstacle_cfg, 'cube_size_z', 1.0))
            
            # Create cube mesh
            cube = trimesh.creation.box(extents=[size_x, size_y, size_z])
            
            # Create transformation matrix with both rotation and translation
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[0:3, 3] = [position[0], position[1], position[2] + size_z/2]
            
            cube.apply_transform(transform)
            return cube
        
        elif obstacle_type == "cylinder":
            # Get cylinder dimensions from sizes, config or use defaults
            radius = obstacle_sizes.get('cylinder_radius', getattr(obstacle_cfg, 'cylinder_radius', 0.5))
            height_cyl = obstacle_sizes.get('cylinder_height', getattr(obstacle_cfg, 'cylinder_height', 1.0))
            
            # Create cylinder mesh
            cylinder = trimesh.creation.cylinder(radius=radius, height=height_cyl)
            
            # For cylinders, we only allow rotation around Z-axis to keep them upright
            if obstacle_type == "cylinder":
                # Cylinders should only rotate around Z-axis
                R_cylinder = matrix_from_euler(torch.tensor([0.0, 0.0, yaw]), convention="XYZ").numpy()
                transform = np.eye(4)
                transform[:3, :3] = R_cylinder
            else:
                transform = np.eye(4)
                transform[:3, :3] = R
            
            transform[0:3, 3] = [position[0], position[1], position[2] + height_cyl/2]
            
            cylinder.apply_transform(transform)
            return cylinder
        
        elif obstacle_type == "sphere":
            # Get sphere dimensions from sizes, config or use defaults
            radius = obstacle_sizes.get('sphere_radius', getattr(obstacle_cfg, 'sphere_radius', 0.5))
            
            # Create sphere mesh
            sphere = trimesh.creation.icosphere(radius=radius)
            
            # For spheres, rotation doesn't matter, just apply translation
            transform = np.eye(4)
            transform[0:3, 3] = [position[0], position[1], position[2] + radius]
            
            sphere.apply_transform(transform)
            return sphere
        
        # Return None for unknown obstacle types
        return None

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