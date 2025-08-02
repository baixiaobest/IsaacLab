# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import trimesh
from typing import List, Optional

from isaaclab.terrains.test_terrain_generator_cfg import TestTerrainGeneratorCfg, SubTerrainTestCfg
from isaaclab.terrains.trimesh.utils import make_plane

class TestTerrainGenerator:
    """Test terrain generator that allows manual definition of subterrains with primitives.
    
    This is a simplified version of TerrainGenerator that allows users to manually define
    each subterrain with primitive shapes like cubes, cylinders, and spheres, as well as
    specify start and goal positions.
    """
    
    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: List[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    start_positions: np.ndarray
    """Start positions for each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    goal_positions: np.ndarray
    """Goal positions for each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    
    def __init__(self, cfg: TestTerrainGeneratorCfg, device: str = "cpu"):
        """Initialize the test terrain generator.
        
        Args:
            cfg: Configuration for the test terrain generator.
            device: The device to use for tensors.
        """
        self.cfg = cfg
        self.device = device
        
        # Validate that we have enough subterrains
        if len(cfg.sub_terrains) < cfg.num_rows * cfg.num_cols:
            raise ValueError(
                f"Not enough subterrains provided. Need {cfg.num_rows * cfg.num_cols}, got {len(cfg.sub_terrains)}."
            )
            
        # Initialize arrays for tracking positions
        self.terrain_meshes = []
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.start_positions = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goal_positions = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        
        # Create a single large ground plane first
        self._create_ground_plane()
        
        # Generate all the subterrains on top of the ground plane
        self._generate_terrains()
        
        # Combine all the meshes into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)
        
        # No need for additional centering since we already center the terrain during generation
    
    def _create_ground_plane(self):
        """Create a single large ground plane for all subterrains."""
        # Calculate the total size of the ground plane
        # Account for spacing between subterrains
        total_width = (self.cfg.num_rows - 1) * self.cfg.subterrain_spacing + sum(self.cfg.sub_terrains[row * self.cfg.num_cols].size[0] 
                                                                            for row in range(self.cfg.num_rows))
        total_length = (self.cfg.num_cols - 1) * self.cfg.subterrain_spacing + sum(self.cfg.sub_terrains[col].size[1] 
                                                                             for col in range(self.cfg.num_cols))
        
        total_width = max(total_width, self.cfg.size[0])
        total_length = max(total_length, self.cfg.size[1])

        # Create the ground plane mesh
        ground_plane = make_plane(
            size=(total_width, total_length),
            height=0.0,  # Default ground height
            center_zero=True  # Center at origin
        )
        
        # Add the ground plane to our list of meshes
        self.terrain_meshes.append(ground_plane)
        
    def _generate_terrains(self):
        """Generate all the subterrains on top of the ground plane."""
        # Calculate the starting offsets to center all subterrains
        total_width = (self.cfg.num_rows - 1) * self.cfg.subterrain_spacing + sum(self.cfg.sub_terrains[row * self.cfg.num_cols].size[0] 
                                                                            for row in range(self.cfg.num_rows))
        total_length = (self.cfg.num_cols - 1) * self.cfg.subterrain_spacing + sum(self.cfg.sub_terrains[col].size[1] 
                                                                             for col in range(self.cfg.num_cols))
        
        start_x = -total_width / 2
        start_y = -total_length / 2
        
        # Keep track of the current position
        current_x = start_x
        
        for row in range(self.cfg.num_rows):
            current_y = start_y
            row_height = 0  # Track the tallest subterrain in this row
            
            for col in range(self.cfg.num_cols):
                index = row * self.cfg.num_cols + col
                sub_terrain_cfg = self.cfg.sub_terrains[index]
                
                # Generate the primitive objects for this subterrain
                mesh_list, origin = self._generate_subterrain_primitives(sub_terrain_cfg)
                
                # Position this subterrain at the current position
                position = [current_x, current_y, 0]
                
                # Add the subterrain meshes to our list
                self._add_sub_terrain(mesh_list, origin, position, row, col, sub_terrain_cfg)
                
                # Update y position for next subterrain in this row
                current_y += sub_terrain_cfg.size[1] + self.cfg.subterrain_spacing
                
                # Update row height if this subterrain is taller
                row_height = max(row_height, sub_terrain_cfg.size[0])
            
            # Move to the next row
            current_x += row_height + self.cfg.subterrain_spacing

    def _generate_subterrain_primitives(self, cfg: SubTerrainTestCfg) -> tuple[List[trimesh.Trimesh], np.ndarray]:
        """Generate only the primitive objects for a subterrain.
        
        Args:
            cfg: The configuration for the subterrain.
            
        Returns:
            A tuple containing the primitive meshes and the origin.
        """
        meshes = []
        
        # Add cubes if specified
        if cfg.cubes:
            for cube in cfg.cubes:
                # Create the cube mesh
                cube_mesh = trimesh.creation.box(
                    extents=cube.dimensions,
                    transform=trimesh.transformations.translation_matrix(cube.position)
                )
                meshes.append(cube_mesh)
        
        # Add cylinders if specified
        if cfg.cylinders:
            for cylinder in cfg.cylinders:
                # Create the cylinder mesh
                cylinder_transform = trimesh.transformations.translation_matrix(cylinder.position)
                cylinder_mesh = trimesh.creation.cylinder(
                    radius=cylinder.radius,
                    height=cylinder.height,
                    transform=cylinder_transform
                )
                meshes.append(cylinder_mesh)
        
        # Add spheres if specified
        if cfg.spheres:
            for sphere in cfg.spheres:
                # Create the sphere mesh
                sphere_transform = trimesh.transformations.translation_matrix(sphere.position)
                sphere_mesh = trimesh.creation.icosphere(
                    radius=sphere.radius,
                    transform=sphere_transform
                )
                meshes.append(sphere_mesh)
        
        # Determine the origin of the subterrain
        if cfg.start_position is not None:
            origin = np.array(cfg.start_position)
        else:
            # Use the center of the subterrain's assigned cell as the origin
            origin = np.array([cfg.size[0] * 0.5, cfg.size[1] * 0.5, 0.0])
        
        return meshes, origin
            
    def _add_sub_terrain(
        self, meshes: List[trimesh.Trimesh], origin: np.ndarray, position: List[float], 
        row: int, col: int, sub_terrain_cfg: SubTerrainTestCfg
    ):
        """Add input sub-terrain primitives to the list of meshes.
        
        Args:
            meshes: The meshes of the sub-terrain primitives.
            origin: The origin of the sub-terrain.
            position: The position where this subterrain should be placed [x, y, z].
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
            sub_terrain_cfg: The configuration of the sub-terrain.
        """
        # Transform the meshes to the correct position
        transform = np.eye(4)
        transform[0:3, -1] = position
        
        for mesh in meshes:
            # Create a copy of the mesh before transforming
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(transform)
            self.terrain_meshes.append(mesh_copy)
            
        # Add origin to the list (origin is relative to the subterrain position)
        self.terrain_origins[row, col] = origin + transform[:3, -1]
        
        # Add start and goal positions if specified
        if sub_terrain_cfg.start_position is not None:
            self.start_positions[row, col] = np.array(sub_terrain_cfg.start_position) + transform[:3, -1]
        else:
            self.start_positions[row, col] = self.terrain_origins[row, col]
            
        if sub_terrain_cfg.goal_position is not None:
            self.goal_positions[row, col] = np.array(sub_terrain_cfg.goal_position) + transform[:3, -1]
        else:
            self.goal_positions[row, col] = self.terrain_origins[row, col]
    
    def get_start_positions(self) -> np.ndarray:
        """Get the start positions of all the sub-terrains.
        
        Returns:
            The start positions of all the sub-terrains. Shape is (num_rows, num_cols, 3).
        """
        return self.start_positions
    
    def get_goal_positions(self) -> np.ndarray:
        """Get the goal positions of all the sub-terrains.
        
        Returns:
            The goal positions of all the sub-terrains. Shape is (num_rows, num_cols, 3).
        """
        return self.goal_positions
    
    def __str__(self):
        """Return a string representation of the test terrain generator."""
        msg = "Test Terrain Generator:"
        msg += f"\n\tNumber of rows: {self.cfg.num_rows}"
        msg += f"\n\tNumber of columns: {self.cfg.num_cols}"
        msg += f"\n\tSub-terrain spacing: {self.cfg.subterrain_spacing}"
        msg += f"\n\tNumber of sub-terrains: {len(self.cfg.sub_terrains)}"
        return msg