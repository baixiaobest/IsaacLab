# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, field
from typing import List, Tuple, Optional

from isaaclab.utils import configclass

@configclass
class PrimitiveCfg:
    """Base configuration for a primitive shape."""
    position: List[float] = MISSING  # [x, y, z]
    
@configclass
class CubeCfg(PrimitiveCfg):
    """Configuration for a cube primitive."""
    dimensions: List[float] = MISSING  # [width, depth, height]
    
@configclass
class CylinderCfg(PrimitiveCfg):
    """Configuration for a cylinder primitive."""
    radius: float = MISSING
    height: float = MISSING
    
@configclass
class SphereCfg(PrimitiveCfg):
    """Configuration for a sphere primitive."""
    radius: float = MISSING

@configclass
class SubTerrainTestCfg:
    """Configuration for a manually defined subterrain."""
    size: Tuple[float, float] = (10.0, 10.0)
    """The width (along x) and length (along y) of the sub-terrain ground plane (in m).
    
    This defines the size of the ground plane for this specific sub-terrain that will be placed
    under any primitive objects. Each sub-terrain can have its own ground plane size.
    """
    cubes: List[CubeCfg] = field(default_factory=list)
    cylinders: List[CylinderCfg] = field(default_factory=list)
    spheres: List[SphereCfg] = field(default_factory=list)
    start_position: Optional[List[float]] = None  # [x, y, z]
    goal_position: Optional[List[float]] = None   # [x, y, z]
    ground_height: float = 0.0  # Height of the ground plane

@configclass
class TestTerrainGeneratorCfg:
    """Configuration for the test terrain generator."""
    size: Tuple[float, float] = MISSING
    """The width (along x) and length (along y) of each sub-terrain cell (in m).
    
    This defines the overall grid cell size used to position each sub-terrain. The actual ground plane
    size for each sub-terrain is defined individually in the SubTerrainTestCfg.size parameter.
    """
    subterrain_spacing: float = 1.0
    """The spacing between adjacent sub-terrains (in m).
    
    This defines the empty space between the edges of adjacent sub-terrains, not the center-to-center distance.
    A value of 0.0 means sub-terrains will be placed right next to each other.
    """
    num_rows: int = 1
    """Number of rows of sub-terrains to generate."""
    num_cols: int = 1
    """Number of columns of sub-terrains to generate."""
    sub_terrains: List[SubTerrainTestCfg] = MISSING
    """List of sub-terrain configurations to place in the grid."""