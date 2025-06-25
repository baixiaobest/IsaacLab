from isaaclab.utils import configclass
from dataclasses import MISSING
from .height_field.hf_terrains_cfg import HfMountainTerrainCfg

@configclass
class SingleTerrainGeneratorCfg:
    """Configuration for the single terrain generator."""

    terrain_config: HfMountainTerrainCfg = MISSING
    """The terrain configuration to use for generating the terrain."""

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the terrain in meters (width, length)."""

    goal_num_rows: int = 5
    """Number of rows for goal locations. Defaults to 5."""

    goal_num_cols: int = 10
    """Number of columns for goal locations. Defaults to 5."""

    goal_grid_area_size: tuple[float, float] = (75.0, 75.0)
    """A grid of goals defined by goal_num_rows and goal_num_cols will be place in an area of this size (width, length) in meters."""

    total_terrain_levels: int = 10
    """Total number of terrain levels to generate. Defaults to 10."""

    distance_increment_per_level: float = 10.0
    """Distance increment per level in meters. Distance to goal will increase with levels. Defaults to 5.0."""

    origins_per_level: int = 4
    """Number of origins per goal."""