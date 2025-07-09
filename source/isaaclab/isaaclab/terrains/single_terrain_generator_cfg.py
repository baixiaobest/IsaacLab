from isaaclab.utils import configclass
from dataclasses import MISSING
from .terrain_generator_cfg import SubTerrainBaseCfg

@configclass
class SingleTerrainGeneratorCfg:
    """Configuration for the single terrain generator."""

    terrain_config: SubTerrainBaseCfg = MISSING
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

    @configclass
    class ObstaclesGeneratorConfig:
        length_pixels: int = 1000
        """The length of the perlin noise generator in pixels."""

        width_pixels: int = 1000
        """The width of the perlin noise generator in pixels."""

        scale: float = 20.0
        """The scale of the perlin noise generator."""

        amplitudes: list[float] = [0.5, 0.3, 0.5, 1.0]
        """The amplitudes of the perlin noise generator."""

        lacunarity: float = 2.0
        """The lacunarity of the perlin noise generator."""

        threshold: float = 0.87
        """The threshold for the perlin noise generator to create obstacles."""

        seed: int = 1
        """The seed for the perlin noise generator."""

        size_range = (0.1, 1.0)
        """The range of sizes for the obstacles."""

        obstacles_types: list[str] = ["cube", "cylinder", "sphere"]
        """The types of obstacles to generate."""

        goal_region_clearance: float = 3.0
        """Clearance around the goal region to not place obstacle."""

    obstacles_generator_config: ObstaclesGeneratorConfig | None = ObstaclesGeneratorConfig()