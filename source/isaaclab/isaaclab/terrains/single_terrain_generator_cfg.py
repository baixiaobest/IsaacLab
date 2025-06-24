from isaaclab.utils import configclass
from dataclasses import MISSING
from .height_field.hf_terrains_cfg import HfMountainTerrainCfg

@configclass
class SingleTerrainGeneratorCfg:
    """Configuration for the single terrain generator."""

    terrain_config: type = HfMountainTerrainCfg
    """The terrain configuration to use for generating the terrain."""

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the terrain in meters (width, length)."""