from .single_terrain_generator_cfg import SingleTerrainGeneratorCfg
from .trimesh.utils import make_plane
from .height_field.hf_terrains import mountain_terrain
import trimesh
import numpy as np

class SingleTerrainGenerator:

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""

    def __init__(self, cfg: SingleTerrainGeneratorCfg, device="cpu"):
        self.cfg = cfg
        self.device = device

        self.terrain_meshes = list()

        mountain_meshes, origin, heights = mountain_terrain(0, self.cfg.terrain_config)
        
        self.terrain_meshes += mountain_meshes

        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        self.terrain_origins = np.zeros((1, 1, 3), dtype=np.float32)
        origin = origin - np.array([self.cfg.terrain_config.size[0] * 0.5, self.cfg.terrain_config.size[1] * 0.5, 0.0])
        self.terrain_origins[0, 0, :] = origin

        # offset the entire terrain and origins so that it is centered
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.terrain_config.size[0] * 0.5, -self.cfg.terrain_config.size[1] * 0.5
        self.terrain_mesh.apply_transform(transform)