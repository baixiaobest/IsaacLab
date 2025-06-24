from .single_terrain_generator_cfg import SingleTerrainGeneratorCfg
from .trimesh.utils import make_plane
import trimesh
import numpy as np

class SingleTerrainGenerator:

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""

    def __init__(self, cfg: SingleTerrainGeneratorCfg, device="cpu"):
        self.cfg = cfg
        self.device = device

        self.terrain_meshes = list()
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        self.terrain_meshes.append(ground_plane)

        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

         # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * 0.5, -self.cfg.size[1] * 0.5
        self.terrain_mesh.apply_transform(transform)