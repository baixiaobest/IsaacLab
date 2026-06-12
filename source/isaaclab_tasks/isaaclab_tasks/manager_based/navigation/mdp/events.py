from __future__ import annotations
import inspect
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class activate_event_terrain_level_reached(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.activated = False
        self.matched_env_ids = None
    
    def __call__(self,
                env: ManagerBasedEnv,
                env_ids: torch.Tensor,
                func: callable,
                terrain_names: list[str],
                operator: str = "max", # max or mean
                terrain_level_threshold: float = 0.0,
                callback_params: dict = {}
    ):
        
        terrain: TerrainImporter = env.scene.terrain
        if self.matched_env_ids is None:
            env_terrain_names = terrain.get_env_terrain_names()
            self.matched_env_ids = torch.tensor([i for i, name in enumerate(env_terrain_names) if name in terrain_names], dtype=torch.int, device=env.device) 

        if self.matched_env_ids.size() == 0:
            raise ValueError("No environments match the specified terrain names.")
        
        matched_env_terrain_levels = terrain.terrain_levels[self.matched_env_ids]

        level = 0
        if operator == "max":
            level = torch.max(matched_env_terrain_levels.float()).item()
        elif operator == "mean":
            level = torch.mean(matched_env_terrain_levels.float()).item()
        else:
            raise ValueError("operator must be 'max' or 'mean'")

        if not self.activated and level >= terrain_level_threshold:
            self.activated = True
        
        if self.activated:
            func(env, env_ids, **callback_params)
        else:
            return


def reset_pedestrian_crowd(env: ManagerBasedEnv, env_ids: torch.Tensor, flow_dir: float = 1.0):
    """(Re)spawn the social-force pedestrian crowd for ``env_ids``.

    Derives the per-env corridor geometry (origin, length, width) from the env's current
    sub-terrain (``terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]`` plus
    the terrain generator's ``size``), then calls ``env.crowd_manager.reset_idx`` — preserving
    the active pedestrian count and speed range last set by :func:`pedestrian_crowd_curriculum`.

    Must be declared after ``reset_base`` (mode="reset") so ``terrain_origins``/``terrain_levels``
    reflect this episode's terrain assignment before being read here.
    """
    terrain: TerrainImporter = env.scene.terrain

    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    corridor_origin = terrain.terrain_origins[levels, types][:, :2]

    size = terrain.cfg.terrain_generator.size
    corridor_length = torch.full((len(env_ids),), size[0], device=env.device)
    corridor_width = torch.full((len(env_ids),), size[1], device=env.device)
    flow_dir_t = torch.full((len(env_ids),), flow_dir, device=env.device)

    crowd_manager = env.crowd_manager
    num_active = crowd_manager.active_mask[env_ids].sum(dim=1)
    speed_range = crowd_manager._speed_range[env_ids]

    crowd_manager.reset_idx(env_ids, corridor_origin, flow_dir_t, corridor_length, corridor_width, num_active, speed_range)