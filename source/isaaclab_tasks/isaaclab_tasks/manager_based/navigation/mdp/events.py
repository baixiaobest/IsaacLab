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