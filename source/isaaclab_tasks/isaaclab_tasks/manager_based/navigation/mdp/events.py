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


def reset_pedestrian_scenario_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    flow_pose_range: dict[str, tuple[float, float]],
    crossing_south_pose_range: dict[str, tuple[float, float]],
    crossing_north_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    crossing_prob: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Sample the per-env pedestrian scenario and reset the robot root state accordingly.

    For each env in ``env_ids`` this draws a scenario mode and stores it in
    ``env.pedestrian_scenario_mode`` (read later by :class:`CorridorPedestrianPose2dCommand` to
    pick the matching goal):

    - ``0`` = flow: spawn from ``flow_pose_range``, goal up/downstream along local-x.
    - ``1`` = crossing south→north: spawn from ``crossing_south_pose_range`` (facing +y), goal
      across the flow on the north side.
    - ``2`` = crossing north→south: spawn from ``crossing_north_pose_range`` (facing -y), goal
      across the flow on the south side.

    ``crossing_prob`` is the probability of a crossing episode (vs. flow); a crossing episode is
    then south→north or north→south with equal probability, so the robot experiences the crowd
    sweeping across its path from both relative directions. The pedestrian crowd itself is
    scenario-independent (it always flows along local-x), so it is (re)spawned separately by
    :func:`reset_pedestrian_crowd`.

    Must be declared as a ``mode="reset"`` event so it runs before the command manager resamples
    the goal within the same reset, and ``env.pedestrian_scenario_mode`` must already exist (it
    is allocated in :class:`PedestrianCrowdNavigationEnv`).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)

    # -- sample the scenario mode (0 = flow, 1 = cross S->N, 2 = cross N->S) --
    is_crossing = torch.rand(n, device=env.device) < crossing_prob
    is_north_start = torch.rand(n, device=env.device) < 0.5  # only meaningful when crossing
    mode = torch.where(
        is_crossing,
        torch.where(is_north_start, torch.full_like(is_crossing, 2, dtype=torch.long),
                    torch.full_like(is_crossing, 1, dtype=torch.long)),
        torch.zeros_like(is_crossing, dtype=torch.long),
    )
    env.pedestrian_scenario_mode[env_ids] = mode

    root_states = asset.data.default_root_state[env_ids].clone()

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]

    def _sample(pose_range: dict[str, tuple[float, float]]) -> torch.Tensor:
        ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in keys], device=asset.device)
        return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)

    flow_samples = _sample(flow_pose_range)
    south_samples = _sample(crossing_south_pose_range)
    north_samples = _sample(crossing_north_pose_range)
    crossing_samples = torch.where(is_north_start.unsqueeze(-1), north_samples, south_samples)
    rand_samples = torch.where(is_crossing.unsqueeze(-1), crossing_samples, flow_samples)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in keys], device=asset.device)
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


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