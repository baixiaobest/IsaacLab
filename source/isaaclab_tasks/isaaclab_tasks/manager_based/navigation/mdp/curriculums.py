import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs import ManagerBasedRLEnv


def pose_2d_command_terrain_curriculum(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        distance_threshold: float = 0.5,
        angular_threshold: float = 0.1):
    """ When pose 2d command is within threshold, goal is considered reached. Then the terrain level is increased."""
    command = env.command_manager.get_command(command_name)[env_ids]
    within_distance = torch.norm(command[:, :3], dim=1) <= distance_threshold
    within_angular_distance = torch.abs(command[:, 3]) <= angular_threshold

    move_up = torch.logical_and(within_distance, within_angular_distance)
    move_down = ~move_up

    # update terrain levels
    terrain: TerrainImporter = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean terrain level
    return {"mean": torch.mean(terrain.terrain_levels.float()), "max": torch.max(terrain.terrain_levels.float())}

def pose_2d_command_terrain_curriculum_with_threshold(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        min_level_thresholds: int,
        max_level_thresholds: int,
        distance_threshold: float = 0.5,
        angular_threshold: float = 0.1
):
    """When the level is beyond a threshold, the level cannot be decreased."""
    command = env.command_manager.get_command(command_name)[env_ids]
    within_distance = torch.norm(command[:, :3], dim=1) <= distance_threshold
    within_angular_distance = torch.abs(command[:, 3]) <= angular_threshold

    # update terrain levels
    terrain: TerrainImporter = env.scene.terrain

    can_move_up = terrain.terrain_levels[env_ids] < max_level_thresholds

    move_up = torch.logical_and(within_distance, within_angular_distance)
    move_up = torch.logical_and(move_up, can_move_up)
    move_down = ~move_up

    # only allow decrease if current level is below threshold
    can_move_down = terrain.terrain_levels[env_ids] < min_level_thresholds
    move_down = torch.logical_and(move_down, can_move_down)

    terrain.update_env_origins(env_ids, move_up, move_down)

    # return the mean terrain level
    return {"mean": torch.mean(terrain.terrain_levels.float()), "max": torch.max(terrain.terrain_levels.float())}


def pedestrian_crowd_curriculum(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        max_level: int,
        count_range_low: tuple[int, int],
        count_range_high: tuple[int, int],
        speed_range_low: tuple[float, float],
        speed_range_high: tuple[float, float],
):
    """Ramp the active pedestrian count and preferred-speed range with terrain level.

    For each env in ``env_ids``, linearly interpolates ``count_range``/``speed_range``
    between the "low" (terrain level 0) and "high" (terrain level == ``max_level``)
    settings according to ``env.scene.terrain.terrain_levels[env_id] / max_level``, then
    forwards the result to ``env.crowd_manager`` (a :class:`SocialForceCrowdManager`).

    This term must be declared AFTER ``terrain_levels`` (``pose_2d_command_terrain_curriculum``)
    in the curriculum config so ``terrain.terrain_levels`` reflects this episode's update
    before being read here.

    ``env_ids`` is filtered down to ``env.is_pedestrian_env`` envs — a no-op for envs sitting on
    a static (non-"ped_corridor") terrain column.
    """
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    env_ids_t = env_ids_t[env.is_pedestrian_env[env_ids_t]]
    if len(env_ids_t) == 0:
        return {"mean_active": torch.tensor(0.0), "mean_speed": torch.tensor(0.0)}

    terrain: TerrainImporter = env.scene.terrain
    levels = terrain.terrain_levels[env_ids_t].float()
    t = (levels / max(max_level, 1)).clamp(0.0, 1.0)

    count_min = count_range_low[0] + t * (count_range_high[0] - count_range_low[0])
    count_max = count_range_low[1] + t * (count_range_high[1] - count_range_low[1])
    num_active = (count_min + torch.rand_like(t) * (count_max - count_min)).round().long()

    speed_min = speed_range_low[0] + t * (speed_range_high[0] - speed_range_low[0])
    speed_max = speed_range_low[1] + t * (speed_range_high[1] - speed_range_low[1])
    speed_range = torch.stack([speed_min, speed_max], dim=-1)

    env.crowd_manager.set_active_count(env_ids_t, num_active)
    env.crowd_manager.set_speed_range(env_ids_t, speed_range)

    return {"mean_active": num_active.float().mean(), "mean_speed": speed_max.mean()}


def terrain_level_contact_penalty_curriculum(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        max_level: int,
        reward_term_name: str = "undesired_contacts",
        weight_initial: float = -200.0,
        weight_final: float = -1000.0,
):
    """Scales a contact-penalty reward weight linearly from weight_initial to weight_final
    as the mean terrain level rises from 0 to max_level."""
    terrain: TerrainImporter = env.scene.terrain
    mean_level = torch.mean(terrain.terrain_levels.float()).item()

    t = min(mean_level / max(max_level, 1), 1.0)
    new_weight = weight_initial + t * (weight_final - weight_initial)

    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    term_cfg.weight = new_weight

    return {"weight": new_weight}
