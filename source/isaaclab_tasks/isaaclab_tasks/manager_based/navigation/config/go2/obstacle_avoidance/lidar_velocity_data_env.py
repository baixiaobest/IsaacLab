"""Fixed-coverage support and scan-time labels for LiDAR velocity collection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.terrains import TerrainImporter
from isaaclab_tasks.manager_based.navigation.mdp.events import reset_pedestrian_crowd

from .pedestrian_crowd_env import PedestrianCrowdNavigationEnv


class FixedCoverageTerrainImporter(TerrainImporter):
    """Assign every generated terrain tile equally instead of sampling rows randomly."""

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        num_rows, num_cols = origins.shape[:2]
        tiles_per_replica = num_rows * num_cols
        if num_envs <= 0 or num_envs % tiles_per_replica != 0:
            raise ValueError(
                "Fixed-coverage LiDAR velocity collection requires num_envs to be a positive multiple of "
                f"{tiles_per_replica} ({num_rows} levels x {num_cols} terrain columns), got {num_envs}."
            )

        tile_ids = torch.arange(num_envs, device=self.device) % tiles_per_replica
        self.terrain_levels = torch.div(tile_ids, num_cols, rounding_mode="floor").to(torch.long)
        self.terrain_types = (tile_ids % num_cols).to(torch.long)
        self.tile_replicas = torch.div(torch.arange(num_envs, device=self.device), tiles_per_replica, rounding_mode="floor")
        self.max_terrain_level = num_rows
        return origins[self.terrain_levels, self.terrain_types].clone()


def configure_fixed_level_pedestrian_profiles(env: PedestrianCrowdNavigationEnv, env_ids: Sequence[int] | torch.Tensor) -> None:
    """Configure per-level crowd difficulty without changing terrain levels."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.is_pedestrian_env[ids]]
    if ids.numel() == 0:
        return

    cfg = env.cfg
    level = env.scene.terrain.terrain_levels[ids].float()
    max_level = max(int(env.scene.terrain.max_terrain_level) - 1, 1)
    alpha = (level / max_level).clamp(0.0, 1.0)

    low_count = torch.tensor((2.0, 3.0), device=env.device)
    high_count = torch.tensor((10.0, 12.0), device=env.device)
    count_range = low_count + alpha.unsqueeze(-1) * (high_count - low_count)
    # Sample once per reset within the level's profile range, just as the former
    # curriculum did, while leaving the assigned terrain level untouched.
    count = (count_range[:, 0] + torch.rand_like(alpha) * (count_range[:, 1] - count_range[:, 0])).round().long()

    low_speed = torch.tensor((0.3, 0.7), device=env.device)
    high_speed = torch.tensor((0.9, 1.5), device=env.device)
    speed_range = low_speed + alpha.unsqueeze(-1) * (high_speed - low_speed)
    heading_max = alpha * math.radians(12.0)

    env.crowd_manager.set_active_count(ids, count)
    env.crowd_manager.set_speed_range(ids, speed_range)
    env.crowd_manager.set_lateral_heading_max(ids, heading_max)


def reset_fixed_level_pedestrian_crowd(
    env: PedestrianCrowdNavigationEnv, env_ids: Sequence[int] | torch.Tensor, flow_dir: float = 1.0
) -> None:
    """Reset crowd state after applying its fixed terrain-level profile."""
    configure_fixed_level_pedestrian_profiles(env, env_ids)
    reset_pedestrian_crowd(env, env_ids, flow_dir=flow_dir)


class FixedCoveragePedestrianCrowdNavigationEnv(PedestrianCrowdNavigationEnv):
    """Crowd environment exposing scan-time per-bin velocity labels to rollout tools."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        env_ids = torch.arange(self.num_envs, device=self.device)
        configure_fixed_level_pedestrian_profiles(self, env_ids)
        reset_pedestrian_crowd(self, env_ids, flow_dir=cfg.pedestrian_flow_dir)
        self._write_pedestrians_to_sim()
        if self._held_scan_lidar_collector is not None:
            self._held_scan_lidar_collector.reset(env_ids)
        self._validate_velocity_label_scene()

    def _validate_velocity_label_scene(self) -> None:
        sensor = self.scene.sensors["obstacle_scanner"]
        if getattr(sensor.cfg, "update_mesh_ids", False) is not True:
            raise RuntimeError("LiDAR velocity data collection requires obstacle_scanner.update_mesh_ids=True.")
        target_counts = sensor._num_meshes_per_env
        if len(target_counts) != 2:
            raise RuntimeError("Expected exactly terrain and pedestrian raycast targets for velocity labels.")
        counts = list(target_counts.values())
        if counts[0] != 1 or counts[1] != self.crowd_manager.max_pedestrians:
            raise RuntimeError(
                "Velocity labels require mesh index 0 for terrain and one mesh per pedestrian slot; "
                f"got target mesh counts {counts}."
            )

    def get_point_velocity_labels(self) -> dict[str, torch.Tensor]:
        """Return labels aligned to the policy's current 128-bin forward LiDAR arc."""
        collector = self._held_scan_lidar_collector
        if collector is None:
            raise RuntimeError("LiDAR velocity labels require the held scan collector.")
        capture = collector.latest_capture()
        pedestrian_velocity = capture["pedestrian_velocity_w"]
        if pedestrian_velocity is None:
            raise RuntimeError("No captured pedestrian velocity is available yet; wait for a live LiDAR capture.")

        hit_xy = capture["hit_xy"]
        ray_state = capture["ray_state"]
        mesh_ids = capture["ray_mesh_ids"].to(torch.long)
        # Rollout samples only new captures, so use the pose frozen by the held
        # collector.  Reading ``sensor.data`` here could force an unintended
        # ray-caster refresh between the 130 ms scan events.
        pos_xy = capture["ego_xy"]
        yaw = capture["ego_yaw"]

        num_bins, fov_bins = 256, 128
        relative = hit_xy - pos_xy.unsqueeze(1)
        distance = torch.linalg.vector_norm(relative, dim=-1)
        ray_angle = torch.atan2(relative[..., 1], relative[..., 0])
        global_bin = ((ray_angle + math.pi) / (2.0 * math.pi) * num_bins).long() % num_bins
        center_bin = ((yaw + math.pi) / (2.0 * math.pi) * num_bins).long() % num_bins
        offsets = torch.arange(-fov_bins // 2, fov_bins - fov_bins // 2, device=self.device)
        fov_global_bins = (center_bin.unsqueeze(1) + offsets.unsqueeze(0)) % num_bins
        lookup = torch.full((self.num_envs, num_bins), -1, device=self.device, dtype=torch.long)
        lookup.scatter_(1, fov_global_bins, torch.arange(fov_bins, device=self.device).expand(self.num_envs, -1))
        local_bin = torch.gather(lookup, 1, global_bin)

        valid = (ray_state == 2) & torch.isfinite(distance) & (local_bin >= 0)
        values = torch.where(valid, distance, torch.full_like(distance, float("inf")))
        per_bin_distance = torch.full((self.num_envs, fov_bins), float("inf"), device=self.device)
        per_bin_distance.scatter_reduce_(1, local_bin.clamp_min(0), values, reduce="amin", include_self=True)
        reflection_mask = torch.isfinite(per_bin_distance)

        # Obtain the first nearest ray for each policy bin. Ties are physically
        # equivalent for distance and extremely rare; argmin makes their label stable.
        belongs_to_bin = local_bin.unsqueeze(1) == torch.arange(fov_bins, device=self.device).view(1, -1, 1)
        candidate_distance = torch.where(belongs_to_bin & valid.unsqueeze(1), distance.unsqueeze(1), float("inf"))
        winner = candidate_distance.argmin(dim=-1)
        winner_mesh = torch.gather(mesh_ids, 1, winner)
        dynamic_mask = reflection_mask & (winner_mesh >= 1) & (winner_mesh <= self.crowd_manager.max_pedestrians)
        slot = (winner_mesh - 1).clamp(0, self.crowd_manager.max_pedestrians - 1)
        velocity = torch.gather(
            pedestrian_velocity,
            1,
            slot.unsqueeze(-1).expand(-1, -1, 2),
        )
        velocity = torch.where(dynamic_mask.unsqueeze(-1), velocity, torch.zeros_like(velocity))
        return {
            "point_velocity_w": velocity,
            "reflection_mask": reflection_mask,
            "dynamic_mask": dynamic_mask,
            "range_m": torch.where(reflection_mask, per_bin_distance, torch.zeros_like(per_bin_distance)),
            "capture_index": capture["capture_index"],
        }
