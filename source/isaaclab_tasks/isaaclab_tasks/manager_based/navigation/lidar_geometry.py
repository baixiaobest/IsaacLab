"""Shared geometry for heading-centred temporal LiDAR consumers."""

from __future__ import annotations

import math

import torch


def forward_lidar_reflection_bins(
    capture: dict[str, torch.Tensor], num_bins: int = 256, fov_bins: int = 128
) -> dict[str, torch.Tensor]:
    """Return nearest valid returns in a scan-heading-centred forward arc."""
    if fov_bins <= 0 or fov_bins > num_bins or fov_bins % 2:
        raise ValueError("fov_bins must be positive, even, and no greater than num_bins.")
    hit_xy = capture["hit_xy"]
    ray_state = capture["ray_state"]
    ego_xy = capture["ego_xy"]
    ego_yaw = capture["ego_yaw"]
    num_envs = hit_xy.shape[0]
    device = hit_xy.device

    relative = hit_xy - ego_xy.unsqueeze(1)
    distance = torch.linalg.vector_norm(relative, dim=-1)
    ray_angle = torch.atan2(relative[..., 1], relative[..., 0])
    global_bin = ((ray_angle + math.pi) / (2.0 * math.pi) * num_bins).long() % num_bins
    center_bin = ((ego_yaw + math.pi) / (2.0 * math.pi) * num_bins).long() % num_bins
    offsets = torch.arange(-fov_bins // 2, fov_bins - fov_bins // 2, device=device)
    fov_global_bins = (center_bin.unsqueeze(1) + offsets.unsqueeze(0)) % num_bins
    lookup = torch.full((num_envs, num_bins), -1, device=device, dtype=torch.long)
    lookup.scatter_(1, fov_global_bins, torch.arange(fov_bins, device=device).expand(num_envs, -1))
    local_bin = torch.gather(lookup, 1, global_bin)

    valid = (ray_state == 2) & torch.isfinite(distance) & torch.isfinite(hit_xy).all(dim=-1) & (local_bin >= 0)
    bin_ids = torch.arange(fov_bins, device=device).view(1, -1, 1)
    candidates = torch.where(
        (local_bin.unsqueeze(1) == bin_ids) & valid.unsqueeze(1), distance.unsqueeze(1), float("inf")
    )
    winner_ray = candidates.argmin(dim=-1)
    range_m = candidates.amin(dim=-1)
    reflection_mask = torch.isfinite(range_m)
    winner_hit_xy = torch.gather(hit_xy, 1, winner_ray.unsqueeze(-1).expand(-1, -1, 2))
    winner_hit_xy = torch.where(reflection_mask.unsqueeze(-1), winner_hit_xy, torch.zeros_like(winner_hit_xy))
    return {
        "hit_xy": winner_hit_xy,
        "reflection_mask": reflection_mask,
        "range_m": torch.where(reflection_mask, range_m, torch.zeros_like(range_m)),
        "winner_ray": winner_ray,
        "ego_yaw": ego_yaw,
    }


def world_to_body_xy(vectors_w: torch.Tensor, yaw_w: torch.Tensor) -> torch.Tensor:
    """Rotate planar world vectors to a yaw-only robot body frame."""
    cos_yaw = torch.cos(yaw_w).unsqueeze(-1)
    sin_yaw = torch.sin(yaw_w).unsqueeze(-1)
    return torch.stack(
        (cos_yaw * vectors_w[..., 0] + sin_yaw * vectors_w[..., 1], -sin_yaw * vectors_w[..., 0] + cos_yaw * vectors_w[..., 1]),
        dim=-1,
    )


def body_to_world_xy(vectors_b: torch.Tensor, yaw_w: torch.Tensor) -> torch.Tensor:
    """Rotate planar yaw-only robot-body vectors to world coordinates."""
    cos_yaw = torch.cos(yaw_w).unsqueeze(-1)
    sin_yaw = torch.sin(yaw_w).unsqueeze(-1)
    return torch.stack(
        (cos_yaw * vectors_b[..., 0] - sin_yaw * vectors_b[..., 1], sin_yaw * vectors_b[..., 0] + cos_yaw * vectors_b[..., 1]),
        dim=-1,
    )
