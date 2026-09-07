# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""A non-learning, reactive LiDAR baseline for Go2 obstacle-avoidance navigation.

At every high-level control step the controller looks only at the current LiDAR scan (no
prediction of future pedestrian motion), scores a set of candidate steering directions by a
mix of available free space and alignment with the goal bearing, and steers toward the
highest-scoring direction while slowing down near obstacles or when a large turn is required.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isaaclab.envs.mdp.observations import lidar_scan
from isaaclab.managers import SceneEntityCfg


class GreedyLidarController:
    """Vectorized greedy reactive controller: ``S(theta) = alpha * D(theta) - beta * |theta - theta_g|``."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.5,
        max_speed_mps: float = 1.0,
        min_speed_mps: float = 0.0,
        slow_distance_m: float = 1.5,
        full_stop_angle_deg: float = 70.0,
        yaw_gain: float = 1.5,
        max_yaw_rate_rad_s: float = 1.5,
        clearance_window_deg: float = 12.0,
        distance_cap_m: float = 3.0,
        num_candidates: int | None = None,
        sensor_name: str = "obstacle_scanner",
        command_name: str = "pose_2d_command",
    ):
        self.alpha = alpha
        self.beta = beta
        self.max_speed_mps = max_speed_mps
        self.min_speed_mps = min_speed_mps
        self.slow_distance_m = slow_distance_m
        self.full_stop_angle_rad = math.radians(full_stop_angle_deg)
        self.yaw_gain = yaw_gain
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self.clearance_window_deg = clearance_window_deg
        self.distance_cap_m = distance_cap_m
        self.num_candidates = num_candidates
        self.sensor_cfg = SceneEntityCfg(sensor_name)
        self.command_name = command_name

    def compute_actions(self, env) -> torch.Tensor:
        """Return a ``(num_envs, 3)`` tensor of body-frame ``(vx, vy, wz)`` commands."""
        sensor = env.scene.sensors[self.sensor_cfg.name]
        pattern_cfg = sensor.cfg.pattern_cfg
        if getattr(pattern_cfg, "channels", 1) != 1:
            raise ValueError("GreedyLidarController expects a single-channel (2D) lidar pattern.")
        fov_min_deg, fov_max_deg = pattern_cfg.horizontal_fov_range

        distances = lidar_scan(env, self.sensor_cfg, max=sensor.cfg.max_distance, scale_distance=False)
        num_envs, num_rays = distances.shape
        device = distances.device

        ray_angles_rad = torch.linspace(
            math.radians(fov_min_deg), math.radians(fov_max_deg), num_rays, device=device
        )
        deg_per_ray = (fov_max_deg - fov_min_deg) / max(num_rays - 1, 1)
        half_window = max(1, round((self.clearance_window_deg / 2.0) / max(deg_per_ray, 1e-6)))

        # Sliding-window minimum distance ("free space") around every raw ray bearing. Padding
        # with zero treats the unobserved region just past the sensor's edge as blocked, so
        # candidates right at the FOV boundary are not scored as falsely open.
        padded = F.pad(distances, (half_window, half_window), mode="constant", value=0.0)
        window_size = 2 * half_window + 1
        clearance_per_ray = padded.unfold(dimension=1, size=window_size, step=1).amin(dim=-1)

        if self.num_candidates is not None and self.num_candidates < num_rays:
            candidate_indices = torch.linspace(0, num_rays - 1, self.num_candidates, device=device).round().long()
        else:
            candidate_indices = torch.arange(num_rays, device=device)
        candidate_angles_rad = ray_angles_rad[candidate_indices]
        candidate_clearance = clearance_per_ray.index_select(1, candidate_indices)

        goal_command = env.command_manager.get_command(self.command_name)
        theta_g = torch.atan2(goal_command[:, 1], goal_command[:, 0])
        theta_g = torch.clamp(theta_g, min=ray_angles_rad[0].item(), max=ray_angles_rad[-1].item())

        distance_norm = torch.clamp(candidate_clearance, max=self.distance_cap_m) / self.distance_cap_m
        angle_error_norm = torch.abs(candidate_angles_rad.unsqueeze(0) - theta_g.unsqueeze(1)) / math.pi
        scores = self.alpha * distance_norm - self.beta * angle_error_norm

        best_index = torch.argmax(scores, dim=1)
        theta_star = candidate_angles_rad[best_index]
        clearance_star = candidate_clearance.gather(1, best_index.unsqueeze(1)).squeeze(1)

        wz = torch.clamp(self.yaw_gain * theta_star, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s)
        clearance_scale = torch.clamp(clearance_star / self.slow_distance_m, 0.0, 1.0)
        turn_scale = torch.clamp(1.0 - torch.abs(theta_star) / self.full_stop_angle_rad, 0.0, 1.0)
        speed_scale = clearance_scale * turn_scale
        vx = self.min_speed_mps + (self.max_speed_mps - self.min_speed_mps) * speed_scale
        vx = torch.clamp(vx, min=0.0)
        vy = torch.zeros_like(vx)

        return torch.stack([vx, vy, wz], dim=-1)
