# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""A Dynamic Window Approach (DWA) baseline for Go2 obstacle-avoidance navigation.

Unlike a purely reactive controller that scores raw LiDAR ray directions directly, DWA reasons
about the robot's own reachable velocities: at every high-level control step it builds a "dynamic
window" of ``(vx, wz)`` pairs reachable from the robot's *current measured* velocity within one
acceleration-limited control step, forward-simulates a short unicycle trajectory for every
candidate in that window, and scores each trajectory by a mix of goal-heading alignment, worst-case
clearance from LiDAR obstacle points along the trajectory, and forward speed (to avoid the
"frozen robot" failure mode of overly conservative candidates). The highest-scoring, collision-free
candidate is steered toward; if every candidate collides, the controller falls back to stopping and
rotating in place toward the goal.
"""

from __future__ import annotations

import math

import torch


class DwaController:
    """Vectorized Dynamic Window Approach controller, unicycle-style (``vy`` is always 0)."""

    _NO_HIT_EPS = 0.01
    """Fraction of ``max_distance`` below which a ray is still treated as a real obstacle hit."""

    def __init__(
        self,
        *,
        vel_abs_lower_mps: float = 0.0,
        vel_abs_upper_mps: float = 1.0,
        accel_lower_mps2: float = -3.0,
        accel_upper_mps2: float = 3.0,
        max_yaw_rate_rad_s: float = 1.5,
        max_yaw_accel_rad_s2: float = 3.0,
        rollout_dt_s: float | None = None,
        num_rollout_steps: int = 10,
        num_vx_samples: int = 11,
        num_wz_samples: int = 21,
        w_heading: float = 1.0,
        w_clearance: float = 1.0,
        w_velocity: float = 0.5,
        distance_cap_m: float = 3.0,
        robot_radius_m: float = 0.4,
        safety_margin_m: float = 0.1,
        fallback_yaw_gain: float = 1.5,
        obstacle_stride: int = 1,
        sensor_name: str = "obstacle_scanner",
        command_name: str = "pose_2d_command",
    ):
        self.vel_abs_lower_mps = vel_abs_lower_mps
        self.vel_abs_upper_mps = vel_abs_upper_mps
        self.accel_lower_mps2 = accel_lower_mps2
        self.accel_upper_mps2 = accel_upper_mps2
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self.max_yaw_accel_rad_s2 = max_yaw_accel_rad_s2
        self.rollout_dt_s = rollout_dt_s
        self.num_rollout_steps = num_rollout_steps
        self.num_vx_samples = num_vx_samples
        self.num_wz_samples = num_wz_samples
        self.w_heading = w_heading
        self.w_clearance = w_clearance
        self.w_velocity = w_velocity
        self.distance_cap_m = distance_cap_m
        self.robot_radius_m = robot_radius_m
        self.safety_margin_m = safety_margin_m
        self.fallback_yaw_gain = fallback_yaw_gain
        self.obstacle_stride = obstacle_stride
        self.sensor_name = sensor_name
        self.command_name = command_name

    def compute_actions(self, env) -> torch.Tensor:
        """Return a ``(num_envs, 3)`` tensor of body-frame ``(vx, vy, wz)`` commands."""
        from isaaclab.envs.mdp.observations import lidar_scan
        from isaaclab.managers import SceneEntityCfg

        sensor_cfg = SceneEntityCfg(self.sensor_name)
        sensor = env.scene.sensors[self.sensor_name]
        pattern_cfg = sensor.cfg.pattern_cfg
        if getattr(pattern_cfg, "channels", 1) != 1:
            raise ValueError("DwaController expects a single-channel (2D) lidar pattern.")
        fov_min_deg, fov_max_deg = pattern_cfg.horizontal_fov_range

        distances = lidar_scan(env, sensor_cfg, max=sensor.cfg.max_distance, scale_distance=False)
        num_rays = distances.shape[1]
        device = distances.device
        ray_angles_rad = torch.linspace(
            math.radians(fov_min_deg), math.radians(fov_max_deg), num_rays, device=device
        )

        goal_command = env.command_manager.get_command(self.command_name)
        robot = env.scene["robot"]
        v_meas = robot.data.root_lin_vel_b[:, 0]
        w_meas = robot.data.root_ang_vel_b[:, 2]
        dt = self.rollout_dt_s if self.rollout_dt_s is not None else env.step_dt

        return self._compute_actions_from_state(
            distances=distances,
            ray_angles_rad=ray_angles_rad,
            max_distance=sensor.cfg.max_distance,
            goal_xy=goal_command[:, :2],
            v_meas=v_meas,
            w_meas=w_meas,
            dt=dt,
        )

    def _compute_actions_from_state(
        self,
        *,
        distances: torch.Tensor,
        ray_angles_rad: torch.Tensor,
        max_distance: float,
        goal_xy: torch.Tensor,
        v_meas: torch.Tensor,
        w_meas: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Pure-tensor DWA core: obstacle points -> dynamic window -> rollout -> scoring -> argmax.

        Kept independent of any Isaac Lab/Isaac Sim types so it can be exercised directly with
        hand-built tensors in unit tests, with no simulator app launch required.
        """
        device = distances.device
        num_envs = distances.shape[0]

        # ---- obstacle points in the current body frame -------------------------------------
        valid = distances < max_distance * (1.0 - self._NO_HIT_EPS)
        cos_ray = torch.cos(ray_angles_rad)
        sin_ray = torch.sin(ray_angles_rad)
        if self.obstacle_stride > 1:
            distances = distances[:, :: self.obstacle_stride]
            valid = valid[:, :: self.obstacle_stride]
            cos_ray = cos_ray[:: self.obstacle_stride]
            sin_ray = sin_ray[:: self.obstacle_stride]
        obs_x = distances * cos_ray.unsqueeze(0)
        obs_y = distances * sin_ray.unsqueeze(0)
        obs_x = obs_x.unsqueeze(1)  # (num_envs, 1, num_points)
        obs_y = obs_y.unsqueeze(1)
        valid = valid.unsqueeze(1)  # (num_envs, 1, num_points)

        # ---- dynamic window: velocities reachable in one accel-limited control step ---------
        vx_lo = torch.clamp(
            v_meas + self.accel_lower_mps2 * dt, min=self.vel_abs_lower_mps, max=self.vel_abs_upper_mps
        )
        vx_hi = torch.clamp(
            v_meas + self.accel_upper_mps2 * dt, min=self.vel_abs_lower_mps, max=self.vel_abs_upper_mps
        )
        vx_hi = torch.maximum(vx_hi, vx_lo)
        wz_lo = torch.clamp(
            w_meas - self.max_yaw_accel_rad_s2 * dt, min=-self.max_yaw_rate_rad_s, max=self.max_yaw_rate_rad_s
        )
        wz_hi = torch.clamp(
            w_meas + self.max_yaw_accel_rad_s2 * dt, min=-self.max_yaw_rate_rad_s, max=self.max_yaw_rate_rad_s
        )
        wz_hi = torch.maximum(wz_hi, wz_lo)

        # ---- candidate grid: per-env outer product of vx/wz samples in the window -----------
        vx_unit = torch.linspace(0.0, 1.0, self.num_vx_samples, device=device)
        wz_unit = torch.linspace(0.0, 1.0, self.num_wz_samples, device=device)
        vx_grid = vx_lo.unsqueeze(1) + (vx_hi - vx_lo).unsqueeze(1) * vx_unit.unsqueeze(0)
        wz_grid = wz_lo.unsqueeze(1) + (wz_hi - wz_lo).unsqueeze(1) * wz_unit.unsqueeze(0)
        vx_cand = vx_grid.unsqueeze(2).expand(num_envs, self.num_vx_samples, self.num_wz_samples)
        wz_cand = wz_grid.unsqueeze(1).expand(num_envs, self.num_vx_samples, self.num_wz_samples)
        vx_cand = vx_cand.reshape(num_envs, -1)
        wz_cand = wz_cand.reshape(num_envs, -1)

        # ---- forward rollout (unicycle kinematics, anchored at the current body frame) ------
        x = torch.zeros_like(vx_cand)
        y = torch.zeros_like(vx_cand)
        theta = torch.zeros_like(vx_cand)
        running_min_clearance = torch.full_like(vx_cand, float("inf"))
        for _ in range(self.num_rollout_steps):
            x = x + vx_cand * torch.cos(theta) * dt
            y = y + vx_cand * torch.sin(theta) * dt
            theta = theta + wz_cand * dt

            dx = x.unsqueeze(2) - obs_x
            dy = y.unsqueeze(2) - obs_y
            point_dist = torch.sqrt(dx * dx + dy * dy)
            point_dist = torch.where(valid, point_dist, torch.full_like(point_dist, float("inf")))
            step_min_dist = point_dist.amin(dim=2)
            running_min_clearance = torch.minimum(running_min_clearance, step_min_dist)

        # ---- scoring: heading alignment at trajectory end, clearance, forward speed ---------
        goal_x = goal_xy[:, 0].unsqueeze(1)
        goal_y = goal_xy[:, 1].unsqueeze(1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        rel_x = cos_t * (goal_x - x) + sin_t * (goal_y - y)
        rel_y = -sin_t * (goal_x - x) + cos_t * (goal_y - y)
        heading_error = torch.atan2(rel_y, rel_x)
        heading_score = 1.0 - torch.abs(heading_error) / math.pi

        clearance = running_min_clearance - self.robot_radius_m
        clearance_score = torch.clamp(clearance, min=0.0, max=self.distance_cap_m) / self.distance_cap_m

        velocity_score = vx_cand / self.vel_abs_upper_mps if self.vel_abs_upper_mps > 0 else torch.zeros_like(vx_cand)

        score = self.w_heading * heading_score + self.w_clearance * clearance_score + self.w_velocity * velocity_score

        # ---- feasibility: exclude colliding candidates from the argmax, don't average them --
        collision_mask = clearance < self.safety_margin_m
        score = torch.where(collision_mask, torch.full_like(score, -1.0e6), score)

        best_index = torch.argmax(score, dim=1)
        vx_star = vx_cand.gather(1, best_index.unsqueeze(1)).squeeze(1)
        wz_star = wz_cand.gather(1, best_index.unsqueeze(1)).squeeze(1)

        # ---- fallback: if every candidate collides, stop and rotate toward the goal ---------
        all_infeasible = collision_mask.all(dim=1)
        fallback_wz = torch.clamp(
            self.fallback_yaw_gain * torch.atan2(goal_xy[:, 1], goal_xy[:, 0]),
            -self.max_yaw_rate_rad_s,
            self.max_yaw_rate_rad_s,
        )
        vx_star = torch.where(all_infeasible, torch.zeros_like(vx_star), vx_star)
        wz_star = torch.where(all_infeasible, fallback_wz, wz_star)

        vy_star = torch.zeros_like(vx_star)
        return torch.stack([vx_star, vy_star, wz_star], dim=-1)
