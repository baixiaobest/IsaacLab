"""Physics-rate collection of completed actor lidar scans."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass


@configclass
class TwoCloudLidarCfg:
    """Calibration-facing parameters for the completed actor lidar.

    The default is the intentionally simple first-round model: one complete fan
    captured every 130 ms, with only i.i.d. endpoint-position and scan-yaw error.
    Set :attr:`use_two_cloud_assembly` once this baseline is solved to enable the
    measured two-raw-cloud / partial-fan model and its additional error sources.
    """

    sensor_name: str = "obstacle_scanner"
    raw_cloud_period_s: float = 0.065
    completed_scan_period_s: float = 0.130
    use_two_cloud_assembly: bool = False
    max_distance: float = 20.0
    full_fan_ray_count: int = 256
    completed_scan_ray_count: int = 128

    # Simple-model errors. Position perturbations are independent for each valid
    # surface-hit endpoint; yaw perturbation is one independent scalar per completed
    # scan/environment and rotates the entire scan coherently.
    iid_hit_position_noise_std_m: float = 0.01
    iid_yaw_noise_std_deg: float = 0.25

    # Each raw message contains two antipodal fans.  The 180-degree simulator fan
    # uses this azimuth lookup to retain the matching ray directions; phase moves by
    # the measured 98.2 degrees from one raw message to the next.
    raw_fan_azimuth_offsets_deg: tuple[float, float] = (0.0, 180.0)
    raw_fan_span_deg: float = 90.0
    raw_center_sweep_deg: float = 98.2
    phase_jitter_std_deg: float = 2.0
    motor_speed_range: tuple[float, float] = (0.98, 1.02)

    mounting_yaw_range_deg: tuple[float, float] = (-2.0, 2.0)
    yaw_bias_init_std_deg: float = 0.25
    yaw_bias_walk_std_deg: float = 0.10
    yaw_bias_limit_deg: float = 5.0
    yaw_rate_gain_std: float = 0.10
    xy_bias_init_std_m: float = 0.01
    xy_bias_walk_std_m: float = 0.005
    xy_speed_gain_std: float = 0.05

    range_noise_base_std_m: float = 0.01
    range_noise_per_m: float = 0.0025
    bin_dropout_probability: float = 0.02
    raw_cloud_dropout_probability: float = 0.01
    completion_latency_s: float = 0.020
    completion_latency_jitter_s: float = 0.010


class TwoCloudLidarCollector:
    """Collect complete actor scans at the configured completed-scan cadence.

    The default path captures the full simulated fan once every 130 ms. The optional
    two-cloud path preserves the measured 65 ms partial-fan schedule for later
    sim-to-real calibration without requiring a separate collector implementation.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        cfg: TwoCloudLidarCfg | None = None,
    ) -> None:
        self.env = env
        self.cfg = cfg if cfg is not None else TwoCloudLidarCfg()
        self.sensor_name = self.cfg.sensor_name
        self.max_distance = self.cfg.max_distance
        self._use_two_cloud_assembly = self.cfg.use_two_cloud_assembly
        self.raw_sweep_rad = math.radians(self.cfg.raw_center_sweep_deg)
        self.raw_half_coverage_rad = math.radians(self.cfg.raw_fan_span_deg) * 0.5
        self._fan_azimuth_offsets = torch.tensor(
            [math.radians(offset) for offset in self.cfg.raw_fan_azimuth_offsets_deg], device=env.device
        )
        self._raw_steps = max(1, round(self.cfg.raw_cloud_period_s / env.physics_dt))
        raw_period_grid = self._raw_steps * env.physics_dt
        self._completed_steps = max(1, round(self.cfg.completed_scan_period_s / env.physics_dt))
        completed_period_grid = self._completed_steps * env.physics_dt
        if not math.isclose(self.cfg.completed_scan_period_s, completed_period_grid, abs_tol=1e-6):
            raise ValueError(
                "TwoCloudLidarCfg.completed_scan_period_s must lie on the physics-time grid."
            )
        if self._use_two_cloud_assembly and not math.isclose(
            self.cfg.completed_scan_period_s, 2.0 * raw_period_grid, abs_tol=1e-6
        ):
            raise ValueError(
                "Two-cloud assembly requires completed_scan_period_s to equal two raw-cloud periods "
                "on the physics-time grid."
            )
        self._physics_steps = 0
        self._time_s = 0.0

        sensor = env.scene.sensors[self.sensor_name]
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_rays = sensor.data.ray_hits_w.shape[1]
        if self.num_rays != self.cfg.full_fan_ray_count:
            raise ValueError(
                f"TwoCloudLidarCollector expected {self.cfg.full_fan_ray_count} full-fan rays, "
                f"but '{self.sensor_name}' provides {self.num_rays}."
            )
        self.completed_num_rays = self.cfg.completed_scan_ray_count
        if self.num_rays % self.completed_num_rays != 0:
            raise ValueError(
                "TwoCloudLidarCfg.completed_scan_ray_count must divide "
                "full_fan_ray_count for fixed-width nearest-hit rebinning."
            )
        self._ray_angles = torch.linspace(-math.pi / 2.0, math.pi / 2.0, self.num_rays, device=self.device)

        # The RayCaster supplies 256 rays, but actor history stores the policy's
        # 128 directions after per-raw-cloud nearest-hit reduction.
        shape_hits = (self.num_envs, self.completed_num_rays, 2)
        shape_state = (self.num_envs, self.completed_num_rays)
        self._first_hit_xy = torch.zeros(shape_hits, device=self.device)
        self._first_state = torch.zeros(shape_state, dtype=torch.uint8, device=self.device)
        self._first_ego_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._first_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._first_lin_speed = torch.zeros(self.num_envs, device=self.device)
        self._has_first = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._pending_hit_xy = torch.zeros(shape_hits, device=self.device)
        self._pending_state = torch.zeros(shape_state, dtype=torch.uint8, device=self.device)
        self._pending_ego_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._pending_ego_yaw = torch.zeros(self.num_envs, device=self.device)
        self._pending_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._pending_available_time_s = torch.zeros(self.num_envs, device=self.device)
        self._pending_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latest_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._has_latest = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._mount_yaw = torch.zeros(self.num_envs, device=self.device)
        self._yaw_bias = torch.zeros(self.num_envs, device=self.device)
        self._xy_bias = torch.zeros(self.num_envs, 2, device=self.device)
        self._fan_phase = torch.zeros(self.num_envs, device=self.device)
        self._speed_scale = torch.ones(self.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self._has_first[env_ids] = False
        self._first_state[env_ids] = 0
        self._pending_valid[env_ids] = False
        self._has_latest[env_ids] = False
        self._latest_reference_time_s[env_ids] = self._time_s
        self._mount_yaw[env_ids] = torch.empty(env_ids.numel(), device=self.device).uniform_(
            math.radians(self.cfg.mounting_yaw_range_deg[0]), math.radians(self.cfg.mounting_yaw_range_deg[1])
        )
        self._yaw_bias[env_ids] = torch.randn(env_ids.numel(), device=self.device) * math.radians(self.cfg.yaw_bias_init_std_deg)
        self._xy_bias[env_ids] = torch.randn(env_ids.numel(), 2, device=self.device) * self.cfg.xy_bias_init_std_m
        self._fan_phase[env_ids] = torch.rand(env_ids.numel(), device=self.device) * math.pi
        self._speed_scale[env_ids] = 1.0

    def on_physics_step(self) -> None:
        self._physics_steps += 1
        self._time_s += self.env.physics_dt
        if self._use_two_cloud_assembly and self._physics_steps % self._raw_steps == 0:
            self._capture_raw_cloud()
        elif not self._use_two_cloud_assembly and self._physics_steps % self._completed_steps == 0:
            self._capture_complete_scan()

    def scan_age_s(self) -> torch.Tensor:
        age = torch.full((self.num_envs,), 0.25, device=self.device)
        available = self._has_latest
        age[available] = self._time_s - self._latest_reference_time_s[available]
        return torch.clamp(age, min=0.0)

    def consume_completed(self) -> dict[str, torch.Tensor] | None:
        ready = self._pending_valid & (self._time_s >= self._pending_available_time_s)
        if not torch.any(ready):
            return None
        env_ids = ready.nonzero(as_tuple=False).squeeze(-1)
        self._pending_valid[env_ids] = False
        self._has_latest[env_ids] = True
        self._latest_reference_time_s[env_ids] = self._pending_reference_time_s[env_ids]
        return {
            "env_ids": env_ids,
            "hit_xy": self._pending_hit_xy[env_ids],
            "ray_state": self._pending_state[env_ids],
            "ego_xy": self._pending_ego_xy[env_ids],
            "ego_yaw": self._pending_ego_yaw[env_ids],
            "scan_age_s": self.scan_age_s()[env_ids],
        }

    def _capture_raw_cloud(self) -> None:
        sensor = self.env.scene.sensors[self.sensor_name]
        data = sensor.data
        pos_w = data.pos_w
        quat_w = data.quat_w
        _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
        hit_w = data.ray_hits_w
        directions_w = sensor._ray_directions_w  # direction corresponding to every ideal simulator ray
        ray_dist = torch.linalg.vector_norm(hit_w - pos_w.unsqueeze(1), dim=-1)
        hit_valid = torch.isfinite(ray_dist) & (ray_dist < self.max_distance * 0.99)
        free_endpoint = pos_w.unsqueeze(1) + directions_w * self.max_distance
        hit_xy = torch.where(hit_valid.unsqueeze(-1), hit_w[..., :2], free_endpoint[..., :2])

        robot = self.env.scene["robot"]
        yaw_rate = torch.abs(robot.data.root_ang_vel_w[:, 2])
        lin_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1)

        first_ids = (~self._has_first).nonzero(as_tuple=False).squeeze(-1)
        second_ids = self._has_first.nonzero(as_tuple=False).squeeze(-1)
        selection = self._sample_selection_mask()
        raw_drop = torch.rand(self.num_envs, device=self.device) < self.cfg.raw_cloud_dropout_probability
        selection[raw_drop] = False
        raw_state = torch.where(hit_valid, 2, 1).to(torch.uint8)
        raw_state[~selection] = 0
        rebinned_hit_xy, rebinned_state = self._rebin_raw_to_policy(hit_xy, raw_state, pos_w[:, :2])

        if first_ids.numel() > 0:
            self._first_hit_xy[first_ids] = rebinned_hit_xy[first_ids]
            self._first_state[first_ids] = rebinned_state[first_ids]
            self._first_ego_xy[first_ids] = pos_w[first_ids, :2]
            self._first_yaw_rate[first_ids] = yaw_rate[first_ids]
            self._first_lin_speed[first_ids] = lin_speed[first_ids]
            self._has_first[first_ids] = True
            self._speed_scale[first_ids] = torch.empty(first_ids.numel(), device=self.device).uniform_(
                self.cfg.motor_speed_range[0], self.cfg.motor_speed_range[1]
            )

        if second_ids.numel() > 0:
            first_hits = self._perturb_raw(
                second_ids,
                self._first_hit_xy[second_ids],
                self._first_state[second_ids],
                self._first_ego_xy[second_ids],
                self._first_yaw_rate[second_ids],
                self._first_lin_speed[second_ids],
                age_s=0.0975,
            )
            second_hits = self._perturb_raw(
                second_ids,
                rebinned_hit_xy[second_ids],
                rebinned_state[second_ids],
                pos_w[second_ids, :2],
                yaw_rate[second_ids],
                lin_speed[second_ids],
                age_s=0.0325,
            )
            merged_xy, merged_state = self._merge_raw(
                first_hits, self._first_state[second_ids], second_hits, rebinned_state[second_ids]
            )
            dropout = (
                torch.rand_like(merged_state, dtype=torch.float32) < self.cfg.bin_dropout_probability
            ) & (merged_state > 0)
            merged_state[dropout] = 0
            merged_xy[dropout] = 0.0

            self._pending_hit_xy[second_ids] = merged_xy
            self._pending_state[second_ids] = merged_state
            self._pending_ego_xy[second_ids] = pos_w[second_ids, :2]
            self._pending_ego_yaw[second_ids] = yaw[second_ids]
            self._pending_reference_time_s[second_ids] = self._time_s
            latency = self.cfg.completion_latency_s + torch.rand(second_ids.numel(), device=self.device) * self.cfg.completion_latency_jitter_s
            self._pending_available_time_s[second_ids] = self._time_s + latency
            self._pending_valid[second_ids] = True
            self._has_first[second_ids] = False
            self._yaw_bias[second_ids] = torch.clamp(
                self._yaw_bias[second_ids]
                + torch.randn(second_ids.numel(), device=self.device) * math.radians(self.cfg.yaw_bias_walk_std_deg),
                -math.radians(self.cfg.yaw_bias_limit_deg),
                math.radians(self.cfg.yaw_bias_limit_deg),
            )
            self._xy_bias[second_ids] += torch.randn(second_ids.numel(), 2, device=self.device) * self.cfg.xy_bias_walk_std_m

        self._fan_phase = torch.remainder(
            self._fan_phase
            + self.raw_sweep_rad * self._speed_scale
            + torch.randn(self.num_envs, device=self.device) * math.radians(self.cfg.phase_jitter_std_deg),
            math.pi,
        )

    def _capture_complete_scan(self) -> None:
        """Capture one full ideal fan and apply only the simple first-round errors."""
        sensor = self.env.scene.sensors[self.sensor_name]
        data = sensor.data
        pos_w = data.pos_w
        quat_w = data.quat_w
        _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
        hit_w = data.ray_hits_w
        directions_w = sensor._ray_directions_w

        ray_dist = torch.linalg.vector_norm(hit_w - pos_w.unsqueeze(1), dim=-1)
        hit_valid = torch.isfinite(ray_dist) & (ray_dist < self.max_distance * 0.99)
        free_endpoint = pos_w.unsqueeze(1) + directions_w * self.max_distance
        hit_xy = torch.where(hit_valid.unsqueeze(-1), hit_w[..., :2], free_endpoint[..., :2])
        ray_state = torch.where(hit_valid, 2, 1).to(torch.uint8)
        hit_xy, ray_state = self._rebin_raw_to_policy(hit_xy, ray_state, pos_w[:, :2])
        hit_xy = self._apply_simple_scan_noise(hit_xy, ray_state, pos_w[:, :2])

        self._pending_hit_xy[:] = hit_xy
        self._pending_state[:] = ray_state
        self._pending_ego_xy[:] = pos_w[:, :2]
        self._pending_ego_yaw[:] = yaw
        self._pending_reference_time_s[:] = self._time_s
        # The simple baseline deliberately has no completion pipeline delay.
        self._pending_available_time_s[:] = self._time_s
        self._pending_valid[:] = True

    def _apply_simple_scan_noise(
        self, hit_xy: torch.Tensor, ray_state: torch.Tensor, ego_xy: torch.Tensor
    ) -> torch.Tensor:
        """Apply i.i.d. hit-position noise and scan-wise i.i.d. yaw noise only."""
        noisy_xy = hit_xy.clone()
        hit_mask = ray_state == 2
        if self.cfg.iid_hit_position_noise_std_m > 0.0:
            position_noise = torch.randn_like(noisy_xy) * self.cfg.iid_hit_position_noise_std_m
            noisy_xy = torch.where(hit_mask.unsqueeze(-1), noisy_xy + position_noise, noisy_xy)

        if self.cfg.iid_yaw_noise_std_deg > 0.0:
            yaw_error = torch.randn(self.num_envs, device=self.device) * math.radians(self.cfg.iid_yaw_noise_std_deg)
            rel = noisy_xy - ego_xy.unsqueeze(1)
            cos_yaw = torch.cos(yaw_error).unsqueeze(1)
            sin_yaw = torch.sin(yaw_error).unsqueeze(1)
            noisy_xy = ego_xy.unsqueeze(1) + torch.stack(
                (cos_yaw * rel[..., 0] - sin_yaw * rel[..., 1], sin_yaw * rel[..., 0] + cos_yaw * rel[..., 1]),
                dim=-1,
            )

        noisy_xy[ray_state == 0] = 0.0
        return noisy_xy

    def _sample_selection_mask(self) -> torch.Tensor:
        # Configurable azimuth lookup for the two opposing fans in one hardware raw
        # cloud.  The simulator ray fan is front-only, so this retains whichever of
        # the two physical fans intersects each simulated azimuth direction.
        centers = self._fan_phase[:, None] + self._fan_azimuth_offsets[None, :]
        relative = self._ray_angles[None, :, None] - centers[:, None, :]
        wrapped = torch.remainder(relative + math.pi, 2.0 * math.pi) - math.pi
        return (torch.abs(wrapped) <= self.raw_half_coverage_rad).any(dim=-1)

    def _rebin_raw_to_policy(
        self, hit_xy: torch.Tensor, state: torch.Tensor, ego_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce full-fan rays to fixed policy directions using nearest-hit semantics.

        A policy direction represents an equal-width block of source rays.  If that
        block contains one or more surface returns, retain the closest one; otherwise
        retain one observed free-space ray.  A block remains invalid only when no
        source ray was selected by this raw cloud's fan schedule.
        """
        group_size = self.num_rays // self.completed_num_rays
        grouped_xy = hit_xy.view(self.num_envs, self.completed_num_rays, group_size, 2)
        grouped_state = state.view(self.num_envs, self.completed_num_rays, group_size)

        hit_mask = grouped_state == 2
        free_mask = grouped_state == 1
        hit_distance_sq = torch.sum((grouped_xy - ego_xy[:, None, None, :]) ** 2, dim=-1)
        nearest_hit_index = hit_distance_sq.masked_fill(~hit_mask, float("inf")).argmin(dim=-1)
        first_free_index = free_mask.to(torch.long).argmax(dim=-1)
        has_hit = hit_mask.any(dim=-1)
        has_free = free_mask.any(dim=-1)
        selected_index = torch.where(has_hit, nearest_hit_index, first_free_index)
        selected_xy = torch.gather(
            grouped_xy, 2, selected_index[..., None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        selected_state = torch.where(
            has_hit,
            torch.full_like(nearest_hit_index, 2, dtype=torch.uint8),
            torch.where(
                has_free,
                torch.ones_like(nearest_hit_index, dtype=torch.uint8),
                torch.zeros_like(nearest_hit_index, dtype=torch.uint8),
            ),
        )
        selected_xy[selected_state == 0] = 0.0
        return selected_xy, selected_state

    def _perturb_raw(
        self,
        env_ids: torch.Tensor,
        hit_xy: torch.Tensor,
        state: torch.Tensor,
        ego_xy: torch.Tensor,
        yaw_rate: torch.Tensor,
        lin_speed: torch.Tensor,
        age_s: float,
    ) -> torch.Tensor:
        rel = hit_xy - ego_xy.unsqueeze(1)
        dist = torch.linalg.vector_norm(rel, dim=-1)
        hit_mask = state == 2
        range_sigma = self.cfg.range_noise_base_std_m + self.cfg.range_noise_per_m * dist
        noisy_dist = torch.clamp(dist + torch.randn_like(dist) * range_sigma, 0.0, self.max_distance)
        scale = torch.where(hit_mask & (dist > 1e-6), noisy_dist / torch.clamp(dist, min=1e-6), torch.ones_like(dist))
        rel = rel * scale.unsqueeze(-1)

        yaw_gain = torch.randn(env_ids.numel(), device=self.device) * self.cfg.yaw_rate_gain_std
        yaw_error = self._mount_yaw[env_ids] + self._yaw_bias[env_ids] + yaw_gain * yaw_rate * age_s
        cos_yaw = torch.cos(yaw_error).unsqueeze(1)
        sin_yaw = torch.sin(yaw_error).unsqueeze(1)
        rotated = torch.stack(
            (cos_yaw * rel[..., 0] - sin_yaw * rel[..., 1], sin_yaw * rel[..., 0] + cos_yaw * rel[..., 1]), dim=-1
        )
        age_xy = (
            torch.randn(env_ids.numel(), 2, device=self.device)
            * self.cfg.xy_speed_gain_std
            * lin_speed.unsqueeze(-1)
            * age_s
        )
        return ego_xy.unsqueeze(1) + rotated + self._xy_bias[env_ids].unsqueeze(1) + age_xy.unsqueeze(1)

    @staticmethod
    def _merge_raw(
        first_xy: torch.Tensor, first_state: torch.Tensor, second_xy: torch.Tensor, second_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_hit = first_state == 2
        second_hit = second_state == 2
        first_real = first_state > 0
        second_real = second_state > 0
        # Overlap is intentional.  Prefer the later cloud because it is younger and
        # therefore has the smaller deskew residual; a hit still overrides an older
        # free-space contribution.  Comparing world-coordinate norms here would be
        # wrong for environments away from the world origin.
        choose_second = second_real & (~first_real | second_hit)
        merged_xy = torch.where(choose_second.unsqueeze(-1), second_xy, first_xy)
        merged_state = torch.where(choose_second, second_state, first_state)
        return merged_xy, merged_state


class TwoCloudTemporalLidarRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv variant that feeds :class:`TwoCloudLidarCollector`."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        # ``ManagerBasedEnv.__init__`` calls ``self.load_managers()`` in the
        # standalone path.  Set this before entering the base constructor so our
        # override can create the collector before ObservationManager instantiates
        # temporal_lidar_scan_age.
        self._two_cloud_lidar_collector: TwoCloudLidarCollector | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _ensure_two_cloud_lidar_collector(self) -> None:
        if self._two_cloud_lidar_collector is None:
            self._two_cloud_lidar_collector = TwoCloudLidarCollector(
                self, getattr(self.cfg, "two_cloud_lidar", None)
            )

    def load_managers(self) -> None:
        """Create the collector before observation terms resolve it by name."""
        self._ensure_two_cloud_lidar_collector()
        super().load_managers()

    def _post_physics_step(self) -> None:
        if self._two_cloud_lidar_collector is not None:
            self._two_cloud_lidar_collector.on_physics_step()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if self._two_cloud_lidar_collector is not None:
            self._two_cloud_lidar_collector.reset(env_ids)
