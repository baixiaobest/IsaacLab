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

    One complete fan is captured every 130 ms.  The only actor perturbations are
    i.i.d. endpoint-position noise and a scan-wise i.i.d. yaw error.
    """

    sensor_name: str = "obstacle_scanner"
    completed_scan_period_s: float = 0.130
    max_distance: float = 20.0
    full_fan_ray_count: int = 256
    completed_scan_ray_count: int = 128

    # Simple-model errors. Position perturbations are independent for each valid
    # surface-hit endpoint; yaw perturbation is one independent scalar per completed
    # scan/environment and rotates the entire scan coherently.
    iid_hit_position_noise_std_m: float = 0.01
    iid_yaw_noise_std_deg: float = 0.25


class TwoCloudLidarCollector:
    """Collect complete actor scans at the configured completed-scan cadence.

    Each capture starts with the full simulated 180-degree fan, then reduces it to
    the policy's fixed output directions.
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
        self._completed_steps = max(1, round(self.cfg.completed_scan_period_s / env.physics_dt))
        completed_period_grid = self._completed_steps * env.physics_dt
        if not math.isclose(self.cfg.completed_scan_period_s, completed_period_grid, abs_tol=1e-6):
            raise ValueError(
                "TwoCloudLidarCfg.completed_scan_period_s must lie on the physics-time grid."
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
        # The RayCaster supplies 256 rays, but actor history stores the policy's
        # 128 directions after nearest-hit reduction.
        shape_hits = (self.num_envs, self.completed_num_rays, 2)
        shape_state = (self.num_envs, self.completed_num_rays)
        self._pending_hit_xy = torch.zeros(shape_hits, device=self.device)
        self._pending_state = torch.zeros(shape_state, dtype=torch.uint8, device=self.device)
        self._pending_ego_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._pending_ego_yaw = torch.zeros(self.num_envs, device=self.device)
        self._pending_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._pending_available_time_s = torch.zeros(self.num_envs, device=self.device)
        self._pending_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latest_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._has_latest = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reset()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self._pending_valid[env_ids] = False
        self._has_latest[env_ids] = False
        self._latest_reference_time_s[env_ids] = self._time_s

    def on_physics_step(self) -> None:
        self._physics_steps += 1
        self._time_s += self.env.physics_dt
        if self._physics_steps % self._completed_steps == 0:
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
        hit_xy, ray_state = self._rebin_to_policy(hit_xy, ray_state, pos_w[:, :2])
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

    def _rebin_to_policy(
        self, hit_xy: torch.Tensor, state: torch.Tensor, ego_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce full-fan rays to fixed policy directions using nearest-hit semantics.

        A policy direction represents an equal-width block of source rays.  If that
        block contains one or more surface returns, retain the closest one; otherwise
        retain one observed free-space ray.  A block remains invalid only when no
        source ray is valid.
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
