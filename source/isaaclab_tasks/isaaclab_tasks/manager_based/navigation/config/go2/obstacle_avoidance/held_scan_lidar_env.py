"""Physics-rate collection of held, full-fan lidar scans."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass


@configclass
class HeldScanLidarCfg:
    """Configuration for a complete lidar fan held between fixed-rate captures."""

    sensor_name: str = "obstacle_scanner"
    scan_period_s: float = 0.130
    max_distance: float = 20.0
    full_fan_ray_count: int = 256


class HeldScanLidarCollector:
    """Capture one ideal full lidar fan every ``scan_period_s`` physics seconds.

    The collector deliberately models timing only.  It never re-bins, assembles
    partial clouds, delays, or corrupts scans: actor-side corruption is applied by
    the temporal observation term exactly as in the ``640034b`` baseline.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: HeldScanLidarCfg | None = None) -> None:
        self.env = env
        self.cfg = cfg if cfg is not None else HeldScanLidarCfg()
        self.sensor_name = self.cfg.sensor_name
        self.max_distance = self.cfg.max_distance
        self._scan_steps = max(1, round(self.cfg.scan_period_s / env.physics_dt))
        period_on_grid = self._scan_steps * env.physics_dt
        if not math.isclose(self.cfg.scan_period_s, period_on_grid, abs_tol=1e-6):
            raise ValueError("HeldScanLidarCfg.scan_period_s must lie on the physics-time grid.")
        self._physics_steps = 0
        self._time_s = 0.0

        sensor = env.scene.sensors[self.sensor_name]
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_rays = sensor.data.ray_hits_w.shape[1]
        if self.num_rays != self.cfg.full_fan_ray_count:
            raise ValueError(
                f"HeldScanLidarCollector expected {self.cfg.full_fan_ray_count} full-fan rays, "
                f"but '{self.sensor_name}' provides {self.num_rays}."
            )

        self._pending_hit_xy = torch.zeros(self.num_envs, self.num_rays, 2, device=self.device)
        self._pending_state = torch.zeros(self.num_envs, self.num_rays, dtype=torch.uint8, device=self.device)
        self._pending_ego_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._pending_ego_yaw = torch.zeros(self.num_envs, device=self.device)
        self._pending_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._pending_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latest_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._has_latest = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reset()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset and immediately queue a current full scan for ``env_ids``."""
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._pending_valid[env_ids] = False
        self._has_latest[env_ids] = False
        self._latest_reference_time_s[env_ids] = self._time_s
        self._capture_full_scan(env_ids)

    def on_physics_step(self) -> None:
        self._physics_steps += 1
        self._time_s += self.env.physics_dt
        if self._physics_steps % self._scan_steps == 0:
            self._capture_full_scan()

    def scan_age_s(self) -> torch.Tensor:
        """Return age of each latest scan, or one full period when unavailable."""
        age = torch.full((self.num_envs,), self.cfg.scan_period_s, device=self.device)
        available = self._has_latest
        age[available] = self._time_s - self._latest_reference_time_s[available]
        return torch.clamp(age, min=0.0)

    def latest_capture(self) -> dict[str, torch.Tensor]:
        """Return the most recently captured ideal scan without consuming it.

        The temporal observation path consumes captures to update its history.  A
        deployment controller, however, must be able to use the same held scan
        at its own control rate without changing that history or forcing a new
        ray-caster update.  The returned hit points are world-frame XY points;
        ``ray_state == 2`` identifies a valid reflection and ``ray_state == 1``
        is a no-return endpoint.
        """
        return {
            "hit_xy": self._pending_hit_xy,
            "ray_state": self._pending_state,
            "ego_xy": self._pending_ego_xy,
            "ego_yaw": self._pending_ego_yaw,
            "scan_age_s": self.scan_age_s(),
        }

    def consume_completed(self) -> dict[str, torch.Tensor] | None:
        """Return each queued full scan once, leaving it held thereafter."""
        if not torch.any(self._pending_valid):
            return None
        env_ids = self._pending_valid.nonzero(as_tuple=False).squeeze(-1)
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

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _capture_full_scan(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Queue the current ideal 256-ray fan for selected environments."""
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        sensor = self.env.scene.sensors[self.sensor_name]
        data = sensor.data
        pos_w = data.pos_w
        quat_w = data.quat_w
        hit_w = data.ray_hits_w
        directions_w = sensor._ray_directions_w

        ray_dist = torch.linalg.vector_norm(hit_w - pos_w.unsqueeze(1), dim=-1)
        hit_valid = torch.isfinite(ray_dist) & (ray_dist < self.max_distance * 0.99)
        free_endpoint = pos_w.unsqueeze(1) + directions_w * self.max_distance
        hit_xy = torch.where(hit_valid.unsqueeze(-1), hit_w[..., :2], free_endpoint[..., :2])
        _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)

        self._pending_hit_xy[env_ids] = hit_xy[env_ids]
        self._pending_state[env_ids] = torch.where(hit_valid[env_ids], 2, 1).to(torch.uint8)
        self._pending_ego_xy[env_ids] = pos_w[env_ids, :2]
        self._pending_ego_yaw[env_ids] = yaw[env_ids]
        self._pending_reference_time_s[env_ids] = self._time_s
        self._pending_valid[env_ids] = True


class HeldScanTemporalLidarRLEnv(ManagerBasedRLEnv):
    """Temporal-lidar environment fed by :class:`HeldScanLidarCollector`."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        self._held_scan_lidar_collector: HeldScanLidarCollector | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _ensure_held_scan_lidar_collector(self) -> None:
        if self._held_scan_lidar_collector is None:
            self._held_scan_lidar_collector = HeldScanLidarCollector(
                self, getattr(self.cfg, "held_scan_lidar", None)
            )

    def load_managers(self) -> None:
        self._ensure_held_scan_lidar_collector()
        super().load_managers()

    def _post_physics_step(self) -> None:
        if self._held_scan_lidar_collector is not None:
            self._held_scan_lidar_collector.on_physics_step()

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if self._held_scan_lidar_collector is not None:
            self._held_scan_lidar_collector.reset(env_ids)
