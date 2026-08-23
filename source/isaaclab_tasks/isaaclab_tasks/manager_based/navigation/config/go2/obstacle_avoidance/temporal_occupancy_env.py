"""Physics-rate temporal occupancy-map collection for mixed navigation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


@configclass
class TemporalOccupancyCfg:
    """Sampling geometry for one independently owned occupancy-map history."""

    sensor_name: str = "obstacle_scanner"
    grid_size: int = 50
    grid_resolution: float = 0.2
    sample_period_s: float = 0.5
    history_frames: int = 6
    min_height_rel: float = -0.1
    max_height_rel: float = 1.5
    max_range: float = 10.0


class TemporalOccupancyCollector:
    """Hold six historical ego-centric occupancy maps sampled on the physics grid.

    The output intentionally excludes a current map.  At time ``t`` it is ordered
    oldest-to-newest as ``[t - 3.0, t - 2.5, ..., t - 0.5]``.  A seventh internal
    slot holds the current capture so this convention remains true at sampling
    instants too.  Frames unavailable immediately after reset are zero-padded.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: TemporalOccupancyCfg | None = None) -> None:
        self.env = env
        self.cfg = cfg if cfg is not None else TemporalOccupancyCfg()
        if self.cfg.grid_size <= 0:
            raise ValueError("TemporalOccupancyCfg.grid_size must be positive.")
        if self.cfg.grid_resolution <= 0.0:
            raise ValueError("TemporalOccupancyCfg.grid_resolution must be positive.")
        if self.cfg.sample_period_s <= 0.0:
            raise ValueError("TemporalOccupancyCfg.sample_period_s must be positive.")
        if self.cfg.history_frames <= 0:
            raise ValueError("TemporalOccupancyCfg.history_frames must be positive.")
        self.num_envs = env.num_envs
        self.device = env.device
        self.frame_size = self.cfg.grid_size * self.cfg.grid_size
        self._sample_steps = max(1, round(self.cfg.sample_period_s / env.physics_dt))
        period_on_grid = self._sample_steps * env.physics_dt
        if not math.isclose(self.cfg.sample_period_s, period_on_grid, abs_tol=1e-6):
            raise ValueError("TemporalOccupancyCfg.sample_period_s must lie on the physics-time grid.")

        self._physics_steps = 0
        self._last_capture_physics_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._capacity = self.cfg.history_frames + 1
        self._head = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frames = torch.zeros(self.num_envs, self._capacity, self.frame_size, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear selected histories and wait for post-reset physics-rate captures.

        This hook runs before the base environment commits the reset state to the
        simulator, so sampling here could retain a pre-reset sensor image.  The
        history is instead zero-padded until the first regular 0.5-second
        physics-grid capture.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._frames[env_ids] = 0.0
        self._head[env_ids] = 0
        self._count[env_ids] = 0
        self._last_capture_physics_step[env_ids] = self._physics_steps

    def on_physics_step(self) -> None:
        """Capture all environments every configured number of physics steps."""
        self._physics_steps += 1
        elapsed = self._physics_steps - self._last_capture_physics_step
        env_ids = (elapsed >= self._sample_steps).nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._capture(env_ids)
            self._last_capture_physics_step[env_ids] = self._physics_steps

    def history_frames(self) -> torch.Tensor:
        """Return chronological non-current frames with shape ``(N, H, grid_size**2)``."""
        ages = torch.arange(self.cfg.history_frames, 0, -1, device=self.device)
        slots = (self._head.unsqueeze(1) - ages.unsqueeze(0)) % self._capacity
        frames = self._frames.gather(1, slots.unsqueeze(-1).expand(-1, -1, self.frame_size))
        valid = self._count.unsqueeze(1) > ages.unsqueeze(0)
        return frames * valid.unsqueeze(-1)

    def history(self) -> torch.Tensor:
        """Return the flattened ``(N, H * grid_size**2)`` chronological non-current history."""
        return self.history_frames().reshape(self.num_envs, -1)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _capture(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._push(self._rasterize()[env_ids], env_ids)

    def _push(self, grids: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Append pre-rasterized frames. Kept small and deterministic for unit tests."""
        expected_shape = (env_ids.numel(), self.frame_size)
        if tuple(grids.shape) != expected_shape:
            raise ValueError(f"Expected occupancy frames with shape {expected_shape}, got {tuple(grids.shape)}.")
        slots = (self._head[env_ids] + 1) % self._capacity
        self._frames[env_ids, slots] = grids
        self._head[env_ids] = slots
        self._count[env_ids] += 1

    def _rasterize(self) -> torch.Tensor:
        """Rasterize the current ray-caster data with the baseline occupancy semantics."""
        sensor = self.env.scene.sensors[self.cfg.sensor_name]
        ray_hits = sensor.data.ray_hits_w
        sensor_pos = sensor.data.pos_w
        rel = ray_hits - sensor_pos.unsqueeze(1)
        dists = torch.norm(rel, dim=-1)
        valid = (
            (dists < self.cfg.max_range * 0.99)
            & (rel[..., 2] >= self.cfg.min_height_rel)
            & (rel[..., 2] <= self.cfg.max_height_rel)
        )
        half = self.cfg.grid_size // 2
        ix = (rel[..., 0] / self.cfg.grid_resolution + half).long()
        iy = (rel[..., 1] / self.cfg.grid_resolution + half).long()
        in_bounds = (ix >= 0) & (ix < self.cfg.grid_size) & (iy >= 0) & (iy < self.cfg.grid_size)
        valid &= in_bounds
        flat_idx = (ix * self.cfg.grid_size + iy).clamp(0, self.frame_size - 1)
        flat_idx = torch.where(valid, flat_idx, torch.zeros_like(flat_idx))
        grid = torch.zeros(self.num_envs, self.frame_size, dtype=torch.float32, device=ray_hits.device)
        grid.scatter_reduce_(1, flat_idx, valid.float(), reduce="amax", include_self=True)
        return grid


def temporal_occupancy_grid(env: ManagerBasedRLEnv, collector_name: str) -> torch.Tensor:
    """Observation term returning one named collector's six-frame occupancy tail."""
    collector = getattr(env, collector_name, None)
    if collector is None:
        raise RuntimeError(f"Temporal occupancy collector '{collector_name}' is not available on the environment.")
    return collector.history()
