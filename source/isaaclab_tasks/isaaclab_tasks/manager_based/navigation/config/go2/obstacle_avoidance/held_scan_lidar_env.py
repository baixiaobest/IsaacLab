"""Physics-rate collection of held lidar scans with optional sparse sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.navigation.lidar_geometry import forward_lidar_reflection_bins, world_to_body_xy


@configclass
class HeldScanLidarCfg:
    """Configuration for a lidar fan held between fixed-rate captures."""

    sensor_name: str = "obstacle_scanner"
    scan_period_s: float = 0.130
    max_distance: float = 20.0
    full_fan_ray_count: int = 256
    sparse_sampling_enabled: bool = False
    density_curriculum_enabled: bool = False
    target_coverage: float | None = None


LIDAR_COVERAGE_STAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, None)
LIDAR_DENSE_WARMUP_ITERATIONS = 500
LIDAR_STAGE_INTERVAL_ITERATIONS = 250


class HeldScanLidarCollector:
    """Hold full geometry scans, optionally sampling sparse policy reflections."""

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
        if self.cfg.sparse_sampling_enabled and self.num_rays != 256:
            raise ValueError("Sparse lidar sampling requires the 256-ray front fan.")
        if self.cfg.target_coverage is not None and not (0.0 < self.cfg.target_coverage <= 1.0):
            raise ValueError("HeldScanLidarCfg.target_coverage must be in (0, 1].")
        if self.cfg.density_curriculum_enabled and not self.cfg.sparse_sampling_enabled:
            raise ValueError("The lidar density curriculum requires sparse sampling.")

        self._density_stage = 0
        self._coverage_target = self.cfg.target_coverage
        self._episode_density_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._training_start_iteration = 0
        self._steps_per_iteration: int | None = None

        self._pending_hit_xy = torch.zeros(self.num_envs, self.num_rays, 2, device=self.device)
        self._pending_policy_hit_xy = torch.zeros_like(self._pending_hit_xy)
        self._pending_state = torch.zeros(self.num_envs, self.num_rays, dtype=torch.uint8, device=self.device)
        self._pending_policy_state = torch.zeros_like(self._pending_state)
        if self.cfg.sparse_sampling_enabled:
            self._sampling_pattern = torch.zeros(self.num_envs, 256, dtype=torch.bool, device=self.device)
            self._sampling_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._sampling_has_capture = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Mesh ids are optional on the base ray caster.  Data-collection tasks enable
        # them to identify which scene object produced each reflection.
        self._pending_ray_mesh_ids = torch.full(
            (self.num_envs, self.num_rays), -1, dtype=torch.int16, device=self.device
        )
        self._pending_ped_velocity_w: torch.Tensor | None = None
        self._pending_ego_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._pending_ego_yaw = torch.zeros(self.num_envs, device=self.device)
        self._pending_reference_time_s = torch.zeros(self.num_envs, device=self.device)
        self._capture_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        if getattr(getattr(self, "cfg", None), "sparse_sampling_enabled", False):
            self._reset_sampling_pattern(env_ids)
            if self.cfg.density_curriculum_enabled:
                self._episode_density_stage[env_ids] = self._density_stage
            self._pending_policy_state[env_ids] = 0
            self._pending_policy_hit_xy[env_ids] = 0
        self._capture_full_scan(env_ids)

    def _reset_sampling_pattern(self, env_ids: torch.Tensor) -> None:
        """Draw independent circular 360-degree templates and initial phases."""
        count = env_ids.numel()
        # Even the shortest possible groups occupy three cells, so 86 groups
        # suffice to cover all 256 cells without a per-environment Python loop.
        two_cell = torch.rand(count, 86, device=self.device) < 0.75
        gap_draw = torch.rand(count, 86, device=self.device)
        gaps = 2 + (gap_draw >= 0.10).long() + (gap_draw >= 0.50).long() + (gap_draw >= 0.90).long()
        widths = 1 + two_cell.long()
        lengths = widths + gaps
        starts = torch.cumsum(lengths, dim=1) - lengths
        pattern = torch.zeros(count, 256, dtype=torch.bool, device=self.device)
        rows = torch.arange(count, device=self.device)[:, None].expand_as(starts)
        first = starts < 256
        second = two_cell & (starts + 1 < 256)
        pattern[rows[first], starts[first]] = True
        pattern[rows[second], (starts + 1)[second]] = True
        if self._coverage_target == 1.0:
            pattern.fill_(True)
        elif self._coverage_target is not None:
            # Fill gaps once per episode, not independently at every capture.
            # The augmented 360-degree template shifts as a single pattern.
            desired = round(256 * self._coverage_target)
            extra = (desired - pattern.sum(dim=1)).clamp_min(0)
            scores = torch.rand(count, 256, device=self.device).masked_fill(pattern, float("inf"))
            ranks = scores.argsort(dim=1).argsort(dim=1)
            pattern |= ranks < extra[:, None]
        self._sampling_pattern[env_ids] = pattern
        self._sampling_phase[env_ids] = torch.randint(0, 256, (count,), device=self.device)
        self._sampling_has_capture[env_ids] = False

    def configure_density_schedule(self, start_iteration: int, steps_per_iteration: int) -> bool:
        """Anchor the schedule to RSL-RL's current iteration, including after resume.

        Returns whether the target stage changed. The trainer must refresh the
        collector and observation history after a resumed-stage change.
        """
        if not self.cfg.density_curriculum_enabled:
            raise RuntimeError("LiDAR density curriculum is disabled for this collector.")
        if start_iteration < 0 or steps_per_iteration <= 0:
            raise ValueError("LiDAR density schedule requires a non-negative iteration and positive rollout length.")
        self._training_start_iteration = start_iteration
        self._steps_per_iteration = steps_per_iteration
        return self.advance_density_schedule()

    def advance_density_schedule(self) -> bool:
        """Update the shared target from completed high-level rollout steps."""
        if not self.cfg.density_curriculum_enabled:
            raise RuntimeError("LiDAR density curriculum is disabled for this collector.")
        iteration = self._training_start_iteration
        if self._steps_per_iteration is not None:
            iteration += self.env.common_step_counter // self._steps_per_iteration
        stage = 0
        if iteration >= LIDAR_DENSE_WARMUP_ITERATIONS:
            stage = min(
                1 + (iteration - LIDAR_DENSE_WARMUP_ITERATIONS) // LIDAR_STAGE_INTERVAL_ITERATIONS,
                len(LIDAR_COVERAGE_STAGES) - 1,
            )
        if stage == self._density_stage:
            return False
        self._density_stage = stage
        self._coverage_target = LIDAR_COVERAGE_STAGES[stage]
        return True

    def density_status(self) -> dict[str, float]:
        """Expose the current coverage target to W&B."""
        return {
            "coverage_percent": 100.0 * (self._coverage_target if self._coverage_target is not None else 1.0 / 3.0),
        }

    def _sparse_ray_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Select one ray in each sampled front cell, shifting toward the left."""
        count = env_ids.numel()
        # Positive phase moves a template cell toward increasing sensor angle.
        # Front cells occupy [64, 192); indexing the full circle, rather than
        # rolling 128 front cells, brings new samples in from the right edge.
        front_cells = torch.arange(64, 192, device=self.device)
        template_cells = (front_cells[None, :] - self._sampling_phase[env_ids, None]) % 256
        active = self._sampling_pattern[env_ids].gather(1, template_cells.expand(count, -1))
        subray = torch.randint(0, 2, (count, 128), device=self.device)
        # The 256-ray fan includes +90 degrees, but the 128 policy bins are
        # half-open [-90, +90).  Ray 255 falls outside the last front bin.
        subray[:, -1] = 0
        ray_indices = 2 * torch.arange(128, device=self.device)[None, :] + subray
        mask = torch.zeros(count, self.num_rays, dtype=torch.bool, device=self.device)
        mask.scatter_(1, ray_indices.expand(count, -1), active)
        return mask

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
            "ray_mesh_ids": self._pending_ray_mesh_ids,
            "pedestrian_velocity_w": self._pending_ped_velocity_w,
            "capture_index": self._capture_index,
        }

    def latest_policy_capture(self) -> dict[str, torch.Tensor]:
        """Return the sparse held capture, with metadata aligned to the full scan."""
        capture = self.latest_capture()
        capture["hit_xy"] = getattr(self, "_pending_policy_hit_xy", self._pending_hit_xy)
        capture["ray_state"] = getattr(self, "_pending_policy_state", self._pending_state)
        return capture

    def consume_completed(self) -> dict[str, torch.Tensor] | None:
        """Return each queued policy scan once, leaving it held thereafter."""
        if not torch.any(self._pending_valid):
            return None
        env_ids = self._pending_valid.nonzero(as_tuple=False).squeeze(-1)
        self._pending_valid[env_ids] = False
        self._has_latest[env_ids] = True
        self._latest_reference_time_s[env_ids] = self._pending_reference_time_s[env_ids]
        return {
            "env_ids": env_ids,
            "hit_xy": getattr(self, "_pending_policy_hit_xy", self._pending_hit_xy)[env_ids],
            "ray_state": getattr(self, "_pending_policy_state", self._pending_state)[env_ids],
            "ego_xy": self._pending_ego_xy[env_ids],
            "ego_yaw": self._pending_ego_yaw[env_ids],
            "scan_age_s": self.scan_age_s()[env_ids],
            "ray_mesh_ids": self._pending_ray_mesh_ids[env_ids],
            "pedestrian_velocity_w": (
                self._pending_ped_velocity_w[env_ids] if self._pending_ped_velocity_w is not None else None
            ),
            "capture_index": self._capture_index[env_ids],
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
        if self.cfg.sparse_sampling_enabled:
            previously_captured = self._sampling_has_capture[env_ids]
            phase_draw = torch.rand(env_ids.numel(), device=self.device)
            phase_step = (phase_draw >= 0.10).long() + (phase_draw >= 0.85).long()
            self._sampling_phase[env_ids] = (
                self._sampling_phase[env_ids] + phase_step * previously_captured.long()
            ) % 256
            selected = self._sparse_ray_mask(env_ids)
            # The first curriculum stage is the original dense held scan. Use
            # the stage assigned at reset so in-flight episodes stay dense when
            # the global curriculum advances to 90%.
            if self.cfg.density_curriculum_enabled:
                dense_episode = self._episode_density_stage[env_ids] == 0
            else:
                dense_episode = torch.full(
                    (env_ids.numel(),), self._coverage_target == 1.0, dtype=torch.bool, device=self.device
                )
            selected[dense_episode] = True
            # A selected no-return ray is measured (state 1); only an
            # unselected ray is unavailable (state 0).
            self._pending_policy_state[env_ids] = torch.where(
                selected, self._pending_state[env_ids], 0
            )
            self._pending_policy_hit_xy[env_ids] = torch.where(
                selected.unsqueeze(-1), hit_xy[env_ids], free_endpoint[env_ids, :, :2]
            )
            self._sampling_has_capture[env_ids] = True
        else:
            self._pending_policy_state[env_ids] = self._pending_state[env_ids]
            self._pending_policy_hit_xy[env_ids] = hit_xy[env_ids]
        self._pending_ego_xy[env_ids] = pos_w[env_ids, :2]
        self._pending_ego_yaw[env_ids] = yaw[env_ids]
        self._pending_reference_time_s[env_ids] = self._time_s
        self._pending_valid[env_ids] = True
        self._capture_index[env_ids] += 1

        ray_mesh_ids = getattr(data, "ray_mesh_ids", None)
        if ray_mesh_ids is not None:
            self._pending_ray_mesh_ids[env_ids] = ray_mesh_ids[env_ids].squeeze(-1).to(torch.int16)
        else:
            self._pending_ray_mesh_ids[env_ids] = -1

        # The crowd is created after the collector during environment startup, so
        # allocate this optional buffer lazily on the first live capture.  Storing
        # velocity here, rather than when rollout observations are read, keeps the
        # pedestrian state synchronized with the LiDAR reflection time.
        crowd_manager = getattr(self.env, "crowd_manager", None)
        if crowd_manager is not None:
            velocity_w = crowd_manager.get_velocities()
            if self._pending_ped_velocity_w is None or self._pending_ped_velocity_w.shape != velocity_w.shape:
                self._pending_ped_velocity_w = torch.zeros_like(velocity_w)
            self._pending_ped_velocity_w[env_ids] = velocity_w[env_ids]


def iteration_lidar_density_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, float]:
    """Advance the training-only density target on the fixed iteration schedule."""
    collector = env._held_scan_lidar_collector
    collector.advance_density_schedule()
    return collector.density_status()


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
