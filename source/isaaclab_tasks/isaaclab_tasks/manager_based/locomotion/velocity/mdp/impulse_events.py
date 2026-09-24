"""Short, curriculum-gated external-wrench events for locomotion training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BaseWrenchImpulse(ManagerTermBase):
    """Apply short, off-centre horizontal base-force pulses.

    The event manager invokes this term every RL control step.  The term writes a
    permanent wrench only while a pulse is active, so the articulation applies it
    at every physics substep.  The wrench is explicitly cleared when the pulse
    expires and whenever an environment resets.

    Force and application point are expressed in the base-link frame.  Providing
    the point to :class:`~isaaclab.utils.wrench_composer.WrenchComposer` produces
    the physically consistent torque ``r x F`` instead of sampling an unrelated
    torque vector.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset = env.scene[self._asset_cfg.name]
        self._body_ids = self._asset_cfg.body_ids
        if isinstance(self._body_ids, slice) or len(self._body_ids) != 1:
            raise ValueError("BaseWrenchImpulse requires exactly one base body.")

        self._step_dt = float(env.step_dt)
        if self._step_dt <= 0.0:
            raise ValueError("BaseWrenchImpulse requires a positive environment step_dt.")

        num_envs = env.num_envs
        self._elapsed_time_s = torch.zeros(num_envs, device=env.device)
        self._time_to_next_pulse_s = torch.zeros(num_envs, device=env.device)
        self._remaining_pulse_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        # Retained for diagnostics and unit tests.  The composer owns the applied
        # wrench; these buffers describe the most recently scheduled pulse.
        self._last_force_b = torch.zeros(num_envs, 3, device=env.device)
        self._last_application_point_b = torch.zeros(num_envs, 3, device=env.device)
        self._last_torque_b = torch.zeros(num_envs, 3, device=env.device)
        self._last_impulse_ns = torch.zeros(num_envs, device=env.device)

        self._schedule_after_reset(torch.arange(num_envs, device=env.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear active wrenches and restart the per-environment schedule."""
        ids = self._as_ids(env_ids)
        if len(ids) == 0:
            return
        self._clear_wrench(ids)
        self._elapsed_time_s[ids] = 0.0
        self._remaining_pulse_steps[ids] = 0
        self._last_force_b[ids] = 0.0
        self._last_application_point_b[ids] = 0.0
        self._last_torque_b[ids] = 0.0
        self._last_impulse_ns[ids] = 0.0
        self._schedule_after_reset(ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        terrain_level_threshold: int,
        settle_time_s: float,
        command_speed_threshold: float,
        pulse_interval_range_s: tuple[float, float],
        pulse_duration_range_s: tuple[float, float],
        impulse_ranges_by_level: tuple[tuple[float, float], ...],
        application_point_range_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> None:
        """Advance pulse timers and apply/clear wrenches for ``env_ids``."""
        del asset_cfg  # Resolved once in __init__; retained for EventTermCfg validation.
        ids = self._as_ids(env_ids)
        if len(ids) == 0:
            return

        self._elapsed_time_s[ids] += self._step_dt
        self._time_to_next_pulse_s[ids] -= self._step_dt

        # A force set during the previous callback has just been applied through
        # one full RL step (and all of its physics substeps).  Clear it exactly
        # after the configured number of those steps.
        active_ids = ids[self._remaining_pulse_steps[ids] > 0]
        if len(active_ids) > 0:
            self._remaining_pulse_steps[active_ids] -= 1
            expired_ids = active_ids[self._remaining_pulse_steps[active_ids] == 0]
            if len(expired_ids) > 0:
                self._clear_wrench(expired_ids)

        terrain_levels = env.scene.terrain.terrain_levels[ids]
        commands = env.command_manager.get_command(command_name)[ids]
        moving = torch.linalg.vector_norm(commands[:, :2], dim=1) >= command_speed_threshold
        eligible = (
            (terrain_levels >= terrain_level_threshold)
            & moving
            & (self._elapsed_time_s[ids] >= settle_time_s)
            & (self._time_to_next_pulse_s[ids] <= 0.0)
            & (self._remaining_pulse_steps[ids] == 0)
        )
        pulse_ids = ids[eligible]
        if len(pulse_ids) == 0:
            return

        levels = terrain_levels[eligible]
        impulse_ranges = self._impulse_ranges_for_levels(levels, terrain_level_threshold, impulse_ranges_by_level)
        impulses = self._sample_uniform(impulse_ranges[:, 0], impulse_ranges[:, 1])
        durations = self._sample_uniform(
            torch.full_like(impulses, pulse_duration_range_s[0]),
            torch.full_like(impulses, pulse_duration_range_s[1]),
        )
        force_magnitudes = impulses / durations
        directions = self._sample_horizontal_directions(len(pulse_ids))
        forces_b = directions * force_magnitudes.unsqueeze(-1)
        points_b = self._sample_application_points(len(pulse_ids), application_point_range_m)

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_b.unsqueeze(1),
            positions=points_b.unsqueeze(1),
            body_ids=self._body_ids,
            env_ids=pulse_ids,
            is_global=False,
        )
        self._remaining_pulse_steps[pulse_ids] = torch.ceil(durations / self._step_dt).to(torch.long)
        self._time_to_next_pulse_s[pulse_ids] = self._sample_interval(len(pulse_ids), pulse_interval_range_s)
        self._last_force_b[pulse_ids] = forces_b
        self._last_application_point_b[pulse_ids] = points_b
        self._last_torque_b[pulse_ids] = torch.linalg.cross(points_b, forces_b)
        self._last_impulse_ns[pulse_ids] = impulses

    def _schedule_after_reset(self, env_ids: torch.Tensor) -> None:
        settle_time_s = float(self.cfg.params["settle_time_s"])
        interval_range_s = self.cfg.params["pulse_interval_range_s"]
        self._time_to_next_pulse_s[env_ids] = settle_time_s + self._sample_interval(len(env_ids), interval_range_s)

    def _clear_wrench(self, env_ids: torch.Tensor) -> None:
        zeros = torch.zeros((len(env_ids), 1, 3), device=self.device)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            body_ids=self._body_ids,
            env_ids=env_ids,
            is_global=False,
        )

    def _as_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None or env_ids == slice(None):
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _sample_interval(self, count: int, interval_range_s: tuple[float, float]) -> torch.Tensor:
        low, high = interval_range_s
        return self._sample_uniform(
            torch.full((count,), low, device=self.device), torch.full((count,), high, device=self.device)
        )

    def _sample_uniform(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return low + torch.rand_like(low) * (high - low)

    def _sample_horizontal_directions(self, count: int) -> torch.Tensor:
        angles = torch.rand(count, device=self.device) * (2.0 * math.pi)
        return torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1)

    def _sample_application_points(
        self,
        count: int,
        point_range_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> torch.Tensor:
        ranges = torch.tensor(point_range_m, dtype=torch.float32, device=self.device)
        return ranges[:, 0] + torch.rand((count, 3), device=self.device) * (ranges[:, 1] - ranges[:, 0])

    @staticmethod
    def _impulse_ranges_for_levels(
        levels: torch.Tensor,
        terrain_level_threshold: int,
        impulse_ranges_by_level: tuple[tuple[float, float], ...],
    ) -> torch.Tensor:
        if len(impulse_ranges_by_level) != 4:
            raise ValueError("Expected impulse ranges for terrain bands 5-6, 7, 8, and 9+.")
        band_indices = torch.clamp(levels - terrain_level_threshold - 1, min=0, max=3)
        # Level 5 and 6 share the first band.  Levels 7, 8, and 9+ use the
        # remaining bands respectively.
        band_indices = torch.where(
            levels <= terrain_level_threshold + 1, torch.zeros_like(band_indices), band_indices
        )
        ranges = torch.tensor(impulse_ranges_by_level, dtype=torch.float32, device=levels.device)
        return ranges[band_indices]
