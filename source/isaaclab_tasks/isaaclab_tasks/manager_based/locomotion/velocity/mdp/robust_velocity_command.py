"""Prior-based velocity commands for robust Go2 locomotion training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RobustVelocityCommandCfg(CommandTermCfg):
    """Direct body-twist curriculum with fixed priors and bounded updates.

    Terrain difficulty alone progresses through levels 0--4. At level 5,
    normal coupled commands begin making prior-relative updates, rapid small
    commands independently resample, sudden-change commands repeat more
    frequently, and full-stop samples become cruise-to-stop cycles. There is
    no command delay or filtering.
    """

    class_type: type | None = None
    asset_name: str = "robot"
    max_planar_speed: float = 1.5
    max_yaw_rate: float = 2.0
    planar_deadzone_mps: float = 0.10
    """Planar commands below this magnitude are emitted as a full stop."""
    yaw_deadzone_radps: float = 0.10
    """Yaw commands below this magnitude are emitted as zero."""
    normal_yaw_full_cap_speed_mps: float = 1.0
    """Planar speed through which normal/sudden commands retain ``max_yaw_rate``."""
    normal_yaw_cap_at_max_planar_speed_radps: float = 1.0
    """Normal/sudden yaw-rate cap at ``max_planar_speed``."""
    slow_coupled_turn_probability: float = 0.15
    slow_straight_probability: float = 0.0
    rapid_small_change_probability: float = 0.0
    rotate_in_place_probability: float = 0.15
    full_stop_probability: float = 0.10
    normal_coupled_motion_probability: float = 0.40
    sudden_change_probability: float = 0.20
    incremental_start_terrain_level: int = 5
    incremental_full_terrain_level: int = 9
    incremental_start_frequency_hz: float = 0.5
    incremental_full_frequency_hz: float = 2.0
    max_planar_delta_mps: float = 0.2
    max_yaw_delta_radps: float = 0.3
    rapid_small_change_start_terrain_level: int = 5
    rapid_small_change_interval_s: float = 0.5
    rapid_small_change_max_speed_mps: float = 0.5
    sudden_change_time_fraction: float = 0.5
    """Single sudden-change time fraction retained for terrain levels below 5."""
    sudden_change_start_terrain_level: int = 5
    sudden_change_start_interval_s: float = 5.0
    sudden_change_full_interval_s: float = 3.0
    stop_cycle_start_terrain_level: int = 5
    stop_cycle_cruise_duration_s: float = 4.0
    stop_cycle_planar_speed_range_mps: tuple[float, float] = (0.75, 1.5)
    stop_cycle_settle_planar_speed_mps: float = 0.10
    stop_cycle_settle_yaw_rate_radps: float = 0.10
    stop_cycle_settle_duration_s: float = 0.5
    stop_cycle_max_dwell_s: float = 3.0

    target_vel_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/robust_velocity_target"
    )
    """Marker for the commanded body-frame planar velocity."""
    measured_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/robust_velocity_measured"
    )
    """Marker for the measured body-frame planar velocity."""

    target_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    measured_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = RobustVelocityCommand
        if self.max_planar_speed <= 0.0 or self.max_yaw_rate <= 0.0:
            raise ValueError("Velocity limits must be positive.")
        if not 0.0 <= self.planar_deadzone_mps <= self.max_planar_speed:
            raise ValueError("planar_deadzone_mps must lie in [0, max_planar_speed].")
        if not 0.0 <= self.yaw_deadzone_radps <= self.max_yaw_rate:
            raise ValueError("yaw_deadzone_radps must lie in [0, max_yaw_rate].")
        if not 0.0 < self.normal_yaw_full_cap_speed_mps < self.max_planar_speed:
            raise ValueError("normal_yaw_full_cap_speed_mps must lie strictly within the planar-speed envelope.")
        if not 0.0 <= self.normal_yaw_cap_at_max_planar_speed_radps <= self.max_yaw_rate:
            raise ValueError("normal_yaw_cap_at_max_planar_speed_radps must lie in [0, max_yaw_rate].")
        mode_probability_sum = sum(
            (
                self.slow_coupled_turn_probability,
                self.slow_straight_probability,
                self.rapid_small_change_probability,
                self.rotate_in_place_probability,
                self.full_stop_probability,
                self.normal_coupled_motion_probability,
                self.sudden_change_probability,
            )
        )
        if any(
            probability < 0.0
            for probability in (
                self.slow_coupled_turn_probability,
                self.slow_straight_probability,
                self.rapid_small_change_probability,
                self.rotate_in_place_probability,
                self.full_stop_probability,
                self.normal_coupled_motion_probability,
                self.sudden_change_probability,
            )
        ) or not math.isclose(mode_probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("Mode probabilities must be nonnegative and sum to one.")
        if self.incremental_start_terrain_level < 0:
            raise ValueError("incremental_start_terrain_level must be nonnegative.")
        if self.incremental_full_terrain_level <= self.incremental_start_terrain_level:
            raise ValueError("incremental_full_terrain_level must exceed incremental_start_terrain_level.")
        if self.incremental_start_frequency_hz <= 0.0 or self.incremental_full_frequency_hz <= 0.0:
            raise ValueError("Incremental command frequencies must be positive.")
        if self.incremental_full_frequency_hz < self.incremental_start_frequency_hz:
            raise ValueError("Incremental command frequency must not decrease with terrain level.")
        if self.max_planar_delta_mps < 0.0 or self.max_yaw_delta_radps < 0.0:
            raise ValueError("Incremental command deltas must be nonnegative.")
        if self.rapid_small_change_start_terrain_level < 0:
            raise ValueError("rapid_small_change_start_terrain_level must be nonnegative.")
        if self.rapid_small_change_interval_s <= 0.0:
            raise ValueError("rapid_small_change_interval_s must be positive.")
        if not 0.10 < self.rapid_small_change_max_speed_mps <= self.max_planar_speed:
            raise ValueError("rapid_small_change_max_speed_mps must lie in (0.10, max_planar_speed].")
        if not 0.0 < self.sudden_change_time_fraction < 1.0:
            raise ValueError("sudden_change_time_fraction must be strictly between zero and one.")
        if self.sudden_change_start_terrain_level < 0:
            raise ValueError("sudden_change_start_terrain_level must be nonnegative.")
        if self.sudden_change_start_terrain_level >= self.incremental_full_terrain_level:
            raise ValueError("sudden_change_start_terrain_level must precede incremental_full_terrain_level.")
        if self.sudden_change_start_interval_s <= 0.0 or self.sudden_change_full_interval_s <= 0.0:
            raise ValueError("Sudden-change intervals must be positive.")
        if self.sudden_change_full_interval_s > self.sudden_change_start_interval_s:
            raise ValueError("Sudden-change interval must not increase with terrain level.")
        if self.stop_cycle_start_terrain_level < 0:
            raise ValueError("stop_cycle_start_terrain_level must be nonnegative.")
        if self.stop_cycle_cruise_duration_s <= 0.0:
            raise ValueError("stop_cycle_cruise_duration_s must be positive.")
        stop_speed_low, stop_speed_high = self.stop_cycle_planar_speed_range_mps
        if not 0.0 < stop_speed_low <= stop_speed_high <= self.max_planar_speed:
            raise ValueError("stop_cycle_planar_speed_range_mps must lie in (0, max_planar_speed].")
        if self.stop_cycle_settle_planar_speed_mps < 0.0 or self.stop_cycle_settle_yaw_rate_radps < 0.0:
            raise ValueError("Stop-cycle settle velocity thresholds must be nonnegative.")
        if self.stop_cycle_settle_duration_s <= 0.0 or self.stop_cycle_max_dwell_s <= 0.0:
            raise ValueError("Stop-cycle settle duration and dwell cap must be positive.")


class RobustVelocityCommand(CommandTerm):
    """Prior-command velocity curriculum without latency, smoothing, or AR noise."""

    SLOW_COUPLED_TURN = 0
    ROTATE_IN_PLACE = 1
    NORMAL_COUPLED_MOTION = 2
    FULL_STOP = 3
    SUDDEN_CHANGE = 4
    SLOW_STRAIGHT = 5
    RAPID_SMALL_CHANGE = 6

    def __init__(self, cfg: RobustVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._target_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._emitted_command = torch.zeros_like(self._target_command)
        self._command = torch.zeros_like(self._target_command)
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sudden_change_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Full-stop mode becomes a stateful high-speed cruise -> zero-command
        # cycle on difficult terrain.  Below that gate it remains the original
        # stationary command sampled at reset.
        self._stop_cycle_braking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stop_cycle_settled_time_s = torch.zeros(self.num_envs, device=self.device)
        self._stop_cycle_dwell_time_s = torch.zeros(self.num_envs, device=self.device)

        self._planar_error_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._yaw_error_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._stop_planar_speed_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._stop_yaw_rate_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._planar_active_steps = torch.zeros(self.num_envs, device=self.device)
        self._yaw_active_steps = torch.zeros(self.num_envs, device=self.device)
        self._stop_steps = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_planar_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_planar_rms_mps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_yaw_rms_radps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_planar_rms_mps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_yaw_rms_radps"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Bounded current prior supplied to the policy and tracking rewards."""
        return self._command

    @property
    def target_command(self) -> torch.Tensor:
        """Current prior target, retained for debugging and evaluation traces."""
        return self._target_command

    @property
    def emitted_command(self) -> torch.Tensor:
        """Bounded command emitted immediately to the policy; equal to ``command``."""
        return self._emitted_command

    @property
    def mode(self) -> torch.Tensor:
        return self._mode

    def __str__(self) -> str:
        return (
            "RobustVelocityCommand:\n"
            "\tCommand dimension: (3,)\n"
            f"\tPlanar speed limit: {self.cfg.max_planar_speed} m/s\n"
            f"\tYaw-rate limit: {self.cfg.max_yaw_rate} rad/s\n"
            f"\tNormal update frequency: {self.cfg.incremental_start_frequency_hz}--"
            f"{self.cfg.incremental_full_frequency_hz} Hz (levels "
            f"{self.cfg.incremental_start_terrain_level}--{self.cfg.incremental_full_terrain_level})"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._target_command[env_ids] = 0.0
        self._emitted_command[env_ids] = 0.0
        self._command[env_ids] = 0.0
        self._mode[env_ids] = self.FULL_STOP
        self._sudden_change_fired[env_ids] = False
        self._stop_cycle_braking[env_ids] = False
        self._stop_cycle_settled_time_s[env_ids] = 0.0
        self._stop_cycle_dwell_time_s[env_ids] = 0.0
        extras = super().reset(env_ids)
        self._planar_error_sq_sum[env_ids] = 0.0
        self._yaw_error_sq_sum[env_ids] = 0.0
        self._stop_planar_speed_sq_sum[env_ids] = 0.0
        self._stop_yaw_rate_sq_sum[env_ids] = 0.0
        self._planar_active_steps[env_ids] = 0.0
        self._yaw_active_steps[env_ids] = 0.0
        self._stop_steps[env_ids] = 0.0
        return extras

    def _terrain_levels(self, env_ids: torch.Tensor) -> torch.Tensor:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_levels"):
            return torch.full(
                (len(env_ids),), self.cfg.incremental_full_terrain_level, dtype=torch.long, device=self.device
            )
        return terrain.terrain_levels[env_ids].to(device=self.device, dtype=torch.long)

    def _normal_update_period_s(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return level-interpolated normal-command update periods in seconds."""
        levels = self._terrain_levels(env_ids).float()
        alpha = (
            (levels - self.cfg.incremental_start_terrain_level)
            / float(self.cfg.incremental_full_terrain_level - self.cfg.incremental_start_terrain_level)
        ).clamp(0.0, 1.0)
        frequency = self.cfg.incremental_start_frequency_hz + alpha * (
            self.cfg.incremental_full_frequency_hz - self.cfg.incremental_start_frequency_hz
        )
        return frequency.reciprocal()

    def _sudden_change_period_s(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the level-interpolated repeated sudden-change interval."""
        levels = self._terrain_levels(env_ids).float()
        alpha = (
            (levels - self.cfg.sudden_change_start_terrain_level)
            / float(self.cfg.incremental_full_terrain_level - self.cfg.sudden_change_start_terrain_level)
        ).clamp(0.0, 1.0)
        return self.cfg.sudden_change_start_interval_s + alpha * (
            self.cfg.sudden_change_full_interval_s - self.cfg.sudden_change_start_interval_s
        )

    def _stop_cycle_enabled(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._terrain_levels(env_ids) >= self.cfg.stop_cycle_start_terrain_level

    def _sample_stop_cycle_cruise_targets(self, count: int) -> torch.Tensor:
        return self._sample_normal_coupled_targets(
            count,
            self.device,
            self.cfg.max_planar_speed,
            self.cfg.max_yaw_rate,
            min_speed_mps=self.cfg.stop_cycle_planar_speed_range_mps[0],
            max_speed_mps=self.cfg.stop_cycle_planar_speed_range_mps[1],
            normal_yaw_full_cap_speed_mps=self.cfg.normal_yaw_full_cap_speed_mps,
            normal_yaw_cap_at_max_planar_speed_radps=self.cfg.normal_yaw_cap_at_max_planar_speed_radps,
        )

    def _start_stop_cycle_cruise(self, env_ids: torch.Tensor) -> None:
        """Start a high-speed cruise phase for the specified stop-cycle environments."""
        if len(env_ids) == 0:
            return
        self._target_command[env_ids] = self._sample_stop_cycle_cruise_targets(len(env_ids))
        self._stop_cycle_braking[env_ids] = False
        self._stop_cycle_settled_time_s[env_ids] = 0.0
        self._stop_cycle_dwell_time_s[env_ids] = 0.0
        self.time_left[env_ids] = self.cfg.stop_cycle_cruise_duration_s

    def _resample(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        initial = self.command_counter[env_ids] == 0
        if torch.any(initial):
            initial_ids = env_ids[initial]
            commands, modes = self.sample_direct_targets(
                len(initial_ids),
                self.device,
                self.cfg.max_planar_speed,
                self.cfg.max_yaw_rate,
                normal_yaw_full_cap_speed_mps=self.cfg.normal_yaw_full_cap_speed_mps,
                normal_yaw_cap_at_max_planar_speed_radps=self.cfg.normal_yaw_cap_at_max_planar_speed_radps,
                slow_coupled_turn_probability=self.cfg.slow_coupled_turn_probability,
                slow_straight_probability=self.cfg.slow_straight_probability,
                rapid_small_change_probability=self.cfg.rapid_small_change_probability,
                rotate_in_place_probability=self.cfg.rotate_in_place_probability,
                full_stop_probability=self.cfg.full_stop_probability,
                normal_coupled_motion_probability=self.cfg.normal_coupled_motion_probability,
                sudden_change_probability=self.cfg.sudden_change_probability,
            )
            self._target_command[initial_ids] = commands
            self._mode[initial_ids] = modes
            rapid_ids = initial_ids[modes == self.RAPID_SMALL_CHANGE]
            if len(rapid_ids) > 0:
                active_ids = rapid_ids[
                    self._terrain_levels(rapid_ids) >= self.cfg.rapid_small_change_start_terrain_level
                ]
                self._target_command[active_ids] = self._sample_small_straight_targets(
                    len(active_ids), self.device, self.cfg.rapid_small_change_max_speed_mps
                )
            self._sudden_change_fired[initial_ids] = False
            self._set_next_resample_time(initial_ids)
            full_stop_ids = initial_ids[modes == self.FULL_STOP]
            if len(full_stop_ids) > 0:
                cycle_ids = full_stop_ids[self._stop_cycle_enabled(full_stop_ids)]
                self._start_stop_cycle_cruise(cycle_ids)
        if torch.any(~initial):
            self._resample_existing_priors(env_ids[~initial])
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Satisfy the CommandTerm contract; scheduling is owned by ``_resample``.

        ``CommandTerm`` declares this hook abstract, but the prior-command
        curriculum must choose both the next target and its hold time together.
        Its overridden ``_resample`` therefore performs the complete operation.
        """
        return

    def compute(self, dt: float) -> None:
        """Advance the stop-settling timer before processing command transitions."""
        braking = (self._mode == self.FULL_STOP) & self._stop_cycle_braking
        if torch.any(braking):
            planar_speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)
            yaw_rate = torch.abs(self.robot.data.root_ang_vel_b[:, 2])
            settled = (planar_speed < self.cfg.stop_cycle_settle_planar_speed_mps) & (
                yaw_rate < self.cfg.stop_cycle_settle_yaw_rate_radps
            )
            settled_braking = braking & settled
            unsettled_braking = braking & ~settled
            self._stop_cycle_settled_time_s[settled_braking] += dt
            self._stop_cycle_settled_time_s[unsettled_braking] = 0.0
            self._stop_cycle_dwell_time_s[braking] += dt
        super().compute(dt)

    def _set_next_resample_time(self, env_ids: torch.Tensor) -> None:
        self.time_left[env_ids] = float("inf")
        modes = self._mode[env_ids]
        rapid_ids = env_ids[modes == self.RAPID_SMALL_CHANGE]
        if len(rapid_ids) > 0:
            active_ids = rapid_ids[self._terrain_levels(rapid_ids) >= self.cfg.rapid_small_change_start_terrain_level]
            self.time_left[active_ids] = self.cfg.rapid_small_change_interval_s
        sudden = modes == self.SUDDEN_CHANGE
        if torch.any(sudden):
            sudden_ids = env_ids[sudden]
            repeated = self._terrain_levels(sudden_ids) >= self.cfg.sudden_change_start_terrain_level
            if torch.any(repeated):
                repeated_ids = sudden_ids[repeated]
                self.time_left[repeated_ids] = self._sudden_change_period_s(repeated_ids)
            if torch.any(~repeated):
                self.time_left[sudden_ids[~repeated]] = (
                    self._env.max_episode_length_s * self.cfg.sudden_change_time_fraction
                )
        normal = modes == self.NORMAL_COUPLED_MOTION
        if torch.any(normal):
            normal_ids = env_ids[normal]
            updates_enabled = self._terrain_levels(normal_ids) >= self.cfg.incremental_start_terrain_level
            if torch.any(updates_enabled):
                update_ids = normal_ids[updates_enabled]
                self.time_left[update_ids] = self._normal_update_period_s(update_ids)

    def _resample_existing_priors(self, env_ids: torch.Tensor) -> None:
        modes = self._mode[env_ids]
        self.time_left[env_ids] = float("inf")
        rapid_ids = env_ids[modes == self.RAPID_SMALL_CHANGE]
        if len(rapid_ids) > 0:
            active_ids = rapid_ids[self._terrain_levels(rapid_ids) >= self.cfg.rapid_small_change_start_terrain_level]
            self._target_command[active_ids] = self._sample_small_straight_targets(
                len(active_ids), self.device, self.cfg.rapid_small_change_max_speed_mps
            )
            self.time_left[active_ids] = self.cfg.rapid_small_change_interval_s
        sudden_modes = modes == self.SUDDEN_CHANGE
        sudden_ids = env_ids[sudden_modes]
        repeated_sudden = torch.zeros(len(sudden_ids), dtype=torch.bool, device=self.device)
        if len(sudden_ids) > 0:
            repeated_sudden = self._terrain_levels(sudden_ids) >= self.cfg.sudden_change_start_terrain_level
        sudden = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if len(sudden_ids) > 0:
            sudden[sudden_modes] = repeated_sudden | ~self._sudden_change_fired[sudden_ids]
        if torch.any(sudden):
            change_ids = env_ids[sudden]
            self._target_command[change_ids] = self._sample_normal_coupled_targets(
                len(change_ids),
                self.device,
                self.cfg.max_planar_speed,
                self.cfg.max_yaw_rate,
                normal_yaw_full_cap_speed_mps=self.cfg.normal_yaw_full_cap_speed_mps,
                normal_yaw_cap_at_max_planar_speed_radps=self.cfg.normal_yaw_cap_at_max_planar_speed_radps,
            )
            self._sudden_change_fired[change_ids] = True
        if len(sudden_ids) > 0 and torch.any(repeated_sudden):
            repeated_ids = sudden_ids[repeated_sudden]
            self.time_left[repeated_ids] = self._sudden_change_period_s(repeated_ids)
        normal = modes == self.NORMAL_COUPLED_MOTION
        if torch.any(normal):
            normal_ids = env_ids[normal]
            updates_enabled = self._terrain_levels(normal_ids) >= self.cfg.incremental_start_terrain_level
            if torch.any(updates_enabled):
                update_ids = normal_ids[updates_enabled]
                self._target_command[update_ids] = self._sample_prior_relative_updates(self._target_command[update_ids])
                self.time_left[update_ids] = self._normal_update_period_s(update_ids)
        full_stop_ids = env_ids[modes == self.FULL_STOP]
        if len(full_stop_ids) > 0:
            cycle_ids = full_stop_ids[self._stop_cycle_enabled(full_stop_ids)]
            if len(cycle_ids) > 0:
                cruise_ids = cycle_ids[~self._stop_cycle_braking[cycle_ids]]
                if len(cruise_ids) > 0:
                    self._target_command[cruise_ids] = 0.0
                    self._stop_cycle_braking[cruise_ids] = True
                    self._stop_cycle_settled_time_s[cruise_ids] = 0.0
                    self._stop_cycle_dwell_time_s[cruise_ids] = 0.0
                    self.time_left[cruise_ids] = self._env.step_dt
                brake_ids = cycle_ids[self._stop_cycle_braking[cycle_ids]]
                if len(brake_ids) > 0:
                    ready = (self._stop_cycle_settled_time_s[brake_ids] >= self.cfg.stop_cycle_settle_duration_s) | (
                        self._stop_cycle_dwell_time_s[brake_ids] >= self.cfg.stop_cycle_max_dwell_s
                    )
                    if torch.any(ready):
                        self._start_stop_cycle_cruise(brake_ids[ready])
                    if torch.any(~ready):
                        self.time_left[brake_ids[~ready]] = self._env.step_dt

    @classmethod
    def _sample_normal_coupled_targets(
        cls,
        count: int,
        device: torch.device | str,
        max_planar_speed: float,
        max_yaw_rate: float,
        *,
        min_speed_mps: float | None = None,
        max_speed_mps: float | None = None,
        normal_yaw_full_cap_speed_mps: float = 1.0,
        normal_yaw_cap_at_max_planar_speed_radps: float = 1.0,
    ) -> torch.Tensor:
        command = torch.zeros(count, 3, device=device)
        if count == 0:
            return command
        lower_speed = min(0.25, max_planar_speed) if min_speed_mps is None else min_speed_mps
        upper_speed = max_planar_speed if max_speed_mps is None else max_speed_mps
        if not 0.0 < lower_speed <= upper_speed <= max_planar_speed:
            raise ValueError("Coupled planar-speed range must lie in (0, max_planar_speed].")
        speed = torch.empty(count, device=device).uniform_(lower_speed, upper_speed)
        angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
        command[:, 0] = speed * torch.cos(angle)
        command[:, 1] = speed * torch.sin(angle)
        speed_alpha = (
            (speed - normal_yaw_full_cap_speed_mps)
            / (max_planar_speed - normal_yaw_full_cap_speed_mps)
        ).clamp(0.0, 1.0)
        yaw_cap = max_yaw_rate + speed_alpha * (normal_yaw_cap_at_max_planar_speed_radps - max_yaw_rate)
        command[:, 2] = torch.empty(count, device=device).uniform_(-1.0, 1.0) * yaw_cap
        return command

    @staticmethod
    def _sample_small_straight_targets(count: int, device: torch.device | str, max_speed_mps: float) -> torch.Tensor:
        command = torch.zeros(count, 3, device=device)
        if count:
            speed = torch.empty(count, device=device).uniform_(0.10, max_speed_mps)
            angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
            command[:, 0] = speed * torch.cos(angle)
            command[:, 1] = speed * torch.sin(angle)
        return command

    @classmethod
    def sample_direct_targets(
        cls,
        count: int,
        device: torch.device | str,
        max_planar_speed: float = 1.5,
        max_yaw_rate: float = 2.0,
        *,
        normal_yaw_full_cap_speed_mps: float = 1.0,
        normal_yaw_cap_at_max_planar_speed_radps: float = 1.0,
        slow_coupled_turn_probability: float = 0.15,
        slow_straight_probability: float = 0.0,
        rapid_small_change_probability: float = 0.0,
        rotate_in_place_probability: float = 0.15,
        full_stop_probability: float = 0.10,
        normal_coupled_motion_probability: float = 0.40,
        sudden_change_probability: float = 0.20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the configured slow/rotate/stop/normal/sudden mixture."""
        if count <= 0:
            return torch.empty(0, 3, device=device), torch.empty(0, dtype=torch.long, device=device)
        probabilities = (
            slow_coupled_turn_probability,
            slow_straight_probability,
            rapid_small_change_probability,
            rotate_in_place_probability,
            full_stop_probability,
            normal_coupled_motion_probability,
            sudden_change_probability,
        )
        if any(probability < 0.0 for probability in probabilities) or not math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise ValueError("Mode probabilities must be nonnegative and sum to one.")
        if not 0.0 < normal_yaw_full_cap_speed_mps < max_planar_speed:
            raise ValueError("normal_yaw_full_cap_speed_mps must lie strictly within the planar-speed envelope.")
        if not 0.0 <= normal_yaw_cap_at_max_planar_speed_radps <= max_yaw_rate:
            raise ValueError("normal_yaw_cap_at_max_planar_speed_radps must lie in [0, max_yaw_rate].")
        command = torch.zeros(count, 3, device=device)
        mode = torch.empty(count, dtype=torch.long, device=device)
        mixture = torch.rand(count, device=device)
        slow_end = slow_coupled_turn_probability
        straight_end = slow_end + slow_straight_probability
        rapid_end = straight_end + rapid_small_change_probability
        rotate_end = rapid_end + rotate_in_place_probability
        stop_end = rotate_end + full_stop_probability
        normal_end = stop_end + normal_coupled_motion_probability
        slow = mixture < slow_end
        slow_straight = (mixture >= slow_end) & (mixture < straight_end)
        rapid_small = (mixture >= straight_end) & (mixture < rapid_end)
        rotate = (mixture >= rapid_end) & (mixture < rotate_end)
        stop = (mixture >= rotate_end) & (mixture < stop_end)
        normal = (mixture >= stop_end) & (mixture < normal_end)
        # The validated unit sum makes the remaining interval precisely the
        # sudden-change probability, while also covering round-off at 1.0.
        sudden = mixture >= normal_end
        slow_count = int(slow.sum().item())
        if slow_count:
            speed = torch.empty(slow_count, device=device).uniform_(0.05, min(0.35, max_planar_speed))
            angle = torch.empty(slow_count, device=device).uniform_(-math.pi, math.pi)
            command[slow, 0] = speed * torch.cos(angle)
            command[slow, 1] = speed * torch.sin(angle)
            command[slow, 2] = torch.empty(slow_count, device=device).uniform_(-max_yaw_rate, max_yaw_rate)
        straight_count = int((slow_straight | rapid_small).sum().item())
        if straight_count:
            command[slow_straight | rapid_small] = cls._sample_small_straight_targets(
                straight_count, device, 0.25
            )
        rotate_count = int(rotate.sum().item())
        if rotate_count:
            command[rotate, 2] = torch.empty(rotate_count, device=device).uniform_(-max_yaw_rate, max_yaw_rate)
        coupled = normal | sudden
        if torch.any(coupled):
            command[coupled] = cls._sample_normal_coupled_targets(
                int(coupled.sum().item()),
                device,
                max_planar_speed,
                max_yaw_rate,
                normal_yaw_full_cap_speed_mps=normal_yaw_full_cap_speed_mps,
                normal_yaw_cap_at_max_planar_speed_radps=normal_yaw_cap_at_max_planar_speed_radps,
            )
        mode[slow] = cls.SLOW_COUPLED_TURN
        mode[slow_straight] = cls.SLOW_STRAIGHT
        mode[rapid_small] = cls.RAPID_SMALL_CHANGE
        mode[rotate] = cls.ROTATE_IN_PLACE
        mode[stop] = cls.FULL_STOP
        mode[normal] = cls.NORMAL_COUPLED_MOTION
        mode[sudden] = cls.SUDDEN_CHANGE
        return command, mode

    def _sample_prior_relative_updates(self, prior: torch.Tensor) -> torch.Tensor:
        count = prior.shape[0]
        delta_magnitude = torch.empty(count, device=self.device).uniform_(0.0, self.cfg.max_planar_delta_mps)
        delta_angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        updated = prior.clone()
        updated[:, 0] += delta_magnitude * torch.cos(delta_angle)
        updated[:, 1] += delta_magnitude * torch.sin(delta_angle)
        updated[:, 2] += torch.empty(count, device=self.device).uniform_(
            -self.cfg.max_yaw_delta_radps, self.cfg.max_yaw_delta_radps
        )
        return self._bound_command(updated)

    def _bound_command(self, command: torch.Tensor) -> torch.Tensor:
        bounded = command.clone()
        planar_norm = torch.linalg.vector_norm(bounded[:, :2], dim=-1, keepdim=True)
        bounded[:, :2] *= torch.clamp(self.cfg.max_planar_speed / planar_norm.clamp_min(1.0e-8), max=1.0)
        bounded[:, 2].clamp_(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        stationary_planar = planar_norm.squeeze(-1) < self.cfg.planar_deadzone_mps
        bounded[stationary_planar, :2] = 0.0
        stationary_yaw = torch.abs(bounded[:, 2]) < self.cfg.yaw_deadzone_radps
        bounded[stationary_yaw, 2] = 0.0
        return bounded

    def _update_metrics(self) -> None:
        asset = self._env.scene[self.cfg.asset_name]
        measured_planar = asset.data.root_lin_vel_b[:, :2]
        measured_yaw = asset.data.root_ang_vel_b[:, 2]
        commanded_planar = self._command[:, :2]
        commanded_yaw = self._command[:, 2]
        planar_active = torch.linalg.vector_norm(commanded_planar, dim=-1) > 0.10
        yaw_active = torch.abs(commanded_yaw) > 0.10
        stopped = ~(planar_active | yaw_active)
        self._planar_error_sq_sum += torch.sum(torch.square(commanded_planar - measured_planar), dim=-1) * planar_active
        self._yaw_error_sq_sum += torch.square(commanded_yaw - measured_yaw) * yaw_active
        self._stop_planar_speed_sq_sum += torch.sum(torch.square(measured_planar), dim=-1) * stopped
        self._stop_yaw_rate_sq_sum += torch.square(measured_yaw) * stopped
        self._planar_active_steps += planar_active
        self._yaw_active_steps += yaw_active
        self._stop_steps += stopped
        self.metrics["tracking_planar_rms_mps"][:] = torch.sqrt(self._planar_error_sq_sum / self._planar_active_steps.clamp_min(1.0))
        self.metrics["tracking_yaw_rms_radps"][:] = torch.sqrt(self._yaw_error_sq_sum / self._yaw_active_steps.clamp_min(1.0))
        self.metrics["stop_planar_rms_mps"][:] = torch.sqrt(self._stop_planar_speed_sq_sum / self._stop_steps.clamp_min(1.0))
        self.metrics["stop_yaw_rms_radps"][:] = torch.sqrt(self._stop_yaw_rate_sq_sum / self._stop_steps.clamp_min(1.0))
        self.metrics["target_planar_speed"][:] = torch.linalg.vector_norm(self._target_command[:, :2], dim=-1)
        self.metrics["target_yaw_rate"][:] = torch.abs(self._target_command[:, 2])

    def episode_tracking_metrics(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return pre-reset episode metrics for binary terrain progression."""
        return {
            "planar_rms_mps": torch.sqrt(self._planar_error_sq_sum[env_ids] / self._planar_active_steps[env_ids].clamp_min(1.0)),
            "yaw_rms_radps": torch.sqrt(self._yaw_error_sq_sum[env_ids] / self._yaw_active_steps[env_ids].clamp_min(1.0)),
            "stop_planar_rms_mps": torch.sqrt(self._stop_planar_speed_sq_sum[env_ids] / self._stop_steps[env_ids].clamp_min(1.0)),
            "stop_yaw_rms_radps": torch.sqrt(self._stop_yaw_rate_sq_sum[env_ids] / self._stop_steps[env_ids].clamp_min(1.0)),
            "planar_active_steps": self._planar_active_steps[env_ids],
            "yaw_active_steps": self._yaw_active_steps[env_ids],
            "stop_steps": self._stop_steps[env_ids],
        }

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Show target and measured body-frame velocity, at separate heights."""
        if debug_vis:
            if not hasattr(self, "target_vel_visualizer"):
                self.target_vel_visualizer = VisualizationMarkers(self.cfg.target_vel_visualizer_cfg)
                self.measured_vel_visualizer = VisualizationMarkers(self.cfg.measured_vel_visualizer_cfg)
            self.target_vel_visualizer.set_visibility(True)
            self.measured_vel_visualizer.set_visibility(True)
        elif hasattr(self, "target_vel_visualizer"):
            self.target_vel_visualizer.set_visibility(False)
            self.measured_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w
        target_pos_w = base_pos_w.clone()
        measured_pos_w = base_pos_w.clone()
        target_pos_w[:, 2] += 0.70
        measured_pos_w[:, 2] += 0.45
        target_scale, target_quat = self._resolve_xy_velocity_to_arrow(self._target_command[:, :2])
        measured_scale, measured_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.target_vel_visualizer.visualize(target_pos_w, target_quat, target_scale)
        self.measured_vel_visualizer.visualize(measured_pos_w, measured_quat, measured_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.target_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.vector_norm(xy_velocity, dim=-1) * 3.0
        heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        arrow_quat_b = math_utils.quat_from_euler_xyz(torch.zeros_like(heading), torch.zeros_like(heading), heading)
        return arrow_scale, math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat_b)

    def _update_command(self) -> None:
        self._emitted_command[:] = self._bound_command(self._target_command)
        self._command[:] = self._emitted_command


@configclass
class ScriptedVelocityCommandCfg(RobustVelocityCommandCfg):
    """Evaluation-only source that sends bounded scripted targets immediately."""

    class_type: type | None = None

    def __post_init__(self) -> None:
        self.class_type = ScriptedVelocityCommand
        super().__post_init__()


class ScriptedVelocityCommand(RobustVelocityCommand):
    """Externally driven zero-delay counterpart of :class:`RobustVelocityCommand`."""

    cfg: ScriptedVelocityCommandCfg

    def _resample(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.time_left[env_ids] = float("inf")
        self.command_counter[env_ids] += 1

    def set_target_commands(self, commands: torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
        """Set unfiltered body-twist targets for selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        commands = torch.as_tensor(commands, device=self.device, dtype=self._target_command.dtype)
        if commands.shape != (len(env_ids), 3):
            raise ValueError(f"Expected command shape ({len(env_ids)}, 3), received {tuple(commands.shape)}.")
        self._target_command[env_ids] = commands

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        return
