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
    normal coupled commands begin making prior-relative updates; the update
    rate reaches 2 Hz at level 9. There is no command delay or filtering.
    """

    class_type: type | None = None
    asset_name: str = "robot"
    max_planar_speed: float = 1.5
    max_yaw_rate: float = 2.0
    incremental_start_terrain_level: int = 5
    incremental_full_terrain_level: int = 9
    incremental_start_frequency_hz: float = 0.5
    incremental_full_frequency_hz: float = 2.0
    max_planar_delta_mps: float = 0.2
    max_yaw_delta_radps: float = 0.3
    sudden_change_time_fraction: float = 0.5

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
        if not 0.0 < self.sudden_change_time_fraction < 1.0:
            raise ValueError("sudden_change_time_fraction must be strictly between zero and one.")


class RobustVelocityCommand(CommandTerm):
    """Prior-command velocity curriculum without latency, smoothing, or AR noise."""

    SLOW_COUPLED_TURN = 0
    ROTATE_IN_PLACE = 1
    NORMAL_COUPLED_MOTION = 2
    FULL_STOP = 3
    SUDDEN_CHANGE = 4

    def __init__(self, cfg: RobustVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._target_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._emitted_command = torch.zeros_like(self._target_command)
        self._command = torch.zeros_like(self._target_command)
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sudden_change_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

    def _resample(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        initial = self.command_counter[env_ids] == 0
        if torch.any(initial):
            initial_ids = env_ids[initial]
            commands, modes = self.sample_direct_targets(
                len(initial_ids), self.device, self.cfg.max_planar_speed, self.cfg.max_yaw_rate
            )
            self._target_command[initial_ids] = commands
            self._mode[initial_ids] = modes
            self._sudden_change_fired[initial_ids] = False
            self._set_next_resample_time(initial_ids)
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

    def _set_next_resample_time(self, env_ids: torch.Tensor) -> None:
        self.time_left[env_ids] = float("inf")
        modes = self._mode[env_ids]
        sudden = modes == self.SUDDEN_CHANGE
        if torch.any(sudden):
            self.time_left[env_ids[sudden]] = self._env.max_episode_length_s * self.cfg.sudden_change_time_fraction
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
        sudden = (modes == self.SUDDEN_CHANGE) & ~self._sudden_change_fired[env_ids]
        if torch.any(sudden):
            sudden_ids = env_ids[sudden]
            self._target_command[sudden_ids] = self._sample_normal_coupled_targets(
                len(sudden_ids), self.device, self.cfg.max_planar_speed, self.cfg.max_yaw_rate
            )
            self._sudden_change_fired[sudden_ids] = True
        normal = modes == self.NORMAL_COUPLED_MOTION
        if torch.any(normal):
            normal_ids = env_ids[normal]
            updates_enabled = self._terrain_levels(normal_ids) >= self.cfg.incremental_start_terrain_level
            if torch.any(updates_enabled):
                update_ids = normal_ids[updates_enabled]
                self._target_command[update_ids] = self._sample_prior_relative_updates(self._target_command[update_ids])
                self.time_left[update_ids] = self._normal_update_period_s(update_ids)

    @staticmethod
    def _sample_signed_magnitude(
        count: int, minimum: float, maximum: float, device: torch.device | str
    ) -> torch.Tensor:
        magnitude = torch.empty(count, device=device).uniform_(minimum, maximum)
        return magnitude * torch.where(torch.rand(count, device=device) < 0.5, -1.0, 1.0)

    @classmethod
    def _sample_normal_coupled_targets(
        cls, count: int, device: torch.device | str, max_planar_speed: float, max_yaw_rate: float
    ) -> torch.Tensor:
        command = torch.zeros(count, 3, device=device)
        if count == 0:
            return command
        speed = torch.empty(count, device=device).uniform_(min(0.25, max_planar_speed), max_planar_speed)
        angle = torch.empty(count, device=device).uniform_(-math.pi, math.pi)
        command[:, 0] = speed * torch.cos(angle)
        command[:, 1] = speed * torch.sin(angle)
        command[:, 2] = cls._sample_signed_magnitude(count, min(0.15, max_yaw_rate), max_yaw_rate, device)
        return command

    @classmethod
    def sample_direct_targets(
        cls, count: int, device: torch.device | str, max_planar_speed: float = 1.5, max_yaw_rate: float = 2.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the 15/15/10/40/20 slow/rotate/stop/normal/sudden mixture."""
        if count <= 0:
            return torch.empty(0, 3, device=device), torch.empty(0, dtype=torch.long, device=device)
        command = torch.zeros(count, 3, device=device)
        mode = torch.empty(count, dtype=torch.long, device=device)
        mixture = torch.rand(count, device=device)
        slow = mixture < 0.15
        rotate = (mixture >= 0.15) & (mixture < 0.30)
        stop = (mixture >= 0.30) & (mixture < 0.40)
        normal = (mixture >= 0.40) & (mixture < 0.80)
        sudden = ~(slow | rotate | stop | normal)
        slow_count = int(slow.sum().item())
        if slow_count:
            speed = torch.empty(slow_count, device=device).uniform_(0.05, min(0.35, max_planar_speed))
            angle = torch.empty(slow_count, device=device).uniform_(-math.pi, math.pi)
            command[slow, 0] = speed * torch.cos(angle)
            command[slow, 1] = speed * torch.sin(angle)
            command[slow, 2] = cls._sample_signed_magnitude(slow_count, min(0.25, max_yaw_rate), max_yaw_rate, device)
        rotate_count = int(rotate.sum().item())
        if rotate_count:
            command[rotate, 2] = cls._sample_signed_magnitude(rotate_count, min(0.15, max_yaw_rate), max_yaw_rate, device)
        coupled = normal | sudden
        if torch.any(coupled):
            command[coupled] = cls._sample_normal_coupled_targets(
                int(coupled.sum().item()), device, max_planar_speed, max_yaw_rate
            )
        mode[slow] = cls.SLOW_COUPLED_TURN
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
