"""Deployment-shaped velocity commands for robust Go2 locomotion training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RobustVelocityCommandCfg(CommandTermCfg):
    """Direct body-twist command curriculum matching the navigation interface.

    The policy observes a delayed, rate-limited ``(vx, vy, wz)`` command.  The
    command target itself is never heading controlled: ``wz`` is always a
    direct yaw-rate request in rad/s.
    """

    class_type: type | None = None

    asset_name: str = "robot"
    """Robot asset used to obtain per-environment terrain levels."""

    max_planar_speed: float = 1.0
    max_yaw_rate: float = 1.2
    max_planar_accel: float = 1.5
    max_yaw_accel: float = 3.0

    curriculum_full_level: int = 6
    """Terrain level where 12.5 Hz targets and full jitter are enabled."""

    initial_target_hold_range_s: tuple[float, float] = (0.8, 1.2)
    final_target_hold_s: float = 0.08

    jitter_start_level: int = 3
    jitter_planar_std: float = 0.08
    jitter_yaw_std: float = 0.20
    jitter_correlation_time_s: float = 0.20

    min_delay_ticks: int = 1
    max_delay_ticks: int = 3

    sudden_transition_start_probability: float = 0.02
    """Probability that a normal resample starts a two-phase intervention.

    The first phase holds a fast approach target long enough for the emitted
    command to build speed. The next raw target is then either a stop or a
    large avoidance switch. The policy still sees only the rate-limited,
    delayed command.
    """

    sudden_transition_approach_hold_s: float = 0.60
    sudden_transition_response_hold_s: float = 0.60

    def __post_init__(self) -> None:
        # The command class is defined below this config declaration.  Resolve it
        # at instantiation time so this remains a normal, optional cfg field.
        if self.class_type is None:
            self.class_type = RobustVelocityCommand
        if self.max_planar_speed <= 0.0 or self.max_yaw_rate <= 0.0:
            raise ValueError("Velocity limits must be positive.")
        if self.max_planar_accel <= 0.0 or self.max_yaw_accel <= 0.0:
            raise ValueError("Acceleration limits must be positive.")
        if self.curriculum_full_level <= 0:
            raise ValueError("curriculum_full_level must be positive.")
        if self.jitter_start_level < 0 or self.jitter_start_level >= self.curriculum_full_level:
            raise ValueError("jitter_start_level must be in [0, curriculum_full_level).")
        if self.jitter_correlation_time_s <= 0.0:
            raise ValueError("jitter_correlation_time_s must be positive.")
        if self.min_delay_ticks < 0 or self.max_delay_ticks < self.min_delay_ticks:
            raise ValueError("Delay ticks must satisfy 0 <= min <= max.")
        if not 0.0 <= self.sudden_transition_start_probability <= 1.0:
            raise ValueError("sudden_transition_start_probability must be in [0, 1].")
        if self.sudden_transition_approach_hold_s <= 0.0:
            raise ValueError("sudden_transition_approach_hold_s must be positive.")
        if self.sudden_transition_response_hold_s <= 0.0:
            raise ValueError("sudden_transition_response_hold_s must be positive.")
        if self.initial_target_hold_range_s[0] <= 0.0 or (
            self.initial_target_hold_range_s[1] < self.initial_target_hold_range_s[0]
        ):
            raise ValueError("initial_target_hold_range_s must be positive and ordered.")
        if self.final_target_hold_s <= 0.0:
            raise ValueError("final_target_hold_s must be positive.")


class RobustVelocityCommand(CommandTerm):
    """Stateful direct-twist command term with transition and latency curricula."""

    cfg: RobustVelocityCommandCfg

    # Mode IDs are intentionally public for deterministic unit tests and logging.
    SLOW_COUPLED_TURN = 0
    ROTATE_IN_PLACE = 1
    NORMAL_COUPLED_MOTION = 2
    STRAIGHT_LATERAL_STOP = 3
    SUDDEN_STOP_APPROACH = 4
    SUDDEN_AVOIDANCE_APPROACH = 5
    SUDDEN_STOP = 6
    SUDDEN_AVOIDANCE_SWITCH = 7

    def __init__(self, cfg: RobustVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._emitted_command = torch.zeros_like(self._target_command)
        self._command = torch.zeros_like(self._target_command)
        self._jitter = torch.zeros_like(self._target_command)
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 0 is inactive, 1 emits a stop next, and 2 emits a lateral/yaw
        # avoidance switch next.  This makes high-speed interventions explicit
        # training scenarios rather than a rare accidental pair of samples.
        self._pending_sudden_transition = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._delay_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._history = torch.zeros(cfg.max_delay_ticks + 1, self.num_envs, 3, device=self.device)
        self._history_index = 0
        self._last_dt = float(env.step_dt)

        self.metrics["delay_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_planar_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Delayed direct body twist supplied to the policy and tracking rewards."""
        return self._command

    @property
    def target_command(self) -> torch.Tensor:
        """Undelayed latent twist target, intended for debugging and tests."""
        return self._target_command

    @property
    def emitted_command(self) -> torch.Tensor:
        """Rate-limited, pre-delay command, intended for debugging and tests."""
        return self._emitted_command

    @property
    def delay_ticks(self) -> torch.Tensor:
        """Fixed per-episode command delay in low-level control ticks."""
        return self._delay_ticks

    @property
    def mode(self) -> torch.Tensor:
        """Current sampled command mode."""
        return self._mode

    def __str__(self) -> str:
        return (
            "RobustVelocityCommand:\n"
            "\tCommand dimension: (3,)\n"
            f"\tPlanar speed limit: {self.cfg.max_planar_speed} m/s\n"
            f"\tYaw-rate limit: {self.cfg.max_yaw_rate} rad/s\n"
            f"\tDelay ticks: [{self.cfg.min_delay_ticks}, {self.cfg.max_delay_ticks}]"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._target_command[env_ids] = 0.0
        self._emitted_command[env_ids] = 0.0
        self._command[env_ids] = 0.0
        self._jitter[env_ids] = 0.0
        self._pending_sudden_transition[env_ids] = 0
        self._history[:, env_ids] = 0.0
        self._delay_ticks[env_ids] = torch.randint(
            self.cfg.min_delay_ticks,
            self.cfg.max_delay_ticks + 1,
            (len(env_ids),),
            device=self.device,
        )
        return super().reset(env_ids)

    def compute(self, dt: float) -> None:
        self._last_dt = dt
        super().compute(dt)

    def _terrain_alpha(self, env_ids: torch.Tensor) -> torch.Tensor:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_levels"):
            return torch.ones(len(env_ids), device=self.device)
        return torch.clamp(
            terrain.terrain_levels[env_ids].float() / float(self.cfg.curriculum_full_level), min=0.0, max=1.0
        )

    def _resample(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        alpha = self._terrain_alpha(env_ids)
        initial_hold = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.initial_target_hold_range_s)
        self.time_left[env_ids] = initial_hold + alpha * (self.cfg.final_target_hold_s - initial_hold)
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    @staticmethod
    def _sample_signed_magnitude(
        count: int, minimum: float, maximum: float, device: torch.device | str
    ) -> torch.Tensor:
        magnitude = torch.empty(count, device=device).uniform_(minimum, maximum)
        sign = torch.where(torch.rand(count, device=device) < 0.5, -1.0, 1.0)
        return magnitude * sign

    @classmethod
    def sample_direct_targets(
        cls, count: int, device: torch.device | str, max_planar_speed: float = 1.0, max_yaw_rate: float = 1.2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the fixed 25/20/40/15 direct-twist training mixture."""
        if count <= 0:
            return torch.empty(0, 3, device=device), torch.empty(0, dtype=torch.long, device=device)

        command = torch.zeros(count, 3, device=device)
        mode = torch.full((count,), cls.STRAIGHT_LATERAL_STOP, dtype=torch.long, device=device)
        mixture = torch.rand(count, device=device)

        slow = mixture < 0.25
        rotate = (mixture >= 0.25) & (mixture < 0.45)
        normal = (mixture >= 0.45) & (mixture < 0.85)
        residual = ~(slow | rotate | normal)

        def assign_planar(mask: torch.Tensor, speed_min: float, speed_max: float) -> None:
            num = int(mask.sum().item())
            if num == 0:
                return
            speed = torch.empty(num, device=device).uniform_(speed_min, speed_max)
            angle = torch.empty(num, device=device).uniform_(-math.pi, math.pi)
            command[mask, 0] = speed * torch.cos(angle)
            command[mask, 1] = speed * torch.sin(angle)

        assign_planar(slow, 0.05, min(0.35, max_planar_speed))
        command[slow, 2] = cls._sample_signed_magnitude(int(slow.sum().item()), 0.25, max_yaw_rate, device)
        mode[slow] = cls.SLOW_COUPLED_TURN

        command[rotate, 2] = cls._sample_signed_magnitude(int(rotate.sum().item()), 0.15, max_yaw_rate, device)
        mode[rotate] = cls.ROTATE_IN_PLACE

        assign_planar(normal, 0.25, max_planar_speed)
        command[normal, 2] = cls._sample_signed_magnitude(int(normal.sum().item()), 0.15, max_yaw_rate, device)
        mode[normal] = cls.NORMAL_COUPLED_MOTION

        # Split the remaining 15% evenly among stop, straight, and lateral cases.
        residual_ids = residual.nonzero(as_tuple=False).flatten()
        residual_kind = torch.rand(len(residual_ids), device=device)
        straight = residual_ids[residual_kind < 1.0 / 3.0]
        lateral = residual_ids[(residual_kind >= 1.0 / 3.0) & (residual_kind < 2.0 / 3.0)]
        if len(straight) > 0:
            speed = torch.empty(len(straight), device=device).uniform_(0.25, max_planar_speed)
            command[straight, 0] = speed * torch.where(torch.rand(len(straight), device=device) < 0.5, -1.0, 1.0)
        if len(lateral) > 0:
            speed = torch.empty(len(lateral), device=device).uniform_(0.25, min(0.8, max_planar_speed))
            command[lateral, 1] = speed * torch.where(torch.rand(len(lateral), device=device) < 0.5, -1.0, 1.0)

        return command, mode

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        commands, modes = self.sample_direct_targets(
            len(env_ids), self.device, self.cfg.max_planar_speed, self.cfg.max_yaw_rate
        )
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        pending = self._pending_sudden_transition[env_ids]
        stop_ids = pending == 1
        avoidance_ids = pending == 2
        # Raw target steps intentionally mirror obstacle intervention cases.
        # Their delivery is still shaped below by the shared limiter and delay.
        commands[stop_ids] = 0.0
        modes[stop_ids] = self.SUDDEN_STOP
        if torch.any(avoidance_ids):
            count = int(avoidance_ids.sum().item())
            lateral_sign = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
            yaw_sign = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
            commands[avoidance_ids, 0] = 0.25
            commands[avoidance_ids, 1] = 0.65 * lateral_sign
            commands[avoidance_ids, 2] = 0.80 * yaw_sign
            modes[avoidance_ids] = self.SUDDEN_AVOIDANCE_SWITCH
        self._pending_sudden_transition[env_ids[pending != 0]] = 0
        if torch.any(pending != 0):
            self.time_left[env_ids[pending != 0]] = self.cfg.sudden_transition_response_hold_s

        # Start a sustained fast approach, then force an intervention on the
        # following resample. The start probability is deliberately low: its
        # 0.60-s approach plus 0.60-s response would otherwise dominate the
        # high-curriculum time mix.
        eligible = pending == 0
        starts = eligible & (
            torch.rand(len(env_ids), device=self.device) < self.cfg.sudden_transition_start_probability
        )
        stop_starts = starts & (torch.rand(len(env_ids), device=self.device) < 0.5)
        avoidance_starts = starts & ~stop_starts
        if torch.any(starts):
            commands[starts] = 0.0
            commands[starts, 0] = 0.75
            stop_count = int(stop_starts.sum().item())
            if stop_count:
                commands[stop_starts, 2] = self._sample_signed_magnitude(stop_count, 0.60, 0.60, self.device)
            modes[stop_starts] = self.SUDDEN_STOP_APPROACH
            modes[avoidance_starts] = self.SUDDEN_AVOIDANCE_APPROACH
            self._pending_sudden_transition[env_ids[stop_starts]] = 1
            self._pending_sudden_transition[env_ids[avoidance_starts]] = 2
            # A 0.75 m/s request needs this sustained approach phase so the
            # robot is truly moving before the raw target changes.
            self.time_left[env_ids[starts]] = self.cfg.sudden_transition_approach_hold_s
        self._target_command[env_ids] = commands
        self._mode[env_ids] = modes

    def _update_metrics(self) -> None:
        self.metrics["delay_s"][:] = self._delay_ticks * self._last_dt
        self.metrics["target_planar_speed"][:] = torch.linalg.vector_norm(self._target_command[:, :2], dim=-1)
        self.metrics["target_yaw_rate"][:] = torch.abs(self._target_command[:, 2])

    def _update_command(self) -> None:
        levels = torch.arange(self.num_envs, device=self.device)
        terrain_alpha = self._terrain_alpha(levels)
        terrain_level = terrain_alpha * self.cfg.curriculum_full_level
        # Begin level 3 with half of the final standard deviation (0.04 m/s,
        # 0.10 rad/s by default) and reach the full magnitude at level 6.
        jitter_alpha = torch.where(
            terrain_level < self.cfg.jitter_start_level,
            torch.zeros_like(terrain_level),
            0.5
            + 0.5
            * (terrain_level - self.cfg.jitter_start_level)
            / float(self.cfg.curriculum_full_level - self.cfg.jitter_start_level),
        ).clamp(max=1.0)
        active = (torch.linalg.vector_norm(self._target_command[:, :2], dim=-1) > 1.0e-6) | (
            torch.abs(self._target_command[:, 2]) > 1.0e-6
        )
        rho = math.exp(-self._last_dt / self.cfg.jitter_correlation_time_s)
        noise_std = torch.stack(
            (
                jitter_alpha * self.cfg.jitter_planar_std,
                jitter_alpha * self.cfg.jitter_planar_std,
                jitter_alpha * self.cfg.jitter_yaw_std,
            ),
            dim=-1,
        )
        self._jitter = rho * self._jitter + math.sqrt(1.0 - rho * rho) * torch.randn_like(self._jitter) * noise_std
        self._jitter[~active] = 0.0

        noisy_target = self._target_command + self._jitter
        self._apply_command_shaping(noisy_target)

    def _apply_command_shaping(self, target_command: torch.Tensor) -> None:
        """Rate-limit and delay a target command.

        Kept separate from target sampling so scripted evaluation can exercise
        precisely the delivery path used during robust-policy training.
        """
        noisy_target = target_command.clone()
        planar_norm = torch.linalg.vector_norm(noisy_target[:, :2], dim=-1, keepdim=True)
        planar_scale = torch.clamp(self.cfg.max_planar_speed / planar_norm.clamp_min(1.0e-8), max=1.0)
        noisy_target[:, :2] *= planar_scale
        noisy_target[:, 2].clamp_(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

        planar_delta = noisy_target[:, :2] - self._emitted_command[:, :2]
        planar_delta_norm = torch.linalg.vector_norm(planar_delta, dim=-1, keepdim=True)
        max_planar_delta = self.cfg.max_planar_accel * self._last_dt
        planar_delta *= torch.clamp(max_planar_delta / planar_delta_norm.clamp_min(1.0e-8), max=1.0)
        self._emitted_command[:, :2] += planar_delta
        yaw_delta = torch.clamp(
            noisy_target[:, 2] - self._emitted_command[:, 2],
            min=-self.cfg.max_yaw_accel * self._last_dt,
            max=self.cfg.max_yaw_accel * self._last_dt,
        )
        self._emitted_command[:, 2] += yaw_delta

        self._history[self._history_index] = self._emitted_command
        # The slot just written is delay zero; one tick reads the preceding slot.
        read_indices = (self._history_index - self._delay_ticks) % self._history.shape[0]
        self._command[:] = self._history[read_indices, torch.arange(self.num_envs, device=self.device)]
        self._history_index = (self._history_index + 1) % self._history.shape[0]


@configclass
class ScriptedVelocityCommandCfg(RobustVelocityCommandCfg):
    """Evaluation-only direct-twist source with robust-v1 delivery semantics."""

    class_type: type | None = None

    def __post_init__(self) -> None:
        self.class_type = ScriptedVelocityCommand
        super().__post_init__()


class ScriptedVelocityCommand(RobustVelocityCommand):
    """Externally driven counterpart of :class:`RobustVelocityCommand`.

    The evaluator sets latent targets explicitly.  This term deliberately
    retains the same rate limiter and delay history as training, while
    removing all sampling, jitter, and terrain-curriculum behavior.
    """

    cfg: ScriptedVelocityCommandCfg

    def _resample(self, env_ids: Sequence[int]) -> None:
        # Scripted targets are owned by the evaluator, never by CommandManager.
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.time_left[env_ids] = float("inf")
        self.command_counter[env_ids] += 1

    def set_target_commands(self, commands: torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
        """Set unfiltered body-twist targets for the selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        commands = torch.as_tensor(commands, device=self.device, dtype=self._target_command.dtype)
        if commands.shape != (len(env_ids), 3):
            raise ValueError(f"Expected command shape ({len(env_ids)}, 3), received {tuple(commands.shape)}.")
        self._target_command[env_ids] = commands

    def set_delay_ticks(self, delay_ticks: torch.Tensor | int, env_ids: Sequence[int] | None = None) -> None:
        """Set fixed delivery latency for evaluation environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        delays = torch.as_tensor(delay_ticks, device=self.device, dtype=torch.long)
        if delays.ndim == 0:
            delays = delays.expand(len(env_ids))
        if delays.shape != (len(env_ids),):
            raise ValueError(f"Expected {len(env_ids)} delay ticks, received {tuple(delays.shape)}.")
        if torch.any(delays < 0) or torch.any(delays > self.cfg.max_delay_ticks):
            raise ValueError(f"Delay ticks must be in [0, {self.cfg.max_delay_ticks}].")
        self._delay_ticks[env_ids] = delays

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # Targets are set through ``set_target_commands``.
        return

    def _update_command(self) -> None:
        self._jitter.zero_()
        self._apply_command_shaping(self._target_command)
