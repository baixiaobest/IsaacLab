# Copyright (c) 2026, Baixiao Huang.
# # All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kp velocity-command preprocessing for navigation policies.

This module deliberately leaves :mod:`pre_trained_policy_action` unchanged.  It
provides a separate action term for experiments that model the navigation
command as a bounded acceleration before forwarding it to the low-level
locomotion policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ObservationManager
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

from .pre_trained_policy_action import PreTrainedPolicyAction, PreTrainedPolicyActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zoh_average_acceleration_gain(step_dt: float, tracking_tau_s: float) -> float:
    """Return the ZOH gain from interval-average acceleration to command offset."""
    if step_dt <= 0.0 or tracking_tau_s <= 0.0:
        raise ValueError("step_dt and tracking_tau_s must be positive.")
    return step_dt / (1.0 - math.exp(-step_dt / tracking_tau_s))


def compute_kp_velocity_command(
    desired_velocity: torch.Tensor,
    measured_velocity: torch.Tensor,
    step_dt: float,
    tracking_tau_s: float,
    kp: torch.Tensor,
    acceleration_lower: torch.Tensor,
    acceleration_upper: torch.Tensor,
    velocity_lower: torch.Tensor,
    velocity_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return model-aware nominal acceleration and a bounded planar command.

    All velocity tensors are planar body-frame velocities. ``step_dt`` is the
    high-level command hold period. The ZOH gain maps a desired *average*
    acceleration over that period to the locomotion velocity command required
    by the fixed first-order tracking model.
    """
    zoh_gain_s = zoh_average_acceleration_gain(step_dt, tracking_tau_s)
    nominal_acceleration = torch.clamp(
        kp * (desired_velocity - measured_velocity),
        min=acceleration_lower,
        max=acceleration_upper,
    )
    effective_lower = torch.maximum(acceleration_lower, (velocity_lower - measured_velocity) / zoh_gain_s)
    effective_upper = torch.minimum(acceleration_upper, (velocity_upper - measured_velocity) / zoh_gain_s)
    feasible = torch.all(effective_lower <= effective_upper, dim=-1, keepdim=True)
    safe_acceleration = torch.clamp(nominal_acceleration, min=effective_lower, max=effective_upper)
    # If a transient measured velocity lies outside the command envelope,
    # command the nearest achievable return rather than advancing stale state.
    safe_acceleration = torch.where(feasible, safe_acceleration, torch.zeros_like(safe_acceleration))
    commanded_velocity = measured_velocity + zoh_gain_s * safe_acceleration
    commanded_velocity = torch.clamp(
        commanded_velocity,
        min=velocity_lower,
        max=velocity_upper,
    )
    return nominal_acceleration, commanded_velocity


class KpPreTrainedPolicyAction(PreTrainedPolicyAction):
    """Pre-trained locomotion action with bounded Kp planar-velocity tracking.

    The high-level policy action remains ``(v_RL,x, v_RL,y, yaw_rate)``.  Its
    planar desired velocity is a body-frame value. The controller compares it
    with measured body velocity and sends a ZOH model-aware velocity command to
    the low-level policy. The yaw command is intentionally passed through
    unchanged.
    """

    cfg: KpPreTrainedPolicyActionCfg

    def __init__(self, cfg: KpPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # This mirrors PreTrainedPolicyAction initialization because the
        # low-level observation binds to the model-aware command, not v_RL.
        ActionTerm.__init__(self, cfg, env)

        if len(cfg.action_scales) != 3:
            raise ValueError("KpPreTrainedPolicyAction requires exactly three actions: (v_x, v_y, yaw_rate).")
        if len(cfg.kp) != 2:
            raise ValueError("'kp' must provide one proportional gain for each planar axis.")

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._action_scales = torch.tensor(cfg.action_scales, device=self.device)
        self._kp = torch.tensor(cfg.kp, device=self.device)
        self._acceleration_lower, self._acceleration_upper = self._limits_to_tensors(
            cfg.acceleration_limits, "acceleration_limits"
        )
        self._velocity_lower, self._velocity_upper = self._limits_to_tensors(cfg.velocity_limits, "velocity_limits")

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._nominal_acceleration = torch.zeros(self.num_envs, 2, device=self.device)

        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        # Only the private low-level-policy observation receives v_cmd.  The
        # high-level mdp.last_action observation remains the policy's v_RL.
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._processed_actions
        cfg.low_level_observations.velocity_commands.params = dict()
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0
        self._control_dt = cfg.low_level_decimation * env.physics_dt

    @property
    def processed_actions(self) -> torch.Tensor:
        """Model-aware body-frame velocity command sent to the low-level policy."""
        return self._processed_actions

    @property
    def nominal_acceleration(self) -> torch.Tensor:
        """Bounded body-frame nominal planar acceleration, for diagnostics."""
        return self._nominal_acceleration

    @property
    def desired_velocity(self) -> torch.Tensor:
        """The scaled RL planar velocity target in the robot body frame."""
        return self._raw_actions[:, :2]

    @property
    def commanded_velocity(self) -> torch.Tensor:
        """The model-aware, velocity-limited planar command in the body frame."""
        return self._processed_actions[:, :2]

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {self.action_dim}, received {actions.shape[-1]}."
            )
        self._raw_actions[:] = actions * self._action_scales

    def _update_model_aware_command(self) -> None:
        """Recompute the held navigation target against the latest measured velocity."""
        measured_velocity = self.robot.data.root_lin_vel_b[:, :2]
        self._nominal_acceleration[:], self._processed_actions[:, :2] = compute_kp_velocity_command(
            self._raw_actions[:, :2],
            measured_velocity,
            self._control_dt,
            self.cfg.tracking_tau_s,
            self._kp,
            self._acceleration_lower,
            self._acceleration_upper,
            self._velocity_lower,
            self._velocity_upper,
        )
        self._processed_actions[:, 2] = self._raw_actions[:, 2]

    def apply_actions(self):
        """Update the model-aware command at the locomotion-policy cadence."""
        if self._counter % self.cfg.low_level_decimation == 0:
            self._update_model_aware_command()
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    def _limits_to_tensors(
        self, limits: tuple[tuple[float, float], tuple[float, float]], name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(limits) != 2 or any(len(axis_limits) != 2 for axis_limits in limits):
            raise ValueError(f"'{name}' must be ((x_min, x_max), (y_min, y_max)).")
        lower = torch.tensor((limits[0][0], limits[1][0]), device=self.device)
        upper = torch.tensor((limits[0][1], limits[1][1]), device=self.device)
        if torch.any(lower > upper):
            raise ValueError(f"Each lower bound in '{name}' must be no greater than its upper bound.")
        return lower, upper


@configclass
class KpPreTrainedPolicyActionCfg(PreTrainedPolicyActionCfg):
    """Configuration for :class:`KpPreTrainedPolicyAction`."""

    class_type: type[ActionTerm] = KpPreTrainedPolicyAction
    kp: tuple[float, float] = (5.0, 5.0)
    """Body-frame planar proportional gains in s^-1, ordered ``(x, y)``."""
    acceleration_limits: tuple[tuple[float, float], tuple[float, float]] = ((-3.0, 3.0), (-3.0, 3.0))
    """Body-frame component-wise acceleration bounds in m/s^2: ``((x_min, x_max), (y_min, y_max))``."""
    velocity_limits: tuple[tuple[float, float], tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0))
    """Existing body-frame component-wise planar velocity bounds in m/s."""
    tracking_tau_s: float = 0.30
    """Fixed first-order locomotion tracking time constant used by ZOH preprocessing, in seconds."""
