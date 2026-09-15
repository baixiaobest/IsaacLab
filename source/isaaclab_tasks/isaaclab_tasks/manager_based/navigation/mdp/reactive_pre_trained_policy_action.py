# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bake the reactive greedy/DWA baselines into the Kp-bounded navigation action term.

``KpPreTrainedPolicyAction.process_actions`` only ever does
``self._raw_actions[:] = actions * self._action_scales`` with whatever ``(num_envs, 3)`` tensor the
caller hands it -- it does not care whether that tensor came from an RL policy or an external Python
controller. The action terms here replace that external hand-off: instead of accepting the desired
body-frame ``(vx, vy, wz)`` command from outside, they compute it themselves each step, from a single
current-frame LiDAR scan, via :mod:`reactive_controllers` -- exactly the way the base class already
computes its low-level joint actions from a frozen locomotion policy instead of accepting them from
outside. There is deliberately no CBF-QP safety filter in this path: the controller's command goes
through the existing Kp-bounded acceleration/velocity tracking (the same preprocessing the non-CBF
"Kp" tasks already use) and straight to the frozen locomotion policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from .kp_pre_trained_policy_action import KpPreTrainedPolicyAction, KpPreTrainedPolicyActionCfg
from .reactive_controllers import DwaController, GreedyLidarController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def build_greedy_controller(cfg: GreedyPreTrainedPolicyActionCfg) -> GreedyLidarController:
    """Build the exact controller a :class:`GreedyPreTrainedPolicyAction` would build from ``cfg``.

    Factored out so an eval script can build an identical "shadow" controller off the same task cfg
    -- e.g. to log the desired command for a replay recorder -- without duplicating field names and
    risking them drifting out of sync with the action term.
    """
    return GreedyLidarController(
        alpha=cfg.alpha,
        beta=cfg.beta,
        max_speed_mps=cfg.max_speed_mps,
        min_speed_mps=cfg.min_speed_mps,
        slow_distance_m=cfg.slow_distance_m,
        full_stop_angle_deg=cfg.full_stop_angle_deg,
        yaw_gain=cfg.yaw_gain,
        max_yaw_rate_rad_s=cfg.max_yaw_rate,
        clearance_window_deg=cfg.clearance_window_deg,
        distance_cap_m=cfg.distance_cap_m,
        num_candidates=cfg.num_candidates,
        sensor_name=cfg.lidar_sensor_name,
        command_name=cfg.command_name,
    )


def build_dwa_controller(cfg: DwaPreTrainedPolicyActionCfg) -> DwaController:
    """Build the exact controller a :class:`DwaPreTrainedPolicyAction` would build from ``cfg``.

    Factored out so an eval script can build an identical "shadow" controller off the same task cfg
    -- e.g. to log the desired command for a replay recorder -- without duplicating field names and
    risking them drifting out of sync with the action term.
    """
    return DwaController(
        vel_abs_lower_mps=cfg.vel_abs_lower_mps,
        vel_abs_upper_mps=cfg.vel_abs_upper_mps,
        accel_lower_mps2=cfg.accel_lower_mps2,
        accel_upper_mps2=cfg.accel_upper_mps2,
        max_yaw_rate_rad_s=cfg.max_yaw_rate,
        max_yaw_accel_rad_s2=cfg.max_yaw_accel,
        rollout_dt_s=cfg.rollout_dt if cfg.rollout_dt > 0.0 else None,
        num_rollout_steps=cfg.num_rollout_steps,
        num_vx_samples=cfg.num_vx_samples,
        num_wz_samples=cfg.num_wz_samples,
        w_heading=cfg.w_heading,
        w_clearance=cfg.w_clearance,
        w_velocity=cfg.w_velocity,
        distance_cap_m=cfg.distance_cap_m,
        robot_radius_m=cfg.robot_radius_m,
        safety_margin_m=cfg.safety_margin_m,
        fallback_yaw_gain=cfg.fallback_yaw_gain,
        obstacle_stride=cfg.obstacle_stride,
        sensor_name=cfg.lidar_sensor_name,
        command_name=cfg.command_name,
    )


class GreedyPreTrainedPolicyAction(KpPreTrainedPolicyAction):
    """Kp-bounded navigation action term driven by the in-environment greedy LiDAR controller."""

    cfg: GreedyPreTrainedPolicyActionCfg

    def __init__(self, cfg: GreedyPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._controller = build_greedy_controller(cfg)

    def process_actions(self, actions: torch.Tensor):
        """Ignore the externally-submitted action; compute the desired command in-environment."""
        self._raw_actions[:] = self._controller.compute_actions(self._env)


@configclass
class GreedyPreTrainedPolicyActionCfg(KpPreTrainedPolicyActionCfg):
    """Configuration for :class:`GreedyPreTrainedPolicyAction`.

    Mirrors ``evaluate_baseline.py``'s ``--greedy_*`` CLI flags, so the same tuning now lives on the
    task's env cfg instead of argparse.
    """

    class_type: type = GreedyPreTrainedPolicyAction
    alpha: float = 1.0
    """Weight on normalized free space D(theta)."""
    beta: float = 1.5
    """Weight on normalized goal-heading error |theta - theta_g|."""
    max_speed_mps: float = 1.0
    """Forward speed with full clearance and no turn."""
    min_speed_mps: float = 0.0
    """Forward speed floor before obstacle/turn scaling."""
    slow_distance_m: float = 1.5
    """Clearance below which forward speed is scaled down."""
    full_stop_angle_deg: float = 70.0
    """Steering angle beyond which forward speed is scaled to zero."""
    yaw_gain: float = 1.5
    """Proportional gain turning the robot toward theta*."""
    max_yaw_rate: float = 1.5
    """Maximum commanded yaw rate (rad/s)."""
    clearance_window_deg: float = 12.0
    """Angular sector width used to compute D(theta) as a worst-case (min) clearance."""
    distance_cap_m: float = 3.0
    """Clearance beyond which D(theta) saturates at 1.0 in the score."""
    num_candidates: int | None = None
    """Candidate steering directions; ``None`` uses the full raw ray resolution."""
    lidar_sensor_name: str = "obstacle_scanner"
    """Scene entity name of the 2D LiDAR sensor the controller reads."""
    command_name: str = "pose_2d_command"
    """Command manager term supplying the goal bearing."""


class DwaPreTrainedPolicyAction(KpPreTrainedPolicyAction):
    """Kp-bounded navigation action term driven by the in-environment DWA controller."""

    cfg: DwaPreTrainedPolicyActionCfg

    def __init__(self, cfg: DwaPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._controller = build_dwa_controller(cfg)

    def process_actions(self, actions: torch.Tensor):
        """Ignore the externally-submitted action; compute the desired command in-environment."""
        self._raw_actions[:] = self._controller.compute_actions(self._env)


@configclass
class DwaPreTrainedPolicyActionCfg(KpPreTrainedPolicyActionCfg):
    """Configuration for :class:`DwaPreTrainedPolicyAction`.

    Mirrors ``evaluate_dwa_baseline.py``'s ``--dwa_*`` CLI flags, so the same tuning now lives on the
    task's env cfg instead of argparse.
    """

    class_type: type = DwaPreTrainedPolicyAction
    vel_abs_lower_mps: float = 0.0
    """Lowest reachable forward speed (no reverse by default)."""
    vel_abs_upper_mps: float = 1.0
    """Highest reachable forward speed."""
    accel_lower_mps2: float = -3.0
    """Deceleration bound used to size the one-step dynamic window."""
    accel_upper_mps2: float = 3.0
    """Acceleration bound used to size the one-step dynamic window."""
    max_yaw_rate: float = 1.5
    """Maximum reachable yaw rate (rad/s)."""
    max_yaw_accel: float = 3.0
    """Yaw-acceleration bound used to size the one-step dynamic window."""
    rollout_dt: float = 0.0
    """Rollout/window-sizing tick in seconds; ``0`` uses the environment's own control-step dt."""
    num_rollout_steps: int = 10
    """Number of forward-simulated rollout ticks per candidate."""
    num_vx_samples: int = 11
    """Number of forward-speed samples in the dynamic window."""
    num_wz_samples: int = 21
    """Number of yaw-rate samples in the dynamic window."""
    w_heading: float = 1.0
    """Weight on goal-heading alignment at the rolled-out trajectory's end."""
    w_clearance: float = 1.0
    """Weight on worst-case clearance from LiDAR obstacle points along the rolled-out trajectory."""
    w_velocity: float = 0.5
    """Weight on forward speed."""
    distance_cap_m: float = 3.0
    """Clearance beyond which the clearance score saturates at 1.0."""
    robot_radius_m: float = 0.4
    """Robot collision radius."""
    safety_margin_m: float = 0.1
    """Extra buffer beyond robot_radius_m before a candidate trajectory is masked infeasible."""
    fallback_yaw_gain: float = 1.5
    """Proportional gain turning toward the goal bearing when every candidate is infeasible."""
    obstacle_stride: int = 1
    """Use every Nth LiDAR ray as an obstacle point; 1 uses full ray resolution."""
    lidar_sensor_name: str = "obstacle_scanner"
    """Scene entity name of the 2D LiDAR sensor the controller reads."""
    command_name: str = "pose_2d_command"
    """Command manager term supplying the goal bearing."""
