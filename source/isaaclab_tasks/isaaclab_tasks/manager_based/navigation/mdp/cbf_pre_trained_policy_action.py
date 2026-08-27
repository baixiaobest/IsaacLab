# Copyright (c) 2026, Baixiao Huang.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment-only static-obstacle CBF filtering for Kp navigation commands.

The high-level navigation action stays a body-frame ``(v_x, v_y, yaw_rate)``
command.  This term holds that target between navigation steps, then recomputes
Kp and the CBF-QP whenever the low-level locomotion policy is updated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

from .kp_pre_trained_policy_action import KpPreTrainedPolicyAction, KpPreTrainedPolicyActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass(frozen=True)
class _OsqpSolveStats:
    """OSQP diagnostics for one QP solve.

    Times are supplied by OSQP itself and therefore cover only the solver-side
    work.  They intentionally do not include PyTorch-to-NumPy transfers or the
    surrounding CBF construction, which can be profiled separately.
    """

    iterations: int
    solve_time_s: float
    update_time_s: float
    polish_time_s: float
    primal_residual: float
    dual_residual: float
    status: str


class _OsqpStaticObstacleCbf:
    """One fixed-sparsity OSQP problem for one vectorized environment."""

    def __init__(self, cfg: StaticObstacleCbfPreTrainedPolicyActionCfg) -> None:
        try:
            import osqp
            import scipy.sparse as sparse
        except ImportError as error:
            raise ImportError(
                "StaticObstacleCbfPreTrainedPolicyAction requires the optional 'osqp' package. "
                "Install Isaac Lab tasks with its deployment dependencies."
            ) from error

        self._max_points = cfg.max_lidar_points
        # Variables are [u_world_x, u_world_y, shared_slack].  Constraint rows
        # are barrier points, slack >= 0, body-frame acceleration bounds, and
        # body-frame one-step velocity bounds.
        self._num_rows = self._max_points + 5
        self._slack_row = self._max_points
        self._accel_rows = slice(self._max_points + 1, self._max_points + 3)
        self._velocity_rows = slice(self._max_points + 3, self._max_points + 5)

        p = sparse.diags([2.0, 2.0, 2.0 * cfg.slack_penalty], format="csc")
        # Keep every A entry structurally present so OSQP can update all point
        # coefficients without reconstructing/factorizing the problem pattern.
        a = sparse.csc_matrix(
            (
                np.ones(self._num_rows * 3),
                np.tile(np.arange(self._num_rows), 3),
                np.arange(0, self._num_rows * 3 + 1, self._num_rows),
            ),
            shape=(self._num_rows, 3),
        )
        self._solver = osqp.OSQP()
        osqp_major_version = int(osqp.__version__.split(".", maxsplit=1)[0])
        solver_version_settings = (
            {"polishing": cfg.solver_polish, "warm_starting": cfg.solver_warm_start}
            if osqp_major_version >= 1
            else {"polish": cfg.solver_polish, "warm_start": cfg.solver_warm_start}
        )
        self._solver.setup(
            P=p,
            q=np.zeros(3),
            A=a,
            l=np.full(self._num_rows, -np.inf),
            u=np.full(self._num_rows, np.inf),
            verbose=False,
            eps_abs=cfg.solver_eps_abs,
            eps_rel=cfg.solver_eps_rel,
            max_iter=cfg.solver_max_iter,
            **solver_version_settings,
        )

    def solve(
        self,
        nominal_acceleration_w: np.ndarray,
        obstacle_vectors_w: np.ndarray,
        barrier_rhs: np.ndarray,
        body_to_world: np.ndarray,
        acceleration_lower_b: np.ndarray,
        acceleration_upper_b: np.ndarray,
        velocity_acceleration_lower_b: np.ndarray,
        velocity_acceleration_upper_b: np.ndarray,
    ) -> tuple[np.ndarray | None, float, _OsqpSolveStats]:
        """Solve the soft CBF QP and return its result with OSQP diagnostics."""
        matrix = np.zeros((self._num_rows, 3), dtype=np.float64)
        lower = np.full(self._num_rows, -np.inf, dtype=np.float64)
        upper = np.full(self._num_rows, np.inf, dtype=np.float64)

        num_points = obstacle_vectors_w.shape[0]
        matrix[:num_points, :2] = 2.0 * obstacle_vectors_w
        matrix[:num_points, 2] = 1.0
        lower[:num_points] = barrier_rhs

        matrix[self._slack_row, 2] = 1.0
        lower[self._slack_row] = 0.0

        # u_body = R_wb.T u_world.  Acceleration and velocity bounds remain
        # exactly the existing component-wise body-frame command bounds.
        world_to_body = body_to_world.T
        matrix[self._accel_rows, :2] = world_to_body
        lower[self._accel_rows] = acceleration_lower_b
        upper[self._accel_rows] = acceleration_upper_b

        matrix[self._velocity_rows, :2] = world_to_body
        lower[self._velocity_rows] = velocity_acceleration_lower_b
        upper[self._velocity_rows] = velocity_acceleration_upper_b

        # CSC stores columns consecutively, hence the transpose before flatten.
        self._solver.update(q=np.array([-2.0 * nominal_acceleration_w[0], -2.0 * nominal_acceleration_w[1], 0.0]),
                            Ax=matrix.T.reshape(-1), l=lower, u=upper)
        result = self._solver.solve()
        info = result.info
        status = str(info.status)
        # OSQP renamed these residual fields between its 0.x and 1.x Python
        # interfaces.  Keep the task compatible with both image variants.
        stats = _OsqpSolveStats(
            iterations=int(info.iter),
            solve_time_s=float(getattr(info, "solve_time", 0.0)),
            update_time_s=float(getattr(info, "update_time", 0.0)),
            polish_time_s=float(getattr(info, "polish_time", 0.0)),
            primal_residual=float(getattr(info, "prim_res", getattr(info, "pri_res", 0.0))),
            dual_residual=float(getattr(info, "dual_res", getattr(info, "dua_res", 0.0))),
            status=status,
        )
        if result.x is None or not status.lower().startswith("solved"):
            return None, 0.0, stats
        return result.x[:2], max(0.0, float(result.x[2])), stats


def integrate_velocity_setpoint_world(
    velocity_setpoint_w: torch.Tensor,
    acceleration_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    control_dt: float,
    velocity_lower_b: torch.Tensor,
    velocity_upper_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a world-frame acceleration into a bounded body-frame command.

    The reference state remains in the world frame so it is not spuriously
    changed merely because the robot turns. The locomotion policy still
    receives its usual body-frame velocity command.
    """
    next_setpoint_w = velocity_setpoint_w + control_dt * acceleration_w
    next_setpoint_b = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(root_quat_w),
        torch.cat((next_setpoint_w, torch.zeros_like(next_setpoint_w[:, :1])), dim=1),
    )[:, :2]
    next_setpoint_b = torch.clamp(next_setpoint_b, min=velocity_lower_b, max=velocity_upper_b)
    next_setpoint_w = math_utils.quat_apply_yaw(
        root_quat_w,
        torch.cat((next_setpoint_b, torch.zeros_like(next_setpoint_b[:, :1])), dim=1),
    )[:, :2]
    return next_setpoint_w, next_setpoint_b


class StaticObstacleCbfPreTrainedPolicyAction(KpPreTrainedPolicyAction):
    """Kp navigation command filtered by a static-obstacle, soft CBF-QP.

    ``process_actions`` only stores the latest high-level RL command.  At each
    low-level policy update, this term obtains the latest held LiDAR scan,
    recomputes the Kp nominal acceleration from the current robot velocity,
    solves the QP, and forwards the resulting body-frame velocity command.  The
    yaw-rate action bypasses both Kp and CBF unchanged.
    """

    cfg: StaticObstacleCbfPreTrainedPolicyActionCfg

    def __init__(self, cfg: StaticObstacleCbfPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if cfg.d_margin <= 0.0 or cfg.d_cbf_active <= 0.0:
            raise ValueError("d_margin and d_cbf_active must be positive.")
        if cfg.d_cbf_active < cfg.d_margin:
            raise ValueError("d_cbf_active must be no smaller than d_margin.")
        if cfg.gamma1 <= 0.0 or cfg.gamma2 <= 0.0 or cfg.slack_penalty <= 0.0:
            raise ValueError("CBF gains and slack_penalty must be positive.")
        if cfg.max_lidar_points < 1:
            raise ValueError("max_lidar_points must be at least one.")

        self._control_dt = cfg.low_level_decimation * env.physics_dt
        self._solvers = [_OsqpStaticObstacleCbf(cfg) for _ in range(self.num_envs)]
        self._safe_acceleration_w = torch.zeros(self.num_envs, 2, device=self.device)
        # The CBF's acceleration is applied to this persistent reference, not
        # repeatedly rebased on the measured robot velocity. Keeping it in the
        # world frame makes the integration invariant to changes in body yaw.
        self._velocity_setpoint_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._velocity_setpoint_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._slack = torch.zeros(self.num_envs, device=self.device)
        self._slack_positive_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._slack_sum = torch.zeros(self.num_envs, device=self.device)
        self._slack_max = torch.zeros(self.num_envs, device=self.device)
        self._cbf_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._solve_failures = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._velocity_feasibility_failures = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # These remain NumPy arrays because OSQP already returns host values.
        # Updating CUDA tensors for every scalar diagnostic would add many tiny
        # kernel launches to the control loop we are trying to measure.
        self._solver_solve_count = np.zeros(self.num_envs, dtype=np.int64)
        self._solver_iteration_total = np.zeros(self.num_envs, dtype=np.int64)
        self._solver_iteration_max = np.zeros(self.num_envs, dtype=np.int64)
        self._solver_solve_time_total_s = np.zeros(self.num_envs, dtype=np.float64)
        self._solver_solve_time_max_s = np.zeros(self.num_envs, dtype=np.float64)
        self._solver_update_time_total_s = np.zeros(self.num_envs, dtype=np.float64)
        self._solver_polish_time_total_s = np.zeros(self.num_envs, dtype=np.float64)
        self._solver_primal_residual_max = np.zeros(self.num_envs, dtype=np.float64)
        self._solver_dual_residual_max = np.zeros(self.num_envs, dtype=np.float64)
        self._solver_inaccurate_count = np.zeros(self.num_envs, dtype=np.int64)
        self._solver_max_iteration_count = np.zeros(self.num_envs, dtype=np.int64)

    @property
    def cbf_control_dt(self) -> float:
        """Actual CBF/locomotion command period in seconds."""
        return self._control_dt

    @property
    def safe_acceleration(self) -> torch.Tensor:
        """World-frame planar acceleration returned by the CBF-QP."""
        return self._safe_acceleration_w

    @property
    def velocity_setpoint(self) -> torch.Tensor:
        """Persistent world-frame planar velocity reference sent to the locomotion policy."""
        return self._velocity_setpoint_w

    @property
    def cbf_slack(self) -> torch.Tensor:
        """Current shared CBF slack for every vectorized environment."""
        return self._slack

    @property
    def cbf_filtered_velocity_command(self) -> torch.Tensor:
        """Latest body-frame CBF-filtered velocity command for PLAY replay."""
        return self._processed_actions

    @property
    def slack_metrics(self) -> dict[str, torch.Tensor]:
        """Per-episode CBF slack and solver diagnostics.

        Consumers can read this action term during PLAY evaluation.  Counters
        reset with the environment; ``mean_nonzero`` is zero until the first
        positive slack step.
        """
        positive = self._slack_positive_steps
        return {
            "current": self._slack.clone(),
            "positive_fraction": positive.to(torch.float32) / self._cbf_steps.clamp_min(1),
            "mean_nonzero": self._slack_sum / positive.clamp_min(1),
            "max": self._slack_max.clone(),
            "solve_failures": self._solve_failures.clone(),
            "velocity_feasibility_failures": self._velocity_feasibility_failures.clone(),
        }

    @property
    def solver_metrics(self) -> dict[str, np.ndarray]:
        """Per-episode OSQP diagnostics for every vectorized environment.

        Values are reset alongside the environment.  The evaluation runner
        records them for completed episodes in ``evaluation_metadata.json``.
        """
        solve_count = self._solver_solve_count
        return {
            "solve_count": solve_count.copy(),
            "iteration_total": self._solver_iteration_total.copy(),
            "iteration_max": self._solver_iteration_max.copy(),
            "solve_time_total_s": self._solver_solve_time_total_s.copy(),
            "solve_time_max_s": self._solver_solve_time_max_s.copy(),
            "update_time_total_s": self._solver_update_time_total_s.copy(),
            "polish_time_total_s": self._solver_polish_time_total_s.copy(),
            "primal_residual_max": self._solver_primal_residual_max.copy(),
            "dual_residual_max": self._solver_dual_residual_max.copy(),
            "inaccurate_count": self._solver_inaccurate_count.copy(),
            "max_iteration_count": self._solver_max_iteration_count.copy(),
        }

    def process_actions(self, actions: torch.Tensor):
        """Hold the latest RL velocity target until the next navigation step."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {self.action_dim}, received {actions.shape[-1]}."
            )
        self._raw_actions[:] = actions * self._action_scales

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self._update_cbf_command()
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._nominal_acceleration[env_ids] = 0.0
        self._safe_acceleration_w[env_ids] = 0.0
        self._velocity_setpoint_w[env_ids] = 0.0
        self._velocity_setpoint_initialized[env_ids] = False
        self._slack[env_ids] = 0.0
        self._slack_positive_steps[env_ids] = 0
        self._slack_sum[env_ids] = 0.0
        self._slack_max[env_ids] = 0.0
        self._cbf_steps[env_ids] = 0
        self._solve_failures[env_ids] = 0
        self._velocity_feasibility_failures[env_ids] = 0
        metric_env_ids: slice | np.ndarray
        if isinstance(env_ids, slice):
            metric_env_ids = env_ids
        elif isinstance(env_ids, torch.Tensor):
            metric_env_ids = env_ids.detach().cpu().numpy()
        else:
            metric_env_ids = np.asarray(env_ids)
        self._solver_solve_count[metric_env_ids] = 0
        self._solver_iteration_total[metric_env_ids] = 0
        self._solver_iteration_max[metric_env_ids] = 0
        self._solver_solve_time_total_s[metric_env_ids] = 0.0
        self._solver_solve_time_max_s[metric_env_ids] = 0.0
        self._solver_update_time_total_s[metric_env_ids] = 0.0
        self._solver_polish_time_total_s[metric_env_ids] = 0.0
        self._solver_primal_residual_max[metric_env_ids] = 0.0
        self._solver_dual_residual_max[metric_env_ids] = 0.0
        self._solver_inaccurate_count[metric_env_ids] = 0
        self._solver_max_iteration_count[metric_env_ids] = 0
        self._low_level_action_term.reset(env_ids=env_ids)

    def _update_cbf_command(self) -> None:
        root_quat_w = self.robot.data.root_quat_w
        root_velocity_w = self.robot.data.root_lin_vel_w[:, :2]
        root_position_w = self.robot.data.root_pos_w[:, :2]
        uninitialized = ~self._velocity_setpoint_initialized
        if torch.any(uninitialized):
            self._velocity_setpoint_w[uninitialized] = root_velocity_w[uninitialized]
            self._velocity_setpoint_initialized[uninitialized] = True
        velocity_setpoint_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(root_quat_w),
            torch.cat((self._velocity_setpoint_w, torch.zeros_like(self._velocity_setpoint_w[:, :1])), dim=1),
        )[:, :2]
        nominal_acceleration_b = torch.clamp(
            self._kp * (self._raw_actions[:, :2] - velocity_setpoint_b),
            min=self._acceleration_lower,
            max=self._acceleration_upper,
        )
        self._nominal_acceleration[:] = nominal_acceleration_b

        nominal_acceleration_w = math_utils.quat_apply_yaw(
            root_quat_w, torch.cat((nominal_acceleration_b, torch.zeros_like(nominal_acceleration_b[:, :1])), dim=1)
        )[:, :2]

        hit_xy_w, ray_state = self._latest_lidar_capture()
        safe_acceleration_w = torch.empty_like(nominal_acceleration_w)
        slack = torch.zeros_like(self._slack)

        # The existing acceleration and velocity limits are body-frame boxes.
        # Apply the one-step velocity bounds to the persistent command state,
        # while the barrier itself continues to use measured robot velocity.
        velocity_acceleration_lower_b = (self._velocity_lower - velocity_setpoint_b) / self._control_dt
        velocity_acceleration_upper_b = (self._velocity_upper - velocity_setpoint_b) / self._control_dt
        effective_lower_b = torch.maximum(self._acceleration_lower, velocity_acceleration_lower_b)
        effective_upper_b = torch.minimum(self._acceleration_upper, velocity_acceleration_upper_b)
        feasible_velocity_bounds = torch.all(effective_lower_b <= effective_upper_b, dim=1)
        nominal_feasible_b = torch.clamp(nominal_acceleration_b, min=effective_lower_b, max=effective_upper_b)
        nominal_feasible_w = math_utils.quat_apply_yaw(
            root_quat_w, torch.cat((nominal_feasible_b, torch.zeros_like(nominal_feasible_b[:, :1])), dim=1)
        )[:, :2]

        for env_id in range(self.num_envs):
            if not feasible_velocity_bounds[env_id]:
                # A malformed or externally changed command reference can make
                # hard acceleration and one-step velocity bounds incompatible.
                # Preserve finite bounded commands and diagnose it.
                safe_acceleration_w[env_id] = nominal_acceleration_w[env_id]
                self._velocity_feasibility_failures[env_id] += 1
                continue

            obstacle_vectors_w = root_position_w[env_id].unsqueeze(0) - hit_xy_w[env_id]
            distances = torch.linalg.vector_norm(obstacle_vectors_w, dim=1)
            valid = (ray_state[env_id] == 2) & torch.isfinite(obstacle_vectors_w).all(dim=1)
            valid &= distances <= self.cfg.d_cbf_active
            valid &= distances > 1.0e-4
            obstacle_vectors_w = obstacle_vectors_w[valid]

            if obstacle_vectors_w.shape[0] == 0:
                # No active barriers: this is the closest acceleration that still
                # produces a velocity command inside the configured envelope.
                safe_acceleration_w[env_id] = nominal_feasible_w[env_id]
                continue
            if obstacle_vectors_w.shape[0] > self.cfg.max_lidar_points:
                nearest = torch.topk(torch.linalg.vector_norm(obstacle_vectors_w, dim=1), self.cfg.max_lidar_points,
                                     largest=False).indices
                obstacle_vectors_w = obstacle_vectors_w[nearest]

            velocity_w = root_velocity_w[env_id]
            squared_distance = torch.sum(obstacle_vectors_w.square(), dim=1)
            rhs = (
                -2.0 * torch.dot(velocity_w, velocity_w)
                - 2.0 * (self.cfg.gamma1 + self.cfg.gamma2) * (obstacle_vectors_w @ velocity_w)
                - self.cfg.gamma1 * self.cfg.gamma2 * (squared_distance - self.cfg.d_margin**2)
            )
            yaw_quat = math_utils.yaw_quat(root_quat_w[env_id].unsqueeze(0))
            x_axis_w = math_utils.quat_apply_yaw(yaw_quat, torch.tensor([[1.0, 0.0, 0.0]], device=self.device))[0, :2]
            y_axis_w = math_utils.quat_apply_yaw(yaw_quat, torch.tensor([[0.0, 1.0, 0.0]], device=self.device))[0, :2]
            body_to_world = torch.stack((x_axis_w, y_axis_w), dim=1)

            solution, env_slack, solver_stats = self._solvers[env_id].solve(
                nominal_acceleration_w[env_id].detach().cpu().numpy(),
                obstacle_vectors_w.detach().cpu().numpy(),
                rhs.detach().cpu().numpy(),
                body_to_world.detach().cpu().numpy(),
                self._acceleration_lower.detach().cpu().numpy(),
                self._acceleration_upper.detach().cpu().numpy(),
                velocity_acceleration_lower_b[env_id].detach().cpu().numpy(),
                velocity_acceleration_upper_b[env_id].detach().cpu().numpy(),
            )
            self._record_solver_stats(env_id, solver_stats)
            if solution is None:
                safe_acceleration_w[env_id] = nominal_acceleration_w[env_id]
                self._solve_failures[env_id] += 1
            else:
                safe_acceleration_w[env_id] = torch.as_tensor(solution, device=self.device, dtype=torch.float32)
                slack[env_id] = env_slack

        self._safe_acceleration_w[:] = safe_acceleration_w
        self._velocity_setpoint_w[:], self._processed_actions[:, :2] = integrate_velocity_setpoint_world(
            self._velocity_setpoint_w,
            safe_acceleration_w,
            root_quat_w,
            self._control_dt,
            self._velocity_lower,
            self._velocity_upper,
        )
        self._processed_actions[:, 2] = self._raw_actions[:, 2]
        self._slack[:] = slack
        self._cbf_steps += 1
        positive = slack > 0.0
        self._slack_positive_steps += positive.to(torch.long)
        self._slack_sum += torch.where(positive, slack, torch.zeros_like(slack))
        self._slack_max = torch.maximum(self._slack_max, slack)

    def _record_solver_stats(self, env_id: int, stats: _OsqpSolveStats) -> None:
        """Accumulate one host-side OSQP result without perturbing GPU timing."""
        self._solver_solve_count[env_id] += 1
        self._solver_iteration_total[env_id] += stats.iterations
        self._solver_iteration_max[env_id] = max(self._solver_iteration_max[env_id], stats.iterations)
        self._solver_solve_time_total_s[env_id] += stats.solve_time_s
        self._solver_solve_time_max_s[env_id] = max(self._solver_solve_time_max_s[env_id], stats.solve_time_s)
        self._solver_update_time_total_s[env_id] += stats.update_time_s
        self._solver_polish_time_total_s[env_id] += stats.polish_time_s
        self._solver_primal_residual_max[env_id] = max(self._solver_primal_residual_max[env_id], stats.primal_residual)
        self._solver_dual_residual_max[env_id] = max(self._solver_dual_residual_max[env_id], stats.dual_residual)
        status = stats.status.lower()
        self._solver_inaccurate_count[env_id] += "inaccurate" in status
        self._solver_max_iteration_count[env_id] += "maximum iterations reached" in status

    def _latest_lidar_capture(self) -> tuple[torch.Tensor, torch.Tensor]:
        collector = getattr(self._env, self.cfg.lidar_collector_name, None)
        if collector is None:
            raise RuntimeError(
                f"CBF PLAY requires the held LiDAR collector '{self.cfg.lidar_collector_name}'. "
                "Use the temporal-LiDAR PLAY environment entry point."
            )
        capture = collector.latest_capture()
        return capture["hit_xy"], capture["ray_state"]


@configclass
class StaticObstacleCbfPreTrainedPolicyActionCfg(KpPreTrainedPolicyActionCfg):
    """Configuration for the PLAY-only static-obstacle CBF action term."""

    class_type: type[ActionTerm] = StaticObstacleCbfPreTrainedPolicyAction
    d_margin: float = 0.70
    """Circular clearance radius around each valid LiDAR reflection, in metres."""
    d_cbf_active: float = 5.0
    """Maximum current-scan point distance considered by the CBF, in metres."""
    gamma1: float = 2.0
    """First relative-degree-two CBF gain in s^-1."""
    gamma2: float = 2.0
    """Second relative-degree-two CBF gain in s^-1."""
    slack_penalty: float = 1000.0
    """Positive shared-slack quadratic penalty rho."""
    max_lidar_points: int = 64
    """Maximum nearest valid active-range reflections retained per QP."""
    lidar_collector_name: str = "_held_scan_lidar_collector"
    """Name of the held-scan collector supplied by the temporal-LiDAR environment."""
    solver_eps_abs: float = 1.0e-3
    solver_eps_rel: float = 1.0e-3
    solver_max_iter: int = 500
    solver_polish: bool = True
    solver_warm_start: bool = True
