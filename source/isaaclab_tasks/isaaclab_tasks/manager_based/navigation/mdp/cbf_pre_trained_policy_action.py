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

from .kp_pre_trained_policy_action import (
    KpPreTrainedPolicyAction,
    KpPreTrainedPolicyActionCfg,
    zoh_average_acceleration_gain,
)

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

        self._cfg = cfg
        self._max_points = cfg.max_lidar_points
        # Variables are [u_world_x, u_world_y, delta_0, ..., delta_M-1].
        # Keeping all possible LiDAR-slack variables structurally present lets
        # OSQP reuse its factorization while individual rays become active.
        self._num_variables = 2 + self._max_points
        self._num_rows = 2 * self._max_points + 2
        self._slack_rows = slice(self._max_points, 2 * self._max_points)
        self._accel_rows = slice(2 * self._max_points, 2 * self._max_points + 2)

        p = sparse.diags(np.concatenate((np.array([2.0, 2.0]), np.zeros(self._max_points))), format="csc")
        # Keep every A entry structurally present so OSQP can update all point
        # coefficients without reconstructing/factorizing the problem pattern.
        a = sparse.csc_matrix(
            (
                np.ones(self._num_rows * self._num_variables),
                np.tile(np.arange(self._num_rows), self._num_variables),
                np.arange(0, self._num_rows * self._num_variables + 1, self._num_rows),
            ),
            shape=(self._num_rows, self._num_variables),
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
            q=np.zeros(self._num_variables),
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
        barrier_offset: np.ndarray,
        body_to_world: np.ndarray,
        acceleration_lower_b: np.ndarray,
        acceleration_upper_b: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, _OsqpSolveStats]:
        """Solve the soft CBF QP and return its result with OSQP diagnostics."""
        matrix = np.zeros((self._num_rows, self._num_variables), dtype=np.float64)
        lower = np.full(self._num_rows, -np.inf, dtype=np.float64)
        upper = np.full(self._num_rows, np.inf, dtype=np.float64)

        num_points = obstacle_vectors_w.shape[0]
        matrix[:num_points, :2] = 2.0 * obstacle_vectors_w
        matrix[np.arange(num_points), 2 + np.arange(num_points)] = 1.0
        # Canonical residual: 2 r.T u + b >= -delta.
        lower[:num_points] = -barrier_offset

        slack_indices = np.arange(self._max_points)
        matrix[self._max_points + slack_indices, 2 + slack_indices] = 1.0
        lower[self._slack_rows] = 0.0

        # u_body = R_wb.T u_world. Bounds are component-wise in body frame.
        world_to_body = body_to_world.T
        matrix[self._accel_rows, :2] = world_to_body
        lower[self._accel_rows] = acceleration_lower_b
        upper[self._accel_rows] = acceleration_upper_b

        q = np.zeros(self._num_variables)
        q[:2] = -2.0 * nominal_acceleration_w
        # Normalize the L1 cost by active constraints so scan density does not
        # change the acceleration/slack tradeoff.
        q[2 : 2 + num_points] = self._cfg.slack_penalty / num_points

        # CSC stores columns consecutively, hence the transpose before flatten.
        self._solver.update(q=q, Ax=matrix.T.reshape(-1), l=lower, u=upper)
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
            return None, np.zeros(num_points), stats
        return result.x[:2], np.maximum(0.0, result.x[2 : 2 + num_points]), stats


def velocity_command_from_average_acceleration(
    measured_velocity_b: torch.Tensor, acceleration_b: torch.Tensor, zoh_gain_s: float
) -> torch.Tensor:
    """Map average acceleration over one held interval to a body-frame velocity command."""
    return measured_velocity_b + zoh_gain_s * acceleration_b


def effective_zoh_acceleration_bounds(
    measured_velocity_b: torch.Tensor,
    zoh_gain_s: float,
    acceleration_lower_b: torch.Tensor,
    acceleration_upper_b: torch.Tensor,
    velocity_lower_b: torch.Tensor,
    velocity_upper_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect physical acceleration limits with ZOH command-envelope limits."""
    command_lower = (velocity_lower_b - measured_velocity_b) / zoh_gain_s
    command_upper = (velocity_upper_b - measured_velocity_b) / zoh_gain_s
    return torch.maximum(acceleration_lower_b, command_lower), torch.minimum(acceleration_upper_b, command_upper)


class StaticObstacleCbfPreTrainedPolicyAction(KpPreTrainedPolicyAction):
    """Kp navigation command filtered by a static-obstacle, soft CBF-QP.

    ``process_actions`` only stores the latest high-level RL command. At each
    low-level policy update, this term measures the robot velocity, constructs
    a nominal average acceleration, solves the QP, and maps it through the
    first-order locomotion model. The yaw-rate action bypasses both Kp and CBF.
    """

    cfg: StaticObstacleCbfPreTrainedPolicyActionCfg

    def __init__(self, cfg: StaticObstacleCbfPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if cfg.d_margin <= 0.0 or cfg.d_cbf_active <= 0.0:
            raise ValueError("d_margin and d_cbf_active must be positive.")
        if cfg.d_cbf_active < cfg.d_margin:
            raise ValueError("d_cbf_active must be no smaller than d_margin.")
        if cfg.gamma1 <= 0.0 or cfg.gamma2 <= 0.0 or cfg.slack_penalty <= 0.0 or cfg.tracking_tau_s <= 0.0:
            raise ValueError("CBF gains, slack_penalty, and tracking_tau_s must be positive.")
        if cfg.max_lidar_points < 1:
            raise ValueError("max_lidar_points must be at least one.")

        self._control_dt = cfg.low_level_decimation * env.physics_dt
        self._zoh_gain_s = zoh_average_acceleration_gain(self._control_dt, cfg.tracking_tau_s)
        self._solvers = [_OsqpStaticObstacleCbf(cfg) for _ in range(self.num_envs)]
        self._safe_acceleration_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._slack = torch.zeros(self.num_envs, device=self.device)
        self._mean_slack = torch.zeros(self.num_envs, device=self.device)
        self._minimum_barrier_residual = torch.zeros(self.num_envs, device=self.device)
        self._active_point_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command_envelope_guard_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
    def cbf_slack(self) -> torch.Tensor:
        """Current maximum per-point CBF slack for every vectorized environment."""
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
            "current_mean": self._mean_slack.clone(),
            "current_min_residual": self._minimum_barrier_residual.clone(),
            "active_point_count": self._active_point_count.clone(),
            "positive_fraction": positive.to(torch.float32) / self._cbf_steps.clamp_min(1),
            "mean_nonzero": self._slack_sum / positive.clamp_min(1),
            "max": self._slack_max.clone(),
            "solve_failures": self._solve_failures.clone(),
            "velocity_feasibility_failures": self._velocity_feasibility_failures.clone(),
            "command_envelope_guard_count": self._command_envelope_guard_count.clone(),
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
        self._slack[env_ids] = 0.0
        self._mean_slack[env_ids] = 0.0
        self._minimum_barrier_residual[env_ids] = 0.0
        self._active_point_count[env_ids] = 0
        self._command_envelope_guard_count[env_ids] = 0
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
        measured_velocity_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(root_quat_w),
            torch.cat((root_velocity_w, torch.zeros_like(root_velocity_w[:, :1])), dim=1),
        )[:, :2]
        nominal_acceleration_b = torch.clamp(
            self._kp * (self._raw_actions[:, :2] - measured_velocity_b),
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
        mean_slack = torch.zeros_like(self._mean_slack)
        minimum_residual = torch.zeros_like(self._minimum_barrier_residual)
        active_point_count = torch.zeros_like(self._active_point_count)

        # The ZOH model maps interval-average acceleration directly to the
        # next locomotion command. Intersect that command envelope with the
        # explicit physical acceleration box before the QP is solved.
        effective_lower_b, effective_upper_b = effective_zoh_acceleration_bounds(
            measured_velocity_b,
            self._zoh_gain_s,
            self._acceleration_lower,
            self._acceleration_upper,
            self._velocity_lower,
            self._velocity_upper,
        )
        feasible_velocity_bounds = torch.all(effective_lower_b <= effective_upper_b, dim=1)
        nominal_feasible_b = torch.clamp(nominal_acceleration_b, min=effective_lower_b, max=effective_upper_b)
        nominal_feasible_w = math_utils.quat_apply_yaw(
            root_quat_w, torch.cat((nominal_feasible_b, torch.zeros_like(nominal_feasible_b[:, :1])), dim=1)
        )[:, :2]

        for env_id in range(self.num_envs):
            if not feasible_velocity_bounds[env_id]:
                # Measured velocity can transiently lie outside the command
                # envelope. Keep the acceleration finite and bounded while
                # recording that no exact command-feasible solution exists.
                safe_acceleration_w[env_id] = nominal_feasible_w[env_id]
                self._velocity_feasibility_failures[env_id] += 1
                continue

            obstacle_vectors_w = root_position_w[env_id].unsqueeze(0) - hit_xy_w[env_id]
            distances = torch.linalg.vector_norm(obstacle_vectors_w, dim=1)
            valid = (ray_state[env_id] == 2) & torch.isfinite(obstacle_vectors_w).all(dim=1)
            valid &= distances <= self.cfg.d_cbf_active
            valid &= distances > 1.0e-4
            obstacle_vectors_w = obstacle_vectors_w[valid]

            if obstacle_vectors_w.shape[0] == 0:
                # No active barriers: use the closest command-feasible nominal.
                safe_acceleration_w[env_id] = nominal_feasible_w[env_id]
                continue
            if obstacle_vectors_w.shape[0] > self.cfg.max_lidar_points:
                nearest = torch.topk(torch.linalg.vector_norm(obstacle_vectors_w, dim=1), self.cfg.max_lidar_points,
                                     largest=False).indices
                obstacle_vectors_w = obstacle_vectors_w[nearest]
            active_point_count[env_id] = obstacle_vectors_w.shape[0]

            velocity_w = root_velocity_w[env_id]
            squared_distance = torch.sum(obstacle_vectors_w.square(), dim=1)
            barrier_offset = (
                2.0 * torch.dot(velocity_w, velocity_w)
                + 2.0 * (self.cfg.gamma1 + self.cfg.gamma2) * (obstacle_vectors_w @ velocity_w)
                + self.cfg.gamma1 * self.cfg.gamma2 * (squared_distance - self.cfg.d_margin**2)
            )
            yaw_quat = math_utils.yaw_quat(root_quat_w[env_id].unsqueeze(0))
            x_axis_w = math_utils.quat_apply_yaw(yaw_quat, torch.tensor([[1.0, 0.0, 0.0]], device=self.device))[0, :2]
            y_axis_w = math_utils.quat_apply_yaw(yaw_quat, torch.tensor([[0.0, 1.0, 0.0]], device=self.device))[0, :2]
            body_to_world = torch.stack((x_axis_w, y_axis_w), dim=1)

            solution, point_slack, solver_stats = self._solvers[env_id].solve(
                nominal_acceleration_w[env_id].detach().cpu().numpy(),
                obstacle_vectors_w.detach().cpu().numpy(),
                barrier_offset.detach().cpu().numpy(),
                body_to_world.detach().cpu().numpy(),
                effective_lower_b[env_id].detach().cpu().numpy(),
                effective_upper_b[env_id].detach().cpu().numpy(),
            )
            self._record_solver_stats(env_id, solver_stats)
            if solution is None:
                safe_acceleration_w[env_id] = math_utils.quat_apply_yaw(
                    root_quat_w[env_id].unsqueeze(0),
                    torch.cat(
                        (
                            torch.zeros_like(measured_velocity_b[env_id : env_id + 1]),
                            torch.zeros(1, 1, device=self.device),
                        ),
                        dim=1,
                    ),
                )[0, :2]
                self._solve_failures[env_id] += 1
            else:
                safe_acceleration_w[env_id] = torch.as_tensor(solution, device=self.device, dtype=torch.float32)
                point_residual = 2.0 * (obstacle_vectors_w @ safe_acceleration_w[env_id]) + barrier_offset
                slack[env_id] = float(np.max(point_slack))
                mean_slack[env_id] = float(np.mean(point_slack))
                minimum_residual[env_id] = torch.min(point_residual)

        self._safe_acceleration_w[:] = safe_acceleration_w
        safe_acceleration_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(root_quat_w),
            torch.cat((safe_acceleration_w, torch.zeros_like(safe_acceleration_w[:, :1])), dim=1),
        )[:, :2]
        commanded_velocity_b = velocity_command_from_average_acceleration(
            measured_velocity_b, safe_acceleration_b, self._zoh_gain_s
        )
        command_guard = (commanded_velocity_b < self._velocity_lower - 1.0e-5) | (
            commanded_velocity_b > self._velocity_upper + 1.0e-5
        )
        self._command_envelope_guard_count += torch.any(command_guard, dim=1).to(torch.long)
        self._processed_actions[:, :2] = torch.clamp(
            commanded_velocity_b, min=self._velocity_lower, max=self._velocity_upper
        )
        self._processed_actions[:, 2] = self._raw_actions[:, 2]
        self._slack[:] = slack
        self._mean_slack[:] = mean_slack
        self._minimum_barrier_residual[:] = minimum_residual
        self._active_point_count[:] = active_point_count
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
    d_margin: float = 1.0
    """Circular clearance radius around each valid LiDAR reflection, in metres."""
    d_cbf_active: float = 5.0
    """Maximum current-scan point distance considered by the CBF, in metres."""
    gamma1: float = 2.0
    """First relative-degree-two CBF gain in s^-1."""
    gamma2: float = 2.0
    """Second relative-degree-two CBF gain in s^-1."""
    slack_penalty: float = 1000.0
    """Positive coefficient rho for normalized per-point L1 slack."""
    max_lidar_points: int = 64
    """Maximum nearest valid active-range reflections retained per QP."""
    lidar_collector_name: str = "_held_scan_lidar_collector"
    """Name of the held-scan collector supplied by the temporal-LiDAR environment."""
    solver_eps_abs: float = 1.0e-3
    solver_eps_rel: float = 1.0e-3
    solver_max_iter: int = 500
    solver_polish: bool = True
    solver_warm_start: bool = True
