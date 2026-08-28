"""Local-only diagnosis for the static-obstacle CBF on frozen pedestrian scenes.

This intentionally does *not* extend ``evaluate.py``: the latter is the cloud
benchmark entry point and publishes Research Agent/W&B evaluation telemetry.
This script freezes the social-force crowd in the local Isaac Lab process and
writes self-contained diagnostic artifacts under ``logs/local_diagnostics``.

The ordinary crossing, with-flow, and against-flow profile geometries are kept
so that a CBF failure can be compared directly with the dynamic benchmark.  The
pedestrians themselves have zero velocity for the whole episode.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

from evaluation import (  # isort: skip
    CollisionReplayRecorder,
    EpisodeMetricsCollector,
    EpisodeVelocityAccumulator,
    _json_safe,
    dynamic_crowd_profiles,
    print_results,
    save_artifacts,
    terminal_goal_region_collision_ids,
)


parser = argparse.ArgumentParser(
    description="Run the CBF locally against frozen pedestrians and save controller diagnostics."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Static-Obstacle-Cbf-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    help="CBF-compatible Isaac Lab task ID. This script is local-only.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL-agent config entry point.")
parser.add_argument("--num_envs", type=int, default=24, help="One vector environment per ordinary profile.")
parser.add_argument("--episodes_per_profile", type=int, default=3, help="Completed frozen-crowd episodes per profile.")
parser.add_argument(
    "--pedestrian_counts",
    type=int,
    nargs="+",
    default=list(range(2, 17, 2)),
    help="Pedestrian counts used for each crossing/with-flow/against-flow profile.",
)
parser.add_argument("--seed", type=int, default=42, help="Local diagnostic seed.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/local_diagnostics/static_cbf",
    help="Local artifact root. A timestamped run directory is created below it.",
)
parser.add_argument("--failure_history_seconds", type=float, default=4.0, help="Replay context retained before a collision.")
parser.add_argument(
    "--candidate_distance_m",
    type=float,
    default=2.0,
    help="Only analyze possible opposite-direction commands when the nearest frozen pedestrian is within this range.",
)
parser.add_argument(
    "--minimum_command_speed_mps",
    type=float,
    default=0.05,
    help="Ignore nearly-zero navigation/CBF arrows in the opposite-direction analysis.",
)
parser.add_argument(
    "--minimum_filter_change_mps",
    type=float,
    default=0.05,
    help="Require the CBF to have materially changed the navigation command before flagging a frame.",
)
parser.add_argument(
    "--trace_stride",
    type=int,
    default=1,
    help="Store one trace sample every N environment steps (default: every step).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if os.environ.get("RESEARCH_EXPERIMENT_ID") or os.environ.get("RESEARCH_AGENT_EVALUATION_ATTEMPT_ID"):
    parser.error("static_cbf_diagnostic.py is local-only and refuses a Research Agent cloud evaluation environment.")
if args_cli.episodes_per_profile < 1:
    parser.error("--episodes_per_profile must be at least one.")
if args_cli.trace_stride < 1:
    parser.error("--trace_stride must be at least one.")
if not args_cli.checkpoint:
    parser.error("--checkpoint is required for the local diagnostic.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (  # noqa: E402
    EVALUATION_SCENARIO_CODES,
    configure_dynamic_crowd_evaluation,
    install_dynamic_crowd_evaluation_profiles,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _create_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    for sequence in range(1_000):
        suffix = "" if sequence == 0 else f"_{sequence:02d}"
        run_dir = output_root / f"{timestamp}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"Could not create a local diagnostic directory in {output_root}.")


def _body_velocity_to_world(command_b: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Convert planar body-frame commands to world frame using the current yaw."""
    vector_b = torch.cat((command_b[:, :2], torch.zeros_like(command_b[:, :1])), dim=1)
    return math_utils.quat_apply_yaw(root_quat_w, vector_b)[:, :2]


class StaticCbfTrace:
    """Keep compact per-environment controller samples and flag contradictory commands."""

    def __init__(self, profiles: list[Any], env_profile_indices: list[int], output_dir: Path) -> None:
        self._profiles = profiles
        self._profile_indices = np.asarray(env_profile_indices, dtype=np.int16)
        self._output_dir = output_dir
        self._samples: dict[str, list[np.ndarray]] = {
            "step": [],
            "profile_index": [],
            "robot_position_xy": [],
            "robot_velocity_xy_world": [],
            "navigation_velocity_body": [],
            "cbf_velocity_body": [],
            "navigation_velocity_xy_world": [],
            "cbf_velocity_xy_world": [],
            "nearest_pedestrian_position_xy": [],
            "nearest_pedestrian_velocity_xy_world": [],
            "nearest_pedestrian_distance_m": [],
            "cbf_slack": [],
            "cbf_mean_slack": [],
            "cbf_minimum_residual": [],
            "cbf_active_point_count": [],
            "cbf_solve_failures": [],
            "cbf_velocity_feasibility_failures": [],
        }
        self.candidates: list[dict[str, Any]] = []
        self.sample_count = 0
        self._step = 0

    def record(
        self,
        raw_env: Any,
        navigation_velocity_body: torch.Tensor,
        *,
        candidate_distance_m: float,
        minimum_command_speed_mps: float,
        minimum_filter_change_mps: float,
        trace_stride: int,
    ) -> None:
        self._step += 1
        if self._step % trace_stride:
            return

        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        cbf_velocity_body = getattr(action_term, "cbf_filtered_velocity_command", None)
        if cbf_velocity_body is None:
            raise RuntimeError("The selected task did not expose a CBF-filtered command.")
        robot = raw_env.scene["robot"]
        robot_pos = robot.data.root_pos_w[:, :2]
        robot_vel = robot.data.root_lin_vel_w[:, :2]
        nav_world = _body_velocity_to_world(navigation_velocity_body, robot.data.root_quat_w)
        cbf_world = _body_velocity_to_world(cbf_velocity_body, robot.data.root_quat_w)

        crowd = raw_env.crowd_manager
        ped_pos = crowd.get_world_positions()
        ped_vel = crowd.get_velocities()
        active = crowd.get_active_mask()
        offsets = ped_pos - robot_pos.unsqueeze(1)
        distances = torch.linalg.vector_norm(offsets, dim=-1)
        distances = torch.where(active, distances, torch.full_like(distances, float("inf")))
        nearest_distance, nearest_index = distances.min(dim=1)
        env_indices = torch.arange(raw_env.num_envs, device=raw_env.device)
        nearest_offset = offsets[env_indices, nearest_index]
        nearest_pos = ped_pos[env_indices, nearest_index]
        nearest_vel = ped_vel[env_indices, nearest_index]

        slack_metrics = getattr(action_term, "slack_metrics", {})
        slack = slack_metrics.get("current", torch.zeros(raw_env.num_envs, device=raw_env.device))
        mean_slack = slack_metrics.get("current_mean", torch.zeros(raw_env.num_envs, device=raw_env.device))
        minimum_residual = slack_metrics.get(
            "current_min_residual", torch.zeros(raw_env.num_envs, device=raw_env.device)
        )
        active_point_count = slack_metrics.get(
            "active_point_count", torch.zeros(raw_env.num_envs, device=raw_env.device, dtype=torch.long)
        )
        solve_failures = slack_metrics.get("solve_failures", torch.zeros(raw_env.num_envs, device=raw_env.device))
        feasibility_failures = slack_metrics.get(
            "velocity_feasibility_failures", torch.zeros(raw_env.num_envs, device=raw_env.device)
        )

        self._append("step", np.full(raw_env.num_envs, self._step, dtype=np.int32))
        self._append("profile_index", self._profile_indices)
        self._append("robot_position_xy", robot_pos)
        self._append("robot_velocity_xy_world", robot_vel)
        self._append("navigation_velocity_body", navigation_velocity_body)
        self._append("cbf_velocity_body", cbf_velocity_body)
        self._append("navigation_velocity_xy_world", nav_world)
        self._append("cbf_velocity_xy_world", cbf_world)
        self._append("nearest_pedestrian_position_xy", nearest_pos)
        self._append("nearest_pedestrian_velocity_xy_world", nearest_vel)
        self._append("nearest_pedestrian_distance_m", nearest_distance)
        self._append("cbf_slack", slack)
        self._append("cbf_mean_slack", mean_slack)
        self._append("cbf_minimum_residual", minimum_residual)
        self._append("cbf_active_point_count", active_point_count)
        self._append("cbf_solve_failures", solve_failures)
        self._append("cbf_velocity_feasibility_failures", feasibility_failures)
        self.sample_count += raw_env.num_envs

        nav_projection = torch.sum(nav_world * nearest_offset, dim=1)
        cbf_projection = torch.sum(cbf_world * nearest_offset, dim=1)
        nav_speed = torch.linalg.vector_norm(nav_world, dim=1)
        cbf_speed = torch.linalg.vector_norm(cbf_world, dim=1)
        filter_change = torch.linalg.vector_norm(cbf_world - nav_world, dim=1)
        candidate_mask = (
            torch.isfinite(nearest_distance)
            & (nearest_distance <= candidate_distance_m)
            & (nav_speed >= minimum_command_speed_mps)
            & (cbf_speed >= minimum_command_speed_mps)
            & (filter_change >= minimum_filter_change_mps)
            & (nav_projection < 0.0)
            & (cbf_projection > 0.0)
        )
        for env_id in torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1).detach().cpu().tolist():
            profile = self._profiles[self._profile_indices[env_id]]
            self.candidates.append(
                {
                    "step": self._step,
                    "env_id": env_id,
                    "profile": {"scenario": profile.scenario, "pedestrian_count": profile.pedestrian_count},
                    "nearest_pedestrian_distance_m": float(nearest_distance[env_id].item()),
                    "navigation_projection_toward_nearest": float(nav_projection[env_id].item()),
                    "cbf_projection_toward_nearest": float(cbf_projection[env_id].item()),
                    "navigation_velocity_body": navigation_velocity_body[env_id].detach().cpu().tolist(),
                    "cbf_velocity_body": cbf_velocity_body[env_id].detach().cpu().tolist(),
                    "navigation_velocity_xy_world": nav_world[env_id].detach().cpu().tolist(),
                    "cbf_velocity_xy_world": cbf_world[env_id].detach().cpu().tolist(),
                    "robot_position_xy": robot_pos[env_id].detach().cpu().tolist(),
                    "nearest_pedestrian_position_xy": nearest_pos[env_id].detach().cpu().tolist(),
                    "nearest_pedestrian_velocity_xy_world": nearest_vel[env_id].detach().cpu().tolist(),
                    "cbf_slack": float(slack[env_id].item()),
                    "cbf_mean_slack": float(mean_slack[env_id].item()),
                    "cbf_minimum_residual": float(minimum_residual[env_id].item()),
                    "cbf_active_point_count": int(active_point_count[env_id].item()),
                    "cbf_solve_failures": int(solve_failures[env_id].item()),
                    "cbf_velocity_feasibility_failures": int(feasibility_failures[env_id].item()),
                }
            )

    def _append(self, name: str, value: torch.Tensor | np.ndarray) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        self._samples[name].append(np.asarray(value).copy())

    def write(self, metadata: dict[str, Any]) -> dict[str, Any]:
        trace_path = self._output_dir / "static_cbf_trace.npz"
        np.savez_compressed(
            trace_path,
            **{
                name: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
                for name, values in self._samples.items()
            },
        )
        candidates_path = self._output_dir / "opposite_direction_candidates.json"
        candidates_path.write_text(json.dumps(_json_safe(self.candidates), indent=2), encoding="utf-8")
        analysis = {
            **metadata,
            "trace_file": trace_path.name,
            "candidate_file": candidates_path.name,
            "trace_samples": self.sample_count,
            "opposite_direction_candidate_count": len(self.candidates),
            "interpretation": (
                "A candidate means the submitted navigation velocity pointed away from the nearest frozen "
                "pedestrian while the post-step CBF setpoint pointed toward it. It is a diagnostic flag, "
                "not proof by itself: inspect its replay and CBF slack/failure counters."
            ),
        }
        (self._output_dir / "static_cbf_analysis.json").write_text(
            json.dumps(_json_safe(analysis), indent=2), encoding="utf-8"
        )
        return analysis


def _freeze_crowd(raw_env: Any) -> None:
    """Freeze every pedestrian after every reset without changing cloud task code."""
    crowd = raw_env.crowd_manager

    def _static_step(_self, *, dt: float, robot_pos: torch.Tensor) -> None:
        del dt, robot_pos

    crowd.step = MethodType(_static_step, crowd)
    original_reset_idx = raw_env._reset_idx

    def _static_reset_idx(env_ids):
        result = original_reset_idx(env_ids)
        crowd.vel[env_ids] = 0.0
        crowd.desired_speed[env_ids] = 0.0
        crowd.goal[env_ids] = crowd.pos[env_ids]
        raw_env._write_pedestrians_to_sim()
        return result

    raw_env._reset_idx = _static_reset_idx


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    profiles = dynamic_crowd_profiles(args_cli.pedestrian_counts, include_slow_leader=False, include_slow_crowd=False)
    if args_cli.num_envs < len(profiles):
        raise ValueError(f"--num_envs must be at least {len(profiles)} for the selected ordinary profiles.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    configure_dynamic_crowd_evaluation(env_cfg)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    output_dir = _create_run_dir(Path(args_cli.output_dir))
    env_cfg.log_dir = str(output_dir)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    checkpoint = handle_deprecated_rsl_rl_checkpoint(checkpoint, INSTALLED_RSL_RL_VERSION)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)
    policy_nn = None
    if version.parse(INSTALLED_RSL_RL_VERSION) < version.parse("4.0.0"):
        policy_nn = runner.alg.policy if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("2.3.0") else runner.alg.actor_critic

    env_profile_indices = [index % len(profiles) for index in range(args_cli.num_envs)]
    install_dynamic_crowd_evaluation_profiles(
        raw_env,
        [profiles[index].pedestrian_count for index in env_profile_indices],
        [EVALUATION_SCENARIO_CODES[profiles[index].scenario] for index in env_profile_indices],
    )
    _freeze_crowd(raw_env)

    collector = EpisodeMetricsCollector(profiles, env_profile_indices, args_cli.episodes_per_profile)
    trace = StaticCbfTrace(profiles, env_profile_indices, output_dir)
    velocity_accumulator = EpisodeVelocityAccumulator(args_cli.num_envs)
    goal_region_collision_ids: set[int] = set()
    replay_dir = output_dir / "episode_cases"
    replay_recorder = CollisionReplayRecorder(
        profiles,
        env_profile_indices,
        replay_dir,
        env.unwrapped.step_dt,
        args_cli.failure_history_seconds,
        record_collisions=True,
        successes_per_scenario=0,
        episode_length_s=env.unwrapped.cfg.episode_length_s,
    )

    original_reset_idx = raw_env._reset_idx

    def _record_cbf_replay_state() -> None:
        """Attach the CBF's post-step velocity and acceleration outputs to replays."""
        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        filtered = getattr(action_term, "cbf_filtered_velocity_command", None)
        if filtered is not None:
            replay_recorder.record_cbf_filtered_command(filtered)
        nominal_acceleration = getattr(action_term, "nominal_acceleration", None)
        filtered_acceleration = getattr(action_term, "safe_acceleration", None)
        if nominal_acceleration is not None and filtered_acceleration is not None:
            replay_recorder.record_cbf_accelerations(nominal_acceleration, filtered_acceleration)

    def _tracked_reset_idx(env_ids):
        terminal_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
        velocity_accumulator.record_terminal(terminal_speed, env_ids)
        goal_region_collision_ids.update(terminal_goal_region_collision_ids(raw_env, env_ids))
        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        _record_cbf_replay_state()
        success_env_ids = torch.nonzero(raw_env.termination_manager.get_term("goal_reached"), as_tuple=False).reshape(-1)
        replay_recorder.capture_terminal_episodes(raw_env, env_ids, success_env_ids)
        return original_reset_idx(env_ids)

    raw_env._reset_idx = _tracked_reset_idx
    obs, _ = env.reset()
    print(
        f"[LOCAL STATIC CBF] Evaluating {checkpoint} on {len(profiles)} frozen-pedestrian profiles "
        f"with {args_cli.episodes_per_profile} episode(s) each. Artifacts: {output_dir}",
        flush=True,
    )
    try:
        while simulation_app.is_running() and not collector.complete:
            step_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
            velocity_accumulator.record_step(step_speed)
            with torch.inference_mode():
                actions = policy(obs)
                submitted_actions = actions
                if env.clip_actions is not None:
                    submitted_actions = torch.clamp(submitted_actions, -env.clip_actions, env.clip_actions)
                action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
                action_scales = torch.as_tensor(action_term.cfg.action_scales, device=submitted_actions.device)
                navigation_velocity_body = submitted_actions * action_scales
                replay_recorder.record_pre_step(
                    raw_env,
                    navigation_velocity_body,
                    cbf_filtered_command=getattr(action_term, "cbf_filtered_velocity_command", None),
                )
                obs, _, dones, extras = env.step(actions)
                _record_cbf_replay_state()
                trace.record(
                    raw_env,
                    navigation_velocity_body,
                    candidate_distance_m=args_cli.candidate_distance_m,
                    minimum_command_speed_mps=args_cli.minimum_command_speed_mps,
                    minimum_filter_change_mps=args_cli.minimum_filter_change_mps,
                    trace_stride=args_cli.trace_stride,
                )
            if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                assert policy_nn is not None
                policy_nn.reset(dones)
            completed_ids = torch.nonzero(dones, as_tuple=False).reshape(-1)
            collector.consume(
                extras,
                velocity_accumulator.completed_means(completed_ids),
                completed_env_ids=completed_ids,
                goal_region_collision_env_ids=goal_region_collision_ids,
            )
            velocity_accumulator.reset(completed_ids)
            goal_region_collision_ids.difference_update(completed_ids.detach().cpu().tolist())
            if collector.total_episodes and collector.total_episodes % len(profiles) == 0:
                print(
                    f"[LOCAL STATIC CBF] {collector.total_episodes}/{len(profiles) * args_cli.episodes_per_profile} "
                    "episodes accepted.",
                    flush=True,
                )
    finally:
        env.close()

    if not collector.complete:
        raise RuntimeError("Local static CBF diagnostic stopped before all profiles completed.")
    rows = collector.rows()
    aggregate_rows = collector.aggregate_rows()
    save_artifacts(
        output_dir,
        rows,
        aggregate_rows,
        {
            "mode": "local_static_cbf_diagnostic",
            "checkpoint": str(checkpoint),
            "seed": args_cli.seed,
            "pedestrian_motion": "frozen: crowd_manager.step is a local no-op after reset",
            "profile_count": len(profiles),
            "episodes_per_profile": args_cli.episodes_per_profile,
            "task": args_cli.task,
        },
    )
    analysis = trace.write(
        {
            "mode": "local_static_cbf_diagnostic",
            "checkpoint": str(checkpoint),
            "profiles": [{"scenario": profile.scenario, "pedestrian_count": profile.pedestrian_count} for profile in profiles],
            "candidate_definition": {
                "nearest_pedestrian_distance_m": args_cli.candidate_distance_m,
                "minimum_command_speed_mps": args_cli.minimum_command_speed_mps,
                "minimum_filter_change_mps": args_cli.minimum_filter_change_mps,
                "navigation_projection": "negative means away from nearest frozen pedestrian",
                "cbf_projection": "positive means toward nearest frozen pedestrian",
            },
        }
    )
    print_results(rows, aggregate_rows)
    print(
        "[LOCAL STATIC CBF] Analysis complete: "
        f"{analysis['opposite_direction_candidate_count']} opposite-direction candidate frame(s). "
        f"Open {output_dir / analysis['candidate_file']} and collision replays under {replay_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
