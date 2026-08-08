# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a mixed Go2 policy on the standardized dynamic-crowd benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

from evaluation import (  # isort: skip
    CollisionReplayRecorder,
    EpisodeMetricsCollector,
    EpisodeVelocityAccumulator,
    GOAL_REGION_COLLISION_RADIUS_M,
    dynamic_crowd_profiles,
    print_results,
    save_artifacts,
    terminal_goal_region_collision_ids,
)


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy in the dynamic-crowd benchmark.")
parser.add_argument("--task", type=str, required=True, help="Existing mixed obstacle-avoidance task ID.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL-agent config entry point.")
parser.add_argument("--num_envs", type=int, default=24, help="Vector environments (must be at least 24).")
parser.add_argument("--seed", type=int, default=42, help="Benchmark random seed.")
parser.add_argument(
    "--episodes_per_profile", type=int, default=50, help="Completed episodes for every scenario/count cell."
)
parser.add_argument("--output_dir", type=str, default=None, help="Artifact directory (defaults under the checkpoint).")
parser.add_argument(
    "--failure_history_seconds", type=float, default=3.0,
    help="Seconds of context before a pedestrian collision to retain in each replay.",
)
parser.add_argument(
    "--failure_output_dir", type=str, default=None,
    help="Collision replay directory (defaults to <output_dir>/failure_cases).",
)
parser.add_argument(
    "--disable_failure_recording", action="store_true",
    help="Do not save pedestrian-collision replay artifacts during evaluation.",
)
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the published checkpoint when available."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from packaging import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (  # noqa: E402
    EVALUATION_CROWD_LATERAL_HEADING_MAX,
    EVALUATION_CROWD_SPEED_RANGE,
    EVALUATION_SCENARIO_CODES,
    configure_dynamic_crowd_evaluation,
    install_dynamic_crowd_evaluation_profiles,
)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> tuple[str, str]:
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not checkpoint:
            raise RuntimeError(f"No published RSL-RL checkpoint is available for {train_task_name}.")
        return checkpoint, os.path.dirname(checkpoint)
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
        return checkpoint, os.path.dirname(checkpoint)
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return checkpoint, os.path.dirname(checkpoint)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run all dynamic-crowd profiles in parallel until every profile reaches its quota."""
    profiles = dynamic_crowd_profiles()
    if args_cli.num_envs < len(profiles):
        raise ValueError(f"--num_envs must be at least {len(profiles)} for the 24 benchmark profiles.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    configure_dynamic_crowd_evaluation(env_cfg)

    checkpoint, log_dir = _resolve_checkpoint(agent_cfg)
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else Path(log_dir) / "evaluations" / "dynamic_crowd"
    failure_output_dir = (
        Path(args_cli.failure_output_dir) if args_cli.failure_output_dir else output_dir / "failure_cases"
    )
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoint = handle_deprecated_rsl_rl_checkpoint(checkpoint, INSTALLED_RSL_RL_VERSION)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if version.parse(INSTALLED_RSL_RL_VERSION) < version.parse("4.0.0"):
        policy_nn = (
            runner.alg.policy
            if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("2.3.0")
            else runner.alg.actor_critic
        )

    env_profile_indices = [index % len(profiles) for index in range(args_cli.num_envs)]
    install_dynamic_crowd_evaluation_profiles(
        env.unwrapped,
        [profiles[index].pedestrian_count for index in env_profile_indices],
        [EVALUATION_SCENARIO_CODES[profiles[index].scenario] for index in env_profile_indices],
    )
    obs, _ = env.reset()
    collector = EpisodeMetricsCollector(profiles, env_profile_indices, args_cli.episodes_per_profile)
    velocity_accumulator = EpisodeVelocityAccumulator(args_cli.num_envs)
    goal_region_collision_ids: set[int] = set()

    # Some task variants do not publish command-manager metrics in ``extras["log"]``.
    # Capture the terminal state immediately before ManagerBasedRLEnv resets it, while the
    # per-step samples below provide the rest of each episode's world-XY speed trace.
    raw_env = env.unwrapped
    step_dt_s = env.unwrapped.step_dt
    episode_length_s = env.unwrapped.cfg.episode_length_s
    replay_recorder = None
    if not args_cli.disable_failure_recording:
        replay_recorder = CollisionReplayRecorder(
            profiles,
            env_profile_indices,
            failure_output_dir,
            step_dt_s,
            args_cli.failure_history_seconds,
            goal_region_radius_m=GOAL_REGION_COLLISION_RADIUS_M,
        )

    original_reset_idx = raw_env._reset_idx

    def _tracked_reset_idx(env_ids):
        terminal_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
        velocity_accumulator.record_terminal(terminal_speed, env_ids)
        goal_region_collision_ids.update(
            terminal_goal_region_collision_ids(raw_env, env_ids, GOAL_REGION_COLLISION_RADIUS_M)
        )
        if replay_recorder is not None:
            replay_recorder.capture_terminal_collisions(raw_env, env_ids)
        return original_reset_idx(env_ids)

    raw_env._reset_idx = _tracked_reset_idx

    print(
        f"[INFO] Evaluating {checkpoint} on {len(profiles)} dynamic-crowd profiles "
        f"with {args_cli.episodes_per_profile} episodes each."
    )
    try:
        while simulation_app.is_running() and not collector.complete:
            step_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
            velocity_accumulator.record_step(step_speed)
            with torch.inference_mode():
                actions = policy(obs)
                if replay_recorder is not None:
                    submitted_actions = actions
                    if env.clip_actions is not None:
                        submitted_actions = torch.clamp(submitted_actions, -env.clip_actions, env.clip_actions)
                    action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
                    action_scales = torch.as_tensor(action_term.cfg.action_scales, device=submitted_actions.device)
                    replay_recorder.record_pre_step(raw_env, submitted_actions * action_scales)
                obs, _, dones, extras = env.step(actions)
            if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
            # ``extras[\"log\"]`` is cleared to scalar zero on idle steps.  Use the RSL-RL
            # done mask as the authoritative completion source; the log is used only for the
            # terminal reasons of those confirmed environments.
            completed_ids = torch.nonzero(dones, as_tuple=False).reshape(-1)
            collector.consume(
                extras,
                velocity_accumulator.completed_means(completed_ids),
                completed_env_ids=completed_ids,
                goal_region_collision_env_ids=goal_region_collision_ids,
            )
            velocity_accumulator.reset(completed_ids)
            goal_region_collision_ids.difference_update(completed_ids.detach().cpu().tolist())
    finally:
        env.close()

    if not collector.complete:
        raise RuntimeError("Evaluation stopped before all benchmark profiles completed.")
    if collector.velocity_metric_source == "direct_world_xy_speed":
        print("[INFO] Mean XY speed was measured directly from the robot world-frame velocity.")
    elif collector.velocity_metric_source != "linear_velocity_xy":
        print(
            "[WARN] The task did not export linear_velocity_xy; using the legacy "
            f"{collector.velocity_metric_source} metric on the flat pedestrian corridor."
        )
    rows = collector.rows()
    aggregates = collector.aggregate_rows()
    artifact_dir = save_artifacts(
        output_dir,
        rows,
        aggregates,
        {
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "seed": agent_cfg.seed,
            "episodes_per_profile": args_cli.episodes_per_profile,
            "pedestrian_counts": sorted({profile.pedestrian_count for profile in profiles}),
            "scenarios": list(EVALUATION_SCENARIO_CODES),
            "crowd_speed_range_mps": EVALUATION_CROWD_SPEED_RANGE,
            "crowd_lateral_heading_max_deg": math.degrees(EVALUATION_CROWD_LATERAL_HEADING_MAX),
            "metrics": {
                "success_rate": "goal_reached term; collisions take precedence when simultaneous",
                "navigation_success_rate": "successes divided by episodes outside the terminal-goal buffer",
                "collision_rate": "pedestrian collisions outside the terminal-goal buffer",
                "goal_region_collision_rate": (
                    f"pedestrian collisions within {GOAL_REGION_COLLISION_RADIUS_M:.2f} m of the goal"
                ),
                "all_collision_rate": "all pedestrian collisions before goal-region classification",
                "mean_xy_speed_mps": "episode-average world-frame horizontal robot speed over all episodes",
            },
            "velocity_metric_source": collector.velocity_metric_source,
            "step_dt_s": step_dt_s,
            "episode_length_s": episode_length_s,
            "failure_replays": {
                "enabled": replay_recorder is not None,
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "history_seconds": args_cli.failure_history_seconds if replay_recorder is not None else None,
                "goal_region_radius_m": GOAL_REGION_COLLISION_RADIUS_M,
                "collision_cases": replay_recorder.case_count if replay_recorder is not None else 0,
            },
        },
    )
    print_results(rows, aggregates)
    print(f"[INFO] Wrote dynamic-crowd evaluation artifacts to: {artifact_dir}")
    if replay_recorder is not None:
        print(f"[INFO] Wrote {replay_recorder.case_count} collision replay(s) to: {failure_output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
