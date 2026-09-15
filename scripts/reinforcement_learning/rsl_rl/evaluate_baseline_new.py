# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a reactive baseline that is baked into the task itself, on the static-plus-dynamic benchmark.

Unlike ``evaluate_baseline.py``/``evaluate_dwa_baseline.py``, this script does not compute the desired
navigation command itself: the task's action term (``GreedyPreTrainedPolicyAction`` or
``DwaPreTrainedPolicyAction``) computes it internally each step from a single current-frame LiDAR
scan, bounded by the same Kp acceleration/velocity tracking the non-CBF "Kp" tasks already use, and
sends it straight to the frozen locomotion policy -- there is no CBF-QP safety filter in this path.
Which algorithm drives the robot is a property of ``--task``, the same way the locomotion checkpoint
already is. This script only steps the environment (with a dummy action the task ignores) and collects
the same ``dynamic_crowd_results.csv``/``.json`` metrics as the other baseline evaluators, so all three
remain directly comparable.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# Import torch before isaaclab/isaacsim: some deprecated Isaac Sim extensions (e.g.
# isaacsim.core.prims) import torch at module scope, and if torch isn't already cached in
# sys.modules by then, that import can resolve to Isaac Sim's own bundled (and here,
# ABI-incompatible) copy instead of this environment's, silently killing the app on startup.
import torch  # noqa: F401, E402

from isaaclab.app import AppLauncher

from evaluation import (  # isort: skip
    CollisionReplayRecorder,
    EpisodeMetricsCollector,
    EpisodeVelocityAccumulator,
    GOAL_REGION_COLLISION_RADIUS_M,
    dynamic_crowd_profiles,
    fixed_grid_profile_indices,
    print_results,
    save_artifacts,
    terminal_collision_ids,
    terminal_goal_region_collision_ids,
)


parser = argparse.ArgumentParser(
    description=(
        "Evaluate a reactive baseline that is baked into the task's action term (Greedy or DWA, "
        "selected by --task) on the fixed static-plus-dynamic benchmark."
    )
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Mixed-Static-Pedestrian-Kp-Greedy-Obstacle-Avoidance-Unitree-Go2-Play-v0",
    help="Self-driving baseline task ID (the Greedy or Dwa Play task registered in obstacle_avoidance).",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point (unused; required to resolve the task's Hydra config).")
parser.add_argument(
    "--num_envs", type=int, default=56,
    help=(
        "Vector environments. The static-plus-dynamic benchmark requires a positive multiple of "
        "56 (7 scenario columns x 8 count rows); defaults to one replica per cell."
    ),
)
parser.add_argument("--seed", type=int, default=42, help="Benchmark random seed.")
parser.add_argument(
    "--seeds", type=int, default=1,
    help=(
        "Number of consecutive benchmark seeds starting at --seed (each contributes an "
        "equal share of --episodes_per_profile, so total runtime is unchanged)."
    ),
)
parser.add_argument(
    "--episodes_per_profile", type=int, default=100, help="Completed episodes for every scenario/count cell."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Evaluation artifact root; each run creates a timestamped subdirectory (defaults under logs/).",
)
parser.add_argument("--clip_actions", type=float, default=100.0, help="Symmetric clip applied to the dummy action tensor (the task ignores its contents).")
parser.add_argument(
    "--failure_history_seconds", type=float, default=3.0,
    help="Seconds of context before a pedestrian collision to retain in each replay.",
)
parser.add_argument(
    "--success_cases_per_scenario",
    type=int,
    default=0,
    help=(
        "Interesting complete successful episodes to save for each scenario; "
        "0 disables success recording (default)."
    ),
)
parser.add_argument(
    "--timeout_cases_per_scenario",
    type=int,
    default=0,
    help=(
        "Complete timed-out episodes to save for each scenario; 0 disables timeout recording (default)."
    ),
)
parser.add_argument(
    "--interesting_interaction_distance_m",
    type=float,
    default=1.5,
    help="A success replay is sampled only if the robot comes within this distance of an active pedestrian.",
)
parser.add_argument(
    "--failure_output_dir",
    "--replay_output_dir",
    dest="replay_output_dir",
    type=str,
    default=None,
    help="Episode-replay root; each run creates a timestamped subdirectory (defaults to the evaluation run).",
)
parser.add_argument(
    "--disable_failure_recording", action="store_true",
    help="Do not save pedestrian-collision replay artifacts during evaluation.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (  # noqa: E402
    EVALUATION_SCENARIO_CODES,
    configure_static_dynamic_evaluation,
    install_dynamic_crowd_evaluation_profiles,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


STATIC_DYNAMIC_GRID_CELLS = 56


def _fixed_grid_profile_indices(env, profiles) -> list[int]:
    """Map row-major fixed terrain cells to column-major benchmark profiles."""
    terrain = env.scene["terrain"]
    levels = terrain.terrain_levels.detach().cpu().tolist()
    columns = terrain.terrain_types.detach().cpu().tolist()
    return fixed_grid_profile_indices(profiles, levels, columns)


def _create_timestamped_run_dir(output_root: Path) -> Path:
    """Create a unique, human-readable evaluation run directory."""
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
    raise RuntimeError(f"Could not create a unique evaluation run directory in {output_root}.")


class _ProgressReporter:
    """Throttled console progress print. No RL policy/W&B involved here, so unlike
    ``evaluate.py``'s reporter this only ever writes to stdout."""

    _INTERVAL_SECONDS = 10.0

    def __init__(self, total_episodes: int, seed_count: int):
        self.total_episodes = total_episodes
        self.seed_count = seed_count
        self.started_at = time.monotonic()
        self.last_report_at = 0.0

    def report(self, accepted_episodes: int, *, seed: int, seed_index: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report_at < self._INTERVAL_SECONDS:
            return
        self.last_report_at = now
        elapsed = max(0.0, now - self.started_at)
        rate = accepted_episodes / elapsed if elapsed else 0.0
        remaining = max(0, self.total_episodes - accepted_episodes)
        eta = f"{remaining / rate:.0f}s" if rate > 0.0 else "n/a"
        percent = round(100.0 * accepted_episodes / self.total_episodes, 1) if self.total_episodes else 100.0
        print(
            f"[EVAL] running: {accepted_episodes}/{self.total_episodes} episodes ({percent}%), "
            f"seed {seed} ({seed_index}/{self.seed_count}), elapsed {elapsed:.0f}s, ETA {eta}",
            flush=True,
        )


def _build_shadow_controller(action_term_cfg):
    """Build a controller matching the task's baked-in one, only to log its command for replays.

    The task's action term computes its desired command internally (see
    ``reactive_pre_trained_policy_action.py``), so unlike ``evaluate_baseline.py`` this script never
    computes the command that drives the robot. To still snapshot "state right before this command
    was applied" the same way ``CollisionReplayRecorder`` expects, this builds an identical
    controller from the same cfg fields the action term itself used, and calls it on the same
    pre-step env state -- a pure, deterministic function of that state, so it reproduces the exact
    value the action term will independently (re)compute moments later inside ``env.step()``.
    """
    if isinstance(action_term_cfg, nav_mdp.GreedyPreTrainedPolicyActionCfg):
        return nav_mdp.build_greedy_controller(action_term_cfg)
    if isinstance(action_term_cfg, nav_mdp.DwaPreTrainedPolicyActionCfg):
        return nav_mdp.build_dwa_controller(action_term_cfg)
    return None


def _controller_label(task_id: str) -> str:
    """Best-effort algorithm label for artifact metadata, derived from the task ID."""
    lowered = task_id.lower()
    if "-greedy-" in lowered:
        return "greedy_reactive_env_baked"
    if "-dwa-" in lowered:
        return "dwa_reactive_env_baked"
    return "reactive_env_baked"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run all static-plus-dynamic profiles in parallel until every profile reaches its quota."""
    profiles = dynamic_crowd_profiles(include_static=True, include_slow_leader=True, include_slow_crowd=True)
    if args_cli.num_envs <= 0 or args_cli.num_envs % STATIC_DYNAMIC_GRID_CELLS:
        raise ValueError(
            f"--num_envs must be a positive multiple of {STATIC_DYNAMIC_GRID_CELLS} "
            "for the 7-column x 8-row static-plus-dynamic benchmark."
        )
    if len(profiles) != STATIC_DYNAMIC_GRID_CELLS:
        raise RuntimeError(
            "The static-plus-dynamic benchmark requires all six dynamic scenario columns plus the static column."
        )

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    configure_static_dynamic_evaluation(env_cfg)

    log_root = Path("logs") / "reactive_baseline" / args_cli.task.replace(":", "_")
    output_root = Path(args_cli.output_dir) if args_cli.output_dir else log_root / "evaluations" / "reactive_baseline"
    output_dir = _create_timestamped_run_dir(output_root)
    if args_cli.replay_output_dir:
        failure_output_dir = _create_timestamped_run_dir(Path(args_cli.replay_output_dir)) / "episode_cases"
    else:
        failure_output_dir = output_dir / "episode_cases"

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=args_cli.clip_actions)

    env_profile_indices = _fixed_grid_profile_indices(env.unwrapped, profiles)
    install_dynamic_crowd_evaluation_profiles(
        env.unwrapped,
        [profiles[index].pedestrian_count for index in env_profile_indices],
        [EVALUATION_SCENARIO_CODES.get(profiles[index].scenario, -1) for index in env_profile_indices],
    )
    env.reset()
    raw_env = env.unwrapped
    step_dt_s = env.unwrapped.step_dt
    episode_length_s = env.unwrapped.cfg.episode_length_s

    # The task's action term computes its own desired command every step (see
    # reactive_pre_trained_policy_action.py); this dummy tensor only satisfies env.step()'s shape
    # requirement and is otherwise ignored.
    dummy_actions = torch.zeros(args_cli.num_envs, env.action_space.shape[-1], device=env.unwrapped.device)

    replay_recorder = None
    if not args_cli.disable_failure_recording or args_cli.success_cases_per_scenario or args_cli.timeout_cases_per_scenario:
        replay_recorder = CollisionReplayRecorder(
            profiles,
            env_profile_indices,
            failure_output_dir,
            step_dt_s,
            args_cli.failure_history_seconds,
            goal_region_radius_m=GOAL_REGION_COLLISION_RADIUS_M,
            successes_per_scenario=args_cli.success_cases_per_scenario,
            timeouts_per_scenario=args_cli.timeout_cases_per_scenario,
            episode_length_s=episode_length_s,
            record_collisions=not args_cli.disable_failure_recording,
            interesting_interaction_distance_m=args_cli.interesting_interaction_distance_m,
        )
    shadow_controller = _build_shadow_controller(env_cfg.actions.pre_trained_policy_action) if replay_recorder is not None else None

    def _record_cbf_replay_state() -> None:
        """Write final CBF velocity and acceleration state for the latest replay frame."""
        if replay_recorder is None:
            return
        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        command = getattr(action_term, "cbf_filtered_velocity_command", None)
        if command is not None:
            replay_recorder.record_cbf_filtered_command(command)
        nominal_acceleration = getattr(action_term, "nominal_acceleration", None)
        filtered_acceleration = getattr(action_term, "safe_acceleration", None)
        if nominal_acceleration is not None and filtered_acceleration is not None:
            replay_recorder.record_cbf_accelerations(nominal_acceleration, filtered_acceleration)

    collector = EpisodeMetricsCollector(profiles, env_profile_indices, args_cli.episodes_per_profile)
    seed_count = args_cli.seeds
    if seed_count < 1:
        raise ValueError("--seeds must be at least 1.")
    seeds = [args_cli.seed + index for index in range(seed_count)]
    per_seed_quota = math.ceil(args_cli.episodes_per_profile / seed_count)
    velocity_accumulator = EpisodeVelocityAccumulator(args_cli.num_envs)
    goal_region_collision_ids: set[int] = set()

    original_reset_idx = raw_env._reset_idx

    def _tracked_reset_idx(env_ids):
        terminal_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
        velocity_accumulator.record_terminal(terminal_speed, env_ids)
        collision_ids = terminal_collision_ids(raw_env, env_ids, profiles, env_profile_indices)
        goal_region_collision_ids.update(
            terminal_goal_region_collision_ids(
                raw_env, env_ids, GOAL_REGION_COLLISION_RADIUS_M, collision_env_ids=collision_ids
            )
        )
        if replay_recorder is not None:
            def _termination_ids(name):
                try:
                    term = raw_env.termination_manager.get_term(name)
                except KeyError:
                    return torch.empty(0, dtype=torch.long, device=env_ids.device)
                return torch.nonzero(term, as_tuple=False).reshape(-1)

            success_env_ids = _termination_ids("goal_reached")
            timeout_env_ids = _termination_ids("time_out")
            _record_cbf_replay_state()
            replay_recorder.capture_terminal_episodes(
                raw_env, env_ids, success_env_ids, collision_env_ids=collision_ids, timeout_env_ids=timeout_env_ids
            )
        return original_reset_idx(env_ids)

    raw_env._reset_idx = _tracked_reset_idx

    controller_label = _controller_label(args_cli.task)
    print(
        f"[INFO] Evaluating the in-environment '{controller_label}' baseline on {len(profiles)} "
        f"static-plus-dynamic profiles with {args_cli.episodes_per_profile} episodes each"
        + (f" across {seed_count} consecutive seeds ({seeds[0]}..{seeds[-1]})." if seed_count > 1 else ".")
    )
    progress_reporter = _ProgressReporter(
        total_episodes=len(profiles) * args_cli.episodes_per_profile, seed_count=seed_count
    )
    try:
        for seed_index, seed in enumerate(seeds):
            if seed_index > 0:
                env.unwrapped.seed(seed)
                print(f"[INFO] Advancing to seed {seed} (stage {seed_index + 1} of {seed_count}).")
            collector.set_stage_limit(min(args_cli.episodes_per_profile, per_seed_quota * (seed_index + 1)))
            while simulation_app.is_running() and not collector.stage_complete:
                step_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
                velocity_accumulator.record_step(step_speed)
                with torch.inference_mode():
                    if replay_recorder is not None and shadow_controller is not None:
                        shadow_command = shadow_controller.compute_actions(raw_env)
                        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
                        cbf_command = getattr(action_term, "cbf_filtered_velocity_command", None)
                        replay_recorder.record_pre_step(raw_env, shadow_command, cbf_filtered_command=cbf_command)
                    _, rewards, dones, extras = env.step(dummy_actions)
                    if replay_recorder is not None:
                        replay_recorder.record_reward(rewards)
                    _record_cbf_replay_state()
                completed_ids = torch.nonzero(dones, as_tuple=False).reshape(-1)
                collector.consume(
                    extras,
                    velocity_accumulator.completed_means(completed_ids),
                    completed_env_ids=completed_ids,
                    goal_region_collision_env_ids=goal_region_collision_ids,
                )
                velocity_accumulator.reset(completed_ids)
                goal_region_collision_ids.difference_update(completed_ids.detach().cpu().tolist())
                progress_reporter.report(collector.total_episodes, seed=seed, seed_index=seed_index + 1)
            progress_reporter.report(collector.total_episodes, seed=seed, seed_index=seed_index + 1, force=True)
            print(f"[INFO] Seed {seed} stage complete: {collector.total_episodes} episodes accepted.")
    finally:
        env.close()

    if not collector.complete:
        raise RuntimeError("Evaluation stopped before all benchmark profiles completed.")

    rows = collector.rows()
    aggregates = collector.aggregate_rows()
    replicas = args_cli.num_envs // STATIC_DYNAMIC_GRID_CELLS
    for row in [*rows, *aggregates]:
        row["terrain_replicas"] = replicas

    artifact_dir = save_artifacts(
        output_dir,
        rows,
        aggregates,
        {
            "controller": controller_label,
            "task": args_cli.task,
            "seed": args_cli.seed,
            "seeds": seeds,
            "seed_count": seed_count,
            "run_id": output_dir.name,
            "output_root": str(output_root),
            "episodes_per_profile": args_cli.episodes_per_profile,
            "scenarios": [
                "static_obstacles", "crossing", "with_flow", "against_flow",
                "with_flow_slow_leader", "crossing_slow", "against_flow_slow",
            ],
            "terrain_grid": {
                "columns": 7,
                "rows": 8,
                "cells": STATIC_DYNAMIC_GRID_CELLS,
                "replicas": replicas,
                "static_column": 0,
                "dynamic_columns": 6,
                "count_levels": [2, 4, 6, 8, 10, 12, 14, 16],
            },
            "note": (
                "Controller hyperparameters live on the task's env cfg "
                "(actions.pre_trained_policy_action), not this script's CLI, since the task's action "
                "term computes the command internally. Override via Hydra, e.g. "
                "'env.actions.pre_trained_policy_action.alpha=0.8'."
            ),
            "failure_replays": {
                "enabled": replay_recorder is not None and not args_cli.disable_failure_recording,
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "history_seconds": args_cli.failure_history_seconds if replay_recorder is not None else None,
                "collision_cases": replay_recorder.collision_case_count if replay_recorder is not None else 0,
            },
            "success_replays": {
                "enabled": replay_recorder is not None and bool(args_cli.success_cases_per_scenario),
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "success_cases": replay_recorder.success_case_count if replay_recorder is not None else 0,
            },
            "timeout_replays": {
                "enabled": replay_recorder is not None and bool(args_cli.timeout_cases_per_scenario),
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "timeout_cases": replay_recorder.timeout_case_count if replay_recorder is not None else 0,
            },
            "metrics": {
                "success_rate": "goal_reached term; collisions take precedence when simultaneous",
                "navigation_success_rate": "successes divided by episodes outside the terminal-goal buffer",
                "collision_rate": (
                    "pedestrian collisions outside the terminal-goal buffer for dynamic scenarios; "
                    "base-contact static-obstacle collisions for static_obstacles"
                ),
                "goal_region_collision_rate": f"benchmark collisions within {GOAL_REGION_COLLISION_RADIUS_M:.2f} m of the goal",
                "all_collision_rate": "all benchmark collisions before goal-region classification",
                "timeout_rate": "episodes terminated by the time_out term",
                "base_contact_rate": "episodes terminated by the base_contact term",
                "mean_xy_speed_mps": "episode-average world-frame horizontal robot speed over all episodes",
            },
            "velocity_metric_source": collector.velocity_metric_source,
            "step_dt_s": env.unwrapped.step_dt,
            "episode_length_s": env.unwrapped.cfg.episode_length_s,
        },
    )
    print_results(rows, aggregates)
    print(f"[INFO] Wrote reactive-baseline evaluation artifacts to: {artifact_dir}")
    if replay_recorder is not None:
        print(
            f"[INFO] Wrote {replay_recorder.collision_case_count} collision replay(s), "
            f"{replay_recorder.success_case_count} complete success replay(s), and "
            f"{replay_recorder.timeout_case_count} complete timeout replay(s) to: {failure_output_dir}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
