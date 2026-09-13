# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the non-learning greedy LiDAR baseline on the static-plus-dynamic benchmark.

This mirrors ``evaluate.py``'s benchmark setup and metrics (same task family, same
``dynamic_crowd_results.csv``/``.json`` schema) so a greedy reactive controller can be compared
directly against a trained RL+CBF policy. Unlike ``evaluate.py`` it has no policy checkpoint to
load, and it skips the CBF-QP diagnostics, collision-replay clips, and leader/interaction-event
tracking that are specific to analyzing a trained policy's behavior.
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
    description="Evaluate the greedy LiDAR baseline on the fixed static-plus-dynamic benchmark."
)
parser.add_argument(
    "--task",
    type=str,
    default=(
        "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Dynamic-Obstacle-Cbf-"
        "Obstacle-Avoidance-Unitree-Go2-Play-v0"
    ),
    help="Existing mixed obstacle-avoidance task ID.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point (unused by the baseline; required to resolve the task's Hydra config).")
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
parser.add_argument("--clip_actions", type=float, default=100.0, help="Symmetric clip applied to the raw (vx, vy, wz) command.")

# Greedy controller hyperparameters.
parser.add_argument("--greedy_alpha", type=float, default=1.0, help="Weight on normalized free space D(theta).")
parser.add_argument("--greedy_beta", type=float, default=1.5, help="Weight on normalized goal-heading error |theta - theta_g|.")
parser.add_argument("--greedy_max_speed_mps", type=float, default=1.0, help="Forward speed with full clearance and no turn.")
parser.add_argument("--greedy_min_speed_mps", type=float, default=0.0, help="Forward speed floor before obstacle/turn scaling.")
parser.add_argument("--greedy_slow_distance_m", type=float, default=1.5, help="Clearance below which forward speed is scaled down.")
parser.add_argument("--greedy_full_stop_angle_deg", type=float, default=70.0, help="Steering angle beyond which forward speed is scaled to zero.")
parser.add_argument("--greedy_yaw_gain", type=float, default=1.5, help="Proportional gain turning the robot toward theta*.")
parser.add_argument("--greedy_max_yaw_rate", type=float, default=1.5, help="Maximum commanded yaw rate (rad/s).")
parser.add_argument("--greedy_clearance_window_deg", type=float, default=12.0, help="Angular sector width used to compute D(theta) as a worst-case (min) clearance.")
parser.add_argument("--greedy_distance_cap_m", type=float, default=3.0, help="Clearance beyond which D(theta) saturates at 1.0 in the score.")
parser.add_argument("--greedy_num_candidates", type=int, default=0, help="Candidate steering directions; 0 uses the full raw ray resolution.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (  # noqa: E402
    EVALUATION_SCENARIO_CODES,
    configure_static_dynamic_evaluation,
    install_dynamic_crowd_evaluation_profiles,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from greedy_baseline import GreedyLidarController  # isort: skip


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

    log_root = Path("logs") / "greedy_baseline" / args_cli.task.replace(":", "_")
    output_root = Path(args_cli.output_dir) if args_cli.output_dir else log_root / "evaluations" / "greedy_baseline"
    output_dir = _create_timestamped_run_dir(output_root)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=args_cli.clip_actions)

    controller = GreedyLidarController(
        alpha=args_cli.greedy_alpha,
        beta=args_cli.greedy_beta,
        max_speed_mps=args_cli.greedy_max_speed_mps,
        min_speed_mps=args_cli.greedy_min_speed_mps,
        slow_distance_m=args_cli.greedy_slow_distance_m,
        full_stop_angle_deg=args_cli.greedy_full_stop_angle_deg,
        yaw_gain=args_cli.greedy_yaw_gain,
        max_yaw_rate_rad_s=args_cli.greedy_max_yaw_rate,
        clearance_window_deg=args_cli.greedy_clearance_window_deg,
        distance_cap_m=args_cli.greedy_distance_cap_m,
        num_candidates=args_cli.greedy_num_candidates if args_cli.greedy_num_candidates > 0 else None,
    )

    env_profile_indices = _fixed_grid_profile_indices(env.unwrapped, profiles)
    install_dynamic_crowd_evaluation_profiles(
        env.unwrapped,
        [profiles[index].pedestrian_count for index in env_profile_indices],
        [EVALUATION_SCENARIO_CODES.get(profiles[index].scenario, -1) for index in env_profile_indices],
    )
    env.reset()
    raw_env = env.unwrapped

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
        return original_reset_idx(env_ids)

    raw_env._reset_idx = _tracked_reset_idx

    print(
        f"[INFO] Evaluating the greedy LiDAR baseline on {len(profiles)} static-plus-dynamic profiles "
        f"with {args_cli.episodes_per_profile} episodes each"
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
                    actions = controller.compute_actions(raw_env)
                    _, _, dones, extras = env.step(actions)
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
            "controller": "greedy_lidar_baseline",
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
            "greedy_hyperparameters": {
                "alpha": args_cli.greedy_alpha,
                "beta": args_cli.greedy_beta,
                "max_speed_mps": args_cli.greedy_max_speed_mps,
                "min_speed_mps": args_cli.greedy_min_speed_mps,
                "slow_distance_m": args_cli.greedy_slow_distance_m,
                "full_stop_angle_deg": args_cli.greedy_full_stop_angle_deg,
                "yaw_gain": args_cli.greedy_yaw_gain,
                "max_yaw_rate_rad_s": args_cli.greedy_max_yaw_rate,
                "clearance_window_deg": args_cli.greedy_clearance_window_deg,
                "distance_cap_m": args_cli.greedy_distance_cap_m,
                "num_candidates": args_cli.greedy_num_candidates or "full_ray_resolution",
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
    print(f"[INFO] Wrote greedy-baseline evaluation artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
