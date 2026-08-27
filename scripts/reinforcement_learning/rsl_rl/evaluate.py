# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a mixed Go2 policy on the standardized dynamic-crowd benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

from evaluation import (  # isort: skip
    CollisionReplayRecorder,
    EpisodeMetricsCollector,
    EpisodeVelocityAccumulator,
    GOAL_REGION_COLLISION_RADIUS_M,
    InteractionEventCollector,
    InteractionEventReplayRecorder,
    _json_safe,
    dynamic_crowd_profiles,
    print_results,
    save_artifacts,
    save_interaction_event_artifacts,
    terminal_goal_region_collision_ids,
)


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy in the dynamic-crowd benchmark.")
parser.add_argument("--task", type=str, required=True, help="Existing mixed obstacle-avoidance task ID.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL-agent config entry point.")
parser.add_argument(
    "--num_envs", type=int, default=48,
    help=(
        "Vector environments (must be at least one per benchmark profile). "
        "Defaults to the full 48-cell grid (3 ordinary + slow-leader + 2 slow-crowd "
        "scenarios x 8 pedestrian counts)."
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
    help="Evaluation artifact root; each run creates a timestamped subdirectory (defaults under the checkpoint).",
)
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
    "--interesting_interaction_distance_m",
    type=float,
    default=1.5,
    help="A success replay is sampled only if the robot comes within this distance of an active pedestrian.",
)
parser.add_argument(
    "--interaction_event_cases_per_label",
    type=int,
    default=20,
    help="Maximum saved interaction-event clips per scenario and canonical label (0 disables event clips).",
)
parser.add_argument(
    "--interaction_event_padding_seconds",
    type=float,
    default=1.0,
    help="Required context before and after a saved interaction event clip.",
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
    EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD,
    EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD,
    EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS,
    EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD,
    EVALUATION_SCENARIO_CODES,
    configure_dynamic_crowd_evaluation,
    install_dynamic_crowd_evaluation_profiles,
)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


# The slow-agent benchmark variants (single slow leader; whole-crowd slow cells)
# require task-level scenario and reset support.  Detect that capability rather
# than importing it unconditionally so older task branches can still run the
# standard benchmark profiles.
try:
    from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (  # noqa: E402
        EVALUATION_CROWD_SLOW_SPEED_RANGE,
        EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
        EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
        EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
    )
except ImportError:
    EVALUATION_CROWD_SLOW_SPEED_RANGE = None
    EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M = None
    EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS = None
    EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M = None

# The RVO2/occupancy task evaluates on its benchmark environment (social-force
# crowd, 16 person slots, corridor terrain) registered by
# rvo2_navigation_eval_mixins; the policy observations/architecture stay the
# training ones.
RVO2_CROWD_EVAL = "RVO2-Crowd" in args_cli.task
if RVO2_CROWD_EVAL:
    from isaaclab_tasks.manager_based.navigation.config.go2.rvo2_navigation_eval_mixins import (  # noqa: E402
        EVALUATION_SCENARIO_CODES as RVO2_SCENARIO_CODES,
        configure_rvo2_dynamic_crowd_evaluation,
        install_rvo2_dynamic_crowd_evaluation_profiles,
        register_rvo2_eval_task,
    )
    register_rvo2_eval_task()


SLOW_LEADER_AVAILABLE = (
    not RVO2_CROWD_EVAL
    and "with_flow_slow_leader" in EVALUATION_SCENARIO_CODES
    and EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS is not None
    and EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M is not None
    and EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M is not None
)

SLOW_CROWD_AVAILABLE = (
    not RVO2_CROWD_EVAL
    and "crossing_slow" in EVALUATION_SCENARIO_CODES
    and "against_flow_slow" in EVALUATION_SCENARIO_CODES
    and EVALUATION_CROWD_SLOW_SPEED_RANGE is not None
)


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


def _slow_leader_conditions_by_env(raw_env, profiles, env_profile_indices) -> dict[int, dict[str, float]]:
    """Snapshot the current slow-leader reset samples before an evaluation step."""
    required = (
        "evaluation_slow_leader_speed_mps",
        "evaluation_slow_leader_start_ahead_m",
        "evaluation_slow_leader_lateral_offset_m",
    )
    if not SLOW_LEADER_AVAILABLE or not all(hasattr(raw_env, name) for name in required):
        return {}
    speeds = raw_env.evaluation_slow_leader_speed_mps.detach().cpu().tolist()
    aheads = raw_env.evaluation_slow_leader_start_ahead_m.detach().cpu().tolist()
    offsets = raw_env.evaluation_slow_leader_lateral_offset_m.detach().cpu().tolist()
    return {
        env_id: {
            "speed_mps": float(speeds[env_id]),
            "start_ahead_m": float(aheads[env_id]),
            "lateral_offset_m": float(offsets[env_id]),
        }
        for env_id, profile_index in enumerate(env_profile_indices)
        if profiles[profile_index].scenario == "with_flow_slow_leader"
    }


def _slow_leader_condition_summary(records: list[dict[str, float]]) -> dict[str, float | int | None]:
    """Produce compact artifact metadata for the sampled slow-leader conditions."""
    summary: dict[str, float | int | None] = {"episodes": len(records)}
    for key in ("speed_mps", "start_ahead_m", "lateral_offset_m"):
        values = [record[key] for record in records]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
        summary[f"min_{key}"] = min(values) if values else None
        summary[f"max_{key}"] = max(values) if values else None
    return summary


def _snapshot_cbf_solver_metrics(action_term, env_ids) -> dict[int, dict[str, float | int]]:
    """Snapshot CBF QP statistics before the action term resets completed environments."""
    metrics = getattr(action_term, "solver_metrics", None)
    if metrics is None:
        return {}
    if hasattr(env_ids, "detach"):
        environment_ids = env_ids.detach().cpu().tolist()
    else:
        environment_ids = list(env_ids)
    snapshots: dict[int, dict[str, float | int]] = {}
    for env_id in environment_ids:
        snapshot: dict[str, float | int] = {}
        for name, values in metrics.items():
            value = values[int(env_id)]
            if hasattr(value, "item"):
                value = value.item()
            snapshot[name] = int(value) if isinstance(value, int) else float(value)
        snapshots[int(env_id)] = snapshot
    return snapshots


def _cbf_solver_summary(records: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    """Aggregate completed-episode CBF QP telemetry for evaluation metadata."""
    if not records:
        return None
    solve_count = sum(int(record["solve_count"]) for record in records)
    solve_time_total_s = sum(float(record["solve_time_total_s"]) for record in records)
    update_time_total_s = sum(float(record["update_time_total_s"]) for record in records)
    polish_time_total_s = sum(float(record["polish_time_total_s"]) for record in records)
    return {
        "episodes": len(records),
        "solve_count": solve_count,
        "mean_iterations_per_solve": (
            sum(int(record["iteration_total"]) for record in records) / solve_count if solve_count else 0.0
        ),
        "max_iterations": max(int(record["iteration_max"]) for record in records),
        "mean_solve_time_ms": 1_000.0 * solve_time_total_s / solve_count if solve_count else 0.0,
        "max_solve_time_ms": 1_000.0 * max(float(record["solve_time_max_s"]) for record in records),
        "total_solve_time_s": solve_time_total_s,
        "total_update_time_s": update_time_total_s,
        "total_polish_time_s": polish_time_total_s,
        "max_primal_residual": max(float(record["primal_residual_max"]) for record in records),
        "max_dual_residual": max(float(record["dual_residual_max"]) for record in records),
        "solved_inaccurately_count": sum(int(record["inaccurate_count"]) for record in records),
        "maximum_iteration_reached_count": sum(int(record["max_iteration_count"]) for record in records),
        "timing_scope": "OSQP update/solve/polish only; excludes CBF construction and PyTorch CPU/GPU transfers",
    }


class EvaluationProgressReporter:
    """Best-effort live evaluation telemetry for the Research Agent UI.

    Evaluation is deliberately independent of W&B: a telemetry outage must
    never slow down or fail a benchmark.  The reporter writes flat summary
    keys so the control plane can read them from the normal run summary while
    the Pod is still executing.
    """

    _INTERVAL_SECONDS = 30.0
    _PREFIX = "research_agent_evaluation_"

    def __init__(self, *, profile_count: int, episodes_per_profile: int, seed_count: int):
        self.total_episodes = profile_count * episodes_per_profile
        self.profile_count = profile_count
        self.episodes_per_profile = episodes_per_profile
        self.seed_count = seed_count
        self.started_at = time.monotonic()
        self.last_report_at = 0.0
        self.run = None
        experiment_id = os.environ.get("RESEARCH_EXPERIMENT_ID")
        project = os.environ.get("WANDB_PROJECT")
        # Each newly submitted evaluation owns an isolated W&B run.  Keep the
        # durable remote-attempt identity in every update so the control plane
        # never mistakes an earlier attempt's telemetry for this one.
        self.evaluation_attempt_id = os.environ.get("RESEARCH_AGENT_EVALUATION_ATTEMPT_ID")
        self.bootstrap_id = os.environ.get("RESEARCH_AGENT_BOOTSTRAP_ID")
        # The bootstrap switches WANDB_RUN_ID before starting the evaluator,
        # but prefer the explicit value as well: it makes this reporter robust
        # when it is launched outside that shell wrapper.  Falling back to the
        # experiment ID preserves the legacy shared-run behavior for old Pods.
        self.wandb_run_id = (
            os.environ.get("RESEARCH_AGENT_EVALUATION_WANDB_RUN_ID")
            or os.environ.get("WANDB_RUN_ID")
            or experiment_id
        )
        if not experiment_id or not project:
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project,
                entity=os.environ.get("WANDB_ENTITY") or None,
                id=self.wandb_run_id,
                resume="allow",
            )
        except Exception as error:
            print(f"[WARN] Research Agent evaluation telemetry disabled: {error}", flush=True)

    def report(
        self,
        accepted_episodes: int,
        *,
        seed: int | None,
        seed_index: int | None,
        status: str,
        force: bool = False,
    ) -> None:
        """Publish a throttled progress snapshot without affecting evaluation."""
        now = time.monotonic()
        if not force and now - self.last_report_at < self._INTERVAL_SECONDS:
            return
        self.last_report_at = now
        elapsed_seconds = max(0.0, now - self.started_at)
        rate = accepted_episodes / elapsed_seconds if elapsed_seconds else 0.0
        remaining_episodes = max(0, self.total_episodes - accepted_episodes)
        remaining_seconds = remaining_episodes / rate if rate > 0.0 else None
        percent = round(100.0 * accepted_episodes / self.total_episodes, 1) if self.total_episodes else 100.0
        eta = f"{remaining_seconds:.0f}s" if remaining_seconds is not None else "n/a"
        seed_desc = f"seed {seed} ({seed_index}/{self.seed_count})" if seed is not None else "finalizing"
        # Mirror the throttled snapshot to stdout: the Pod console stream is
        # captured by the Research Agent pod-log logger, so live evaluation
        # progress is visible even if W&B telemetry is unavailable.
        print(
            f"[EVAL] {status}: {accepted_episodes}/{self.total_episodes} episodes ({percent}%), "
            f"{seed_desc}, elapsed {elapsed_seconds:.0f}s, ETA {eta}",
            flush=True,
        )
        if self.run is None:
            return
        snapshot = {
            f"{self._PREFIX}status": status,
            f"{self._PREFIX}accepted_episodes": int(accepted_episodes),
            f"{self._PREFIX}total_episodes": int(self.total_episodes),
            f"{self._PREFIX}percent": round(100.0 * accepted_episodes / self.total_episodes, 1)
            if self.total_episodes
            else 100.0,
            f"{self._PREFIX}current_seed": seed,
            f"{self._PREFIX}seed_index": seed_index,
            f"{self._PREFIX}seed_count": self.seed_count,
            f"{self._PREFIX}profile_count": self.profile_count,
            f"{self._PREFIX}episodes_per_profile": self.episodes_per_profile,
            f"{self._PREFIX}elapsed_seconds": round(elapsed_seconds, 1),
            f"{self._PREFIX}estimated_remaining_seconds": round(remaining_seconds, 1)
            if remaining_seconds is not None
            else None,
            f"{self._PREFIX}updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.evaluation_attempt_id:
            snapshot["research_agent_evaluation_attempt_id"] = self.evaluation_attempt_id
        if self.bootstrap_id:
            snapshot["research_agent_bootstrap_id"] = self.bootstrap_id
        try:
            self.run.summary.update(snapshot)
            # ``log`` causes W&B to sync the just-updated summary during a
            # long-running evaluation, rather than only when the process exits.
            self.run.log(snapshot, commit=True)
        except Exception as error:
            # Do not repeatedly add expensive failed network calls to a live
            # evaluation. A later W&B reconnect is handled by its own client.
            print(f"[WARN] Research Agent evaluation telemetry update skipped: {error}", flush=True)
            self.run = None

    def close(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as error:
            print(f"[WARN] Research Agent evaluation telemetry shutdown skipped: {error}", flush=True)
        finally:
            self.run = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run all dynamic-crowd profiles in parallel until every profile reaches its quota."""
    profiles = dynamic_crowd_profiles(
        include_slow_leader=SLOW_LEADER_AVAILABLE,
        include_slow_crowd=SLOW_CROWD_AVAILABLE,
    )
    if args_cli.num_envs < len(profiles):
        raise ValueError(f"--num_envs must be at least {len(profiles)} for the benchmark profiles.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if RVO2_CROWD_EVAL:
        configure_rvo2_dynamic_crowd_evaluation(env_cfg)
    else:
        configure_dynamic_crowd_evaluation(env_cfg)

    checkpoint, log_dir = _resolve_checkpoint(agent_cfg)
    output_root = Path(args_cli.output_dir) if args_cli.output_dir else Path(log_dir) / "evaluations" / "dynamic_crowd"
    output_dir = _create_timestamped_run_dir(output_root)
    if args_cli.replay_output_dir:
        failure_output_dir = _create_timestamped_run_dir(Path(args_cli.replay_output_dir)) / "episode_cases"
    else:
        failure_output_dir = output_dir / "episode_cases"
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
    if RVO2_CROWD_EVAL:
        install_rvo2_dynamic_crowd_evaluation_profiles(
            env.unwrapped,
            [profiles[index].pedestrian_count for index in env_profile_indices],
            [RVO2_SCENARIO_CODES[profiles[index].scenario] for index in env_profile_indices],
        )
    else:
        install_dynamic_crowd_evaluation_profiles(
            env.unwrapped,
            [profiles[index].pedestrian_count for index in env_profile_indices],
            [EVALUATION_SCENARIO_CODES[profiles[index].scenario] for index in env_profile_indices],
        )
    obs, _ = env.reset()
    collector = EpisodeMetricsCollector(profiles, env_profile_indices, args_cli.episodes_per_profile)
    slow_leader_records: list[dict[str, float | int]] = []
    seed_count = args_cli.seeds
    if seed_count < 1:
        raise ValueError("--seeds must be at least 1.")
    seeds = [args_cli.seed + index for index in range(seed_count)]
    # Each seed contributes an equal share of the per-profile episode budget; the last
    # seed absorbs the remainder (e.g. 100 episodes / 3 seeds -> 34, 33, 33).
    per_seed_quota = math.ceil(args_cli.episodes_per_profile / seed_count)
    progress_reporter = EvaluationProgressReporter(
        profile_count=len(profiles),
        episodes_per_profile=args_cli.episodes_per_profile,
        seed_count=seed_count,
    )
    velocity_accumulator = EpisodeVelocityAccumulator(args_cli.num_envs)
    goal_region_collision_ids: set[int] = set()
    cbf_solver_terminal_metrics: dict[int, dict[str, float | int]] = {}
    cbf_solver_episode_metrics: list[dict[str, float | int]] = []

    # Some task variants do not publish command-manager metrics in ``extras["log"]``.
    # Capture the terminal state immediately before ManagerBasedRLEnv resets it, while the
    # per-step samples below provide the rest of each episode's world-XY speed trace.
    raw_env = env.unwrapped
    step_dt_s = env.unwrapped.step_dt
    episode_length_s = env.unwrapped.cfg.episode_length_s
    interaction_collector = InteractionEventCollector(profiles, env_profile_indices, step_dt_s)
    replay_recorder = None
    if (
        not args_cli.disable_failure_recording
        or args_cli.success_cases_per_scenario
        or args_cli.interaction_event_cases_per_label
    ):
        replay_recorder = CollisionReplayRecorder(
            profiles,
            env_profile_indices,
            failure_output_dir,
            step_dt_s,
            args_cli.failure_history_seconds,
            goal_region_radius_m=GOAL_REGION_COLLISION_RADIUS_M,
            successes_per_scenario=args_cli.success_cases_per_scenario,
            episode_length_s=episode_length_s,
            record_collisions=not args_cli.disable_failure_recording,
            interesting_interaction_distance_m=args_cli.interesting_interaction_distance_m,
        )
    interaction_replay_recorder = (
        InteractionEventReplayRecorder(
            failure_output_dir / "interaction_events",
            replay_recorder,
            args_cli.interaction_event_cases_per_label,
            args_cli.interaction_event_padding_seconds,
        )
        if replay_recorder is not None and args_cli.interaction_event_cases_per_label
        else None
    )

    def _record_cbf_filtered_command() -> None:
        """Write the final CBF command once per navigation-rate replay frame."""
        if replay_recorder is None:
            return
        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        command = getattr(action_term, "cbf_filtered_velocity_command", None)
        if command is not None:
            replay_recorder.record_cbf_filtered_command(command)

    original_reset_idx = raw_env._reset_idx

    def _tracked_reset_idx(env_ids):
        terminal_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
        velocity_accumulator.record_terminal(terminal_speed, env_ids)
        goal_region_collision_ids.update(
            terminal_goal_region_collision_ids(raw_env, env_ids, GOAL_REGION_COLLISION_RADIUS_M)
        )
        interaction_collector.finalize_terminal(env_ids)
        try:
            action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        except KeyError:
            action_term = None
        if action_term is not None:
            cbf_solver_terminal_metrics.update(_snapshot_cbf_solver_metrics(action_term, env_ids))
        success_env_ids = torch.nonzero(
            raw_env.termination_manager.get_term("goal_reached"), as_tuple=False
        ).reshape(-1)
        if interaction_replay_recorder is not None:
            for env_id in success_env_ids.detach().cpu().tolist():
                if env_id in set(env_ids.detach().cpu().tolist()):
                    interaction_replay_recorder.stage_terminal_success(
                        raw_env, int(env_id), interaction_collector.pending_events(int(env_id))
                    )
        _record_cbf_filtered_command()
        if replay_recorder is not None:
            replay_recorder.capture_terminal_episodes(raw_env, env_ids, success_env_ids)
        return original_reset_idx(env_ids)

    raw_env._reset_idx = _tracked_reset_idx

    print(
        f"[INFO] Evaluating {checkpoint} on {len(profiles)} dynamic-crowd profiles "
        f"with {args_cli.episodes_per_profile} episodes each"
        + (f" across {seed_count} consecutive seeds ({seeds[0]}..{seeds[-1]})." if seed_count > 1 else ".")
    )
    progress_reporter.report(
        collector.total_episodes,
        seed=seeds[0],
        seed_index=1,
        status="running",
        force=True,
    )
    if args_cli.success_cases_per_scenario:
        print(
            "[INFO] Recording "
            f"{args_cli.success_cases_per_scenario} interesting complete success replay(s) per scenario "
            f"(robot-agent distance < {args_cli.interesting_interaction_distance_m:.2f} m)."
        )
    try:
        for seed_index, seed in enumerate(seeds):
            if seed_index > 0:
                # Re-seed all global RNGs so the next chunk draws a fresh episode
                # sequence. Episodes already in flight finish under their original
                # draws; each profile has at most one in-flight episode, which is
                # attributed to the stage in which it completes (see
                # EpisodeMetricsCollector.per_seed_counts).
                env.unwrapped.seed(seed)
                print(f"[INFO] Advancing to seed {seed} (stage {seed_index + 1} of {seed_count}).")
            collector.set_stage_limit(min(args_cli.episodes_per_profile, per_seed_quota * (seed_index + 1)))
            progress_reporter.report(
                collector.total_episodes,
                seed=seed,
                seed_index=seed_index + 1,
                status="running",
                force=True,
            )
            while simulation_app.is_running() and not collector.stage_complete:
                step_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
                velocity_accumulator.record_step(step_speed)
                slow_leader_conditions = _slow_leader_conditions_by_env(
                    raw_env, profiles, env_profile_indices
                )
                with torch.inference_mode():
                    actions = policy(obs)
                    if replay_recorder is not None:
                        submitted_actions = actions
                        if env.clip_actions is not None:
                            submitted_actions = torch.clamp(submitted_actions, -env.clip_actions, env.clip_actions)
                        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
                        action_scales = torch.as_tensor(action_term.cfg.action_scales, device=submitted_actions.device)
                        cbf_command = getattr(action_term, "cbf_filtered_velocity_command", None)
                        replay_recorder.record_pre_step(
                            raw_env, submitted_actions * action_scales, cbf_filtered_command=cbf_command
                        )
                    interaction_collector.record_pre_step(raw_env)
                    obs, _, dones, extras = env.step(actions)
                    _record_cbf_filtered_command()
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
                for env_id in collector.last_accepted_ids:
                    metrics = cbf_solver_terminal_metrics.pop(env_id, None)
                    if metrics is not None:
                        cbf_solver_episode_metrics.append(metrics)
                for env_id in completed_ids.detach().cpu().tolist():
                    cbf_solver_terminal_metrics.pop(int(env_id), None)
                for env_id in sorted(collector.last_accepted_ids):
                    condition = slow_leader_conditions.get(env_id)
                    if condition is None:
                        continue
                    profile = profiles[env_profile_indices[env_id]]
                    slow_leader_records.append(
                        {
                            **condition,
                            "seed": seed,
                            "pedestrian_count": profile.pedestrian_count,
                        }
                    )
                progress_reporter.report(
                    collector.total_episodes,
                    seed=seed,
                    seed_index=seed_index + 1,
                    status="running",
                )
                interaction_collector.resolve_terminal(completed_ids, collector.last_accepted_success_ids)
                if interaction_replay_recorder is not None:
                    interaction_replay_recorder.resolve_terminal(completed_ids, collector.last_accepted_success_ids)
                velocity_accumulator.reset(completed_ids)
                goal_region_collision_ids.difference_update(completed_ids.detach().cpu().tolist())
            print(f"[INFO] Seed {seed} stage complete: {collector.total_episodes} episodes accepted.")
            progress_reporter.report(
                collector.total_episodes,
                seed=seed,
                seed_index=seed_index + 1,
                status="running",
                force=True,
            )
    finally:
        env.close()

    if not collector.complete:
        progress_reporter.report(
            collector.total_episodes,
            seed=None,
            seed_index=None,
            status="stopped",
            force=True,
        )
        progress_reporter.close()
        raise RuntimeError("Evaluation stopped before all benchmark profiles completed.")
    progress_reporter.report(
        collector.total_episodes,
        seed=seeds[-1],
        seed_index=seed_count,
        status="writing_artifacts",
        force=True,
    )
    if collector.velocity_metric_source == "direct_world_xy_speed":
        print("[INFO] Mean XY speed was measured directly from the robot world-frame velocity.")
    elif collector.velocity_metric_source != "linear_velocity_xy":
        print(
            "[WARN] The task did not export linear_velocity_xy; using the legacy "
            f"{collector.velocity_metric_source} metric on the flat pedestrian corridor."
        )
    rows = collector.rows()
    aggregates = collector.aggregate_rows()
    slow_leader_summary = _slow_leader_condition_summary(slow_leader_records)
    artifact_dir = save_artifacts(
        output_dir,
        rows,
        aggregates,
        {
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "seed": agent_cfg.seed,
            "seeds": seeds,
            "seed_count": seed_count,
            "run_id": output_dir.name,
            "output_root": str(output_root),
            "episodes_per_profile": args_cli.episodes_per_profile,
            "success_cases_per_scenario": args_cli.success_cases_per_scenario,
            "interesting_interaction_distance_m": args_cli.interesting_interaction_distance_m,
            "interaction_events": {
                "success_only": True,
                "cases_per_label": args_cli.interaction_event_cases_per_label,
                "padding_seconds": args_cli.interaction_event_padding_seconds,
                "canonical_yield_speed_ratio": 0.70,
                "canonical_assert_speed_ratio": 0.85,
                "crossing_assert": "pedestrian-frame front crossing",
                "crossing_yield": "total planar robot speed reduction without a front crossing",
                "speed_measurement": "robot total planar speed",
            },
            "pedestrian_counts": sorted({profile.pedestrian_count for profile in profiles}),
            "scenarios": list(RVO2_SCENARIO_CODES) if RVO2_CROWD_EVAL else list(EVALUATION_SCENARIO_CODES),
            "crowd_speed_range_mps": EVALUATION_CROWD_SPEED_RANGE,
            "slow_leader": {
                "available": SLOW_LEADER_AVAILABLE,
                "scenario": "with_flow_slow_leader",
                "pedestrian_counts": [
                    profile.pedestrian_count for profile in profiles
                    if profile.scenario == "with_flow_slow_leader"
                ],
                "pedestrian_slot": 0,
                "speed_range_mps": EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
                "start_ahead_range_m": EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
                "lateral_offset_range_m": EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
                "sampled_conditions_file": "slow_leader_conditions.json",
                "sampled_conditions": slow_leader_summary,
            },
            "slow_crowd": {
                "available": SLOW_CROWD_AVAILABLE,
                "scenarios": ["crossing_slow", "against_flow_slow"],
                "pedestrian_counts": sorted({
                    profile.pedestrian_count for profile in profiles
                    if profile.scenario in ("crossing_slow", "against_flow_slow")
                }),
                "speed_range_mps": EVALUATION_CROWD_SLOW_SPEED_RANGE,
            },
            "crowd_lateral_heading_max_deg": math.degrees(EVALUATION_CROWD_LATERAL_HEADING_MAX),
            "goal_reach_condition": {
                "distance_threshold_m": EVALUATION_GOAL_REACHED_DISTANCE_THRESHOLD,
                "heading_error_threshold_deg": math.degrees(EVALUATION_GOAL_REACHED_ANGULAR_THRESHOLD),
                "xy_speed_threshold_mps": EVALUATION_GOAL_REACHED_VELOCITY_THRESHOLD,
                "stay_for_seconds": EVALUATION_GOAL_REACHED_STAY_FOR_SECONDS,
            },
            "metrics": {
                "success_rate": "goal_reached term; collisions take precedence when simultaneous",
                "navigation_success_rate": "successes divided by episodes outside the terminal-goal buffer",
                "collision_rate": "pedestrian collisions outside the terminal-goal buffer",
                "goal_region_collision_rate": (
                    f"pedestrian collisions within {GOAL_REGION_COLLISION_RADIUS_M:.2f} m of the goal"
                ),
                "all_collision_rate": "all pedestrian collisions before goal-region classification",
                "timeout_rate": "episodes terminated by the time_out term",
                "base_contact_rate": "episodes terminated by the base_contact term",
                "mean_xy_speed_mps": "episode-average world-frame horizontal robot speed over all episodes",
            },
            "velocity_metric_source": collector.velocity_metric_source,
            "step_dt_s": step_dt_s,
            "episode_length_s": episode_length_s,
            "cbf_qp_solver": _cbf_solver_summary(cbf_solver_episode_metrics),
            "failure_replays": {
                "enabled": replay_recorder is not None and not args_cli.disable_failure_recording,
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "history_seconds": args_cli.failure_history_seconds if replay_recorder is not None else None,
                "goal_region_radius_m": GOAL_REGION_COLLISION_RADIUS_M,
                "collision_cases": replay_recorder.collision_case_count if replay_recorder is not None else 0,
            },
            "success_replays": {
                "enabled": args_cli.success_cases_per_scenario > 0,
                "output_dir": str(failure_output_dir) if replay_recorder is not None else None,
                "cases_per_scenario": args_cli.success_cases_per_scenario,
                "interesting_interaction_distance_m": args_cli.interesting_interaction_distance_m,
                "success_cases": replay_recorder.success_case_count if replay_recorder is not None else 0,
            },
        },
    )
    with (artifact_dir / "slow_leader_conditions.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": 1,
                "scenario": "with_flow_slow_leader",
                "configured_ranges": {
                    "speed_mps": EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
                    "start_ahead_m": EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
                    "lateral_offset_m": EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
                },
                "summary": slow_leader_summary,
                "episodes": slow_leader_records,
            },
            file,
            indent=2,
            allow_nan=False,
        )
    save_interaction_event_artifacts(artifact_dir, interaction_collector.events, interaction_collector.summary_rows())
    if seed_count > 1:
        per_seed_breakdown = {
            "seeds": seeds,
            "episodes_per_profile": args_cli.episodes_per_profile,
            "per_profile": collector.per_seed_rows(seeds),
            "per_scenario": collector.per_seed_aggregate_rows(seeds),
        }
        with (artifact_dir / "per_seed_aggregates.json").open("w", encoding="utf-8") as file:
            json.dump(_json_safe(per_seed_breakdown), file, indent=2, allow_nan=False)
        print("[INFO] Wrote per-seed breakdown to per_seed_aggregates.json")
    print_results(rows, aggregates)
    print(f"[INFO] Wrote dynamic-crowd evaluation artifacts to: {artifact_dir}")
    if replay_recorder is not None:
        print(
            f"[INFO] Wrote {replay_recorder.collision_case_count} collision replay(s) and "
            f"{replay_recorder.success_case_count} complete success replay(s) to: {failure_output_dir}"
        )
    if interaction_replay_recorder is not None:
        print(
            f"[INFO] Wrote {interaction_replay_recorder.case_count} interaction-event replay(s) to: "
            f"{interaction_replay_recorder.output_dir}"
        )
    progress_reporter.report(
        collector.total_episodes,
        seed=seeds[-1],
        seed_index=seed_count,
        status="complete",
        force=True,
    )
    progress_reporter.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
