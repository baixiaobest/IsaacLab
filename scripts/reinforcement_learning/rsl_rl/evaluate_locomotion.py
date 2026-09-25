"""Research Agent evaluation for the robust Go2 low-level locomotion policy.

This evaluator intentionally does not import navigation scenarios, CBF code, or
mining services.  It drives the locomotion command term directly with a fixed,
held-out profile matrix and compares a candidate final checkpoint with a pinned
native RSL-RL baseline checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from locomotion_evaluation import (  # isort: skip
    RESAMPLING_STRESS_UPDATE_S,
    evaluation_profiles,
    evaluate_gates,
    percentile,
    target_for_profile,
)
from research_agent_evaluation import resolve_evaluation_spec  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate robust Go2 locomotion checkpoints for Research Agent.")
parser.add_argument("--task", default="Isaac-Locomotion-Vel-Unitree-Go2-Robust-v1")
parser.add_argument("--evaluation_family", default="locomotion")
parser.add_argument("--baseline_checkpoint", default=None, help="Optional pinned native RSL-RL baseline checkpoint.")
parser.add_argument("--baseline_experiment_id", default=None, help="Research Agent provenance for an optional baseline.")
parser.add_argument("--episodes_per_profile", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=24, help="At least one environment per fixed profile cell.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--max_failure_artifacts", type=int, default=20)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if not args_cli.checkpoint:
    parser.error("--checkpoint is required and must point to the final candidate native RSL-RL checkpoint.")
if args_cli.episodes_per_profile < 1:
    parser.error("--episodes_per_profile must be positive.")
if args_cli.max_failure_artifacts < 0:
    parser.error("--max_failure_artifacts must be nonnegative.")
spec = resolve_evaluation_spec(args_cli.evaluation_family, args_cli.task)
if spec.mining_enabled:
    parser.error("The locomotion evaluation stage must not enable mining.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.manager_based.locomotion.velocity import mdp  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


class ResearchAgentProgressReporter:
    """Best-effort progress and final-gate publication for Research Agent/W&B."""

    def __init__(self, total_episodes: int, profile_count: int):
        self.total_episodes = total_episodes
        self.profile_count = profile_count
        self.started = time.monotonic()
        self.last_report = 0.0
        self.run = None
        if not os.environ.get("RESEARCH_EXPERIMENT_ID") or not os.environ.get("WANDB_PROJECT"):
            return
        try:
            import wandb

            self.run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                entity=os.environ.get("WANDB_ENTITY") or None,
                id=(
                    os.environ.get("RESEARCH_AGENT_EVALUATION_WANDB_RUN_ID")
                    or os.environ.get("WANDB_RUN_ID")
                    or os.environ.get("RESEARCH_EXPERIMENT_ID")
                ),
                resume="allow",
            )
        except Exception as error:
            print(f"[WARN] Research Agent locomotion telemetry disabled: {error}", flush=True)

    def report(self, completed: int, label: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < 30.0:
            return
        self.last_report = now
        elapsed = max(now - self.started, 1.0e-6)
        rate = completed / elapsed
        remaining = (self.total_episodes - completed) / rate if rate > 0.0 else None
        payload = {
            "research_agent_evaluation_status": f"locomotion_{label}",
            "research_agent_evaluation_accepted_episodes": completed,
            "research_agent_evaluation_total_episodes": self.total_episodes,
            "research_agent_evaluation_percent": round(100.0 * completed / self.total_episodes, 1),
            "research_agent_evaluation_profile_count": self.profile_count,
            "research_agent_evaluation_estimated_remaining_seconds": (
                round(remaining, 1) if remaining is not None else None
            ),
            "research_agent_evaluation_updated_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"[LOCOMOTION EVAL] {label}: {completed}/{self.total_episodes} episodes", flush=True)
        if self.run is not None:
            try:
                self.run.summary.update(payload)
                self.run.log(payload, commit=True)
            except Exception as error:
                print(f"[WARN] Research Agent locomotion telemetry update skipped: {error}", flush=True)
                self.run = None

    def close(self, summary: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.summary.update({"research_agent_evaluation_status": "complete", **summary["gating"]})
            self.run.finish()
        except Exception as error:
            print(f"[WARN] Research Agent locomotion telemetry shutdown skipped: {error}", flush=True)


def _run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    path.mkdir(exist_ok=False)
    return path


def _load_policy(env, agent_cfg: RslRlBaseRunnerCfg, checkpoint: str):
    checkpoint = handle_deprecated_rsl_rl_checkpoint(retrieve_file_path(checkpoint), INSTALLED_RSL_RL_VERSION)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = None
    if version.parse(INSTALLED_RSL_RL_VERSION) < version.parse("4.0.0"):
        policy_nn = (
            runner.alg.policy
            if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("2.3.0")
            else runner.alg.actor_critic
        )
    return policy, policy_nn, checkpoint


def _termination(raw_env, name: str, env_ids: torch.Tensor) -> torch.Tensor:
    try:
        value = raw_env.termination_manager.get_term(name)
    except KeyError:
        return torch.zeros(len(env_ids), dtype=torch.bool, device=env_ids.device)
    return value[env_ids].bool()


def _transition_time(trajectory: str) -> float | None:
    # Includes the one-second settling phase in ``target_for_profile``.
    return {
        "stop_rotate_restart": 3.0,
        "obstacle_stop": 4.0,
        "obstacle_avoidance_switch": 4.0,
    }.get(trajectory)


class EpisodeCollector:
    """Small bounded per-environment recorder; never retains all episode traces."""

    def __init__(self, profiles, assignments: list[int], max_failure_artifacts: int):
        self.profiles = profiles
        self.assignments = assignments
        self.max_failure_artifacts = max_failure_artifacts
        self.samples: list[list[dict[str, float]]] = [[] for _ in assignments]
        self.rows: list[dict[str, Any]] = []
        self.artifacts: dict[str, tuple[float, list[dict[str, float]], dict[str, Any]]] = {}
        self.failure_count = 0

    def record(
        self, raw_env, command_term, actions: torch.Tensor, previous_actions: torch.Tensor, step_counts: torch.Tensor
    ) -> None:
        robot = raw_env.scene["robot"]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        linear = robot.data.root_lin_vel_b[:, :2]
        yaw = robot.data.root_ang_vel_b[:, 2]
        torque_ratio = torch.abs(robot.data.applied_torque) / robot.data.joint_effort_limits.clamp_min(1.0e-6)
        error = torch.stack(
            (
                linear[:, 0] - command_term.command[:, 0],
                linear[:, 1] - command_term.command[:, 1],
                yaw - command_term.command[:, 2],
            ),
            dim=-1,
        )
        action_rate = torch.linalg.vector_norm(actions - previous_actions, dim=-1)
        for env_id, profile_index in enumerate(self.assignments):
            self.samples[env_id].append(
                {
                    "time_s": float(step_counts[env_id].item() * raw_env.step_dt),
                    "target_vx": float(command_term.target_command[env_id, 0]),
                    "target_vy": float(command_term.target_command[env_id, 1]),
                    "target_wz": float(command_term.target_command[env_id, 2]),
                    "emitted_vx": float(command_term.emitted_command[env_id, 0]),
                    "emitted_vy": float(command_term.emitted_command[env_id, 1]),
                    "emitted_wz": float(command_term.emitted_command[env_id, 2]),
                    "command_vx": float(command_term.command[env_id, 0]),
                    "command_vy": float(command_term.command[env_id, 1]),
                    "command_wz": float(command_term.command[env_id, 2]),
                    "error_vx": float(error[env_id, 0]),
                    "error_vy": float(error[env_id, 1]),
                    "error_wz": float(error[env_id, 2]),
                    "roll_rad": float(roll[env_id]),
                    "pitch_rad": float(pitch[env_id]),
                    "angular_rate_rad_s": float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[env_id])),
                    "action_rate": float(action_rate[env_id]),
                    "torque_saturation_fraction": float((torque_ratio[env_id] >= 0.98).float().mean()),
                    "world_x": float(robot.data.root_pos_w[env_id, 0]),
                    "world_y": float(robot.data.root_pos_w[env_id, 1]),
                }
            )

    def finalize(self, env_ids: torch.Tensor, episode_numbers: torch.Tensor, raw_env) -> None:
        for env_id in env_ids.detach().cpu().tolist():
            samples = self.samples[env_id]
            if not samples:
                continue
            profile = self.profiles[self.assignments[env_id]]
            errors = np.asarray([[s["error_vx"], s["error_vy"], s["error_wz"]] for s in samples])
            tracking_norm = np.linalg.norm(errors, axis=1)
            tilt = np.asarray([math.hypot(s["roll_rad"], s["pitch_rad"]) for s in samples])
            transition_time = _transition_time(profile.family)
            after_transition = [s for s in samples if transition_time is not None and s["time_s"] >= transition_time]
            settling_time_s = None
            if after_transition:
                stable_steps = max(1, round(0.30 / raw_env.step_dt))
                post_error = [
                    math.sqrt(s["error_vx"] ** 2 + s["error_vy"] ** 2 + s["error_wz"] ** 2)
                    for s in after_transition
                ]
                for index in range(max(0, len(post_error) - stable_steps + 1)):
                    if max(post_error[index : index + stable_steps]) <= 0.15:
                        settling_time_s = after_transition[index]["time_s"] - transition_time
                        break
            fall = bool(_termination(raw_env, "base_contact", torch.tensor([env_id], device=raw_env.device))[0])
            velocity_limit = bool(
                _termination(raw_env, "base_vel_out_of_limit", torch.tensor([env_id], device=raw_env.device))[0]
            )
            row = {
                "profile": profile.name,
                "trajectory": profile.family,
                "condition": profile.name.split(":", 1)[1],
                "episode_index": int(episode_numbers[env_id]),
                "mirror_sign": -1 if int(episode_numbers[env_id]) % 2 else 1,
                "fall": fall or velocity_limit,
                "base_contact": fall,
                "base_velocity_limit": velocity_limit,
                "rms_vx_error": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                "rms_vy_error": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
                "rms_wz_error": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
                "p95_vx_error": float(np.percentile(np.abs(errors[:, 0]), 95)),
                "p95_vy_error": float(np.percentile(np.abs(errors[:, 1]), 95)),
                "p95_wz_error": float(np.percentile(np.abs(errors[:, 2]), 95)),
                "p95_tracking_error": float(np.percentile(tracking_norm, 95)),
                "p99_tilt_rad": float(np.percentile(tilt, 99)),
                "peak_angular_rate_rad_s": max(s["angular_rate_rad_s"] for s in samples),
                "p95_action_rate": percentile((s["action_rate"] for s in samples), 95),
                "torque_saturation_fraction": float(np.mean([s["torque_saturation_fraction"] for s in samples])),
                "transition_peak_tracking_error": max(
                    (math.sqrt(s["error_vx"] ** 2 + s["error_vy"] ** 2 + s["error_wz"] ** 2) for s in after_transition),
                    default=None,
                ),
                "settling_time_s": settling_time_s,
            }
            if profile.family == "obstacle_stop" and after_transition:
                start = after_transition[0]
                row["stopping_distance_m"] = max(
                    math.hypot(s["world_x"] - start["world_x"], s["world_y"] - start["world_y"])
                    for s in after_transition
                )
            self.rows.append(row)
            trace_key = profile.name
            should_retain = row["fall"] and self.failure_count < self.max_failure_artifacts
            previous = self.artifacts.get(trace_key)
            if should_retain or previous is None or row["p95_tracking_error"] > previous[0]:
                self.artifacts[trace_key] = (row["p95_tracking_error"], samples, row)
                if should_retain:
                    self.failure_count += 1
            self.samples[env_id] = []


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    metric_keys = (
        "rms_vx_error", "rms_vy_error", "rms_wz_error", "p95_vx_error", "p95_vy_error", "p95_wz_error",
        "p95_tracking_error", "p99_tilt_rad", "peak_angular_rate_rad_s", "p95_action_rate",
        "torque_saturation_fraction", "transition_peak_tracking_error", "settling_time_s", "stopping_distance_m",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    aggregates = []
    for group, group_rows in sorted(groups.items()):
        aggregate = dict(zip(group_keys, group, strict=True))
        aggregate["episodes"] = len(group_rows)
        aggregate["fall_rate"] = sum(bool(row["fall"]) for row in group_rows) / len(group_rows)
        aggregate["base_contact_rate"] = sum(bool(row["base_contact"]) for row in group_rows) / len(group_rows)
        aggregate["base_velocity_limit_rate"] = (
            sum(bool(row["base_velocity_limit"]) for row in group_rows) / len(group_rows)
        )
        for key in metric_keys:
            values = [float(row[key]) for row in group_rows if row.get(key) is not None]
            aggregate[f"mean_{key}"] = float(np.mean(values)) if values else None
        aggregates.append(aggregate)
    return aggregates


def _make_evaluation_env(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Create the one Isaac environment shared by all evaluated checkpoints.

    Isaac Sim keeps its simulation context alive for the lifetime of this
    process.  Closing a Gym wrapper and then constructing a second manager
    environment in that same process can leave the second construction
    spinning in Isaac's teardown/startup path.  Candidate and baseline use an
    identical fixed profile matrix, so one environment is both safer and the
    fairer comparison.
    """
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.base_velocity = mdp.ScriptedVelocityCommandCfg(
        asset_name="robot",
        # CommandTermCfg requires this even though scripted targets are owned by
        # the evaluator.  Keep CommandManager resampling inert: the scripted
        # term sets ``time_left`` to infinity on reset and receives all target
        # changes explicitly from the evaluation profiles.
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, env.unwrapped


def _evaluate_checkpoint(
    env,
    raw_env,
    agent_cfg,
    checkpoint: str,
    label: str,
    output_dir: Path,
    reporter: ResearchAgentProgressReporter,
    completed_offset: int,
) -> tuple[list[dict[str, Any]], str]:
    profiles = evaluation_profiles()
    if args_cli.num_envs < len(profiles):
        raise ValueError(f"--num_envs must be at least {len(profiles)} for the 24 locomotion evaluation profiles.")
    policy, policy_nn, resolved_checkpoint = _load_policy(env, agent_cfg, checkpoint)
    assignments = [index % len(profiles) for index in range(args_cli.num_envs)]
    term = raw_env.command_manager.get_term("base_velocity")
    collector = EpisodeCollector(profiles, assignments, args_cli.max_failure_artifacts if label == "candidate" else 0)
    counts = np.zeros(len(profiles), dtype=np.int64)
    episode_numbers = torch.zeros(args_cli.num_envs, dtype=torch.long, device=raw_env.device)
    step_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=raw_env.device)
    previous_actions = torch.zeros_like(raw_env.action_manager.action)
    original_reset_idx = raw_env._reset_idx
    tracking_started = [False]

    def tracked_reset(env_ids):
        if not tracking_started[0]:
            return original_reset_idx(env_ids)
        collector.finalize(env_ids, episode_numbers, raw_env)
        for env_id in env_ids.detach().cpu().tolist():
            counts[assignments[env_id]] += 1
        step_counts[env_ids] = 0
        episode_numbers[env_ids] += 1
        return original_reset_idx(env_ids)

    raw_env._reset_idx = tracked_reset
    obs, _ = env.reset()
    tracking_started[0] = True
    try:
        while simulation_app.is_running() and np.any(counts < args_cli.episodes_per_profile):
            targets = term.target_command.detach().cpu().numpy().copy()
            for env_id in range(args_cli.num_envs):
                profile = profiles[assignments[env_id]]
                # Stress profiles replace the prior at 2 Hz (every 500 ms).
                # All commands reach the policy directly, with no delay path.
                update_period_steps = round(RESAMPLING_STRESS_UPDATE_S / raw_env.step_dt)
                if profile.resampling_stress and int(step_counts[env_id]) % update_period_steps:
                    continue
                targets[env_id] = target_for_profile(
                    profile,
                    float(step_counts[env_id].item() * raw_env.step_dt),
                    int(episode_numbers[env_id]),
                )
            term.set_target_commands(torch.as_tensor(targets, device=raw_env.device))
            # Only the policy is inference-only.  Isaac's environment step
            # mutates simulation and reset buffers; running it under
            # ``inference_mode`` converts those buffers to inference tensors.
            # A second checkpoint then cannot reset the shared environment
            # because Isaac must update the root-state tensors in place.
            # ``no_grad`` avoids autograd work without changing the tensor
            # kind used by the environment.
            with torch.no_grad():
                actions = policy(obs)
            collector.record(raw_env, term, actions, previous_actions, step_counts)
            obs, _, dones, _ = env.step(actions)
            if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
            previous_actions[:] = actions
            step_counts[~dones] += 1
            reporter.report(
                completed_offset + int(np.minimum(counts, args_cli.episodes_per_profile).sum()), label
            )
    finally:
        # The next checkpoint shares this environment.  Never leave a prior
        # collector installed, otherwise every reset would be finalized by two
        # checkpoint-specific recorders.
        raw_env._reset_idx = original_reset_idx
    if np.any(counts < args_cli.episodes_per_profile):
        raise RuntimeError(f"{label} evaluation stopped before every locomotion profile reached its episode quota.")
    # Concurrent environments can finish one extra episode while another cell
    # reaches quota. Preserve the first deterministic quota for every cell.
    accepted = []
    accepted_counts = {profile.name: 0 for profile in profiles}
    for row in collector.rows:
        if accepted_counts[row["profile"]] >= args_cli.episodes_per_profile:
            continue
        accepted.append(row)
        accepted_counts[row["profile"]] += 1
    _write_rows(output_dir / f"{label}_episodes.csv", accepted)
    (output_dir / f"{label}_episodes.json").write_text(
        json.dumps(accepted, indent=2, allow_nan=False), encoding="utf-8"
    )
    artifact_dir = output_dir / f"{label}_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for name, (_, trace, row) in collector.artifacts.items():
        safe_name = name.replace(":", "_")
        _write_rows(artifact_dir / f"{safe_name}.csv", trace)
        (artifact_dir / f"{safe_name}.json").write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
    return accepted, resolved_checkpoint


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), INSTALLED_RSL_RL_VERSION)
    output_dir = _run_dir(
        Path(args_cli.output_dir) if args_cli.output_dir else Path("logs/rsl_rl/locomotion_evaluations")
    )
    profiles = evaluation_profiles()
    per_checkpoint_total = len(profiles) * args_cli.episodes_per_profile
    total_checkpoints = 2 if args_cli.baseline_checkpoint else 1
    reporter = ResearchAgentProgressReporter(per_checkpoint_total * total_checkpoints, len(profiles))
    env, raw_env = _make_evaluation_env(env_cfg, agent_cfg)
    try:
        reporter.report(0, "candidate", force=True)
        candidate_rows, candidate_checkpoint = _evaluate_checkpoint(
            env, raw_env, agent_cfg, args_cli.checkpoint, "candidate", output_dir, reporter, 0
        )
        baseline_rows: list[dict[str, Any]] = []
        baseline_checkpoint = None
        if args_cli.baseline_checkpoint:
            reporter.report(per_checkpoint_total, "baseline", force=True)
            baseline_rows, baseline_checkpoint = _evaluate_checkpoint(
                env, raw_env, agent_cfg, args_cli.baseline_checkpoint, "baseline", output_dir, reporter, per_checkpoint_total
            )
    finally:
        env.close()
    summary = {
        "evaluation_family": "locomotion",
        "mining_enabled": False,
        "task": args_cli.task,
        "candidate_checkpoint": candidate_checkpoint,
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_experiment_id": args_cli.baseline_experiment_id,
        "episodes_per_profile": args_cli.episodes_per_profile,
        "profile_count": len(evaluation_profiles()),
        "candidate_per_profile": _aggregate_rows(candidate_rows, ("profile", "trajectory", "condition")),
        "baseline_per_profile": _aggregate_rows(baseline_rows, ("profile", "trajectory", "condition")),
        "candidate_by_mirror_sign": _aggregate_rows(candidate_rows, ("trajectory", "condition", "mirror_sign")),
        "baseline_by_mirror_sign": _aggregate_rows(baseline_rows, ("trajectory", "condition", "mirror_sign")),
        "gating": evaluate_gates(candidate_rows, baseline_rows, args_cli.episodes_per_profile),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    reporter.report(per_checkpoint_total * total_checkpoints, "writing_artifacts", force=True)
    reporter.close(summary)
    print(json.dumps(summary["gating"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
