"""Validate inverse first-order velocity tracking locally on the Go2 policy.

This deliberately commands the low-level locomotion environment directly.  It
does not use a navigation checkpoint, CBF action term, Research Agent service,
or any cloud-facing evaluation path.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Validate first-order-model-aware Go2 velocity tracking locally.")
parser.add_argument("--task", default="Isaac-Locomotion-Vel-Unitree-Go2-Play-v0", help="Low-level locomotion task.")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="RL-agent config entry point.")
parser.add_argument("--num_envs", type=int, default=24, help="Parallel local trials per batch.")
parser.add_argument("--repetitions", type=int, default=3, help="Repeats for every controller and trajectory profile.")
parser.add_argument("--rest_seconds", type=float, default=2.0, help="Zero-command settling time before every ramp.")
parser.add_argument("--hold_seconds", type=float, default=3.0, help="Constant-reference hold after every ramp.")
parser.add_argument("--linear_target_mps", type=float, default=1.0, help="Linear-ramp target; leaves inverse-model command headroom.")
parser.add_argument("--ramp_rates_mps2", default="0.25,0.5,0.75", help="Comma-separated linear ramp rates in m/s^2.")
parser.add_argument("--smooth_target_mps", type=float, default=1.5, help="Full-range smooth-ramp target in m/s.")
parser.add_argument("--smooth_time_constant_s", type=float, default=1.0, help="Exponential reference time constant in seconds.")
parser.add_argument("--smooth_horizon_time_constants", type=float, default=5.0, help="Number of smooth-ramp time constants before hold.")
parser.add_argument("--directions", default="forward,left", help="Comma-separated body-frame directions: forward,right,back,left.")
parser.add_argument("--tau_accel_s", type=float, default=0.5859380136551247, help="Identified acceleration tracking tau.")
parser.add_argument("--tau_decel_s", type=float, default=0.3528624432644826, help="Identified deceleration tracking tau.")
parser.add_argument("--feedback_gain_s_inv", type=float, default=2.0, help="Velocity-error gain for feedback inverse control.")
parser.add_argument("--velocity_limit_mps", type=float, default=1.5, help="Symmetric command clipping limit.")
parser.add_argument("--seed", type=int, default=42, help="Local simulator seed.")
parser.add_argument("--output_dir", default="logs/local_diagnostics/locomotion_tracking_validation", help="Timestamped local output root.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if os.environ.get("RESEARCH_EXPERIMENT_ID") or os.environ.get("RESEARCH_AGENT_EVALUATION_ATTEMPT_ID"):
    parser.error("locomotion_tracking_validation.py is local-only and refuses a Research Agent cloud environment.")
if not args_cli.checkpoint:
    parser.error("--checkpoint is required for local tracking validation.")
if args_cli.num_envs < 1 or args_cli.repetitions < 1:
    parser.error("--num_envs and --repetitions must both be positive.")
if min(
    args_cli.rest_seconds,
    args_cli.hold_seconds,
    args_cli.linear_target_mps,
    args_cli.smooth_target_mps,
    args_cli.smooth_time_constant_s,
    args_cli.smooth_horizon_time_constants,
    args_cli.tau_accel_s,
    args_cli.tau_decel_s,
    args_cli.feedback_gain_s_inv,
    args_cli.velocity_limit_mps,
) <= 0.0:
    parser.error("All timing, target, tau, feedback-gain, and velocity-limit values must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from locomotion_tracking_validation_utils import (  # noqa: E402
    first_order_prediction_step,
    tracking_velocity_command,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")
CONTROLLERS = ("baseline", "feedforward", "feedback_inverse")
DIRECTIONS = {
    "forward": (1.0, 0.0),
    "right": (0.0, -1.0),
    "back": (-1.0, 0.0),
    "left": (0.0, 1.0),
}


def _create_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    for index in range(1000):
        path = root / (stamp if index == 0 else f"{stamp}_{index:02d}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"Could not allocate an output directory below {root}.")


def _parse_positive_csv(raw: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of numbers.") from error
    if not values or any(value <= 0.0 for value in values):
        raise ValueError(f"{name} must contain at least one positive value.")
    return values


def _parse_directions(raw: str) -> tuple[str, ...]:
    directions = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    unsupported = sorted(set(directions).difference(DIRECTIONS))
    if not directions or unsupported:
        raise ValueError(f"--directions must use {', '.join(DIRECTIONS)}; received {raw!r}.")
    return directions


def _profile_specs() -> list[dict[str, Any]]:
    rates = _parse_positive_csv(args_cli.ramp_rates_mps2, "--ramp_rates_mps2")
    directions = _parse_directions(args_cli.directions)
    specs: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        for direction in directions:
            for rate in rates:
                for repeat in range(args_cli.repetitions):
                    specs.append(
                        {
                            "trial_id": f"{controller}_linear_{direction}_{rate:g}_repeat_{repeat + 1}",
                            "controller": controller,
                            "trajectory": "linear",
                            "direction": direction,
                            "direction_xy": DIRECTIONS[direction],
                            "ramp_rate_mps2": rate,
                            "target_mps": args_cli.linear_target_mps,
                            "repeat": repeat + 1,
                        }
                    )
            for repeat in range(args_cli.repetitions):
                specs.append(
                    {
                        "trial_id": f"{controller}_smooth_{direction}_repeat_{repeat + 1}",
                        "controller": controller,
                        "trajectory": "smooth",
                        "direction": direction,
                        "direction_xy": DIRECTIONS[direction],
                        "ramp_rate_mps2": None,
                        "target_mps": args_cli.smooth_target_mps,
                        "repeat": repeat + 1,
                    }
                )
    return specs


def _duration_s(spec: dict[str, Any]) -> float:
    if spec["trajectory"] == "linear":
        ramp_duration = float(spec["target_mps"]) / float(spec["ramp_rate_mps2"])
    else:
        ramp_duration = args_cli.smooth_horizon_time_constants * args_cli.smooth_time_constant_s
    return args_cli.rest_seconds + ramp_duration + args_cli.hold_seconds


def _reference(spec: dict[str, Any], time_s: float) -> tuple[np.ndarray, np.ndarray, str]:
    """Return body-frame velocity reference, derivative, and phase."""
    direction = np.asarray(spec["direction_xy"], dtype=np.float64)
    if time_s < args_cli.rest_seconds:
        return np.zeros(2), np.zeros(2), "rest"
    elapsed = time_s - args_cli.rest_seconds
    target = float(spec["target_mps"])
    if spec["trajectory"] == "linear":
        rate = float(spec["ramp_rate_mps2"])
        ramp_duration = target / rate
        if elapsed < ramp_duration:
            return direction * rate * elapsed, direction * rate, "ramp"
        return direction * target, np.zeros(2), "hold"
    horizon = args_cli.smooth_horizon_time_constants * args_cli.smooth_time_constant_s
    if elapsed < horizon:
        exponent = np.exp(-elapsed / args_cli.smooth_time_constant_s)
        return (
            direction * target * (1.0 - exponent),
            direction * target * exponent / args_cli.smooth_time_constant_s,
            "ramp",
        )
    return direction * target, np.zeros(2), "hold"


def _override_for_local_validation(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Run the learned locomotion policy on a disturbance-free plane."""
    terrain_cfg = env_cfg.scene.terrain
    terrain_cfg.terrain_type = "plane"
    terrain_cfg.terrain_generator = None
    terrain_cfg.single_terrain_generator = None
    terrain_cfg.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_resampling_time = None
    env_cfg.events.joint_torque_offset_curriculum = None
    env_cfg.events.push_robot = None
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    command_cfg.heading_command = False
    command_cfg.velocity_heading = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_rotating_standing_envs = 0.0
    command_cfg.ranges.lin_vel_x = (-args_cli.velocity_limit_mps, args_cli.velocity_limit_mps)
    command_cfg.ranges.lin_vel_y = (-args_cli.velocity_limit_mps, args_cli.velocity_limit_mps)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)


def _record_batch(
    env: RslRlVecEnvWrapper,
    policy: Any,
    policy_nn: Any | None,
    specs: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw_env = env.unwrapped
    command_term = raw_env.command_manager.get_term("base_velocity")
    step_dt = float(raw_env.step_dt)
    total_envs = raw_env.num_envs
    max_steps = int(np.ceil(max(_duration_s(spec) for spec in specs) / step_dt))
    lower = np.full(2, -args_cli.velocity_limit_mps)
    upper = np.full(2, args_cli.velocity_limit_mps)
    traces: dict[str, list[dict[str, Any]]] = {spec["trial_id"]: [] for spec in specs}
    states = [{"done": False, "predicted_velocity": np.zeros(2)} for _ in specs]

    obs, _ = env.reset()
    reset_mask = torch.ones(total_envs, dtype=torch.bool, device=raw_env.device)
    if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
        policy.reset(reset_mask)
    elif policy_nn is not None:
        policy_nn.reset(reset_mask)

    for step in range(max_steps):
        time_s = step * step_dt
        commands = torch.zeros((total_envs, 3), device=raw_env.device)
        diagnostic: list[dict[str, Any] | None] = [None] * total_envs
        measured_before = raw_env.scene["robot"].data.root_lin_vel_b[:, :2].detach().cpu().numpy().copy()
        for env_id, spec in enumerate(specs):
            if time_s >= _duration_s(spec):
                continue
            reference, reference_acceleration, phase = _reference(spec, time_s)
            command, unclipped, desired_acceleration, tau_s = tracking_velocity_command(
                reference,
                reference_acceleration,
                measured_before[env_id],
                controller=spec["controller"],
                tau_accel_s=args_cli.tau_accel_s,
                tau_decel_s=args_cli.tau_decel_s,
                feedback_gain_s_inv=args_cli.feedback_gain_s_inv,
                velocity_lower=lower,
                velocity_upper=upper,
            )
            commands[env_id, :2] = torch.as_tensor(command, device=raw_env.device, dtype=torch.float32)
            diagnostic[env_id] = {
                "reference": reference,
                "reference_acceleration": reference_acceleration,
                "desired_acceleration": desired_acceleration,
                "unclipped": unclipped,
                "command": command,
                "tau_s": tau_s,
                "phase": phase,
            }
        command_term.vel_command_b[:] = commands
        with torch.no_grad():
            actions = policy(obs)
            if env.clip_actions is not None:
                actions = torch.clamp(actions, -env.clip_actions, env.clip_actions)
        obs, _, dones, _ = env.step(actions)
        measured_after = raw_env.scene["robot"].data.root_lin_vel_b[:, :2].detach().cpu().numpy().copy()
        if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
            policy.reset(dones)
        elif policy_nn is not None:
            policy_nn.reset(dones)

        for env_id, spec in enumerate(specs):
            details = diagnostic[env_id]
            if details is None:
                continue
            predicted, model_tau_s = first_order_prediction_step(
                states[env_id]["predicted_velocity"], details["command"], step_dt, args_cli.tau_accel_s, args_cli.tau_decel_s
            )
            states[env_id]["predicted_velocity"] = predicted
            states[env_id]["done"] = bool(states[env_id]["done"] or dones[env_id].item())
            direction = np.asarray(spec["direction_xy"], dtype=np.float64)
            traces[spec["trial_id"]].append(
                {
                    "trial_id": spec["trial_id"],
                    "controller": spec["controller"],
                    "trajectory": spec["trajectory"],
                    "direction": spec["direction"],
                    "repeat": spec["repeat"],
                    "phase": details["phase"],
                    "timestamp_s": (step + 1) * step_dt,
                    "reference_vx_mps": float(details["reference"][0]),
                    "reference_vy_mps": float(details["reference"][1]),
                    "reference_ax_mps2": float(details["reference_acceleration"][0]),
                    "reference_ay_mps2": float(details["reference_acceleration"][1]),
                    "measured_vx_mps": float(measured_after[env_id, 0]),
                    "measured_vy_mps": float(measured_after[env_id, 1]),
                    "model_vx_mps": float(predicted[0]),
                    "model_vy_mps": float(predicted[1]),
                    "command_vx_mps": float(details["command"][0]),
                    "command_vy_mps": float(details["command"][1]),
                    "unclipped_command_vx_mps": float(details["unclipped"][0]),
                    "unclipped_command_vy_mps": float(details["unclipped"][1]),
                    "desired_ax_mps2": float(details["desired_acceleration"][0]),
                    "desired_ay_mps2": float(details["desired_acceleration"][1]),
                    "controller_tau_s": float(details["tau_s"]),
                    "model_tau_s": float(model_tau_s),
                    "command_clipped": bool(not np.allclose(details["command"], details["unclipped"], atol=1.0e-8)),
                    "terminated_or_reset": bool(states[env_id]["done"]),
                }
            )
    for spec, state in zip(specs, states, strict=True):
        spec["terminated_or_reset"] = state["done"]
    return traces, specs


def _project(rows: list[dict[str, Any]], prefix: str, direction: tuple[float, float]) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    return np.asarray([vector[0] * float(row[f"{prefix}_vx_mps"]) + vector[1] * float(row[f"{prefix}_vy_mps"]) for row in rows])


def _trial_summary(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    direction = spec["direction_xy"]
    reference = _project(rows, "reference", direction)
    measured = _project(rows, "measured", direction)
    predicted = _project(rows, "model", direction)
    ramp_indices = np.asarray([row["phase"] == "ramp" for row in rows], dtype=bool)
    hold_indices = np.asarray([row["phase"] == "hold" for row in rows], dtype=bool)

    def metrics(mask: np.ndarray, observed: np.ndarray) -> dict[str, float | None]:
        if not np.any(mask):
            return {"mae_mps": None, "rmse_mps": None, "max_abs_error_mps": None}
        error = observed[mask] - reference[mask]
        return {
            "mae_mps": float(np.mean(np.abs(error))),
            "rmse_mps": float(np.sqrt(np.mean(np.square(error)))),
            "max_abs_error_mps": float(np.max(np.abs(error))),
        }

    return {
        **spec,
        "sample_count": len(rows),
        "ramp_tracking": metrics(ramp_indices, measured),
        "hold_tracking": metrics(hold_indices, measured),
        "model_prediction": metrics(np.ones(len(rows), dtype=bool), predicted),
        "command_clipping_fraction": float(np.mean([row["command_clipped"] for row in rows])) if rows else None,
        "terminated_or_reset": bool(spec["terminated_or_reset"]),
    }


def _plot_trial(spec: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    time = np.asarray([float(row["timestamp_s"]) for row in rows])
    direction = spec["direction_xy"]
    reference = _project(rows, "reference", direction)
    measured = _project(rows, "measured", direction)
    predicted = _project(rows, "model", direction)
    command = _project(rows, "command", direction)
    unclipped = _project(rows, "unclipped_command", direction)
    figure, (velocity_axis, command_axis) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=(2, 1))
    velocity_axis.plot(time, reference, color="black", linestyle="--", linewidth=2, label="reference velocity")
    velocity_axis.plot(time, measured, color="tab:blue", linewidth=2, label="measured velocity")
    velocity_axis.plot(time, predicted, color="tab:orange", linewidth=1.6, label="first-order model prediction")
    velocity_axis.set_ylabel("Velocity along test direction (m/s)")
    velocity_axis.grid(alpha=0.3)
    velocity_axis.legend(loc="best")
    command_axis.plot(time, unclipped, color="tab:purple", linestyle=":", label="unclipped command")
    command_axis.plot(time, command, color="tab:green", linewidth=1.8, label="issued locomotion command")
    command_axis.axhline(args_cli.velocity_limit_mps, color="tab:red", linewidth=0.8, linestyle="--")
    command_axis.axhline(-args_cli.velocity_limit_mps, color="tab:red", linewidth=0.8, linestyle="--", label="command limit")
    command_axis.set_xlabel("Time (s)")
    command_axis.set_ylabel("Velocity command (m/s)")
    command_axis.grid(alpha=0.3)
    command_axis.legend(loc="best")
    figure.suptitle(f"{spec['controller']} · {spec['trajectory']} ramp · {spec['direction']} · repeat {spec['repeat']}")
    figure.tight_layout()
    directory = output_dir / "trial_plots"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / f"{spec['trial_id']}.png", dpi=170)
    plt.close(figure)


def _plot_comparisons(specs: list[dict[str, Any]], traces: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    groups: dict[tuple[str, str, float | None], list[dict[str, Any]]] = {}
    for spec in specs:
        if spec["repeat"] != 1:
            continue
        key = (spec["trajectory"], spec["direction"], spec["ramp_rate_mps2"])
        groups.setdefault(key, []).append(spec)
    colors = {"baseline": "tab:gray", "feedforward": "tab:orange", "feedback_inverse": "tab:blue"}
    directory = output_dir / "comparison_plots"
    directory.mkdir(exist_ok=True)
    for (trajectory, direction, rate), group in groups.items():
        figure, axis = plt.subplots(figsize=(10, 4.8))
        first = group[0]
        for spec in sorted(group, key=lambda item: CONTROLLERS.index(item["controller"])):
            rows = traces[spec["trial_id"]]
            time = np.asarray([float(row["timestamp_s"]) for row in rows])
            measured = _project(rows, "measured", spec["direction_xy"])
            axis.plot(time, measured, color=colors[spec["controller"]], linewidth=2, label=spec["controller"].replace("_", " "))
        rows = traces[first["trial_id"]]
        time = np.asarray([float(row["timestamp_s"]) for row in rows])
        axis.plot(time, _project(rows, "reference", first["direction_xy"]), color="black", linestyle="--", linewidth=2, label="reference")
        title_profile = f"{rate:g} m/s²" if trajectory == "linear" else f"tau_ref={args_cli.smooth_time_constant_s:g} s"
        axis.set_title(f"Controller comparison · {trajectory} ramp · {direction} · {title_profile}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Velocity along test direction (m/s)")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        suffix = f"{rate:g}" if rate is not None else "smooth"
        figure.savefig(directory / f"{trajectory}_{direction}_{suffix}.png", dpi=180)
        plt.close(figure)


def _write_csv(traces: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    rows = [row for trace in traces.values() for row in trace]
    if not rows:
        return
    with (output_dir / "control_step_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _override_for_local_validation(env_cfg)
    output_dir = _create_run_dir(Path(args_cli.output_dir))
    checkpoint = retrieve_file_path(args_cli.checkpoint)
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
    runner.load(handle_deprecated_rsl_rl_checkpoint(checkpoint, INSTALLED_RSL_RL_VERSION))
    policy = runner.get_inference_policy(device=raw_env.device)
    policy_nn = None
    if version.parse(INSTALLED_RSL_RL_VERSION) < version.parse("4.0.0"):
        policy_nn = runner.alg.policy if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("2.3.0") else runner.alg.actor_critic

    specs = _profile_specs()
    traces: dict[str, list[dict[str, Any]]] = {}
    completed_specs: list[dict[str, Any]] = []
    print(f"[LOCAL TRACKING VALIDATION] Running {len(specs)} trials in batches of {args_cli.num_envs}. Artifacts: {output_dir}", flush=True)
    try:
        for start in range(0, len(specs), args_cli.num_envs):
            batch = [dict(spec) for spec in specs[start : start + args_cli.num_envs]]
            batch_traces, batch_specs = _record_batch(env, policy, policy_nn, batch)
            traces.update(batch_traces)
            completed_specs.extend(batch_specs)
            print(f"[LOCAL TRACKING VALIDATION] Completed {min(start + len(batch), len(specs))}/{len(specs)} trials.", flush=True)
    finally:
        env.close()

    _write_csv(traces, output_dir)
    results = []
    for spec in completed_specs:
        _plot_trial(spec, traces[spec["trial_id"]], output_dir)
        results.append(_trial_summary(spec, traces[spec["trial_id"]]))
    _plot_comparisons(completed_specs, traces, output_dir)
    by_controller: dict[str, dict[str, float | int | None]] = {}
    for controller in CONTROLLERS:
        controller_results = [item for item in results if item["controller"] == controller and not item["terminated_or_reset"]]
        ramp_rmses = [item["ramp_tracking"]["rmse_mps"] for item in controller_results if item["ramp_tracking"]["rmse_mps"] is not None]
        by_controller[controller] = {
            "completed_trial_count": len(controller_results),
            "mean_ramp_rmse_mps": float(np.mean(ramp_rmses)) if ramp_rmses else None,
            "median_ramp_rmse_mps": float(np.median(ramp_rmses)) if ramp_rmses else None,
            "mean_command_clipping_fraction": float(np.mean([item["command_clipping_fraction"] for item in controller_results])) if controller_results else None,
        }
    summary = {
        "mode": "local_locomotion_tracking_validation",
        "checkpoint": str(checkpoint),
        "task": args_cli.task,
        "terrain": "plane",
        "control_dt_s": float(raw_env.step_dt),
        "command_delivery": "direct UniformVelocityCommand.vel_command_b write",
        "controllers": {
            "baseline": "v_cmd = v_ref",
            "feedforward": "v_cmd = v_ref + tau * v_ref_dot",
            "feedback_inverse": "v_cmd = v + tau * (v_ref_dot + k_e * (v_ref - v))",
        },
        "model": {
            "tau_accel_s": args_cli.tau_accel_s,
            "tau_decel_s": args_cli.tau_decel_s,
            "feedback_gain_s_inv": args_cli.feedback_gain_s_inv,
            "tau_switch": "tau_decel only when desired acceleration opposes measured velocity; tau_accel otherwise",
        },
        "profiles": {
            "directions": _parse_directions(args_cli.directions),
            "linear_target_mps": args_cli.linear_target_mps,
            "linear_ramp_rates_mps2": _parse_positive_csv(args_cli.ramp_rates_mps2, "--ramp_rates_mps2"),
            "smooth_target_mps": args_cli.smooth_target_mps,
            "smooth_time_constant_s": args_cli.smooth_time_constant_s,
            "repetitions": args_cli.repetitions,
            "velocity_limit_mps": args_cli.velocity_limit_mps,
        },
        "by_controller": by_controller,
        "trials": results,
    }
    (output_dir / "tracking_validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[LOCAL TRACKING VALIDATION] Complete. Open {output_dir}.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
