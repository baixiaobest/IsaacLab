"""Local-only step-response identification of the Go2 locomotion controller.

Unlike ``evaluate.py``, this script never creates a Research Agent evaluation,
contacts W&B, or changes cloud benchmark state.  It commands the *low-level*
locomotion environment directly and writes local, timestamped diagnostics.
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


parser = argparse.ArgumentParser(description="Identify first-order Go2 low-level velocity tracking locally.")
parser.add_argument("--task", default="Isaac-Locomotion-Vel-Unitree-Go2-Play-v0", help="Low-level locomotion task.")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="RL-agent config entry point.")
parser.add_argument("--num_envs", type=int, default=24, help="Parallel local trials per batch.")
parser.add_argument("--repetitions", type=int, default=5, help="Repeated trials for each step/direction case.")
parser.add_argument("--settle_seconds", type=float, default=3.0, help="Pre-step rest or cruising time.")
parser.add_argument("--response_seconds", type=float, default=3.0, help="Post-step response window to fit.")
parser.add_argument("--cruise_tolerance_mps", type=float, default=0.20, help="Allowed pre-step error for 1.5 m/s deceleration trials.")
parser.add_argument("--output_dir", default="logs/local_diagnostics/locomotion_system_id", help="Timestamped local output root.")
parser.add_argument("--seed", type=int, default=42, help="Local simulator seed.")
parser.add_argument("--tau_percentile", type=float, default=95.0, help="Upper percentile used for the final tau values.")
parser.add_argument(
    "--tau_histogram_bin_width_s",
    type=float,
    default=0.01,
    help="Histogram resolution in seconds for fitted tau distributions.",
)
parser.add_argument("--min_r_squared", type=float, default=0.80, help="Minimum R-squared for final-tau eligibility.")
parser.add_argument("--max_nrmse", type=float, default=0.20, help="Maximum normalized RMSE for final-tau eligibility.")
parser.add_argument("--max_residual_lag1", type=float, default=0.90, help="Maximum residual lag-one correlation magnitude.")
parser.add_argument(
    "--min_valid_fraction",
    type=float,
    default=0.75,
    help="Minimum eligible-trial fraction per mode before publishing a recommended tau.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if os.environ.get("RESEARCH_EXPERIMENT_ID") or os.environ.get("RESEARCH_AGENT_EVALUATION_ATTEMPT_ID"):
    parser.error("locomotion_system_identification.py is local-only and refuses a Research Agent cloud environment.")
if not args_cli.checkpoint:
    parser.error("--checkpoint is required for local locomotion system identification.")
if args_cli.num_envs < 1 or args_cli.repetitions < 1:
    parser.error("--num_envs and --repetitions must both be positive.")
if args_cli.settle_seconds <= 0.0 or args_cli.response_seconds <= 0.0:
    parser.error("--settle_seconds and --response_seconds must both be positive.")
if not 0.0 < args_cli.min_valid_fraction <= 1.0:
    parser.error("--min_valid_fraction must be in (0, 1].")
if args_cli.tau_histogram_bin_width_s <= 0.0:
    parser.error("--tau_histogram_bin_width_s must be positive.")

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
from isaaclab.terrains.config.rough import FLAT_TERRAINS_CFG  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from locomotion_system_identification_analysis import (  # noqa: E402
    conservative_tau,
    first_order_velocity,
    fit_first_order_response,
    fit_quality_reasons,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")
CRUISE_SPEED_MPS = 1.5
STEP_MAGNITUDES_MPS = (0.5, 1.0, 1.5)


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


def _trial_specs(repetitions: int) -> list[dict[str, Any]]:
    directions = (("forward", 0, 1.0), ("lateral_left", 1, 1.0), ("lateral_right", 1, -1.0))
    specs: list[dict[str, Any]] = []
    for mode in ("acceleration", "deceleration"):
        targets = STEP_MAGNITUDES_MPS if mode == "acceleration" else (1.0, 0.5, 0.0)
        for direction, axis, sign in directions:
            for target in targets:
                for repeat in range(repetitions):
                    specs.append(
                        {
                            "trial_id": f"{mode}_{direction}_{CRUISE_SPEED_MPS:g}_to_{target:g}_repeat_{repeat + 1}",
                            "mode": mode,
                            "direction": direction,
                            "axis": axis,
                            "sign": sign,
                            "source_speed_mps": 0.0 if mode == "acceleration" else CRUISE_SPEED_MPS,
                            "target_speed_mps": target,
                            "repeat": repeat + 1,
                            "extrapolation": bool(max(CRUISE_SPEED_MPS, target) > 1.0),
                        }
                    )
    return specs


def _command_tensor(
    specs: list[dict[str, Any]], phase: str, device: torch.device, total_envs: int
) -> torch.Tensor:
    """Build commands for active trial slots and hold unused final-batch slots at zero."""
    commands = torch.zeros((total_envs, 3), device=device)
    for env_id, spec in enumerate(specs):
        speed = spec["source_speed_mps"] if phase != "response" else spec["target_speed_mps"]
        commands[env_id, spec["axis"]] = spec["sign"] * speed
    return commands


def _override_for_identification(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Keep the learned low-level controller intact while removing experiment confounders."""
    env_cfg.scene.terrain.single_terrain_generator = FLAT_TERRAINS_CFG
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    command_cfg.heading_command = False
    command_cfg.velocity_heading = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_rotating_standing_envs = 0.0
    # This is a local identification configuration only.  Direct buffer writes
    # still provide the command, but the declared range makes 1.5 m/s explicit.
    command_cfg.ranges.lin_vel_x = (-1.5, 1.5)
    command_cfg.ranges.lin_vel_y = (-1.5, 1.5)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)


def _record_batch(
    env: RslRlVecEnvWrapper,
    policy: Any,
    policy_nn: Any | None,
    specs: list[dict[str, Any]],
    *,
    settle_steps: int,
    response_steps: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Run one independent batch and return full traces plus per-trial metadata."""
    raw_env = env.unwrapped
    command_term = raw_env.command_manager.get_term("base_velocity")
    step_dt = float(raw_env.step_dt)
    latch_steps = 2
    total_envs = raw_env.num_envs
    traces: dict[str, list[dict[str, Any]]] = {spec["trial_id"]: [] for spec in specs}
    states: list[dict[str, Any]] = [
        {"dones": False, "response_initial": None, "pre_step_speed": None, "response_samples": 0} for _ in specs
    ]
    obs, _ = env.reset()
    reset_mask = torch.ones(total_envs, dtype=torch.bool, device=raw_env.device)
    if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
        policy.reset(reset_mask)
    elif policy_nn is not None:
        policy_nn.reset(reset_mask)

    scheduled_phases = ["latch"] * latch_steps + ["pre_step"] * settle_steps + ["response"] * (response_steps + 1)
    # ``obs`` contains the command written during the prior call to ``env.step``.
    # Keep its phase separately from the command written now for the next action.
    applied_phase = "latch"
    applied_commands = _command_tensor(specs, applied_phase, raw_env.device, total_envs)
    for step, scheduled_phase in enumerate(scheduled_phases):
        commands = _command_tensor(specs, scheduled_phase, raw_env.device, total_envs)
        # UniformVelocityCommand exposes this persistent body-frame buffer to
        # the low-level policy observation.  No navigation or CBF action term
        # participates in this local actuator experiment.
        command_term.vel_command_b[:] = commands
        before_velocity = raw_env.scene["robot"].data.root_lin_vel_b[:, :2].detach().clone()
        # Do not run ``env.step`` under inference_mode.  Isaac Lab keeps
        # simulator state tensors and later updates them in-place during
        # ``env.reset``; tensors created in inference mode reject such updates
        # once execution leaves that context.  ``no_grad`` avoids autograd for
        # policy inference without changing the tensor type/lifetime.
        with torch.no_grad():
            actions = policy(obs)
            if env.clip_actions is not None:
                actions = torch.clamp(actions, -env.clip_actions, env.clip_actions)
        obs, _, dones, _ = env.step(actions)
        after_velocity = raw_env.scene["robot"].data.root_lin_vel_b[:, :2].detach().clone()
        acceleration = (after_velocity - before_velocity) / step_dt
        if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
            policy.reset(dones)
        elif policy_nn is not None:
            policy_nn.reset(dones)

        for env_id, spec in enumerate(specs):
            axis, sign = spec["axis"], spec["sign"]
            pre_projected = float((sign * before_velocity[env_id, axis]).item())
            post_projected = float((sign * after_velocity[env_id, axis]).item())
            if applied_phase == "response" and states[env_id]["response_initial"] is None:
                states[env_id]["response_initial"] = pre_projected
                states[env_id]["pre_step_speed"] = pre_projected
            if applied_phase == "response":
                states[env_id]["response_samples"] += 1
            states[env_id]["dones"] = bool(states[env_id]["dones"] or dones[env_id].item())
            traces[spec["trial_id"]].append(
                {
                    "trial_id": spec["trial_id"],
                    "mode": spec["mode"],
                    "direction": spec["direction"],
                    "repeat": spec["repeat"],
                    "phase": applied_phase,
                    "control_step": step,
                    "timestamp_s": (step + 1) * step_dt,
                    "response_time_s": states[env_id]["response_samples"] * step_dt if applied_phase == "response" else "",
                    "command_vx_mps": float(applied_commands[env_id, 0].item()),
                    "command_vy_mps": float(applied_commands[env_id, 1].item()),
                    "measured_vx_mps": float(after_velocity[env_id, 0].item()),
                    "measured_vy_mps": float(after_velocity[env_id, 1].item()),
                    "realized_ax_mps2": float(acceleration[env_id, 0].item()),
                    "realized_ay_mps2": float(acceleration[env_id, 1].item()),
                    "projected_velocity_mps": post_projected,
                    "projected_acceleration_mps2": float((sign * acceleration[env_id, axis]).item()),
                }
            )
        applied_phase = scheduled_phase
        applied_commands = commands
    for spec, state in zip(specs, states, strict=True):
        spec["terminated_or_reset"] = state["dones"]
        spec["response_initial_mps"] = state["response_initial"]
        spec["pre_step_speed_mps"] = state["pre_step_speed"]
    return traces, specs


def _analyse_trial(spec: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    response = [row for row in rows if row["phase"] == "response"]
    reasons: list[str] = []
    initial = spec.get("response_initial_mps")
    if spec["terminated_or_reset"]:
        reasons.append("terminated_or_reset")
    if initial is None or len(response) < 5:
        reasons.append("insufficient_response_samples")
        return {**spec, "valid_for_pooling": False, "exclusion_reasons": reasons}, np.empty(0), np.empty(0), np.empty(0)
    expected = spec["source_speed_mps"]
    tolerance = args_cli.cruise_tolerance_mps if spec["mode"] == "deceleration" else args_cli.cruise_tolerance_mps
    if abs(initial - expected) > tolerance:
        reasons.append("pre_step_velocity_not_settled")
    times = np.asarray([float(row["response_time_s"]) for row in response])
    measured = np.asarray([float(row["projected_velocity_mps"]) for row in response])
    acceleration = np.asarray([float(row["projected_acceleration_mps2"]) for row in response])
    fit = fit_first_order_response(times, measured, spec["target_speed_mps"], float(initial))
    reasons.extend(
        fit_quality_reasons(
            fit,
            min_r_squared=args_cli.min_r_squared,
            max_nrmse=args_cli.max_nrmse,
            max_abs_residual_lag1=args_cli.max_residual_lag1,
        )
    )
    return (
        {
            **spec,
            "fit": fit.to_dict(),
            "valid_for_pooling": not reasons,
            "exclusion_reasons": reasons,
        },
        times,
        measured,
        acceleration,
    )


def _plot_trial(result: dict[str, Any], times: np.ndarray, measured: np.ndarray, acceleration: np.ndarray, output_dir: Path) -> None:
    if not len(times) or "fit" not in result:
        return
    fit = result["fit"]
    command = float(result["target_speed_mps"])
    initial = float(result["response_initial_mps"])
    predicted = first_order_velocity(times, command, initial, float(fit["tau_s"]))
    (output_dir / "trial_plots").mkdir(exist_ok=True)
    figure, (velocity_axis, acceleration_axis) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=(2, 1))
    velocity_axis.step(times, np.full_like(times, command), where="post", label="commanded velocity", color="tab:gray")
    velocity_axis.plot(times, measured, label="measured velocity", color="tab:blue", linewidth=2)
    velocity_axis.plot(times, predicted, "--", label="fitted first-order response", color="tab:orange", linewidth=2)
    velocity_axis.set_ylabel("Projected velocity (m/s)")
    velocity_axis.grid(alpha=0.3)
    velocity_axis.legend(loc="best")
    annotation = (
        f"{fit['equation']}\n"
        f"tau = {fit['tau_s']:.3f} s, RMSE = {fit['rmse_mps']:.3f} m/s\n"
        f"NRMSE = {fit['nrmse']:.3f}, R² = {fit['r_squared']:.3f}"
    )
    velocity_axis.text(0.02, 0.04, annotation, transform=velocity_axis.transAxes, va="bottom", fontsize=9,
                       bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
    acceleration_axis.plot(times, acceleration, color="tab:green", label="realized acceleration")
    acceleration_axis.axhline(0.0, color="black", linewidth=0.8)
    acceleration_axis.set_xlabel("Time after velocity step (s)")
    acceleration_axis.set_ylabel("Acceleration (m/s²)")
    acceleration_axis.grid(alpha=0.3)
    acceleration_axis.legend(loc="best")
    figure.suptitle(f"{result['trial_id']} ({'eligible' if result['valid_for_pooling'] else 'excluded'})")
    figure.tight_layout()
    figure.savefig(output_dir / "trial_plots" / f"{result['trial_id']}.png", dpi=160)
    plt.close(figure)


def _plot_tau_distributions(results: list[dict[str, Any]], output_dir: Path) -> dict[str, float | None]:
    selected: dict[str, float | None] = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for axis, mode in zip(axes, ("acceleration", "deceleration"), strict=True):
        grouped = {
            direction: [item["fit"]["tau_s"] for item in results if item["mode"] == mode and item["direction"] == direction
                        and item["valid_for_pooling"]]
            for direction in ("forward", "lateral_left", "lateral_right")
        }
        pooled = [value for values in grouped.values() for value in values]
        if pooled:
            # Use a physical bin width instead of a small number of bins.  A
            # 10 ms default makes the response-time distribution visible with
            # the small number of repeated trials used by this experiment.
            bin_width = args_cli.tau_histogram_bin_width_s
            lower = np.floor(min(pooled) / bin_width) * bin_width
            upper = np.ceil(max(pooled) / bin_width) * bin_width
            if upper <= lower:
                lower -= bin_width / 2.0
                upper += bin_width / 2.0
            bins = np.arange(lower, upper + bin_width * 1.01, bin_width)
            axis.hist(pooled, bins=bins, color="tab:blue", alpha=0.35, label="pooled valid trials")
            for direction, values in grouped.items():
                if values:
                    axis.hist(values, bins=bins, histtype="step", linewidth=1.5, label=direction)
            # Preserve the individual observations: histograms alone can hide
            # a sparse or multimodal fit distribution.
            axis.scatter(pooled, np.full(len(pooled), -0.08), marker="|", s=100, color="tab:blue", clip_on=False)
            selected[mode] = conservative_tau(pooled, args_cli.tau_percentile)
            axis.axvline(selected[mode], color="tab:red", linestyle="--", linewidth=2,
                         label=f"P{args_cli.tau_percentile:g} = {selected[mode]:.3f} s")
        else:
            selected[mode] = None
            axis.text(0.5, 0.5, "No valid trials", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"{mode.title()} tau distribution")
        axis.set_xlabel("Fitted tau (s)")
        axis.set_ylabel("Trial count")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "tau_histograms.png", dpi=180)
    plt.close(figure)

    box_data = []
    labels = []
    for mode in ("acceleration", "deceleration"):
        values = [item["fit"]["tau_s"] for item in results if item["mode"] == mode and item["valid_for_pooling"]]
        if values:
            box_data.append(values)
            labels.append(mode)
    if box_data:
        figure, axis = plt.subplots(figsize=(6, 4.5))
        # Isaac Sim ships Matplotlib versions that predate ``tick_labels``.
        # ``labels`` is the compatible spelling and has the same behavior.
        axis.boxplot(box_data, labels=labels, showmeans=True)
        axis.set_ylabel("Fitted tau (s)")
        axis.set_title("Valid-trial tau comparison")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "tau_boxplot.png", dpi=180)
        plt.close(figure)
    return selected


def _write_raw_csv(traces: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
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
    _override_for_identification(env_cfg)
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

    settle_steps = round(args_cli.settle_seconds / raw_env.step_dt)
    response_steps = round(args_cli.response_seconds / raw_env.step_dt)
    all_specs = _trial_specs(args_cli.repetitions)
    all_traces: dict[str, list[dict[str, Any]]] = {}
    all_results: list[dict[str, Any]] = []
    print(f"[LOCAL SYSTEM ID] Running {len(all_specs)} trials in batches of {args_cli.num_envs}. Artifacts: {output_dir}", flush=True)
    try:
        for start in range(0, len(all_specs), args_cli.num_envs):
            batch = [dict(spec) for spec in all_specs[start : start + args_cli.num_envs]]
            traces, completed_specs = _record_batch(
                env, policy, policy_nn, batch, settle_steps=settle_steps, response_steps=response_steps
            )
            all_traces.update(traces)
            for spec in completed_specs:
                result, times, measured, acceleration = _analyse_trial(spec, traces[spec["trial_id"]])
                all_results.append(result)
                _plot_trial(result, times, measured, acceleration, output_dir)
            print(f"[LOCAL SYSTEM ID] Completed {min(start + len(batch), len(all_specs))}/{len(all_specs)} trials.", flush=True)
    finally:
        env.close()

    _write_raw_csv(all_traces, output_dir)
    selected = _plot_tau_distributions(all_results, output_dir)
    for result in all_results:
        result["fit_equation"] = result.get("fit", {}).get("equation")
    valid = [result for result in all_results if result["valid_for_pooling"]]
    adequacy_by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("acceleration", "deceleration"):
        mode_results = [result for result in all_results if result["mode"] == mode]
        mode_valid = [result for result in mode_results if result["valid_for_pooling"]]
        valid_fraction = len(mode_valid) / len(mode_results) if mode_results else 0.0
        adequate = selected[mode] is not None and valid_fraction >= args_cli.min_valid_fraction
        adequacy_by_mode[mode] = {
            "adequate": adequate,
            "valid_trial_count": len(mode_valid),
            "trial_count": len(mode_results),
            "valid_fraction": valid_fraction,
            "minimum_valid_fraction": args_cli.min_valid_fraction,
            "reason": None if adequate else "insufficient_first_order_fit_quality_or_trial_stability",
        }
    summary = {
        "mode": "local_locomotion_system_identification",
        "checkpoint": str(checkpoint),
        "task": args_cli.task,
        "control_dt_s": float(raw_env.step_dt),
        "command_frame": "robot body frame",
        "command_delivery": "direct UniformVelocityCommand.vel_command_b write",
        "terrain": "FLAT_TERRAINS_CFG",
        "requested_steps_mps": {"acceleration": [0.5, 1.0, 1.5], "deceleration": [[1.5, 1.0], [1.5, 0.5], [1.5, 0.0]]},
        "extrapolation_note": "1.5 m/s is exercised directly but lies beyond the original [-1, 1] command sampling range.",
        "tau_percentile": args_cli.tau_percentile,
        "adequacy_by_mode": adequacy_by_mode,
        "recommended_model": {
            "tau_accel_s": selected["acceleration"] if adequacy_by_mode["acceleration"]["adequate"] else None,
            "tau_decel_s": selected["deceleration"] if adequacy_by_mode["deceleration"]["adequate"] else None,
            "pooled_tau_percentile_s": selected,
            "adequate": adequacy_by_mode["acceleration"]["adequate"] and adequacy_by_mode["deceleration"]["adequate"],
            "switching_rule": "use tau_accel while speed magnitude is increasing; tau_decel while decreasing",
        },
        "trial_count": len(all_results),
        "valid_trial_count": len(valid),
        "excluded_trial_count": len(all_results) - len(valid),
        "trials": all_results,
    }
    (output_dir / "system_identification_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "recommended_model.json").write_text(
        json.dumps(summary["recommended_model"], indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        "[LOCAL SYSTEM ID] Complete. "
        f"valid={len(valid)}/{len(all_results)}, tau_accel={selected['acceleration']}, tau_decel={selected['deceleration']}. "
        f"Open {output_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
