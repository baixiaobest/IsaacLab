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
parser.add_argument("--min_r_squared", type=float, default=0.80, help="R-squared threshold for a first-order-fit warning.")
parser.add_argument("--max_nrmse", type=float, default=0.20, help="Normalized-RMSE threshold for a first-order-fit warning.")
parser.add_argument("--max_residual_lag1", type=float, default=0.90, help="Residual lag-one-correlation threshold for a first-order-fit warning.")
parser.add_argument(
    "--min_valid_fraction",
    type=float,
    default=0.75,
    help="Minimum non-excluded-trial fraction per mode for an adequate identification data set.",
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
    fit_quality_warnings,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")
CRUISE_SPEED_MPS = 1.5
STEP_MAGNITUDES_MPS = (0.5, 1.0, 1.5)
NUM_DIRECTION_SAMPLES = 16
CARDINAL_EXEMPLARS = (
    ("Forward", 0),
    ("Right", 12),
    ("Back", 8),
    ("Left", 4),
)


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
    # Body-frame angles: 0° is forward (+x), 90° is left (+y).  The full
    # circle exposes anisotropy in the learned locomotion velocity response.
    directions = [
        {
            "direction": f"{index * 360.0 / NUM_DIRECTION_SAMPLES:05.1f}deg",
            "direction_index": index,
            "direction_angle_deg": index * 360.0 / NUM_DIRECTION_SAMPLES,
            "direction_xy": (
                float(np.cos(2.0 * np.pi * index / NUM_DIRECTION_SAMPLES)),
                float(np.sin(2.0 * np.pi * index / NUM_DIRECTION_SAMPLES)),
            ),
        }
        for index in range(NUM_DIRECTION_SAMPLES)
    ]
    specs: list[dict[str, Any]] = []
    for mode in ("acceleration", "deceleration"):
        targets = STEP_MAGNITUDES_MPS if mode == "acceleration" else (1.0, 0.5, 0.0)
        for direction in directions:
            for target in targets:
                for repeat in range(repetitions):
                    specs.append(
                        {
                            "trial_id": (
                                f"{mode}_{direction['direction']}_{CRUISE_SPEED_MPS:g}_to_{target:g}_repeat_{repeat + 1}"
                            ),
                            "mode": mode,
                            **direction,
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
        direction_x, direction_y = spec["direction_xy"]
        commands[env_id, 0] = direction_x * speed
        commands[env_id, 1] = direction_y * speed
    return commands


def _override_for_identification(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Keep the learned low-level controller intact while removing experiment confounders."""
    # The selected locomotion task normally uses ``terrain_type='generator'``
    # with ROUGH_AND_GRIDS.  A ``single_terrain_generator`` override is ignored
    # for that type, so use Isaac Lab's explicit plane instead.
    terrain_cfg = env_cfg.scene.terrain
    terrain_cfg.terrain_type = "plane"
    terrain_cfg.terrain_generator = None
    terrain_cfg.single_terrain_generator = None
    terrain_cfg.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_resampling_time = None
    # These training perturbations either require terrain levels or inject a
    # disturbance during the response window; neither belongs in a controlled
    # flat-ground actuator-identification trial.
    env_cfg.events.joint_torque_offset_curriculum = None
    env_cfg.events.push_robot = None
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
            direction_x, direction_y = spec["direction_xy"]
            pre_projected = float((direction_x * before_velocity[env_id, 0] + direction_y * before_velocity[env_id, 1]).item())
            post_projected = float((direction_x * after_velocity[env_id, 0] + direction_y * after_velocity[env_id, 1]).item())
            projected_acceleration = float((direction_x * acceleration[env_id, 0] + direction_y * acceleration[env_id, 1]).item())
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
                    "direction_index": spec["direction_index"],
                    "direction_angle_deg": spec["direction_angle_deg"],
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
                    "projected_acceleration_mps2": projected_acceleration,
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
        return {
            **spec,
            "valid_for_pooling": False,
            "exclusion_reasons": reasons,
            "quality_warnings": [],
        }, np.empty(0), np.empty(0), np.empty(0)
    expected = spec["source_speed_mps"]
    tolerance = args_cli.cruise_tolerance_mps if spec["mode"] == "deceleration" else args_cli.cruise_tolerance_mps
    if abs(initial - expected) > tolerance:
        reasons.append("pre_step_velocity_not_settled")
    times = np.asarray([float(row["response_time_s"]) for row in response])
    measured = np.asarray([float(row["projected_velocity_mps"]) for row in response])
    acceleration = np.asarray([float(row["projected_acceleration_mps2"]) for row in response])
    fit = fit_first_order_response(times, measured, spec["target_speed_mps"], float(initial))
    quality_warnings = fit_quality_warnings(
        fit,
        min_r_squared=args_cli.min_r_squared,
        max_nrmse=args_cli.max_nrmse,
        max_abs_residual_lag1=args_cli.max_residual_lag1,
    )
    return (
        {
            **spec,
            "fit": fit.to_dict(),
            "valid_for_pooling": not reasons,
            "exclusion_reasons": reasons,
            "quality_warnings": quality_warnings,
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
    if not result["valid_for_pooling"]:
        status = "excluded: " + ", ".join(result["exclusion_reasons"])
    elif result["quality_warnings"]:
        status = "pooled with fit warnings: " + ", ".join(result["quality_warnings"])
    else:
        status = "pooled: no first-order fit warnings"
    figure.suptitle(f"{result['trial_id']} ({status})")
    figure.tight_layout()
    figure.savefig(output_dir / "trial_plots" / f"{result['trial_id']}.png", dpi=160)
    plt.close(figure)


def _plot_tau_distributions(results: list[dict[str, Any]], output_dir: Path) -> dict[str, float | None]:
    selected: dict[str, float | None] = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for axis, mode in zip(axes, ("acceleration", "deceleration"), strict=True):
        poolable = [item for item in results if item["mode"] == mode and item["valid_for_pooling"]]
        pooled = [item["fit"]["tau_s"] for item in poolable]
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
            clean = [item["fit"]["tau_s"] for item in poolable if not item["quality_warnings"]]
            warned = [item["fit"]["tau_s"] for item in poolable if item["quality_warnings"]]
            histogram_values = [values for values in (clean, warned) if values]
            histogram_labels = [label for values, label in ((clean, "no fit warnings"), (warned, "fit warning")) if values]
            histogram_colors = [color for values, color in ((clean, "tab:blue"), (warned, "tab:orange")) if values]
            axis.hist(
                histogram_values,
                bins=bins,
                stacked=True,
                color=histogram_colors,
                alpha=0.45,
                label=histogram_labels,
            )
            # Preserve the individual observations: histograms alone can hide
            # a sparse or multimodal fit distribution.
            axis.scatter(clean, np.full(len(clean), -0.08), marker="|", s=100, color="tab:blue", clip_on=False)
            axis.scatter(warned, np.full(len(warned), -0.08), marker="|", s=100, color="tab:orange", clip_on=False)
            selected[mode] = conservative_tau(pooled, args_cli.tau_percentile)
            axis.axvline(selected[mode], color="tab:red", linestyle="--", linewidth=2,
                         label=f"P{args_cli.tau_percentile:g} = {selected[mode]:.3f} s")
        else:
            selected[mode] = None
            axis.text(0.5, 0.5, "No valid trials", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"{mode.title()} tau distribution (pooled trials)")
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
        axis.set_title("Pooled-trial tau comparison")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "tau_boxplot.png", dpi=180)
        plt.close(figure)
    return selected


def _directional_tau_profiles(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate poolable fits by the 16 commanded body-frame directions."""
    profiles: dict[str, list[dict[str, Any]]] = {}
    for mode in ("acceleration", "deceleration"):
        by_index: dict[int, list[dict[str, Any]]] = {index: [] for index in range(NUM_DIRECTION_SAMPLES)}
        for item in results:
            if item["mode"] == mode and item["valid_for_pooling"]:
                by_index[item["direction_index"]].append(item)
        profile: list[dict[str, Any]] = []
        for index, items in by_index.items():
            angle_deg = index * 360.0 / NUM_DIRECTION_SAMPLES
            taus = [item["fit"]["tau_s"] for item in items]
            profile.append(
                {
                    "direction_index": index,
                    "direction_angle_deg": angle_deg,
                    "direction": f"{angle_deg:05.1f}deg",
                    "valid_trial_count": len(taus),
                    "quality_warning_trial_count": sum(bool(item["quality_warnings"]) for item in items),
                    "tau_median_s": float(np.median(taus)) if taus else None,
                    "tau_percentile_s": conservative_tau(taus, args_cli.tau_percentile) if taus else None,
                }
            )
        profiles[mode] = profile
    return profiles


def _plot_spatial_tau_profiles(profiles: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Plot median and conservative tau as a polar, robot-centred directional profile."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})
    for axis, mode in zip(axes, ("acceleration", "deceleration"), strict=True):
        profile = profiles[mode]
        angles = np.deg2rad([item["direction_angle_deg"] for item in profile])
        median = np.asarray([np.nan if item["tau_median_s"] is None else item["tau_median_s"] for item in profile])
        conservative = np.asarray(
            [np.nan if item["tau_percentile_s"] is None else item["tau_percentile_s"] for item in profile]
        )
        finite = np.isfinite(median) | np.isfinite(conservative)
        if np.any(finite):
            closed_angles = np.append(angles, angles[0])
            axis.plot(closed_angles, np.append(median, median[0]), color="tab:blue", linewidth=2, label="median tau")
            axis.scatter(angles, median, color="tab:blue", s=28, zorder=3)
            axis.plot(
                closed_angles,
                np.append(conservative, conservative[0]),
                color="tab:orange",
                linestyle="--",
                linewidth=2,
                label=f"P{args_cli.tau_percentile:g} tau",
            )
            axis.scatter(angles, conservative, color="tab:orange", marker="x", s=36, zorder=3)
            radial_max = float(np.nanmax(np.concatenate((median, conservative))))
            axis.set_ylim(0.0, radial_max * 1.15 if radial_max > 0.0 else 1.0)
        else:
            axis.text(0.5, 0.5, "No valid trials", transform=axis.transAxes, ha="center", va="center")
        # Robot-centric convention: forward at the top, left to the plot's left.
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(1)
        axis.set_thetagrids([0, 90, 180, 270], labels=["Forward", "Left", "Back", "Right"])
        axis.set_title(f"{mode.title()} directional tau", pad=20)
        axis.set_rlabel_position(45)
        axis.set_ylabel("tau (s)", labelpad=28)
        axis.grid(alpha=0.35)
        if np.any(finite):
            axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.14), fontsize=8)
    figure.suptitle("Robot-centred first-order velocity-response time constants")
    figure.tight_layout()
    figure.savefig(output_dir / "tau_spatial_profiles.png", dpi=200)
    plt.close(figure)


def _select_representative_trial(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer an eligible fit nearest the candidate set's median tau."""
    fitted = [item for item in candidates if "fit" in item]
    eligible = [item for item in fitted if item["valid_for_pooling"]]
    pool = eligible or fitted
    if not pool:
        return None
    median_tau = float(np.median([item["fit"]["tau_s"] for item in pool]))
    return min(pool, key=lambda item: abs(item["fit"]["tau_s"] - median_tau))


def _plot_exemplary_step_responses(
    results: list[dict[str, Any]], traces: dict[str, list[dict[str, Any]]], output_dir: Path
) -> list[dict[str, Any]]:
    """Render cardinal representative responses with command, data, and fitted curve."""
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey="row")
    selected: list[dict[str, Any]] = []
    for row_index, mode in enumerate(("acceleration", "deceleration")):
        target = 1.5 if mode == "acceleration" else 0.0
        for column_index, (name, direction_index) in enumerate(CARDINAL_EXEMPLARS):
            axis = axes[row_index, column_index]
            candidates = [
                item
                for item in results
                if item["mode"] == mode
                and item["direction_index"] == direction_index
                and item["target_speed_mps"] == target
            ]
            result = _select_representative_trial(candidates)
            if result is None:
                axis.text(0.5, 0.5, "No fitted trial", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(f"{name}: {mode}")
                continue
            response = [entry for entry in traces[result["trial_id"]] if entry["phase"] == "response"]
            times = np.asarray([float(entry["response_time_s"]) for entry in response])
            measured = np.asarray([float(entry["projected_velocity_mps"]) for entry in response])
            fit = result["fit"]
            initial = float(result["response_initial_mps"])
            fitted = first_order_velocity(times, target, initial, float(fit["tau_s"]))
            # The first plotted point at t=0 makes the ideal discontinuous
            # command step explicit; measured data begins one control step later.
            axis.step(
                np.insert(times, 0, 0.0),
                np.concatenate(([float(result["source_speed_mps"])], np.full(len(times), target))),
                where="post",
                color="tab:gray",
                label="command",
            )
            axis.plot(times, measured, color="tab:blue", linewidth=2, label="measured")
            axis.plot(times, fitted, "--", color="tab:orange", linewidth=2, label="first-order fit")
            warning_label = " · fit warning" if result["quality_warnings"] else ""
            axis.set_title(f"{name}: {mode}{warning_label}\n{initial:.1f} → {target:.1f} m/s")
            axis.grid(alpha=0.3)
            axis.text(
                0.03,
                0.04,
                f"tau={fit['tau_s']:.3f} s\nR²={fit['r_squared']:.3f}",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )
            selected.append(
                {
                    "mode": mode,
                    "cardinal_direction": name.lower(),
                    "trial_id": result["trial_id"],
                    "valid_for_pooling": result["valid_for_pooling"],
                    "quality_warnings": result["quality_warnings"],
                    "tau_s": fit["tau_s"],
                }
            )
    for axis in axes[-1, :]:
        axis.set_xlabel("Time after velocity step (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Velocity along commanded direction (m/s)")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.suptitle("Representative cardinal-direction velocity step responses")
    figure.tight_layout()
    figure.savefig(output_dir / "exemplary_cardinal_step_responses.png", dpi=200)
    plt.close(figure)
    return selected


def _rejection_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count hard exclusion reasons without making reasons exclusive."""
    rejected = [item for item in results if not item["valid_for_pooling"]]

    def count_reasons(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            for reason in item["exclusion_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))

    by_mode = {
        mode: {
            "rejected_trial_count": len([item for item in rejected if item["mode"] == mode]),
            "reason_counts": count_reasons([item for item in rejected if item["mode"] == mode]),
        }
        for mode in ("acceleration", "deceleration")
    }
    by_direction = {
        f"{angle:05.1f}deg": {
            "rejected_trial_count": len(
                [item for item in rejected if item["direction_index"] == direction_index]
            ),
            "reason_counts": count_reasons(
                [item for item in rejected if item["direction_index"] == direction_index]
            ),
        }
        for direction_index, angle in (
            (index, index * 360.0 / NUM_DIRECTION_SAMPLES) for index in range(NUM_DIRECTION_SAMPLES)
        )
    }
    by_step_profile: dict[str, dict[str, Any]] = {}
    for item in results:
        key = f"{item['mode']}_{item['source_speed_mps']:g}_to_{item['target_speed_mps']:g}"
        profile = by_step_profile.setdefault(
            key,
            {"trial_count": 0, "rejected_trial_count": 0, "reason_counts": {}},
        )
        profile["trial_count"] += 1
        if not item["valid_for_pooling"]:
            profile["rejected_trial_count"] += 1
            for reason in item["exclusion_reasons"]:
                profile["reason_counts"][reason] = profile["reason_counts"].get(reason, 0) + 1
    for profile in by_step_profile.values():
        profile["reason_counts"] = dict(sorted(profile["reason_counts"].items()))

    return {
        "note": "Hard-exclusion reason counts are non-exclusive: one rejected trial can increment multiple reasons.",
        "trial_count": len(results),
        "rejected_trial_count": len(rejected),
        "reason_counts": count_reasons(rejected),
        "by_mode": by_mode,
        "by_direction": by_direction,
        "by_step_profile": by_step_profile,
    }


def _quality_warning_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize soft first-order-model diagnostics for poolable trials."""
    poolable = [item for item in results if item["valid_for_pooling"]]
    warned = [item for item in poolable if item["quality_warnings"]]

    def count_warnings(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            for warning in item["quality_warnings"]:
                counts[warning] = counts.get(warning, 0) + 1
        return dict(sorted(counts.items()))

    def by_mode(mode: str) -> dict[str, Any]:
        mode_items = [item for item in poolable if item["mode"] == mode]
        mode_warned = [item for item in mode_items if item["quality_warnings"]]
        return {
            "poolable_trial_count": len(mode_items),
            "warning_trial_count": len(mode_warned),
            "warning_free_trial_count": len(mode_items) - len(mode_warned),
            "reason_counts": count_warnings(mode_warned),
        }

    return {
        "note": "Fit-quality warning counts are non-exclusive and do not exclude a complete, settled trial from tau pooling.",
        "poolable_trial_count": len(poolable),
        "warning_trial_count": len(warned),
        "warning_free_trial_count": len(poolable) - len(warned),
        "reason_counts": count_warnings(warned),
        "by_mode": {mode: by_mode(mode) for mode in ("acceleration", "deceleration")},
    }


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
    directional_profiles = _directional_tau_profiles(all_results)
    _plot_spatial_tau_profiles(directional_profiles, output_dir)
    exemplary_trials = _plot_exemplary_step_responses(all_results, all_traces, output_dir)
    rejection_summary = _rejection_summary(all_results)
    quality_warning_summary = _quality_warning_summary(all_results)
    for result in all_results:
        result["fit_equation"] = result.get("fit", {}).get("equation")
    valid = [result for result in all_results if result["valid_for_pooling"]]
    adequacy_by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("acceleration", "deceleration"):
        mode_results = [result for result in all_results if result["mode"] == mode]
        mode_valid = [result for result in mode_results if result["valid_for_pooling"]]
        mode_warned = [result for result in mode_valid if result["quality_warnings"]]
        mode_warning_free = [result for result in mode_valid if not result["quality_warnings"]]
        valid_fraction = len(mode_valid) / len(mode_results) if mode_results else 0.0
        warning_free_fraction = len(mode_warning_free) / len(mode_valid) if mode_valid else 0.0
        adequate = (
            selected[mode] is not None
            and valid_fraction >= args_cli.min_valid_fraction
            and warning_free_fraction >= args_cli.min_valid_fraction
        )
        if selected[mode] is None:
            adequacy_reason = "no_poolable_trials"
        elif valid_fraction < args_cli.min_valid_fraction:
            adequacy_reason = "insufficient_complete_settled_trials"
        elif warning_free_fraction < args_cli.min_valid_fraction:
            adequacy_reason = "systematic_first_order_fit_warnings"
        else:
            adequacy_reason = None
        adequacy_by_mode[mode] = {
            "adequate": adequate,
            "poolable_trial_count": len(mode_valid),
            "trial_count": len(mode_results),
            "poolable_fraction": valid_fraction,
            "quality_warning_trial_count": len(mode_warned),
            "warning_free_trial_count": len(mode_warning_free),
            "warning_free_fraction_of_poolable": warning_free_fraction,
            "minimum_fraction": args_cli.min_valid_fraction,
            "reason": adequacy_reason,
        }
    summary = {
        "mode": "local_locomotion_system_identification",
        "checkpoint": str(checkpoint),
        "task": args_cli.task,
        "control_dt_s": float(raw_env.step_dt),
        "command_frame": "robot body frame",
        "command_delivery": "direct UniformVelocityCommand.vel_command_b write",
        "disabled_training_perturbations": ["joint_torque_offset_curriculum", "push_robot"],
        "direction_samples": {
            "count": NUM_DIRECTION_SAMPLES,
            "convention": "0 degrees is forward (+x); 90 degrees is left (+y); angles increase counter-clockwise",
        },
        "terrain": "plane",
        "requested_steps_mps": {"acceleration": [0.5, 1.0, 1.5], "deceleration": [[1.5, 1.0], [1.5, 0.5], [1.5, 0.0]]},
        "extrapolation_note": "1.5 m/s is exercised directly but lies beyond the original [-1, 1] command sampling range.",
        "tau_percentile": args_cli.tau_percentile,
        "adequacy_by_mode": adequacy_by_mode,
        "recommended_model": {
            "tau_accel_s": selected["acceleration"],
            "tau_decel_s": selected["deceleration"],
            "pooled_tau_percentile_s": selected,
            "adequate": adequacy_by_mode["acceleration"]["adequate"] and adequacy_by_mode["deceleration"]["adequate"],
            "use_note": "Tau values use all complete, settled trials. Review fit-quality warnings before treating the first-order model as an accurate predictive model.",
            "switching_rule": "use tau_accel while speed magnitude is increasing; tau_decel while decreasing",
        },
        "trial_count": len(all_results),
        "valid_trial_count": len(valid),
        "excluded_trial_count": len(all_results) - len(valid),
        "rejection_summary": rejection_summary,
        "quality_warning_summary": quality_warning_summary,
        "directional_tau_profiles": directional_profiles,
        "exemplary_cardinal_trials": exemplary_trials,
        "trials": all_results,
    }
    (output_dir / "system_identification_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "recommended_model.json").write_text(
        json.dumps(summary["recommended_model"], indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "rejection_summary.json").write_text(
        json.dumps(rejection_summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "fit_quality_warning_summary.json").write_text(
        json.dumps(quality_warning_summary, indent=2, allow_nan=False), encoding="utf-8"
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
