# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a policy with a learned velocity estimator substituted into the policy observations."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# Allow importing shared RSL-RL CLI helpers from the parent script folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a policy with a learned velocity estimator.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--estimator_checkpoint",
    type=str,
    default=None,
    help="Path to the trained estimator checkpoint produced by train_estimator.py.",
)
parser.add_argument(
    "--policy_estimator_jit",
    type=str,
    default=None,
    help="Optional TorchScript policy-estimator file exported by policy_estimator_jit_generator.py.",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=None,
    help="Maximum number of completed episodes to play. If not specified, runs indefinitely.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log_interval", type=float, default=5.0, help="Seconds between RMSE/progress logs.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.policy_estimator_jit is None:
    if args_cli.estimator_checkpoint is None or args_cli.checkpoint is None:
        parser.error("Either provide --policy_estimator_jit, or provide both --estimator_checkpoint and --checkpoint.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

from scripts.reinforcement_learning.rsl_rl.velocity_estimator.src.checkpoint_utils import get_checkpoint_string_list, load_estimator_checkpoint, resolve_policy_checkpoint
from scripts.reinforcement_learning.rsl_rl.velocity_estimator.src.observation_utils import ObservationTermSpec, build_observation_term_specs, get_estimator_target_paths, split_observation_groups


def _get_nested_tensor(mapping: dict[str, dict[str, torch.Tensor]] | dict[str, torch.Tensor], path: str) -> torch.Tensor:
    """Resolve a slash-delimited key path inside nested observation dictionaries."""
    value = mapping
    for key in path.split("/"):
        value = value[key]  # type: ignore[index]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Expected tensor at path '{path}', got {type(value).__name__}")
    return value


def _gather_estimator_inputs(
    expanded_obs: dict[str, dict[str, torch.Tensor]],
    input_paths: list[str],
) -> torch.Tensor:
    """Flatten estimator input terms into a batch tensor of shape (num_envs, input_dim)."""
    input_chunks = []
    for input_path in input_paths:
        term_tensor = _get_nested_tensor(expanded_obs, input_path)
        input_chunks.append(term_tensor.reshape(term_tensor.shape[0], -1))
    return torch.cat(input_chunks, dim=-1)


def _build_target_layout(
    expanded_obs: dict[str, dict[str, torch.Tensor]],
    target_paths: list[str],
) -> list[tuple[str, int, int]]:
    """Infer output slices for the estimator based on the ground-truth term shapes."""
    layout: list[tuple[str, int, int]] = []
    offset = 0
    for target_path in target_paths:
        term_name = Path(target_path).name
        term_tensor = expanded_obs["ground_truth"][term_name]
        width = int(math.prod(term_tensor.shape[1:]))
        layout.append((term_name, offset, offset + width))
        offset += width
    return layout


def _infer_jit_schema_paths(observation_specs: dict[str, list[ObservationTermSpec]]) -> tuple[list[str], list[str]]:
    """Infer JIT input and target paths directly from the environment observation layout."""
    policy_specs = observation_specs.get("policy")
    if policy_specs is None:
        raise RuntimeError("The environment must expose both 'policy' and 'ground_truth' observation groups.")

    target_paths = get_estimator_target_paths(observation_specs)
    target_term_names = {Path(path).name for path in target_paths}
    input_paths = sorted(f"policy/{spec.name}" for spec in policy_specs if spec.name not in target_term_names)
    return input_paths, target_paths


def _gather_ground_truth_targets(
    expanded_obs: dict[str, dict[str, torch.Tensor]],
    target_layout: list[tuple[str, int, int]],
) -> torch.Tensor:
    """Flatten ground-truth target terms into a batch tensor matching the estimator output order."""
    target_chunks = []
    for term_name, _, _ in target_layout:
        target_tensor = expanded_obs["ground_truth"][term_name]
        target_chunks.append(target_tensor.reshape(target_tensor.shape[0], -1))
    return torch.cat(target_chunks, dim=-1)


def _inject_estimated_velocities(
    policy_obs: torch.Tensor,
    estimator_output: torch.Tensor,
    policy_specs: list[ObservationTermSpec],
    target_layout: list[tuple[str, int, int]],
) -> torch.Tensor:
    """Replace policy velocity inputs with the estimator predictions."""
    updated_policy_obs = policy_obs.clone()
    policy_spec_map = {spec.name: spec for spec in policy_specs}

    for term_name, start, stop in target_layout:
        if term_name not in policy_spec_map:
            continue
        policy_spec = policy_spec_map[term_name]
        updated_policy_obs[:, policy_spec.start : policy_spec.stop] = estimator_output[:, start:stop]

    return updated_policy_obs


def _advance_history(
    history: torch.Tensor,
    next_features: torch.Tensor,
    done_mask: torch.Tensor,
) -> torch.Tensor:
    """Roll the history window and reset finished environments to their new initial observation."""
    updated_history = torch.roll(history, shifts=-1, dims=1)
    updated_history[:, -1, :] = next_features
    if torch.any(done_mask):
        reset_history = next_features[done_mask].unsqueeze(1).repeat(1, history.shape[1], 1)
        updated_history[done_mask] = reset_history
    return updated_history


def _update_rmse_accumulators(
    accumulators: dict[str, dict[str, float]],
    estimator_output: torch.Tensor,
    ground_truth: torch.Tensor,
    target_layout: list[tuple[str, int, int]],
) -> None:
    """Accumulate squared errors for later RMSE reporting."""
    error = estimator_output - ground_truth
    for term_name, start, stop in target_layout:
        squared_error = error[:, start:stop].pow(2)
        accumulators[term_name]["sum_sq"] += squared_error.sum().item()
        accumulators[term_name]["count"] += squared_error.numel()

    accumulators["total"]["sum_sq"] += error.pow(2).sum().item()
    accumulators["total"]["count"] += error.numel()


def _format_rmse(accumulators: dict[str, dict[str, float]]) -> str:
    """Format cumulative RMSE values for logging."""
    parts = []
    for term_name, stats in accumulators.items():
        if stats["count"] == 0:
            continue
        rmse = math.sqrt(stats["sum_sq"] / stats["count"])
        parts.append(f"{term_name}={rmse:.5f}")
    return ", ".join(parts)


def _load_policy_estimator_jit(jit_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """Load the exported policy-estimator TorchScript module."""
    scripted_module = torch.jit.load(jit_path, map_location=device)
    scripted_module.eval()
    return scripted_module


def main() -> None:
    """Play the policy while replacing velocity observations with estimator outputs."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    estimator_checkpoint_path = retrieve_file_path(args_cli.estimator_checkpoint) if args_cli.estimator_checkpoint else None
    policy_estimator_jit_path = retrieve_file_path(args_cli.policy_estimator_jit) if args_cli.policy_estimator_jit else None

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    if not isinstance(base_env.unwrapped, ManagerBasedRLEnv):
        raise RuntimeError(
            f"This estimator play script currently supports manager-based RL environments, got {type(base_env.unwrapped)}."
        )

    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    estimator_device = torch.device(str(env.unwrapped.device))
    policy_estimator_jit = None
    policy = None
    if policy_estimator_jit_path is not None:
        print(f"[INFO] Loading policy-estimator TorchScript from: {policy_estimator_jit_path}")
        policy_estimator_jit = _load_policy_estimator_jit(policy_estimator_jit_path, estimator_device)
    else:
        policy_checkpoint = resolve_policy_checkpoint(
            agent_cfg.experiment_name,
            args_cli.checkpoint,
            "play the policy with an estimator",
        )
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading policy checkpoint from: {policy_checkpoint}")
        ppo_runner.load(policy_checkpoint)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    observation_specs = build_observation_term_specs(env.unwrapped, "play estimator script")
    if "policy" not in observation_specs or "ground_truth" not in observation_specs:
        raise RuntimeError("The environment must expose both 'policy' and 'ground_truth' observation groups.")

    obs, extras = env.get_observations()
    current_obs_dict = extras["observations"]
    expanded_obs = split_observation_groups(current_obs_dict, observation_specs)

    estimator = None
    if policy_estimator_jit is not None:
        input_paths, target_paths = _infer_jit_schema_paths(observation_specs)
    else:
        if estimator_checkpoint_path is None:
            raise RuntimeError("Estimator checkpoint path is required when --policy_estimator_jit is not provided.")
        print(f"[INFO] Loading estimator checkpoint from: {estimator_checkpoint_path}")
        estimator, estimator_checkpoint = load_estimator_checkpoint(estimator_checkpoint_path, estimator_device)
        input_paths = get_checkpoint_string_list(estimator_checkpoint, "input_paths")
        target_paths = get_checkpoint_string_list(estimator_checkpoint, "target_paths")

    target_layout = _build_target_layout(expanded_obs, target_paths)

    current_features = _gather_estimator_inputs(expanded_obs, input_paths)
    if policy_estimator_jit is not None:
        history_horizon = int(policy_estimator_jit.horizon)
    else:
        if estimator is None:
            raise RuntimeError("Estimator model is not initialized.")
        history_horizon = estimator.horizon
    if policy_estimator_jit is not None and current_features.shape[1] != int(policy_estimator_jit.input_dim):
        raise RuntimeError(
            "Derived policy-estimator JIT input dimension does not match the scripted module contract. "
            f"derived={current_features.shape[1]}, jit={int(policy_estimator_jit.input_dim)}"
        )
    history = current_features.unsqueeze(1).repeat(1, history_horizon, 1)

    rmse_accumulators = {
        term_name: {"sum_sq": 0.0, "count": 0.0} for term_name, _, _ in target_layout
    }
    rmse_accumulators["total"] = {"sum_sq": 0.0, "count": 0.0}

    completed_episodes = 0
    step_count = 0
    dt = env.unwrapped.step_dt
    ratio_window_start_wall = time.time()
    ratio_window_sim_time = 0.0

    if args_cli.max_episodes is not None:
        print(f"[INFO] Playing with estimator for up to {args_cli.max_episodes} completed episodes.")
    else:
        print("[INFO] Playing with estimator indefinitely. Press Ctrl+C to stop.")

    if policy_estimator_jit is not None:
        print("[INFO] Using direct policy-estimator TorchScript inference.")
    else:
        print("[INFO] Using separate estimator and policy checkpoints.")

    try:
        while simulation_app.is_running():
            loop_start_time = time.time()

            ground_truth = _gather_ground_truth_targets(expanded_obs, target_layout)
            with torch.inference_mode():
                if policy_estimator_jit is not None:
                    estimator_output = policy_estimator_jit.estimate_velocity(history)
                    actions = policy_estimator_jit(history)
                else:
                    if estimator is None:
                        raise RuntimeError("Estimator model is not initialized.")
                    estimator_output = estimator(history)
                    policy_obs = _inject_estimated_velocities(
                        current_obs_dict["policy"],
                        estimator_output,
                        observation_specs["policy"],
                        target_layout,
                    )
                    if policy is None:
                        raise RuntimeError("Policy callable is not initialized.")
                    actions = policy(policy_obs)

            _update_rmse_accumulators(rmse_accumulators, estimator_output, ground_truth, target_layout)

            with torch.inference_mode():
                _, _, dones, extras = env.step(actions)

            next_obs_dict = extras["observations"]
            next_expanded_obs = split_observation_groups(next_obs_dict, observation_specs)
            next_features = _gather_estimator_inputs(next_expanded_obs, input_paths)
            history = _advance_history(history, next_features, dones.to(dtype=torch.bool))

            step_count += 1
            ratio_window_sim_time += dt
            completed_episodes += int(dones.sum().item())

            current_obs_dict = next_obs_dict
            expanded_obs = next_expanded_obs

            if args_cli.max_episodes is not None and completed_episodes >= args_cli.max_episodes:
                print(f"[INFO] Reached the requested {completed_episodes} completed episodes.")
                break

            elapsed_wall = time.time() - ratio_window_start_wall
            if elapsed_wall >= args_cli.log_interval:
                sim_to_realtime_ratio = ratio_window_sim_time / elapsed_wall
                print(
                    f"[INFO] steps={step_count} completed_episodes={completed_episodes} "
                    f"sim_to_real={sim_to_realtime_ratio:.3f}x rmse[{_format_rmse(rmse_accumulators)}]"
                )
                ratio_window_start_wall = time.time()
                ratio_window_sim_time = 0.0

            sleep_time = dt - (time.time() - loop_start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        print(f"[INFO] Final RMSE: {_format_rmse(rmse_accumulators)}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()