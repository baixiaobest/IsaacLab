# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a TorchScript policy wrapper that estimates base velocity before acting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent

# Allow importing shared RSL-RL CLI helpers from the parent script folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(SCRIPT_DIR))
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Export a policy+velocity-estimator inference wrapper as TorchScript.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments used to instantiate the policy runner.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--estimator_checkpoint",
    type=str,
    required=True,
    help="Path to the trained estimator checkpoint produced by train_estimator.py.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output TorchScript file path. Defaults to <policy_checkpoint_dir>/exported/policy_estimator.pt.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks  # noqa: F401

from src.checkpoint_utils import (
    get_checkpoint_int,
    get_checkpoint_string_list,
    load_estimator_checkpoint,
    resolve_policy_checkpoint,
)
from src.model import VelocityEstimator
from src.observation_utils import build_observation_term_specs, get_estimator_target_term_names


class PolicyEstimatorJitModule(torch.nn.Module):
    """TorchScript wrapper that estimates velocity before evaluating the policy."""

    def __init__(
        self,
        estimator: VelocityEstimator,
        policy: torch.nn.Module,
        input_feature_starts: list[int],
        input_feature_stops: list[int],
        policy_input_starts: list[int],
        policy_input_stops: list[int],
        estimator_target_starts: list[int],
        estimator_target_stops: list[int],
        policy_target_starts: list[int],
        policy_target_stops: list[int],
        input_dim: int,
        horizon: int,
        policy_obs_dim: int,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.policy = policy
        self.input_dim = input_dim
        self.horizon = horizon
        self.policy_obs_dim = policy_obs_dim

        self.register_buffer("input_feature_starts", torch.tensor(input_feature_starts, dtype=torch.long))
        self.register_buffer("input_feature_stops", torch.tensor(input_feature_stops, dtype=torch.long))
        self.register_buffer("policy_input_starts", torch.tensor(policy_input_starts, dtype=torch.long))
        self.register_buffer("policy_input_stops", torch.tensor(policy_input_stops, dtype=torch.long))
        self.register_buffer("estimator_target_starts", torch.tensor(estimator_target_starts, dtype=torch.long))
        self.register_buffer("estimator_target_stops", torch.tensor(estimator_target_stops, dtype=torch.long))
        self.register_buffer("policy_target_starts", torch.tensor(policy_target_starts, dtype=torch.long))
        self.register_buffer("policy_target_stops", torch.tensor(policy_target_stops, dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise RuntimeError("Expected inputs with shape (batch, horizon, input_dim).")
        if inputs.shape[1] != self.horizon:
            raise RuntimeError("Unexpected horizon in policy-estimator inputs.")
        if inputs.shape[2] != self.input_dim:
            raise RuntimeError("Unexpected input dimension in policy-estimator inputs.")

        estimated_velocity = self.estimator(inputs)
        current_features = inputs[:, -1, :]
        policy_obs = inputs.new_zeros((inputs.shape[0], self.policy_obs_dim))

        for idx in range(int(self.input_feature_starts.shape[0])):
            source_start = int(self.input_feature_starts[idx].item())
            source_stop = int(self.input_feature_stops[idx].item())
            target_start = int(self.policy_input_starts[idx].item())
            target_stop = int(self.policy_input_stops[idx].item())
            policy_obs[:, target_start:target_stop] = current_features[:, source_start:source_stop]

        for idx in range(int(self.estimator_target_starts.shape[0])):
            source_start = int(self.estimator_target_starts[idx].item())
            source_stop = int(self.estimator_target_stops[idx].item())
            target_start = int(self.policy_target_starts[idx].item())
            target_stop = int(self.policy_target_stops[idx].item())
            policy_obs[:, target_start:target_stop] = estimated_velocity[:, source_start:source_stop]

        return self.policy(policy_obs)

    @torch.jit.export
    def estimate_velocity(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return only the estimator prediction for the provided input horizon."""
        return self.estimator(inputs)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent policy state if the wrapped policy uses one."""
        self.policy.reset()
def _resolve_output_path(policy_checkpoint: str) -> str:
    """Resolve the output TorchScript path."""
    if args_cli.output:
        return os.path.abspath(args_cli.output)
    export_dir = os.path.join(os.path.dirname(policy_checkpoint), "exported")
    return os.path.join(export_dir, "policy_estimator.pt")


def _split_schema_path(path: str) -> tuple[str, str]:
    """Split a checkpoint schema path into group and leaf term name."""
    parts = path.split("/")
    if len(parts) < 2:
        raise RuntimeError(f"Expected checkpoint schema path in '<group>/<term>' form, got: {path}")
    return parts[0], parts[-1]


def _build_wrapper_layout(
    observation_specs,
    input_paths: list[str],
    target_paths: list[str],
    input_dim: int,
    target_dim: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[int], int]:
    """Build tensor slice mappings from estimator inputs/outputs to policy observations."""
    policy_specs = observation_specs.get("policy")
    if policy_specs is None:
        raise RuntimeError("The environment must expose a 'policy' observation group.")

    policy_spec_map = {spec.name: spec for spec in policy_specs}
    target_term_names = [Path(path).name for path in target_paths]
    target_term_name_set = set(target_term_names)

    input_feature_starts: list[int] = []
    input_feature_stops: list[int] = []
    policy_input_starts: list[int] = []
    policy_input_stops: list[int] = []
    seen_policy_terms: set[str] = set()

    current_input_offset = 0
    for path in input_paths:
        group_name, term_name = _split_schema_path(path)
        if group_name != "policy":
            raise RuntimeError(
                "The exported JIT wrapper expects estimator inputs from the policy observation group only. "
                f"Unsupported input path: {path}"
            )
        if term_name in target_term_name_set:
            raise RuntimeError(f"Estimator input path unexpectedly includes target velocity term: {path}")
        if term_name not in policy_spec_map:
            raise RuntimeError(f"Estimator input term '{term_name}' is not present in the policy observation group.")
        if term_name in seen_policy_terms:
            raise RuntimeError(f"Duplicate estimator input term detected: {term_name}")

        policy_spec = policy_spec_map[term_name]
        width = policy_spec.stop - policy_spec.start
        input_feature_starts.append(current_input_offset)
        input_feature_stops.append(current_input_offset + width)
        policy_input_starts.append(policy_spec.start)
        policy_input_stops.append(policy_spec.stop)
        current_input_offset += width
        seen_policy_terms.add(term_name)

    if current_input_offset != input_dim:
        raise RuntimeError(
            "Estimator input schema width does not match the checkpoint input dimension: "
            f"schema={current_input_offset}, checkpoint={input_dim}"
        )

    required_policy_terms = {spec.name for spec in policy_specs if spec.name not in target_term_name_set}
    missing_policy_terms = sorted(required_policy_terms.difference(seen_policy_terms))
    if missing_policy_terms:
        raise RuntimeError(
            "Estimator input schema does not cover all non-velocity policy terms required for inference: "
            + ", ".join(missing_policy_terms)
        )

    estimator_target_starts: list[int] = []
    estimator_target_stops: list[int] = []
    policy_target_starts: list[int] = []
    policy_target_stops: list[int] = []
    current_target_offset = 0
    for path in target_paths:
        term_name = Path(path).name
        if term_name not in policy_spec_map:
            raise RuntimeError(f"Estimator target term '{term_name}' is not present in the policy observation group.")
        policy_spec = policy_spec_map[term_name]
        width = policy_spec.stop - policy_spec.start
        estimator_target_starts.append(current_target_offset)
        estimator_target_stops.append(current_target_offset + width)
        policy_target_starts.append(policy_spec.start)
        policy_target_stops.append(policy_spec.stop)
        current_target_offset += width

    if current_target_offset != target_dim:
        raise RuntimeError(
            "Estimator target schema width does not match the checkpoint target dimension: "
            f"schema={current_target_offset}, checkpoint={target_dim}"
        )

    policy_obs_dim = policy_specs[-1].stop if policy_specs else 0
    return (
        input_feature_starts,
        input_feature_stops,
        policy_input_starts,
        policy_input_stops,
        estimator_target_starts,
        estimator_target_stops,
        policy_target_starts,
        policy_target_stops,
        policy_obs_dim,
    )


def main() -> None:
    """Load the policy and estimator checkpoints and export a combined TorchScript module."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # handle deprecated configurations (e.g. `policy` -> `actor`/`critic`)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    policy_checkpoint = resolve_policy_checkpoint(agent_cfg.experiment_name, args_cli.checkpoint, "export the policy estimator JIT")
    estimator_checkpoint_path = retrieve_file_path(args_cli.estimator_checkpoint)
    output_path = _resolve_output_path(policy_checkpoint)

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    if not isinstance(base_env.unwrapped, ManagerBasedRLEnv):
        raise RuntimeError(
            f"This exporter currently supports manager-based RL environments, got {type(base_env.unwrapped)}."
        )

    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    try:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading policy checkpoint from: {policy_checkpoint}")
        ppo_runner.load(policy_checkpoint)

        export_device = torch.device("cpu")
        print(f"[INFO] Loading estimator checkpoint from: {estimator_checkpoint_path}")
        estimator, estimator_checkpoint = load_estimator_checkpoint(estimator_checkpoint_path, export_device)

        observation_specs = build_observation_term_specs(env.unwrapped, "policy estimator JIT generator")
        input_paths = get_checkpoint_string_list(estimator_checkpoint, "input_paths")
        target_paths = get_checkpoint_string_list(estimator_checkpoint, "target_paths")
        checkpoint_target_terms = {Path(path).name for path in target_paths}
        expected_target_terms = set(get_estimator_target_term_names())
        if checkpoint_target_terms != expected_target_terms:
            raise RuntimeError(
                "The estimator checkpoint target terms do not match the current policy-estimator contract. "
                f"expected={sorted(expected_target_terms)}, checkpoint={sorted(checkpoint_target_terms)}. "
                "Retrain the estimator so it predicts linear velocity only."
            )
        input_dim = get_checkpoint_int(estimator_checkpoint, "input_dim")
        target_dim = get_checkpoint_int(estimator_checkpoint, "target_dim")
        horizon = get_checkpoint_int(estimator_checkpoint, "horizon")

        (
            input_feature_starts,
            input_feature_stops,
            policy_input_starts,
            policy_input_stops,
            estimator_target_starts,
            estimator_target_stops,
            policy_target_starts,
            policy_target_stops,
            policy_obs_dim,
        ) = _build_wrapper_layout(observation_specs, input_paths, target_paths, input_dim, target_dim)

        policy_module = ppo_runner.get_inference_policy(device=export_device)
        exported_policy = policy_module.as_jit().to(export_device)
        exported_policy.eval()
        estimator = estimator.to(export_device)
        estimator.eval()

        scripted_module = PolicyEstimatorJitModule(
            estimator=estimator,
            policy=exported_policy,
            input_feature_starts=input_feature_starts,
            input_feature_stops=input_feature_stops,
            policy_input_starts=policy_input_starts,
            policy_input_stops=policy_input_stops,
            estimator_target_starts=estimator_target_starts,
            estimator_target_stops=estimator_target_stops,
            policy_target_starts=policy_target_starts,
            policy_target_stops=policy_target_stops,
            input_dim=input_dim,
            horizon=horizon,
            policy_obs_dim=policy_obs_dim,
        ).to(export_device)
        scripted_module.eval()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.jit.script(scripted_module).save(output_path)

        print(f"[INFO] Saved policy-estimator TorchScript to: {output_path}")
        print(f"[INFO] Expected input shape: (batch, {horizon}, {input_dim})")
        print(f"[INFO] Input terms: {input_paths}")
        print(f"[INFO] Estimated velocity terms: {target_paths}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()