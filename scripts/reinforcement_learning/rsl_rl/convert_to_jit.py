# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert an RSL-RL checkpoint to a TorchScript (JIT) module.

This script mirrors patterns from the rollout script and the OnPolicyRunner save
path: it will instantiate the environment and runner for the specified task,
load the checkpoint (state_dict saved by the runner), and export a TorchScript
inference module (actor or encoder+actor) for deployment.

Usage examples:
  python convert_to_jit.py --task <GymTaskName> --checkpoint logs/rsl_rl/.../model_000.pt
  python convert_to_jit.py --task Isaac-MyTask-v0 --checkpoint /path/to/model.pt --output /tmp/my_jit.pt

Note: This must be run in the same workspace / Python environment used for training
so repo modules and Isaac Sim are importable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Allow importing the CLI helpers residing in the scripts folder.
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))
import cli_args  # isort: skip

from isaaclab.app import AppLauncher

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Convert RSL-RL checkpoint to TorchScript module.")
parser.add_argument("--task", type=str, required=True, help="Gym task name for which the policy was trained.")
parser.add_argument("--output", type=str, default=None, help="Output JIT path. Defaults to <checkpoint>_jit.pt")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric (use USD I/O).")
parser.add_argument("--force_trace", action="store_true", default=False, help="Force torch.jit.trace instead of scripting fallback.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim before any torch/numpy imports — matches train.py pattern so
# that Isaac Sim's bundled numpy is on sys.path before torch links against it.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All heavy imports come AFTER AppLauncher (Isaac Sim configures sys.path first) ──
import torch
import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401


def _resolve_resume_path(checkpoint_arg: str) -> str:
    """Resolve checkpoint path (prefer local path, fallback to Nucleus retrieval)."""
    if checkpoint_arg is None:
        raise ValueError("--checkpoint is required")

    if os.path.isfile(checkpoint_arg):
        return os.path.abspath(checkpoint_arg)

    try:
        from isaaclab.utils.assets import retrieve_file_path
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import isaaclab.utils.assets.retrieve_file_path (missing 'omni.client'). "
            "Use a local checkpoint path, or run this script via ./isaaclab.sh -p so Isaac Sim modules are available."
        ) from exc

    return retrieve_file_path(checkpoint_arg)


def main() -> None:
    resume_path = _resolve_resume_path(args_cli.checkpoint)
    print(f"[INFO] Resolved checkpoint: {resume_path}")

    # Fast path: if input is already TorchScript, skip env construction.
    try:
        _ = torch.jit.load(resume_path, map_location="cpu")
        print("[INFO] Checkpoint appears to be a TorchScript module already.")
        out = args_cli.output or (
            resume_path if resume_path.endswith("_jit.pt") else resume_path.replace(".pt", "_jit.pt")
        )
        if os.path.abspath(resume_path) != os.path.abspath(out):
            shutil.copyfile(resume_path, out)
            print(f"[INFO] Copied existing JIT to: {out}")
        else:
            print(f"[INFO] JIT already at: {out}")
        return
    except Exception:
        pass

    # Build environment config and instantiate env (matching training) so the runner can construct the policy.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, use_fabric=not args_cli.disable_fabric)

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    if not isinstance(base_env.unwrapped, ManagerBasedRLEnv):
        raise RuntimeError(f"This conversion script supports manager-based RL envs, got {type(base_env.unwrapped)}")

    # Parse the RSL-RL runner config and build the runner.
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # Default output path.
    jit_path = args_cli.output or resume_path.replace(".pt", "_jit.pt")

    # If the policy provides a built-in JIT exporter, use it.
    policy_model = runner.alg.policy
    if hasattr(policy_model, "save_as_jit_script"):
        print("[INFO] Using policy.save_as_jit_script() to export TorchScript.")
        policy_model.save_as_jit_script(jit_path)
        print(f"[INFO] Saved JIT to: {jit_path}")
    else:
        print("[INFO] Policy has no save_as_jit_script. Falling back to scripting/tracing the inference module.")

        obs, extras = env.get_observations()
        sample_obs = obs[:1].to(args_cli.device)

        class _PolicyWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                if hasattr(self.policy, "act_inference"):
                    return self.policy.act_inference(x)
                if hasattr(self.policy, "actor"):
                    return self.policy.actor(x)
                return self.policy(x)

        wrapper = _PolicyWrapper(policy_model).to(args_cli.device)
        wrapper.eval()

        scripted = None
        if not args_cli.force_trace:
            try:
                scripted = torch.jit.script(wrapper)
                scripted.save(jit_path)
                print(f"[INFO] Successfully scripted and saved to: {jit_path}")
            except Exception as e:
                print(f"[WARN] Scripting failed ({e}). Trying to trace...")

        if scripted is None:
            with torch.inference_mode():
                traced = torch.jit.trace(wrapper, sample_obs.to(args_cli.device))
                traced.save(jit_path)
                print(f"[INFO] Successfully traced and saved to: {jit_path}")

    # Verify we can load the saved JIT and do a forward pass.
    try:
        loaded = torch.jit.load(jit_path, map_location=args_cli.device)
        obs, _ = env.get_observations()
        test_in = obs[:1].to(args_cli.device)
        with torch.inference_mode():
            out = loaded(test_in)
        print(f"[INFO] Verification pass succeeded, output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"[WARN] Verification failed: {e}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
