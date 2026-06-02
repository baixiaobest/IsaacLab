# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--timesteps", type=int, default=None, help="Total trainer timesteps override.")
parser.add_argument("--memory_size", type=int, default=None, help="Replay memory size override for off-policy agents.")
parser.add_argument("--checkpoint_interval", type=int, default=None, help="Checkpoint save interval override (in timesteps).")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--logger",
    type=str,
    default=None,
    choices=["tensorboard", "wandb", "none"],
    help="Logger module to use. Set to 'none' to disable TensorBoard and wandb.",
)
parser.add_argument(
    "--log_project_name",
    type=str,
    default=None,
    help="Project name used when --logger wandb is selected.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _expand_action_bound(bound, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Expand scalar/vector bound to match action shape."""
    bound_array = np.asarray(bound, dtype=np.float32)
    if bound_array.ndim == 0:
        return np.full(shape, float(bound_array), dtype=np.float32)
    if tuple(bound_array.shape) != tuple(shape):
        raise ValueError(
            f"Invalid action_space_bounds.{name} shape: {bound_array.shape}. Expected scalar or {shape}."
        )
    return bound_array.astype(np.float32, copy=False)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    experiment_cfg = agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
    trainer_cfg = agent_cfg.setdefault("trainer", {})

    # optional logger overrides from CLI
    if args_cli.logger == "tensorboard":
        experiment_cfg["wandb"] = False
        experiment_cfg["write_interval"] = "auto"
    elif args_cli.logger == "wandb":
        experiment_cfg["wandb"] = True
        experiment_cfg.setdefault("write_interval", "auto")
        if args_cli.log_project_name:
            wandb_kwargs = experiment_cfg.setdefault("wandb_kwargs", {})
            wandb_kwargs["project"] = args_cli.log_project_name
    elif args_cli.logger == "none":
        experiment_cfg["wandb"] = False
        experiment_cfg["write_interval"] = 0

    # optional replay memory size override (e.g. SAC)
    if args_cli.memory_size is not None and "memory" in agent_cfg:
        agent_cfg["memory"]["memory_size"] = args_cli.memory_size

    # optional checkpoint interval override
    if args_cli.checkpoint_interval is not None:
        experiment_cfg["checkpoint_interval"] = args_cli.checkpoint_interval

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # explicit timesteps override
    if args_cli.timesteps is not None:
        trainer_cfg["timesteps"] = args_cli.timesteps
    # max iterations for training
    elif args_cli.max_iterations:
        rollouts = agent_cfg["agent"].get("rollouts")
        if rollouts is None:
            trainer_cfg["timesteps"] = args_cli.max_iterations
        else:
            trainer_cfg["timesteps"] = args_cli.max_iterations * rollouts
    trainer_cfg["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Isaac Lab manager environments may expose unbounded Box action spaces.
    # skrl Model.random_act uses Uniform(low, high), which becomes NaN for inf bounds.
    # Bounds can be provided via agent_cfg["action_space_bounds"], otherwise fallback is [-1, 1].
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if isinstance(getattr(base_env, "single_action_space", None), gym.spaces.Box):
        single_action_space = base_env.single_action_space
        action_space_bounds_cfg = agent_cfg.get("action_space_bounds", {})
        override_force = bool(action_space_bounds_cfg.get("force", False))
        has_unbounded_bounds = not np.isfinite(single_action_space.low).all() or not np.isfinite(single_action_space.high).all()

        if override_force or has_unbounded_bounds:
            low_bound = _expand_action_bound(action_space_bounds_cfg.get("low", -1.0), single_action_space.shape, "low")
            high_bound = _expand_action_bound(
                action_space_bounds_cfg.get("high", 1.0), single_action_space.shape, "high"
            )
            if not np.all(high_bound > low_bound):
                raise ValueError("Invalid action_space_bounds: all elements in 'high' must be greater than 'low'.")

            source = "forced override" if override_force else "unbounded fallback"
            print(
                f"[WARNING] Using action-space bounds from config ({source}): "
                f"low={np.array2string(low_bound, precision=3)}, high={np.array2string(high_bound, precision=3)}"
            )
            base_env.single_action_space = gym.spaces.Box(
                low=low_bound,
                high=high_bound,
                shape=single_action_space.shape,
                dtype=np.float32,
            )
            base_env.action_space = gym.vector.utils.batch_space(base_env.single_action_space, base_env.num_envs)
            # Keep wrapper attributes synchronized when writable.
            try:
                env.action_space = base_env.action_space
            except Exception:
                pass

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # patch write_checkpoint to also upload .pt files to wandb when enabled
    if agent_cfg.get("agent", {}).get("experiment", {}).get("wandb", False):
        import wandb as _wandb
        _orig_write_checkpoint = runner.agent.write_checkpoint

        def _write_checkpoint_with_wandb(timestep, timesteps):
            _orig_write_checkpoint(timestep, timesteps)
            checkpoints_dir = os.path.join(runner.agent.experiment_dir, "checkpoints")
            if _wandb.run is not None and os.path.isdir(checkpoints_dir):
                _wandb.save(os.path.join(checkpoints_dir, "*.pt"), base_path=runner.agent.experiment_dir)

        runner.agent.write_checkpoint = _write_checkpoint_with_wandb

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
