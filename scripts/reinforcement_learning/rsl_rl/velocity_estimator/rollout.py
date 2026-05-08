# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect rollouts from an RSL-RL policy and export them as episode datasets."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# Allow importing shared RSL-RL CLI helpers from the parent script folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cli_args  # isort: skip

# Launch Isaac Sim Simulator first.


parser = argparse.ArgumentParser(description="Collect rollout datasets with an RSL-RL policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max_episodes",
    type=int,
    default=100,
    help="Maximum number of completed episodes to save. Set to 0 to run indefinitely.",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="datasets/",
    help="Directory where rollout datasets will be written.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="rollout",
    help="Prefix used for generated dataset files.",
)
parser.add_argument(
    "--episodes_per_file",
    type=int,
    default=100,
    help="Maximum number of completed episodes to store in a single HDF5 file before flushing and rotating.",
)
parser.add_argument(
    "--log_interval",
    type=float,
    default=5.0,
    help="Seconds between progress logs.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from scripts.reinforcement_learning.rsl_rl.velocity_estimator.src.observation_utils import ObservationTermSpec, build_observation_term_specs, get_estimator_target_paths, get_estimator_target_term_names, serialize_observation_specs, split_observation_groups


def _resolve_resume_path(agent_cfg: RslRlOnPolicyRunnerCfg) -> tuple[str, str]:
    """Resolve the checkpoint path and the base logging directory."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for rollout collection.")

    resume_path = retrieve_file_path(args_cli.checkpoint)
    return resume_path, os.path.dirname(resume_path)


def _add_observation_step(
    episode: EpisodeData,
    obs_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    observation_specs: dict[str, list[ObservationTermSpec]],
    env_id: int,
) -> None:
    """Append the current observation snapshot for one environment."""
    extracted_groups = split_observation_groups(obs_dict, observation_specs, env_id)
    ground_truth = extracted_groups.pop("ground_truth", None)
    estimator_target_terms = get_estimator_target_term_names()

    episode.add("observations", extracted_groups)
    if ground_truth is not None:
        episode.add(
            "ground_truth",
            {term_name: ground_truth[term_name] for term_name in estimator_target_terms},
        )


def _add_step_transition(
    episode: EpisodeData,
    actions: torch.Tensor,
    env_id: int,
    step_index: int,
) -> None:
    """Append pre-step data for one environment."""
    episode.add("actions", actions[env_id].detach().clone())
    episode.add(
        "step_index",
        torch.tensor([step_index], dtype=torch.int64, device=actions.device),
    )


def _add_step_outcome(
    episode: EpisodeData,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    env_id: int,
) -> None:
    """Append post-step rollout signals for one environment."""
    episode.add("rewards", rewards[env_id].detach().clone().reshape(1))
    episode.add("dones", dones[env_id].detach().to(dtype=torch.bool).reshape(1).clone())
    episode.add("time_outs", time_outs[env_id].detach().to(dtype=torch.bool).reshape(1).clone())


def _add_episode_metadata(episode: EpisodeData, env_id: int, episode_steps: int, termination_flags: dict[str, bool], device: str) -> None:
    """Store metadata that is constant for a finished episode."""
    episode.add("metadata/source_env_id", torch.tensor([env_id], dtype=torch.int64, device=device))
    episode.add("metadata/num_steps", torch.tensor([episode_steps], dtype=torch.int64, device=device))
    for term_name, is_active in termination_flags.items():
        episode.add(
            f"metadata/termination_terms/{term_name}",
            torch.tensor([is_active], dtype=torch.bool, device=device),
        )


def _extract_termination_flags(extras: dict, env: RslRlVecEnvWrapper, env_id: int) -> dict[str, bool]:
    """Resolve named termination flags for a completed episode from extras logs."""
    log_data = extras.get("log", {})
    termination_flags: dict[str, bool] = {}

    for term_name in env.unwrapped.termination_manager.active_terms:
        env_ids = log_data.get(f"Episode_Termination/Envs/Ids/{term_name}")
        if env_ids is None:
            termination_flags[term_name] = False
            continue

        flattened_env_ids = env_ids.reshape(-1)
        termination_flags[term_name] = bool(torch.any(flattened_env_ids == env_id).item())

    return termination_flags


class RolloutDatasetWriter:
    """Write completed episodes into a sequence of HDF5 files."""

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        env_name: str,
        episodes_per_file: int,
        env_args: dict[str, object],
    ) -> None:
        self.dataset_root = os.path.abspath(dataset_root)
        self.dataset_name = dataset_name
        self.env_name = env_name
        self.episodes_per_file = episodes_per_file
        self.env_args = env_args

        self.total_episodes = 0
        self._file_index = -1
        self._handler: HDF5DatasetFileHandler | None = None

        os.makedirs(self.dataset_root, exist_ok=True)
        self._open_next_file()

    @property
    def current_file_path(self) -> str:
        """Path to the currently open dataset file."""
        return os.path.join(self.dataset_root, f"{self.dataset_name}_{self._file_index:04d}.hdf5")

    def write_episode(self, episode: EpisodeData) -> None:
        """Write a completed episode and rotate files when needed."""
        if self._handler is None:
            raise RuntimeError("Dataset writer is not initialized.")
        if self._handler.get_num_episodes() >= self.episodes_per_file:
            self._rotate_file()

        self._handler.write_episode(episode)
        self.total_episodes += 1

        if self._handler.get_num_episodes() >= self.episodes_per_file:
            self._handler.flush()

    def close(self) -> None:
        """Flush and close the current dataset file."""
        if self._handler is not None:
            self._handler.flush()
            self._handler.close()
            self._handler = None

    def _rotate_file(self) -> None:
        """Close the current file and start a new one."""
        if self._handler is not None:
            self._handler.flush()
            self._handler.close()
        self._open_next_file()

    def _open_next_file(self) -> None:
        """Create the next chunk file and attach dataset metadata."""
        self._file_index += 1
        handler = HDF5DatasetFileHandler()
        self._handler = handler
        handler.create(self.current_file_path, env_name=self.env_name)
        handler.add_env_args(self.env_args)
        print(f"[INFO] Writing rollout episodes to: {self.current_file_path}")


def main() -> None:
    """Run the rollout loop and export completed episodes."""
    if args_cli.episodes_per_file <= 0:
        raise ValueError("--episodes_per_file must be greater than zero.")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    resume_path, _ = _resolve_resume_path(agent_cfg)

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    if not isinstance(base_env.unwrapped, ManagerBasedRLEnv):
        raise RuntimeError(
            f"This rollout script currently supports manager-based RL environments, got {type(base_env.unwrapped)}."
        )

    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    observation_specs = build_observation_term_specs(env.unwrapped, "rollout script")
    dataset_writer = RolloutDatasetWriter(
        dataset_root=args_cli.dataset_root,
        dataset_name=args_cli.dataset_name,
        env_name=args_cli.task,
        episodes_per_file=args_cli.episodes_per_file,
        env_args={
            "source": "rsl_rl.rollout",
            "checkpoint": resume_path,
            "observation_spec": serialize_observation_specs(observation_specs),
            "ground_truth_group": "ground_truth" if "ground_truth" in observation_specs else "",
            "estimator_target_paths": get_estimator_target_paths(observation_specs),
            "num_envs": env.num_envs,
            "device": str(agent_cfg.device),
        },
    )

    obs, extras = env.get_observations()
    current_obs_dict = extras["observations"]
    episode_buffers = [EpisodeData() for _ in range(env.num_envs)]
    episode_step_counts = [0 for _ in range(env.num_envs)]

    completed_episodes = 0
    dt = env.unwrapped.step_dt
    step_count = 0
    ratio_window_start_wall = time.time()
    ratio_window_sim_time = 0.0
    max_episodes = None if args_cli.max_episodes == 0 else args_cli.max_episodes

    print(
        f"[INFO] Saving completed episodes to {os.path.abspath(args_cli.dataset_root)} "
        f"with up to {args_cli.episodes_per_file} episodes per file."
    )
    if max_episodes is None:
        print("[INFO] Collecting rollout data indefinitely. Press Ctrl+C to stop.")
    else:
        print(f"[INFO] Collecting {max_episodes} completed episodes.")

    try:
        while simulation_app.is_running():
            loop_start_time = time.time()

            with torch.inference_mode():
                actions = policy(obs)

            for env_id, episode in enumerate(episode_buffers):
                _add_observation_step(episode, current_obs_dict, observation_specs, env_id)
                _add_step_transition(episode, actions, env_id, episode_step_counts[env_id])

            with torch.inference_mode():
                obs, rewards, dones, extras = env.step(actions)

            time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            next_obs_dict = extras["observations"]

            for env_id, episode in enumerate(episode_buffers):
                _add_step_outcome(episode, rewards, dones, time_outs, env_id)
                episode_step_counts[env_id] += 1

                if not bool(dones[env_id].item()):
                    continue

                termination_flags = _extract_termination_flags(extras, env, env_id)
                _add_episode_metadata(
                    episode,
                    env_id=env_id,
                    episode_steps=episode_step_counts[env_id],
                    termination_flags=termination_flags,
                    device=env.device,
                )
                dataset_writer.write_episode(episode)
                completed_episodes += 1

                episode_buffers[env_id] = EpisodeData()
                episode_step_counts[env_id] = 0

                if max_episodes is not None and completed_episodes >= max_episodes:
                    break

            step_count += 1
            ratio_window_sim_time += dt
            current_obs_dict = next_obs_dict

            if max_episodes is not None and completed_episodes >= max_episodes:
                print(f"[INFO] Reached the requested {completed_episodes} completed episodes.")
                break

            elapsed_wall = time.time() - ratio_window_start_wall
            if elapsed_wall >= args_cli.log_interval:
                sim_to_realtime_ratio = ratio_window_sim_time / elapsed_wall
                print(
                    f"[INFO] steps={step_count} completed_episodes={completed_episodes} "
                    f"sim_to_real={sim_to_realtime_ratio:.3f}x"
                )
                ratio_window_start_wall = time.time()
                ratio_window_sim_time = 0.0

            sleep_time = dt - (time.time() - loop_start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        dataset_writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()