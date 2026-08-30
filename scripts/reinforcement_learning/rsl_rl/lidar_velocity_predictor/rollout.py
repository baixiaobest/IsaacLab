"""Collect scan-event point-velocity supervision from a trained Kp policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect temporal-LiDAR point velocity labels.")
parser.add_argument("--task", type=str, default="Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Point-Velocity-Data-Unitree-Go2-Play-v0")
parser.add_argument("--num_envs", type=int, default=40)
parser.add_argument("--max_episodes", type=int, default=1000)
parser.add_argument("--dataset_root", type=str, default="datasets/lidar_point_velocity")
parser.add_argument("--dataset_name", type=str, default="mixed_kp")
parser.add_argument("--episodes_per_file", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs <= 0 or args_cli.num_envs % 40:
    parser.error("--num_envs must be a positive multiple of 40 for exact tile coverage.")
if args_cli.max_episodes <= 0 or args_cli.episodes_per_file <= 0:
    parser.error("--max_episodes and --episodes_per_file must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import importlib.metadata as metadata
import isaaclab_tasks  # noqa: F401


def _slice_term(env: ManagerBasedRLEnv, observations: dict, group: str, term_name: str) -> torch.Tensor:
    names = env.observation_manager.active_terms[group]
    shapes = env.observation_manager.group_obs_term_dim[group]
    try:
        term_index = names.index(term_name)
    except ValueError as error:
        raise RuntimeError(f"Observation group '{group}' is missing '{term_name}'.") from error
    start = sum(int(np.prod(shape)) for shape in shapes[:term_index])
    stop = start + int(np.prod(shapes[term_index]))
    tensor = observations[group][:, start:stop]
    return tensor.reshape(tensor.shape[0], *shapes[term_index])


@dataclass
class EpisodeBuffer:
    fields: dict[str, list[torch.Tensor]] = field(default_factory=lambda: {key: [] for key in (
        "lidar_noisy", "lidar_clean", "point_velocity_b", "reflection_mask", "dynamic_mask", "range_m", "capture_index"
    )})
    terrain_name: str = "unknown"
    terrain_level: int = -1
    replica_index: int = -1
    scenario_mode: int = -1

    def append(self, **values: torch.Tensor) -> None:
        for key, value in values.items():
            self.fields[key].append(value.detach().cpu().clone())

    def __len__(self) -> int:
        return len(self.fields["lidar_noisy"])


class ChunkWriter:
    """Atomically write complete HDF5 episode chunks and never overwrite old chunks."""

    def __init__(self, root: str, name: str, episodes_per_file: int, metadata: dict) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.name, self.episodes_per_file, self.metadata = name, episodes_per_file, metadata
        existing = sorted(self.root.glob(f"{name}_*.hdf5"))
        for path in existing:
            with h5py.File(path, "r") as handle:
                data = handle.get("data")
                encoded = data.attrs.get("metadata") if data is not None else None
                try:
                    existing_metadata = json.loads(encoded) if encoded is not None else {}
                except (TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError(f"Cannot safely resume dataset: invalid metadata in {path}.") from error
                if existing_metadata.get("schema_version") != 2 or existing_metadata.get("velocity_frame") != "body_xy":
                    raise RuntimeError(
                        f"{path} uses an incompatible LiDAR velocity schema. Archive or remove the old world-frame "
                        "files before collecting body-frame schema-v2 samples."
                    )
        self.file_index = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1 if existing else 0
        self.episode_index = 0
        self.pending: list[EpisodeBuffer] = []

    def add(self, episode: EpisodeBuffer) -> None:
        self.pending.append(episode)
        if len(self.pending) >= self.episodes_per_file:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        destination = self.root / f"{self.name}_{self.file_index:04d}.hdf5"
        temporary = destination.with_suffix(".tmp")
        with h5py.File(temporary, "w") as handle:
            data = handle.create_group("data")
            data.attrs["metadata"] = json.dumps(self.metadata)
            for episode in self.pending:
                group = data.create_group(f"episode_{self.episode_index:07d}")
                for key, values in episode.fields.items():
                    group.create_dataset(key, data=torch.stack(values).numpy(), compression="gzip", compression_opts=4)
                group.attrs["terrain_name"] = episode.terrain_name
                group.attrs["terrain_level"] = episode.terrain_level
                group.attrs["replica_index"] = episode.replica_index
                group.attrs["scenario_mode"] = episode.scenario_mode
                self.episode_index += 1
        os.replace(temporary, destination)
        print(f"[INFO] Wrote {len(self.pending)} episodes to {destination}")
        self.pending.clear()
        self.file_index += 1


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for point-velocity rollout collection.")

    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    if not isinstance(base_env.unwrapped, ManagerBasedRLEnv):
        raise RuntimeError("Point velocity collection requires a manager-based RL environment.")
    # Gymnasium's outer ``OrderEnforcing`` wrapper does not proxy ``seed``.
    # Seed the Isaac environment directly, as the other Isaac rollout tools do.
    base_env.unwrapped.seed(args_cli.seed)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    terrain = env.unwrapped.scene.terrain
    terrain_names = terrain.get_env_terrain_names()
    replicas = getattr(terrain, "tile_replicas", torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    writer = ChunkWriter(args_cli.dataset_root, args_cli.dataset_name, args_cli.episodes_per_file, {
        "task": args_cli.task, "checkpoint": checkpoint, "num_envs": env.num_envs, "sample_period_s": 0.130,
        "schema_version": 2, "velocity_frame": "body_xy", "seed": args_cli.seed,
    })
    buffers = [EpisodeBuffer() for _ in range(env.num_envs)]
    previous_capture = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    completed = 0
    observations = env.get_observations()
    try:
        while simulation_app.is_running() and completed < args_cli.max_episodes:
            labels = env.unwrapped.get_point_velocity_labels()
            capture_index = labels["capture_index"]
            new_scan = capture_index != previous_capture
            noisy = _slice_term(env.unwrapped, observations, "policy", "obstacle_scan")
            clean = _slice_term(env.unwrapped, observations, "critic", "obstacle_scan")
            for env_id in new_scan.nonzero(as_tuple=False).squeeze(-1).tolist():
                buffer = buffers[env_id]
                if len(buffer) == 0:
                    buffer.terrain_name = terrain_names[env_id]
                    buffer.terrain_level = int(terrain.terrain_levels[env_id].item())
                    buffer.replica_index = int(replicas[env_id].item())
                    buffer.scenario_mode = int(env.unwrapped.pedestrian_scenario_mode[env_id].item())
                buffer.append(
                    lidar_noisy=noisy[env_id], lidar_clean=clean[env_id],
                    point_velocity_b=labels["point_velocity_b"][env_id],
                    reflection_mask=labels["reflection_mask"][env_id], dynamic_mask=labels["dynamic_mask"][env_id],
                    range_m=labels["range_m"][env_id], capture_index=capture_index[env_id].reshape(1),
                )
            previous_capture = capture_index.clone()
            with torch.inference_mode():
                action = policy(observations)
                observations, _, dones, _ = env.step(action)
                policy.reset(dones)
            for env_id in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
                if len(buffers[env_id]) > 0:
                    writer.add(buffers[env_id])
                    completed += 1
                buffers[env_id] = EpisodeBuffer()
                if completed >= args_cli.max_episodes:
                    break
            if completed and completed % 50 == 0:
                print(f"[INFO] completed_episodes={completed}/{args_cli.max_episodes}")
    finally:
        writer.flush()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
