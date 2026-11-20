# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark profile, environment, checkpoint, and policy setup."""

from __future__ import annotations

import importlib.metadata as metadata
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from packaging import version

import gymnasium as gym

import cli_args
from evaluation import (
    dynamic_crowd_profiles,
    fixed_grid_profile_indices,
    indoor_dynamic_profiles,
    room_corridor_profiles,
    select_scenario_profiles,
)
from evaluation_coverage import coverage_quotas
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    EVALUATION_SCENARIO_CODES,
    configure_indoor_dynamic_evaluation,
    configure_static_dynamic_evaluation,
    install_dynamic_crowd_evaluation_profiles,
)
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

try:
    from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
        EVALUATION_CROWD_SLOW_SPEED_RANGE,
        EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M,
        EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS,
        EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M,
        EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M,
        EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS,
        EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M,
    )
except ImportError:
    EVALUATION_CROWD_SLOW_SPEED_RANGE = None
    EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M = None
    EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS = None
    EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M = None
    EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M = None
    EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS = None
    EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M = None

@dataclass
class EvaluationSetup:
    agent_cfg: object
    profiles: list
    required_cells: int
    coverage_sweep: bool
    sweep_quotas: list[list[int]] | None
    checkpoint: str
    output_root: Path
    output_dir: Path
    failure_output_dir: Path
    env: object
    runner: object
    policy: object
    policy_nn: object | None
    env_profile_indices: list[int]
    rvo2_crowd_eval: bool
    leader_outcomes_available: bool
    slow_crowd_available: bool


def _fixed_grid_profile_indices(env, profiles, benchmark_suite) -> list[int]:
    """Map selected terrain cells to their benchmark profiles."""
    terrain = env.scene["terrain"]
    levels = terrain.terrain_levels.detach().cpu().tolist()
    columns = terrain.terrain_types.detach().cpu().tolist()
    if benchmark_suite == "indoor_dynamic":
        return fixed_grid_profile_indices(profiles, levels, columns, num_rows=4, num_cols=3)
    return fixed_grid_profile_indices(profiles, levels, columns, num_rows=8, num_cols=8)


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _resolve_checkpoint(agent_cfg, args_cli) -> tuple[str, str]:
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not checkpoint:
            raise RuntimeError(f"No published RSL-RL checkpoint is available for {train_task_name}.")
        return checkpoint, os.path.dirname(checkpoint)
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
        return checkpoint, os.path.dirname(checkpoint)
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return checkpoint, os.path.dirname(checkpoint)


def _create_timestamped_run_dir(output_root: Path) -> Path:
    """Create a unique, human-readable evaluation run directory."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    for sequence in range(1_000):
        suffix = "" if sequence == 0 else f"_{sequence:02d}"
        run_dir = output_root / f"{timestamp}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"Could not create a unique evaluation run directory in {output_root}.")



def prepare_evaluation(env_cfg, agent_cfg, *, args_cli, enabled_scenarios) -> EvaluationSetup:
    """Construct the selected benchmark and load the inference policy."""
    RVO2_CROWD_EVAL = "RVO2-Crowd" in args_cli.task
    if RVO2_CROWD_EVAL:
        from isaaclab_tasks.manager_based.navigation.config.go2.rvo2_navigation_eval_mixins import (
            EVALUATION_SCENARIO_CODES as RVO2_SCENARIO_CODES,
            configure_rvo2_dynamic_crowd_evaluation,
            install_rvo2_dynamic_crowd_evaluation_profiles,
        )
    LEADER_OUTCOMES_AVAILABLE = (
        not RVO2_CROWD_EVAL
        and "with_flow" in EVALUATION_SCENARIO_CODES
        and "with_flow_slow_leader" in EVALUATION_SCENARIO_CODES
        and EVALUATION_WITH_FLOW_LEADER_SPEED_RANGE_MPS is not None
        and EVALUATION_WITH_FLOW_LEADER_START_AHEAD_RANGE_M is not None
        and EVALUATION_WITH_FLOW_LEADER_LATERAL_OFFSET_RANGE_M is not None
        and EVALUATION_SLOW_LEADER_SPEED_RANGE_MPS is not None
        and EVALUATION_SLOW_LEADER_START_AHEAD_RANGE_M is not None
        and EVALUATION_SLOW_LEADER_LATERAL_OFFSET_RANGE_M is not None
    )
    SLOW_CROWD_AVAILABLE = (
        not RVO2_CROWD_EVAL
        and "crossing_slow" in EVALUATION_SCENARIO_CODES
        and "against_flow_slow" in EVALUATION_SCENARIO_CODES
        and EVALUATION_CROWD_SLOW_SPEED_RANGE is not None
    )
    if args_cli.benchmark_suite == "indoor_dynamic":
        if RVO2_CROWD_EVAL:
            raise ValueError("The indoor benchmark is supported only by the mixed temporal-LiDAR task family.")
        profiles = indoor_dynamic_profiles()
    else:
        profiles = dynamic_crowd_profiles(
            include_static=not RVO2_CROWD_EVAL,
            include_slow_leader=LEADER_OUTCOMES_AVAILABLE,
            include_slow_crowd=SLOW_CROWD_AVAILABLE,
        )
        if not RVO2_CROWD_EVAL:
            profiles.extend(room_corridor_profiles())
    profiles = select_scenario_profiles(profiles, enabled_scenarios[args_cli.benchmark_suite])
    required_cells = len(profiles)
    selected_cell_ids = tuple(sorted(
        profile.terrain_level * (3 if args_cli.benchmark_suite == "indoor_dynamic" else 8)
        + profile.terrain_column for profile in profiles
    ))
    if args_cli.num_envs is None:
        args_cli.num_envs = required_cells
    if RVO2_CROWD_EVAL:
        if args_cli.num_envs < len(profiles):
            raise ValueError(f"--num_envs must be at least {len(profiles)} for the benchmark profiles.")
    elif args_cli.num_envs <= 0 or args_cli.num_envs % required_cells:
        raise ValueError(
            f"--num_envs must be a positive multiple of {required_cells} for the selected benchmark suite."
        )

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if RVO2_CROWD_EVAL:
        configure_rvo2_dynamic_crowd_evaluation(env_cfg)
    elif args_cli.benchmark_suite == "indoor_dynamic":
        configure_indoor_dynamic_evaluation(env_cfg, selected_cell_ids=selected_cell_ids)
    else:
        configure_static_dynamic_evaluation(env_cfg, selected_cell_ids=selected_cell_ids)

    lidar_cfg = getattr(env_cfg, "held_scan_lidar", None)
    coverage_sweep = bool(lidar_cfg is not None and lidar_cfg.sparse_sampling_enabled)
    sweep_quotas = None
    if coverage_sweep:
        # The benchmark controls episode coverage, regardless of the task's
        # training-time density schedule or PLAY default.
        lidar_cfg.density_curriculum_enabled = False
        lidar_cfg.target_coverage = None
        sweep_quotas = coverage_quotas(args_cli.episodes_per_profile, args_cli.seeds)

    checkpoint, log_dir = _resolve_checkpoint(agent_cfg, args_cli)
    output_root = Path(args_cli.output_dir) if args_cli.output_dir else Path(log_dir) / "evaluations" / args_cli.benchmark_suite
    output_dir = _create_timestamped_run_dir(output_root)
    if args_cli.replay_output_dir:
        failure_output_dir = _create_timestamped_run_dir(Path(args_cli.replay_output_dir)) / "episode_cases"
    else:
        failure_output_dir = output_dir / "episode_cases"
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoint = handle_deprecated_rsl_rl_checkpoint(checkpoint, INSTALLED_RSL_RL_VERSION)
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

    env_profile_indices = (
        [index % len(profiles) for index in range(args_cli.num_envs)]
        if RVO2_CROWD_EVAL
        else _fixed_grid_profile_indices(env.unwrapped, profiles, args_cli.benchmark_suite)
    )
    if RVO2_CROWD_EVAL:
        install_rvo2_dynamic_crowd_evaluation_profiles(
            env.unwrapped,
            [profiles[index].pedestrian_count for index in env_profile_indices],
            [RVO2_SCENARIO_CODES[profiles[index].scenario] for index in env_profile_indices],
        )
    else:
        install_dynamic_crowd_evaluation_profiles(
            env.unwrapped,
            [profiles[index].pedestrian_count for index in env_profile_indices],
            [EVALUATION_SCENARIO_CODES.get(profiles[index].scenario, -1) for index in env_profile_indices],
        )
    return EvaluationSetup(
        agent_cfg=agent_cfg, profiles=profiles, required_cells=required_cells,
        coverage_sweep=coverage_sweep, sweep_quotas=sweep_quotas, checkpoint=checkpoint,
        output_root=output_root, output_dir=output_dir, failure_output_dir=failure_output_dir,
        env=env, runner=runner, policy=policy, policy_nn=policy_nn,
        env_profile_indices=env_profile_indices, rvo2_crowd_eval=RVO2_CROWD_EVAL,
        leader_outcomes_available=LEADER_OUTCOMES_AVAILABLE,
        slow_crowd_available=SLOW_CROWD_AVAILABLE,
    )
