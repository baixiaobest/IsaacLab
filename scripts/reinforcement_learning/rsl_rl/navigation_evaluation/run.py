# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a mixed Go2 policy on the fixed static-plus-dynamic benchmark."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from packaging import version

from evaluation import (
    CollisionReplayRecorder,
    EpisodeMetricsCollector,
    EpisodeVelocityAccumulator,
    GOAL_REGION_COLLISION_RADIUS_M,
    InteractionEventCollector,
    InteractionEventReplayRecorder,
    LeaderOutcomeCollector,
    terminal_collision_ids,
    terminal_goal_region_collision_ids,
)
from evaluation_coverage import COVERAGE_TARGETS, subtract_counts
from evaluation_telemetry import ParquetTelemetryRecorder
from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.manager_based.navigation.lidar_geometry import forward_lidar_reflection_bins

from navigation_evaluation.progress import EvaluationProgressReporter
from navigation_evaluation.run_output import write_run_output
from navigation_evaluation.setup import INSTALLED_RSL_RL_VERSION, prepare_evaluation

def _leader_conditions_by_env(raw_env, profiles, env_profile_indices, leader_outcomes_available) -> dict[int, dict[str, float]]:
    """Snapshot designated-leader reset samples before an evaluation step."""
    required = (
        "evaluation_leader_speed_mps",
        "evaluation_leader_start_ahead_m",
        "evaluation_leader_lateral_offset_m",
    )
    if not leader_outcomes_available or not all(hasattr(raw_env, name) for name in required):
        return {}
    speeds = raw_env.evaluation_leader_speed_mps.detach().cpu().tolist()
    aheads = raw_env.evaluation_leader_start_ahead_m.detach().cpu().tolist()
    offsets = raw_env.evaluation_leader_lateral_offset_m.detach().cpu().tolist()
    return {
        env_id: {
            "speed_mps": float(speeds[env_id]),
            "start_ahead_m": float(aheads[env_id]),
            "lateral_offset_m": float(offsets[env_id]),
        }
        for env_id, profile_index in enumerate(env_profile_indices)
        if profiles[profile_index].scenario in ("with_flow", "with_flow_slow_leader")
    }


def _leader_condition_summary(records: list[dict[str, float]]) -> dict[str, float | int | None]:
    """Produce compact artifact metadata for sampled designated-leader conditions."""
    summary: dict[str, float | int | None] = {"episodes": len(records)}
    for key in ("speed_mps", "start_ahead_m", "lateral_offset_m"):
        values = [record[key] for record in records]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
        summary[f"min_{key}"] = min(values) if values else None
        summary[f"max_{key}"] = max(values) if values else None
    return summary


def _snapshot_cbf_solver_metrics(action_term, env_ids) -> dict[int, dict[str, float | int]]:
    """Snapshot CBF QP and barrier diagnostics before completed environments reset."""
    metrics = getattr(action_term, "solver_metrics", None)
    slack_metrics = getattr(action_term, "slack_metrics", None)
    if metrics is None and slack_metrics is None:
        return {}
    if hasattr(env_ids, "detach"):
        environment_ids = env_ids.detach().cpu().tolist()
    else:
        environment_ids = list(env_ids)
    snapshots: dict[int, dict[str, float | int]] = {}
    for env_id in environment_ids:
        snapshot: dict[str, float | int] = {}
        for name, values in (metrics or {}).items():
            value = values[int(env_id)]
            if hasattr(value, "item"):
                value = value.item()
            snapshot[name] = int(value) if isinstance(value, int) else float(value)
        for name, values in (slack_metrics or {}).items():
            value = values[int(env_id)]
            if hasattr(value, "item"):
                value = value.item()
            snapshot[f"slack_{name}"] = int(value) if isinstance(value, int) else float(value)
        snapshots[int(env_id)] = snapshot
    return snapshots


def _cbf_solver_summary(records: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    """Aggregate completed-episode CBF QP telemetry for evaluation metadata."""
    if not records:
        return None
    solve_count = sum(int(record["solve_count"]) for record in records)
    solve_time_total_s = sum(float(record["solve_time_total_s"]) for record in records)
    update_time_total_s = sum(float(record["update_time_total_s"]) for record in records)
    polish_time_total_s = sum(float(record["polish_time_total_s"]) for record in records)
    return {
        "episodes": len(records),
        "solve_count": solve_count,
        "mean_iterations_per_solve": (
            sum(int(record["iteration_total"]) for record in records) / solve_count if solve_count else 0.0
        ),
        "max_iterations": max(int(record["iteration_max"]) for record in records),
        "mean_solve_time_ms": 1_000.0 * solve_time_total_s / solve_count if solve_count else 0.0,
        "max_solve_time_ms": 1_000.0 * max(float(record["solve_time_max_s"]) for record in records),
        "total_solve_time_s": solve_time_total_s,
        "total_update_time_s": update_time_total_s,
        "total_polish_time_s": polish_time_total_s,
        "max_primal_residual": max(float(record["primal_residual_max"]) for record in records),
        "max_dual_residual": max(float(record["dual_residual_max"]) for record in records),
        "solved_inaccurately_count": sum(int(record["inaccurate_count"]) for record in records),
        "maximum_iteration_reached_count": sum(int(record["max_iteration_count"]) for record in records),
        "max_point_slack": max(float(record.get("slack_max", 0.0)) for record in records),
        "mean_nonzero_max_point_slack": (
            sum(float(record.get("slack_mean_nonzero", 0.0)) for record in records) / len(records)
        ),
        "max_active_lidar_points": max(int(record.get("slack_active_point_count", 0)) for record in records),
        "minimum_barrier_residual_at_episode_end": min(
            float(record.get("slack_current_min_residual", 0.0)) for record in records
        ),
        "command_envelope_guard_count": sum(
            int(record.get("slack_command_envelope_guard_count", 0)) for record in records
        ),
        "timing_scope": "OSQP update/solve/polish only; excludes CBF construction and PyTorch CPU/GPU transfers",
    }



def run_evaluation(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    *, args_cli, simulation_app, enabled_scenarios,
):
    """Run all static-plus-dynamic profiles in parallel until every profile reaches its quota."""
    setup = prepare_evaluation(env_cfg, agent_cfg, args_cli=args_cli, enabled_scenarios=enabled_scenarios)
    agent_cfg = setup.agent_cfg
    profiles = setup.profiles
    required_cells = setup.required_cells
    coverage_sweep = setup.coverage_sweep
    sweep_quotas = setup.sweep_quotas
    checkpoint = setup.checkpoint
    output_root = setup.output_root
    output_dir = setup.output_dir
    failure_output_dir = setup.failure_output_dir
    env = setup.env
    runner = setup.runner
    policy = setup.policy
    policy_nn = setup.policy_nn
    env_profile_indices = setup.env_profile_indices
    RVO2_CROWD_EVAL = setup.rvo2_crowd_eval
    LEADER_OUTCOMES_AVAILABLE = setup.leader_outcomes_available
    SLOW_CROWD_AVAILABLE = setup.slow_crowd_available
    obs, _ = env.reset()
    collector = EpisodeMetricsCollector(profiles, env_profile_indices, args_cli.episodes_per_profile)
    leader_records: list[dict[str, float | int]] = []
    seed_count = args_cli.seeds
    if seed_count < 1:
        raise ValueError("--seeds must be at least 1.")
    seeds = [args_cli.seed + index for index in range(seed_count)]
    # Each seed contributes an equal share of the per-profile episode budget; the last
    # seed absorbs the remainder (e.g. 100 episodes / 3 seeds -> 34, 33, 33).
    per_seed_quota = math.ceil(args_cli.episodes_per_profile / seed_count)
    stage_counts: list[tuple[float, list[dict]]] = []
    accepted_coverage: dict[float, dict[str, float]] = {
        target: {"ray_sum": 0.0, "reflection_sum": 0.0, "episodes": 0.0} for target in COVERAGE_TARGETS
    }
    terminal_coverage: dict[int, tuple[float, float]] = {}
    progress_reporter = EvaluationProgressReporter(
        profile_count=len(profiles),
        episodes_per_profile=args_cli.episodes_per_profile,
        seed_count=seed_count,
    )
    velocity_accumulator = EpisodeVelocityAccumulator(args_cli.num_envs)
    goal_region_collision_ids: set[int] = set()
    cbf_solver_terminal_metrics: dict[int, dict[str, float | int]] = {}
    cbf_solver_episode_metrics: list[dict[str, float | int]] = []

    # Some task variants do not publish command-manager metrics in ``extras["log"]``.
    # Capture the terminal state immediately before ManagerBasedRLEnv resets it, while the
    # per-step samples below provide the rest of each episode's world-XY speed trace.
    raw_env = env.unwrapped
    step_dt_s = env.unwrapped.step_dt
    episode_length_s = env.unwrapped.cfg.episode_length_s
    coverage_action_metadata = {}
    if coverage_sweep:
        action_cfg = raw_env.action_manager.get_term("pre_trained_policy_action").cfg
        coverage_action_metadata = {
            "scan_source": getattr(action_cfg, "scan_source", None),
            "scan_type": getattr(action_cfg, "scan_type", None),
            "predictor_100_percent_outside_training_range": bool(
                getattr(action_cfg, "velocity_predictor_jit_path", None)
            ),
        }
    structured_telemetry = args_cli.telemetry_dir is not None
    interaction_collector = InteractionEventCollector(profiles, env_profile_indices, step_dt_s)
    leader_outcome_collector = LeaderOutcomeCollector(profiles, env_profile_indices, step_dt_s)
    replay_recorder = None
    if (
        not args_cli.disable_failure_recording
        or args_cli.success_cases_per_scenario
        or args_cli.interaction_event_cases_per_label
        or structured_telemetry
    ):
        replay_recorder = CollisionReplayRecorder(
            profiles,
            env_profile_indices,
            failure_output_dir,
            step_dt_s,
            args_cli.failure_history_seconds,
            goal_region_radius_m=GOAL_REGION_COLLISION_RADIUS_M,
            successes_per_scenario=args_cli.success_cases_per_scenario,
            episode_length_s=episode_length_s,
            record_collisions=not args_cli.disable_failure_recording,
            interesting_interaction_distance_m=args_cli.interesting_interaction_distance_m,
            # Event clips are admitted only after terminal success. Retaining each episode
            # ensures a slow-leader pass that occurs well before goal reach remains available.
            retain_full_episode=bool(args_cli.interaction_event_cases_per_label) or structured_telemetry,
        )
    interaction_replay_recorder = (
        InteractionEventReplayRecorder(
            failure_output_dir / "interaction_events",
            replay_recorder,
            args_cli.interaction_event_cases_per_label,
            args_cli.interaction_event_padding_seconds,
        )
        if replay_recorder is not None and args_cli.interaction_event_cases_per_label
        else None
    )
    telemetry_recorder = (
        ParquetTelemetryRecorder(
            Path(args_cli.telemetry_dir),
            os.environ["RESEARCH_AGENT_TELEMETRY_DATASET_ID"],
            replay_recorder,
            profiles,
            env_profile_indices,
            step_dt_s,
            source_commit=os.environ.get("RESEARCH_AGENT_EVALUATION_COMMIT"),
        )
        if structured_telemetry and replay_recorder is not None else None
    )
    active_seed = [seeds[0]]

    def _record_cbf_replay_state() -> None:
        """Write final CBF velocity and acceleration state for the latest replay frame."""
        if replay_recorder is None:
            return
        action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        command = getattr(action_term, "cbf_filtered_velocity_command", None)
        if command is not None:
            replay_recorder.record_cbf_filtered_command(command)
        nominal_acceleration = getattr(action_term, "nominal_acceleration", None)
        filtered_acceleration = getattr(action_term, "safe_acceleration", None)
        if nominal_acceleration is not None and filtered_acceleration is not None:
            replay_recorder.record_cbf_accelerations(nominal_acceleration, filtered_acceleration)

    original_reset_idx = raw_env._reset_idx

    def _tracked_reset_idx(env_ids):
        if coverage_sweep:
            capture = raw_env._held_scan_lidar_collector.latest_policy_capture()
            scan_state = capture["ray_state"]
            reflected_bins = forward_lidar_reflection_bins(capture)["reflection_mask"]
            for env_id in env_ids.detach().cpu().tolist():
                state = scan_state[env_id]
                terminal_coverage[int(env_id)] = (
                    min(1.0, float((state > 0).sum().item()) / 128.0),
                    float(reflected_bins[env_id].float().mean().item()),
                )
        terminal_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
        velocity_accumulator.record_terminal(terminal_speed, env_ids)
        collision_ids = terminal_collision_ids(raw_env, env_ids, profiles, env_profile_indices)
        goal_region_collision_ids.update(
            terminal_goal_region_collision_ids(
                raw_env, env_ids, GOAL_REGION_COLLISION_RADIUS_M, collision_env_ids=collision_ids
            )
        )
        if not structured_telemetry:
            interaction_collector.finalize_terminal(env_ids)
            leader_outcome_collector.finalize_terminal(raw_env, env_ids)
        try:
            action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
        except KeyError:
            action_term = None
        if action_term is not None:
            cbf_solver_terminal_metrics.update(_snapshot_cbf_solver_metrics(action_term, env_ids))
        def _termination_ids(name):
            try:
                term = raw_env.termination_manager.get_term(name)
            except KeyError:
                return torch.empty(0, dtype=torch.long, device=env_ids.device)
            return torch.nonzero(term, as_tuple=False).reshape(-1)

        success_env_ids = _termination_ids("goal_reached")
        timeout_env_ids = _termination_ids("time_out")
        base_contact_env_ids = _termination_ids("base_contact")
        if interaction_replay_recorder is not None:
            resetting_env_ids = set(env_ids.detach().cpu().tolist())
            for env_id in success_env_ids.detach().cpu().tolist():
                if env_id in resetting_env_ids:
                    interaction_replay_recorder.stage_terminal_success(
                        raw_env, int(env_id), interaction_collector.pending_events(int(env_id))
                    )
                    leader_outcome = leader_outcome_collector.pending_outcome(int(env_id))
                    if leader_outcome is not None:
                        interaction_replay_recorder.stage_terminal_success(raw_env, int(env_id), [leader_outcome])
        _record_cbf_replay_state()
        if telemetry_recorder is not None:
            telemetry_recorder.stage_terminal(
                raw_env, env_ids, seed=active_seed[0], success_ids=success_env_ids,
                collision_ids=collision_ids, timeout_ids=timeout_env_ids,
                base_contact_ids=base_contact_env_ids,
                goal_region_collision_ids=goal_region_collision_ids,
            )
        if replay_recorder is not None:
            replay_recorder.capture_terminal_episodes(
                raw_env, env_ids, success_env_ids, collision_env_ids=collision_ids
            )
        return original_reset_idx(env_ids)

    raw_env._reset_idx = _tracked_reset_idx

    @torch.inference_mode()
    def _reset_for_coverage_stage(target: float) -> None:
        """Discard in-flight episodes before changing the capture template."""
        nonlocal obs
        all_ids = torch.arange(args_cli.num_envs, device=raw_env.device)
        lidar = raw_env._held_scan_lidar_collector
        lidar.set_episode_coverage_targets(all_ids, torch.full((args_cli.num_envs,), target, device=raw_env.device))
        if replay_recorder is not None:
            replay_recorder.coverage_target = target
            replay_recorder.reset(all_ids)
        if telemetry_recorder is not None:
            telemetry_recorder.coverage_target = target
        if not structured_telemetry:
            interaction_collector.finalize_terminal(all_ids)
            interaction_collector.resolve_terminal(all_ids, set())
            leader_outcome_collector.finalize_terminal(raw_env, all_ids)
            leader_outcome_collector.resolve_terminal(all_ids, set())
        velocity_accumulator.reset(all_ids)
        goal_region_collision_ids.clear()
        terminal_coverage.clear()
        cbf_solver_terminal_metrics.clear()
        # This is a controlled stage reset, not a completed benchmark episode.
        raw_env._reset_idx = original_reset_idx
        try:
            obs, _ = env.reset()
        finally:
            raw_env._reset_idx = _tracked_reset_idx
        reset_mask = torch.ones(args_cli.num_envs, dtype=torch.bool, device=raw_env.device)
        if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
            policy.reset(reset_mask)
        else:
            policy_nn.reset(reset_mask)

    print(
        f"[INFO] Evaluating {checkpoint} on {len(profiles)} static-plus-dynamic profiles "
        f"with {args_cli.episodes_per_profile} episodes each"
        + (f" across {seed_count} consecutive seeds ({seeds[0]}..{seeds[-1]})." if seed_count > 1 else ".")
    )
    progress_reporter.report(
        collector.total_episodes,
        seed=seeds[0],
        seed_index=1,
        status="running",
        force=True,
    )
    if args_cli.success_cases_per_scenario:
        print(
            "[INFO] Recording "
            f"{args_cli.success_cases_per_scenario} interesting complete success replay(s) per scenario "
            f"(robot-agent distance < {args_cli.interesting_interaction_distance_m:.2f} m)."
        )
    try:
        cumulative_limit = 0
        for seed_index, seed in enumerate(seeds):
            active_seed[0] = seed
            if seed_index > 0 and not coverage_sweep:
                # Re-seed all global RNGs so the next chunk draws a fresh episode
                # sequence. Episodes already in flight finish under their original
                # draws; each profile has at most one in-flight episode, which is
                # attributed to the stage in which it completes (see
                # EpisodeMetricsCollector.per_seed_counts).
                env.unwrapped.seed(seed)
                print(f"[INFO] Advancing to seed {seed} (stage {seed_index + 1} of {seed_count}).")
            targets = COVERAGE_TARGETS if coverage_sweep else (None,)
            for coverage_index, target in enumerate(targets):
                quota = sweep_quotas[seed_index][coverage_index] if coverage_sweep else min(
                    per_seed_quota, args_cli.episodes_per_profile - cumulative_limit
                )
                before_stage = collector.snapshot_counts() if coverage_sweep else None
                cumulative_limit += quota
                collector.set_stage_limit(cumulative_limit, record_seed_boundary=coverage_index == 0)
                if quota and target is not None:
                    raw_env.seed(seed)
                    _reset_for_coverage_stage(target)
                progress_reporter.report(
                    collector.total_episodes, seed=seed, seed_index=seed_index + 1,
                    status="running", force=True,
                )
                while simulation_app.is_running() and not collector.stage_complete:
                    step_speed = torch.linalg.vector_norm(raw_env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
                    velocity_accumulator.record_step(step_speed)
                    leader_conditions = _leader_conditions_by_env(
                        raw_env, profiles, env_profile_indices, LEADER_OUTCOMES_AVAILABLE
                    )
                    with torch.inference_mode():
                        actions = policy(obs)
                        if replay_recorder is not None:
                            submitted_actions = actions
                            if env.clip_actions is not None:
                                submitted_actions = torch.clamp(submitted_actions, -env.clip_actions, env.clip_actions)
                            action_term = raw_env.action_manager.get_term("pre_trained_policy_action")
                            action_scales = torch.as_tensor(action_term.cfg.action_scales, device=submitted_actions.device)
                            cbf_command = getattr(action_term, "cbf_filtered_velocity_command", None)
                            replay_recorder.record_pre_step(
                                raw_env, submitted_actions * action_scales, cbf_filtered_command=cbf_command
                            )
                        if not structured_telemetry:
                            interaction_collector.record_pre_step(raw_env)
                            leader_outcome_collector.record_pre_step(raw_env, leader_conditions)
                        obs, rewards, dones, extras = env.step(actions)
                        if replay_recorder is not None:
                            replay_recorder.record_reward(rewards)
                        _record_cbf_replay_state()
                    if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                    # ``extras[\"log\"]`` is cleared to scalar zero on idle steps.  Use the RSL-RL
                    # done mask as the authoritative completion source; the log is used only for the
                    # terminal reasons of those confirmed environments.
                    completed_ids = torch.nonzero(dones, as_tuple=False).reshape(-1)
                    if telemetry_recorder is not None:
                        telemetry_recorder.attach_terminal_rewards(rewards, completed_ids)
                    collector.consume(
                        extras,
                        velocity_accumulator.completed_means(completed_ids),
                        completed_env_ids=completed_ids,
                        goal_region_collision_env_ids=goal_region_collision_ids,
                    )
                    for env_id in collector.last_accepted_ids:
                        metrics = cbf_solver_terminal_metrics.pop(env_id, None)
                        if metrics is not None:
                            cbf_solver_episode_metrics.append(metrics)
                        if target is not None and env_id in terminal_coverage:
                            ray_fraction, reflection_fraction = terminal_coverage[env_id]
                            accepted_coverage[target]["ray_sum"] += ray_fraction
                            accepted_coverage[target]["reflection_sum"] += reflection_fraction
                            accepted_coverage[target]["episodes"] += 1
                    for env_id in completed_ids.detach().cpu().tolist():
                        cbf_solver_terminal_metrics.pop(int(env_id), None)
                        terminal_coverage.pop(int(env_id), None)
                    for env_id in sorted(collector.last_accepted_ids):
                        condition = leader_conditions.get(env_id)
                        if condition is None:
                            continue
                        profile = profiles[env_profile_indices[env_id]]
                        leader_records.append(
                            {
                                **condition,
                                "seed": seed,
                                "pedestrian_count": profile.pedestrian_count,
                            }
                        )
                    progress_reporter.report(
                        collector.total_episodes,
                        seed=seed,
                        seed_index=seed_index + 1,
                        status="running",
                    )
                    if structured_telemetry:
                        telemetry_recorder.resolve_terminal(completed_ids, collector.last_accepted_ids)
                    else:
                        interaction_collector.resolve_terminal(completed_ids, collector.last_accepted_success_ids)
                        leader_outcome_collector.resolve_terminal(
                            completed_ids, collector.last_accepted_success_ids, seed=seed
                        )
                    if interaction_replay_recorder is not None:
                        interaction_replay_recorder.resolve_terminal(completed_ids, collector.last_accepted_success_ids)
                    velocity_accumulator.reset(completed_ids)
                    goal_region_collision_ids.difference_update(completed_ids.detach().cpu().tolist())
                if target is not None:
                    stage_counts.append((target, subtract_counts(collector.snapshot_counts(), before_stage)))
            print(f"[INFO] Seed {seed} stage complete: {collector.total_episodes} episodes accepted.")
            progress_reporter.report(
                collector.total_episodes,
                seed=seed,
                seed_index=seed_index + 1,
                status="running",
                force=True,
            )
    finally:
        env.close()

    if telemetry_recorder is not None:
        result = telemetry_recorder.close()
        print(f"[INFO] Finalized structured telemetry dataset {result['dataset_id']}.")

    if not collector.complete:
        progress_reporter.report(
            collector.total_episodes,
            seed=None,
            seed_index=None,
            status="stopped",
            force=True,
        )
        progress_reporter.close()
        raise RuntimeError("Evaluation stopped before all benchmark profiles completed.")
    progress_reporter.report(
        collector.total_episodes,
        seed=seeds[-1],
        seed_index=seed_count,
        status="writing_artifacts",
        force=True,
    )
    write_run_output(
        args_cli=args_cli, agent_cfg=agent_cfg, output_dir=output_dir, output_root=output_root,
        checkpoint=checkpoint, profiles=profiles, seeds=seeds, seed_count=seed_count,
        required_cells=required_cells, coverage_sweep=coverage_sweep, sweep_quotas=sweep_quotas,
        coverage_action_metadata=coverage_action_metadata, step_dt_s=step_dt_s,
        episode_length_s=episode_length_s, collector=collector, leader_records=leader_records,
        leader_outcome_collector=leader_outcome_collector,
        leader_summary=_leader_condition_summary(leader_records),
        cbf_solver_summary=_cbf_solver_summary(cbf_solver_episode_metrics),
        stage_counts=stage_counts, accepted_coverage=accepted_coverage,
        failure_output_dir=failure_output_dir, replay_recorder=replay_recorder,
        interaction_collector=interaction_collector, interaction_replay_recorder=interaction_replay_recorder,
        progress_reporter=progress_reporter, rvo2_crowd_eval=RVO2_CROWD_EVAL,
        leader_outcomes_available=LEADER_OUTCOMES_AVAILABLE,
        slow_crowd_available=SLOW_CROWD_AVAILABLE,
    )
