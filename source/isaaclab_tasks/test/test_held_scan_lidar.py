"""Unit tests for the held full-scan and sparse lidar collector."""

import math
from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.navigation.lidar_geometry import (
    body_to_world_xy,
    forward_lidar_reflection_bins,
    world_to_body_xy,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.held_scan_lidar_env import (
    LIDAR_COVERAGE_STAGES,
    LIDAR_CURRICULUM_CHECKPOINT_KEY,
    HeldScanLidarCfg,
    HeldScanLidarCollector,
    attach_lidar_density_checkpoint_saving,
    goal_reached_lidar_density_curriculum,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.lidar_velocity_data_env import (
    FixedCoveragePedestrianCrowdNavigationEnv,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.lidar_velocity_data_env_cfg import (
    MixedTemporalLidarKpPointVelocityDataEnvCfg,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.kp_mixed_scenario_env_cfg import (
    MixedTemporalLidarKpDynamicObstacleCbfObstacleAvoidanceEnvCfg_PLAY,
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg,
    MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.mixed_scenario_mixins import (
    MixedObstacleAvoidanceEnvCfg,
    MixedTemporalLidarObstacleAvoidanceEnvCfg,
    MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY,
    MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg,
    MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg_PLAY,
    configure_static_dynamic_evaluation,
)
from isaaclab_tasks.manager_based.navigation.config.go2.obstacle_avoidance.temporal_lidar_env_cfg import (
    TEMPORAL_LIDAR_COLLECTOR_NAME,
    TEMPORAL_LIDAR_HISTORY_KEY,
    TEMPORAL_LIDAR_POS_NOISE_STD,
    TEMPORAL_LIDAR_RAYS,
    TemporalLidarObservationsCfg,
    TemporalLidarObstacleAvoidanceEnvCfg,
)


def test_full_scan_clock_fires_every_twenty_six_physics_steps() -> None:
    """A 130 ms full scan is captured every 26 physics steps on the 5 ms grid."""
    collector = object.__new__(HeldScanLidarCollector)
    collector.env = SimpleNamespace(physics_dt=0.005)
    collector._physics_steps = 0
    collector._time_s = 0.0
    collector._scan_steps = 26
    captures = []
    collector._capture_full_scan = lambda: captures.append(collector._physics_steps)

    for _ in range(78):
        collector.on_physics_step()

    assert captures == [26, 52, 78]
    assert abs(collector._time_s - 0.390) < 1.0e-12


def test_consume_returns_full_scan_once_then_holds_it() -> None:
    """Collector output preserves all source rays and is consumed only once."""
    collector = object.__new__(HeldScanLidarCollector)
    collector.num_envs = 2
    collector.device = "cpu"
    collector.cfg = SimpleNamespace(scan_period_s=0.130)
    collector._time_s = 0.130
    collector._pending_valid = torch.tensor([True, False])
    collector._has_latest = torch.tensor([False, False])
    collector._latest_reference_time_s = torch.zeros(2)
    collector._pending_reference_time_s = torch.tensor([0.130, 0.0])
    collector._pending_hit_xy = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]],
        ]
    )
    collector._pending_state = torch.tensor([[2, 1, 2, 1], [1, 1, 1, 1]], dtype=torch.uint8)
    collector._pending_ego_xy = torch.zeros(2, 2)
    collector._pending_ego_yaw = torch.zeros(2)

    completed = collector.consume_completed()

    assert completed is not None
    assert completed["hit_xy"].shape == (1, 4, 2)
    assert torch.equal(completed["hit_xy"][0], collector._pending_hit_xy[0])
    assert torch.equal(completed["ray_state"], torch.tensor([[2, 1, 2, 1]], dtype=torch.uint8))
    assert collector.consume_completed() is None


def test_scan_age_grows_while_a_full_scan_is_held() -> None:
    collector = object.__new__(HeldScanLidarCollector)
    collector.num_envs = 2
    collector.device = "cpu"
    collector.cfg = SimpleNamespace(scan_period_s=0.130)
    collector._time_s = 0.210
    collector._has_latest = torch.tensor([True, False])
    collector._latest_reference_time_s = torch.tensor([0.130, 0.0])

    age = collector.scan_age_s()

    assert torch.allclose(age, torch.tensor([0.080, 0.130]))


def test_reset_queues_an_immediate_scan_for_only_reset_environments() -> None:
    """Reset environments get a valid scan before their next policy action."""
    collector = object.__new__(HeldScanLidarCollector)
    collector.num_envs = 3
    collector.device = "cpu"
    collector._time_s = 0.210
    collector._pending_valid = torch.tensor([True, True, True])
    collector._has_latest = torch.tensor([True, True, True])
    collector._latest_reference_time_s = torch.zeros(3)
    captured = []
    collector._capture_full_scan = lambda env_ids: captured.append(env_ids.clone())

    collector.reset(torch.tensor([1, 2]))

    assert torch.equal(collector._pending_valid, torch.tensor([True, False, False]))
    assert torch.equal(collector._has_latest, torch.tensor([True, False, False]))
    assert torch.equal(collector._latest_reference_time_s, torch.tensor([0.0, 0.210, 0.210]))
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([1, 2]))


def test_rebinning_and_collector_noise_are_absent() -> None:
    """The held-scan model intentionally contains timing only."""
    assert not hasattr(HeldScanLidarCollector, "_rebin_to_policy")
    assert not hasattr(HeldScanLidarCollector, "_apply_simple_scan_noise")


def test_body_world_velocity_rotation_is_invertible() -> None:
    velocity_w = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    yaw = torch.tensor([torch.pi / 2.0])

    velocity_b = world_to_body_xy(velocity_w, yaw)

    assert torch.allclose(velocity_b, torch.tensor([[[0.0, -1.0], [1.0, 0.0]]]), atol=1.0e-6)
    assert torch.allclose(body_to_world_xy(velocity_b, yaw), velocity_w, atol=1.0e-6)


def test_forward_bins_keep_only_nearest_valid_reflection() -> None:
    # Both first rays lie in the forward bin at yaw zero; the 1 m return owns it.
    capture = {
        "hit_xy": torch.tensor([[[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]]),
        "ray_state": torch.tensor([[2, 2, 1]], dtype=torch.uint8),
        "ego_xy": torch.zeros(1, 2),
        "ego_yaw": torch.zeros(1),
    }
    binned = forward_lidar_reflection_bins(capture, num_bins=256, fov_bins=128)
    forward = 64

    assert binned["reflection_mask"][0, forward]
    assert binned["range_m"][0, forward] == 1.0
    assert binned["winner_ray"][0, forward] == 1


def test_actor_and_critic_share_held_history_and_scan_age() -> None:
    """Only actor corruption may differ; lidar timing and layout must match."""
    policy = TemporalLidarObservationsCfg.PolicyCfg()
    critic = TemporalLidarObservationsCfg.CriticCfg()

    assert policy.scan_age.params["collector_name"] == TEMPORAL_LIDAR_COLLECTOR_NAME
    assert critic.scan_age.params == policy.scan_age.params
    assert policy.obstacle_scan.params["history_key"] == TEMPORAL_LIDAR_HISTORY_KEY
    assert critic.obstacle_scan.params["history_key"] == TEMPORAL_LIDAR_HISTORY_KEY
    assert policy.obstacle_scan.params["history_num_rays"] == TEMPORAL_LIDAR_RAYS
    assert critic.obstacle_scan.params["history_num_rays"] == TEMPORAL_LIDAR_RAYS
    assert policy.obstacle_scan.params["pos_noise_std"] == TEMPORAL_LIDAR_POS_NOISE_STD
    assert critic.obstacle_scan.params["pos_noise_std"] == 0.0
    assert policy.obstacle_scan.noise.n_min == -0.05
    assert policy.obstacle_scan.noise.n_max == 0.05
    assert critic.obstacle_scan.noise is None


def _make_sparse_collector(num_envs: int = 4, curriculum: bool = False) -> HeldScanLidarCollector:
    angles = torch.linspace(-math.pi / 2, math.pi / 2, 256)
    directions = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    directions = directions.unsqueeze(0).expand(num_envs, -1, -1).clone()
    positions = torch.zeros(num_envs, 3)
    data = SimpleNamespace(
        pos_w=positions,
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, -1),
        ray_hits_w=positions[:, None, :] + directions * 5.0,
        ray_mesh_ids=torch.zeros(num_envs, 256, 1, dtype=torch.int16),
    )
    sensor = SimpleNamespace(data=data, _ray_directions_w=directions)
    env = SimpleNamespace(
        physics_dt=0.005,
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(sensors={"obstacle_scanner": sensor}),
    )
    return HeldScanLidarCollector(
        env,
        HeldScanLidarCfg(
            sparse_sampling_enabled=True,
            density_curriculum_enabled=curriculum,
            target_coverage=1.0 if curriculum else None,
        ),
    )


def test_sparse_templates_are_independent_with_randomized_wall_occupancy() -> None:
    collector = _make_sparse_collector(256)
    patterns = collector._sampling_pattern
    assert torch.unique(patterns, dim=0).shape[0] > 240
    assert torch.unique(collector._sampling_phase).numel() > 100
    assert 0.29 < patterns.float().mean().item() < 0.38
    # The drawn groups contain both neighboring reflections and several gap lengths.
    pair_count = (patterns[:, :-1] & patterns[:, 1:]).sum().item()
    assert pair_count > 1000
    assert len({int(gap) for gap in torch.diff(patterns[0].nonzero().flatten()) if gap > 1}) >= 3
    policy_state = collector.latest_policy_capture()["ray_state"]
    assert set(policy_state.unique().tolist()) == {0, 2}
    # One of two physical rays is chosen for each populated angular cell.
    assert 0.14 < (policy_state == 2).float().mean().item() < 0.19
    reflected_bins = forward_lidar_reflection_bins(collector.latest_policy_capture())["reflection_mask"]
    assert 0.29 < reflected_bins.float().mean().item() < 0.38


def test_sparse_phase_enters_from_right_edge_and_wraps_full_circle(monkeypatch) -> None:
    collector = _make_sparse_collector(1)
    collector._sampling_pattern[:] = False
    collector._sampling_pattern[0, 63] = True  # just outside the right edge
    collector._sampling_phase[:] = 0
    collector._sampling_has_capture[:] = True
    monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: torch.full(shape, 0.5, device=kwargs["device"]))

    collector._capture_full_scan()
    assert collector._sampling_phase.item() == 1
    assert collector.latest_policy_capture()["ray_state"][0, :2].max().item() == 2
    assert collector.latest_policy_capture()["ray_state"][0, -2:].max().item() == 0

    collector._sampling_pattern[:] = False
    collector._sampling_pattern[0, 64] = True
    collector._sampling_phase[:] = 255
    collector._capture_full_scan()
    assert collector._sampling_phase.item() == 0
    assert collector.latest_policy_capture()["ray_state"][0, :2].max().item() == 2


def test_sparse_phase_steps_vary_only_on_completed_captures() -> None:
    collector = _make_sparse_collector(256)
    observed_steps = []
    for _ in range(12):
        prior = collector._sampling_phase.clone()
        collector._capture_full_scan()
        observed_steps.append((collector._sampling_phase - prior) % 256)
    steps = torch.cat(observed_steps)
    assert set(steps.unique().tolist()) == {0, 1, 2}
    assert 0.07 < (steps == 0).float().mean().item() < 0.13
    assert 0.71 < (steps == 1).float().mean().item() < 0.79
    assert 0.12 < (steps == 2).float().mean().item() < 0.18


def test_sparse_capture_holds_and_reset_is_isolated() -> None:
    collector = _make_sparse_collector(2)
    before_phase = collector._sampling_phase.clone()
    before_state = collector.latest_policy_capture()["ray_state"].clone()
    before_index = collector.latest_policy_capture()["capture_index"].clone()
    for _ in range(25):
        collector.on_physics_step()
    assert torch.equal(collector._sampling_phase, before_phase)
    assert torch.equal(collector.latest_policy_capture()["ray_state"], before_state)
    collector.on_physics_step()
    assert torch.equal(collector.latest_policy_capture()["capture_index"], before_index + 1)

    unaffected_pattern = collector._sampling_pattern[0].clone()
    unaffected_phase = collector._sampling_phase[0].clone()
    unaffected_index = collector._capture_index[0].clone()
    collector.reset(torch.tensor([1]))
    assert torch.equal(collector._sampling_pattern[0], unaffected_pattern)
    assert torch.equal(collector._sampling_phase[0], unaffected_phase)
    assert torch.equal(collector._capture_index[0], unaffected_index)
    assert collector._sampling_has_capture[1]


def test_sparse_policy_has_max_range_invalids_while_cbf_keeps_full_geometry() -> None:
    collector = _make_sparse_collector(2)
    full = collector.latest_capture()
    sparse = collector.latest_policy_capture()
    missing = sparse["ray_state"] == 0
    assert torch.all(full["ray_state"] == 2)
    assert missing.any()
    sparse_ranges = torch.linalg.vector_norm(sparse["hit_xy"][missing], dim=-1)
    full_ranges = torch.linalg.vector_norm(full["hit_xy"][missing], dim=-1)
    assert torch.allclose(sparse_ranges, torch.full_like(sparse_ranges, 20.0))
    assert torch.allclose(full_ranges, torch.full_like(full_ranges, 5.0))
    assert torch.equal(full["capture_index"], sparse["capture_index"])
    assert full["ray_mesh_ids"] is sparse["ray_mesh_ids"]
    completed = collector.consume_completed()
    assert completed is not None
    assert torch.equal(completed["ray_state"], sparse["ray_state"])
    assert torch.equal(completed["hit_xy"], sparse["hit_xy"])
    assert collector.consume_completed() is None

    collector.env.scene.sensors["obstacle_scanner"].data.ray_hits_w[:] = float("inf")
    collector._capture_full_scan()
    assert not collector.latest_policy_capture()["ray_state"].any()
    assert torch.all(collector.latest_capture()["ray_state"] == 1)


def test_only_mixed_temporal_configs_enable_sparse_sampling() -> None:
    assert not HeldScanLidarCfg().sparse_sampling_enabled
    assert not hasattr(MixedObstacleAvoidanceEnvCfg(), "held_scan_lidar")
    assert not TemporalLidarObstacleAvoidanceEnvCfg().held_scan_lidar.sparse_sampling_enabled
    assert MixedTemporalLidarObstacleAvoidanceEnvCfg().held_scan_lidar.sparse_sampling_enabled
    assert MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg().held_scan_lidar.sparse_sampling_enabled
    for cfg_type in (
        MixedTemporalLidarObstacleAvoidanceEnvCfg_PLAY,
        MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg_PLAY,
        MixedTemporalLidarKpObstacleAvoidanceEnvCfg,
        MixedTemporalLidarKpObstacleAvoidanceEnvCfg_PLAY,
        MixedTemporalLidarKpDynamicObstacleCbfObstacleAvoidanceEnvCfg_PLAY,
        MixedTemporalLidarKpPointVelocityDataEnvCfg,
    ):
        cfg = cfg_type()
        assert cfg.held_scan_lidar.sparse_sampling_enabled
        assert not cfg.held_scan_lidar.density_curriculum_enabled
        assert cfg.held_scan_lidar.target_coverage is None
        assert cfg.curriculum.lidar_density is None
    for cfg_type in (MixedTemporalLidarObstacleAvoidanceEnvCfg, MixedTemporalLidarPredictionObstacleAvoidanceEnvCfg):
        cfg = cfg_type()
        assert cfg.held_scan_lidar.density_curriculum_enabled
        assert cfg.held_scan_lidar.target_coverage == 1.0
        assert cfg.curriculum.lidar_density is not None
    eval_cfg = configure_static_dynamic_evaluation(MixedTemporalLidarObstacleAvoidanceEnvCfg())
    assert eval_cfg.curriculum.lidar_density is None
    assert not eval_cfg.held_scan_lidar.density_curriculum_enabled
    assert eval_cfg.held_scan_lidar.target_coverage is None


def test_velocity_labels_use_sparse_capture_with_aligned_metadata() -> None:
    collector = _make_sparse_collector(2)
    collector._pending_ray_mesh_ids[:] = 1
    collector._pending_ped_velocity_w = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    data_env = object.__new__(FixedCoveragePedestrianCrowdNavigationEnv)
    data_env._held_scan_lidar_collector = collector
    data_env.crowd_manager = SimpleNamespace(max_pedestrians=1)

    labels = data_env.get_point_velocity_labels()
    sparse_bins = forward_lidar_reflection_bins(collector.latest_policy_capture())
    full_bins = forward_lidar_reflection_bins(collector.latest_capture())
    assert torch.equal(labels["reflection_mask"], sparse_bins["reflection_mask"])
    assert torch.equal(labels["dynamic_mask"], sparse_bins["reflection_mask"])
    assert torch.equal(labels["capture_index"], collector.latest_policy_capture()["capture_index"])
    assert torch.all(labels["point_velocity_b"][~labels["reflection_mask"]] == 0)
    assert full_bins["reflection_mask"].sum() > labels["reflection_mask"].sum()


def test_density_filling_keeps_capture_only_shift_and_full_cbf_geometry() -> None:
    collector = _make_sparse_collector(64, curriculum=True)
    assert collector._sampling_pattern.all()
    assert torch.all(collector.latest_policy_capture()["ray_state"].sum(dim=1) == 256)
    assert torch.all(forward_lidar_reflection_bins(collector.latest_policy_capture())["reflection_mask"])
    assert torch.all(collector.latest_capture()["ray_state"] == 2)
    for target in LIDAR_COVERAGE_STAGES[:-1]:
        collector._coverage_target = target
        collector.reset()
        assert torch.all(collector._sampling_pattern.sum(dim=1) == round(256 * target))
    collector._coverage_target = 0.6
    collector.reset()
    pattern = collector._sampling_pattern.clone()
    phase = collector._sampling_phase.clone()
    for _ in range(25):
        collector.on_physics_step()
    assert torch.equal(pattern, collector._sampling_pattern)
    assert torch.equal(phase, collector._sampling_phase)
    collector.on_physics_step()
    assert torch.equal(pattern, collector._sampling_pattern)
    reflected = forward_lidar_reflection_bins(collector.latest_policy_capture())["reflection_mask"]
    assert 0.52 < reflected.float().mean().item() < 0.68
    assert torch.all(collector.latest_capture()["ray_state"] == 2)
    collector._coverage_target = None
    collector.reset()
    assert 0.29 < collector._sampling_pattern.float().mean().item() < 0.38


def test_density_curriculum_window_threshold_checks_and_old_episode_exclusion() -> None:
    assert LIDAR_COVERAGE_STAGES == (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, None)
    collector = _make_sparse_collector(500, curriculum=True)
    ids = torch.arange(500)
    collector.record_density_outcomes(ids[:499], torch.ones(499, dtype=torch.bool))
    assert collector._density_stage == 0
    collector.record_density_outcomes(ids[499:], torch.ones(1, dtype=torch.bool))
    assert collector._density_stage == 1
    assert collector._coverage_target == 0.9
    assert collector._density_completed == 0
    # The other episodes in flight at stage zero cannot feed the new stage's window.
    collector.record_density_outcomes(ids, torch.ones(500, dtype=torch.bool))
    assert collector._density_completed == 0
    collector.reset(ids)
    collector.record_density_outcomes(ids[:500], torch.zeros(500, dtype=torch.bool))
    assert collector._density_stage == 1
    assert collector._density_next_check == 600
    collector.record_density_outcomes(ids[:99], torch.ones(99, dtype=torch.bool))
    assert collector._density_stage == 1
    collector.record_density_outcomes(ids[:1], torch.ones(1, dtype=torch.bool))
    assert collector._density_stage == 1  # only 100/500 recent episodes succeeded
    assert collector._density_next_check == 700

    boundary = _make_sparse_collector(500, curriculum=True)
    outcomes = torch.cat((torch.ones(350, dtype=torch.bool), torch.zeros(150, dtype=torch.bool)))
    boundary.record_density_outcomes(ids, outcomes)
    assert boundary._density_stage == 1  # exactly 70% passes
    assert LIDAR_COVERAGE_STAGES[-1] is None
    boundary._density_stage = len(LIDAR_COVERAGE_STAGES) - 1
    boundary._coverage_target = None
    boundary.reset(ids)
    boundary.record_density_outcomes(ids, torch.ones(500, dtype=torch.bool))
    assert boundary._density_stage == len(LIDAR_COVERAGE_STAGES) - 1
    assert boundary._coverage_target is None


def test_density_curriculum_uses_goal_reached_before_reset_and_ignores_initial_resets() -> None:
    collector = _make_sparse_collector(3, curriculum=True)
    collector.env._held_scan_lidar_collector = collector
    collector.env.episode_length_buf = torch.tensor([20, 0, 5])
    collector.env.termination_manager = SimpleNamespace(
        get_term=lambda name: torch.tensor([True, True, False])
    )
    goal_reached_lidar_density_curriculum(collector.env, torch.tensor([0, 1, 2]))
    assert list(collector._density_outcomes) == [True, False]
    assert collector._density_completed == 2


def test_density_checkpoint_infos_restore_and_legacy_fallback() -> None:
    collector = _make_sparse_collector(4, curriculum=True)
    collector.env._held_scan_lidar_collector = collector
    collector._density_stage = 3
    collector._coverage_target = LIDAR_COVERAGE_STAGES[3]
    collector._density_completed = 504
    collector._density_next_check = 600
    collector._density_outcomes.extend([True] * 400 + [False] * 100)
    runner = SimpleNamespace(save=lambda path, infos=None: saved.append((path, infos)))
    saved = []
    assert attach_lidar_density_checkpoint_saving(runner, collector.env) is collector
    runner.save("model_10.pt", infos={"existing": 1})
    infos = saved[0][1]
    assert infos["existing"] == 1
    state = infos[LIDAR_CURRICULUM_CHECKPOINT_KEY]
    restored = _make_sparse_collector(4, curriculum=True)
    restored.restore_density_checkpoint_state(state)
    assert restored.density_checkpoint_state() == state
    assert torch.all(restored._episode_density_stage == 3)
    assert torch.all(restored._sampling_pattern.sum(dim=1) == round(256 * 0.7))
    restored.restore_density_checkpoint_state(None)
    assert restored._density_stage == 0
    assert restored._density_completed == 0
    assert restored._sampling_pattern.all()


def test_density_curriculum_exposes_progress_through_curriculum_term() -> None:
    cfg = MixedTemporalLidarObstacleAvoidanceEnvCfg()
    assert cfg.curriculum.lidar_density.func is goal_reached_lidar_density_curriculum
    collector = _make_sparse_collector(4, curriculum=True)
    collector.env._held_scan_lidar_collector = collector
    collector.env.episode_length_buf = torch.tensor([20, 0, 5, 8])
    collector.env.termination_manager = SimpleNamespace(
        get_term=lambda name: torch.tensor([True, True, False, True])
    )

    logged = cfg.curriculum.lidar_density.func(collector.env, torch.arange(4))

    assert logged["coverage_percent"] == 100.0
    assert logged["stage"] == 0.0
    assert logged["rolling_goal_percent"] == 200.0 / 3.0
    assert logged["goal_threshold_percent"] == 70.0
    assert logged["rolling_goal_count"] == 2.0
    assert logged["rolling_episode_count"] == 3.0
    assert logged["stage_completed_episodes"] == 3.0
    assert logged["episodes_until_check"] == 497.0
    collector._density_stage = len(LIDAR_COVERAGE_STAGES) - 1
    collector._coverage_target = None
    assert round(collector.density_status()["coverage_percent"], 1) == 33.3
    assert collector.density_status()["episodes_until_check"] == 0.0
