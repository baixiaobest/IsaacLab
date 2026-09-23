"""Pure tests for the Research Agent robust-locomotion evaluator contract."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from locomotion_evaluation import (  # noqa: E402
    NOMINAL_CONDITION,
    RESAMPLING_STRESS_CONDITION,
    evaluation_profiles,
    evaluate_gates,
    target_for_profile,
)
from research_agent_evaluation import resolve_evaluation_spec  # noqa: E402


def _row(profile, *, fall: bool = False, tracking: float = 0.1, quality: float = 0.1):
    return {
        "profile": profile.name,
        "trajectory": profile.family,
        "condition": profile.name.split(":", 1)[1],
        "fall": fall,
        "p95_vx_error": tracking,
        "p95_vy_error": tracking,
        "p95_wz_error": tracking,
        "p95_tracking_error": tracking,
        "p99_tilt_rad": quality,
        "p95_action_rate": quality,
        "torque_saturation_fraction": quality,
    }


def test_profile_matrix_and_sign_mirroring() -> None:
    profiles = evaluation_profiles()
    assert len(profiles) == 24
    assert sum(profile.name.endswith(NOMINAL_CONDITION) for profile in profiles) == 12
    assert sum(profile.name.endswith(RESAMPLING_STRESS_CONDITION) for profile in profiles) == 12
    assert sum(profile.resampling_stress for profile in profiles) == 12
    lateral = next(profile for profile in profiles if profile.family == "lateral_tracking")
    assert target_for_profile(lateral, 2.0, 0)[1] == -target_for_profile(lateral, 2.0, 1)[1]
    obstacle_stop = next(profile for profile in profiles if profile.family == "obstacle_stop")
    assert target_for_profile(obstacle_stop, 3.9, 0)[0] > 0.0
    assert target_for_profile(obstacle_stop, 4.1, 0).sum() == 0.0
    sequence_nominal = next(
        profile for profile in profiles if profile.family == "navigation_rate_sequence" and not profile.resampling_stress
    )
    sequence_stress = next(
        profile for profile in profiles if profile.family == "navigation_rate_sequence" and profile.resampling_stress
    )
    assert (target_for_profile(sequence_nominal, 2.1, 0) == target_for_profile(sequence_nominal, 1.1, 0)).all()
    assert not (target_for_profile(sequence_stress, 2.1, 0) == target_for_profile(sequence_stress, 1.1, 0)).all()


def test_explicit_evaluation_family_rejects_navigation_task() -> None:
    assert resolve_evaluation_spec("locomotion", "Isaac-Locomotion-Vel-Unitree-Go2-Robust-v1").family == "locomotion"
    assert (
        resolve_evaluation_spec(
            "navigation", "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-Play-v0"
        ).family
        == "navigation"
    )
    try:
        resolve_evaluation_spec("locomotion", "Isaac-Obstacle-Avoidance-Unitree-Go2-v0")
    except ValueError:
        pass
    else:
        raise AssertionError("Locomotion evaluator must reject navigation task IDs.")


def test_promotion_gates_require_safety_and_confident_coupled_improvement() -> None:
    profiles = evaluation_profiles()
    candidate = []
    baseline = []
    for profile in profiles:
        for episode in range(100):
            # Baseline falls in 20% of coupled stress trials, candidate in 0%.
            coupled_stress = not profile.name.endswith(NOMINAL_CONDITION) and "turn" in profile.family
            baseline.append(_row(profile, fall=coupled_stress and episode < 20))
            candidate.append(_row(profile, fall=False))
    result = evaluate_gates(candidate, baseline, episodes_per_profile=100)
    assert result["eligible_for_hardware_validation"]

    candidate[0]["fall"] = True
    assert not evaluate_gates(candidate, baseline, episodes_per_profile=100)["eligible_for_hardware_validation"]
    assert not evaluate_gates(candidate, baseline, episodes_per_profile=10)["gates"]["sufficient_samples"]


def test_standalone_gates_are_absolute_only() -> None:
    candidate = [_row(profile) for profile in evaluation_profiles() for _ in range(100)]
    result = evaluate_gates(candidate, None, episodes_per_profile=100)
    assert result["hardware_eligible"]
    assert not result["comparison_available"]
    assert result["comparative_gates"]["coupled_turn_improvement"] == "not_assessed"
