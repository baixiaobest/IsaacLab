"""Explicit Research Agent evaluation-family registry.

The control plane should pass ``evaluation_family`` from this registry rather
than infer the evaluator from a Gym task-name convention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSpec:
    family: str
    entrypoint: str
    allowed_tasks: frozenset[str] | None = None
    mining_enabled: bool = False

    def validate_task(self, task: str) -> None:
        if self.allowed_tasks is not None and task not in self.allowed_tasks:
            raise ValueError(f"Task {task!r} is not registered for the {self.family!r} evaluation family.")


EVALUATION_SPECS: dict[str, EvaluationSpec] = {
    "navigation": EvaluationSpec(
        family="navigation",
        entrypoint="scripts/reinforcement_learning/rsl_rl/evaluate.py",
        # Kept in sync with RESEARCH_AGENT_NAVIGATION_TASKS beside the Gym
        # registrations. This standalone copy lets the control-plane registry
        # be read without importing Isaac Sim/Gym at bootstrap time.
        allowed_tasks=frozenset(
            {
                "Isaac-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                "Isaac-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                "Isaac-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Pedestrian-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Pedestrian-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                "Isaac-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Pedestrian-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Mixed-Static-Pedestrian-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Mixed-Static-Pedestrian-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                (
                    "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Static-Obstacle-Cbf-"
                    "Obstacle-Avoidance-Unitree-Go2-Play-v0"
                ),
                (
                    "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Kp-Dynamic-Obstacle-Cbf-"
                    "Obstacle-Avoidance-Unitree-Go2-Play-v0"
                ),
                "Isaac-Mixed-Static-Pedestrian-Occupancy-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Mixed-Static-Pedestrian-Occupancy-Obstacle-Avoidance-Unitree-Go2-Play-v0",
                "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-v0",
                "Isaac-Mixed-Static-Pedestrian-Temporal-Lidar-Prediction-Obstacle-Avoidance-Unitree-Go2-Play-v0",
            }
        ),
        mining_enabled=False,
    ),
    "locomotion": EvaluationSpec(
        family="locomotion",
        entrypoint="scripts/reinforcement_learning/rsl_rl/evaluate_locomotion.py",
        allowed_tasks=frozenset({"Isaac-Locomotion-Vel-Unitree-Go2-Robust-v1"}),
        mining_enabled=False,
    ),
}


def resolve_evaluation_spec(family: str, task: str) -> EvaluationSpec:
    """Return and validate the explicit evaluator selected by a stage."""
    try:
        spec = EVALUATION_SPECS[family]
    except KeyError as error:
        raise ValueError(f"Unknown evaluation family {family!r}; choose one of {sorted(EVALUATION_SPECS)}.") from error
    spec.validate_task(task)
    return spec
