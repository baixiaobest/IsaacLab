"""Tests for linear and angular estimator metrics."""

import torch

from scripts.reinforcement_learning.rsl_rl.velocity_estimator.src.metrics import VelocityMetrics


def test_metrics_follow_target_paths_and_average_all_components():
    accumulator = VelocityMetrics(["ground_truth/base_ang_vel", "ground_truth/base_lin_vel"], 6)
    target = torch.zeros((2, 6))
    prediction = torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    accumulator.update(prediction[:1], target[:1])
    accumulator.update(prediction[1:], target[1:])

    values = accumulator.values("validation")
    assert values["validation_loss"] == 2.5
    assert values["validation_mae"] == 1.5
    assert values["validation_base_ang_vel_mse"] == 1.0
    assert values["validation_base_ang_vel_rmse"] == 1.0
    assert values["validation_base_lin_vel_mse"] == 4.0
    assert values["validation_base_lin_vel_mae"] == 2.0
    assert values["validation_base_lin_vel_rmse"] == 2.0


def test_metrics_use_checkpoint_order_not_fixed_slices():
    accumulator = VelocityMetrics(["ground_truth/base_lin_vel", "ground_truth/base_ang_vel"], 6)
    accumulator.update(torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0, 1.0]]), torch.zeros((1, 6)))
    values = accumulator.values("train")
    assert values["train_base_lin_vel_mse"] == 4.0
    assert values["train_base_ang_vel_mse"] == 1.0
