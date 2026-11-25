"""Metrics for the linear and angular velocity estimator targets."""

from pathlib import Path

import torch


class VelocityMetrics:
    """Accumulate component-wise errors without changing the training objective."""

    def __init__(self, target_paths: list[str], target_dim: int) -> None:
        self.target_slices: dict[str, slice] = {}
        for index, path in enumerate(target_paths):
            name = Path(path).name
            if name not in {"base_lin_vel", "base_ang_vel"} or name in self.target_slices:
                raise ValueError(f"Unexpected or duplicate velocity target: {path}")
            self.target_slices[name] = slice(3 * index, 3 * (index + 1))
        if not self.target_slices or target_dim != 3 * len(self.target_slices):
            raise ValueError("Velocity targets must contain three components per target path.")

        self.totals = {name: [0.0, 0.0, 0] for name in ("total", *self.target_slices)}

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate squared and absolute errors over samples and components."""
        error = (predictions.detach() - targets.detach())
        if error.ndim != 2 or error.shape != targets.shape or error.shape[1] != 3 * len(self.target_slices):
            raise ValueError("Prediction and target shapes do not match the velocity target schema.")
        for name, component_slice in (("total", slice(None)), *self.target_slices.items()):
            values = error[:, component_slice]
            totals = self.totals[name]
            totals[0] += values.square().sum().item()
            totals[1] += values.abs().sum().item()
            totals[2] += values.numel()

    def values(self, prefix: str) -> dict[str, float]:
        """Return mean MSE, MAE, and RMSE for each target and the combined output."""
        metrics: dict[str, float] = {}
        for name, (sum_sq, sum_abs, count) in self.totals.items():
            metric_prefix = prefix if name == "total" else f"{prefix}_{name}"
            mse = sum_sq / count if count else 0.0
            metrics[f"{metric_prefix}_{'loss' if name == 'total' else 'mse'}"] = mse
            metrics[f"{metric_prefix}_mae"] = sum_abs / count if count else 0.0
            metrics[f"{metric_prefix}_rmse"] = mse**0.5
        return metrics
