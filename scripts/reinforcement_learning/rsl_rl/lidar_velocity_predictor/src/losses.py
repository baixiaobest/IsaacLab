"""Losses and metrics for masked LiDAR point velocity regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distance_weight(range_m: torch.Tensor) -> torch.Tensor:
    """Keep all reflections through 5 m equally weighted, then decay smoothly."""
    far = 0.25 + 0.75 * torch.exp(-(range_m - 5.0).clamp_min(0.0) / 5.0)
    return torch.where(range_m <= 5.0, torch.ones_like(range_m), far)


def masked_class_balanced_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reflection_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    range_m: torch.Tensor,
    static_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a class-balanced masked Smooth-L1 loss with configurable class weights."""
    if not 0.0 <= static_loss_weight <= 1.0:
        raise ValueError("static_loss_weight must be in [0, 1].")
    totals = masked_class_balanced_huber_totals(prediction, target, reflection_mask, dynamic_mask, range_m)
    static_weight = totals["static_weight"]
    dynamic_weight = totals["dynamic_weight"]
    static_loss = totals["static_numerator"] / static_weight.clamp_min(1.0)
    dynamic_loss = totals["dynamic_numerator"] / dynamic_weight.clamp_min(1.0)
    dynamic_loss_weight = 1.0 - static_loss_weight
    static_active = (static_weight > 0).to(torch.float32)
    dynamic_active = (dynamic_weight > 0).to(torch.float32)
    active_weight = static_loss_weight * static_active + dynamic_loss_weight * dynamic_active
    total = (
        static_loss_weight * static_loss * static_active + dynamic_loss_weight * dynamic_loss * dynamic_active
    ) / active_weight.clamp_min(1.0)
    return total, {
        "static_loss": static_loss.detach(),
        "dynamic_loss": dynamic_loss.detach(),
        "static_weight": static_weight.detach(),
        "dynamic_weight": dynamic_weight.detach(),
    }


def masked_class_balanced_huber_totals(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reflection_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    range_m: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return unnormalized class terms for exact aggregation across batches."""
    point_loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    weights = distance_weight(range_m)
    static_mask = reflection_mask & ~dynamic_mask
    return {
        "static_numerator": (weights * point_loss * static_mask).sum(),
        "static_weight": (weights * static_mask).sum(),
        "dynamic_numerator": (weights * point_loss * dynamic_mask).sum(),
        "dynamic_weight": (weights * dynamic_mask).sum(),
    }


@torch.no_grad()
def masked_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reflection_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    range_m: torch.Tensor,
) -> dict[str, tuple[float, int]]:
    """Return squared/absolute error totals for all requested evaluation subsets."""
    error = prediction - target
    squared = error.square().sum(dim=-1)
    absolute = error.abs().sum(dim=-1)
    subsets = {
        "all": reflection_mask,
        "static": reflection_mask & ~dynamic_mask,
        "dynamic": dynamic_mask,
        "within_5m": reflection_mask & (range_m <= 5.0),
        "within_2m": reflection_mask & (range_m <= 2.0),
        "dynamic_within_5m": dynamic_mask & (range_m <= 5.0),
        "dynamic_within_2m": dynamic_mask & (range_m <= 2.0),
    }
    return {
        name: (float(squared[mask].sum().item()), int(mask.sum().item()))
        for name, mask in subsets.items()
    } | {
        f"{name}_abs": (float(absolute[mask].sum().item()), int(mask.sum().item()))
        for name, mask in subsets.items()
    } | {
        f"zero_{name}": (float(target.square().sum(dim=-1)[mask].sum().item()), int(mask.sum().item()))
        for name, mask in subsets.items()
    } | {
        "static_false_motion": (
            float(torch.linalg.vector_norm(prediction, dim=-1)[subsets["static"]].sum().item()),
            int(subsets["static"].sum().item()),
        )
    } | _dynamic_heading_metric(prediction, target, dynamic_mask)


@torch.no_grad()
def _dynamic_heading_metric(
    prediction: torch.Tensor, target: torch.Tensor, dynamic_mask: torch.Tensor
) -> dict[str, tuple[float, int]]:
    target_speed = torch.linalg.vector_norm(target, dim=-1)
    mask = dynamic_mask & (target_speed > 1.0e-4)
    if not torch.any(mask):
        return {"dynamic_heading_error": (0.0, 0)}
    prediction_angle = torch.atan2(prediction[..., 1], prediction[..., 0])
    target_angle = torch.atan2(target[..., 1], target[..., 0])
    error = torch.atan2(torch.sin(prediction_angle - target_angle), torch.cos(prediction_angle - target_angle)).abs()
    return {"dynamic_heading_error": (float(error[mask].sum().item()), int(mask.sum().item()))}
