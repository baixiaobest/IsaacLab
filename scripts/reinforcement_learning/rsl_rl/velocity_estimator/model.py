# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    """Create an activation module by name."""
    activation_name = name.lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class VelocityEstimator(nn.Module):
    """MLP estimator that consumes a fixed horizon of flattened observations."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        activation: str = "elu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero.")
        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero.")

        flattened_input_dim = input_dim * horizon
        layers: list[nn.Module] = []
        previous_dim = flattened_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))

        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Estimate current velocity from a horizon of past observations."""
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs with shape (batch, horizon, input_dim), got {tuple(inputs.shape)}")
        if inputs.shape[1] != self.horizon:
            raise ValueError(f"Expected horizon={self.horizon}, got {inputs.shape[1]}")
        if inputs.shape[2] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {inputs.shape[2]}")

        flattened = inputs.reshape(inputs.shape[0], self.horizon * self.input_dim)
        return self.network(flattened)