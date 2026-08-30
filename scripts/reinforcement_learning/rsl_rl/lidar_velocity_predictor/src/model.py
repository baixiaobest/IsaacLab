"""CNN/deconvolution model for temporal-LiDAR per-bin velocity prediction."""

from __future__ import annotations

import torch
from torch import nn


class TemporalLidarVelocityCNN(nn.Module):
    """Map ``(distance, validity) x history x bins`` to body-XY bin velocity."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ELU(),
        )
        self.fusion = nn.Sequential(nn.Linear(512, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        # Keep validation TorchScript-compatible: tuple formatting of dynamic
        # shapes cannot be compiled by ``torch.jit.script``.
        if lidar.dim() != 4 or lidar.size(1) != 2 or lidar.size(2) != 4 or lidar.size(3) != 128:
            raise ValueError("Expected LiDAR input with shape (B, 2, 4, 128).")
        encoded = self.encoder(lidar)
        if encoded.size(1) != 64 or encoded.size(2) != 1 or encoded.size(3) != 8:
            raise RuntimeError("Unexpected encoder output shape.")
        latent = self.fusion(encoded.flatten(start_dim=1))
        decoded = self.decoder(latent.view(latent.size(0), 64, 8))
        return decoded.transpose(1, 2)
