"""CNN/deconvolution model for temporal-LiDAR per-bin velocity prediction."""

from __future__ import annotations

import math

import torch
from torch import nn

# Bottleneck width the encoder always reduces to (and the decoder always starts from),
# independent of fov_bins. Keeping this fixed means the fusion Linear(512, 512) layers
# below (512 = 64 channels x bottleneck width 8) never change size across fov_bins.
_BOTTLENECK_WIDTH = 8


class TemporalLidarVelocityCNN(nn.Module):
    """Map ``(distance, validity) x history x bins`` to body-XY bin velocity."""

    def __init__(self, fov_bins: int = 128) -> None:
        super().__init__()
        repeat_count = _validate_fov_bins(fov_bins)
        self.fov_bins: int = fov_bins

        encoder_layers: list[nn.Module] = [
            nn.Conv2d(2, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ELU(),
        ]
        for _ in range(repeat_count):
            encoder_layers += [nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ELU()]
        self.encoder = nn.Sequential(*encoder_layers)

        self.fusion = nn.Sequential(nn.Linear(512, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU())

        decoder_layers: list[nn.Module] = []
        for _ in range(repeat_count):
            decoder_layers += [nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1), nn.ELU()]
        decoder_layers += [
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

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
        if lidar.dim() != 4 or lidar.size(1) != 2 or lidar.size(2) != 4 or lidar.size(3) != self.fov_bins:
            raise ValueError("Expected LiDAR input with shape (B, 2, 4, fov_bins).")
        encoded = self.encoder(lidar)
        if encoded.size(1) != 64 or encoded.size(2) != 1 or encoded.size(3) != 8:
            raise RuntimeError("Unexpected encoder output shape.")
        latent = self.fusion(encoded.flatten(start_dim=1))
        decoded = self.decoder(latent.view(latent.size(0), 64, 8))
        return decoded.transpose(1, 2)


def _validate_fov_bins(fov_bins: int) -> int:
    """Return the number of repeated 64->64 stride-2 stages needed for ``fov_bins``.

    The fixed 4-stage encoder prefix always reduces width by a factor of 8
    (128 forward bins reach the encoder's fixed bottleneck width of 8 with
    exactly one extra halving stage; 512 bins need three). ``fov_bins`` must
    therefore be a power of two no smaller than 8 * _BOTTLENECK_WIDTH = 64, so
    every intermediate stage width stays an even integer down to the fixed
    bottleneck.
    """
    if fov_bins < 8 * _BOTTLENECK_WIDTH or (fov_bins & (fov_bins - 1)) != 0:
        raise ValueError(
            f"fov_bins must be a power of two >= {8 * _BOTTLENECK_WIDTH}, got {fov_bins}."
        )
    return int(math.log2(fov_bins)) - 6
