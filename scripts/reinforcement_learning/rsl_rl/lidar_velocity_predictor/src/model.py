"""CNN/deconvolution model for temporal-LiDAR per-bin velocity prediction."""

from __future__ import annotations

import torch
from torch import nn


class TemporalLidarVelocityCNN(nn.Module):
    """Map ``(distance, validity) x history x bins`` to body-XY bin velocity."""

    def __init__(self, num_frames: int = 4, aux_classifier: bool = False, dynamic_threshold: float = 0.3) -> None:
        super().__init__()
        if num_frames < 1:
            raise ValueError("num_frames must be positive.")
        if not 0.0 < dynamic_threshold < 1.0:
            raise ValueError("dynamic_threshold must be between zero and one.")
        self.num_frames = num_frames
        self.aux_classifier = aux_classifier
        self.dynamic_threshold = dynamic_threshold
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
        encoded_frames = (num_frames + 1) // 2
        encoded_frames = (encoded_frames + 1) // 2
        self.fusion = nn.Sequential(nn.Linear(64 * encoded_frames * 8, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1),
        )
        self.classifier_head = nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1) if aux_classifier else None
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _decoded_features(self, lidar: torch.Tensor) -> torch.Tensor:
        # Keep validation TorchScript-compatible: tuple formatting of dynamic
        # shapes cannot be compiled by ``torch.jit.script``.
        if lidar.dim() != 4 or lidar.size(1) != 2 or lidar.size(2) != self.num_frames or lidar.size(3) != 128:
            raise ValueError("Expected LiDAR input with shape (B, 2, num_frames, 128).")
        encoded = self.encoder(lidar)
        if encoded.size(1) != 64 or encoded.size(3) != 8:
            raise RuntimeError("Unexpected encoder output shape.")
        latent = self.fusion(encoded.flatten(start_dim=1))
        features = latent.view(latent.size(0), 64, 8)
        features = self.decoder[0](features)
        features = self.decoder[1](features)
        features = self.decoder[2](features)
        features = self.decoder[3](features)
        features = self.decoder[4](features)
        return self.decoder[5](features)

    @torch.jit.export
    def predict_components(self, lidar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw velocity and dynamic logits for auxiliary training/audits."""
        if self.classifier_head is None:
            raise RuntimeError("This model has no auxiliary classifier.")
        features = self._decoded_features(lidar)
        velocity = self.decoder[6](features).transpose(1, 2)
        logits = self.classifier_head(features).squeeze(1)
        return velocity, logits

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        if self.classifier_head is None:
            features = self._decoded_features(lidar)
            return self.decoder[6](features).transpose(1, 2)
        velocity, logits = self.predict_components(lidar)
        moving = torch.sigmoid(logits) >= self.dynamic_threshold
        return torch.where(moving.unsqueeze(-1), velocity, torch.zeros_like(velocity))
