from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.modifiers import ModifierBase, ModifierCfg


def _expand_param(value: float | Sequence[float], tail_shape: tuple[int, ...], device: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 1:
        return tensor.reshape((1,) * len(tail_shape)).expand(tail_shape)
    return tensor.reshape(tail_shape)


class UniformEpisodeBias(ModifierBase):
    """Apply an episode-constant additive bias sampled independently per environment."""

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg=cfg, data_dim=data_dim, device=device)
        self._sample_shape = data_dim[1:]
        self._bias_min = _expand_param(cfg.params["bias_min"], self._sample_shape, device)
        self._bias_max = _expand_param(cfg.params["bias_max"], self._sample_shape, device)
        self._bias = torch.zeros(data_dim, dtype=torch.float32, device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
            batch_size = self._data_dim[0]
        else:
            batch_size = len(env_ids)

        random_sample = torch.rand((batch_size, *self._sample_shape), device=self._device)
        self._bias[env_ids] = self._bias_min.unsqueeze(0) + random_sample * (self._bias_max - self._bias_min).unsqueeze(0)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data + self._bias


class UniformEpisodeScale(ModifierBase):
    """Apply an episode-constant multiplicative scale sampled independently per environment."""

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg=cfg, data_dim=data_dim, device=device)
        self._sample_shape = data_dim[1:]
        self._scale_min = _expand_param(cfg.params["scale_min"], self._sample_shape, device)
        self._scale_max = _expand_param(cfg.params["scale_max"], self._sample_shape, device)
        self._scale = torch.ones(data_dim, dtype=torch.float32, device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
            batch_size = self._data_dim[0]
        else:
            batch_size = len(env_ids)

        random_sample = torch.rand((batch_size, *self._sample_shape), device=self._device)
        self._scale[env_ids] = self._scale_min.unsqueeze(0) + random_sample * (self._scale_max - self._scale_min).unsqueeze(0)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data * self._scale


class UniformRandomWalkBias(ModifierBase):
    """Apply a bounded random-walk additive bias to emulate low-frequency sensor drift."""

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg=cfg, data_dim=data_dim, device=device)
        self._sample_shape = data_dim[1:]
        self._step_min = _expand_param(cfg.params["step_min"], self._sample_shape, device)
        self._step_max = _expand_param(cfg.params["step_max"], self._sample_shape, device)
        self._drift_min = _expand_param(cfg.params["drift_min"], self._sample_shape, device)
        self._drift_max = _expand_param(cfg.params["drift_max"], self._sample_shape, device)
        self._bias = torch.zeros(data_dim, dtype=torch.float32, device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._bias.zero_()
        else:
            self._bias[env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        random_sample = torch.rand(self._bias.shape, device=self._device)
        delta = self._step_min.unsqueeze(0) + random_sample * (self._step_max - self._step_min).unsqueeze(0)
        self._bias.add_(delta)
        self._bias.clamp_(min=self._drift_min.unsqueeze(0), max=self._drift_max.unsqueeze(0))
        return data + self._bias


class ElementwiseDropout(ModifierBase):
    """Randomly mask elements with a configured probability on every call."""

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg=cfg, data_dim=data_dim, device=device)
        self._sample_shape = data_dim[1:]
        self._drop_probability = _expand_param(cfg.params["drop_probability"], self._sample_shape, device)
        fill_value = cfg.params.get("fill_value", 0.0)
        self._fill_value = _expand_param(fill_value, self._sample_shape, device)

    def reset(self, env_ids: Sequence[int] | None = None):
        return None

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        drop_mask = torch.rand(data.shape, device=self._device) < self._drop_probability.unsqueeze(0)
        fill_value = self._fill_value.unsqueeze(0).expand_as(data)
        return torch.where(drop_mask, fill_value, data)