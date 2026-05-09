from __future__ import annotations

import importlib.util
import isaaclab.utils.modifiers as modifiers
import torch
from pathlib import Path


def _load_observation_modifiers_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "isaaclab_tasks"
        / "manager_based"
        / "navigation"
        / "mdp"
        / "observation_modifiers.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_tasks_observation_modifiers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load observation modifiers module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_observation_modifiers = _load_observation_modifiers_module()
ElementwiseDropout = _observation_modifiers.ElementwiseDropout
UniformEpisodeBias = _observation_modifiers.UniformEpisodeBias
UniformEpisodeScale = _observation_modifiers.UniformEpisodeScale
UniformRandomWalkBias = _observation_modifiers.UniformRandomWalkBias


def main() -> None:
    test_functions = [
        test_uniform_episode_bias_reset_is_env_local,
        test_uniform_episode_scale_supports_deterministic_scales,
        test_random_walk_bias_stays_within_bounds,
        test_elementwise_dropout_respects_probability_extremes,
    ]

    for test_function in test_functions:
        test_function()
        print(f"PASS: {test_function.__name__}")

    print(f"Executed {len(test_functions)} observation modifier tests.")


def test_uniform_episode_bias_reset_is_env_local():
    torch.manual_seed(0)
    modifier_cfg = modifiers.ModifierCfg(
        func=UniformEpisodeBias,
        params={"bias_min": (-0.2, -0.2, -0.2), "bias_max": (0.2, 0.2, 0.2)},
    )
    modifier = UniformEpisodeBias(cfg=modifier_cfg, data_dim=(4, 3), device="cpu")

    original_output = modifier(torch.zeros(4, 3))
    modifier.reset(env_ids=[1, 3])
    updated_output = modifier(torch.zeros(4, 3))

    torch.testing.assert_close(updated_output[0], original_output[0])
    torch.testing.assert_close(updated_output[2], original_output[2])
    assert not torch.allclose(updated_output[1], original_output[1])
    assert not torch.allclose(updated_output[3], original_output[3])


def test_uniform_episode_scale_supports_deterministic_scales():
    modifier_cfg = modifiers.ModifierCfg(
        func=UniformEpisodeScale,
        params={"scale_min": (1.1, 0.9, 1.0), "scale_max": (1.1, 0.9, 1.0)},
    )
    modifier = UniformEpisodeScale(cfg=modifier_cfg, data_dim=(2, 3), device="cpu")

    scaled = modifier(torch.ones(2, 3))

    expected = torch.tensor([[1.1, 0.9, 1.0], [1.1, 0.9, 1.0]])
    torch.testing.assert_close(scaled, expected)


def test_random_walk_bias_stays_within_bounds():
    torch.manual_seed(0)
    modifier_cfg = modifiers.ModifierCfg(
        func=UniformRandomWalkBias,
        params={
            "step_min": (-0.05, -0.05, -0.05),
            "step_max": (0.05, 0.05, 0.05),
            "drift_min": (-0.2, -0.2, -0.2),
            "drift_max": (0.2, 0.2, 0.2),
        },
    )
    modifier = UniformRandomWalkBias(cfg=modifier_cfg, data_dim=(8, 3), device="cpu")

    output = torch.zeros(8, 3)
    for _ in range(64):
        output = modifier(torch.zeros(8, 3))

    assert torch.all(output <= 0.2 + 1e-6)
    assert torch.all(output >= -0.2 - 1e-6)


def test_elementwise_dropout_respects_probability_extremes():
    modifier_cfg = modifiers.ModifierCfg(
        func=ElementwiseDropout,
        params={"drop_probability": (1.0, 0.0, 1.0), "fill_value": 0.0},
    )
    modifier = ElementwiseDropout(cfg=modifier_cfg, data_dim=(3, 3), device="cpu")

    output = modifier(torch.ones(3, 3))

    assert torch.all(output[:, 0] == 0.0)
    assert torch.all(output[:, 1] == 1.0)
    assert torch.all(output[:, 2] == 0.0)


if __name__ == "__main__":
    main()