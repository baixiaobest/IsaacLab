"""Focused checks for the optional per-return classifier."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.losses import class_balanced_classifier_bce
from src.model import TemporalLidarVelocityCNN
from train import _load_resume_checkpoint, _save_torchscript, dashboard_metrics, evaluate


def test_aux_forward_gate_and_torchscript(tmp_path):
    lidar = torch.zeros(1, 2, 4, 128)
    legacy = TemporalLidarVelocityCNN(4).eval()
    assert legacy(lidar).shape == (1, 128, 2)
    assert not any(key.startswith("classifier_head.") for key in legacy.state_dict())

    model = TemporalLidarVelocityCNN(4, aux_classifier=True).eval()
    with torch.no_grad():
        model.decoder[6].weight.zero_()
        model.decoder[6].bias.copy_(torch.tensor([1.0, 0.0]))
        model.classifier_head.weight.zero_()
        model.classifier_head.bias.fill_(-10.0)
    raw, logits = model.predict_components(lidar)
    assert raw.shape == (1, 128, 2) and logits.shape == (1, 128)
    assert torch.all(raw[..., 0] == 1)
    assert torch.count_nonzero(model(lidar)) == 0
    scripted = torch.jit.script(model)
    assert torch.count_nonzero(scripted(lidar)) == 0
    with torch.no_grad():
        model.classifier_head.bias.fill_(10.0)
    assert torch.all(model(lidar)[..., 0] == 1)

    path = tmp_path / "aux_jit.pt"
    _save_torchscript(model, path)
    extras = {"aux_classifier.txt": b"", "dynamic_threshold.txt": b""}
    loaded = torch.jit.load(str(path), _extra_files=extras)
    assert extras == {"aux_classifier.txt": b"1", "dynamic_threshold.txt": b"0.3"}
    assert loaded(lidar).shape == (1, 128, 2)


def test_classifier_loss_balances_classes_and_ignores_missing_returns():
    logits = torch.tensor([[0.0, 2.0, -2.0, 99.0]], requires_grad=True)
    reflection = torch.tensor([[True, True, True, False]])
    dynamic = torch.tensor([[False, True, True, False]])
    loss = class_balanced_classifier_bce(logits, reflection, dynamic)
    expected = (torch.nn.functional.softplus(logits[0, 0])
                + torch.nn.functional.softplus(-logits[0, 1:3]).mean()) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 3] == 0
    assert torch.isfinite(class_balanced_classifier_bce(logits, reflection & ~dynamic, dynamic))


def test_evaluate_logs_only_essential_classifier_metrics():
    model = TemporalLidarVelocityCNN(4, aux_classifier=True).eval()
    with torch.no_grad():
        model.decoder[6].weight.zero_()
        model.decoder[6].bias.copy_(torch.tensor([1.0, 0.0]))
        model.classifier_head.weight.zero_()
        model.classifier_head.bias.fill_(-10.0)
    target = torch.zeros(128, 2)
    target[0, 0] = 1.0
    reflection = torch.zeros(128, dtype=torch.bool)
    reflection[:2] = True
    dynamic = torch.zeros(128, dtype=torch.bool)
    dynamic[0] = True
    loader = DataLoader([{
        "lidar": torch.zeros(2, 4, 128), "target": target,
        "reflection_mask": reflection, "dynamic_mask": dynamic,
        "range_m": torch.ones(128), "ray_coverage": torch.tensor(0.5),
    }], batch_size=1)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert metrics["dynamic_rmse"] == pytest.approx(1.0)
    assert metrics["static_false_motion"] == 0.0
    assert metrics["dynamic_recall"] == 0.0
    assert metrics["static_false_positive_rate"] == 0.0
    selected = dashboard_metrics(metrics)
    assert {"classifier_bce", "dynamic_recall", "static_false_positive_rate"} <= selected.keys()
    assert not any(key.startswith("raw_") for key in selected)


def test_legacy_checkpoint_initializes_aux_model_only(tmp_path):
    legacy = TemporalLidarVelocityCNN(4)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": legacy.state_dict(), "projection_contract": "temporal_lidar_v4",
                "num_frames": 4}, checkpoint)
    aux = TemporalLidarVelocityCNN(4, aux_classifier=True)
    args = (checkpoint, aux, object(), torch.device("cpu"))
    assert _load_resume_checkpoint(*args, True, 3e-4, 1e-5, (None, None))[0] == 1
    torch.testing.assert_close(aux.decoder[6].weight, legacy.decoder[6].weight)
    with pytest.raises(RuntimeError, match="variants differ"):
        _load_resume_checkpoint(*args, False, 3e-4, 1e-5, (None, None))
