"""Train the temporal-LiDAR point velocity predictor."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import DynamicAwareBatchSampler, PointVelocityDataset, save_split_metadata
from src.losses import (
    class_balanced_classifier_bce,
    class_balanced_classifier_bce_totals,
    masked_class_balanced_huber,
    masked_class_balanced_huber_totals,
    masked_metrics,
)
from src.model import TemporalLidarVelocityCNN

REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiDAR per-bin velocity predictor.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="logs/lidar_velocity_predictor")
    parser.add_argument("--run_name", default=None, help="Optional run name; defaults to a timestamp.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help=(
            "Path to a training .pt checkpoint (for example last.pt). Restores model and AdamW state and "
            "continues at the following epoch. --epochs is the total target epoch count."
        ),
    )
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Load only model weights from --resume_checkpoint and start a new optimizer/epoch schedule.",
    )
    parser.add_argument(
        "--checkpoint_save_interval",
        type=int,
        default=10,
        help="Save and upload a retained epoch checkpoint every N epochs; set 0 to disable.",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--aux_classifier", action="store_true", help="Add a per-return dynamic/static classifier head.")
    parser.add_argument("--dynamic_threshold", type=float, default=0.3, help="Aux classifier probability cutoff for deployed velocity.")
    parser.add_argument("--classification_loss_weight", type=float, default=0.25)
    parser.add_argument("--num_frames", type=int, default=None, help="Use the newest N frames; defaults to all frames in the dataset.")
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--static_loss_weight",
        type=float,
        default=0.5,
        help="Static term weight in masked class-balanced Smooth-L1; dynamic weight is 1 - this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--logger",
        choices=("wandb", "none"),
        default="wandb",
        help="Training logger backend.",
    )
    parser.add_argument("--wandb_project", default="lidar velocity predictor")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument(
        "--deployment_jit_path",
        default=str(REPO_ROOT / "logs/lidar_velocity_predictor/best_jit.pt"),
        help="Fixed body-frame TorchScript artifact replaced atomically whenever validation finds a new best model.",
    )
    return parser.parse_args()


def _uses_wandb(logger_name: str) -> bool:
    """Return whether Weights & Biases logging is enabled."""
    return logger_name == "wandb"


def _init_wandb(args: argparse.Namespace, run_dir: Path):
    """Initialize W&B using the same lazy-import pattern as velocity_estimator."""
    if not _uses_wandb(args.logger):
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("Weights & Biases logging requested but the 'wandb' package is not installed.") from error
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        dir=str(run_dir),
        config={key: value for key, value in vars(args).items() if key != "run_name"},
    )


def _upload_file_to_wandb(wandb_run: Any, file_path: Path, base_path: Path) -> None:
    """Upload a run artifact immediately, matching velocity_estimator behavior."""
    if wandb_run is not None:
        wandb_run.save(str(file_path), base_path=str(base_path), policy="now")


def _save_torchscript(model: torch.nn.Module, output_path: Path,
                      noise_contract: tuple[float | None, str | None] | None = None) -> None:
    """Export the current model with its deployment input/output signature."""
    model.eval()
    extra_files = {"projection_contract.txt": b"temporal_lidar_v4", "num_frames.txt": str(model.num_frames).encode()}
    extra_files["aux_classifier.txt"] = str(int(model.aux_classifier)).encode()
    if model.aux_classifier:
        extra_files["dynamic_threshold.txt"] = str(model.dynamic_threshold).encode()
    if noise_contract is not None and noise_contract[0] is not None:
        extra_files["range_noise_m.txt"] = str(noise_contract[0]).encode()
        extra_files["range_noise_applies_to.txt"] = str(noise_contract[1]).encode()
    torch.jit.save(
        torch.jit.script(model), str(output_path),
        _extra_files=extra_files,
    )


def _publish_deployment_torchscript(model: torch.nn.Module, output_path: Path,
                                    noise_contract: tuple[float | None, str | None] | None = None) -> None:
    """Atomically replace the fixed CBF deployment JIT artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    _save_torchscript(model, temporary, noise_contract)
    os.replace(temporary, output_path)


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_only: bool,
    learning_rate: float,
    weight_decay: float,
    expected_noise_contract: tuple[float | None, str | None],
) -> tuple[int, float]:
    """Restore a trainable checkpoint and return the next epoch and best loss."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load {checkpoint_path} as a training checkpoint. "
            "Use a .pt checkpoint such as last.pt or best.pt, not the TorchScript best_jit.pt artifact."
        ) from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError(
            f"{checkpoint_path} is not a LiDAR predictor training checkpoint. "
            "Expected a dictionary containing model_state_dict."
        )
    if checkpoint.get("projection_contract") not in ("temporal_lidar_v3", "temporal_lidar_v4"):
        raise RuntimeError(f"{checkpoint_path} predates the evaluation-pose projection contract; retrain from schema-v3 data.")
    if int(checkpoint.get("num_frames", 4)) != model.num_frames:
        raise RuntimeError(f"{checkpoint_path} uses {checkpoint.get('num_frames', 4)} frames, but training requested {model.num_frames}.")
    checkpoint_aux = bool(checkpoint.get("aux_classifier", False))
    if checkpoint_aux != model.aux_classifier and not (model_only and model.aux_classifier and not checkpoint_aux):
        raise RuntimeError("Checkpoint and requested model variants differ. Only --resume_model_only from velocity-only to aux is supported.")
    if checkpoint_aux and not model_only and float(checkpoint.get("dynamic_threshold", 0.3)) != model.dynamic_threshold:
        raise RuntimeError("Full resume requires the checkpoint's dynamic threshold; use --resume_model_only to change it.")
    checkpoint_noise = (checkpoint.get("range_noise_m"), checkpoint.get("range_noise_applies_to"))
    if not model_only and checkpoint_noise != expected_noise_contract:
        raise RuntimeError(
            "Checkpoint and dataset have different LiDAR range-noise settings. "
            "Use --resume_model_only to intentionally fine-tune across noise distributions."
        )
    if model.aux_classifier and not checkpoint_aux:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if set(missing) != {"classifier_head.weight", "classifier_head.bias"} or unexpected:
            raise RuntimeError(f"Unexpected legacy checkpoint weights: missing={missing}, unexpected={unexpected}.")
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if model_only:
        return 1, float("inf")
    if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
        raise RuntimeError(
            f"{checkpoint_path} has no optimizer_state_dict. Use --resume_model_only to fine-tune from model weights only."
        )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Keep explicit command-line hyperparameters effective after optimizer-state
    # restoration, while retaining AdamW moments for a genuine continuation.
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["weight_decay"] = weight_decay
    completed_epoch = checkpoint.get("epoch")
    if not isinstance(completed_epoch, int) or completed_epoch < 0:
        raise RuntimeError(f"{checkpoint_path} has an invalid completed epoch: {completed_epoch!r}.")
    metrics = checkpoint.get("metrics", {})
    prior_best = checkpoint.get("best_validation_loss")
    # Checkpoints made before resume support did not save the running best
    # score in last.pt. Recover it from its sibling best.pt when available.
    if prior_best is None:
        best_checkpoint_path = checkpoint_path.with_name("best.pt")
        if best_checkpoint_path != checkpoint_path and best_checkpoint_path.is_file():
            try:
                best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
                prior_best = best_checkpoint.get("metrics", {}).get("loss")
            except Exception:
                prior_best = None
    if prior_best is None:
        prior_best = metrics.get("loss", float("inf"))
    if not isinstance(prior_best, (float, int)):
        prior_best = float("inf")
    return completed_epoch + 1, float(prior_best)


def _resume_best_checkpoint_path(checkpoint_path: Path) -> Path | None:
    """Find the best checkpoint belonging to a resumed training run, if retained."""
    candidate = checkpoint_path if checkpoint_path.name == "best.pt" else checkpoint_path.with_name("best.pt")
    return candidate if candidate.is_file() else None


def evaluate(
    model: TemporalLidarVelocityCNN, loader: DataLoader, device: torch.device,
    static_loss_weight: float = 0.5, classification_loss_weight: float = 0.25,
) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    loss_totals = {"static_numerator": 0.0, "static_weight": 0.0, "dynamic_numerator": 0.0, "dynamic_weight": 0.0}
    classifier_totals = {"static_numerator": 0.0, "static_count": 0.0, "dynamic_numerator": 0.0, "dynamic_count": 0.0}
    true_positive = false_positive = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            lidar = batch["lidar"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)
            reflection = batch["reflection_mask"].to(device)
            dynamic = batch["dynamic_mask"].to(device)
            ranges = batch["range_m"].to(device)
            if model.aux_classifier:
                raw_velocity, logits = model.predict_components(lidar)
                moving = torch.sigmoid(logits) >= model.dynamic_threshold
                prediction = torch.where(moving.unsqueeze(-1), raw_velocity, torch.zeros_like(raw_velocity))
                for key, value in class_balanced_classifier_bce_totals(logits, reflection, dynamic).items():
                    classifier_totals[key] += float(value.item())
                true_positive += int((moving & dynamic & reflection).sum().item())
                false_positive += int((moving & ~dynamic & reflection).sum().item())
            else:
                raw_velocity = prediction = model(lidar)
            for key, value in masked_class_balanced_huber_totals(
                raw_velocity, target, reflection, dynamic, ranges
            ).items():
                loss_totals[key] += float(value.item())
            for key, (value, count) in masked_metrics(prediction, target, reflection, dynamic, ranges).items():
                total = totals.setdefault(key, [0.0, 0.0])
                total[0] += value
                total[1] += count
            coverage = batch["ray_coverage"]
            for name, subset in (("sparse", coverage < 0.37), ("dense", coverage >= 0.77)):
                if not subset.any():
                    continue
                selected = subset.to(device)
                for key, (value, count) in masked_metrics(
                    prediction[selected], target[selected], reflection[selected], dynamic[selected], ranges[selected]
                ).items():
                    if key not in ("dynamic", "zero_dynamic"):
                        continue
                    total = totals.setdefault(f"{name}_{key}", [0.0, 0.0])
                    total[0] += value
                    total[1] += count
    metrics: dict[str, float] = {}
    class_losses = []
    for class_name in ("static", "dynamic"):
        weight = loss_totals[f"{class_name}_weight"]
        if weight > 0.0:
            class_loss = loss_totals[f"{class_name}_numerator"] / weight
            metrics[f"{class_name}_loss"] = class_loss
            class_losses.append(class_loss)
    if class_losses:
        # Same static/dynamic weighting as the training objective, aggregated
        # over all validation samples rather than per batch.
        class_weights = {"static": static_loss_weight, "dynamic": 1.0 - static_loss_weight}
        active_weight = sum(class_weights[name] for name in ("static", "dynamic") if f"{name}_loss" in metrics)
        metrics["loss"] = (
            sum(class_weights[name] * metrics[f"{name}_loss"] for name in ("static", "dynamic") if f"{name}_loss" in metrics)
            / max(active_weight, 1.0e-12)
        )
    if model.aux_classifier:
        classifier_losses = [
            classifier_totals[f"{name}_numerator"] / classifier_totals[f"{name}_count"]
            for name in ("static", "dynamic") if classifier_totals[f"{name}_count"] > 0
        ]
        metrics["classifier_bce"] = sum(classifier_losses) / len(classifier_losses) if classifier_losses else 0.0
        metrics["loss"] = metrics.get("loss", 0.0) + classification_loss_weight * metrics["classifier_bce"]
        dynamic_count = classifier_totals["dynamic_count"]
        static_count = classifier_totals["static_count"]
        metrics["dynamic_recall"] = true_positive / dynamic_count if dynamic_count else 0.0
        metrics["static_false_positive_rate"] = false_positive / static_count if static_count else 0.0
    for key, (value, count) in totals.items():
        if count:
            metrics[key] = value / count
    metrics["coverage_endpoints_pass"] = float(all(
        metrics.get(f"{name}_dynamic", float("inf")) < metrics.get(f"{name}_zero_dynamic", 0.0)
        for name in ("sparse", "dense")
    ))
    for name in ("all", "static", "dynamic", "within_5m", "within_2m", "dynamic_within_5m", "dynamic_within_2m"):
        if name in metrics:
            metrics[f"{name}_rmse"] = metrics[name] ** 0.5
        if f"{name}_abs" in metrics:
            # L1 was accumulated over x/y, report per-component MAE.
            metrics[f"{name}_mae"] = metrics[f"{name}_abs"] / 2.0
    if "zero_dynamic" in metrics:
        metrics["zero_dynamic_rmse"] = metrics["zero_dynamic"] ** 0.5
    return metrics


def dashboard_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep the W&B dashboard focused on velocity quality and coverage endpoints."""
    selected = {
        name: metrics[name]
        for name in (
            "loss",
            "static_loss",
            "static_rmse",
            "static_mae",
            "dynamic_rmse",
            "zero_dynamic_rmse",
            "dynamic_within_2m_rmse",
            "static_false_motion",
            "classifier_bce",
            "dynamic_recall",
            "static_false_positive_rate",
        )
        if name in metrics
    }
    for coverage in ("sparse", "dense"):
        key = f"{coverage}_dynamic"
        if key in metrics:
            selected[f"{coverage}_dynamic_rmse"] = metrics[key] ** 0.5
    return selected


def main() -> None:
    args = parse_args()
    if args.checkpoint_save_interval < 0:
        raise ValueError("--checkpoint_save_interval must be greater than or equal to zero.")
    if not 0.0 <= args.static_loss_weight <= 1.0:
        raise ValueError("--static_loss_weight must be in [0, 1].")
    if not 0.0 < args.dynamic_threshold < 1.0:
        raise ValueError("--dynamic_threshold must be between zero and one.")
    if args.classification_loss_weight < 0.0:
        raise ValueError("--classification_loss_weight must be nonnegative.")
    if args.resume_model_only and args.resume_checkpoint is None:
        raise ValueError("--resume_model_only requires --resume_checkpoint.")
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).expanduser().resolve() / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    deployment_jit_path = Path(args.deployment_jit_path).expanduser().resolve()
    dataset = PointVelocityDataset(args.dataset_path, input_name="lidar_noisy", num_frames=args.num_frames)
    args.num_frames = dataset.num_frames
    train, validation, test = dataset.split(args.seed)
    if len(validation) == 0 or len(test) == 0:
        raise RuntimeError("Dataset is too small for stratified validation/test splits.")
    save_split_metadata(str(output / "splits.json"), train, validation, test)
    train_loader = DataLoader(dataset, batch_sampler=DynamicAwareBatchSampler(train, args.batch_size, args.seed), num_workers=args.num_workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.backends.cudnn.enabled:
        # Some Isaac Sim Python environments expose cuDNN but cannot initialize
        # it. Check before the first training batch and use PyTorch's CUDA
        # convolution implementation when that backend is unavailable.
        try:
            probe = torch.zeros((1, 2, args.num_frames, 128), device=device)
            torch.nn.functional.conv2d(probe, torch.zeros((16, 2, 1, 5), device=device), padding=(0, 2))
        except RuntimeError as error:
            if "CUDNN_STATUS_NOT_INITIALIZED" not in str(error):
                raise
            torch.backends.cudnn.enabled = False
            print("cuDNN could not initialize; using PyTorch CUDA convolutions instead.", flush=True)
    model = TemporalLidarVelocityCNN(args.num_frames, args.aux_classifier, args.dynamic_threshold).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_validation_loss = float("inf")
    start_epoch = 1
    resume_checkpoint_path = None
    resume_best_checkpoint_path = None
    if args.resume_checkpoint is not None:
        resume_checkpoint_path = Path(args.resume_checkpoint).expanduser().resolve()
        start_epoch, best_validation_loss = _load_resume_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer,
            device,
            args.resume_model_only,
            args.learning_rate,
            args.weight_decay,
            dataset.noise_contract or (None, None),
        )
        resume_best_checkpoint_path = _resume_best_checkpoint_path(resume_checkpoint_path)
        mode = "model weights only" if args.resume_model_only else "model and AdamW state"
        print(f"Resumed {mode} from {resume_checkpoint_path}; starting epoch {start_epoch}.")
    if start_epoch > args.epochs:
        raise ValueError(
            f"The resume checkpoint completed epoch {start_epoch - 1}, but --epochs is {args.epochs}. "
            "Set --epochs higher than the checkpoint epoch to continue training."
        )
    periodic_checkpoint_dir = output / "checkpoints"
    if args.checkpoint_save_interval > 0:
        periodic_checkpoint_dir.mkdir(exist_ok=True)
    metadata = {
        "args": vars(args), "num_samples": len(dataset), "train": len(train), "validation": len(validation), "test": len(test),
        "target_velocity_frame": "evaluation_yaw_xy", "projection_contract": "temporal_lidar_v4",
        "num_frames": args.num_frames, "available_frames": dataset.available_frames,
        "aux_classifier": args.aux_classifier, "dynamic_threshold": args.dynamic_threshold if args.aux_classifier else None,
        "range_noise_m": dataset.noise_contract[0] if dataset.noise_contract else None,
        "range_noise_applies_to": dataset.noise_contract[1] if dataset.noise_contract else None,
        "deployment_jit_path": str(deployment_jit_path),
        "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        "resume_model_only": args.resume_model_only,
        "start_epoch": start_epoch,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    wandb_run = _init_wandb(args, output)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "run_dir": str(output),
                "num_samples": len(dataset),
                "num_episodes": len(dataset.entries),
                "num_training_samples": len(train),
                "num_validation_samples": len(validation),
                "num_test_samples": len(test),
                "input_shape": [2, args.num_frames, 128],
                "target_shape": [128, 2],
                "target_velocity_frame": "evaluation_yaw_xy",
            },
            allow_val_change=True,
        )
        _upload_file_to_wandb(wandb_run, output / "metadata.json", output)
        _upload_file_to_wandb(wandb_run, output / "splits.json", output)
    # A resumed run writes to its own timestamped directory by default. Carry
    # forward the previous best artifact so the run remains deployable even
    # when no later epoch improves on it.
    if not args.resume_model_only and resume_best_checkpoint_path is not None:
        output_best_checkpoint = output / "best.pt"
        if resume_best_checkpoint_path.resolve() != output_best_checkpoint.resolve():
            shutil.copy2(resume_best_checkpoint_path, output_best_checkpoint)
            _upload_file_to_wandb(wandb_run, output_best_checkpoint, output)
        source_best_jit = resume_best_checkpoint_path.with_name("best_jit.pt")
        output_best_jit = output / "best_jit.pt"
        if source_best_jit.is_file() and source_best_jit.resolve() != output_best_jit.resolve():
            shutil.copy2(source_best_jit, output_best_jit)
            _upload_file_to_wandb(wandb_run, output_best_jit, output)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                lidar = batch["lidar"].to(device, dtype=torch.float32)
                target = batch["target"].to(device, dtype=torch.float32)
                reflection = batch["reflection_mask"].to(device)
                dynamic = batch["dynamic_mask"].to(device)
                ranges = batch["range_m"].to(device)
                if model.aux_classifier:
                    raw_velocity, logits = model.predict_components(lidar)
                else:
                    raw_velocity = model(lidar)
                velocity_loss, _ = masked_class_balanced_huber(
                    raw_velocity, target, reflection, dynamic, ranges, static_loss_weight=args.static_loss_weight
                )
                loss = velocity_loss
                if model.aux_classifier:
                    loss = loss + args.classification_loss_weight * class_balanced_classifier_bce(logits, reflection, dynamic)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.item())
            average_train_loss = train_loss / max(len(train_loader), 1)
            metrics = evaluate(model, validation_loader, device, args.static_loss_weight, args.classification_loss_weight)
            score = metrics.get("loss", float("inf"))
            is_best = score < best_validation_loss and bool(metrics["coverage_endpoints_pass"])
            if is_best:
                best_validation_loss = score
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "best_validation_loss": best_validation_loss,
                "model": "TemporalLidarVelocityCNN",
                "projection_contract": "temporal_lidar_v4",
                "num_frames": args.num_frames,
                "aux_classifier": args.aux_classifier,
                "dynamic_threshold": args.dynamic_threshold if args.aux_classifier else None,
                "range_noise_m": dataset.noise_contract[0] if dataset.noise_contract else None,
                "range_noise_applies_to": dataset.noise_contract[1] if dataset.noise_contract else None,
                "args": vars(args),
            }
            torch.save(checkpoint, output / "last.pt")
            _upload_file_to_wandb(wandb_run, output / "last.pt", output)
            if args.checkpoint_save_interval > 0 and epoch % args.checkpoint_save_interval == 0:
                periodic_checkpoint_path = periodic_checkpoint_dir / f"epoch_{epoch:04d}.pt"
                torch.save(checkpoint, periodic_checkpoint_path)
                _upload_file_to_wandb(wandb_run, periodic_checkpoint_path, output)
            if is_best:
                torch.save(checkpoint, output / "best.pt")
                _upload_file_to_wandb(wandb_run, output / "best.pt", output)
                _save_torchscript(model, output / "best_jit.pt", dataset.noise_contract)
                _publish_deployment_torchscript(model, deployment_jit_path, dataset.noise_contract)
                _upload_file_to_wandb(wandb_run, output / "best_jit.pt", output)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": average_train_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        **{f"validation/{name}": value for name, value in dashboard_metrics(metrics).items()},
                    },
                    step=epoch,
                )
            print(f"[Epoch {epoch:03d}] train_loss={average_train_loss:.6f} validation_loss={score:.6f}")
        final_metrics = {
            "validation": evaluate(model, validation_loader, device, args.static_loss_weight, args.classification_loss_weight),
            "test": evaluate(model, DataLoader(test, batch_size=args.batch_size), device, args.static_loss_weight, args.classification_loss_weight),
        }
        (output / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        if wandb_run is not None:
            wandb_run.summary.update({
                f"final/{split}/{name}": value
                for split, values in final_metrics.items()
                for name, value in dashboard_metrics(values).items()
            })
        print(json.dumps(final_metrics, indent=2))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
