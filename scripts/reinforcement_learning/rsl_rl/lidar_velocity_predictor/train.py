"""Train the temporal-LiDAR point velocity predictor."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import DynamicAwareBatchSampler, PointVelocityDataset, save_split_metadata
from src.losses import masked_class_balanced_huber, masked_class_balanced_huber_totals, masked_metrics
from src.model import TemporalLidarVelocityCNN

REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiDAR per-bin velocity predictor.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="logs/lidar_velocity_predictor")
    parser.add_argument("--run_name", default=None, help="Optional run name; defaults to a timestamp.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--checkpoint_save_interval",
        type=int,
        default=10,
        help="Save and upload a retained epoch checkpoint every N epochs; set 0 to disable.",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
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


def _save_torchscript(model: torch.nn.Module, output_path: Path) -> None:
    """Export the current model with its deployment input/output signature."""
    model.eval()
    torch.jit.script(model).save(str(output_path))


def _publish_deployment_torchscript(model: torch.nn.Module, output_path: Path) -> None:
    """Atomically replace the fixed CBF deployment JIT artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    _save_torchscript(model, temporary)
    os.replace(temporary, output_path)


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, static_loss_weight: float = 0.5
) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    loss_totals = {"static_numerator": 0.0, "static_weight": 0.0, "dynamic_numerator": 0.0, "dynamic_weight": 0.0}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            lidar = batch["lidar"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)
            reflection = batch["reflection_mask"].to(device)
            dynamic = batch["dynamic_mask"].to(device)
            ranges = batch["range_m"].to(device)
            prediction = model(lidar)
            for key, value in masked_class_balanced_huber_totals(
                prediction, target, reflection, dynamic, ranges
            ).items():
                loss_totals[key] += float(value.item())
            for key, (value, count) in masked_metrics(prediction, target, reflection, dynamic, ranges).items():
                total = totals.setdefault(key, [0.0, 0.0])
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
    for key, (value, count) in totals.items():
        if count:
            metrics[key] = value / count
    for name in ("all", "static", "dynamic", "within_5m", "within_2m", "dynamic_within_5m", "dynamic_within_2m"):
        if name in metrics:
            metrics[f"{name}_rmse"] = metrics[name] ** 0.5
        if f"{name}_abs" in metrics:
            # L1 was accumulated over x/y, report per-component MAE.
            metrics[f"{name}_mae"] = metrics[f"{name}_abs"] / 2.0
    return metrics


def main() -> None:
    args = parse_args()
    if args.checkpoint_save_interval < 0:
        raise ValueError("--checkpoint_save_interval must be greater than or equal to zero.")
    if not 0.0 <= args.static_loss_weight <= 1.0:
        raise ValueError("--static_loss_weight must be in [0, 1].")
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).expanduser().resolve() / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    deployment_jit_path = Path(args.deployment_jit_path).expanduser().resolve()
    dataset = PointVelocityDataset(args.dataset_path, input_name="lidar_noisy")
    train, validation, test = dataset.split(args.seed)
    if len(validation) == 0 or len(test) == 0:
        raise RuntimeError("Dataset is too small for stratified validation/test splits.")
    save_split_metadata(str(output / "splits.json"), train, validation, test)
    train_loader = DataLoader(dataset, batch_sampler=DynamicAwareBatchSampler(train, args.batch_size, args.seed), num_workers=args.num_workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    model = TemporalLidarVelocityCNN().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_validation_loss = float("inf")
    periodic_checkpoint_dir = output / "checkpoints"
    if args.checkpoint_save_interval > 0:
        periodic_checkpoint_dir.mkdir(exist_ok=True)
    metadata = {
        "args": vars(args), "num_samples": len(dataset), "train": len(train), "validation": len(validation), "test": len(test),
        "target_velocity_frame": "body_xy", "deployment_jit_path": str(deployment_jit_path),
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
                "input_shape": [2, 4, 128],
                "target_shape": [128, 2],
                "target_velocity_frame": "body_xy",
            },
            allow_val_change=True,
        )
        _upload_file_to_wandb(wandb_run, output / "metadata.json", output)
        _upload_file_to_wandb(wandb_run, output / "splits.json", output)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                lidar = batch["lidar"].to(device, dtype=torch.float32)
                target = batch["target"].to(device, dtype=torch.float32)
                reflection = batch["reflection_mask"].to(device)
                dynamic = batch["dynamic_mask"].to(device)
                ranges = batch["range_m"].to(device)
                loss, _ = masked_class_balanced_huber(
                    model(lidar), target, reflection, dynamic, ranges, static_loss_weight=args.static_loss_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.item())
            average_train_loss = train_loss / max(len(train_loader), 1)
            metrics = evaluate(model, validation_loader, device, args.static_loss_weight)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "model": "TemporalLidarVelocityCNN",
                "args": vars(args),
            }
            torch.save(checkpoint, output / "last.pt")
            _upload_file_to_wandb(wandb_run, output / "last.pt", output)
            if args.checkpoint_save_interval > 0 and epoch % args.checkpoint_save_interval == 0:
                periodic_checkpoint_path = periodic_checkpoint_dir / f"epoch_{epoch:04d}.pt"
                torch.save(checkpoint, periodic_checkpoint_path)
                _upload_file_to_wandb(wandb_run, periodic_checkpoint_path, output)
            score = metrics.get("loss", float("inf"))
            is_best = score < best_validation_loss
            if is_best:
                best_validation_loss = score
                torch.save(checkpoint, output / "best.pt")
                _upload_file_to_wandb(wandb_run, output / "best.pt", output)
                _save_torchscript(model, output / "best_jit.pt")
                _publish_deployment_torchscript(model, deployment_jit_path)
                _upload_file_to_wandb(wandb_run, output / "best_jit.pt", output)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": average_train_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "validation/selection_loss": score,
                        "validation/is_best": int(is_best),
                        **{f"validation/{name}": value for name, value in metrics.items()},
                    },
                    step=epoch,
                )
            print(f"[Epoch {epoch:03d}] train_loss={average_train_loss:.6f} validation_loss={score:.6f}")
        final_metrics = {
            "validation": evaluate(model, validation_loader, device, args.static_loss_weight),
            "test": evaluate(model, DataLoader(test, batch_size=args.batch_size), device, args.static_loss_weight),
        }
        if wandb_run is not None:
            wandb_run.summary.update({f"final/{split}/{name}": value for split, values in final_metrics.items() for name, value in values.items()})
        print(json.dumps(final_metrics, indent=2))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
