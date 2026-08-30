"""Train the temporal-LiDAR point velocity predictor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import DynamicAwareBatchSampler, PointVelocityDataset, save_split_metadata
from src.losses import masked_class_balanced_huber, masked_metrics
from src.model import TemporalLidarVelocityCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiDAR per-bin velocity predictor.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="logs/lidar_velocity_predictor")
    parser.add_argument("--run_name", default="run")
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


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            lidar = batch["lidar"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)
            reflection = batch["reflection_mask"].to(device)
            dynamic = batch["dynamic_mask"].to(device)
            ranges = batch["range_m"].to(device)
            for key, (value, count) in masked_metrics(model(lidar), target, reflection, dynamic, ranges).items():
                total = totals.setdefault(key, [0.0, 0.0])
                total[0] += value
                total[1] += count
    metrics: dict[str, float] = {}
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).expanduser().resolve() / args.run_name
    output.mkdir(parents=True, exist_ok=True)
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
    best_rmse = float("inf")
    periodic_checkpoint_dir = output / "checkpoints"
    if args.checkpoint_save_interval > 0:
        periodic_checkpoint_dir.mkdir(exist_ok=True)
    metadata = {"args": vars(args), "num_samples": len(dataset), "train": len(train), "validation": len(validation), "test": len(test)}
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
                loss, _ = masked_class_balanced_huber(model(lidar), target, reflection, dynamic, ranges)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.item())
            average_train_loss = train_loss / max(len(train_loader), 1)
            metrics = evaluate(model, validation_loader, device)
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
            score = metrics.get("dynamic_within_5m_rmse", metrics.get("dynamic_rmse", float("inf")))
            is_best = score < best_rmse
            if is_best:
                best_rmse = score
                torch.save(checkpoint, output / "best.pt")
                _upload_file_to_wandb(wandb_run, output / "best.pt", output)
                _save_torchscript(model, output / "best_jit.pt")
                _upload_file_to_wandb(wandb_run, output / "best_jit.pt", output)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": average_train_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "validation/selection_dynamic_within_5m_rmse": score,
                        "validation/is_best": int(is_best),
                        **{f"validation/{name}": value for name, value in metrics.items()},
                    },
                    step=epoch,
                )
            print(f"[Epoch {epoch:03d}] train_loss={average_train_loss:.6f} dynamic_5m_rmse={score:.6f}")
        final_metrics = {
            "validation": evaluate(model, validation_loader, device),
            "test": evaluate(model, DataLoader(test, batch_size=args.batch_size), device),
        }
        if wandb_run is not None:
            wandb_run.summary.update({f"final/{split}/{name}": value for split, values in final_metrics.items() for name, value in values.items()})
        print(json.dumps(final_metrics, indent=2))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
