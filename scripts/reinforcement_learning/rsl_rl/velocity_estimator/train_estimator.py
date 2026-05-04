# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import create_estimator_datasets
from model import VelocityEstimator


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for estimator training."""
    parser = argparse.ArgumentParser(description="Train a velocity estimator from rollout datasets.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to an HDF5 rollout file or a directory of HDF5 files.")
    parser.add_argument("--output_dir", type=str, default="logs/velocity_estimator", help="Directory for checkpoints and logs.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name. Defaults to a timestamp.")
    parser.add_argument("--horizon", type=int, default=10, help="Number of past observations, including the current step, used by the estimator.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument(
        "--checkpoint_save_interval",
        type=int,
        default=10,
        help="Save an extra epoch checkpoint every N epochs. Set to 0 to disable periodic checkpoints.",
    )
    parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-5, help="Optimizer weight decay.")
    parser.add_argument("--validation_fraction", type=float, default=0.1, help="Fraction of episodes reserved for validation.")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128], help="Hidden layer sizes for the estimator MLP.")
    parser.add_argument("--activation", type=str, default="elu", choices=["relu", "elu", "gelu", "silu"], help="Activation function.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Optional dropout rate in the MLP.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for training.")
    parser.add_argument(
        "--exclude_input_terms",
        type=str,
        nargs="*",
        default=None,
        help="Optional extra observation term names to exclude from estimator inputs. Ground-truth term names are excluded automatically.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_run_dir(output_dir: str, run_name: str | None) -> str:
    """Create a unique run directory."""
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(output_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _evaluate(model: VelocityEstimator, data_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Run validation metrics."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_examples = 0

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["inputs"].to(device=device, dtype=torch.float32)
            targets = batch["targets"].to(device=device, dtype=torch.float32)
            predictions = model(inputs)

            batch_size = inputs.shape[0]
            total_examples += batch_size
            total_loss += F.mse_loss(predictions, targets, reduction="sum").item()
            total_mae += F.l1_loss(predictions, targets, reduction="sum").item()

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_mae / total_examples


def _save_checkpoint(
    checkpoint_path: str,
    model: VelocityEstimator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    dataset,
    args: argparse.Namespace,
) -> None:
    """Persist a training checkpoint with dataset schema metadata."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "input_paths": dataset.input_paths,
            "target_paths": dataset.target_paths,
            "input_dim": dataset.input_dim,
            "target_dim": dataset.target_dim,
            "horizon": dataset.horizon,
        },
        checkpoint_path,
    )


def _save_run_metadata(run_dir: str, dataset, args: argparse.Namespace, train_subset: Subset, validation_subset: Subset) -> None:
    """Write readable metadata for the training run."""
    metadata = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "horizon": args.horizon,
        "input_paths": dataset.input_paths,
        "target_paths": dataset.target_paths,
        "input_dim": dataset.input_dim,
        "target_dim": dataset.target_dim,
        "num_episodes": len(dataset.episode_entries),
        "num_training_samples": len(train_subset),
        "num_validation_samples": len(validation_subset),
        "env_args": dataset.env_args,
    }
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    """Train an MLP velocity estimator on rollout episodes."""
    args = parse_args()
    if args.checkpoint_save_interval < 0:
        raise ValueError("--checkpoint_save_interval must be greater than or equal to zero.")

    set_seed(args.seed)

    run_dir = _build_run_dir(args.output_dir, args.run_name)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    dataset, train_subset, validation_subset = create_estimator_datasets(
        dataset_path=args.dataset_path,
        horizon=args.horizon,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        excluded_input_terms=set(args.exclude_input_terms or []),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = VelocityEstimator(
        input_dim=dataset.input_dim,
        horizon=args.horizon,
        output_dim=dataset.target_dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    _save_run_metadata(run_dir, dataset, args, train_subset, validation_subset)

    best_validation_loss = float("inf")
    best_checkpoint_path = os.path.join(run_dir, "best.pt")
    last_checkpoint_path = os.path.join(run_dir, "last.pt")
    periodic_checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if args.checkpoint_save_interval > 0:
        os.makedirs(periodic_checkpoint_dir, exist_ok=True)

    print(f"[INFO] Training velocity estimator in: {run_dir}")
    print(f"[INFO] Training samples: {len(train_subset)} | Validation samples: {len(validation_subset)}")
    print(f"[INFO] Input dim: {dataset.input_dim} | Target dim: {dataset.target_dim} | Horizon: {args.horizon}")
    print(f"[INFO] Input terms: {sorted(dataset.input_paths)}")
    print(f"[INFO] Target terms: {sorted(dataset.target_paths)}")
    print(f"[INFO] Excluded input terms: {sorted(set(args.exclude_input_terms or []))}")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_train_loss = 0.0
            total_train_mae = 0.0
            total_examples = 0

            for batch in train_loader:
                inputs = batch["inputs"].to(device=device, dtype=torch.float32)
                targets = batch["targets"].to(device=device, dtype=torch.float32)

                predictions = model(inputs)
                loss = F.mse_loss(predictions, targets)
                mae = F.l1_loss(predictions, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = inputs.shape[0]
                total_examples += batch_size
                total_train_loss += loss.item() * batch_size
                total_train_mae += mae.item() * batch_size

            train_loss = total_train_loss / max(total_examples, 1)
            train_mae = total_train_mae / max(total_examples, 1)

            validation_loss, validation_mae = _evaluate(model, validation_loader, device) if len(validation_subset) > 0 else (0.0, 0.0)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("mae/train", train_mae, epoch)
            if len(validation_subset) > 0:
                writer.add_scalar("loss/val", validation_loss, epoch)
                writer.add_scalar("mae/val", validation_mae, epoch)

            metrics = {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
            _save_checkpoint(last_checkpoint_path, model, optimizer, epoch, metrics, dataset, args)

            if args.checkpoint_save_interval > 0 and epoch % args.checkpoint_save_interval == 0:
                periodic_checkpoint_path = os.path.join(periodic_checkpoint_dir, f"epoch_{epoch:04d}.pt")
                _save_checkpoint(periodic_checkpoint_path, model, optimizer, epoch, metrics, dataset, args)

            monitored_loss = validation_loss if len(validation_subset) > 0 else train_loss
            if monitored_loss < best_validation_loss:
                best_validation_loss = monitored_loss
                _save_checkpoint(best_checkpoint_path, model, optimizer, epoch, metrics, dataset, args)

            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} train_mae={train_mae:.6f} "
                f"val_loss={validation_loss:.6f} val_mae={validation_mae:.6f}"
            )
    finally:
        writer.close()

    print(f"[INFO] Saved best checkpoint to: {best_checkpoint_path}")
    print(f"[INFO] Saved last checkpoint to: {last_checkpoint_path}")


if __name__ == "__main__":
    main()