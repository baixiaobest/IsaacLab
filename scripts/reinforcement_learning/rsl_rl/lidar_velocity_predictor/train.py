"""Train the temporal-LiDAR point velocity predictor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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
    metadata = {"args": vars(args), "num_samples": len(dataset), "train": len(train), "validation": len(validation), "test": len(test)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        metrics = evaluate(model, validation_loader, device)
        checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics, "model": "TemporalLidarVelocityCNN"}
        torch.save(checkpoint, output / "last.pt")
        score = metrics.get("dynamic_within_5m_rmse", metrics.get("dynamic_rmse", float("inf")))
        if score < best_rmse:
            best_rmse = score
            torch.save(checkpoint, output / "best.pt")
        print(f"[Epoch {epoch:03d}] train_loss={train_loss / max(len(train_loader), 1):.6f} dynamic_5m_rmse={score:.6f}")
    print(json.dumps({"validation": evaluate(model, validation_loader, device), "test": evaluate(model, DataLoader(test, batch_size=args.batch_size), device)}, indent=2))


if __name__ == "__main__":
    main()
