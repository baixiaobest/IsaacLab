"""Evaluate a trained point-velocity predictor on noisy and clean HDF5 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import PointVelocityDataset
from src.losses import masked_metrics
from src.model import TemporalLidarVelocityCNN


def _evaluate(model, dataset, device, batch_size):
    totals: dict[str, list[float]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            prediction = model(batch["lidar"].to(device, dtype=torch.float32))
            results = masked_metrics(prediction, batch["target"].to(device), batch["reflection_mask"].to(device), batch["dynamic_mask"].to(device), batch["range_m"].to(device))
            for key, (value, count) in results.items():
                total = totals.setdefault(key, [0.0, 0.0])
                total[0] += value
                total[1] += count
    raw = {key: value / count for key, (value, count) in totals.items() if count}
    output = {"velocity_frame": "body_xy"}
    for subset in ("all", "static", "dynamic", "within_5m", "within_2m", "dynamic_within_5m", "dynamic_within_2m"):
        if subset in raw:
            output[f"{subset}_rmse"] = raw[subset] ** 0.5
        if f"{subset}_abs" in raw:
            output[f"{subset}_mae"] = raw[f"{subset}_abs"] / 2.0
        if f"zero_{subset}" in raw:
            output[f"zero_{subset}_rmse"] = raw[f"zero_{subset}"] ** 0.5
    for key in ("static_false_motion", "dynamic_heading_error"):
        if key in raw:
            output[key] = raw[key]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = TemporalLidarVelocityCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    output = {}
    for name in ("lidar_noisy", "lidar_clean"):
        dataset = PointVelocityDataset(args.dataset_path, input_name=name)
        split_file = Path(args.checkpoint).resolve().parent / "splits.json"
        if split_file.exists():
            test = Subset(dataset, json.loads(split_file.read_text(encoding="utf-8"))["test_indices"])
        else:
            _, _, test = dataset.split()
        result = _evaluate(model, test, device, args.batch_size)
        output[name] = result
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
