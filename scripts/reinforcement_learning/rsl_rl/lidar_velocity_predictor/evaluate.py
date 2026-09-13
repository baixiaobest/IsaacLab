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


def _resolve_fov_bins(checkpoint: str, explicit: int | None) -> int:
    """Return ``explicit`` if given, else auto-detect from the checkpoint's sibling metadata.json."""
    if explicit is not None:
        return explicit
    metadata_file = Path(checkpoint).resolve().parent / "metadata.json"
    if metadata_file.exists():
        fov_bins = json.loads(metadata_file.read_text(encoding="utf-8")).get("args", {}).get("fov_bins")
        if fov_bins is not None:
            return int(fov_bins)
    return 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--fov_bins",
        type=int,
        default=None,
        help="Forward-arc bin count. Defaults to auto-detecting from the checkpoint's metadata.json, else 128.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    fov_bins = _resolve_fov_bins(args.checkpoint, args.fov_bins)
    model = TemporalLidarVelocityCNN(fov_bins=fov_bins).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    output = {}
    for name in ("lidar_noisy", "lidar_clean"):
        dataset = PointVelocityDataset(args.dataset_path, input_name=name, fov_bins=fov_bins)
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
