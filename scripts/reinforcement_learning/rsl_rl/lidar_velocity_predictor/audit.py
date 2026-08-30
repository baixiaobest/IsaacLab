"""Inspect collected LiDAR velocity HDF5 datasets before model training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from src.losses import distance_weight


def _files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.glob("*.hdf5"))


def _histogram(path: Path, static_distance: np.ndarray, dynamic_distance: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; skipping PNG plots.")
        return
    bins = np.linspace(0.0, 20.0, 81)
    plt.figure(figsize=(8, 4))
    plt.hist(static_distance, bins=bins, density=True, alpha=0.65, label="static")
    plt.hist(dynamic_distance, bins=bins, density=True, alpha=0.65, label="pedestrian")
    plt.xlabel("Reflection distance (m)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "distance_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LiDAR point-velocity dataset distribution.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    strata = Counter()
    static_ranges: list[np.ndarray] = []
    dynamic_ranges: list[np.ndarray] = []
    dynamic_speeds: list[np.ndarray] = []
    dynamic_heading: list[np.ndarray] = []
    close = Counter()

    for file_path in _files(Path(args.dataset_path).expanduser().resolve()):
        with h5py.File(file_path, "r") as handle:
            for group in handle["data"].values():
                reflection = np.asarray(group["reflection_mask"], dtype=bool)
                dynamic = np.asarray(group["dynamic_mask"], dtype=bool)
                ranges = np.asarray(group["range_m"], dtype=np.float32)
                velocity = np.asarray(group["point_velocity_w"], dtype=np.float32)
                static = reflection & ~dynamic
                counts.update({"total_cells": int(reflection.size), "no_return": int((~reflection).sum()), "static": int(static.sum()), "dynamic": int(dynamic.sum())})
                key = f"{group.attrs.get('terrain_name', 'unknown')}/level_{int(group.attrs.get('terrain_level', -1))}/scenario_{int(group.attrs.get('scenario_mode', -1))}"
                strata[key] += int(reflection.sum())
                static_ranges.append(ranges[static])
                dynamic_ranges.append(ranges[dynamic])
                if dynamic.any():
                    speed = np.linalg.norm(velocity[dynamic], axis=-1)
                    dynamic_speeds.append(speed)
                    dynamic_heading.append(np.arctan2(velocity[..., 1][dynamic], velocity[..., 0][dynamic]))
                close["dynamic_within_2m"] += int((dynamic & (ranges <= 2.0)).sum())
                close["dynamic_within_5m"] += int((dynamic & (ranges <= 5.0)).sum())

    static_distance = np.concatenate(static_ranges) if static_ranges else np.empty(0)
    dynamic_distance = np.concatenate(dynamic_ranges) if dynamic_ranges else np.empty(0)
    speed = np.concatenate(dynamic_speeds) if dynamic_speeds else np.empty(0)
    heading = np.concatenate(dynamic_heading) if dynamic_heading else np.empty(0)
    if counts["dynamic"] == 0:
        raise RuntimeError("Dataset contains no pedestrian reflection labels.")
    weight_static = float(distance_weight(torch.from_numpy(static_distance)).sum().item()) if static_distance.size else 0.0
    weight_dynamic = float(distance_weight(torch.from_numpy(dynamic_distance)).sum().item()) if dynamic_distance.size else 0.0
    summary = {
        "counts": dict(counts),
        "fractions": {name: value / counts["total_cells"] for name, value in counts.items() if name != "total_cells"},
        "close_dynamic": dict(close),
        "static_distance_quantiles_m": np.quantile(static_distance, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist() if static_distance.size else [],
        "dynamic_distance_quantiles_m": np.quantile(dynamic_distance, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
        "dynamic_speed_quantiles_mps": np.quantile(speed, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist() if speed.size else [],
        "dynamic_heading_quantiles_rad": np.quantile(heading, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist() if heading.size else [],
        "effective_distance_weight": {"static": weight_static, "dynamic": weight_dynamic},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "strata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["terrain_level_scenario", "valid_reflections"])
        writer.writerows(sorted(strata.items()))
    _histogram(output, static_distance, dynamic_distance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
