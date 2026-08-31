"""Inspect collected LiDAR velocity HDF5 datasets before model training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from src.losses import distance_weight


def _files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {path}.")
    return files


@dataclass(frozen=True)
class ScanSample:
    """Location of one scan-event visualization in an HDF5 rollout file."""

    category: str
    file_path: str
    episode_name: str
    sample_index: int


class _Reservoir:
    """Deterministically retain a uniform sample without retaining every scan id."""

    def __init__(self, category: str, capacity: int, rng: np.random.Generator) -> None:
        self.category = category
        self.capacity = capacity
        self.rng = rng
        self.count = 0
        self.samples: list[ScanSample] = []

    def consider(self, file_path: Path, episode_name: str, sample_index: int) -> None:
        if self.capacity <= 0:
            return
        sample = ScanSample(self.category, str(file_path), episode_name, int(sample_index))
        self.count += 1
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return
        replacement = int(self.rng.integers(self.count))
        if replacement < self.capacity:
            self.samples[replacement] = sample


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


def _bin_positions(range_m: np.ndarray) -> np.ndarray:
    """Map the heading-centred 128 forward bins to robot-body XY positions."""
    if range_m.shape != (128,):
        raise ValueError(f"Expected 128 forward LiDAR bins, got {range_m.shape}.")
    # ``forward_lidar_reflection_bins`` maps local bin 64 to the heading bin
    # and advances by 2*pi/256 per output bin.  This makes the visualization
    # use precisely the same body-frame convention as the training targets.
    angle = (np.arange(128, dtype=np.float32) - 64.0) * (2.0 * np.pi / 256.0)
    return np.column_stack((range_m * np.cos(angle), range_m * np.sin(angle)))


def _plot_scan_sample(output: Path, sample: ScanSample, max_range_m: float, arrow_seconds: float) -> dict:
    """Render one labelled scan in body coordinates and return manifest metadata."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required to render LiDAR scan sanity-check plots.") from None

    with h5py.File(sample.file_path, "r") as handle:
        group = handle["data"][sample.episode_name]
        reflection = np.asarray(group["reflection_mask"][sample.sample_index], dtype=bool)
        dynamic = np.asarray(group["dynamic_mask"][sample.sample_index], dtype=bool)
        ranges = np.asarray(group["range_m"][sample.sample_index], dtype=np.float32)
        velocity = np.asarray(group["point_velocity_b"][sample.sample_index], dtype=np.float32)
        capture_index = int(np.asarray(group["capture_index"][sample.sample_index]).reshape(-1)[0])
        terrain_name = str(group.attrs.get("terrain_name", "unknown"))
        terrain_level = int(group.attrs.get("terrain_level", -1))
        scenario_mode = int(group.attrs.get("scenario_mode", -1))

    valid = reflection & np.isfinite(ranges) & (ranges > 0.0) & (ranges <= max_range_m)
    points = _bin_positions(ranges)
    static = valid & ~dynamic
    moving = valid & dynamic
    fig, axes = plt.subplots(figsize=(7, 7))
    if static.any():
        axes.scatter(points[static, 0], points[static, 1], s=12, c="0.55", label="static return (v = 0)")
    if moving.any():
        axes.scatter(points[moving, 0], points[moving, 1], s=22, c="tab:orange", label="pedestrian return")
        axes.quiver(
            points[moving, 0],
            points[moving, 1],
            velocity[moving, 0] * arrow_seconds,
            velocity[moving, 1] * arrow_seconds,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="tab:red",
            width=0.005,
            headwidth=4.0,
            headlength=5.0,
            label=f"body-frame velocity × {arrow_seconds:g} s",
        )
    axes.scatter([0.0], [0.0], c="tab:blue", marker="^", s=70, zorder=3, label="robot")
    axes.axhline(0.0, color="0.85", linewidth=0.8)
    axes.axvline(0.0, color="0.85", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(-0.5, max_range_m + 0.5)
    axes.set_ylim(-max_range_m - 0.5, max_range_m + 0.5)
    axes.set_xlabel("body X: forward (m)")
    axes.set_ylabel("body Y: left (m)")
    axes.set_title(
        f"{sample.category} labelled scan | {terrain_name}, level {terrain_level}, scenario {scenario_mode}\n"
        f"{Path(sample.file_path).name}:{sample.episode_name}, capture {capture_index}"
    )
    axes.legend(loc="upper right", fontsize=8)
    axes.grid(alpha=0.2)
    fig.tight_layout()
    filename = f"{sample.category}_{Path(sample.file_path).stem}_{sample.episode_name}_{sample.sample_index:06d}.png"
    figure_path = output / "scan_samples" / filename
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return {
        **asdict(sample),
        "plot": str(figure_path),
        "capture_index": capture_index,
        "terrain_name": terrain_name,
        "terrain_level": terrain_level,
        "scenario_mode": scenario_mode,
        "valid_returns_plotted": int(valid.sum()),
        "dynamic_returns_plotted": int(moving.sum()),
        "max_range_m": max_range_m,
        "velocity_arrow_seconds": arrow_seconds,
    }


def _render_scan_samples(output: Path, samples: list[ScanSample], max_range_m: float, arrow_seconds: float) -> list[dict]:
    if not samples:
        return []
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        print("[WARN] matplotlib is unavailable; skipping LiDAR scan sanity-check plots.")
        return []
    return [_plot_scan_sample(output, sample, max_range_m, arrow_seconds) for sample in samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LiDAR point-velocity dataset distribution.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--num_scan_samples",
        type=int,
        default=4,
        help="Number of uniformly sampled static and dynamic scan-event plots each (default: 4).",
    )
    parser.add_argument("--scan_sample_seed", type=int, default=42)
    parser.add_argument(
        "--scan_plot_range_m",
        type=float,
        default=10.0,
        help="Display only valid returns within this body-frame range (default: 10 m).",
    )
    parser.add_argument(
        "--velocity_arrow_seconds",
        type=float,
        default=1.0,
        help="Arrow length represents this many seconds of body-frame target velocity (default: 1).",
    )
    args = parser.parse_args()
    if args.num_scan_samples < 0:
        parser.error("--num_scan_samples must be nonnegative.")
    if args.scan_plot_range_m <= 0.0 or args.velocity_arrow_seconds <= 0.0:
        parser.error("--scan_plot_range_m and --velocity_arrow_seconds must be positive.")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    strata = Counter()
    static_ranges: list[np.ndarray] = []
    dynamic_ranges: list[np.ndarray] = []
    dynamic_speeds: list[np.ndarray] = []
    dynamic_heading: list[np.ndarray] = []
    close = Counter()
    rng = np.random.default_rng(args.scan_sample_seed)
    dynamic_scan_samples = _Reservoir("dynamic", args.num_scan_samples, rng)
    static_scan_samples = _Reservoir("static", args.num_scan_samples, rng)

    for file_path in _files(Path(args.dataset_path).expanduser().resolve()):
        with h5py.File(file_path, "r") as handle:
            data = handle.get("data")
            if data is None:
                raise RuntimeError(f"{file_path} is missing its data group.")
            try:
                metadata = json.loads(data.attrs.get("metadata", "{}"))
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{file_path} has invalid dataset metadata.") from error
            if metadata.get("schema_version") != 2 or metadata.get("velocity_frame") != "body_xy":
                raise RuntimeError(f"{file_path} is not a body-frame schema-v2 LiDAR velocity dataset.")
            for group in data.values():
                reflection = np.asarray(group["reflection_mask"], dtype=bool)
                dynamic = np.asarray(group["dynamic_mask"], dtype=bool)
                ranges = np.asarray(group["range_m"], dtype=np.float32)
                velocity = np.asarray(group["point_velocity_b"], dtype=np.float32)
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
                # Dynamic examples have at least one visible pedestrian label.
                # Static examples have valid returns but no pedestrian label, so
                # every shown reflection is supervised to the zero vector.
                visible = reflection & (ranges > 0.0) & (ranges <= args.scan_plot_range_m)
                for sample_index in np.flatnonzero((dynamic & visible).any(axis=1)):
                    dynamic_scan_samples.consider(file_path, group.name.rsplit("/", 1)[-1], int(sample_index))
                for sample_index in np.flatnonzero(visible.any(axis=1) & ~dynamic.any(axis=1)):
                    static_scan_samples.consider(file_path, group.name.rsplit("/", 1)[-1], int(sample_index))

    static_distance = np.concatenate(static_ranges) if static_ranges else np.empty(0)
    dynamic_distance = np.concatenate(dynamic_ranges) if dynamic_ranges else np.empty(0)
    speed = np.concatenate(dynamic_speeds) if dynamic_speeds else np.empty(0)
    heading = np.concatenate(dynamic_heading) if dynamic_heading else np.empty(0)
    if counts["dynamic"] == 0:
        raise RuntimeError("Dataset contains no pedestrian reflection labels.")
    weight_static = float(distance_weight(torch.from_numpy(static_distance)).sum().item()) if static_distance.size else 0.0
    weight_dynamic = float(distance_weight(torch.from_numpy(dynamic_distance)).sum().item()) if dynamic_distance.size else 0.0
    summary = {
        "schema_version": 2,
        "velocity_frame": "body_xy",
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
    visualizations = _render_scan_samples(
        output,
        static_scan_samples.samples + dynamic_scan_samples.samples,
        args.scan_plot_range_m,
        args.velocity_arrow_seconds,
    )
    (output / "scan_samples.json").write_text(json.dumps(visualizations, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {len(visualizations)} labelled LiDAR scan plots to {output / 'scan_samples'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
